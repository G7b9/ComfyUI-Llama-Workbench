from __future__ import annotations

import json

import pytest

from lwb.backend import ChatResponse
from lwb.nodes import LlamaWorkbenchPromptEnhancer, NODE_CLASS_MAPPINGS
from lwb.prompt_rewrite import (
    PROMPT_REWRITE_PROFILES,
    PromptRewriteFormatError,
    build_prompt_rewrite_messages,
    load_system_prompt,
    parse_prompt_rewrite_json,
    rewrite_prompt,
)


class SequenceBackend:
    label = "test backend"

    def __init__(self, *responses: ChatResponse):
        self.responses = list(responses)
        self.calls = []

    def chat_response(self, messages, **settings):
        self.calls.append((messages, settings))
        return self.responses.pop(0)


def test_official_prompt_rewrite_profiles_are_task_specific():
    t2i = PROMPT_REWRITE_PROFILES["t2i"]
    edit = PROMPT_REWRITE_PROFILES["edit"]

    assert t2i.request_settings(42) == {
        "max_tokens": 16256,
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 1.5,
        "seed": 42,
        "enable_thinking": True,
    }
    assert edit.max_tokens == 24000
    assert edit.presence_penalty == 0.0
    assert edit.takes_images is True


def test_system_prompt_must_come_from_one_user_or_local_source(tmp_path):
    prompt_file = tmp_path / "system_prompt.txt"
    prompt_file.write_text("local prompt\n", encoding="utf-8")

    assert load_system_prompt(" pasted prompt ") == "pasted prompt"
    assert load_system_prompt(system_prompt_path=str(prompt_file)) == "local prompt"
    assert load_system_prompt(system_prompt_path=str(tmp_path)) == "local prompt"
    with pytest.raises(ValueError, match="not both"):
        load_system_prompt("inline", str(prompt_file))
    with pytest.raises(ValueError, match="required"):
        load_system_prompt()


def test_edit_messages_keep_images_in_numbered_order_before_text():
    messages = build_prompt_rewrite_messages(
        PROMPT_REWRITE_PROFILES["edit"],
        "system",
        "edit these",
        ["data:image/jpeg;base64,one", "data:image/jpeg;base64,two"],
    )

    content = messages[1]["content"]
    assert [part["type"] for part in content] == ["image_url", "image_url", "text"]
    assert content[0]["image_url"]["url"].endswith("one")
    assert content[1]["image_url"]["url"].endswith("two")
    assert content[2]["text"] == "edit these"


def test_strict_t2i_parser_normalizes_the_structured_outputs():
    parsed = parse_prompt_rewrite_json(
        '{"rewritten_prompt":"a detailed scene","wh_ratio":"16:9"}',
        PROMPT_REWRITE_PROFILES["t2i"],
    )

    assert parsed == {
        "rewritten_prompt": "a detailed scene",
        "wh_ratio": "16:9",
        "ratio_follow": "",
    }


@pytest.mark.parametrize(
    "answer,match",
    [
        ('```json\n{"rewritten_prompt":"x","wh_ratio":"1:1"}\n```', "single valid JSON"),
        ('{"rewritten_prompt":"x","wh_ratio":"1:1","extra":true}', "unexpected"),
        ('{"rewritten_prompt":"x","wh_ratio":"1:1","wh_ratio":"3:2"}', "repeats"),
        ('{"rewritten_prompt":"x","wh_ratio":""}', "non-empty wh_ratio"),
        ('{"rewritten_prompt":"x\\ny","wh_ratio":"1:1"}', "single paragraph"),
    ],
)
def test_strict_parser_rejects_wrappers_unknown_fields_and_invalid_values(answer, match):
    with pytest.raises(PromptRewriteFormatError, match=match):
        parse_prompt_rewrite_json(answer, PROMPT_REWRITE_PROFILES["t2i"])


def test_edit_parser_enforces_mutual_exclusion_and_image_bounds():
    profile = PROMPT_REWRITE_PROFILES["edit"]
    with pytest.raises(PromptRewriteFormatError, match="exactly one"):
        parse_prompt_rewrite_json(
            '{"rewritten_prompt":"x","wh_ratio":"1:1","ratio_follow":"<image1>"}',
            profile,
            image_count=1,
        )
    with pytest.raises(PromptRewriteFormatError, match="only 2"):
        parse_prompt_rewrite_json(
            '{"rewritten_prompt":"x","wh_ratio":"","ratio_follow":"<image3>"}',
            profile,
            image_count=2,
        )


def test_format_failure_retries_once_with_same_official_sampling_profile():
    backend = SequenceBackend(
        ChatResponse(content="not json", reasoning="first thought"),
        ChatResponse(
            content='<think>second thought</think>{"rewritten_prompt":"rainy corgi","wh_ratio":"3:2"}'
        ),
    )

    result = rewrite_prompt(backend, "corgi", system_prompt="user supplied", seed=7)

    assert result.rewritten_prompt == "rainy corgi"
    assert result.wh_ratio == "3:2"
    assert result.ratio_follow == ""
    assert result.thinking == "first thought\n\nsecond thought"
    assert result.retried is True
    assert len(backend.calls) == 2
    assert backend.calls[0][1] == backend.calls[1][1]
    assert backend.calls[0][1]["presence_penalty"] == 1.5
    assert backend.calls[0][1]["min_p"] == 0.0
    assert backend.calls[0][1]["enable_thinking"] is True
    assert [message["role"] for message in backend.calls[1][0]] == [
        "system",
        "user",
        "assistant",
        "user",
    ]


def test_second_format_failure_is_terminal():
    backend = SequenceBackend(ChatResponse(content="bad"), ChatResponse(content="still bad"))

    with pytest.raises(PromptRewriteFormatError, match="invalid structured output twice"):
        rewrite_prompt(backend, "corgi", system_prompt="user supplied")
    assert len(backend.calls) == 2


def test_prompt_enhancer_node_is_registered_and_returns_separate_fields():
    backend = SequenceBackend(
        ChatResponse(content='{"rewritten_prompt":"expanded","wh_ratio":"1:1"}', reasoning="think")
    )
    node_inputs = LlamaWorkbenchPromptEnhancer.INPUT_TYPES()

    assert NODE_CLASS_MAPPINGS["LlamaWorkbench_PromptEnhancer"] is LlamaWorkbenchPromptEnhancer
    assert {"image", *(f"image{index}" for index in range(1, 11))} <= set(node_inputs["optional"])
    output = LlamaWorkbenchPromptEnhancer().enhance(
        backend,
        "short",
        "t2i",
        "user supplied",
        "",
    )

    rewritten, ratio, follow, result_json, thinking, _ = output["result"]
    assert (rewritten, ratio, follow, thinking) == ("expanded", "1:1", "", "think")
    assert json.loads(result_json) == {
        "rewritten_prompt": "expanded",
        "wh_ratio": "1:1",
        "ratio_follow": "",
    }
