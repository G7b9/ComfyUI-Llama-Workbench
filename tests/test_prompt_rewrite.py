from __future__ import annotations

import json
import sys
import types

import pytest

import lwb.nodes as nodes_module
from lwb.backend import ChatResponse
from lwb.nodes import (
    LlamaWorkbenchQwenImage21PECanvas,
    LlamaWorkbenchPromptEnhancer,
    LlamaWorkbenchQwenImage21PEResolution,
    NODE_CLASS_MAPPINGS,
)
from lwb.prompt_rewrite import (
    PROMPT_REWRITE_PROFILES,
    PROMPT_REWRITE_AUTO_ASPECT_RATIO,
    QWEN_IMAGE_21_ASPECT_RATIOS,
    PromptRewriteFormatError,
    build_edit_image_reference_rule,
    build_prompt_rewrite_messages,
    load_system_prompt,
    parse_prompt_rewrite_json,
    prompt_rewrite_canvas_spec,
    prompt_rewrite_dimensions,
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
    assert "exactly 2 input image(s)" in messages[0]["content"]
    assert "<image1>, <image2>" in messages[0]["content"]
    assert [part["type"] for part in content] == ["image_url", "image_url", "text"]
    assert content[0]["image_url"]["url"].endswith("one")
    assert content[1]["image_url"]["url"].endswith("two")
    assert content[2]["text"] == "edit these"


def test_dynamic_edit_image_rule_distinguishes_single_and_multi_image_contracts():
    assert "Do not use an image tag inside rewritten_prompt" in build_edit_image_reference_rule(1)
    rule = build_edit_image_reference_rule(3)
    assert "<image1>, <image2>, <image3>" in rule
    assert "reference every input image at least once" in rule


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
    ],
)
def test_strict_parser_rejects_wrappers_unknown_fields_and_invalid_values(answer, match):
    with pytest.raises(PromptRewriteFormatError, match=match):
        parse_prompt_rewrite_json(answer, PROMPT_REWRITE_PROFILES["t2i"])


def test_strict_parser_accepts_escaped_line_breaks_inside_rewritten_prompt():
    parsed = parse_prompt_rewrite_json(
        '{"rewritten_prompt":"top region\\n\\nbottom region","wh_ratio":"3:4"}',
        PROMPT_REWRITE_PROFILES["t2i"],
    )

    assert parsed["rewritten_prompt"] == "top region\n\nbottom region"


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
            '{"rewritten_prompt":"use <image1> with <image2>","wh_ratio":"","ratio_follow":"<image3>"}',
            profile,
            image_count=2,
        )


def test_edit_parser_validates_rewritten_prompt_image_references():
    profile = PROMPT_REWRITE_PROFILES["edit"]
    parsed = parse_prompt_rewrite_json(
        '{"rewritten_prompt":"Move <image1> into <image2>","wh_ratio":"","ratio_follow":"<image2>"}',
        profile,
        image_count=2,
    )
    assert parsed["ratio_follow"] == "<image2>"
    with pytest.raises(PromptRewriteFormatError, match="missing: <image2>"):
        parse_prompt_rewrite_json(
            '{"rewritten_prompt":"Edit <image1>","wh_ratio":"","ratio_follow":"<image1>"}',
            profile,
            image_count=2,
        )
    with pytest.raises(PromptRewriteFormatError, match="single-image edit"):
        parse_prompt_rewrite_json(
            '{"rewritten_prompt":"Edit <image1>","wh_ratio":"","ratio_follow":"<image1>"}',
            profile,
            image_count=1,
        )


def test_edit_parser_accepts_exactly_ordered_ten_image_reference_set():
    references = " ".join(f"<image{index}>" for index in range(1, 11))
    parsed = parse_prompt_rewrite_json(
        json.dumps(
            {
                "rewritten_prompt": f"Use each source independently: {references}",
                "wh_ratio": "",
                "ratio_follow": "<image10>",
            }
        ),
        PROMPT_REWRITE_PROFILES["edit"],
        image_count=10,
    )

    assert parsed["ratio_follow"] == "<image10>"


def test_format_failure_retries_once_without_thinking():
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
    assert backend.calls[0][1]["enable_thinking"] is True
    assert backend.calls[1][1]["enable_thinking"] is False
    assert backend.calls[0][1]["accept_truncated_response"] is True
    assert backend.calls[0][1]["presence_penalty"] == 1.5
    assert backend.calls[0][1]["min_p"] == 0.0
    assert [message["role"] for message in backend.calls[1][0]] == [
        "system",
        "user",
        "user",
    ]
    assert "without a thinking trace" in backend.calls[1][0][-1]["content"]


def test_generation_truncation_retries_once_even_when_partial_json_is_valid():
    backend = SequenceBackend(
        ChatResponse(
            content='{"rewritten_prompt":"partial","wh_ratio":"1:1"}',
            finish_reason="length",
        ),
        ChatResponse(
            content='{"rewritten_prompt":"complete","wh_ratio":"1:1"}',
            finish_reason="stop",
        ),
    )

    result = rewrite_prompt(backend, "corgi", system_prompt="user supplied")

    assert result.rewritten_prompt == "complete"
    assert result.retried is True
    assert backend.calls[1][1]["enable_thinking"] is False


def test_second_format_failure_is_terminal():
    backend = SequenceBackend(ChatResponse(content="bad"), ChatResponse(content="still bad"))

    with pytest.raises(PromptRewriteFormatError, match="failed structured output after one retry"):
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


def test_edit_prompt_enhancer_uses_bounded_lossless_image_transport(monkeypatch):
    first, second = object(), object()
    calls = []

    def encode(image, max_images, max_edge, **options):
        calls.append((image, max_images, max_edge, options))
        return [f"data:image/png;base64,{len(calls)}"]

    monkeypatch.setattr("lwb.nodes.image_tensor_to_data_urls", encode)
    backend = SequenceBackend(
        ChatResponse(
            content=(
                '{"rewritten_prompt":"Move <image1> into <image2>",'
                '"wh_ratio":"","ratio_follow":"<image2>"}'
            )
        )
    )

    output = LlamaWorkbenchPromptEnhancer().enhance(
        backend,
        "combine them",
        "edit",
        "user supplied",
        "",
        max_images=5,
        max_image_edge=0,
        max_image_pixels=16777216,
        image1=first,
        image2=second,
    )

    assert output["result"][2] == "<image2>"
    assert calls == [
        (first, 5, 4096, {"max_pixels": 1048576, "image_format": "png"}),
        (second, 4, 4096, {"max_pixels": 1048576, "image_format": "png"}),
    ]


def test_prompt_rewrite_dimensions_preserve_ratio_and_pixel_budget():
    width, height, ratio = prompt_rewrite_dimensions("16:9", megapixels=1.0, multiple=8)

    assert (width, height, ratio) == (1368, 768, "16:9")
    assert width % 8 == height % 8 == 0
    assert width * height == pytest.approx(1024 * 1024, rel=0.02)
    assert prompt_rewrite_dimensions("1920:1080")[2] == "16:9"
    assert prompt_rewrite_dimensions("16:9", aspect_ratio_override="1:1") == (1024, 1024, "1:1")


@pytest.mark.parametrize("ratio", ["", "auto", "0:1", "16/9"])
def test_prompt_rewrite_dimensions_reject_invalid_ratios(ratio):
    with pytest.raises(ValueError, match="positive W:H"):
        prompt_rewrite_dimensions(ratio)


def test_qwen_pe_resolution_node_is_registered_and_uses_force_input():
    inputs = LlamaWorkbenchQwenImage21PEResolution.INPUT_TYPES()["required"]

    assert NODE_CLASS_MAPPINGS["LlamaWorkbench_QwenImage21PEResolution"] is LlamaWorkbenchQwenImage21PEResolution
    assert inputs["wh_ratio"][1]["forceInput"] is True
    assert inputs["aspect_ratio_override"][0] == [
        PROMPT_REWRITE_AUTO_ASPECT_RATIO,
        *QWEN_IMAGE_21_ASPECT_RATIOS,
    ]
    assert LlamaWorkbenchQwenImage21PEResolution().select("1:1") == (1024, 1024, "1:1")
    assert LlamaWorkbenchQwenImage21PEResolution().select(
        "16:9", aspect_ratio_override="2:3"
    ) == (840, 1256, "2:3")


def test_qwen_pe_canvas_resolves_ratio_follow_and_creates_native_latent(monkeypatch):
    class FakeImage:
        shape = (1, 768, 1280, 3)

    monkeypatch.setattr(
        "lwb.nodes._qwen_image_21_empty_latent",
        lambda width, height: {"samples": {"shape": (1, 64, height // 16, width // 16)}},
    )
    canvas = LlamaWorkbenchQwenImage21PECanvas()

    width, height, latent, source = canvas.build(
        "",
        "<image1>",
        image1=FakeImage(),
        follow_input_size=True,
    )

    assert (width, height) == (1280, 768)
    assert latent["samples"]["shape"] == (1, 64, 48, 80)
    assert source == "ratio_follow:<image1>:input_size"
    assert NODE_CLASS_MAPPINGS["LlamaWorkbench_QwenImage21PECanvas"] is LlamaWorkbenchQwenImage21PECanvas


def test_qwen_image_21_empty_latent_uses_64_channels_and_sixteenth_scale(monkeypatch):
    calls = []
    torch_module = types.ModuleType("torch")
    torch_module.zeros = lambda shape, device=None: calls.append((tuple(shape), device)) or "samples"
    comfy_module = types.ModuleType("comfy")
    comfy_module.model_management = types.SimpleNamespace(intermediate_device=lambda: "test-device")
    monkeypatch.setitem(sys.modules, "torch", torch_module)
    monkeypatch.setitem(sys.modules, "comfy", comfy_module)

    latent = nodes_module._qwen_image_21_empty_latent(1280, 768)

    assert latent == {"samples": "samples"}
    assert calls == [((1, 64, 48, 80), "test-device")]


def test_qwen_pe_canvas_uses_ratio_budget_when_not_following_input_size():
    spec = prompt_rewrite_canvas_spec(
        "",
        "<image2>",
        [(640, 640), (1200, 800)],
        megapixels=1.0,
        multiple=32,
        follow_input_size=False,
    )

    assert spec.width % 32 == spec.height % 32 == 0
    assert spec.width * spec.height == pytest.approx(1024 * 1024, rel=0.04)
    assert spec.ratio_source == "ratio_follow:<image2>:aspect_only"
