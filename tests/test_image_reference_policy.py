"""Policy regression tests: retain schema safety without rewriting valid references."""
import json

import pytest

from lwb.backend import ChatResponse
from lwb.nodes import LlamaWorkbenchPromptEnhancer
from lwb.prompt_rewrite import (
    IMAGE_REFERENCE_POLICIES,
    PROMPT_REWRITE_PROFILES,
    PromptRewriteFormatError,
    build_edit_image_reference_rule,
    build_prompt_rewrite_messages,
    parse_prompt_rewrite_json,
    rewrite_prompt,
)


CHARACTER_SHEET = (
    "Create a professional character reference sheet using <image1> as the sole identity and design reference. "
    "Preserve the exact facial identity, hairstyle, clothing, footwear, body proportions and visible accessories from <image1>. "
    "Place a large frontal portrait on the left, front/true-side/back full-body turnaround views on the upper right, "
    "and several close-up identity-relevant detail views on the lower right. "
    "Use a clean neutral studio background and maintain consistent identity, clothing and materials across every view."
)


def answer(prompt=CHARACTER_SHEET, ratio="16:9", follow=""):
    return json.dumps(dict(rewritten_prompt=prompt, wh_ratio=ratio, ratio_follow=follow))


class Backend:
    def __init__(self, *answers):
        self.answers = iter(answers)
        self.calls = []

    def chat_response(self, messages, **settings):
        self.calls.append((messages, settings))
        return ChatResponse(content=next(self.answers))


@pytest.mark.parametrize("prompt", [CHARACTER_SHEET, "Preserve the exact identity from <image1>.",
                                    "Preserve the exact identity from the input reference image."])
def test_compatible_preserves_single_image_prompt(prompt):
    result = parse_prompt_rewrite_json(answer(prompt), PROMPT_REWRITE_PROFILES["edit"], image_count=1)
    assert result["rewritten_prompt"] == prompt


@pytest.mark.parametrize("policy", IMAGE_REFERENCE_POLICIES)
@pytest.mark.parametrize("prompt,ratio,follow,count,error", [
    ("Use <image2>", "16:9", "", 1, "only 1"),
    ("Use <image0>", "16:9", "", 1, "invalid image reference"),
    ("Use <image-1>", "16:9", "", 1, "invalid image reference"),
    ("Use <image01>", "16:9", "", 1, "invalid image reference"),
    ("Use <image1", "16:9", "", 1, "invalid image reference"),
    ("Use <image1 <image1>", "16:9", "", 1, "invalid image reference"),
    ("Use <IMAGE1>", "16:9", "", 1, "invalid image reference"),
    ("Use <image1>", "16:9", "", 2, "missing: <image2>"),
    ("Use the image", "16:9", "<image1>", 1, "exactly one"),
    ("Use the image", "", "", 1, "exactly one"),
    ("Use the image", "", "image1", 1, "ratio_follow"),
    ("Use the image", "", "<image2>", 1, "only 1"),
    ("Use the image", "16/9", "", 1, "wh_ratio"),
    ("", "16:9", "", 1, "must not be empty"),
])
def test_hard_validation_applies_to_both_policies(policy, prompt, ratio, follow, count, error):
    with pytest.raises(PromptRewriteFormatError, match=error):
        parse_prompt_rewrite_json(answer(prompt, ratio, follow), PROMPT_REWRITE_PROFILES["edit"],
                                  image_count=count, image_reference_policy=policy)


@pytest.mark.parametrize("policy", IMAGE_REFERENCE_POLICIES)
def test_valid_multi_image_and_single_ratio_follow(policy):
    for prompt, count, follow in [("Use <image1> and <image2>", 2, "<image2>"),
                                  ("Use the reference image", 1, "<image1>")]:
        parsed = parse_prompt_rewrite_json(answer(prompt, "", follow), PROMPT_REWRITE_PROFILES["edit"],
                                           image_count=count, image_reference_policy=policy)
        assert parsed["ratio_follow"] == follow


@pytest.mark.parametrize("policy", IMAGE_REFERENCE_POLICIES)
@pytest.mark.parametrize("raw", ['{', '{"rewritten_prompt":"x"}',
    '{"rewritten_prompt":3,"wh_ratio":"1:1","ratio_follow":""}',
    '{"rewritten_prompt":"x","wh_ratio":"1:1","wh_ratio":"2:1","ratio_follow":""}',
    '{"rewritten_prompt":"x","wh_ratio":"1:1","ratio_follow":"","extra":"x"}'])
def test_schema_remains_strict(policy, raw):
    with pytest.raises(PromptRewriteFormatError):
        parse_prompt_rewrite_json(raw, PROMPT_REWRITE_PROFILES["edit"], image_count=1,
                                  image_reference_policy=policy)


@pytest.mark.parametrize("policy", IMAGE_REFERENCE_POLICIES)
@pytest.mark.parametrize("raw", [answer(), answer("scene", "", "<image1>")])
def test_t2i_rejects_image_references(policy, raw):
    with pytest.raises(PromptRewriteFormatError):
        parse_prompt_rewrite_json(raw, PROMPT_REWRITE_PROFILES["t2i"], image_reference_policy=policy)


def test_runtime_rules_match_policy():
    rule = build_edit_image_reference_rule(1, "compatible")
    assert 'refer naturally to "the input reference image"' in rule
    assert "or use the explicit tag <image1>. Both forms are valid" in rule
    assert "Never emit any other image tag" in rule
    assert "Do not use an image tag inside rewritten_prompt" in build_edit_image_reference_rule(1, "official_strict")
    assert build_edit_image_reference_rule(2, "compatible") == build_edit_image_reference_rule(2, "official_strict")
    assert "reference every input image at least once" in build_edit_image_reference_rule(2)


@pytest.mark.parametrize("policy", IMAGE_REFERENCE_POLICIES)
def test_single_image_retry_contract(policy, capsys):
    backend = Backend(answer(), answer())
    kwargs = dict(task="edit", system_prompt="custom", image_data_urls=["image"],
                  image_reference_policy=policy, debug=True)
    if policy == "compatible":
        result = rewrite_prompt(backend, "make a sheet", **kwargs)
        assert result.rewritten_prompt == CHARACTER_SHEET
        assert result.retried is False
        assert len(backend.calls) == 1
    else:
        with pytest.raises(PromptRewriteFormatError, match="single-image edit"):
            rewrite_prompt(backend, "make a sheet", **kwargs)
        assert len(backend.calls) == 2
        assert backend.calls[1][1]["enable_thinking"] is False
        assert build_edit_image_reference_rule(1, policy) in backend.calls[1][0][-1]["content"]
    output = capsys.readouterr().out
    assert f"image_reference_policy={policy} image_count=1" in output


def test_compatible_retry_allows_valid_tag_after_invalid_json():
    backend = Backend("bad", answer())
    result = rewrite_prompt(backend, "edit", task="edit", system_prompt="custom", image_data_urls=["image"])
    assert result.retried
    corrective = backend.calls[1][0][-1]["content"]
    assert "Both forms are valid" in corrective
    assert "Do not use an image tag" not in corrective


def test_invalid_policy_fails_before_generation():
    backend = Backend()
    with pytest.raises(ValueError, match="Unknown image_reference_policy"):
        rewrite_prompt(backend, "edit", image_reference_policy="typo")
    assert not backend.calls
    for call in [
        lambda: build_edit_image_reference_rule(1, "typo"),
        lambda: build_prompt_rewrite_messages(PROMPT_REWRITE_PROFILES["edit"], "s", "p", ["i"], "typo"),
        lambda: parse_prompt_rewrite_json(answer(), PROMPT_REWRITE_PROFILES["edit"], image_reference_policy="typo"),
    ]:
        with pytest.raises(ValueError, match="Unknown image_reference_policy"):
            call()


@pytest.mark.parametrize("pixels,expected", [(1048576, 1048576), (2097152, 2097152),
                                            (4194304, 4194304), (8388608, 4194304)])
def test_node_pixel_budget_and_legacy_default_policy(monkeypatch, pixels, expected):
    calls = []
    def encode(image, **kwargs):
        calls.append(kwargs)
        return ["data:image/png;base64,test"]
    monkeypatch.setattr("lwb.nodes.image_tensor_to_data_urls", encode)
    backend = Backend(answer())
    output = LlamaWorkbenchPromptEnhancer().enhance(
        backend, "edit", "edit", "custom", "", image1=object(), max_image_pixels=pixels)
    assert calls[0]["max_pixels"] == expected
    assert calls[0]["max_edge"] == 4096
    assert output["result"][0] == CHARACTER_SHEET
    assert output["ui"]["retried"] == [False]


def test_node_schema_appends_optional_policy():
    schema = LlamaWorkbenchPromptEnhancer.INPUT_TYPES()
    assert list(schema["optional"])[-2:] == ["debug", "image_reference_policy"]
    assert schema["optional"]["image_reference_policy"][0] == ["compatible", "official_strict"]
    assert schema["optional"]["image_reference_policy"][1]["default"] == "compatible"
    pixels = schema["required"]["max_image_pixels"][1]
    assert (pixels["default"], pixels["min"], pixels["max"], pixels["step"]) == (1048576, 65536, 4194304, 65536)


def test_node_forwards_official_strict_policy(monkeypatch):
    monkeypatch.setattr("lwb.nodes.image_tensor_to_data_urls", lambda *args, **kwargs: ["image"])
    backend = Backend(answer(), answer())
    with pytest.raises(PromptRewriteFormatError, match="single-image edit"):
        LlamaWorkbenchPromptEnhancer().enhance(
            backend, "edit", "edit", "custom", "", image1=object(),
            image_reference_policy="official_strict")
    assert len(backend.calls) == 2
