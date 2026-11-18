import json

import pytest

from lwb.nodes import (
    LlamaWorkbenchPromptEnhancer,
    LlamaWorkbenchQwenImage21PECanvas,
    LlamaWorkbenchQwenImage21PEManual,
    NODE_CLASS_MAPPINGS,
)


def test_manual_prompt_preserves_text_and_pe_output_contract(monkeypatch):
    def unexpected_inference(*args, **kwargs):
        pytest.fail("Manual parameters must not run inference")

    monkeypatch.setattr("lwb.nodes.rewrite_prompt", unexpected_inference)
    prompt = '  中文提示词 "保持原样"\n<image1>  '
    result = LlamaWorkbenchQwenImage21PEManual().build(prompt, "9:19.5")

    assert result[:3] == (prompt, "6:13", "")
    assert json.loads(result[3]) == dict(zip(LlamaWorkbenchPromptEnhancer.RETURN_NAMES[:3], result[:3]))
    assert LlamaWorkbenchQwenImage21PEManual.RETURN_NAMES == LlamaWorkbenchPromptEnhancer.RETURN_NAMES[:4]
    assert NODE_CLASS_MAPPINGS["LlamaWorkbench_QwenImage21PEManual"] is LlamaWorkbenchQwenImage21PEManual
    assert set(LlamaWorkbenchQwenImage21PEManual.INPUT_TYPES()["required"]) == {
        "prompt", "wh_ratio", "ratio_follow"
    }


@pytest.mark.parametrize("follow", ["none", ""])
def test_manual_ratio_feeds_canvas(monkeypatch, follow):
    monkeypatch.setattr("lwb.nodes._qwen_image_21_empty_latent", lambda w, h: {"size": (w, h)})
    _, ratio, reference, _ = LlamaWorkbenchQwenImage21PEManual().build("", "1:1", follow)
    width, height, latent, _ = LlamaWorkbenchQwenImage21PECanvas().build(ratio, reference)
    assert (width, height) == (1024, 1024)
    assert latent == {"size": (1024, 1024)}


@pytest.mark.parametrize("index", range(1, 11))
def test_manual_follow_clears_ratio_and_feeds_canvas(monkeypatch, index):
    class Image:
        shape = (1, 768, 1280, 3)

    monkeypatch.setattr("lwb.nodes._qwen_image_21_empty_latent", lambda w, h: {"size": (w, h)})
    follow = f"<image{index}>"
    result = LlamaWorkbenchQwenImage21PEManual().build("edit", "ignored", follow)
    assert result[:3] == ("edit", "", follow)
    assert json.loads(result[3])["ratio_follow"] == follow
    images = {f"image{i}": Image() for i in range(1, index + 1)}
    width, height, _, source = LlamaWorkbenchQwenImage21PECanvas().build(*result[1:3], **images)
    assert (width, height, source) == (1280, 768, f"ratio_follow:{follow}:input_size")


@pytest.mark.parametrize("ratio", ["", "0:1", "16/9", "-1:2", "nan:1"])
def test_manual_rejects_invalid_ratios(ratio):
    with pytest.raises(ValueError, match="positive W:H"):
        LlamaWorkbenchQwenImage21PEManual().build("", ratio)


@pytest.mark.parametrize("follow", ["image1", "<image0>", "<image11>", "<image01>"])
def test_manual_rejects_invalid_references(follow):
    with pytest.raises(ValueError, match="ratio_follow"):
        LlamaWorkbenchQwenImage21PEManual().build("", ratio_follow=follow)


def test_canvas_reports_missing_manual_reference():
    result = LlamaWorkbenchQwenImage21PEManual().build("edit", ratio_follow="<image2>")
    with pytest.raises(ValueError, match="Canvas received 0 image"):
        LlamaWorkbenchQwenImage21PECanvas().build(*result[1:3])
