from __future__ import annotations

from lwb.nodes import H3_AUTO_ASPECT_RATIO, H3_ASPECT_RATIOS, LlamaWorkbenchH3AutoResolutionSelector


class FakeImage:
    def __init__(self, height: int, width: int):
        self.shape = (1, height, width, 3)


def test_h3_resolution_selector_uses_the_nearest_supported_input_aspect_ratio():
    selector = LlamaWorkbenchH3AutoResolutionSelector()
    width, height, selected = selector.select(FakeImage(1600, 900), megapixels=1.0, multiple=8)
    assert selected == "9:16 (Portrait Widescreen)"
    assert width % 8 == 0 and height % 8 == 0
    assert abs(width / height - 9 / 16) < 0.005

    width, height, selected = selector.select(FakeImage(900, 1600), megapixels=1.0, multiple=8)
    assert selected == "16:9 (Widescreen)"
    assert width % 8 == 0 and height % 8 == 0
    assert abs(width / height - 16 / 9) < 0.005


def test_h3_resolution_selector_matches_the_given_selector_formula_and_allows_manual_ratio():
    selector = LlamaWorkbenchH3AutoResolutionSelector()
    width, height, selected = selector.select(FakeImage(900, 1600), aspect_ratio="1:1 (Square)", megapixels=1.0, multiple=8)
    assert (width, height, selected) == (1024, 1024, "1:1 (Square)")

    inputs = LlamaWorkbenchH3AutoResolutionSelector.INPUT_TYPES()["required"]
    assert inputs["aspect_ratio"][0] == [H3_AUTO_ASPECT_RATIO, *H3_ASPECT_RATIOS]
    assert inputs["megapixels"][1]["default"] == 1.0
    assert inputs["multiple"][1]["default"] == 8
