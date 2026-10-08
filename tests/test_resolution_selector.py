from __future__ import annotations

import json

import pytest

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
    assert inputs["multiple"][1]["default"] == 32


def test_h3_resolution_selector_input_and_output_contract():
    inputs = LlamaWorkbenchH3AutoResolutionSelector.INPUT_TYPES()["required"]
    assert list(inputs) == ["image", "aspect_ratio", "megapixels", "multiple"]
    assert inputs["image"][0] == "IMAGE"
    assert inputs["megapixels"][0] == "FLOAT"
    assert inputs["megapixels"][1] | {"tooltip": ""} == {
        "default": 1.0, "min": 0.1, "max": 16.0,
        "step": 0.01, "round": 0.01, "tooltip": "",
    }
    assert inputs["multiple"][0] == "INT"
    assert inputs["multiple"][1] | {"tooltip": ""} == {
        "default": 32, "min": 8, "max": 128, "step": 4, "tooltip": "",
    }
    assert LlamaWorkbenchH3AutoResolutionSelector.RETURN_TYPES == ("INT", "INT", "STRING")
    assert LlamaWorkbenchH3AutoResolutionSelector.RETURN_NAMES == (
        "width", "height", "selected_aspect_ratio",
    )
    assert LlamaWorkbenchH3AutoResolutionSelector.FUNCTION == "select"


@pytest.mark.parametrize(("megapixels", "expected"), [
    (0.90, (1280, 736)), (0.91, (1312, 736)),
    (0.92, (1312, 736)), (0.94, (1312, 736)),
    (0.95, (1344, 736)), (0.96, (1344, 768)),
    (0.98, (1344, 768)), (1.00, (1376, 768)),
])
def test_h3_resolution_selector_two_decimal_megapixels(megapixels, expected):
    selector = LlamaWorkbenchH3AutoResolutionSelector()
    image = FakeImage(720, 1280)
    selected = "16:9 (Widescreen)"
    assert selector.select(image, selected, megapixels, 32) == (*expected, selected)
    assert selector.select(image, megapixels=megapixels, multiple=32) == (*expected, selected)


@pytest.mark.parametrize(("image", "selected", "expected"), [
    (FakeImage(720, 1280), "16:9 (Widescreen)", (1344, 768)),
    (FakeImage(1280, 720), "9:16 (Portrait Widescreen)", (768, 1344)),
    (FakeImage(720, 720), "1:1 (Square)", (1024, 1024)),
])
def test_h3_resolution_selector_auto_aligned_ratios(image, selected, expected):
    assert LlamaWorkbenchH3AutoResolutionSelector().select(
        image, megapixels=0.98, multiple=32,
    ) == (*expected, selected)


@pytest.mark.parametrize("megapixels", [0.92, 0.98])
@pytest.mark.parametrize("multiple", [8, 16, 32])
def test_h3_resolution_selector_saved_parameters_and_float_input(megapixels, multiple):
    selector = LlamaWorkbenchH3AutoResolutionSelector()
    parameters = {
        "aspect_ratio": "16:9 (Widescreen)",
        "megapixels": megapixels, "multiple": multiple,
    }
    restored = json.loads(json.dumps(parameters))
    assert restored == parameters
    image = FakeImage(720, 1280)
    assert selector.select(image, **restored) == selector.select(
        image, "16:9 (Widescreen)", float(megapixels), multiple,
    )
    if multiple == 8:
        assert selector.select(image, **restored)[:2] == {
            0.92: (1312, 736), 0.98: (1352, 760),
        }[megapixels]
