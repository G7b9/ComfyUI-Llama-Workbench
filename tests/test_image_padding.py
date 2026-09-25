from __future__ import annotations

import numpy as np
import pytest

from lwb.image_padding import (
    IMAGE_PADDING_INFO_TYPE,
    LlamaWorkbenchPadImageToMultiple,
    LlamaWorkbenchRestoreImageFromPadding,
    calculate_padding,
    pad_image_to_multiple,
    restore_image_from_padding,
)
from lwb.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS


class FakeTensor:
    def __init__(self, data):
        self.data = np.asarray(data, dtype=np.float32)
        self.shape = self.data.shape
        self.dtype = "torch.float32"
        self.device = "cpu"

    def numel(self):
        return self.data.size

    def element_size(self):
        return self.data.itemsize

    def new_empty(self, shape):
        return FakeTensor(np.empty(shape, dtype=self.data.dtype))

    def new_tensor(self, values):
        return FakeTensor(np.asarray(values, dtype=self.data.dtype))

    def reshape(self, *shape):
        return FakeTensor(self.data.reshape(*shape))

    def contiguous(self):
        return FakeTensor(np.ascontiguousarray(self.data))

    def __getitem__(self, key):
        return FakeTensor(self.data[key])

    def __setitem__(self, key, value):
        self.data[key] = value.data if isinstance(value, FakeTensor) else value


def sample_image(height=5, width=7, channels=3):
    values = np.arange(height * width * channels, dtype=np.float32).reshape(
        1, height, width, channels
    )
    return FakeTensor(values / max(1, values.max()))


def test_calculate_center_padding_distributes_odd_remainders_to_right_and_bottom():
    assert calculate_padding(7, 5, 4, "center") == {
        "padded_width": 8,
        "padded_height": 8,
        "pad_left": 0,
        "pad_top": 1,
        "pad_right": 1,
        "pad_bottom": 2,
    }


@pytest.mark.parametrize(
    "placement,expected",
    [
        ("top-left", (0, 0, 1, 3)),
        ("top-right", (1, 0, 0, 3)),
        ("bottom-left", (0, 3, 1, 0)),
        ("bottom-right", (1, 3, 0, 0)),
    ],
)
def test_calculate_padding_supports_corner_placements(placement, expected):
    info = calculate_padding(7, 5, 4, placement)

    assert (info["pad_left"], info["pad_top"], info["pad_right"], info["pad_bottom"]) == expected


def test_pad_image_to_multiple_preserves_content_and_uses_selected_rgb_color():
    image = sample_image()
    padded, info = pad_image_to_multiple(
        image,
        multiple=4,
        placement="center",
        color=0x336699,
    )

    assert padded.shape == (1, 8, 8, 3)
    np.testing.assert_allclose(padded.data[:, 1:6, 0:7, :], image.data)
    np.testing.assert_allclose(padded.data[0, 0, 0], [0x33 / 255, 0x66 / 255, 0x99 / 255])
    assert info == {
        "schema_version": 1,
        "original_width": 7,
        "original_height": 5,
        "padded_width": 8,
        "padded_height": 8,
        "pad_left": 0,
        "pad_top": 1,
        "pad_right": 1,
        "pad_bottom": 2,
        "multiple": 4,
        "placement": "center",
        "color": 0x336699,
        "color_hex": "#336699",
        "pad_alpha": 1.0,
        "batch_size": 1,
        "channels": 3,
    }


def test_pad_image_to_multiple_uses_alpha_for_rgba_padding():
    image = sample_image(channels=4)
    padded, _ = pad_image_to_multiple(
        image,
        multiple=4,
        placement="top-left",
        color=0xFF8000,
        pad_alpha=0.25,
    )

    np.testing.assert_allclose(padded.data[0, -1, -1], [1.0, 128 / 255, 0.0, 0.25])


def test_pad_image_to_multiple_returns_original_tensor_when_already_aligned():
    image = sample_image(height=8, width=16)
    padded, info = pad_image_to_multiple(image, multiple=8)

    assert padded is image
    assert (info["pad_left"], info["pad_top"], info["pad_right"], info["pad_bottom"]) == (
        0,
        0,
        0,
        0,
    )


@pytest.mark.parametrize("placement", ["center", "top-left", "bottom-right", "center-right"])
def test_restore_image_from_padding_round_trips_each_placement(placement):
    image = sample_image()
    padded, info = pad_image_to_multiple(image, 4, placement, 0x112233)

    restored = restore_image_from_padding(padded, info)

    assert restored.shape == image.shape
    np.testing.assert_allclose(restored.data, image.data)


def test_restore_crops_the_processed_image_not_the_original_input():
    image = sample_image()
    padded, info = pad_image_to_multiple(image, 4, "center", 0)
    processed = FakeTensor(padded.data + 0.125)

    restored = restore_image_from_padding(processed, info)

    np.testing.assert_allclose(restored.data, image.data + 0.125)


def test_restore_rejects_a_changed_padded_size_by_default():
    image = sample_image()
    padded, info = pad_image_to_multiple(image, 4)
    resized = FakeTensor(np.zeros((1, padded.shape[1] + 4, padded.shape[2], 3), dtype=np.float32))

    with pytest.raises(ValueError, match="does not match padding_info"):
        restore_image_from_padding(resized, info)


def test_padding_nodes_expose_native_color_widget_and_typed_metadata():
    pad_inputs = LlamaWorkbenchPadImageToMultiple.INPUT_TYPES()["required"]
    restore_inputs = LlamaWorkbenchRestoreImageFromPadding.INPUT_TYPES()["required"]

    assert pad_inputs["multiple"][1]["default"] == 16
    assert pad_inputs["color"][1]["display"] == "color"
    assert LlamaWorkbenchPadImageToMultiple.RETURN_TYPES[1] == IMAGE_PADDING_INFO_TYPE
    assert restore_inputs["padding_info"][0] == IMAGE_PADDING_INFO_TYPE
    assert restore_inputs["padding_info"][1]["forceInput"] is True


def test_padding_nodes_return_dimensions_and_are_registered():
    image = sample_image()
    outputs = LlamaWorkbenchPadImageToMultiple().pad(image, 4, "center", 0, 1.0)
    restored, width, height = LlamaWorkbenchRestoreImageFromPadding().restore(
        outputs[0], outputs[1]
    )

    assert outputs[2:] == (7, 5, 8, 8, 0, 1, 1, 2)
    assert (width, height) == (7, 5)
    np.testing.assert_allclose(restored.data, image.data)
    assert (
        NODE_CLASS_MAPPINGS["LlamaWorkbench_PadImageToMultiple"] is LlamaWorkbenchPadImageToMultiple
    )
    assert (
        NODE_CLASS_MAPPINGS["LlamaWorkbench_RestoreImageFromPadding"]
        is LlamaWorkbenchRestoreImageFromPadding
    )
    assert (
        NODE_DISPLAY_NAME_MAPPINGS["LlamaWorkbench_PadImageToMultiple"]
        == "Llama Workbench Pad Image to Multiple"
    )
    assert (
        NODE_DISPLAY_NAME_MAPPINGS["LlamaWorkbench_RestoreImageFromPadding"]
        == "Llama Workbench Restore Image from Padding"
    )
