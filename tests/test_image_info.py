from __future__ import annotations

import json
from pathlib import Path

import pytest

from lwb.image_info import (
    IMAGE_INFO_TYPE,
    LlamaWorkbenchImageInfo,
    LlamaWorkbenchImageInfoDisplay,
    build_image_info,
    format_image_info,
)
from lwb.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS


class FakeImage:
    def __init__(
        self,
        shape: tuple[int, ...],
        *,
        dtype: str = "torch.float16",
        device: str = "cuda:0",
        element_size: int = 2,
    ):
        self.shape = shape
        self.dtype = dtype
        self.device = device
        self._element_size = element_size

    def numel(self):
        result = 1
        for value in self.shape:
            result *= value
        return result

    def element_size(self):
        return self._element_size


def test_build_image_info_reports_geometry_batch_and_tensor_metadata():
    info = build_image_info(FakeImage((2, 1080, 1920, 4)))

    assert info["width"] == 1920
    assert info["height"] == 1080
    assert info["longest_side"] == 1920
    assert info["shortest_side"] == 1080
    assert info["total_pixels"] == 2_073_600
    assert info["megapixels"] == pytest.approx(2.0736)
    assert info["aspect_ratio"] == "16:9"
    assert info["aspect_ratio_value"] == pytest.approx(16 / 9)
    assert info["orientation"] == "landscape"
    assert info["batch_size"] == 2
    assert info["batch_total_pixels"] == 4_147_200
    assert info["channels"] == 4
    assert info["color_model"] == "RGBA"
    assert info["has_alpha"] is True
    assert info["tensor_shape"] == [2, 1080, 1920, 4]
    assert info["tensor_layout"] == "BHWC"
    assert info["dtype"] == "float16"
    assert info["device"] == "cuda:0"
    assert info["estimated_tensor_bytes"] == 33_177_600


@pytest.mark.parametrize(
    "shape,ratio,orientation",
    [
        ((1, 1024, 1024, 3), "1:1", "square"),
        ((1, 1600, 900, 3), "9:16", "portrait"),
        ((1, 1000, 1500, 3), "3:2", "landscape"),
    ],
)
def test_build_image_info_reduces_aspect_ratio_and_detects_orientation(shape, ratio, orientation):
    info = build_image_info(FakeImage(shape))

    assert info["aspect_ratio"] == ratio
    assert info["orientation"] == orientation


@pytest.mark.parametrize("shape", [(1080, 1920, 3), (0, 1080, 1920, 3), (1, -1, 100, 3)])
def test_build_image_info_rejects_invalid_image_shapes(shape):
    with pytest.raises(ValueError, match="IMAGE tensor|positive"):
        build_image_info(FakeImage(shape))


def test_image_info_node_exposes_required_scalar_and_structured_outputs():
    node = LlamaWorkbenchImageInfo()
    outputs = node.inspect(FakeImage((3, 720, 1280, 3), dtype="torch.float32", device="cpu", element_size=4))

    assert outputs[:10] == (1280, 720, 1280, 720, 921_600, 0.9216, "16:9", "landscape", 3, 3)
    assert outputs[10]["batch_total_pixels"] == 2_764_800
    assert LlamaWorkbenchImageInfo.RETURN_TYPES[-1] == IMAGE_INFO_TYPE


def test_image_info_display_formats_detailed_compact_and_json_outputs():
    info = build_image_info(FakeImage((2, 1080, 1920, 4)))

    detailed_zh = format_image_info(info, "Detailed", "中文")
    compact_en = format_image_info(info, "Compact", "English")
    json_text = format_image_info(info, "JSON", "English")

    assert "分辨率: 1920 x 1080 px" in detailed_zh
    assert "最长边: 1920 px" in detailed_zh
    assert "批次总像素: 4,147,200" in detailed_zh
    assert "估算张量内存: 31.64 MiB" in detailed_zh
    assert "1920 x 1080 px | 16:9 | landscape" in compact_en
    assert json.loads(json_text) == info


def test_image_info_display_returns_canvas_ui_and_reusable_text():
    info = build_image_info(FakeImage((1, 800, 600, 3)))
    result = LlamaWorkbenchImageInfoDisplay().display(info, "Detailed", "English")

    formatted, info_json = result["result"]
    assert "Resolution: 600 x 800 px" in formatted
    assert json.loads(info_json)["orientation"] == "portrait"
    assert result["ui"]["formatted_info"] == [formatted]
    assert result["ui"]["title"] == ["Image Information"]


def test_image_info_nodes_are_registered():
    assert NODE_CLASS_MAPPINGS["LlamaWorkbench_ImageInfo"] is LlamaWorkbenchImageInfo
    assert NODE_CLASS_MAPPINGS["LlamaWorkbench_ImageInfoDisplay"] is LlamaWorkbenchImageInfoDisplay
    assert NODE_DISPLAY_NAME_MAPPINGS["LlamaWorkbench_ImageInfo"] == "Llama Workbench Image Info"
    assert (
        NODE_DISPLAY_NAME_MAPPINGS["LlamaWorkbench_ImageInfoDisplay"]
        == "Llama Workbench Image Info Display"
    )


def test_image_info_frontend_adds_a_canvas_display_without_replacing_node_type():
    source = (Path(__file__).parents[1] / "web" / "image_info.js").read_text(encoding="utf-8")

    assert 'name: "LlamaWorkbench.ImageInfoDisplay"' in source
    assert "beforeRegisterNodeDef" in source
    assert 'const DISPLAY_NODE = "LlamaWorkbench_ImageInfoDisplay"' in source
    assert 'api.addEventListener("executed"' in source
    assert 'outputValue(output, "formatted_info")' in source
    assert "LiteGraph.registerNodeType" not in source
