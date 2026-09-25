"""Image tensor metadata and display nodes for Llama Workbench."""

from __future__ import annotations

import json
import math
import time
from functools import reduce
from operator import mul
from typing import Any, Mapping


IMAGE_INFO_TYPE = "LLAMA_WORKBENCH_IMAGE_INFO"
IMAGE_INFO_FORMATS = ("Detailed", "Compact", "JSON")
IMAGE_INFO_LANGUAGES = ("中文", "English")


def _tensor_shape(image: Any) -> tuple[int, int, int, int]:
    shape = getattr(image, "shape", None)
    if shape is None:
        raise ValueError("image must be a ComfyUI IMAGE tensor shaped [B, H, W, C]")
    try:
        dimensions = tuple(int(value) for value in shape)
    except (TypeError, ValueError) as exc:
        raise ValueError("image tensor shape must contain integer dimensions") from exc
    if len(dimensions) != 4:
        raise ValueError("image must be a ComfyUI IMAGE tensor shaped [B, H, W, C]")
    if any(value <= 0 for value in dimensions):
        raise ValueError("image tensor dimensions must be positive")
    return dimensions


def _clean_tensor_label(value: Any, prefix: str = "") -> str:
    label = str(value if value is not None else "unknown")
    return label[len(prefix) :] if prefix and label.startswith(prefix) else label


def _tensor_bytes(image: Any, shape: tuple[int, ...], dtype: str) -> int:
    try:
        numel = int(image.numel())
        element_size = int(image.element_size())
        if numel >= 0 and element_size > 0:
            return numel * element_size
    except (AttributeError, TypeError, ValueError):
        pass

    byte_sizes = {
        "bool": 1,
        "uint8": 1,
        "int8": 1,
        "float16": 2,
        "bfloat16": 2,
        "int16": 2,
        "float32": 4,
        "int32": 4,
        "float64": 8,
        "int64": 8,
    }
    return reduce(mul, shape, 1) * byte_sizes.get(dtype, 0)


def build_image_info(image: Any) -> dict[str, Any]:
    """Return JSON-safe metadata for one ComfyUI IMAGE batch."""

    batch_size, height, width, channels = _tensor_shape(image)
    common_divisor = math.gcd(width, height)
    total_pixels = width * height
    batch_total_pixels = total_pixels * batch_size
    orientation = "square" if width == height else "landscape" if width > height else "portrait"
    color_model = {
        1: "Gray",
        2: "Gray+Alpha",
        3: "RGB",
        4: "RGBA",
    }.get(channels, f"{channels}-channel")
    dtype = _clean_tensor_label(getattr(image, "dtype", None), "torch.")
    device = _clean_tensor_label(getattr(image, "device", None))
    shape = (batch_size, height, width, channels)
    estimated_tensor_bytes = _tensor_bytes(image, shape, dtype)

    return {
        "schema_version": 1,
        "width": width,
        "height": height,
        "resolution": f"{width}x{height}",
        "longest_side": max(width, height),
        "shortest_side": min(width, height),
        "total_pixels": total_pixels,
        "megapixels": total_pixels / 1_000_000,
        "aspect_ratio": f"{width // common_divisor}:{height // common_divisor}",
        "aspect_ratio_value": width / height,
        "orientation": orientation,
        "batch_size": batch_size,
        "batch_total_pixels": batch_total_pixels,
        "batch_megapixels": batch_total_pixels / 1_000_000,
        "channels": channels,
        "color_model": color_model,
        "has_alpha": channels in {2, 4},
        "tensor_shape": list(shape),
        "tensor_layout": "BHWC",
        "dtype": dtype,
        "device": device,
        "estimated_tensor_bytes": estimated_tensor_bytes,
    }


def _format_bytes(value: Any) -> str:
    try:
        size = max(0, int(value))
    except (TypeError, ValueError):
        return "unknown"
    if size == 0:
        return "unknown"
    amount = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if amount < 1024 or unit == "TiB":
            return f"{amount:.0f} {unit}" if unit == "B" else f"{amount:.2f} {unit}"
        amount /= 1024
    return f"{size} B"


def _validated_info(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("image_info must come from Llama Workbench Image Info")
    required = {
        "width",
        "height",
        "longest_side",
        "shortest_side",
        "total_pixels",
        "megapixels",
        "aspect_ratio",
        "orientation",
        "batch_size",
        "batch_total_pixels",
        "batch_megapixels",
        "channels",
        "color_model",
        "tensor_shape",
        "tensor_layout",
        "dtype",
        "device",
        "estimated_tensor_bytes",
    }
    missing = required - value.keys()
    if missing:
        raise ValueError(f"image_info is missing field(s): {', '.join(sorted(missing))}")
    return dict(value)


def format_image_info(
    image_info: Any,
    display_format: str = "Detailed",
    language: str = "中文",
) -> str:
    """Format image metadata for canvas display or downstream text nodes."""

    info = _validated_info(image_info)
    selected_format = str(display_format or "Detailed")
    if selected_format not in IMAGE_INFO_FORMATS:
        raise ValueError(f"Unknown image info format: {selected_format}")
    selected_language = str(language or "中文")
    if selected_language not in IMAGE_INFO_LANGUAGES:
        raise ValueError(f"Unknown image info language: {selected_language}")
    if selected_format == "JSON":
        return json.dumps(info, ensure_ascii=False, indent=2)

    width = int(info["width"])
    height = int(info["height"])
    megapixels = float(info["megapixels"])
    batch_megapixels = float(info["batch_megapixels"])
    ratio_value = float(info.get("aspect_ratio_value", width / height))
    orientation_en = str(info["orientation"])
    orientation_zh = {
        "landscape": "横向",
        "portrait": "纵向",
        "square": "正方形",
    }.get(orientation_en, orientation_en)
    shape = "[" + ", ".join(str(value) for value in info["tensor_shape"]) + "]"
    memory = _format_bytes(info["estimated_tensor_bytes"])

    if selected_format == "Compact":
        if selected_language == "中文":
            return (
                f"{width} x {height} px | {info['aspect_ratio']} | {orientation_zh} | "
                f"{megapixels:.3f} MP/张 | 批次 {info['batch_size']} | "
                f"{info['color_model']} | {info['dtype']} | {info['device']}"
            )
        return (
            f"{width} x {height} px | {info['aspect_ratio']} | {orientation_en} | "
            f"{megapixels:.3f} MP/image | batch {info['batch_size']} | "
            f"{info['color_model']} | {info['dtype']} | {info['device']}"
        )

    if selected_language == "中文":
        return "\n".join(
            (
                f"分辨率: {width} x {height} px",
                f"方向: {orientation_zh}",
                f"宽高比: {info['aspect_ratio']} ({ratio_value:.4f})",
                f"最长边: {info['longest_side']} px",
                f"最短边: {info['shortest_side']} px",
                f"单张总像素: {int(info['total_pixels']):,} ({megapixels:.3f} MP)",
                f"批次数量: {info['batch_size']}",
                f"批次总像素: {int(info['batch_total_pixels']):,} ({batch_megapixels:.3f} MP)",
                f"通道: {info['channels']} ({info['color_model']})",
                f"张量形状: {shape} ({info['tensor_layout']})",
                f"数据类型: {info['dtype']}",
                f"设备: {info['device']}",
                f"估算张量内存: {memory}",
            )
        )
    return "\n".join(
        (
            f"Resolution: {width} x {height} px",
            f"Orientation: {orientation_en}",
            f"Aspect ratio: {info['aspect_ratio']} ({ratio_value:.4f})",
            f"Longest side: {info['longest_side']} px",
            f"Shortest side: {info['shortest_side']} px",
            f"Pixels per image: {int(info['total_pixels']):,} ({megapixels:.3f} MP)",
            f"Batch size: {info['batch_size']}",
            f"Batch pixels: {int(info['batch_total_pixels']):,} ({batch_megapixels:.3f} MP)",
            f"Channels: {info['channels']} ({info['color_model']})",
            f"Tensor shape: {shape} ({info['tensor_layout']})",
            f"Data type: {info['dtype']}",
            f"Device: {info['device']}",
            f"Estimated tensor memory: {memory}",
        )
    )


class LlamaWorkbenchImageInfo:
    """Inspect one ComfyUI IMAGE batch without copying pixel data."""

    CATEGORY = "Llama Workbench / Image"
    RETURN_TYPES = (
        "INT",
        "INT",
        "INT",
        "INT",
        "INT",
        "FLOAT",
        "STRING",
        "STRING",
        "INT",
        "INT",
        IMAGE_INFO_TYPE,
    )
    RETURN_NAMES = (
        "width",
        "height",
        "longest_side",
        "shortest_side",
        "total_pixels",
        "megapixels",
        "aspect_ratio",
        "orientation",
        "batch_size",
        "channels",
        "image_info",
    )
    FUNCTION = "inspect"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": (
                    "IMAGE",
                    {
                        "tooltip": "ComfyUI IMAGE batch shaped [B, H, W, C]. Metadata is read without copying pixels.",
                    },
                )
            }
        }

    def inspect(self, image: Any):
        info = build_image_info(image)
        return (
            info["width"],
            info["height"],
            info["longest_side"],
            info["shortest_side"],
            info["total_pixels"],
            info["megapixels"],
            info["aspect_ratio"],
            info["orientation"],
            info["batch_size"],
            info["channels"],
            info,
        )


class LlamaWorkbenchImageInfoDisplay:
    """Format and display all metadata produced by :class:`LlamaWorkbenchImageInfo`."""

    CATEGORY = "Llama Workbench / Image"
    OUTPUT_NODE = True
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("formatted_info", "info_json")
    FUNCTION = "display"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_info": (
                    IMAGE_INFO_TYPE,
                    {"forceInput": True, "tooltip": "Connect Llama Workbench Image Info's image_info output."},
                ),
                "display_format": (
                    list(IMAGE_INFO_FORMATS),
                    {"default": "Detailed", "tooltip": "Detailed, one-line Compact, or pretty JSON output."},
                ),
                "language": (
                    list(IMAGE_INFO_LANGUAGES),
                    {"default": "中文", "tooltip": "Label language for Detailed and Compact output."},
                ),
            }
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def display(
        self,
        image_info: Any,
        display_format: str = "Detailed",
        language: str = "中文",
    ):
        info = _validated_info(image_info)
        formatted = format_image_info(info, display_format, language)
        info_json = json.dumps(info, ensure_ascii=False, indent=2)
        title = "图片信息" if language == "中文" else "Image Information"
        return {
            "ui": {
                "formatted_info": [formatted],
                "info_json": [info_json],
                "title": [title],
                "updated_at": [int(time.time() * 1000)],
            },
            "result": (formatted, info_json),
        }
