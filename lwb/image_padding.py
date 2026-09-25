"""Reversible image padding nodes for Llama Workbench."""

from __future__ import annotations

from typing import Any, Mapping

from .image_info import build_image_info


IMAGE_PADDING_INFO_TYPE = "LLAMA_WORKBENCH_IMAGE_PADDING_INFO"
IMAGE_PADDING_PLACEMENTS = (
    "center",
    "top-left",
    "top-center",
    "top-right",
    "center-left",
    "center-right",
    "bottom-left",
    "bottom-center",
    "bottom-right",
)


def _normalized_multiple(value: Any) -> int:
    try:
        multiple = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("multiple must be a positive integer") from exc
    if multiple <= 0:
        raise ValueError("multiple must be a positive integer")
    return multiple


def _padding_split(extra: int, alignment: str) -> tuple[int, int]:
    if alignment in {"left", "top"}:
        return 0, extra
    if alignment in {"right", "bottom"}:
        return extra, 0
    leading = extra // 2
    return leading, extra - leading


def calculate_padding(
    width: int,
    height: int,
    multiple: int,
    placement: str = "center",
) -> dict[str, int]:
    """Calculate exact padding needed to reach a spatial multiple."""

    source_width = int(width)
    source_height = int(height)
    if source_width <= 0 or source_height <= 0:
        raise ValueError("image width and height must be positive")
    normalized_multiple = _normalized_multiple(multiple)
    normalized_placement = str(placement or "center")
    if normalized_placement not in IMAGE_PADDING_PLACEMENTS:
        raise ValueError(f"Unknown padding placement: {normalized_placement}")

    padded_width = (
        (source_width + normalized_multiple - 1) // normalized_multiple
    ) * normalized_multiple
    padded_height = (
        (source_height + normalized_multiple - 1) // normalized_multiple
    ) * normalized_multiple
    extra_width = padded_width - source_width
    extra_height = padded_height - source_height

    if normalized_placement == "center":
        vertical, horizontal = "center", "center"
    else:
        vertical, horizontal = normalized_placement.split("-", 1)
    pad_left, pad_right = _padding_split(extra_width, horizontal)
    pad_top, pad_bottom = _padding_split(extra_height, vertical)
    return {
        "padded_width": padded_width,
        "padded_height": padded_height,
        "pad_left": pad_left,
        "pad_top": pad_top,
        "pad_right": pad_right,
        "pad_bottom": pad_bottom,
    }


def _color_components(color: Any, alpha: Any, channels: int) -> list[float]:
    try:
        color_value = int(color)
    except (TypeError, ValueError) as exc:
        raise ValueError("color must be an RGB integer between 0x000000 and 0xFFFFFF") from exc
    if color_value < 0 or color_value > 0xFFFFFF:
        raise ValueError("color must be an RGB integer between 0x000000 and 0xFFFFFF")
    try:
        alpha_value = float(alpha)
    except (TypeError, ValueError) as exc:
        raise ValueError("pad_alpha must be between 0 and 1") from exc
    if not 0.0 <= alpha_value <= 1.0:
        raise ValueError("pad_alpha must be between 0 and 1")

    red = ((color_value >> 16) & 0xFF) / 255.0
    green = ((color_value >> 8) & 0xFF) / 255.0
    blue = (color_value & 0xFF) / 255.0
    gray = 0.299 * red + 0.587 * green + 0.114 * blue
    if channels == 1:
        return [gray]
    if channels == 2:
        return [gray, alpha_value]
    values = [red, green, blue]
    if channels >= 4:
        values.append(alpha_value)
    if channels > 4:
        values.extend([0.0] * (channels - 4))
    return values


def _padding_info(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("padding_info must come from Llama Workbench Pad Image to Multiple")
    required = {
        "original_width",
        "original_height",
        "padded_width",
        "padded_height",
        "pad_left",
        "pad_top",
        "pad_right",
        "pad_bottom",
    }
    missing = required - value.keys()
    if missing:
        raise ValueError(f"padding_info is missing field(s): {', '.join(sorted(missing))}")
    try:
        info = dict(value)
        for key in required:
            info[key] = int(info[key])
    except (TypeError, ValueError) as exc:
        raise ValueError("padding_info dimensions must be integers") from exc
    if info["original_width"] <= 0 or info["original_height"] <= 0:
        raise ValueError("padding_info original dimensions must be positive")
    if any(info[key] < 0 for key in ("pad_left", "pad_top", "pad_right", "pad_bottom")):
        raise ValueError("padding_info padding values must be non-negative")
    expected_width = info["original_width"] + info["pad_left"] + info["pad_right"]
    expected_height = info["original_height"] + info["pad_top"] + info["pad_bottom"]
    if expected_width != info["padded_width"] or expected_height != info["padded_height"]:
        raise ValueError("padding_info dimensions are internally inconsistent")
    return info


def pad_image_to_multiple(
    image: Any,
    multiple: int = 16,
    placement: str = "center",
    color: int = 0,
    pad_alpha: float = 1.0,
) -> tuple[Any, dict[str, Any]]:
    """Pad a BHWC image batch while preserving dtype and device."""

    source = build_image_info(image)
    padding = calculate_padding(source["width"], source["height"], multiple, placement)
    color_values = _color_components(color, pad_alpha, source["channels"])
    info = {
        "schema_version": 1,
        "original_width": source["width"],
        "original_height": source["height"],
        **padding,
        "multiple": _normalized_multiple(multiple),
        "placement": str(placement or "center"),
        "color": int(color),
        "color_hex": f"#{int(color):06X}",
        "pad_alpha": float(pad_alpha),
        "batch_size": source["batch_size"],
        "channels": source["channels"],
    }
    if padding["padded_width"] == source["width"] and padding["padded_height"] == source["height"]:
        return image, info

    new_empty = getattr(image, "new_empty", None)
    new_tensor = getattr(image, "new_tensor", None)
    if not callable(new_empty) or not callable(new_tensor):
        raise ValueError("image must support ComfyUI/PyTorch tensor allocation")
    padded = new_empty(
        (
            source["batch_size"],
            padding["padded_height"],
            padding["padded_width"],
            source["channels"],
        )
    )
    fill = new_tensor(color_values).reshape(1, 1, 1, source["channels"])
    padded[...] = fill
    top = padding["pad_top"]
    left = padding["pad_left"]
    padded[:, top : top + source["height"], left : left + source["width"], :] = image
    return padded, info


def restore_image_from_padding(
    image: Any,
    padding_info: Any,
    *,
    strict_size: bool = True,
) -> Any:
    """Crop the original content rectangle from a processed padded image."""

    current = build_image_info(image)
    info = _padding_info(padding_info)
    if strict_size and (
        current["width"] != info["padded_width"] or current["height"] != info["padded_height"]
    ):
        raise ValueError(
            "processed image size does not match padding_info: "
            f"got {current['width']}x{current['height']}, expected "
            f"{info['padded_width']}x{info['padded_height']}"
        )
    left = info["pad_left"]
    top = info["pad_top"]
    right = left + info["original_width"]
    bottom = top + info["original_height"]
    if right > current["width"] or bottom > current["height"]:
        raise ValueError("processed image is too small for the crop recorded in padding_info")
    restored = image[:, top:bottom, left:right, :]
    contiguous = getattr(restored, "contiguous", None)
    return contiguous() if callable(contiguous) else restored


class LlamaWorkbenchPadImageToMultiple:
    """Pad image width and height to a selected multiple."""

    CATEGORY = "Llama Workbench / Image"
    RETURN_TYPES = (
        "IMAGE",
        IMAGE_PADDING_INFO_TYPE,
        "INT",
        "INT",
        "INT",
        "INT",
        "INT",
        "INT",
        "INT",
        "INT",
    )
    RETURN_NAMES = (
        "padded_image",
        "padding_info",
        "original_width",
        "original_height",
        "padded_width",
        "padded_height",
        "pad_left",
        "pad_top",
        "pad_right",
        "pad_bottom",
    )
    FUNCTION = "pad"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "multiple": (
                    "INT",
                    {
                        "default": 16,
                        "min": 1,
                        "max": 4096,
                        "step": 1,
                        "tooltip": "Pad width and height up to the nearest multiple, such as 8, 16, 32, or 64.",
                    },
                ),
                "placement": (
                    list(IMAGE_PADDING_PLACEMENTS),
                    {
                        "default": "center",
                        "tooltip": "Position of the original image inside the padded canvas.",
                    },
                ),
                "color": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFF,
                        "step": 1,
                        "display": "color",
                        "tooltip": "Padding color.",
                    },
                ),
                "pad_alpha": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Padding alpha for RGBA or Gray+Alpha images; ignored for RGB images.",
                    },
                ),
            }
        }

    def pad(
        self,
        image: Any,
        multiple: int = 16,
        placement: str = "center",
        color: int = 0,
        pad_alpha: float = 1.0,
    ):
        padded, info = pad_image_to_multiple(image, multiple, placement, color, pad_alpha)
        return (
            padded,
            info,
            info["original_width"],
            info["original_height"],
            info["padded_width"],
            info["padded_height"],
            info["pad_left"],
            info["pad_top"],
            info["pad_right"],
            info["pad_bottom"],
        )


class LlamaWorkbenchRestoreImageFromPadding:
    """Restore the original content rectangle recorded by the padding node."""

    CATEGORY = "Llama Workbench / Image"
    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("restored_image", "width", "height")
    FUNCTION = "restore"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": (
                    "IMAGE",
                    {"tooltip": "Processed image that still has the padded spatial dimensions."},
                ),
                "padding_info": (
                    IMAGE_PADDING_INFO_TYPE,
                    {
                        "forceInput": True,
                        "tooltip": "Connect Pad Image to Multiple's padding_info output.",
                    },
                ),
                "strict_size": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Reject images whose size no longer matches the recorded padded size.",
                    },
                ),
            }
        }

    def restore(self, image: Any, padding_info: Any, strict_size: bool = True):
        info = _padding_info(padding_info)
        restored = restore_image_from_padding(image, info, strict_size=bool(strict_size))
        return restored, info["original_width"], info["original_height"]
