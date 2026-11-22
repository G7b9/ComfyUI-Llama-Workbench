"""Collect mapped IMAGE outputs and batches into a single A/B preview."""

from __future__ import annotations

import operator
from typing import Any
from uuid import uuid4


def image_batches(images: Any, group: str) -> list[Any]:
    """Preserve list order, batch order, and heterogeneous image dimensions."""
    if isinstance(images, (list, tuple)):
        return [batch for item in images for batch in image_batches(item, group)]
    shape = getattr(images, "shape", ())
    if len(shape) != 4 or shape[-1] not in (3, 4) or shape[1] <= 0 or shape[2] <= 0:
        raise ValueError(f"{group} requires IMAGE tensors shaped [B, H, W, 3 or 4]")
    return [images] if shape[0] else []


def save_preview_group(previewer: Any, batches: list[Any], prefix: str) -> list[dict]:
    images = []
    for index, batch in enumerate(batches):
        # Use ComfyUI's temporary PNG previews, including RGBA, at original resolution.
        saved = previewer.save_images(batch, filename_prefix=f"{prefix}_{index}")
        images.extend(
            {**entry, "width": int(batch.shape[2]), "height": int(batch.shape[1])}
            for entry in saved["ui"]["images"]
        )
    return images


def scalar_input(value, default=None):
    """Widget values are singleton lists when INPUT_IS_LIST is enabled."""
    if isinstance(value, (list, tuple)):
        return value[0] if value else default
    return default if value is None else value


def selected_image(batches: list[Any], index: int, repeat: bool):
    count = sum(int(batch.shape[0]) for batch in batches)
    if not count or index < 0 or (index >= count and not repeat):
        return None
    remaining = min(index, count - 1)
    for batch in batches:
        if remaining < batch.shape[0]:
            # Keep a batch dimension and preserve original pixels, alpha, dtype and device.
            return batch[remaining:remaining + 1]
        remaining -= batch.shape[0]
    return None


class LlamaWorkbenchImageCompare:
    CATEGORY = "Llama Workbench / Utility"
    OUTPUT_NODE = True
    INPUT_IS_LIST = True
    RETURN_TYPES = ("INT", "IMAGE", "IMAGE")
    RETURN_NAMES = ("index", "selected_a", "selected_b")
    FUNCTION = "compare"
    DESCRIPTION = "Preview all A/B image pairs, repeat the shorter group's last image, and optionally output one selected pair with its zero-based index."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_a": ("IMAGE", {"tooltip": "Group A: one image, an image batch, or a list of image batches."}),
                "image_b": ("IMAGE", {"tooltip": "Group B: paired with A in list order, then batch order."}),
                "repeat": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "Repeat the shorter group's last image. Can be toggled after preview without rerunning. Off shows missing-image placeholders."},
                ),
            },
            "optional": {
                "selection_enabled": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "Show one-choice selection below each pair. Run again to send the selection downstream."},
                ),
                "selected_index": (
                    "INT",
                    {"default": -1, "min": -1, "max": 2147483647, "control_after_generate": False,
                     "tooltip": "Zero-based pair index; -1 means no selection. Updated by the pair radio buttons. Missing pairs block downstream outputs."},
                ),
            },
        }

    def compare(self, image_a, image_b, repeat=None, selection_enabled=False, selected_index=-1):
        batches_a = image_batches(image_a, "image_a")
        batches_b = image_batches(image_b, "image_b")
        # Loaded only during execution inside the ComfyUI host.
        from nodes import PreviewImage
        from comfy_execution.graph import ExecutionBlocker

        previewer = PreviewImage()
        prefix = f"LlamaWorkbench_compare_{uuid4().hex}"
        repeat_value = bool(scalar_input(repeat, False))
        enabled = bool(scalar_input(selection_enabled, False))
        try:
            index = operator.index(scalar_input(selected_index, -1))
        except TypeError as exc:
            raise ValueError("selected_index must be an integer") from exc
        count = max(
            sum(int(batch.shape[0]) for batch in batches) for batches in (batches_a, batches_b)
        )
        selected_a = selected_b = None
        if enabled and 0 <= index < count:
            selected_a = selected_image(batches_a, index, repeat_value)
            selected_b = selected_image(batches_b, index, repeat_value)
        if selected_a is not None and selected_b is not None:
            outputs = (index, selected_a, selected_b)
        else:
            outputs = (ExecutionBlocker(None), ExecutionBlocker(None), ExecutionBlocker(None))
        return {
            "ui": {
                "comparison": [{
                    "a": save_preview_group(previewer, batches_a, f"{prefix}_A"),
                    "b": save_preview_group(previewer, batches_b, f"{prefix}_B"),
                    "repeat": bool(repeat_value),
                    "selection_enabled": enabled,
                    "selected_index": index,
                }],
            },
            "result": outputs,
        }
