"""Collect mapped IMAGE outputs and batches into a single A/B preview."""

from __future__ import annotations

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


class LlamaWorkbenchImageCompare:
    CATEGORY = "Llama Workbench / Utility"
    OUTPUT_NODE = True
    INPUT_IS_LIST = True
    RETURN_TYPES = ()
    FUNCTION = "compare"
    DESCRIPTION = "Preview all A/B image pairs, including batches and mapped lists. Optionally repeat the shorter group's last image."

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
            }
        }

    def compare(self, image_a, image_b, repeat=None):
        batches_a = image_batches(image_a, "image_a")
        batches_b = image_batches(image_b, "image_b")
        # Loaded only during execution inside the ComfyUI host.
        from nodes import PreviewImage

        previewer = PreviewImage()
        prefix = f"LlamaWorkbench_compare_{uuid4().hex}"
        repeat_value = repeat[0] if isinstance(repeat, (list, tuple)) and repeat else repeat
        return {
            "ui": {
                "comparison": [{
                    "a": save_preview_group(previewer, batches_a, f"{prefix}_A"),
                    "b": save_preview_group(previewer, batches_b, f"{prefix}_B"),
                    "repeat": bool(repeat_value),
                }],
            }
        }
