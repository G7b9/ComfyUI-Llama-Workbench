"""Seed utility node for Llama Workbench."""

from __future__ import annotations

from typing import Any


SEED_MAX = 0x7FFFFFFF


class LlamaWorkbenchSeed:
    """Output a seed using ComfyUI's native post-generation controls."""

    CATEGORY = "Llama Workbench / Utility"
    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("SEED",)
    FUNCTION = "generate"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": SEED_MAX,
                        "control_after_generate": True,
                        "tooltip": "Use ComfyUI's native control to keep the seed fixed, randomize it, or increment/decrement it after generation.",
                    },
                ),
            }
        }

    def generate(self, seed: Any = 0):
        try:
            value = int(seed)
        except (TypeError, ValueError) as exc:
            raise ValueError("seed must be an integer") from exc
        if value < 0 or value > SEED_MAX:
            raise ValueError(f"seed must be between 0 and {SEED_MAX}")
        return (value,)
