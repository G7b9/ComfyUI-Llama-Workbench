"""Seed utility node for Llama Workbench."""

from __future__ import annotations

import json
import operator
import random
from typing import Any


SEED_MAX = (1 << 53) - 1
SEED_LIST_MAX_COUNT = 1024


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
                        "tooltip": "Seed range: 0–9007199254740991 (JavaScript safe integers). Use the node buttons or ComfyUI's native control to keep it fixed, randomize it, or increment/decrement it after generation. Downstream nodes may have smaller seed ranges.",
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


class LlamaWorkbenchSeedList(LlamaWorkbenchSeed):
    """Expand a base seed into a repeatable list for ComfyUI list execution."""

    RETURN_TYPES = ("INT", "INT", "STRING")
    RETURN_NAMES = ("SEED", "SEEDS", "seeds_json")
    OUTPUT_IS_LIST = (False, True, False)

    @classmethod
    def INPUT_TYPES(cls):
        inputs = super().INPUT_TYPES()
        inputs["required"]["seed"][1]["tooltip"] = (
            "Base seed and first list item. The same seed and count reproduce the same list. "
            "New Fixed Random picks a new fixed list; Randomize Each Run changes the list each run. "
            "Increment/decrement changes the base seed, then regenerates the list."
        )
        inputs["required"]["count"] = (
            "INT",
            {
                "default": 4,
                "min": 1,
                "max": SEED_LIST_MAX_COUNT,
                "step": 1,
                "tooltip": "Number of distinct seeds. Increasing count preserves the existing prefix. SEEDS runs connected ordinary nodes once per seed.",
            },
        )
        return inputs

    def generate(self, seed: Any = 0, count: int = 4):
        value = super().generate(seed)[0]
        if isinstance(count, bool):
            raise ValueError("count must be an integer")
        try:
            count = operator.index(count)
        except TypeError as exc:
            raise ValueError("count must be an integer") from exc
        if not 1 <= count <= SEED_LIST_MAX_COUNT:
            raise ValueError(f"count must be between 1 and {SEED_LIST_MAX_COUNT}")

        # A local generator avoids changing sampling randomness elsewhere in the host.
        rng = random.Random(value)
        seeds = [value]
        seen = {value}
        while len(seeds) < count:
            candidate = rng.randrange(SEED_MAX + 1)
            if candidate not in seen:
                seeds.append(candidate)
                seen.add(candidate)
        return value, seeds, json.dumps(seeds)
