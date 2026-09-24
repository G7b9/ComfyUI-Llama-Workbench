from __future__ import annotations

import pytest

from lwb.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from lwb.seed import LlamaWorkbenchSeed, SEED_MAX


def test_seed_node_uses_native_comfyui_seed_controls():
    specification = LlamaWorkbenchSeed.INPUT_TYPES()["required"]["seed"]
    assert specification[0] == "INT"
    assert specification[1]["default"] == 0
    assert specification[1]["min"] == 0
    assert specification[1]["max"] == SEED_MAX
    assert specification[1]["control_after_generate"] is True


@pytest.mark.parametrize("seed", [0, 1, 42, SEED_MAX])
def test_seed_node_returns_fixed_seed(seed):
    assert LlamaWorkbenchSeed().generate(seed) == (seed,)


@pytest.mark.parametrize("seed", [-1, SEED_MAX + 1, "not-a-seed"])
def test_seed_node_rejects_invalid_seed(seed):
    with pytest.raises(ValueError):
        LlamaWorkbenchSeed().generate(seed)


def test_seed_node_is_registered_with_a_namespaced_display_name():
    assert NODE_CLASS_MAPPINGS["LlamaWorkbench_Seed"] is LlamaWorkbenchSeed
    assert NODE_DISPLAY_NAME_MAPPINGS["LlamaWorkbench_Seed"] == "Llama Workbench Seed"
