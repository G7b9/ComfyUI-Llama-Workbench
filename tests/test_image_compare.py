import json
from pathlib import Path
import subprocess
import sys
import types

import numpy as np
import pytest

from lwb.image_compare import LlamaWorkbenchImageCompare
from lwb.nodes import NODE_CLASS_MAPPINGS


class ExecutionBlocker:
    def __init__(self, message):
        self.message = message


@pytest.fixture
def previewer(monkeypatch):
    calls = []

    class PreviewImage:
        def save_images(self, images, filename_prefix):
            calls.append((images, filename_prefix))
            return {"ui": {"images": [
                {"filename": f"{filename_prefix}_{index}.png", "subfolder": "", "type": "temp"}
                for index in range(images.shape[0])
            ]}}

    monkeypatch.setitem(sys.modules, "nodes", types.SimpleNamespace(PreviewImage=PreviewImage))
    monkeypatch.setitem(sys.modules, "comfy_execution.graph", types.SimpleNamespace(ExecutionBlocker=ExecutionBlocker))
    return calls


def test_compare_aggregates_mapped_outputs_and_batches_in_order(previewer):
    a = np.zeros((1, 24, 32, 4))
    b1 = np.ones((2, 12, 16, 3))
    b2 = np.ones((2, 30, 20, 4))
    result = LlamaWorkbenchImageCompare().compare([a], [b1, b2], [True])
    data = result["ui"]["comparison"][0]
    assert len(data["a"]) == 1
    assert [(x["width"], x["height"]) for x in data["b"]] == [(16, 12)] * 2 + [(20, 30)] * 2
    assert data["repeat"] is True
    assert len(previewer) == 3  # Repeating an image never duplicates tensor data or PNGs.
    assert all(saved is original for (saved, _), original in zip(previewer, [a, b1, b2]))
    assert len({item["filename"] for item in data["a"] + data["b"]}) == 5
    assert "images" not in result["ui"]  # Avoid ComfyUI's additional generic gallery.
    json.dumps(result["ui"])
    assert all(isinstance(value, ExecutionBlocker) for value in result["result"])


def test_empty_batch_and_native_list_contract(previewer):
    empty = np.zeros((0, 24, 32, 3))
    result = LlamaWorkbenchImageCompare().compare([empty], [], [False])
    assert result["ui"]["comparison"] == [{
        "a": [], "b": [], "repeat": False, "selection_enabled": False, "selected_index": -1,
    }]
    assert previewer == []
    assert LlamaWorkbenchImageCompare.INPUT_IS_LIST is True
    assert LlamaWorkbenchImageCompare.OUTPUT_NODE is True
    assert LlamaWorkbenchImageCompare.RETURN_TYPES == ("INT", "IMAGE", "IMAGE")
    assert LlamaWorkbenchImageCompare.RETURN_NAMES == ("index", "selected_a", "selected_b")
    assert NODE_CLASS_MAPPINGS["LlamaWorkbench_ImageCompare"] is LlamaWorkbenchImageCompare


def test_accepts_nested_lists_without_concatenating_different_sizes(previewer):
    a = np.zeros((1, 24, 32, 3))
    b = np.zeros((1, 32, 24, 4))
    output = LlamaWorkbenchImageCompare().compare([[a], [b]], [a], False)
    assert len(output["ui"]["comparison"][0]["a"]) == 2
    assert output["ui"]["comparison"][0]["repeat"] is False


@pytest.mark.parametrize("shape", [(24, 32, 3), (1, 24, 32, 2), (1, 0, 32, 3)])
def test_invalid_input_fails_before_writing_previews(previewer, shape):
    with pytest.raises(ValueError, match="image_b requires IMAGE"):
        LlamaWorkbenchImageCompare().compare([np.zeros((1, 8, 8, 3))], [np.zeros(shape)])
    assert not previewer


def test_runs_have_distinct_preview_urls(previewer):
    a = np.zeros((1, 8, 8, 3))
    node = LlamaWorkbenchImageCompare()
    assert node.compare([a], [a])["ui"] != node.compare([a], [a])["ui"]


def test_selected_pair_indexes_across_batches_and_preserves_original_images(previewer):
    a1 = np.zeros((2, 24, 32, 3), dtype=np.float32)
    a2 = np.full((2, 8, 16, 4), 0.125, dtype=np.float32)
    b = np.stack([np.full((4, 6, 3), i / 7, dtype=np.float32) for i in range(4)])
    output = LlamaWorkbenchImageCompare().compare([a1, a2], [b], [False], [True], [3])
    index, selected_a, selected_b = output["result"]
    assert index == 3
    assert selected_a.shape == (1, 8, 16, 4)
    assert selected_b.shape == (1, 4, 6, 3)
    assert selected_a.dtype == a2.dtype
    np.testing.assert_array_equal(selected_a, a2[1:2])
    np.testing.assert_array_equal(selected_b, b[3:4])
    assert np.shares_memory(selected_a, a2)
    assert np.shares_memory(selected_b, b)


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("short_count", [1, 2])
def test_selected_repeat_pair_outputs_last_image_of_short_group(previewer, reverse, short_count):
    short = np.stack([np.full((8, 8, 3), i / 3) for i in range(short_count)])
    long = np.stack([np.full((8, 8, 3), i / 5) for i in range(4)])
    a, b = (long, short) if reverse else (short, long)
    index, selected_a, selected_b = LlamaWorkbenchImageCompare().compare([a], [b], True, True, 3)["result"]
    assert index == 3
    np.testing.assert_array_equal(selected_a, a[-1:])
    np.testing.assert_array_equal(selected_b, b[-1:])


@pytest.mark.parametrize("enabled,index,repeat", [
    (False, 0, True), (True, -1, True), (True, 4, True), (True, 2, False),
])
def test_invalid_or_disabled_selection_blocks_outputs_but_keeps_preview(previewer, enabled, index, repeat):
    a = np.zeros((1, 8, 8, 3))
    b = np.zeros((4, 8, 8, 3))
    output = LlamaWorkbenchImageCompare().compare([a], [b], [repeat], [enabled], [index])
    assert len(output["ui"]["comparison"][0]["b"]) == 4
    assert all(isinstance(value, ExecutionBlocker) and value.message is None for value in output["result"])


def test_selected_first_pair_uses_zero_based_index(previewer):
    a = np.zeros((1, 8, 8, 3))
    result = LlamaWorkbenchImageCompare().compare([a], [a], False, True, 0)["result"]
    assert result[0] == 0
    np.testing.assert_array_equal(result[1], a)


def test_empty_group_cannot_be_selected_even_with_repeat(previewer):
    result = LlamaWorkbenchImageCompare().compare([], [np.zeros((1, 8, 8, 3))], True, True, 0)
    assert all(isinstance(value, ExecutionBlocker) for value in result["result"])


def test_invalid_index_type_is_rejected(previewer):
    a = np.zeros((1, 8, 8, 3))
    with pytest.raises(ValueError, match="selected_index must be an integer"):
        LlamaWorkbenchImageCompare().compare([a], [a], False, True, 0.5)


def test_frontend_pairs_by_index_and_repeats_only_the_last_item():
    source = (Path(__file__).parents[1] / "web" / "image_compare.js").read_text()
    source = "\n".join(line for line in source.splitlines() if not line.startswith("import "))
    script = """
import assert from 'node:assert/strict';
const app = {registerExtension() {}};
const api = {};
""" + source + """
const a = ['a1'], b = ['b1', 'b2', 'b3', 'b4'];
assert.deepEqual(pairImages(a, b, true).map(p => [p.a, p.b]),
  [['a1','b1'], ['a1','b2'], ['a1','b3'], ['a1','b4']]);
assert.deepEqual(pairImages(b, a, true).map(p => p.b), ['a1','a1','a1','a1']);
assert.deepEqual(pairImages(['a1','a2'], b, true).map(p => p.a), ['a1','a2','a2','a2']);
assert.deepEqual(pairImages(a, b, false).map(p => p.a), ['a1',undefined,undefined,undefined]);
assert.deepEqual(pairImages(a, b, true).map(p => p.repeatA), [false,true,true,true]);
assert.equal(pairImages([], b, true)[0].a, undefined);
assert.equal(pairImages([], b, true)[0].repeatA, false);
assert.equal(pairImages([], [], true).length, 0);
assert.deepEqual(a, ['a1']);
assert.equal(b.length, 4);
"""
    subprocess.run(["node", "--input-type=module", "-e", script], check=True, capture_output=True, text=True)
