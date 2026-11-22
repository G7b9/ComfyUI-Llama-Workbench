import json
from pathlib import Path
import subprocess
import sys
import types

import numpy as np
import pytest

from lwb.image_compare import LlamaWorkbenchImageCompare
from lwb.nodes import NODE_CLASS_MAPPINGS


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
    json.dumps(result)


def test_empty_batch_and_native_list_contract(previewer):
    empty = np.zeros((0, 24, 32, 3))
    result = LlamaWorkbenchImageCompare().compare([empty], [], [False])
    assert result["ui"]["comparison"] == [{"a": [], "b": [], "repeat": False}]
    assert previewer == []
    assert LlamaWorkbenchImageCompare.INPUT_IS_LIST is True
    assert LlamaWorkbenchImageCompare.OUTPUT_NODE is True
    assert LlamaWorkbenchImageCompare.RETURN_TYPES == ()
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
    assert node.compare([a], [a]) != node.compare([a], [a])


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
