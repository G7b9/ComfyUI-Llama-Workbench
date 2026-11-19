import json
from pathlib import Path
import random
import subprocess

import pytest

from lwb.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from lwb.seed import LlamaWorkbenchSeed, LlamaWorkbenchSeedList, SEED_LIST_MAX_COUNT, SEED_MAX


def test_seed_list_registration_and_list_execution_contract():
    assert NODE_CLASS_MAPPINGS["LlamaWorkbench_SeedList"] is LlamaWorkbenchSeedList
    assert NODE_DISPLAY_NAME_MAPPINGS["LlamaWorkbench_SeedList"] == "Llama Workbench Seed List"
    assert LlamaWorkbenchSeedList.RETURN_TYPES == ("INT", "INT", "STRING")
    assert LlamaWorkbenchSeedList.OUTPUT_IS_LIST == (False, True, False)
    assert LlamaWorkbenchSeedList.INPUT_TYPES()["required"]["seed"][1]["control_after_generate"]
    assert set(LlamaWorkbenchSeed.INPUT_TYPES()["required"]) == {"seed"}
    assert LlamaWorkbenchSeed().generate(42) == (42,)


@pytest.mark.parametrize("seed", [0, 42, SEED_MAX])
@pytest.mark.parametrize("count", [1, 4, SEED_LIST_MAX_COUNT])
def test_seed_list_is_repeatable_unique_and_json_safe(seed, count):
    original_state = random.getstate()
    result = LlamaWorkbenchSeedList().generate(seed, count)
    base, seeds, encoded = result
    assert base == seeds[0] == seed
    assert len(seeds) == len(set(seeds)) == count
    assert all(0 <= value <= SEED_MAX for value in seeds)
    assert json.loads(encoded) == seeds
    assert LlamaWorkbenchSeedList().generate(seed, count) == result
    assert random.getstate() == original_state


def test_changing_count_preserves_prefix_and_new_base_changes_list():
    node = LlamaWorkbenchSeedList()
    assert node.generate(123, 4)[1] == node.generate(123, 10)[1][:4]
    assert node.generate(123, 4)[1][1:] != node.generate(456, 4)[1][1:]


@pytest.mark.parametrize("count", [0, -1, SEED_LIST_MAX_COUNT + 1, 1.5, "4", None, True])
def test_seed_list_rejects_invalid_count(count):
    with pytest.raises(ValueError, match="count"):
        LlamaWorkbenchSeedList().generate(42, count)


@pytest.mark.parametrize("seed", [-1, SEED_MAX + 1, "invalid"])
def test_seed_list_rejects_invalid_base(seed):
    with pytest.raises(ValueError, match="seed"):
        LlamaWorkbenchSeedList().generate(seed, 4)


@pytest.mark.parametrize("node_name", ["LlamaWorkbench_Seed", "LlamaWorkbench_SeedList"])
def test_seed_buttons_work_for_both_nodes_and_fixed_list_survives_reload(node_name):
    source = (Path(__file__).parents[1] / "web" / "seed.js").read_text()
    source = "\n".join(line for line in source.splitlines() if not line.startswith("import "))
    script = """
import assert from 'node:assert/strict';
let extension;
const app = {registerExtension(value) {extension = value;}};
""" + source + """
const nodeName = process.argv[1];
class Node {
  constructor() {
    this.widgets = [{name: 'seed', value: 0}, {name: 'control_after_generate', value: 'fixed'}];
    if (nodeName.endsWith('SeedList')) this.widgets.push({name: 'count', value: 4});
  }
  addWidget(type, name, value, callback, options) {
    const widget = {type, name, value, callback, options};
    this.widgets.push(widget);
    return widget;
  }
}
extension.beforeRegisterNodeDef(Node, {name: nodeName});
const node = new Node();
node.onNodeCreated();
node.onNodeCreated();
assert.equal(node.widgets.filter(w => w.type === 'button').length, 5);
const seed = node.widgets[0];
const mode = node.widgets[1];
function click(label) {node.widgets.find(w => w.name.includes(label)).callback();}
let randomValue = 100;
Object.defineProperty(globalThis, 'crypto', {configurable: true, value: {
  getRandomValues(array) {array[0] = 0; array[1] = ++randomValue; return array;}
}});
click('New Fixed Random');
assert.equal(seed.value, 101);
assert.equal(mode.value, 'fixed');
const fixed = seed.value;
click('Randomize Each Run');
assert.equal(seed.value, 102);
assert.equal(mode.value, 'randomize');
click('Keep Fixed');
assert.equal(seed.value, 102);
assert.equal(mode.value, 'fixed');
seed.value = Number.MAX_SAFE_INTEGER;
click('Increment Each Run');
assert.equal(seed.value, 0);
assert.equal(mode.value, 'increment');
click('Decrement Each Run');
assert.equal(seed.value, Number.MAX_SAFE_INTEGER);
assert.equal(mode.value, 'decrement');
click('New Fixed Random');
const saved = JSON.parse(JSON.stringify(node.widgets
  .filter(w => w.options?.serialize !== false).map(w => w.value)));
assert.deepEqual(saved, nodeName.endsWith('SeedList') ? [103, 'fixed', 4] : [103, 'fixed']);
const reloaded = new Node();
reloaded.onNodeCreated();
saved.forEach((value, index) => reloaded.widgets[index].value = value);
assert.equal(reloaded.widgets[0].value, 103);
assert.equal(reloaded.widgets[1].value, 'fixed');
console.log(JSON.stringify({fixed, seed: reloaded.widgets[0].value, count: saved[2] ?? 1}));
"""
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script, node_name],
        check=True, capture_output=True, text=True,
    )
    restored = json.loads(result.stdout)
    seed, count = restored["seed"], restored["count"]
    assert LlamaWorkbenchSeedList().generate(seed, count) == LlamaWorkbenchSeedList().generate(103, count)
    assert LlamaWorkbenchSeedList().generate(restored["fixed"], count)[1] != LlamaWorkbenchSeedList().generate(seed, count)[1]
