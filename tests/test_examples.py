from __future__ import annotations

import json
from pathlib import Path

from lwb.nodes import NODE_CLASS_MAPPINGS


EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
EXPECTED_FILES = {
    "01_start-server-chat.json",
    "02_existing-server-image-to-prompt.json",
    "03_skill-chat.json",
    "04_embedded-qwen-gemma-vlm.json",
    "05_qwen-image-2.1-pe-t2i-gguf.json",
}


def test_examples_are_valid_comfyui_workflow_documents():
    files = {path.name for path in EXAMPLES.glob("*.json")}
    assert files == EXPECTED_FILES
    for path in sorted(EXAMPLES.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["version"] == 0.4
        nodes = {node["id"]: node for node in payload["nodes"]}
        assert nodes
        for node in nodes.values():
            node_type = node["type"]
            assert node_type == "LoadImage" or node_type in NODE_CLASS_MAPPINGS
        for link_id, source_id, source_slot, target_id, target_slot, link_type in payload["links"]:
            assert source_id in nodes and target_id in nodes
            source = nodes[source_id]["outputs"][source_slot]
            target = nodes[target_id]["inputs"][target_slot]
            assert source["type"] == link_type
            assert target["type"] == link_type
            assert target["link"] == link_id


def test_examples_use_only_workbench_node_ids():
    for path in EXAMPLES.glob("*.json"):
        text = path.read_text(encoding="utf-8")
        assert "LlamaCpp" not in text
        assert "QwenTE" not in text
