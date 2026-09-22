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
COMFY_CORE_NODE_IDS = {
    "LoadImage",
    "UNETLoader",
    "CLIPLoader",
    "VAELoader",
    "TextEncodeQwenImage21",
    "EmptyLatentImage",
    "KSampler",
    "VAEDecode",
    "SaveImage",
}


def _serialized_workbench_widgets(node):
    """Pair saved widget values with names, including Comfy's seed control."""

    node_class = NODE_CLASS_MAPPINGS[node["type"]]
    schema = node_class.INPUT_TYPES()
    declared = {**schema.get("required", {}), **schema.get("optional", {})}
    names = []
    for input_slot in node.get("inputs", []):
        widget = input_slot.get("widget")
        if not widget:
            continue
        name = widget["name"]
        names.append(name)
        specification = declared[name]
        options = specification[1] if len(specification) > 1 and isinstance(specification[1], dict) else {}
        if options.get("control_after_generate"):
            names.append(f"{name}:control_after_generate")

    values = node.get("widgets_values") or []
    assert len(values) == len(names), f"{node['type']} has shifted or missing serialized widget values"
    return dict(zip(names, values, strict=True)), declared


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
            assert node_type in COMFY_CORE_NODE_IDS or node_type in NODE_CLASS_MAPPINGS
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


def test_workbench_widget_values_are_not_shifted_by_seed_controls():
    for path in EXAMPLES.glob("*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for node in payload["nodes"]:
            if node["type"] not in NODE_CLASS_MAPPINGS:
                continue
            values, declared = _serialized_workbench_widgets(node)
            controls = {name: value for name, value in values.items() if name.endswith(":control_after_generate")}
            assert all(value in {"fixed", "increment", "decrement", "randomize"} for value in controls.values())
            for name, value in values.items():
                if name.endswith(":control_after_generate"):
                    continue
                specification = declared[name]
                options = specification[1] if len(specification) > 1 and isinstance(specification[1], dict) else {}
                if specification[0] == "INT" and isinstance(value, int):
                    assert value >= options.get("min", value)
                    assert value <= options.get("max", value)


def test_qwen_prompt_enhancer_example_is_an_end_to_end_generation_workflow():
    payload = json.loads((EXAMPLES / "05_qwen-image-2.1-pe-t2i-gguf.json").read_text(encoding="utf-8"))
    nodes = {node["id"]: node for node in payload["nodes"]}
    types = {node["type"] for node in nodes.values()}

    assert {
        "LlamaWorkbench_StartServer",
        "LlamaWorkbench_PromptEnhancer",
        "LlamaWorkbench_QwenImage21PEResolution",
        "UNETLoader",
        "CLIPLoader",
        "VAELoader",
        "TextEncodeQwenImage21",
        "EmptyLatentImage",
        "KSampler",
        "VAEDecode",
        "SaveImage",
    } <= types

    enhancer = next(node for node in nodes.values() if node["type"] == "LlamaWorkbench_PromptEnhancer")
    enhancer_values, _ = _serialized_workbench_widgets(enhancer)
    assert enhancer_values["seed:control_after_generate"] == "fixed"
    assert enhancer_values["max_images"] == 10
    assert enhancer_values["auto_unload"] is True

    resolution = next(
        node for node in nodes.values() if node["type"] == "LlamaWorkbench_QwenImage21PEResolution"
    )
    resolution_values, _ = _serialized_workbench_widgets(resolution)
    assert resolution_values["aspect_ratio_override"] == "Auto (use Prompt Enhancer)"

    links = {(nodes[source]["type"], source_slot, nodes[target]["type"], target_slot) for _, source, source_slot, target, target_slot, _ in payload["links"]}
    assert ("LlamaWorkbench_PromptEnhancer", 0, "TextEncodeQwenImage21", 3) in links
    assert ("LlamaWorkbench_PromptEnhancer", 1, "LlamaWorkbench_QwenImage21PEResolution", 0) in links
    assert ("LlamaWorkbench_QwenImage21PEResolution", 0, "EmptyLatentImage", 0) in links
    assert ("LlamaWorkbench_QwenImage21PEResolution", 1, "EmptyLatentImage", 1) in links
    assert ("VAEDecode", 0, "SaveImage", 0) in links
