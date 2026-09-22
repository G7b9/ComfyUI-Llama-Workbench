from __future__ import annotations

import json

from lwb.nodes import LlamaWorkbenchSkillLoader
from lwb.skills import (
    build_runtime_contract_instruction,
    build_skill_instruction,
    choose_skill,
    discover_skills,
    empty_flow_state,
    output_contract_for_route,
    parse_skill_state,
    read_reference,
    select_runtime_route,
    validate_output_contract,
)


def _write_skill(root, skill_id="shot-planner"):
    skill = root / skill_id
    (skill / "references").mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        "---\nname: Shot Planner\ndescription: Build a video shot plan\n---\nPlan shots safely.", encoding="utf-8"
    )
    (skill / "references" / "guide.md").write_text("A reference.", encoding="utf-8")
    return skill


def test_discovery_selection_and_declared_reference_only(tmp_path):
    _write_skill(tmp_path)
    skills = discover_skills(tmp_path)
    assert skills[0].name == "Shot Planner"
    assert choose_skill(skills, "Please plan video shots").id == "shot-planner"
    assert read_reference(skills[0], "references/guide.md", tmp_path) == "A reference."
    try:
        read_reference(skills[0], "../secret.txt", tmp_path)
    except ValueError:
        pass
    else:
        raise AssertionError("undeclared path traversal should be rejected")


def test_skill_state_is_removed_from_user_visible_reply(tmp_path):
    _write_skill(tmp_path)
    skill = discover_skills(tmp_path)[0]
    state = empty_flow_state()
    state["loaded_references"] = ["references/guide.md"]
    instruction = build_skill_instruction(skill, state, tmp_path)
    assert "A reference." in instruction
    reply, protocol = parse_skill_state(
        'Need a format choice. <lwb_skill_state>{"stage":"brief","options":["vertical"],"final":false}</lwb_skill_state>'
    )
    assert reply == "Need a format choice."
    assert protocol["options"] == ["vertical"]


def test_generic_runtime_manifest_selects_reference_guides_and_validates_declared_contracts(tmp_path):
    skill_directory = _write_skill(tmp_path)
    (skill_directory / "runtime.json").write_text(
        json.dumps(
            {
                "version": 1,
                "reference_router": {
                    "field": "Mode",
                    "normalize": "upper",
                    "routes": {"REFERENCE": ["references/guide.md"]},
                },
                "output_contracts": {
                    "REFERENCE": {
                        "required_headings": ["subject_definitions:", "summary:"],
                        "forbidden_headings": ["integrated_multimodal_description:"],
                        "forbidden_patterns": ["(?i)\\bvisual anchor\\b"],
                        "require_first_heading": True,
                        "timeline_duration_field": "Target duration",
                        "label_count_fields": {"Subject": "Reference subject count"},
                        "exact_value_fields": {"summary:": "Summary"},
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    skill = discover_skills(tmp_path)[0]

    route, references = select_runtime_route(skill, "Mode: BASIC\nMode: REFERENCE")
    assert route == "REFERENCE"
    assert references == ("references/guide.md",)
    contract = output_contract_for_route(skill, route)
    assert contract is not None
    assert contract.require_first_heading is True
    assert contract.timeline_duration_field == "Target duration"
    assert "subject_definitions:" in contract.required_headings
    assert "integrated_multimodal_description:" in contract.forbidden_headings
    runtime_instruction = build_runtime_contract_instruction(skill, route, references)
    assert "references/guide.md" in runtime_instruction

    valid = """subject_definitions:
<Subject 1> is a person.
summary:
[reference generation] The target uses <Subject 1>.
retention_analysis:
<Subject 1>: fully_preserved - identity is retained.
detailed_description:
[Shot 1] <Subject 1> moves.
overall_soundscape:
Room tone.
non_diegetic_music:
N/A"""
    assert validate_output_contract(valid, contract) == []
    invalid = """subject_definitions:
summary:
missing detail
integrated_multimodal_description:
wrong mode"""
    violations = validate_output_contract(invalid, contract)
    assert any("has no content" in item for item in violations)
    assert any("forbidden heading" in item for item in violations)
    out_of_order = """summary:
[reference generation] Wrong order.
subject_definitions:
<Subject 1> is a person.
retention_analysis:
<Subject 1>: fully_preserved - identity is retained.
detailed_description:
[Shot 1] <Subject 1> moves.
overall_soundscape:
Room tone.
non_diegetic_music:
N/A"""
    assert "required headings are not in the declared order" in validate_output_contract(out_of_order, contract)
    too_long = valid.replace("[Shot 1] <Subject 1> moves.", "[Shot 1] At 00:10.001, <Subject 1> moves.")
    duration_violations = validate_output_contract(too_long, contract, "Target duration: 10.00 seconds")
    assert any("exceeds Target duration" in item for item in duration_violations)
    ref2va_frame_anchor = valid.replace(
        "[Shot 1] <Subject 1> moves.",
        "[Shot 1] At 00:12.000, <Picture 1> is the visual anchor for <Subject 1>.",
    )
    frame_violations = validate_output_contract(ref2va_frame_anchor, contract, "Target duration: 10.00 seconds")
    assert any("forbidden output pattern" in item for item in frame_violations)
    assert any("exceeds Target duration" in item for item in frame_violations)
    extra_subject = valid.replace(
        "summary:\n",
        "<Subject 2> is an unrelated cafe environment.\nsummary:\n",
    )
    subject_violations = validate_output_contract(extra_subject, contract, "Reference subject count: 1")
    assert "Subject labels exceed Reference subject count (1)" in subject_violations
    picture_anchor = valid.replace(
        "summary:\n",
        "<Picture 1> is the primary visual anchor for the target video.\nsummary:\n",
    )
    assert any("forbidden output pattern" in item for item in validate_output_contract(picture_anchor, contract))


def test_skill_loader_node_lists_and_selects_the_bundled_prompt_refiner_skill():
    choices = LlamaWorkbenchSkillLoader.INPUT_TYPES()["required"]["skill"][0]
    assert "prompt-refiner" in choices
    payload, _ = LlamaWorkbenchSkillLoader().load("prompt-refiner")
    assert payload["selected"] == "prompt-refiner"
