"""A small, data-only Skill registry and conversation coordinator.

Skills are instruction documents.  They do not gain filesystem, network, shell,
or ComfyUI graph execution authority merely by being loaded.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


SKILLS_ROOT = Path(__file__).resolve().parent.parent / "skills"
MAX_DOCUMENT_BYTES = 128 * 1024
_STATE_TAG = re.compile(r"<lwb_skill_state>\s*(\{.*?\})\s*</lwb_skill_state>", re.DOTALL)
_SAFE_SKILL_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_SAFE_REFERENCE_SUFFIXES = {".md", ".txt", ".json", ".yaml", ".yml"}
_RUNTIME_FILE = "runtime.json"
_RUNTIME_NORMALIZERS = {"upper", "lower", "exact"}


@dataclass(frozen=True, slots=True)
class OutputContract:
    """Optional, declarative final-output checks supplied by one Skill."""

    required_headings: tuple[str, ...] = ()
    forbidden_headings: tuple[str, ...] = ()
    forbidden_patterns: tuple[str, ...] = ()
    require_first_heading: bool = False
    timeline_duration_field: str = ""
    label_count_fields: dict[str, str] = field(default_factory=dict)
    exact_value_fields: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SkillRuntime:
    """Optional generic runtime metadata loaded from a Skill's runtime.json."""

    selector_field: str = ""
    selector_normalize: str = "upper"
    reference_routes: dict[str, tuple[str, ...]] = field(default_factory=dict)
    output_contracts: dict[str, OutputContract] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Skill:
    id: str
    name: str
    description: str
    document: str
    references: tuple[str, ...]
    runtime: SkillRuntime = field(default_factory=SkillRuntime)


def _read_text(path: Path) -> str:
    if not path.is_file():
        raise ValueError(f"Skill file does not exist: {path.name}")
    if path.stat().st_size > MAX_DOCUMENT_BYTES:
        raise ValueError(f"Skill file is too large (limit {MAX_DOCUMENT_BYTES} bytes): {path.name}")
    return path.read_text(encoding="utf-8-sig")


def _frontmatter(text: str) -> tuple[dict[str, str], str]:
    if not text.startswith("---"):
        return {}, text
    closing = text.find("\n---", 3)
    if closing < 0:
        return {}, text
    metadata: dict[str, str] = {}
    for line in text[3:closing].splitlines():
        key, separator, value = line.partition(":")
        if separator and key.strip():
            metadata[key.strip().lower()] = value.strip().strip("\"'")
    return metadata, text[closing + 4 :].lstrip("\r\n")


def _reference_list(skill_directory: Path) -> tuple[str, ...]:
    refs_dir = skill_directory / "references"
    if not refs_dir.is_dir():
        return ()
    files: list[str] = []
    for candidate in refs_dir.rglob("*"):
        if candidate.is_file() and candidate.suffix.lower() in _SAFE_REFERENCE_SUFFIXES:
            files.append(candidate.relative_to(skill_directory).as_posix())
    return tuple(sorted(files))


def _normalize_route_value(value: str, normalizer: str) -> str:
    if normalizer == "lower":
        return value.lower()
    if normalizer == "exact":
        return value
    return value.upper()


def _safe_runtime_strings(value: Any, *, limit: int, item_limit: int) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    result: list[str] = []
    for item in value[:item_limit]:
        if not isinstance(item, str):
            continue
        text = item.strip()
        if text and len(text) <= limit and text not in result:
            result.append(text)
    return tuple(result)


def _read_runtime_config(skill_directory: Path, references: tuple[str, ...]) -> SkillRuntime:
    """Read an optional, data-only Skill runtime declaration.

    A malformed declaration is ignored so an ordinary document-only Skill
    stays usable. Runtime metadata has no executable fields: it can only map
    one user-message field value to already-declared reference files and
    declare lightweight text-output checks.
    """

    path = skill_directory / _RUNTIME_FILE
    if not path.is_file():
        return SkillRuntime()
    try:
        value = json.loads(_read_text(path))
    except (ValueError, json.JSONDecodeError):
        return SkillRuntime()
    if not isinstance(value, dict) or value.get("version") != 1:
        return SkillRuntime()

    router = value.get("reference_router")
    selector_field = ""
    normalizer = "upper"
    routes: dict[str, tuple[str, ...]] = {}
    if isinstance(router, dict):
        candidate_field = str(router.get("field") or "").strip()
        candidate_normalizer = str(router.get("normalize") or "upper").strip().lower()
        if 1 <= len(candidate_field) <= 80 and candidate_normalizer in _RUNTIME_NORMALIZERS:
            selector_field = candidate_field
            normalizer = candidate_normalizer
            raw_routes = router.get("routes")
            if isinstance(raw_routes, dict):
                for raw_key, raw_references in raw_routes.items():
                    if not isinstance(raw_key, str) or not raw_key.strip():
                        continue
                    selected = _safe_runtime_strings(raw_references, limit=240, item_limit=16)
                    if selected and all(item in references for item in selected):
                        key = _normalize_route_value(raw_key.strip(), normalizer)
                        routes[key] = selected

    contracts: dict[str, OutputContract] = {}
    raw_contracts = value.get("output_contracts")
    if isinstance(raw_contracts, dict):
        for raw_key, raw_contract in raw_contracts.items():
            if not isinstance(raw_key, str) or not isinstance(raw_contract, dict):
                continue
            required = _safe_runtime_strings(raw_contract.get("required_headings"), limit=120, item_limit=24)
            forbidden = _safe_runtime_strings(raw_contract.get("forbidden_headings"), limit=120, item_limit=24)
            patterns = _safe_runtime_strings(raw_contract.get("forbidden_patterns"), limit=240, item_limit=24)
            # Invalid regular expressions should not make a selected Skill
            # unusable. Retain only patterns that are safe to compile.
            patterns = tuple(pattern for pattern in patterns if _is_valid_runtime_pattern(pattern))
            raw_label_limits = raw_contract.get("label_count_fields")
            label_count_fields: dict[str, str] = {}
            if isinstance(raw_label_limits, dict):
                for raw_label, raw_field in raw_label_limits.items():
                    label = str(raw_label or "").strip()
                    count_field = str(raw_field or "").strip()
                    if (
                        re.fullmatch(r"[A-Za-z][A-Za-z0-9 _-]{0,39}", label)
                        and 1 <= len(count_field) <= 80
                    ):
                        label_count_fields[label] = count_field
            raw_exact_values = raw_contract.get("exact_value_fields")
            exact_value_fields: dict[str, str] = {}
            if isinstance(raw_exact_values, dict):
                for raw_heading, raw_field in raw_exact_values.items():
                    heading = str(raw_heading or "").strip()
                    value_field = str(raw_field or "").strip()
                    if 1 <= len(heading) <= 120 and 1 <= len(value_field) <= 80:
                        exact_value_fields[heading] = value_field
            if required or forbidden or patterns or label_count_fields or exact_value_fields:
                key = _normalize_route_value(raw_key.strip(), normalizer)
                duration_field = str(raw_contract.get("timeline_duration_field") or "").strip()
                if len(duration_field) > 80:
                    duration_field = ""
                contracts[key] = OutputContract(
                    required_headings=required,
                    forbidden_headings=forbidden,
                    forbidden_patterns=patterns,
                    require_first_heading=bool(raw_contract.get("require_first_heading", False)),
                    timeline_duration_field=duration_field,
                    label_count_fields=label_count_fields,
                    exact_value_fields=exact_value_fields,
                )

    return SkillRuntime(
        selector_field=selector_field,
        selector_normalize=normalizer,
        reference_routes=routes,
        output_contracts=contracts,
    )


def _is_valid_runtime_pattern(pattern: str) -> bool:
    try:
        re.compile(pattern)
    except re.error:
        return False
    return True


def discover_skills(root: Path = SKILLS_ROOT) -> list[Skill]:
    """Read well-formed Skill directories from this package only."""

    if not root.is_dir():
        return []
    found: list[Skill] = []
    for directory in sorted(root.iterdir(), key=lambda item: item.name.lower()):
        if not directory.is_dir() or not _SAFE_SKILL_ID.fullmatch(directory.name):
            continue
        document_path = directory / "SKILL.md"
        if not document_path.is_file():
            continue
        raw = _read_text(document_path)
        metadata, document = _frontmatter(raw)
        references = _reference_list(directory)
        found.append(
            Skill(
                id=directory.name,
                name=metadata.get("name") or directory.name,
                description=metadata.get("description") or "",
                document=document.strip(),
                references=references,
                runtime=_read_runtime_config(directory, references),
            )
        )
    return found


def get_skill(skill_id: str, root: Path = SKILLS_ROOT) -> Skill | None:
    return next((skill for skill in discover_skills(root) if skill.id == skill_id), None)


def read_reference(skill: Skill, relative_path: str, root: Path = SKILLS_ROOT) -> str:
    normalized = str(relative_path or "").replace("\\", "/").strip("/")
    if normalized not in skill.references:
        raise ValueError("The requested Skill reference is not declared by this Skill")
    candidate = (root / skill.id / normalized).resolve()
    skill_root = (root / skill.id).resolve()
    if skill_root not in candidate.parents:
        raise ValueError("Skill reference resolved outside its Skill directory")
    return _read_text(candidate)


def select_runtime_route(skill: Skill, user_text: str) -> tuple[str, tuple[str, ...]]:
    """Select a declared route from a plain ``Field: value`` user line."""

    runtime = skill.runtime
    if not runtime.selector_field or not runtime.reference_routes:
        return "", ()
    pattern = re.compile(rf"(?im)^\s*{re.escape(runtime.selector_field)}\s*:\s*([^\r\n]+)")
    matches = pattern.findall(str(user_text or ""))
    if not matches:
        return "", ()
    route = _normalize_route_value(matches[-1].strip(), runtime.selector_normalize)
    return route, runtime.reference_routes.get(route, ())


def output_contract_for_route(skill: Skill, route: str) -> OutputContract | None:
    if not route:
        return None
    return skill.runtime.output_contracts.get(route)


def _duration_from_user_field(user_text: str, field: str) -> float | None:
    if not field:
        return None
    pattern = re.compile(
        rf"(?im)^\s*{re.escape(field)}\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*(?:s|sec|secs|second|seconds)\b"
    )
    matches = pattern.findall(str(user_text or ""))
    if not matches:
        return None
    try:
        duration = float(matches[-1])
    except ValueError:
        return None
    return duration if duration > 0 else None


def _integer_from_user_field(user_text: str, field: str) -> int | None:
    if not field:
        return None
    pattern = re.compile(rf"(?im)^\s*{re.escape(field)}\s*:\s*(\d+)\b")
    matches = pattern.findall(str(user_text or ""))
    if not matches:
        return None
    try:
        return max(0, int(matches[-1]))
    except ValueError:
        return None


def _string_from_user_field(user_text: str, field: str) -> str | None:
    if not field:
        return None
    pattern = re.compile(rf"(?im)^\s*{re.escape(field)}\s*:\s*([^\r\n]+)")
    matches = pattern.findall(str(user_text or ""))
    return matches[-1].strip() if matches and matches[-1].strip() else None


def _timeline_timestamps(text: str) -> list[tuple[str, float]]:
    """Read conventional ``MM:SS.mmm`` or ``HH:MM:SS.mmm`` timestamps."""

    found: list[tuple[str, float]] = []
    pattern = re.compile(r"(?<!\d)(?:(\d{1,3}):)?(\d{1,2}):(\d{1,2}(?:\.\d+)?)(?!\d)")
    for match in pattern.finditer(str(text or "")):
        try:
            hours = int(match.group(1) or 0)
            minutes = int(match.group(2))
            seconds = float(match.group(3))
        except ValueError:
            continue
        if minutes >= 60 or seconds >= 60:
            continue
        found.append((match.group(0), hours * 3600 + minutes * 60 + seconds))
    return found


def validate_output_contract(text: str, contract: OutputContract | None, user_text: str = "") -> list[str]:
    """Return human-readable violations for an optional Skill output contract."""

    if contract is None:
        return []
    source = str(text or "").strip()
    violations: list[str] = []
    matches: list[tuple[str, re.Match[str]]] = []
    for heading in contract.required_headings:
        match = re.search(rf"(?im)^[ \t]*{re.escape(heading)}(?P<inline>[^\r\n]*)$", source)
        if match is None:
            violations.append(f"missing required heading `{heading}`")
        else:
            matches.append((heading, match))
    if len(matches) == len(contract.required_headings) and [match.start() for _, match in matches] != sorted(match.start() for _, match in matches):
        violations.append("required headings are not in the declared order")
    if contract.require_first_heading and contract.required_headings:
        first = contract.required_headings[0]
        if not re.match(rf"(?is)^\s*{re.escape(first)}", source):
            violations.append(f"output must begin with `{first}`")
    ordered = sorted(matches, key=lambda item: item[1].start())
    section_contents: dict[str, str] = {}
    for index, (heading, match) in enumerate(ordered):
        next_start = ordered[index + 1][1].start() if index + 1 < len(ordered) else len(source)
        content = (match.group("inline") + source[match.end() : next_start]).strip()
        section_contents[heading] = content
        if not content:
            violations.append(f"required heading `{heading}` has no content")
    for heading in contract.forbidden_headings:
        if re.search(rf"(?im)^[ \t]*{re.escape(heading)}", source):
            violations.append(f"forbidden heading `{heading}` is present")
    for pattern in contract.forbidden_patterns:
        if re.search(pattern, source):
            violations.append(f"forbidden output pattern `{pattern}` is present")
    duration = _duration_from_user_field(user_text, contract.timeline_duration_field)
    if duration is not None:
        for timestamp, seconds in _timeline_timestamps(source):
            if seconds > duration + 0.0001:
                violations.append(
                    f"timestamp `{timestamp}` exceeds {contract.timeline_duration_field} ({duration:g} seconds)"
                )
    for label, count_field in contract.label_count_fields.items():
        limit = _integer_from_user_field(user_text, count_field)
        if limit is None:
            continue
        label_pattern = re.compile(rf"(?i)<\s*{re.escape(label)}\s+(\d+)\s*>")
        indices = {int(item) for item in label_pattern.findall(source)}
        if len(indices) > limit or any(index > limit for index in indices):
            violations.append(f"{label} labels exceed {count_field} ({limit})")
    for heading, value_field in contract.exact_value_fields.items():
        expected = _string_from_user_field(user_text, value_field)
        actual = section_contents.get(heading)
        if expected is not None and actual is not None and actual.casefold() != expected.casefold():
            violations.append(f"content of `{heading}` must equal {value_field} (`{expected}`)")
    return violations


def build_runtime_contract_instruction(skill: Skill, route: str, references: tuple[str, ...]) -> str:
    """Render generic Skill-declared runtime context after its long guide."""

    if not route:
        return ""
    chunks = [f"Skill runtime mode: {route}."]
    if references:
        chunks.append("The following Skill references were selected and are already loaded: " + ", ".join(references) + ". Do not request them again.")
    contract = output_contract_for_route(skill, route)
    if contract is None:
        return " ".join(chunks)
    if contract.required_headings:
        chunks.append("Your final response must contain these non-empty headings in this exact order: " + ", ".join(contract.required_headings) + ".")
    if contract.require_first_heading and contract.required_headings:
        chunks.append(f"The response must begin with `{contract.required_headings[0]}`.")
    if contract.forbidden_headings:
        chunks.append("Do not emit these headings: " + ", ".join(contract.forbidden_headings) + ".")
    if contract.forbidden_patterns:
        chunks.append("Do not emit the Skill-declared forbidden output patterns.")
    if contract.timeline_duration_field:
        chunks.append(f"Keep all timeline timestamps at or before the user-supplied `{contract.timeline_duration_field}`.")
    for label, count_field in contract.label_count_fields.items():
        chunks.append(f"Do not use more `{label} N` labels than the user-supplied `{count_field}` permits.")
    for heading, value_field in contract.exact_value_fields.items():
        chunks.append(f"When `{value_field}` is supplied, the content of `{heading}` must equal that value exactly.")
    return " ".join(chunks)


def choose_skill(skills: list[Skill], user_text: str) -> Skill | None:
    """A deterministic offline selector used when the node is set to Auto."""

    terms = set(re.findall(r"[\w-]+", str(user_text or "").casefold()))
    if not skills:
        return None
    ranked: list[tuple[int, str, Skill]] = []
    for skill in skills:
        haystack = f"{skill.id} {skill.name} {skill.description}".casefold()
        score = sum(1 for term in terms if len(term) > 1 and term in haystack)
        ranked.append((score, skill.id, skill))
    best = max(ranked)
    return best[2] if best[0] else None


def empty_flow_state() -> dict[str, Any]:
    return {"skill": "", "skill_name": "", "stage": "not started", "loaded_references": [], "final": ""}


def parse_flow_state(raw: str) -> dict[str, Any]:
    state = empty_flow_state()
    try:
        received = json.loads(raw or "{}")
    except json.JSONDecodeError:
        received = {}
    if not isinstance(received, dict):
        return state
    state["skill"] = str(received.get("skill") or "")
    state["skill_name"] = str(received.get("skill_name") or "")[:120]
    state["stage"] = str(received.get("stage") or "not started")[:120]
    state["loaded_references"] = [
        str(item) for item in received.get("loaded_references", []) if isinstance(item, str)
    ]
    state["final"] = str(received.get("final") or "")
    return state


def parse_skill_state(reply: str) -> tuple[str, dict[str, Any]]:
    matches = list(_STATE_TAG.finditer(reply or ""))
    if not matches:
        return str(reply or "").strip(), {}
    match = matches[-1]
    try:
        state = json.loads(match.group(1))
    except json.JSONDecodeError:
        state = {}
    if not isinstance(state, dict):
        state = {}
    without_tag = (str(reply)[: match.start()] + str(reply)[match.end() :]).strip()
    return without_tag, state


def build_skill_instruction(skill: Skill, flow_state: dict[str, Any], root: Path = SKILLS_ROOT) -> str:
    loaded = [item for item in flow_state.get("loaded_references", []) if item in skill.references]
    chunks = [
        "You are following a local ComfyUI Skill. A Skill is an instruction document, not permission to execute tools.",
        "Do not claim to have accessed the network, filesystem, external applications, or generated media unless the user supplied evidence in this conversation.",
        f"Skill: {skill.name} ({skill.id})\n\n{skill.document}",
        "Available references:\n" + ("\n".join(f"- {item}" for item in skill.references) or "- none"),
    ]
    for reference in loaded:
        chunks.append(f"Reference {reference}:\n{read_reference(skill, reference, root)}")
    chunks.append(
        "At the end of every answer append one machine-readable state tag exactly once: "
        '<lwb_skill_state>{"stage":"short stage","options":[],"load_references":[],"final":false}</lwb_skill_state>. '
        "Only request listed references. Set final true only when the requested text deliverable is complete."
    )
    return "\n\n".join(chunks)
