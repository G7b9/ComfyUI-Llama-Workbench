"""Structured Qwen-Image prompt-enhancer support.

The prompt-enhancer checkpoints are ordinary chat models served through the
same backend socket as the rest of Workbench. This module contains the
task-specific message shape, production sampling profiles, strict output
contract, and task-aware system-prompt resolution. Official prompt assets are
still not bundled; callers can provide them explicitly or place them next to
the matching checkpoint.
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable


MAX_PROMPT_REWRITE_IMAGES = 10
MAX_SYSTEM_PROMPT_BYTES = 1024 * 1024
PROMPT_REWRITE_DEBUG_ENV = "LWB_PROMPT_REWRITE_DEBUG"
PROMPT_REWRITE_DEBUG_MAX_CHARS = 4000
PROMPT_REWRITE_AUTO_ASPECT_RATIO = "Auto (use Prompt Enhancer)"
QWEN_IMAGE_21_ASPECT_RATIOS = (
    "1:1",
    "3:2",
    "2:3",
    "4:3",
    "3:4",
    "5:4",
    "4:5",
    "16:9",
    "9:16",
    "2:1",
    "1:2",
    "21:9",
    "9:21",
    "3:1",
    "1:3",
)
_THINK_BLOCK = re.compile(r"<think>\s*(.*?)\s*</think>", re.IGNORECASE | re.DOTALL)
_RATIO_COMPONENT = r"(?:0\.\d+|[1-9]\d*(?:\.\d+)?)"
_RATIO = re.compile(rf"^({_RATIO_COMPONENT}):({_RATIO_COMPONENT})$")
_IMAGE_REFERENCE = re.compile(r"^<image([1-9]\d*)>$")
_IMAGE_REFERENCE_CANDIDATE = re.compile(r"<image[^>]*>", re.IGNORECASE)
_TRUNCATED_FINISH_REASONS = {"length", "limit", "max_tokens", "max_output_tokens"}
_TASK_SYSTEM_PROMPT_FILENAMES = {
    "t2i": ("system_prompt_t2i.txt", "system_prompt.txt"),
    "edit": ("system_prompt_edit.txt", "system_prompt.txt"),
}


class PromptRewriteFormatError(ValueError):
    """The model did not return the declared prompt-enhancer JSON contract."""


@dataclass(frozen=True, slots=True)
class LoadedSystemPrompt:
    """A validated prompt and a non-sensitive description of its source."""

    text: str
    source: str


def _prompt_rewrite_debug_enabled(explicit: bool = False) -> bool:
    if explicit:
        return True
    value = os.environ.get(PROMPT_REWRITE_DEBUG_ENV, "")
    return value.strip().lower() in {"1", "true", "yes", "on", "debug"}


def _prompt_rewrite_debug_excerpt(value: Any) -> str:
    """Return a bounded repr suitable for the ComfyUI console."""

    text = str(value or "")
    if len(text) > PROMPT_REWRITE_DEBUG_MAX_CHARS:
        text = text[:PROMPT_REWRITE_DEBUG_MAX_CHARS] + "...<truncated>"
    return repr(text)


def _normalize_ratio(value: str) -> tuple[int, int, str]:
    """Convert positive integer or finite-decimal W:H text to simplest integers."""

    match = _RATIO.fullmatch(str(value or "").strip())
    if not match:
        raise ValueError("wh_ratio must be a positive W:H ratio with numeric components")
    try:
        width = Fraction(match.group(1))
        height = Fraction(match.group(2))
    except (TypeError, ValueError, ZeroDivisionError) as exc:
        raise ValueError("wh_ratio must be a positive W:H ratio with numeric components") from exc
    if width <= 0 or height <= 0:
        raise ValueError("wh_ratio must be a positive W:H ratio with numeric components")
    ratio = width / height
    return ratio.numerator, ratio.denominator, f"{ratio.numerator}:{ratio.denominator}"


@dataclass(frozen=True, slots=True)
class PromptRewriteProfile:
    """Production inference settings for one Qwen-Image PE task."""

    name: str
    takes_images: bool
    temperature: float
    top_p: float
    top_k: int
    min_p: float
    presence_penalty: float
    max_tokens: int

    def request_settings(self, seed: int, *, enable_thinking: bool = True) -> dict[str, Any]:
        return {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "min_p": self.min_p,
            "presence_penalty": self.presence_penalty,
            "seed": int(seed),
            # Both PE checkpoints are trained to reason before emitting the
            # small JSON answer. Keep it on by default, while allowing the
            # node to disable reasoning for faster or lower-token requests.
            "enable_thinking": bool(enable_thinking),
        }


PROMPT_REWRITE_PROFILES: dict[str, PromptRewriteProfile] = {
    "t2i": PromptRewriteProfile(
        name="t2i",
        takes_images=False,
        temperature=1.0,
        top_p=0.95,
        top_k=20,
        min_p=0.0,
        presence_penalty=1.5,
        max_tokens=16256,
    ),
    # The transport and output contract are ready for PE-I2I.  A compatible
    # multimodal projector must still be supplied by Start Server/its runtime.
    "edit": PromptRewriteProfile(
        name="edit",
        takes_images=True,
        temperature=1.0,
        top_p=0.95,
        top_k=20,
        min_p=0.0,
        presence_penalty=0.0,
        max_tokens=24000,
    ),
}


@dataclass(frozen=True, slots=True)
class PromptRewriteResult:
    rewritten_prompt: str
    wh_ratio: str
    ratio_follow: str
    thinking: str
    retried: bool = False

    def as_dict(self) -> dict[str, str]:
        return {
            "rewritten_prompt": self.rewritten_prompt,
            "wh_ratio": self.wh_ratio,
            "ratio_follow": self.ratio_follow,
        }

    def as_json(self) -> str:
        return json.dumps(self.as_dict(), ensure_ascii=False, separators=(",", ":"))


@dataclass(frozen=True, slots=True)
class PromptRewriteCanvasSpec:
    width: int
    height: int
    ratio_source: str


def get_prompt_rewrite_profile(task: str) -> PromptRewriteProfile:
    name = str(task or "").strip().lower()
    if name == "i2i":
        name = "edit"
    try:
        return PROMPT_REWRITE_PROFILES[name]
    except KeyError as exc:
        choices = ", ".join(PROMPT_REWRITE_PROFILES)
        raise ValueError(f"Unknown prompt-enhancer task {task!r}; choose one of: {choices}") from exc


def prompt_rewrite_dimensions(
    wh_ratio: str,
    *,
    megapixels: float = 1.0,
    multiple: int = 8,
    aspect_ratio_override: str = PROMPT_REWRITE_AUTO_ASPECT_RATIO,
) -> tuple[int, int, str]:
    """Convert a PE ``wh_ratio`` into generation dimensions.

    Qwen PE deliberately returns a ratio rather than fixed pixel dimensions.
    Keeping this conversion separate lets workflows choose their own pixel
    budget while preserving the model-selected composition.
    """

    override = str(aspect_ratio_override or "").strip()
    ratio = str(wh_ratio or "").strip()
    if override and override != PROMPT_REWRITE_AUTO_ASPECT_RATIO:
        ratio = override
    try:
        ratio_width, ratio_height, normalized_ratio = _normalize_ratio(ratio)
    except ValueError as exc:
        raise ValueError(str(exc)) from exc
    try:
        target_megapixels = float(megapixels)
    except (TypeError, ValueError) as exc:
        raise ValueError("megapixels must be a positive number") from exc
    if not math.isfinite(target_megapixels) or target_megapixels <= 0:
        raise ValueError("megapixels must be a positive number")
    try:
        rounding_multiple = int(multiple)
    except (TypeError, ValueError) as exc:
        raise ValueError("multiple must be a positive integer") from exc
    if rounding_multiple <= 0:
        raise ValueError("multiple must be a positive integer")

    target_pixels = target_megapixels * 1024 * 1024
    scale = math.sqrt(target_pixels / (ratio_width * ratio_height))
    width = max(rounding_multiple, round(ratio_width * scale / rounding_multiple) * rounding_multiple)
    height = max(rounding_multiple, round(ratio_height * scale / rounding_multiple) * rounding_multiple)
    return width, height, normalized_ratio


def prompt_rewrite_canvas_spec(
    wh_ratio: str,
    ratio_follow: str,
    image_dimensions: Iterable[tuple[int, int]] = (),
    *,
    megapixels: float = 1.0,
    multiple: int = 32,
    aspect_ratio_override: str = PROMPT_REWRITE_AUTO_ASPECT_RATIO,
    follow_input_size: bool = True,
) -> PromptRewriteCanvasSpec:
    """Resolve PE sizing fields into a Qwen-Image-2.1 canvas specification."""

    override = str(aspect_ratio_override or "").strip()
    ratio = str(wh_ratio or "").strip()
    follow = str(ratio_follow or "").strip()
    dimensions = tuple((int(width), int(height)) for width, height in image_dimensions)
    try:
        rounding_multiple = int(multiple)
    except (TypeError, ValueError) as exc:
        raise ValueError("multiple must be a positive integer") from exc
    if rounding_multiple <= 0 or rounding_multiple % 16:
        raise ValueError("multiple must be a positive multiple of 16 for Qwen-Image-2.1")

    if override and override != PROMPT_REWRITE_AUTO_ASPECT_RATIO:
        width, height, normalized = prompt_rewrite_dimensions(
            override,
            megapixels=megapixels,
            multiple=rounding_multiple,
            aspect_ratio_override=override,
        )
        return PromptRewriteCanvasSpec(width, height, f"override:{normalized}")

    if bool(ratio) == bool(follow):
        raise ValueError("exactly one of wh_ratio and ratio_follow must be set in Auto mode")
    if ratio:
        width, height, normalized = prompt_rewrite_dimensions(
            ratio,
            megapixels=megapixels,
            multiple=rounding_multiple,
        )
        return PromptRewriteCanvasSpec(width, height, f"wh_ratio:{normalized}")

    match = _IMAGE_REFERENCE.fullmatch(follow)
    if not match:
        raise ValueError("ratio_follow must use the form <imageN>")
    image_index = int(match.group(1))
    if image_index > len(dimensions):
        raise ValueError(
            f"ratio_follow references image {image_index}, but Canvas received {len(dimensions)} image(s)"
        )
    source_width, source_height = dimensions[image_index - 1]
    if source_width <= 0 or source_height <= 0:
        raise ValueError(f"image {image_index} has invalid dimensions {source_width}x{source_height}")
    if follow_input_size:
        width = max(rounding_multiple, round(source_width / rounding_multiple) * rounding_multiple)
        height = max(rounding_multiple, round(source_height / rounding_multiple) * rounding_multiple)
        return PromptRewriteCanvasSpec(width, height, f"ratio_follow:{follow}:input_size")

    width, height, _ = prompt_rewrite_dimensions(
        f"{source_width}:{source_height}",
        megapixels=megapixels,
        multiple=rounding_multiple,
    )
    return PromptRewriteCanvasSpec(width, height, f"ratio_follow:{follow}:aspect_only")


def _normalize_prompt_task(task: str) -> str:
    normalized = str(task or "").strip().lower()
    if normalized == "i2i":
        normalized = "edit"
    if normalized not in _TASK_SYSTEM_PROMPT_FILENAMES:
        raise ValueError(f"Unknown prompt-enhancer task {task!r}; choose from: t2i, edit")
    return normalized


def _validate_system_prompt_text(value: str, source: str) -> LoadedSystemPrompt:
    text = str(value or "").strip().lstrip("\ufeff")
    if len(text.encode("utf-8")) > MAX_SYSTEM_PROMPT_BYTES:
        raise ValueError("System Prompt text exceeds the 1 MiB safety limit")
    if not text:
        raise ValueError(f"System Prompt is empty: {source}")
    return LoadedSystemPrompt(text=text, source=source)


def _load_system_prompt_file(path: Path, source_label: str = "System Prompt") -> LoadedSystemPrompt:
    if not path.is_file():
        raise ValueError(f"{source_label} file does not exist: {path}")
    if path.stat().st_size > MAX_SYSTEM_PROMPT_BYTES:
        raise ValueError(f"{source_label} file exceeds the 1 MiB safety limit: {path}")
    try:
        loaded = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{source_label} file must be UTF-8 text: {path}") from exc
    return _validate_system_prompt_text(loaded, str(path))


def _prompt_search_directories(model_path: str) -> list[Path]:
    """Return likely checkpoint directories without guessing remote models."""

    raw = str(model_path or "").strip()
    if not raw:
        return []
    path = Path(raw).expanduser()
    directories: list[Path] = []
    if path.is_dir():
        directories.append(path)
    elif path.is_file():
        directories.append(path.parent)
        sibling_directory = path.parent / path.stem
        if sibling_directory.is_dir():
            directories.append(sibling_directory)
    else:
        # A cached backend may retain a path after its model has been released.
        # Do not turn a missing auto-discovery path into an explicit-file error.
        return []

    unique: list[Path] = []
    seen: set[Path] = set()
    for directory in directories:
        resolved = directory.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    return unique


def _load_prompt_from_path(raw_path: str, task: str, *, explicit: bool) -> LoadedSystemPrompt:
    path = Path(str(raw_path).strip()).expanduser()
    if path.is_file():
        return _load_system_prompt_file(path)
    if not path.is_dir():
        if explicit:
            raise ValueError(f"System Prompt path does not exist: {path}")
        raise FileNotFoundError(path)

    for filename in _TASK_SYSTEM_PROMPT_FILENAMES[task]:
        candidate = path / filename
        if candidate.is_file():
            return _load_system_prompt_file(candidate)
    if explicit:
        expected = ", ".join(_TASK_SYSTEM_PROMPT_FILENAMES[task])
        raise ValueError(f"No {task} System Prompt found in {path}; expected one of: {expected}")
    raise FileNotFoundError(path)


def resolve_system_prompt(
    task: str = "t2i",
    *,
    system_prompt: str = "",
    system_prompt_path: str = "",
    task_system_prompt: str = "",
    task_system_prompt_path: str = "",
    model_path: str = "",
    auto_load: bool = True,
) -> LoadedSystemPrompt:
    """Resolve one task-matching prompt with explicit overrides first.

    ``system_prompt`` and ``system_prompt_path`` are retained as legacy
    current-task overrides. The task-specific fields win when present. When
    auto loading is enabled, a model directory or GGUF sibling directory is
    searched for the task filename before the generic ``system_prompt.txt``.
    """

    normalized_task = _normalize_prompt_task(task)
    specific_text = str(task_system_prompt or "").strip()
    specific_path = str(task_system_prompt_path or "").strip()
    if specific_text and specific_path:
        raise ValueError(
            f"Provide the {normalized_task} System Prompt as text or a local file, not both"
        )
    if specific_text:
        return _validate_system_prompt_text(specific_text, f"inline:{normalized_task}")
    if specific_path:
        return _load_prompt_from_path(specific_path, normalized_task, explicit=True)

    # Preserve existing workflows: the generic fields remain valid as an
    # override for whichever task is selected.
    inline = str(system_prompt or "").strip()
    raw_path = str(system_prompt_path or "").strip()
    if inline and raw_path:
        raise ValueError("Provide the System Prompt as text or a local file, not both")
    if inline:
        return _validate_system_prompt_text(inline, f"inline:{normalized_task}")
    if raw_path:
        return _load_prompt_from_path(raw_path, normalized_task, explicit=True)

    if auto_load:
        for directory in _prompt_search_directories(model_path):
            try:
                return _load_prompt_from_path(str(directory), normalized_task, explicit=False)
            except FileNotFoundError:
                continue

    filenames = ", ".join(_TASK_SYSTEM_PROMPT_FILENAMES[normalized_task])
    raise ValueError(
        f"A matching {normalized_task} Qwen PE System Prompt is required. "
        f"Paste it into {normalized_task}_system_prompt, set its path, or place "
        f"{filenames} next to the model."
    )


def load_system_prompt(
    system_prompt: str = "",
    system_prompt_path: str = "",
    *,
    task: str = "t2i",
    task_system_prompt: str = "",
    task_system_prompt_path: str = "",
    model_path: str = "",
    auto_load: bool = True,
) -> str:
    """Return the selected task-matching system prompt text.

    This compatibility wrapper keeps the original two-argument API available
    to integrations while the node uses :func:`resolve_system_prompt` to also
    record the selected source for debug output.
    """

    return resolve_system_prompt(
        task,
        system_prompt=system_prompt,
        system_prompt_path=system_prompt_path,
        task_system_prompt=task_system_prompt,
        task_system_prompt_path=task_system_prompt_path,
        model_path=model_path,
        auto_load=auto_load,
    ).text


def build_edit_image_reference_rule(image_count: int) -> str:
    """Describe the runtime image numbering without embedding Qwen's prompt."""

    count = int(image_count)
    if count < 1 or count > MAX_PROMPT_REWRITE_IMAGES:
        raise ValueError(f"edit image count must be between 1 and {MAX_PROMPT_REWRITE_IMAGES}")
    valid_tags = ", ".join(f"<image{index}>" for index in range(1, count + 1))
    mapping = "; ".join(
        f"the {index}{'st' if index == 1 else 'nd' if index == 2 else 'rd' if index == 3 else 'th'} "
        f"image part is <image{index}>"
        for index in range(1, count + 1)
    )
    if count == 1:
        prompt_rule = (
            "Do not use an image tag inside rewritten_prompt for this single-image request; "
            "<image1> remains valid for ratio_follow."
        )
    else:
        prompt_rule = (
            "When rewritten_prompt refers to the inputs, use these exact tags and reference every "
            "input image at least once; do not use natural-language numbering in their place."
        )
    return (
        "# Runtime Image Mapping\n"
        f"This request contains exactly {count} input image(s), in message order. {mapping}. "
        f"The only valid image tags are: {valid_tags}. {prompt_rule} Never emit any other image tag."
    )


def build_prompt_rewrite_messages(
    profile: PromptRewriteProfile,
    system_prompt: str,
    prompt: str,
    image_data_urls: Iterable[str] = (),
) -> list[dict[str, Any]]:
    """Build the PE conversation, keeping edit images before the user text."""

    user_prompt = str(prompt or "").strip()
    if not user_prompt:
        raise ValueError("prompt must not be empty")
    images = [str(item) for item in image_data_urls if str(item)]
    if len(images) > MAX_PROMPT_REWRITE_IMAGES:
        raise ValueError(f"Prompt Enhancer accepts at most {MAX_PROMPT_REWRITE_IMAGES} images")
    if profile.takes_images and not images:
        raise ValueError("The edit prompt-enhancer profile requires at least one input image")
    if not profile.takes_images and images:
        raise ValueError("The t2i prompt-enhancer profile does not accept input images")

    effective_system_prompt = str(system_prompt)
    if profile.takes_images:
        effective_system_prompt = (
            effective_system_prompt.rstrip() + "\n\n" + build_edit_image_reference_rule(len(images))
        )

    if images:
        user_content: str | list[dict[str, Any]] = [
            *({"type": "image_url", "image_url": {"url": image}} for image in images),
            {"type": "text", "text": user_prompt},
        ]
    else:
        user_content = user_prompt
    return [
        {"role": "system", "content": effective_system_prompt},
        {"role": "user", "content": user_content},
    ]


def split_prompt_rewrite_thinking(text: Any) -> tuple[str, str]:
    """Return ``(answer, thinking)`` for common Qwen/llama-server layouts."""

    raw = str(text or "").strip()
    matches = list(_THINK_BLOCK.finditer(raw))
    if matches:
        thinking = "\n\n".join(match.group(1).strip() for match in matches if match.group(1).strip())
        return _THINK_BLOCK.sub("", raw).strip(), thinking
    if "</think>" in raw.lower():
        # Some chat templates pre-fill the opening tag, so decoded content
        # starts inside the thought and contains only the closing tag.
        position = raw.lower().find("</think>")
        return raw[position + len("</think>") :].strip(), raw[:position].strip()
    if raw.lower().startswith("<think>"):
        return "", raw[len("<think>") :].strip()
    return raw, ""


def parse_prompt_rewrite_json(
    answer: str,
    profile: PromptRewriteProfile,
    *,
    image_count: int = 0,
) -> dict[str, str]:
    """Strictly parse and validate one PE answer.

    Unlike a chat-oriented best-effort parser, this function accepts no prose,
    Markdown fences, repaired JSON, aliases, or unknown fields.  Downstream
    image nodes must be able to trust every returned field.
    """

    raw = str(answer or "").strip()
    if not raw:
        raise PromptRewriteFormatError("assistant answer is empty")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        parsed: dict[str, Any] = {}
        for key, item in pairs:
            if key in parsed:
                raise PromptRewriteFormatError(f"JSON object repeats field: {key}")
            parsed[key] = item
        return parsed

    try:
        value = json.loads(raw, object_pairs_hook=unique_object)
    except json.JSONDecodeError as exc:
        raise PromptRewriteFormatError(f"assistant answer is not a single valid JSON object: {exc.msg}") from exc
    if not isinstance(value, dict):
        raise PromptRewriteFormatError("assistant answer must be a JSON object")

    required = {"rewritten_prompt", "wh_ratio", "ratio_follow"} if profile.takes_images else {
        "rewritten_prompt",
        "wh_ratio",
    }
    allowed = {"rewritten_prompt", "wh_ratio", "ratio_follow"}
    missing = required - value.keys()
    extra = value.keys() - allowed
    if missing:
        raise PromptRewriteFormatError(f"JSON object is missing field(s): {', '.join(sorted(missing))}")
    if extra:
        raise PromptRewriteFormatError(f"JSON object has unexpected field(s): {', '.join(sorted(extra))}")
    if any(not isinstance(value.get(key, ""), str) for key in allowed):
        raise PromptRewriteFormatError("rewritten_prompt, wh_ratio, and ratio_follow must be strings")

    rewritten = value["rewritten_prompt"].strip()
    wh_ratio = value["wh_ratio"].strip()
    ratio_follow = value.get("ratio_follow", "").strip()
    if not rewritten:
        raise PromptRewriteFormatError("rewritten_prompt must not be empty")
    if wh_ratio:
        try:
            _, _, wh_ratio = _normalize_ratio(wh_ratio)
        except ValueError as exc:
            raise PromptRewriteFormatError(
                "wh_ratio must be empty or a positive W:H ratio with numeric components"
            ) from exc

    prompt_references: list[int] = []
    for candidate in _IMAGE_REFERENCE_CANDIDATE.findall(rewritten):
        match = _IMAGE_REFERENCE.fullmatch(candidate)
        if not match:
            raise PromptRewriteFormatError(f"rewritten_prompt contains an invalid image reference: {candidate}")
        image_index = int(match.group(1))
        if image_index > image_count:
            raise PromptRewriteFormatError(
                f"rewritten_prompt references image {image_index}, but only {image_count} image(s) were supplied"
            )
        prompt_references.append(image_index)

    if not profile.takes_images:
        if prompt_references:
            raise PromptRewriteFormatError("t2i rewritten_prompt cannot contain image references")
        if not wh_ratio:
            raise PromptRewriteFormatError("t2i output requires a non-empty wh_ratio")
        if ratio_follow:
            raise PromptRewriteFormatError("t2i output cannot set ratio_follow")
    else:
        if image_count == 1 and prompt_references:
            raise PromptRewriteFormatError(
                "single-image edit rewritten_prompt must refer to the image naturally, without <image1>"
            )
        if image_count >= 2:
            missing_references = set(range(1, image_count + 1)) - set(prompt_references)
            if missing_references:
                missing = ", ".join(f"<image{index}>" for index in sorted(missing_references))
                raise PromptRewriteFormatError(
                    f"multi-image edit rewritten_prompt must reference every input image; missing: {missing}"
                )
        if bool(wh_ratio) == bool(ratio_follow):
            raise PromptRewriteFormatError("edit output must set exactly one of wh_ratio and ratio_follow")
        if ratio_follow:
            match = _IMAGE_REFERENCE.fullmatch(ratio_follow)
            if not match:
                raise PromptRewriteFormatError("ratio_follow must be empty or use the form <imageN>")
            image_index = int(match.group(1))
            if image_index > image_count:
                raise PromptRewriteFormatError(
                    f"ratio_follow references image {image_index}, but only {image_count} image(s) were supplied"
                )

    return {
        "rewritten_prompt": rewritten,
        "wh_ratio": wh_ratio,
        "ratio_follow": ratio_follow,
    }


def _backend_completion(
    backend: Any,
    messages: list[dict[str, Any]],
    settings: dict[str, Any],
) -> tuple[str, str, str]:
    structured = getattr(backend, "chat_response", None)
    if callable(structured):
        completion = structured(messages, **settings)
        content = getattr(completion, "content", "")
        reported_thinking = str(getattr(completion, "reasoning", "") or "").strip()
        finish_reason = str(getattr(completion, "finish_reason", "") or "").strip()
    else:
        content = backend.chat(messages, **settings)
        reported_thinking = ""
        finish_reason = ""
    answer, inline_thinking = split_prompt_rewrite_thinking(content)
    thinking = "\n\n".join(part for part in (reported_thinking, inline_thinking) if part)
    return answer, thinking, finish_reason


def _generation_was_truncated(finish_reason: str) -> bool:
    reason = str(finish_reason or "").strip().lower()
    return reason in _TRUNCATED_FINISH_REASONS or "length" in reason or "max_token" in reason


def rewrite_prompt(
    backend: Any,
    prompt: str,
    *,
    task: str = "t2i",
    system_prompt: str = "",
    system_prompt_path: str = "",
    task_system_prompt: str = "",
    task_system_prompt_path: str = "",
    model_path: str = "",
    auto_load_system_prompt: bool = True,
    enable_thinking: bool = True,
    image_data_urls: Iterable[str] = (),
    seed: int = 42,
    debug: bool = False,
) -> PromptRewriteResult:
    """Run a Qwen PE request with one format/truncation retry."""

    profile = get_prompt_rewrite_profile(task)
    loaded_prompt = resolve_system_prompt(
        profile.name,
        system_prompt=system_prompt,
        system_prompt_path=system_prompt_path,
        task_system_prompt=task_system_prompt,
        task_system_prompt_path=task_system_prompt_path,
        model_path=model_path,
        auto_load=bool(auto_load_system_prompt),
    )
    prompt_text = loaded_prompt.text
    if _prompt_rewrite_debug_enabled(debug):
        print(
            "[Llama Workbench][Prompt Enhancer][debug] "
            f"task={profile.name} thinking={'on' if enable_thinking else 'off'} "
            f"system_prompt_source={loaded_prompt.source}",
            flush=True,
        )
    images = tuple(image_data_urls)
    base_messages = build_prompt_rewrite_messages(profile, prompt_text, prompt, images)
    settings = {
        **profile.request_settings(seed, enable_thinking=bool(enable_thinking)),
        "accept_truncated_response": True,
    }
    messages = list(base_messages)
    all_thinking: list[str] = []
    failures: list[str] = []

    for attempt in range(2):
        attempt_settings = dict(settings)
        if attempt:
            attempt_settings["enable_thinking"] = False
        answer, thinking, finish_reason = _backend_completion(backend, messages, attempt_settings)
        if _prompt_rewrite_debug_enabled(debug):
            print(
                "[Llama Workbench][Prompt Enhancer][debug] "
                f"attempt={attempt + 1} finish_reason={finish_reason or 'unknown'} "
                f"thinking_chars={len(thinking)} answer={_prompt_rewrite_debug_excerpt(answer)}",
                flush=True,
            )
        if thinking:
            all_thinking.append(thinking)
        failure: PromptRewriteFormatError | None = None
        parsed: dict[str, str] | None = None
        if _generation_was_truncated(finish_reason):
            failure = PromptRewriteFormatError(
                f"generation was truncated (finish_reason={finish_reason or 'unknown'})"
            )
        else:
            try:
                parsed = parse_prompt_rewrite_json(answer, profile, image_count=len(images))
            except PromptRewriteFormatError as exc:
                failure = exc
        if failure is not None:
            if _prompt_rewrite_debug_enabled(debug):
                print(
                    "[Llama Workbench][Prompt Enhancer][debug] "
                    f"attempt={attempt + 1} validation_error={failure}",
                    flush=True,
                )
            failures.append(str(failure))
            if attempt == 1:
                details = "; retry: ".join(failures)
                raise PromptRewriteFormatError(
                    "Qwen Prompt Enhancer failed structured output after one retry: " + details
                ) from failure
            if profile.takes_images:
                expected = (
                    'Use exactly the string fields "rewritten_prompt", "wh_ratio", and "ratio_follow". '
                    'Exactly one of "wh_ratio" and "ratio_follow" must be non-empty; a non-empty '
                    'ratio_follow must use one of the runtime image tags. Obey the Runtime Image Mapping '
                    'rule for references inside rewritten_prompt.'
                )
            else:
                expected = (
                    'Use exactly the string fields "rewritten_prompt" and "wh_ratio", with wh_ratio '
                    'set to the selected positive W:H ratio; decimal components must be normalized to '
                    'the simplest integer ratio.'
                )
            messages = [
                *base_messages,
                {
                    "role": "user",
                    "content": (
                        "Retry the request without a thinking trace. Return the final JSON object only. "
                        f"{expected} Use no Markdown fence, commentary, or additional fields."
                    ),
                },
            ]
            continue
        assert parsed is not None
        return PromptRewriteResult(
            rewritten_prompt=parsed["rewritten_prompt"],
            wh_ratio=parsed["wh_ratio"],
            ratio_follow=parsed["ratio_follow"],
            thinking="\n\n".join(all_thinking),
            retried=bool(attempt),
        )

    raise AssertionError("unreachable prompt-rewrite attempt state")
