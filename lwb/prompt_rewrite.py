"""Structured Qwen-Image prompt-enhancer support.

The prompt-enhancer checkpoints are ordinary chat models served through the
same backend socket as the rest of Workbench.  This module contains only the
task-specific message shape, production sampling profiles, and strict output
contract.  In particular, it deliberately does not embed either Qwen's system
prompts or any model assets: callers must supply the matching prompt as text or
as a local file.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


MAX_PROMPT_REWRITE_IMAGES = 10
MAX_SYSTEM_PROMPT_BYTES = 1024 * 1024
_THINK_BLOCK = re.compile(r"<think>\s*(.*?)\s*</think>", re.IGNORECASE | re.DOTALL)
_RATIO = re.compile(r"^([1-9]\d*):([1-9]\d*)$")
_IMAGE_REFERENCE = re.compile(r"^<image([1-9]\d*)>$")


class PromptRewriteFormatError(ValueError):
    """The model did not return the declared prompt-enhancer JSON contract."""


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

    def request_settings(self, seed: int) -> dict[str, Any]:
        return {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "min_p": self.min_p,
            "presence_penalty": self.presence_penalty,
            "seed": int(seed),
            # Both PE checkpoints are trained to reason before emitting the
            # small JSON answer.  This must not inherit Prompt's generic
            # thinking default, which is intentionally off.
            "enable_thinking": True,
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
) -> tuple[int, int, str]:
    """Convert a PE ``wh_ratio`` into generation dimensions.

    Qwen PE deliberately returns a ratio rather than fixed pixel dimensions.
    Keeping this conversion separate lets workflows choose their own pixel
    budget while preserving the model-selected composition.
    """

    ratio = str(wh_ratio or "").strip()
    match = _RATIO.fullmatch(ratio)
    if not match:
        raise ValueError("wh_ratio must be a positive W:H integer ratio")
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

    ratio_width, ratio_height = (int(part) for part in match.groups())
    divisor = math.gcd(ratio_width, ratio_height)
    normalized_ratio = f"{ratio_width // divisor}:{ratio_height // divisor}"
    target_pixels = target_megapixels * 1024 * 1024
    scale = math.sqrt(target_pixels / (ratio_width * ratio_height))
    width = max(rounding_multiple, round(ratio_width * scale / rounding_multiple) * rounding_multiple)
    height = max(rounding_multiple, round(ratio_height * scale / rounding_multiple) * rounding_multiple)
    return width, height, normalized_ratio


def load_system_prompt(system_prompt: str = "", system_prompt_path: str = "") -> str:
    """Load exactly one user-supplied system prompt source.

    A directory path is accepted as a convenience and resolves to its
    ``system_prompt.txt``.  No prompt is downloaded or bundled by Workbench.
    """

    inline = str(system_prompt or "").strip()
    raw_path = str(system_prompt_path or "").strip()
    if inline and raw_path:
        raise ValueError("Provide the System Prompt as text or a local file, not both")
    if inline:
        if len(inline.encode("utf-8")) > MAX_SYSTEM_PROMPT_BYTES:
            raise ValueError("System Prompt text exceeds the 1 MiB safety limit")
        return inline.lstrip("\ufeff")
    if not raw_path:
        raise ValueError(
            "A matching Qwen PE System Prompt is required. Paste it into system_prompt "
            "or set system_prompt_path to a local system_prompt.txt file."
        )

    path = Path(raw_path).expanduser()
    if path.is_dir():
        path = path / "system_prompt.txt"
    if not path.is_file():
        raise ValueError(f"System Prompt file does not exist: {path}")
    if path.stat().st_size > MAX_SYSTEM_PROMPT_BYTES:
        raise ValueError(f"System Prompt file exceeds the 1 MiB safety limit: {path}")
    try:
        loaded = path.read_text(encoding="utf-8").strip().lstrip("\ufeff")
    except UnicodeDecodeError as exc:
        raise ValueError(f"System Prompt file must be UTF-8 text: {path}") from exc
    if not loaded:
        raise ValueError(f"System Prompt file is empty: {path}")
    return loaded


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

    if images:
        user_content: str | list[dict[str, Any]] = [
            *({"type": "image_url", "image_url": {"url": image}} for image in images),
            {"type": "text", "text": user_prompt},
        ]
    else:
        user_content = user_prompt
    return [
        {"role": "system", "content": system_prompt},
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
    if "\n" in rewritten or "\r" in rewritten:
        raise PromptRewriteFormatError("rewritten_prompt must be a single paragraph without line breaks")
    if wh_ratio and not _RATIO.fullmatch(wh_ratio):
        raise PromptRewriteFormatError("wh_ratio must be empty or a positive W:H integer ratio")

    if not profile.takes_images:
        if not wh_ratio:
            raise PromptRewriteFormatError("t2i output requires a non-empty wh_ratio")
        if ratio_follow:
            raise PromptRewriteFormatError("t2i output cannot set ratio_follow")
    else:
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
) -> tuple[str, str]:
    structured = getattr(backend, "chat_response", None)
    if callable(structured):
        completion = structured(messages, **settings)
        content = getattr(completion, "content", "")
        reported_thinking = str(getattr(completion, "reasoning", "") or "").strip()
    else:
        content = backend.chat(messages, **settings)
        reported_thinking = ""
    answer, inline_thinking = split_prompt_rewrite_thinking(content)
    thinking = "\n\n".join(part for part in (reported_thinking, inline_thinking) if part)
    return answer, thinking


def rewrite_prompt(
    backend: Any,
    prompt: str,
    *,
    task: str = "t2i",
    system_prompt: str = "",
    system_prompt_path: str = "",
    image_data_urls: Iterable[str] = (),
    seed: int = 42,
) -> PromptRewriteResult:
    """Run a Qwen PE request with one format-only retry."""

    profile = get_prompt_rewrite_profile(task)
    prompt_text = load_system_prompt(system_prompt, system_prompt_path)
    images = tuple(image_data_urls)
    base_messages = build_prompt_rewrite_messages(profile, prompt_text, prompt, images)
    settings = profile.request_settings(seed)
    messages = list(base_messages)
    all_thinking: list[str] = []
    failures: list[str] = []

    for attempt in range(2):
        answer, thinking = _backend_completion(backend, messages, settings)
        if thinking:
            all_thinking.append(thinking)
        try:
            parsed = parse_prompt_rewrite_json(answer, profile, image_count=len(images))
        except PromptRewriteFormatError as exc:
            failures.append(str(exc))
            if attempt == 1:
                details = "; retry: ".join(failures)
                raise PromptRewriteFormatError(
                    "Qwen Prompt Enhancer returned invalid structured output twice: " + details
                ) from exc
            if profile.takes_images:
                expected = (
                    'Use exactly the string fields "rewritten_prompt", "wh_ratio", and "ratio_follow". '
                    'Exactly one of "wh_ratio" and "ratio_follow" must be non-empty; a non-empty '
                    'ratio_follow must look like "<image1>".'
                )
            else:
                expected = (
                    'Use exactly the string fields "rewritten_prompt" and "wh_ratio", with wh_ratio '
                    'set to the selected positive W:H ratio.'
                )
            messages = [
                *base_messages,
                {"role": "assistant", "content": answer},
                {
                    "role": "user",
                    "content": (
                        "Your previous answer did not match the required machine-readable format. "
                        f"Return exactly one valid JSON object. {expected} Use no Markdown "
                        "fence, commentary, or additional fields."
                    ),
                },
            ]
            continue
        return PromptRewriteResult(
            rewritten_prompt=parsed["rewritten_prompt"],
            wh_ratio=parsed["wh_ratio"],
            ratio_follow=parsed["ratio_follow"],
            thinking="\n\n".join(all_thinking),
            retried=bool(attempt),
        )

    raise AssertionError("unreachable prompt-rewrite attempt state")
