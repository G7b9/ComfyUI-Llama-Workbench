"""ComfyUI nodes for the independent Llama Workbench package."""

from __future__ import annotations

import gc
import json
import math
import re
import time
from typing import Any

from .backend import EmbeddedBackend, ServerBackend, backend_descriptor, make_text_message
from .embedded import EMBEDDED_MODELS, create_embedded_backend
from .media import image_tensor_to_data_urls
from .process import OWNED_SERVER, ServerLaunchConfig
from .prompt_rewrite import (
    MAX_PROMPT_REWRITE_IMAGES,
    PROMPT_REWRITE_AUTO_ASPECT_RATIO,
    QWEN_IMAGE_21_ASPECT_RATIOS,
    prompt_rewrite_dimensions,
    rewrite_prompt,
)
from .skills import (
    Skill,
    build_skill_instruction,
    choose_skill,
    discover_skills,
    empty_flow_state,
    get_skill,
    parse_flow_state,
    parse_skill_state,
)


BACKEND_TYPE = "LLAMA_WORKBENCH_BACKEND"
SETTINGS_TYPE = "LLAMA_WORKBENCH_CHAT_SETTINGS"
SKILL_TYPE = "LLAMA_WORKBENCH_SKILL"
DYNAMIC_IMAGE_LIMIT = 10
_THINKING_BLOCK = re.compile(r"<(?:think|thinking)>\s*(.*?)\s*</(?:think|thinking)>", re.IGNORECASE | re.DOTALL)
_UNCLOSED_THINKING_BLOCK = re.compile(r"^\s*<(?:think|thinking)>\s*(.*)$", re.IGNORECASE | re.DOTALL)
_THINKING_HEADING = re.compile(
    r"^\s*(?:here(?:'s| is) (?:(?:the|a) )?thinking process|thinking process|reasoning|analysis)\s*:\s*",
    re.IGNORECASE,
)
_FINAL_HEADING = re.compile(r"(?im)^\s*(?:final(?: answer| response)?|answer|response)\s*:\s*")
H3_AUTO_ASPECT_RATIO = "Auto (nearest input image)"
H3_ASPECT_RATIOS: dict[str, tuple[int, int]] = {
    "1:1 (Square)": (1, 1),
    "2:3 (Portrait Photo)": (2, 3),
    "3:2 (Photo)": (3, 2),
    "3:4 (Portrait Standard)": (3, 4),
    "4:3 (Standard)": (4, 3),
    "9:16 (Portrait Widescreen)": (9, 16),
    "16:9 (Widescreen)": (16, 9),
    "21:9 (Ultrawide)": (21, 9),
}


def _safe_int(value: Any, default: int, minimum: int | None = None) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError):
        result = default
    return max(minimum, result) if minimum is not None else result


def _input_image_dimensions(image: Any) -> tuple[int, int]:
    """Read a ComfyUI IMAGE tensor's width and height without importing torch."""

    shape = getattr(image, "shape", None)
    if shape is None or len(shape) < 3:
        raise ValueError("image must be a ComfyUI IMAGE tensor shaped [B, H, W, C]")
    try:
        height = int(shape[-3])
        width = int(shape[-2])
    except (TypeError, ValueError, IndexError) as exc:
        raise ValueError("Could not read width and height from the IMAGE input") from exc
    if width <= 0 or height <= 0:
        raise ValueError("IMAGE input width and height must be positive")
    return width, height


def _nearest_h3_aspect_ratio(width: int, height: int) -> str:
    input_ratio = width / height
    # Log-space distance treats a portrait and its landscape inverse fairly.
    return min(
        H3_ASPECT_RATIOS,
        key=lambda name: abs(math.log(input_ratio / (H3_ASPECT_RATIOS[name][0] / H3_ASPECT_RATIOS[name][1]))),
    )


def _split_inline_thinking(text: Any) -> tuple[str, str]:
    """Separate common inline reasoning formats from a model's final text.

    llama-server often returns ``reasoning_content`` separately. Some templates
    instead place an XML-like thinking block, or a labelled reasoning section,
    inside normal content. Preserve both forms without treating reasoning as the
    chat answer or writing it back into the conversation history.
    """

    raw = str(text or "").strip()
    blocks = [match.group(1).strip() for match in _THINKING_BLOCK.finditer(raw) if match.group(1).strip()]
    if blocks:
        answer = _THINKING_BLOCK.sub("", raw).strip()
        return answer, "\n\n".join(blocks)

    unclosed = _UNCLOSED_THINKING_BLOCK.match(raw)
    if unclosed:
        return "", unclosed.group(1).strip()

    heading = _THINKING_HEADING.match(raw)
    if not heading:
        return raw, ""
    labelled = raw[heading.end() :].strip()
    final_heading = _FINAL_HEADING.search(labelled)
    if final_heading:
        return labelled[final_heading.end() :].strip(), labelled[: final_heading.start()].strip()
    # There is no final-answer marker. This commonly happens when generation
    # stops while a reasoning model is still thinking, so do not mislabel it as
    # an answer.
    return "", labelled


def _chat_reply_and_thinking(backend: Any, messages: list[dict[str, Any]], **settings: Any) -> tuple[str, str]:
    """Call rich backends without breaking simple third-party backend objects."""

    structured = getattr(backend, "chat_response", None)
    if callable(structured):
        completion = structured(messages, **settings)
        answer, inline_thinking = _split_inline_thinking(getattr(completion, "content", ""))
        reasoning = str(getattr(completion, "reasoning", "") or "").strip()
        thinking = "\n\n".join(part for part in (reasoning, inline_thinking) if part)
        return answer, thinking
    return _split_inline_thinking(backend.chat(messages, **settings))


def _auto_unload_backend(backend: Any, enabled: Any) -> bool:
    """Release only Workbench-owned model resources after a completed request."""

    if not bool(enabled):
        return False
    if isinstance(backend, EmbeddedBackend):
        released = EMBEDDED_MODELS.release(backend._active_backend or backend)
        print(f"[Llama Workbench] auto_unload embedded model: {'released' if released else 'not owned or already released'}.")
        return released
    # Do not require isinstance(ServerBackend) here. During a custom-node
    # reload, a live backend can originate from the previous module object;
    # its explicit ownership marker remains the reliable lifecycle contract.
    if bool(getattr(backend, "owned_by_workbench", False)):
        pid = OWNED_SERVER.status().get("pid")
        stopped = OWNED_SERVER.stop()
        print(f"[Llama Workbench] auto_unload owned llama-server (pid={pid}): {'stopped' if stopped else 'no owned process to stop'}.")
        return stopped
    # An explicit Connection URL never proves process ownership, so it is
    # intentionally left running even when auto_unload is selected.
    print("[Llama Workbench] auto_unload skipped: the server is attached externally and is not safe to stop.")
    return False


def _release_comfy_model_cache() -> bool:
    """Release ComfyUI-managed GPU models before loading a large llama-server.

    This mirrors the model-unload half of ComfyUI's "Free Model and Node
    Cache" action. It deliberately uses ComfyUI's model-management API rather
    than deleting model objects retained by unrelated custom nodes.
    """

    try:
        import comfy.model_management as model_management
    except ImportError:
        print("[Llama Workbench] ComfyUI model management is unavailable; skipping pre-start cache release.")
        return False
    try:
        model_management.unload_all_models()
        gc.collect()
        model_management.soft_empty_cache()
    except Exception as exc:
        print(f"[Llama Workbench] could not fully release ComfyUI model cache before server start: {type(exc).__name__}: {exc}")
        return False
    print("[Llama Workbench] released ComfyUI-managed model cache before starting llama-server.")
    return True


def _schedule_comfy_cache_release_after_run(enabled: Any) -> bool:
    """Request ComfyUI's full model *and executor-cache* cleanup after this job.

    Calling ``unload_all_models`` inside Start Server clears Comfy-managed
    models, but it cannot reset the active PromptExecutor: that reset is only
    safe after the current graph finishes. This sets the same queue flags as
    ComfyUI's ``POST /free`` action. The worker consumes them after downstream
    H3/video nodes finish, readying the next workflow for llama.cpp.
    """

    if not bool(enabled):
        return False
    try:
        from server import PromptServer

        prompt_queue = getattr(getattr(PromptServer, "instance", None), "prompt_queue", None)
        if prompt_queue is None or not hasattr(prompt_queue, "set_flag"):
            raise RuntimeError("ComfyUI prompt queue is unavailable")
        prompt_queue.set_flag("unload_models", True)
        prompt_queue.set_flag("free_memory", True)
    except Exception as exc:
        print(f"[Llama Workbench] could not schedule post-run ComfyUI cache release: {type(exc).__name__}: {exc}")
        return False
    print("[Llama Workbench] scheduled full ComfyUI model and node-cache release after this workflow.")
    return True


def _parse_history(raw: str) -> list[dict[str, str]]:
    try:
        decoded = json.loads(raw or "[]")
    except json.JSONDecodeError:
        decoded = []
    if not isinstance(decoded, list):
        return []
    history: list[dict[str, str]] = []
    for item in decoded[-200:]:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "")
        if role not in {"user", "assistant"}:
            continue
        history.append({"role": role, "content": str(item.get("content") or "")})
    return history


def _chat_settings(value: dict[str, Any] | None) -> dict[str, Any]:
    defaults = {
        "system_prompt": "You are a helpful local AI assistant.",
        "max_history_messages": 24,
        "max_tokens": -1,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 40,
        "repeat_penalty": 1.0,
        "seed": -1,
        "use_cache": True,
        "max_image_edge": 0,
    }
    if isinstance(value, dict):
        defaults.update({key: item for key, item in value.items() if key in defaults})
    defaults["max_history_messages"] = _safe_int(defaults["max_history_messages"], 24, 0)
    defaults["max_tokens"] = _safe_int(defaults["max_tokens"], -1, -1)
    defaults["seed"] = _safe_int(defaults["seed"], -1, -1)
    defaults["use_cache"] = bool(defaults["use_cache"])
    defaults["max_image_edge"] = _safe_int(defaults["max_image_edge"], 0, 0)
    return defaults


def _estimate_text_tokens(value: Any) -> int:
    """Fast, model-agnostic token estimate used only for the Chat UI meter."""

    text = str(value or "")
    if not text:
        return 0
    cjk_characters = len(re.findall(r"[\u3400-\u9fff\uf900-\ufaff]", text))
    other_characters = max(0, len(text) - cjk_characters)
    # CJK text is usually close to one token per character; prose in Latin
    # scripts averages roughly four characters per token. Round upward so the
    # display is cautious rather than deceptively optimistic.
    return cjk_characters + math.ceil(other_characters / 4)


def _context_state(backend: Any, system_prompt: str, history: list[dict[str, str]], image_count: int) -> dict[str, Any]:
    # Include a small per-message allowance for chat-template role markers.
    estimated_tokens = _estimate_text_tokens(system_prompt) + sum(
        _estimate_text_tokens(item.get("content")) + 4 for item in history
    )
    context_size = max(0, _safe_int(getattr(backend, "context_size", 0), 0, 0))
    remaining = max(0, context_size - estimated_tokens) if context_size else None
    return {
        "estimated_tokens": estimated_tokens,
        "context_size": context_size,
        "remaining_tokens": remaining,
        "usage_percent": round(estimated_tokens / context_size * 100, 1) if context_size else None,
        "image_count": max(0, int(image_count)),
        "method": "approximate_text",
    }


def _skill_payload(selected: str) -> dict[str, Any]:
    skills = discover_skills()
    return {
        "selected": "" if selected in {"Auto", "None"} else selected,
        "auto": selected == "Auto",
        "skills": [
            {"id": item.id, "name": item.name, "description": item.description, "references": list(item.references)}
            for item in skills
        ],
    }


def _resolve_skill(payload: Any, user_message: str, state: dict[str, Any]) -> Skill | None:
    if not isinstance(payload, dict):
        return None
    selected = str(payload.get("selected") or state.get("skill") or "")
    if selected:
        return get_skill(selected)
    if not bool(payload.get("auto")):
        return None
    return choose_skill(discover_skills(), user_message)


def _apply_requested_references(skill: Skill, state: dict[str, Any], model_state: dict[str, Any]) -> bool:
    requested = model_state.get("load_references", [])
    if not isinstance(requested, list):
        return False
    existing = list(state.get("loaded_references", []))
    additions = [item for item in requested if isinstance(item, str) and item in skill.references and item not in existing]
    if not additions:
        return False
    state["loaded_references"] = existing + additions
    return True


class LlamaWorkbenchConnection:
    CATEGORY = "Llama Workbench / Backend"
    RETURN_TYPES = (BACKEND_TYPE, "STRING")
    RETURN_NAMES = ("backend", "connection")
    FUNCTION = "connect"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "server_url": ("STRING", {"default": "http://127.0.0.1:8080"}),
                "model_name": ("STRING", {"default": "", "placeholder": "optional model ID for a router"}),
                "api_key_env": ("STRING", {"default": "", "placeholder": "optional environment variable"}),
                "timeout_seconds": ("FLOAT", {"default": 120.0, "min": 1.0, "max": 3600.0, "step": 1.0, "tooltip": "Maximum wait for one remote response. With max_tokens=-1 or a reasoning model, use 600 seconds or more if 120 seconds is insufficient."}),
                "context_size": ("INT", {"default": 0, "min": 0, "max": 1048576, "step": 256, "tooltip": "0 means unknown. Set this when attaching an external server to show the Chat context meter."}),
                "use_environment_proxy": ("BOOLEAN", {"default": False, "tooltip": "Default off: connect directly and ignore HTTP_PROXY/HTTPS_PROXY. Enable only when this server intentionally requires the ComfyUI process's environment proxy."}),
            }
        }

    def connect(
        self,
        server_url: str,
        model_name: str,
        api_key_env: str,
        timeout_seconds: float,
        context_size: int = 0,
        use_environment_proxy: bool = False,
    ):
        backend = ServerBackend(
            server_url,
            api_key_env.strip(),
            float(timeout_seconds),
            "attached llama-server",
            model_name.strip(),
            context_size=max(0, int(context_size or 0)),
            use_environment_proxy=bool(use_environment_proxy),
        )
        return backend, backend_descriptor(backend)


class LlamaWorkbenchStartServer:
    CATEGORY = "Llama Workbench / Backend"
    RETURN_TYPES = (BACKEND_TYPE, "STRING", "BOOLEAN")
    RETURN_NAMES = ("backend", "server_url", "started")
    FUNCTION = "start"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": ("STRING", {"default": "", "placeholder": "absolute path to a GGUF model"}),
                "binary_path": ("STRING", {"default": "", "placeholder": "llama-server / custom compatible binary"}),
                "host": ("STRING", {"default": "127.0.0.1"}),
                "port": ("INT", {"default": 8080, "min": 1, "max": 65535}),
                "context_size": ("INT", {"default": 8192, "min": 512, "max": 1048576, "step": 256}),
                "gpu_layers": ("INT", {"default": -1, "min": -1, "max": 9999}),
                "extra_args": ("STRING", {"default": "", "multiline": True, "placeholder": "for example: --flash-attn on --cache-type-k q8_0"}),
            },
            "optional": {
                "mmproj_path": ("STRING", {"default": "", "placeholder": "optional multimodal projector"}),
                "wait_seconds": ("FLOAT", {"default": 600.0, "min": 1.0, "max": 1800.0, "step": 1.0, "tooltip": "Maximum time to wait for llama-server readiness. Large 35B VLMs may need 600 seconds or more on a cold load."}),
                "timeout_seconds": ("FLOAT", {"default": 120.0, "min": 1.0, "max": 3600.0, "step": 1.0, "tooltip": "Maximum time to wait for one Chat or Prompt response from this server. Increase to 600 seconds or more for long reasoning or unlimited-token responses."}),
                "release_comfy_models": ("BOOLEAN", {"default": True, "tooltip": "Before starting llama-server, unload ComfyUI-managed GPU models and empty PyTorch cache. Keep enabled when alternating with MiniMax H3 on one GPU."}),
                "cleanup_previous_server": ("BOOLEAN", {"default": True, "tooltip": "Before starting, stop a leftover llama-server using this binary and port. This clears a timed-out prior Workbench launch. For a server managed outside Workbench, use Llama Workbench Connection instead."}),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Inputs are already part of ComfyUI's cache key. A cached backend
        # restarts itself lazily if a later uncached request needs it.
        return False

    def start(
        self,
        model_path,
        binary_path,
        host,
        port,
        context_size,
        gpu_layers,
        extra_args,
        mmproj_path="",
        wait_seconds=600.0,
        timeout_seconds=120.0,
        release_comfy_models=True,
        cleanup_previous_server=True,
    ):
        config = ServerLaunchConfig(
            binary_path=str(binary_path),
            model_path=str(model_path),
            host=str(host),
            port=int(port),
            context_size=int(context_size),
            gpu_layers=int(gpu_layers),
            mmproj_path=str(mmproj_path),
            extra_args=str(extra_args),
        )
        if bool(release_comfy_models):
            _release_comfy_model_cache()
        backend = OWNED_SERVER.start(
            config,
            float(wait_seconds),
            cleanup_previous_server=bool(cleanup_previous_server),
            timeout_seconds=float(timeout_seconds),
        )

        def ensure_owned_server() -> None:
            if OWNED_SERVER.is_running_for(config):
                return
            if bool(release_comfy_models):
                _release_comfy_model_cache()
            OWNED_SERVER.start(
                config,
                float(wait_seconds),
                cleanup_previous_server=bool(cleanup_previous_server),
                timeout_seconds=float(timeout_seconds),
            )

        backend.ensure_available = ensure_owned_server
        return backend, backend.server_url, True


class LlamaWorkbenchStopServer:
    CATEGORY = "Llama Workbench / Backend"
    RETURN_TYPES = ("BOOLEAN", "STRING")
    RETURN_NAMES = ("stopped", "status")
    FUNCTION = "stop"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"stop": ("BOOLEAN", {"default": True})}}

    def stop(self, stop=True):
        stopped = OWNED_SERVER.stop() if stop else False
        return stopped, json.dumps(OWNED_SERVER.status(), ensure_ascii=False)


class LlamaWorkbenchServerStatus:
    CATEGORY = "Llama Workbench / Backend"
    RETURN_TYPES = ("STRING", "BOOLEAN")
    RETURN_NAMES = ("status_json", "running")
    FUNCTION = "get_status"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"refresh": ("BOOLEAN", {"default": True})}}

    def get_status(self, refresh=True):
        status = OWNED_SERVER.status()
        return json.dumps(status, ensure_ascii=False, indent=2), bool(status["running"])


class LlamaWorkbenchH3AutoResolutionSelector:
    """Match an IMAGE to the nearest MiniMax H3 resolution-selector ratio."""

    CATEGORY = "Llama Workbench / Utilities"
    RETURN_TYPES = ("INT", "INT", "STRING")
    RETURN_NAMES = ("width", "height", "selected_aspect_ratio")
    FUNCTION = "select"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "The input image used to choose the nearest aspect ratio in Auto mode."}),
                "aspect_ratio": (
                    [H3_AUTO_ASPECT_RATIO, *H3_ASPECT_RATIOS],
                    {
                        "default": H3_AUTO_ASPECT_RATIO,
                        "tooltip": "Auto selects the closest MiniMax H3 Resolution Selector ratio from the input image."
                    },
                ),
                "megapixels": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.1,
                        "max": 16.0,
                        "step": 0.1,
                        "tooltip": "Target total megapixels. 1.0 MP is about 1024×1024 for 1:1.",
                    },
                ),
                "multiple": (
                    "INT",
                    {
                        "default": 8,
                        "min": 8,
                        "max": 128,
                        "step": 4,
                        "tooltip": "Round each output dimension to the nearest multiple, matching the H3 selector calculation.",
                    },
                ),
            }
        }

    def select(self, image, aspect_ratio=H3_AUTO_ASPECT_RATIO, megapixels=1.0, multiple=8):
        source_width, source_height = _input_image_dimensions(image)
        selected = str(aspect_ratio)
        if selected == H3_AUTO_ASPECT_RATIO or selected not in H3_ASPECT_RATIOS:
            selected = _nearest_h3_aspect_ratio(source_width, source_height)
        width_ratio, height_ratio = H3_ASPECT_RATIOS[selected]
        target_pixels = max(0.1, float(megapixels)) * 1024 * 1024
        rounding_multiple = _safe_int(multiple, 8, 8)
        scale = math.sqrt(target_pixels / (width_ratio * height_ratio))
        width = round(width_ratio * scale / rounding_multiple) * rounding_multiple
        height = round(height_ratio * scale / rounding_multiple) * rounding_multiple
        return width, height, selected


class LlamaWorkbenchEmbeddedModel:
    CATEGORY = "Llama Workbench / Embedded"
    RETURN_TYPES = (BACKEND_TYPE, "STRING")
    RETURN_NAMES = ("backend", "diagnostics")
    FUNCTION = "load"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": ("STRING", {"default": "", "placeholder": "absolute GGUF model path"}),
                "family": (["auto", "qwen3-vl", "qwen3.5-vl", "qwen3.6-vl", "qwen3.8-vl", "gemma4"],),
                "context_size": ("INT", {"default": 8192, "min": 512, "max": 327680, "step": 256}),
                "gpu_layers": ("INT", {"default": -1, "min": -1, "max": 9999}),
                "cache_type_k": (["f16", "q8_0"],),
                "cache_type_v": (["f16", "q8_0"],),
                "cpu_moe": ("BOOLEAN", {"default": False}),
                "n_cpu_moe": ("INT", {"default": 0, "min": 0, "max": 256}),
            },
            "optional": {"mmproj_path": ("STRING", {"default": "", "placeholder": "required for multimodal models"})},
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Inputs are already part of ComfyUI's cache key. A cached backend
        # recreates the model only if a later uncached request needs it.
        return False

    def load(self, model_path, family, context_size, gpu_layers, cache_type_k, cache_type_v, cpu_moe, n_cpu_moe, mmproj_path=""):
        configuration = {
            "model_path": str(model_path),
            "family": str(family),
            "mmproj_path": str(mmproj_path),
            "context_size": int(context_size),
            "gpu_layers": int(gpu_layers),
            "cache_type_k": str(cache_type_k),
            "cache_type_v": str(cache_type_v),
            "cpu_moe": bool(cpu_moe),
            "n_cpu_moe": int(n_cpu_moe),
        }
        backend = create_embedded_backend(**configuration)
        backend.ensure_available = lambda: create_embedded_backend(**configuration)
        backend._active_backend = backend
        return backend, backend_descriptor(backend)


class LlamaWorkbenchReleaseEmbedded:
    CATEGORY = "Llama Workbench / Embedded"
    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("released",)
    FUNCTION = "release"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"backend": (BACKEND_TYPE,)}}

    def release(self, backend):
        return (EMBEDDED_MODELS.release(backend),)


class LlamaWorkbenchPrompt:
    CATEGORY = "Llama Workbench / Generation"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("response", "backend_info")
    FUNCTION = "generate"

    @classmethod
    def INPUT_TYPES(cls):
        images = {
            "image": ("IMAGE", {"tooltip": "Legacy image socket; it becomes image1 after connection."}),
            **{
                f"image{index}": ("IMAGE", {"tooltip": f"Image {index}; dynamic image sockets support up to 10 images."})
                for index in range(1, DYNAMIC_IMAGE_LIMIT + 1)
            },
        }
        return {
            "required": {
                "backend": (BACKEND_TYPE,),
                "prompt": ("STRING", {"default": "Describe the image.", "multiline": True}),
                "system_prompt": ("STRING", {"default": "", "multiline": True}),
                "max_tokens": (
                    "INT",
                    {
                        "default": -1,
                        "min": -1,
                        "max": 65536,
                        "tooltip": "-1 means unlimited output tokens; generation still ends at EOS or the context limit.",
                    },
                ),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.01}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}),
                "top_k": ("INT", {"default": 40, "min": 0, "max": 200}),
                "thinking": (["off", "auto", "on"], {"default": "off", "tooltip": "Use off for direct image-to-prompt output."}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0x7FFFFFFF, "control_after_generate": True}),
                "max_images": ("INT", {"default": 10, "min": 1, "max": 10, "tooltip": "Maximum total images sent from dynamic sockets and IMAGE batches."}),
                "max_image_edge": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 64, "tooltip": "0 preserves input resolution; a positive limit downsizes images before sending."}),
                "auto_unload": ("BOOLEAN", {"default": False, "tooltip": "After generation, unload an embedded model or stop a Start Server-owned llama-server. Attached Connection servers are never stopped."}),
            },
            "optional": images,
        }

    def generate(
        self,
        backend,
        prompt,
        system_prompt,
        max_tokens,
        temperature,
        top_p,
        top_k,
        thinking="off",
        seed=-1,
        max_images=10,
        max_image_edge=0,
        auto_unload=False,
        image=None,
        image1=None,
        image2=None,
        image3=None,
        image4=None,
        image5=None,
        image6=None,
        image7=None,
        image8=None,
        image9=None,
        image10=None,
    ):
        image_limit = min(DYNAMIC_IMAGE_LIMIT, _safe_int(max_images, DYNAMIC_IMAGE_LIMIT, 1))
        image_edge = _safe_int(max_image_edge, 0, 0)
        images: list[str] = []
        for supplied in (
            image,
            image1,
            image2,
            image3,
            image4,
            image5,
            image6,
            image7,
            image8,
            image9,
            image10,
        ):
            remaining = image_limit - len(images)
            if supplied is None or remaining <= 0:
                continue
            images.extend(image_tensor_to_data_urls(supplied, max_images=remaining, max_edge=image_edge))
        messages: list[dict[str, Any]] = []
        if str(system_prompt).strip():
            messages.append({"role": "system", "content": str(system_prompt)})
        messages.append(make_text_message(str(prompt), images))
        thinking_mode = str(thinking or "off").strip().lower()
        enable_thinking = None if thinking_mode == "auto" else thinking_mode == "on"
        try:
            response = backend.chat(
                messages,
                max_tokens=int(max_tokens),
                temperature=float(temperature),
                top_p=float(top_p),
                top_k=int(top_k),
                enable_thinking=enable_thinking,
                seed=_safe_int(seed, -1, -1),
            )
        finally:
            _auto_unload_backend(backend, auto_unload)
        return response, backend_descriptor(backend)


class LlamaWorkbenchPromptEnhancer:
    """Qwen-Image-2.1 PE with strict, separately wired output fields."""

    CATEGORY = "Llama Workbench / Generation"
    OUTPUT_NODE = True
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = (
        "rewritten_prompt",
        "wh_ratio",
        "ratio_follow",
        "result_json",
        "thinking",
        "backend_info",
    )
    FUNCTION = "enhance"

    @classmethod
    def INPUT_TYPES(cls):
        images = {
            "image": ("IMAGE", {"tooltip": "First PE-I2I source image; connecting it reveals image2."}),
            **{
                f"image{index}": (
                    "IMAGE",
                    {"tooltip": f"PE-I2I source image {index}; inputs preserve image1..image10 order."},
                )
                for index in range(1, MAX_PROMPT_REWRITE_IMAGES + 1)
            },
        }
        return {
            "required": {
                "backend": (BACKEND_TYPE,),
                "prompt": ("STRING", {"default": "", "multiline": True, "placeholder": "Short image prompt to enhance"}),
                "task": (
                    ["t2i", "edit"],
                    {
                        "default": "t2i",
                        "tooltip": "t2i targets PE-T2I. edit is the prepared PE-I2I path and requires a matching model plus mmproj.",
                    },
                ),
                "system_prompt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "placeholder": "Paste the matching Qwen PE system prompt, or use system_prompt_path",
                    },
                ),
                "system_prompt_path": (
                    "STRING",
                    {
                        "default": "",
                        "placeholder": "local system_prompt.txt or its containing directory",
                    },
                ),
                "seed": ("INT", {"default": 42, "min": -1, "max": 0x7FFFFFFF, "control_after_generate": True}),
                "max_images": (
                    "INT",
                    {
                        "default": 10,
                        "min": 1,
                        "max": 10,
                        "tooltip": "Prepared for Qwen-Image-2.1 PE-I2I; t2i rejects connected images.",
                    },
                ),
                "max_image_edge": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 8192,
                        "step": 64,
                        "tooltip": "0 preserves input resolution; a positive limit downsizes PE-I2I images before sending.",
                    },
                ),
                "auto_unload": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Release a Workbench-owned backend after the enhancement request and its possible format retry.",
                    },
                ),
            },
            "optional": images,
        }

    def enhance(
        self,
        backend,
        prompt,
        task,
        system_prompt,
        system_prompt_path,
        seed=42,
        max_images=10,
        max_image_edge=0,
        auto_unload=False,
        image=None,
        image1=None,
        image2=None,
        image3=None,
        image4=None,
        image5=None,
        image6=None,
        image7=None,
        image8=None,
        image9=None,
        image10=None,
    ):
        image_limit = min(MAX_PROMPT_REWRITE_IMAGES, _safe_int(max_images, 10, 1))
        image_edge = _safe_int(max_image_edge, 0, 0)
        image_urls: list[str] = []
        for supplied in (
            image,
            image1,
            image2,
            image3,
            image4,
            image5,
            image6,
            image7,
            image8,
            image9,
            image10,
        ):
            remaining = image_limit - len(image_urls)
            if supplied is None or remaining <= 0:
                continue
            image_urls.extend(image_tensor_to_data_urls(supplied, max_images=remaining, max_edge=image_edge))
        try:
            result = rewrite_prompt(
                backend,
                str(prompt),
                task=str(task),
                system_prompt=str(system_prompt),
                system_prompt_path=str(system_prompt_path),
                image_data_urls=image_urls,
                seed=_safe_int(seed, 42, -1),
            )
        finally:
            _auto_unload_backend(backend, auto_unload)
        result_json = result.as_json()
        ui = {
            "result_json": [result_json],
            "rewritten_prompt": [result.rewritten_prompt],
            "retried": [result.retried],
        }
        outputs = (
            result.rewritten_prompt,
            result.wh_ratio,
            result.ratio_follow,
            result_json,
            result.thinking,
            backend_descriptor(backend),
        )
        return {"ui": ui, "result": outputs}


class LlamaWorkbenchQwenImage21PEResolution:
    """Turn Prompt Enhancer's aspect-ratio output into latent dimensions."""

    CATEGORY = "Llama Workbench / Generation"
    RETURN_TYPES = ("INT", "INT", "STRING")
    RETURN_NAMES = ("width", "height", "normalized_ratio")
    FUNCTION = "select"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "wh_ratio": (
                    "STRING",
                    {
                        "forceInput": True,
                        "tooltip": "Connect Prompt Enhancer's wh_ratio output.",
                    },
                ),
                "megapixels": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.1,
                        "max": 16.0,
                        "step": 0.1,
                        "tooltip": "Target pixel budget; 1.0 MP is about 1024×1024 at 1:1.",
                    },
                ),
                "multiple": (
                    "INT",
                    {
                        "default": 8,
                        "min": 8,
                        "max": 128,
                        "step": 8,
                        "tooltip": "Round width and height to this generation-compatible multiple.",
                    },
                ),
                "aspect_ratio_override": (
                    [PROMPT_REWRITE_AUTO_ASPECT_RATIO, *QWEN_IMAGE_21_ASPECT_RATIOS],
                    {
                        "default": PROMPT_REWRITE_AUTO_ASPECT_RATIO,
                        "tooltip": "Auto uses the connected PE wh_ratio; a concrete ratio overrides it.",
                    },
                ),
            }
        }

    def select(
        self,
        wh_ratio: str,
        megapixels: float = 1.0,
        multiple: int = 8,
        aspect_ratio_override: str = PROMPT_REWRITE_AUTO_ASPECT_RATIO,
    ):
        return prompt_rewrite_dimensions(
            wh_ratio,
            megapixels=megapixels,
            multiple=multiple,
            aspect_ratio_override=aspect_ratio_override,
        )


class LlamaWorkbenchChatSettings:
    CATEGORY = "Llama Workbench / Chat"
    RETURN_TYPES = (SETTINGS_TYPE,)
    RETURN_NAMES = ("settings",)
    FUNCTION = "build"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "system_prompt": ("STRING", {"default": "You are a helpful local AI assistant.", "multiline": True}),
                "max_history_messages": ("INT", {"default": 24, "min": 0, "max": 200}),
                "max_tokens": ("INT", {"default": -1, "min": -1, "max": 65536}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.01}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}),
                "top_k": ("INT", {"default": 40, "min": 0, "max": 200}),
                "repeat_penalty": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 3.0, "step": 0.01}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0x7FFFFFFF, "control_after_generate": True}),
                "use_cache": ("BOOLEAN", {"default": True, "tooltip": "Default on: reuse Chat output when the complete request is unchanged. This is independent of seed; turn off once to force a new backend request and refresh the cached result."}),
                "max_image_edge": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 64, "tooltip": "0 preserves input resolution; a positive limit downsizes images before sending."}),
            }
        }

    def build(self, **kwargs):
        result = dict(kwargs)
        result["seed"] = _safe_int(result.get("seed"), -1, -1)
        return (result,)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Every widget value is already part of ComfyUI's cache key.
        return False


class LlamaWorkbenchSkillLoader:
    CATEGORY = "Llama Workbench / Skills"
    RETURN_TYPES = (SKILL_TYPE, "STRING")
    RETURN_NAMES = ("skill", "catalog_json")
    FUNCTION = "load"

    @classmethod
    def INPUT_TYPES(cls):
        choices = ["None", "Auto"] + [skill.id for skill in discover_skills()]
        return {"required": {"skill": (choices,)}}

    def load(self, skill):
        payload = _skill_payload(str(skill))
        return payload, json.dumps(payload["skills"], ensure_ascii=False)


class LlamaWorkbenchChat:
    """Stateful graph chat with a native ComfyUI message field."""

    CATEGORY = "Llama Workbench / Chat"
    OUTPUT_NODE = True
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("assistant_message", "history_json", "flow_state_json", "thinking")
    FUNCTION = "chat"

    @classmethod
    def INPUT_TYPES(cls):
        images = {
            "image": ("IMAGE", {"tooltip": "First image; connecting it reveals image2. Dynamic image sockets support up to 10 images."}),
            **{
                f"image{index}": ("IMAGE", {"tooltip": f"Image {index}; dynamic image sockets support up to 10 images."})
                for index in range(1, DYNAMIC_IMAGE_LIMIT + 1)
            },
        }
        return {
            "required": {
                "backend": (BACKEND_TYPE,),
                "lwb_user_message": (
                    "STRING",
                    {
                        "default": "",
                        # The Vue multiline widget reserves a large DOM area
                        # whose reported canvas height is only one row. That
                        # makes a canvas conversation panel overlap controls.
                        # Keep manual chat compact; long/multiline text remains
                        # fully supported through the text input socket.
                        "placeholder": "输入消息后点击 Queue Prompt",
                    },
                ),
                # These are compact persisted state values. They are hidden by
                # the canvas frontend; keeping them single-line avoids creating
                # DOM textarea widgets solely for invisible JSON state.
                "lwb_history_json": ("STRING", {"default": "[]", "hidden": True}),
                "lwb_flow_state_json": ("STRING", {"default": "{}", "hidden": True}),
                "lwb_request_id": ("STRING", {"default": "", "hidden": True}),
                "max_tokens": ("INT", {"default": -1, "min": -1, "max": 65536, "tooltip": "-1 means unlimited output tokens."}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0x7FFFFFFF, "control_after_generate": True}),
                "use_cache": ("BOOLEAN", {"default": True, "tooltip": "Default on: reuse the result for an unchanged complete request. This is independent of seed; turn off to force a fresh model request."}),
                "thinking": (["off", "auto", "on"], {"default": "off", "tooltip": "off suppresses supported reasoning-model thinking output; auto leaves the model template unchanged."}),
                "max_images": ("INT", {"default": 10, "min": 1, "max": 10, "tooltip": "Maximum total images sent from dynamic sockets and IMAGE batches."}),
                "max_image_edge": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 64, "tooltip": "0 preserves input resolution; a positive limit downsizes images before sending."}),
                "auto_unload": ("BOOLEAN", {"default": False, "tooltip": "After generation, unload an embedded model or stop a Start Server-owned llama-server. Attached Connection servers are never stopped."}),
                "clear_context_before_run": ("BOOLEAN", {"default": True, "tooltip": "Default on: discard this node's saved conversation and skill state before every execution. Turn off for a continuing multi-turn conversation."}),
                "release_comfy_cache_after_run": ("BOOLEAN", {"default": True, "tooltip": "When use_cache is off, free ComfyUI's model and node cache after the complete workflow. It is skipped while use_cache is on so an unchanged Chat request can reuse its result."}),
                "release_owned_server_after_run": ("BOOLEAN", {"default": True, "tooltip": "Default on: stop the llama-server started by Llama Workbench immediately after Chat responds, before downstream H3/video nodes allocate VRAM. Turn off only when intentionally keeping the owned server resident."}),
            },
            "optional": {
                "settings": (SETTINGS_TYPE,),
                "skill": (SKILL_TYPE,),
                **images,
            },
        }

    @classmethod
    def IS_CHANGED(cls, use_cache=True, settings=None, **kwargs):
        enabled = settings.get("use_cache", use_cache) if isinstance(settings, dict) else use_cache
        # Returning NaN deliberately bypasses ComfyUI's result cache. The
        # explicit switch is separate from the model's sampling seed.
        return False if bool(enabled) else float("nan")

    def chat(
        self,
        backend,
        lwb_user_message,
        lwb_history_json,
        lwb_flow_state_json,
        lwb_request_id,
        max_tokens=-1,
        seed=-1,
        use_cache=True,
        thinking="off",
        max_images=10,
        max_image_edge=0,
        auto_unload=False,
        clear_context_before_run=True,
        release_comfy_cache_after_run=True,
        release_owned_server_after_run=True,
        settings=None,
        skill=None,
        image=None,
        image1=None,
        image2=None,
        image3=None,
        image4=None,
        image5=None,
        image6=None,
        image7=None,
        image8=None,
        image9=None,
        image10=None,
    ):
        message = str(lwb_user_message or "").strip()
        history = _parse_history(lwb_history_json)
        flow_state = parse_flow_state(lwb_flow_state_json)
        if bool(clear_context_before_run):
            # A queued workflow should be independent by default. Keep the
            # original graph-persisted behaviour available through the toggle
            # for interactive multi-turn chat.
            history = []
            flow_state = empty_flow_state()
        if not message:
            previous = next((item["content"] for item in reversed(history) if item["role"] == "assistant"), "")
            return self._result(previous, "", history, flow_state, False, [], _context_state(backend, "", history, 0))

        options: list[str] = []
        effective = _chat_settings(settings)
        # A connected Settings node supplies generation values it explicitly
        # contains. Older partial Settings payloads have no seed, so retain
        # Chat's seed in that compatibility case.
        if not isinstance(settings, dict):
            effective["max_tokens"] = _safe_int(max_tokens, -1, -1)
            effective["seed"] = _safe_int(seed, -1, -1)
            effective["use_cache"] = bool(use_cache)
        else:
            chat_seed = _safe_int(seed, -1, -1)
            if "seed" not in settings and chat_seed >= 0:
                effective["seed"] = chat_seed
            if "use_cache" not in settings:
                effective["use_cache"] = bool(use_cache)
        thinking_mode = str(thinking or "off").strip().lower()
        enable_thinking = None if thinking_mode == "auto" else thinking_mode == "on"
        image_limit = min(DYNAMIC_IMAGE_LIMIT, _safe_int(max_images, DYNAMIC_IMAGE_LIMIT, 1))
        image_edge = _safe_int(max_image_edge, 0, 0)
        resolved_skill = _resolve_skill(skill, message, flow_state)
        base_system = str(effective["system_prompt"] or "").strip()
        if resolved_skill is not None:
            if flow_state.get("skill") != resolved_skill.id:
                flow_state = empty_flow_state()
            flow_state["skill"] = resolved_skill.id
            flow_state["skill_name"] = resolved_skill.name
            skill_system = build_skill_instruction(resolved_skill, flow_state)
            system = "\n\n".join(part for part in (base_system, skill_system) if part)
        else:
            system = base_system

        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        if effective["max_history_messages"]:
            messages.extend(history[-effective["max_history_messages"] :])
        image_urls: list[str] = []
        for supplied in (
            image,
            image1,
            image2,
            image3,
            image4,
            image5,
            image6,
            image7,
            image8,
            image9,
            image10,
        ):
            remaining = image_limit - len(image_urls)
            if supplied is None or remaining <= 0:
                continue
            image_urls.extend(image_tensor_to_data_urls(supplied, max_images=remaining, max_edge=image_edge))
        messages.append(make_text_message(message, image_urls))
        generation_settings = {key: effective[key] for key in ("max_tokens", "temperature", "top_p", "top_k", "repeat_penalty", "seed")}
        generation_settings["enable_thinking"] = enable_thinking
        release_owned_backend = bool(auto_unload) or bool(release_owned_server_after_run)
        try:
            reply, thinking_text = _chat_reply_and_thinking(backend, messages, **generation_settings)
            reply, model_state = parse_skill_state(reply)

            # Some reasoning templates exhaust their completion on the reasoning
            # trace and return no `content` at all. Preserve that trace, then make
            # one direct-answer retry with thinking disabled. This prevents a
            # populated Thinking panel and an empty Final answer panel while
            # avoiding an unbounded retry loop.
            if not reply and thinking_text and thinking_mode == "on":
                final_settings = dict(generation_settings)
                final_settings["enable_thinking"] = False
                reply, recovery_thinking = _chat_reply_and_thinking(backend, messages, **final_settings)
                if recovery_thinking:
                    thinking_text = "\n\n".join(part for part in (thinking_text, recovery_thinking) if part)
                reply, model_state = parse_skill_state(reply)

            # A Skill may ask for declared references. Re-run once with those exact files,
            # preventing unbounded context expansion or an execution loop.
            if resolved_skill is not None and _apply_requested_references(resolved_skill, flow_state, model_state):
                system = "\n\n".join(
                    part for part in (base_system, build_skill_instruction(resolved_skill, flow_state)) if part
                )
                messages[0:1] = [{"role": "system", "content": system}] if messages and messages[0].get("role") == "system" else [{"role": "system", "content": system}]
                reply, follow_up_thinking = _chat_reply_and_thinking(backend, messages, **generation_settings)
                if follow_up_thinking:
                    thinking_text = "\n\n".join(part for part in (thinking_text, follow_up_thinking) if part)
                reply, model_state = parse_skill_state(reply)
                if not reply and thinking_text and thinking_mode == "on":
                    final_settings = dict(generation_settings)
                    final_settings["enable_thinking"] = False
                    reply, recovery_thinking = _chat_reply_and_thinking(backend, messages, **final_settings)
                    if recovery_thinking:
                        thinking_text = "\n\n".join(part for part in (thinking_text, recovery_thinking) if part)
                    reply, model_state = parse_skill_state(reply)
            # Skill output contracts are prompt guidance only. The node returns
            # the model's reply verbatim (apart from its optional runtime state
            # tag) and never validates, rewrites, or rejects that reply.
            if resolved_skill is not None:
                flow_state["stage"] = str(model_state.get("stage") or flow_state.get("stage") or "in progress")[:120]
                raw_options = model_state.get("options", [])
                options = [str(item)[:240] for item in raw_options if isinstance(item, str)][:6] if isinstance(raw_options, list) else []
                if bool(model_state.get("final")):
                    flow_state["final"] = reply

            history.extend(({"role": "user", "content": message}, {"role": "assistant", "content": reply}))
            history = history[-max(2, effective["max_history_messages"] * 2 or 200) :]
            return self._result(reply, thinking_text, history, flow_state, True, options, _context_state(backend, system, history, len(image_urls)))
        finally:
            # A 5xx is still a completed attempt from a resource-lifecycle
            # perspective. Do not leave an owned llama-server occupying the
            # GPU/port just because inference itself failed.
            _auto_unload_backend(backend, release_owned_backend)
            # Releasing ComfyUI's node cache would discard this Chat result
            # before the next identical workflow can reuse it. The explicit
            # cache switch therefore takes precedence over the legacy GPU-cache
            # release default.
            _schedule_comfy_cache_release_after_run(
                bool(release_comfy_cache_after_run) and not bool(effective["use_cache"])
            )

    @staticmethod
    def _result(
        reply: str,
        thinking_text: str,
        history: list[dict[str, str]],
        state: dict[str, Any],
        sent: bool,
        options: list[str],
        context_state: dict[str, Any],
    ):
        history_encoded = json.dumps(history, ensure_ascii=False, separators=(",", ":"))
        state_encoded = json.dumps(state, ensure_ascii=False, separators=(",", ":"))
        ui = {
            "history_json": [history_encoded],
            "flow_state_json": [state_encoded],
            "assistant_message": [reply],
            "thinking": [thinking_text],
            "context_state_json": [json.dumps(context_state, ensure_ascii=False, separators=(",", ":"))],
            "skill_options": [options],
            "sent": [sent],
            "updated_at": [int(time.time() * 1000)],
        }
        return {"ui": ui, "result": (reply, history_encoded, state_encoded, thinking_text)}


class LlamaWorkbenchChatDisplay:
    """Canvas-only terminal display for a Chat node's reasoning and final answer."""

    CATEGORY = "Llama Workbench / Chat"
    OUTPUT_NODE = True
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("thinking", "assistant_message")
    FUNCTION = "display"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "thinking": ("STRING", {"forceInput": True, "tooltip": "Connect Chat's thinking output."}),
                "assistant_message": ("STRING", {"forceInput": True, "tooltip": "Connect Chat's assistant_message output."}),
            }
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def display(self, thinking: str, assistant_message: str):
        reasoning = str(thinking or "")
        answer = str(assistant_message or "")
        return {
            "ui": {
                "thinking": [reasoning],
                "assistant_message": [answer],
                "updated_at": [int(time.time() * 1000)],
            },
            "result": (reasoning, answer),
        }


NODE_CLASS_MAPPINGS = {
    "LlamaWorkbench_Connection": LlamaWorkbenchConnection,
    "LlamaWorkbench_StartServer": LlamaWorkbenchStartServer,
    "LlamaWorkbench_StopServer": LlamaWorkbenchStopServer,
    "LlamaWorkbench_ServerStatus": LlamaWorkbenchServerStatus,
    "LlamaWorkbench_H3AutoResolutionSelector": LlamaWorkbenchH3AutoResolutionSelector,
    "LlamaWorkbench_EmbeddedModel": LlamaWorkbenchEmbeddedModel,
    "LlamaWorkbench_ReleaseEmbedded": LlamaWorkbenchReleaseEmbedded,
    "LlamaWorkbench_Prompt": LlamaWorkbenchPrompt,
    "LlamaWorkbench_PromptEnhancer": LlamaWorkbenchPromptEnhancer,
    "LlamaWorkbench_QwenImage21PEResolution": LlamaWorkbenchQwenImage21PEResolution,
    "LlamaWorkbench_ChatSettings": LlamaWorkbenchChatSettings,
    "LlamaWorkbench_SkillLoader": LlamaWorkbenchSkillLoader,
    "LlamaWorkbench_Chat": LlamaWorkbenchChat,
    "LlamaWorkbench_ChatDisplay": LlamaWorkbenchChatDisplay,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LlamaWorkbench_Connection": "Llama Workbench Connection",
    "LlamaWorkbench_StartServer": "Llama Workbench Start Server",
    "LlamaWorkbench_StopServer": "Llama Workbench Stop Owned Server",
    "LlamaWorkbench_ServerStatus": "Llama Workbench Server Status",
    "LlamaWorkbench_H3AutoResolutionSelector": "Llama Workbench H3 Auto Resolution Selector",
    "LlamaWorkbench_EmbeddedModel": "Llama Workbench Embedded VL Model",
    "LlamaWorkbench_ReleaseEmbedded": "Llama Workbench Release Embedded Model",
    "LlamaWorkbench_Prompt": "Llama Workbench Prompt / Image2Prompt",
    "LlamaWorkbench_PromptEnhancer": "Llama Workbench Qwen Image 2.1 Prompt Enhancer",
    "LlamaWorkbench_QwenImage21PEResolution": "Llama Workbench Qwen Image 2.1 PE Resolution",
    "LlamaWorkbench_ChatSettings": "Llama Workbench Chat Settings",
    "LlamaWorkbench_SkillLoader": "Llama Workbench Skill Loader",
    "LlamaWorkbench_Chat": "Llama Workbench Chat",
    "LlamaWorkbench_ChatDisplay": "Llama Workbench Chat Output Display",
}
