"""Optional llama-cpp-python model construction for model-specific local use."""

from __future__ import annotations

import inspect
import json
import threading
from pathlib import Path
from typing import Any

from .backend import EmbeddedBackend


class EmbeddedModelRegistry:
    """A namespaced one-model cache; it never hooks ComfyUI global unload."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._backend: EmbeddedBackend | None = None
        self._fingerprint = ""

    def load(self, **configuration: Any) -> EmbeddedBackend:
        fingerprint = json.dumps(configuration, sort_keys=True, ensure_ascii=False)
        with self._lock:
            if self._backend is not None and self._fingerprint == fingerprint:
                return self._backend
            if self._backend is not None:
                self._backend.close()
            self._backend = _create_embedded_backend(**configuration)
            self._fingerprint = fingerprint
            return self._backend

    def release(self, backend: Any = None) -> bool:
        with self._lock:
            if self._backend is None:
                return False
            if backend is not None and backend is not self._backend:
                return False
            self._backend.close()
            self._backend = None
            self._fingerprint = ""
            return True


EMBEDDED_MODELS = EmbeddedModelRegistry()


def _import_llama() -> tuple[Any, Any]:
    try:
        import llama_cpp
        from llama_cpp import Llama
    except ImportError as exc:
        raise RuntimeError(
            "Embedded mode needs llama-cpp-python. Install it with the same Python that launches ComfyUI."
        ) from exc
    return llama_cpp, Llama


def _handler_instance(family: str, mmproj_path: str) -> Any | None:
    if not mmproj_path.strip():
        return None
    projector = Path(mmproj_path).expanduser()
    if not projector.is_file():
        raise ValueError(f"mmproj_path was not found: {projector}")
    try:
        from llama_cpp import llama_chat_format as formats
    except ImportError as exc:  # pragma: no cover - supplied by llama-cpp-python
        raise RuntimeError("The installed llama-cpp-python does not include chat handlers") from exc

    normalized = family.casefold()
    candidates = {
        "qwen3-vl": ("Qwen3VLChatHandler",),
        "qwen3.5-vl": ("Qwen35ChatHandler",),
        "qwen3.6-vl": ("Qwen35ChatHandler",),
        "qwen3.8-vl": ("Qwen35ChatHandler",),
        "gemma4": ("Gemma4ChatHandler",),
        "auto": ("Qwen35ChatHandler", "Qwen3VLChatHandler", "Gemma4ChatHandler"),
    }
    handlers = candidates.get(normalized, candidates["auto"])
    errors: list[str] = []
    for name in handlers:
        handler_type = getattr(formats, name, None)
        if handler_type is None:
            continue
        for keyword in ("mmproj_path", "clip_model_path"):
            try:
                return handler_type(**{keyword: str(projector.resolve()), "verbose": False})
            except TypeError as exc:
                errors.append(f"{name}/{keyword}: {exc}")
            except Exception as exc:
                raise RuntimeError(f"Could not create {name}: {exc}") from exc
    detail = "; ".join(errors[-2:])
    raise RuntimeError(f"No compatible {family} multimodal chat handler is available. {detail}")


def _create_embedded_backend(
    *,
    model_path: str,
    family: str,
    mmproj_path: str,
    context_size: int,
    gpu_layers: int,
    cache_type_k: str,
    cache_type_v: str,
    cpu_moe: bool,
    n_cpu_moe: int,
) -> EmbeddedBackend:
    """Load a local model, exposing Qwen/Gemma handler and common memory options."""

    model = Path(model_path).expanduser()
    if not model.is_file():
        raise ValueError(f"model_path was not found: {model}")
    llama_cpp, Llama = _import_llama()
    kwargs: dict[str, Any] = {
        "model_path": str(model.resolve()),
        "n_ctx": int(context_size),
        "n_gpu_layers": int(gpu_layers),
        "verbose": False,
    }
    handler = _handler_instance(family, mmproj_path)
    if handler is not None:
        kwargs["chat_handler"] = handler
    signature = inspect.signature(Llama.__init__).parameters
    if cache_type_k == "q8_0" and "type_k" in signature:
        kwargs["type_k"] = getattr(llama_cpp, "GGML_TYPE_Q8_0", 8)
    if cache_type_v == "q8_0" and "type_v" in signature:
        kwargs["type_v"] = getattr(llama_cpp, "GGML_TYPE_Q8_0", 8)
    if family.casefold() in {"qwen3.6-vl", "auto"}:
        if cpu_moe and "cpu_moe" in signature:
            kwargs["cpu_moe"] = True
        elif int(n_cpu_moe) > 0 and "n_cpu_moe" in signature:
            kwargs["n_cpu_moe"] = int(n_cpu_moe)
    try:
        llm = Llama(**kwargs)
    except Exception as exc:
        raise RuntimeError(f"Embedded llama-cpp-python model load failed: {type(exc).__name__}: {exc}") from exc
    return EmbeddedBackend(llm=llm, label=f"embedded {family}")


def create_embedded_backend(**configuration: Any) -> EmbeddedBackend:
    """Load or return the package's explicit embedded-model cache entry."""

    return EMBEDDED_MODELS.load(**configuration)
