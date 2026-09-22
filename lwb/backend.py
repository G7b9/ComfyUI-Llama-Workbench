"""Backend adapters shared by prompt and chat nodes.

The public node socket carries one of these objects as
``LLAMA_WORKBENCH_BACKEND``.  Keeping the transport behind this narrow API lets
the chat and Skill layers work with either a standalone llama-server process or
an optional llama-cpp-python instance.
"""

from __future__ import annotations

import inspect
import json
import os
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol
from urllib.parse import urlparse

import requests


class BackendError(RuntimeError):
    """A backend could not complete a generation request."""


@dataclass(frozen=True, slots=True)
class ChatResponse:
    """A completed answer with an optional separately reported reasoning trace."""

    content: str = ""
    reasoning: str = ""
    finish_reason: str = ""


class ChatBackend(Protocol):
    """The common local-LLM transport contract."""

    label: str

    def chat(self, messages: list[dict[str, Any]], **settings: Any) -> str:
        """Return the completed assistant message."""


def normalize_server_url(value: str) -> str:
    url = str(value or "").strip().rstrip("/")
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("server_url must be an http(s) URL, for example http://127.0.0.1:8080")
    return url


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") in {"text", "output_text"}:
                texts.append(str(item.get("text", "")))
        return "".join(texts)
    if content is None:
        return ""
    return str(content)


def _chat_response_from_choice(choice: dict[str, Any]) -> ChatResponse:
    message = choice.get("message") or {}
    content = _content_to_text(message.get("content"))
    reasoning = _content_to_text(message.get("reasoning_content"))
    if not content:
        content = _content_to_text(choice.get("text"))
    return ChatResponse(
        content=content,
        reasoning=reasoning,
        finish_reason=str(choice.get("finish_reason") or ""),
    )


@dataclass(slots=True)
class ServerBackend:
    """OpenAI-compatible llama-server connection."""

    server_url: str
    api_key_env: str = ""
    timeout_seconds: float = 120.0
    label: str = "llama-server"
    model_name: str = ""
    owned_by_workbench: bool = False
    context_size: int = 0
    use_environment_proxy: bool = False
    # Start Server supplies a callback here. Keeping it optional means a
    # Connection node remains a plain OpenAI-compatible client, while an owned
    # server can expose the error that llama.cpp wrote to its own stderr/stdout.
    diagnostic_provider: Callable[[], str] | None = field(default=None, repr=False, compare=False)
    # A cached Start Server node may hand Chat a backend after its owned process
    # was released. The callback makes that cached object usable again only
    # when a real request needs it.
    ensure_available: Callable[[], None] | None = field(default=None, repr=False, compare=False)
    _session: requests.Session = field(default_factory=requests.Session, repr=False)

    def __post_init__(self) -> None:
        self.server_url = normalize_server_url(self.server_url)
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.context_size = max(0, int(self.context_size or 0))
        # A ComfyUI process often inherits HTTP(S)_PROXY for downloads. It
        # must not silently route a configured LAN llama-server through that
        # proxy: proxies commonly reject RFC1918 hosts with an opaque 502.
        # Keep explicit opt-in for users intentionally connecting to a public
        # endpoint through their environment proxy.
        self._session.trust_env = bool(self.use_environment_proxy)

    def _headers(self) -> dict[str, str]:
        env_name = self.api_key_env.strip()
        if not env_name:
            return {}
        secret = os.environ.get(env_name, "")
        if not secret:
            raise BackendError(f"API key environment variable is empty: {env_name}")
        return {"Authorization": f"Bearer {secret}"}

    def _with_server_diagnostics(self, message: str) -> str:
        """Append a bounded owned-server log tail to an otherwise opaque 5xx."""

        provider = self.diagnostic_provider
        if callable(provider):
            try:
                diagnostics = str(provider() or "").strip()
            except Exception as exc:  # Diagnostics must never hide the original error.
                diagnostics = f"owned llama-server diagnostics unavailable: {type(exc).__name__}: {exc}"
        else:
            diagnostics = self._probe_external_server()
        return f"{message}\n{diagnostics}" if diagnostics else message

    def _probe_external_server(self) -> str:
        """Collect only public read-only endpoint state after an external 5xx.

        Connection cannot access the remote process logs or stop its process.
        These probes distinguish a reachable llama.cpp server from a reverse
        proxy/router whose model worker has failed, without exposing secrets or
        resending the potentially large multimodal request.
        """

        # Diagnostics should not turn a failed request into a second long
        # timeout. If /health is already a 5xx, the remaining llama.cpp
        # endpoints share the same broken gateway and add no useful signal.
        timeout = min(2.0, max(0.5, float(self.timeout_seconds)))
        reports: list[str] = []
        for path in ("/health", "/v1/models", "/props"):
            try:
                response = self._session.get(
                    self.server_url + path,
                    headers=self._headers(),
                    timeout=timeout,
                )
            except Exception as exc:
                reports.append(f"{path}: {type(exc).__name__}: {exc}")
                continue
            content_type = str(getattr(response, "headers", {}).get("content-type", "")).split(";", 1)[0]
            excerpt = str(getattr(response, "text", "") or "").strip().replace("\n", " ")[:300]
            detail = f"HTTP {response.status_code}"
            if content_type:
                detail += f" ({content_type})"
            if excerpt:
                detail += f": {excerpt}"
            reports.append(f"{path}: {detail}")
            if path == "/health" and response.status_code >= 500:
                break
        return "external server diagnostics (read-only; process logs are on the remote host):\n" + "\n".join(reports)

    def chat_response(self, messages: list[dict[str, Any]], **settings: Any) -> ChatResponse:
        """Return final text and model reasoning as distinct fields when available."""

        if self.ensure_available is not None:
            self.ensure_available()
        body: dict[str, Any] = {"messages": messages, "stream": False}
        for key in ("max_tokens", "temperature", "top_p", "top_k", "repeat_penalty", "seed"):
            value = settings.get(key)
            if value is not None:
                body[key] = value
        model = settings.get("model") or self.model_name
        if model:
            body["model"] = str(model)
        enable_thinking = settings.get("enable_thinking")
        if enable_thinking is not None:
            # llama-server passes this to its Jinja chat template. Models that
            # do not define the variable simply retain their normal template.
            body["chat_template_kwargs"] = {"enable_thinking": bool(enable_thinking)}

        endpoints = ("/v1/chat/completions", "/chat/completions")
        errors: list[str] = []
        for endpoint in endpoints:
            # Older Windows builds and reverse-proxy wrappers can reject the
            # llama.cpp-specific chat_template_kwargs field with an otherwise
            # unhelpful 5xx response. Retry once using only standard OpenAI
            # fields; this preserves normal thinking support on newer servers.
            attempts = [("requested payload", body)]
            if "chat_template_kwargs" in body:
                compatibility_body = dict(body)
                compatibility_body.pop("chat_template_kwargs", None)
                attempts.append(("standard compatibility payload", compatibility_body))

            endpoint_missing = False
            for description, payload_body in attempts:
                try:
                    response = self._session.post(
                        self.server_url + endpoint,
                        json=payload_body,
                        headers=self._headers(),
                        timeout=self.timeout_seconds,
                    )
                except requests.Timeout as exc:
                    # A read timeout does not mean llama-server cancelled the
                    # generation. Replaying this payload may enqueue the same
                    # long request behind the original one and doubles both
                    # the wait time and the remote model work.
                    timeout_error = (
                        f"{endpoint} ({description}): {type(exc).__name__}: {exc}. "
                        f"The request exceeded Connection timeout_seconds ({self.timeout_seconds:g}s); "
                        "no compatibility retry was sent because the remote server may still be generating it. "
                        "Wait for the remote llama-server to finish, then queue once, or increase timeout_seconds "
                        "for long reasoning/unlimited-token responses."
                    )
                    raise BackendError(self._with_server_diagnostics("llama-server request timed out: " + timeout_error)) from exc
                except requests.RequestException as exc:
                    errors.append(f"{endpoint} ({description}): {type(exc).__name__}: {exc}")
                    # Network failures are unrelated to template compatibility;
                    # do not submit a duplicate prompt with a different body.
                    break
                if response.status_code == 404:
                    errors.append(f"{endpoint}: HTTP 404")
                    endpoint_missing = True
                    break
                if not response.ok:
                    excerpt = response.text[:800].strip() or "<empty response body>"
                    errors.append(f"{endpoint} ({description}): HTTP {response.status_code}: {excerpt}")
                    continue
                try:
                    response_payload = response.json()
                    choices = response_payload.get("choices") or []
                    if not choices:
                        raise ValueError("response has no choices")
                    completion = _chat_response_from_choice(choices[0])
                    if completion.content or completion.reasoning:
                        return completion
                    finish_reason = completion.finish_reason or "unknown"
                    raise BackendError(
                        "llama-server returned no assistant text "
                        f"(finish_reason={finish_reason}). For a reasoning model, set thinking to off "
                        "or increase max_tokens."
                    )
                except (ValueError, TypeError, KeyError) as exc:
                    raise BackendError(f"Invalid chat-completions response: {exc}") from exc
            if not endpoint_missing and errors:
                raise BackendError(self._with_server_diagnostics("llama-server request failed: " + " | ".join(errors[-len(attempts) :])))
        raise BackendError(self._with_server_diagnostics("Could not reach a compatible chat endpoint: " + " | ".join(errors)))

    def chat(self, messages: list[dict[str, Any]], **settings: Any) -> str:
        """Return answer text, retaining the legacy text-only backend contract."""

        completion = self.chat_response(messages, **settings)
        return completion.content or completion.reasoning


@dataclass(slots=True)
class EmbeddedBackend:
    """Thread-safe adapter around an optional llama-cpp-python ``Llama`` object."""

    llm: Any
    label: str = "embedded llama-cpp-python"
    ensure_available: Callable[[], "EmbeddedBackend"] | None = field(default=None, repr=False, compare=False)
    _active_backend: "EmbeddedBackend | None" = field(default=None, repr=False, compare=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    def chat_response(self, messages: list[dict[str, Any]], **settings: Any) -> ChatResponse:
        if self.ensure_available is not None:
            active = self.ensure_available()
            if active is not self:
                self.llm = active.llm
                self._active_backend = active
        kwargs: dict[str, Any] = {"messages": messages, "stream": False}
        for key in ("max_tokens", "temperature", "top_p", "top_k", "repeat_penalty", "seed"):
            value = settings.get(key)
            if value is not None:
                kwargs[key] = value
        try:
            signature = inspect.signature(self.llm.create_chat_completion)
            if not any(p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()):
                kwargs = {name: value for name, value in kwargs.items() if name in signature.parameters}
            with self._lock:
                result = self.llm.create_chat_completion(**kwargs)
            choices = result.get("choices") or []
            if not choices:
                raise BackendError("Embedded llama-cpp-python returned no choices")
            completion = _chat_response_from_choice(choices[0])
            if completion.content or completion.reasoning:
                return completion
            raise BackendError(
                "Embedded llama-cpp-python returned no assistant text "
                f"(finish_reason={completion.finish_reason or 'unknown'})."
            )
        except BackendError:
            raise
        except Exception as exc:
            raise BackendError(f"Embedded llama-cpp-python generation failed: {type(exc).__name__}: {exc}") from exc

    def chat(self, messages: list[dict[str, Any]], **settings: Any) -> str:
        """Return answer text, retaining the legacy text-only backend contract."""

        completion = self.chat_response(messages, **settings)
        return completion.content or completion.reasoning

    def close(self) -> None:
        close = getattr(self.llm, "close", None)
        if callable(close):
            close()


def make_text_message(text: str, image_data_urls: list[str] | None = None) -> dict[str, Any]:
    """Create one OpenAI-style user message, optionally with VLM image parts."""

    clean_text = str(text or "")
    images = [item for item in (image_data_urls or []) if item]
    if not images:
        return {"role": "user", "content": clean_text}
    content: list[dict[str, Any]] = [{"type": "text", "text": clean_text}]
    content.extend({"type": "image_url", "image_url": {"url": data}} for data in images)
    return {"role": "user", "content": content}


def backend_descriptor(backend: ChatBackend) -> str:
    """A serializable, secret-free diagnostics string for node output."""

    if isinstance(backend, ServerBackend):
        return json.dumps(
            {
                "kind": "server",
                "url": backend.server_url,
                "label": backend.label,
                "model": backend.model_name,
                "context_size": backend.context_size,
            }
        )
    return json.dumps({"kind": "embedded", "label": getattr(backend, "label", "embedded")})
