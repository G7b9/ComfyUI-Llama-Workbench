from __future__ import annotations

import pytest
import requests

from lwb.backend import BackendError, ServerBackend, make_text_message, normalize_server_url


class FakeResponse:
    status_code = 200
    ok = True
    text = ""

    @staticmethod
    def json():
        return {"choices": [{"message": {"content": "local answer"}}]}


class FakeSession:
    def __init__(self):
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return FakeResponse()


def test_normalize_server_url_rejects_non_http():
    assert normalize_server_url("http://127.0.0.1:8080/") == "http://127.0.0.1:8080"
    try:
        normalize_server_url("file:///tmp/server")
    except ValueError as error:
        assert "http(s)" in str(error)
    else:
        raise AssertionError("expected invalid URL")


def test_server_backend_posts_openai_chat_payload():
    backend = ServerBackend("http://localhost:8080", model_name="qwen-router-id")
    session = FakeSession()
    backend._session = session
    result = backend.chat(
        [{"role": "user", "content": "hello"}], max_tokens=-1, temperature=0.2, enable_thinking=False
    )
    assert result == "local answer"
    assert session.calls[0][0].endswith("/v1/chat/completions")
    assert session.calls[0][1]["json"]["max_tokens"] == -1
    assert session.calls[0][1]["json"]["model"] == "qwen-router-id"
    assert session.calls[0][1]["json"]["chat_template_kwargs"] == {"enable_thinking": False}


def test_owned_server_backend_ensures_its_cached_process_is_available_before_request():
    ensured = []
    backend = ServerBackend("http://localhost:8080", owned_by_workbench=True, ensure_available=lambda: ensured.append(True))
    backend._session = FakeSession()

    assert backend.chat([{"role": "user", "content": "hello"}]) == "local answer"
    assert ensured == [True]


def test_server_backend_retries_external_502_without_llama_template_kwargs():
    class UnsupportedTemplateResponse:
        status_code = 502
        ok = False
        text = ""

    class CompatibilitySession(FakeSession):
        def post(self, url, **kwargs):
            self.calls.append((url, kwargs))
            return UnsupportedTemplateResponse() if len(self.calls) == 1 else FakeResponse()

    backend = ServerBackend("http://192.0.2.1:50003")
    session = CompatibilitySession()
    backend._session = session

    assert backend.chat([{"role": "user", "content": "hello"}], enable_thinking=True) == "local answer"
    assert session.calls[0][1]["json"]["chat_template_kwargs"] == {"enable_thinking": True}
    assert "chat_template_kwargs" not in session.calls[1][1]["json"]


def test_server_backend_does_not_resubmit_a_generation_after_read_timeout():
    class ProbeResponse:
        status_code = 200
        text = '{"status":"ok"}'
        headers = {"content-type": "application/json"}

    class TimeoutSession(FakeSession):
        def post(self, url, **kwargs):
            self.calls.append((url, kwargs))
            raise requests.exceptions.ReadTimeout("model is still generating")

        @staticmethod
        def get(url, **kwargs):
            return ProbeResponse()

    backend = ServerBackend("http://192.0.2.1:50003")
    session = TimeoutSession()
    backend._session = session

    with pytest.raises(BackendError, match="no compatibility retry was sent"):
        backend.chat([{"role": "user", "content": "hello"}], enable_thinking=False)
    assert len(session.calls) == 1


def test_server_backend_attaches_owned_server_log_tail_to_a_5xx():
    class FailedResponse:
        status_code = 502
        ok = False
        text = ""

    class FailedSession(FakeSession):
        def post(self, url, **kwargs):
            self.calls.append((url, kwargs))
            return FailedResponse()

    backend = ServerBackend(
        "http://localhost:8080",
        diagnostic_provider=lambda: "owned llama-server diagnostics (pid=42, state=running):\\nCUDA error: out of memory",
    )
    backend._session = FailedSession()

    with pytest.raises(BackendError, match="CUDA error: out of memory"):
        backend.chat([{"role": "user", "content": "hello"}])


def test_server_backend_probes_an_external_server_after_a_5xx():
    class FailedResponse:
        status_code = 502
        ok = False
        text = ""

    class ProbeResponse:
        status_code = 200
        text = '{"status":"ok"}'
        headers = {"content-type": "application/json"}

    class ExternalSession(FakeSession):
        def post(self, url, **kwargs):
            self.calls.append((url, kwargs))
            return FailedResponse()

        @staticmethod
        def get(url, **kwargs):
            return ProbeResponse()

    backend = ServerBackend("http://localhost:8080")
    backend._session = ExternalSession()

    with pytest.raises(BackendError, match="external server diagnostics") as error:
        backend.chat([{"role": "user", "content": "hello"}])
    assert "/health: HTTP 200" in str(error.value)
    assert "/v1/models: HTTP 200" in str(error.value)


def test_server_backend_keeps_the_optional_context_size_for_the_chat_meter():
    backend = ServerBackend("http://localhost:8080", context_size=32768)
    assert backend.context_size == 32768
    assert backend._session.trust_env is False


def test_server_backend_can_explicitly_use_the_environment_proxy():
    backend = ServerBackend("http://localhost:8080", use_environment_proxy=True)
    assert backend._session.trust_env is True


def test_image_message_uses_openai_content_parts():
    message = make_text_message("describe", ["data:image/jpeg;base64,abc"])
    assert message["role"] == "user"
    assert message["content"][0]["text"] == "describe"
    assert message["content"][1]["image_url"]["url"].startswith("data:image/")


def test_server_backend_exposes_reasoning_when_content_is_empty():
    class ReasoningResponse(FakeResponse):
        @staticmethod
        def json():
            return {"choices": [{"message": {"content": "", "reasoning_content": "image analysis"}}]}

    class ReasoningSession(FakeSession):
        def post(self, url, **kwargs):
            self.calls.append((url, kwargs))
            return ReasoningResponse()

    backend = ServerBackend("http://localhost:8080")
    backend._session = ReasoningSession()
    assert backend.chat([{"role": "user", "content": "describe"}]) == "image analysis"
    completion = backend.chat_response([{"role": "user", "content": "describe"}])
    assert completion.content == ""
    assert completion.reasoning == "image analysis"
