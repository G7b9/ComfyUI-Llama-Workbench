"""Native workflow access policy; mocked TCP peers include actual LAN addresses."""

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest
from aiohttp import web
from aiohttp.test_utils import make_mocked_request

from lwb.scheduler import authorize, create_lifecycle_middleware, lifecycle_restriction_enabled


@pytest.fixture(autouse=True)
def clear_policy(monkeypatch):
    for name in ("LWB_RESTRICT_LIFECYCLE", "LWB_API_TOKEN", "LWB_API_TOKENS_JSON"):
        monkeypatch.delenv(name, raising=False)


def request(path, node="LlamaWorkbench_StartServer", peer="192.168.0.42", headers=None, extra=None):
    transport = Mock()
    transport.get_extra_info.return_value = (peer, 52000)
    req = make_mocked_request("POST", path, headers=headers or {}, transport=transport)
    req.json = AsyncMock(
        return_value={
            "prompt": {
                "1": {
                    "class_type": node,
                    "inputs": {
                        "binary_path": "/custom/llama-server",
                        "model_path": "/models/model.gguf",
                        "port": 8080,
                        "extra_args": "--custom-option unchanged",
                    },
                }
            },
            **(extra or {}),
        }
    )
    return req


@pytest.mark.parametrize(
    "raw, expected",
    [
        (None, False),
        ("false", False),
        (" FALSE ", False),
        ("0", False),
        ("no", False),
        ("off", False),
        ("true", True),
        ("TRUE", True),
        ("1", True),
        ("yes", True),
        ("on", True),
    ],
)
def test_server_boolean_is_explicit(monkeypatch, raw, expected):
    if raw is not None:
        monkeypatch.setenv("LWB_RESTRICT_LIFECYCLE", raw)
    assert lifecycle_restriction_enabled() is expected


@pytest.mark.parametrize("raw", ["", "maybe", "null", "2", "disabled"])
def test_invalid_configuration_prevents_middleware_installation(monkeypatch, raw):
    monkeypatch.setenv("LWB_RESTRICT_LIFECYCLE", raw)
    with pytest.raises(ValueError, match="LWB_RESTRICT_LIFECYCLE"):
        create_lifecycle_middleware()


@pytest.mark.parametrize("path", ["/prompt", "/api/prompt"])
@pytest.mark.parametrize("node", ["LlamaWorkbench_StartServer", "LlamaWorkbench_StopServer"])
@pytest.mark.parametrize("mode", [None, "false"])
def test_default_native_workflows_defer_to_comfy_access_control(monkeypatch, path, node, mode):
    if mode is not None:
        monkeypatch.setenv("LWB_RESTRICT_LIFECYCLE", mode)
    # Configuring API authentication must not impose it on native workflows.
    monkeypatch.setenv("LWB_API_TOKEN", "api-only-secret")
    req = request(
        path, node, headers={"Host": "192.168.0.28:8188", "Origin": "http://192.168.0.28:8188"}
    )
    handler = AsyncMock(return_value=web.Response(status=202))
    original = asyncio.run(req.json())
    response = asyncio.run(create_lifecycle_middleware()(req, handler))
    assert response.status == 202
    handler.assert_awaited_once_with(req)
    assert asyncio.run(req.json()) == original  # no rewriting workflow inputs


@pytest.mark.parametrize("path", ["/prompt", "/api/prompt"])
@pytest.mark.parametrize("peer", ["127.0.0.1", "::1", "192.168.0.42"])
@pytest.mark.parametrize("node", ["LlamaWorkbench_StartServer", "LlamaWorkbench_StopServer"])
def test_strict_native_workflows_preserve_local_ui_rules(monkeypatch, path, peer, node):
    monkeypatch.setenv("LWB_RESTRICT_LIFECYCLE", "true")
    req = request(
        path,
        node=node,
        peer=peer,
        headers={"Host": "localhost:8188", "Origin": "http://localhost:8188"},
    )
    handler = AsyncMock(return_value=web.Response(status=202))
    middleware = create_lifecycle_middleware()
    if peer.startswith("192."):
        with pytest.raises(web.HTTPForbidden):
            asyncio.run(middleware(req, handler))
        handler.assert_not_awaited()
    else:
        assert asyncio.run(middleware(req, handler)).status == 202


@pytest.mark.parametrize("path", ["/prompt", "/api/prompt"])
def test_strict_mode_rejects_even_authenticated_remote_clients(monkeypatch, path):
    monkeypatch.setenv("LWB_RESTRICT_LIFECYCLE", "true")
    monkeypatch.setenv("LWB_API_TOKEN", "secret")
    req = request(path, headers={"Authorization": "Bearer secret"})
    with pytest.raises(web.HTTPForbidden) as error:
        asyncio.run(create_lifecycle_middleware()(req, AsyncMock()))
    assert "Remote lifecycle nodes" in error.value.text


@pytest.mark.parametrize("mode", ["false", "true"])
@pytest.mark.parametrize("path", ["/prompt", "/api/prompt"])
def test_internal_queued_text_never_accepts_direct_submissions(monkeypatch, mode, path):
    monkeypatch.setenv("LWB_RESTRICT_LIFECYCLE", mode)
    with pytest.raises(web.HTTPForbidden) as error:
        asyncio.run(
            create_lifecycle_middleware()(request(path, "LlamaWorkbench_QueuedText"), AsyncMock())
        )
    assert "QueuedText is internal" in error.value.text


@pytest.mark.parametrize("mode", ["false", "true"])
def test_management_api_authentication_is_independent(monkeypatch, mode):
    monkeypatch.setenv("LWB_RESTRICT_LIFECYCLE", mode)
    middleware = create_lifecycle_middleware()

    async def api_handler(req):
        authorize(req)
        return web.Response(status=200)

    # No token configured: remote still rejected, loopback still allowed.
    with pytest.raises(PermissionError):
        asyncio.run(middleware(request("/lwb/v1/status"), api_handler))
    assert (
        asyncio.run(middleware(request("/lwb/v1/status", peer="127.0.0.1"), api_handler)).status
        == 200
    )
    monkeypatch.setenv("LWB_API_TOKEN", "secret")
    with pytest.raises(PermissionError):
        asyncio.run(middleware(request("/lwb/v1/status", peer="127.0.0.1"), api_handler))
    assert (
        asyncio.run(
            middleware(
                request("/lwb/v1/status", headers={"Authorization": "Bearer secret"}), api_handler
            )
        ).status
        == 200
    )
    with pytest.raises(PermissionError):
        asyncio.run(
            middleware(
                request(
                    "/lwb/v1/status",
                    headers={
                        "Authorization": "Bearer secret",
                        "Origin": "http://192.168.0.28:8188",
                    },
                ),
                api_handler,
            )
        )


@pytest.mark.parametrize("path", ["/prompt", "/api/prompt"])
def test_request_cannot_override_captured_server_policy(monkeypatch, path):
    monkeypatch.setenv("LWB_RESTRICT_LIFECYCLE", "true")
    middleware = create_lifecycle_middleware()
    monkeypatch.setenv("LWB_RESTRICT_LIFECYCLE", "false")
    req = request(
        path + "?LWB_RESTRICT_LIFECYCLE=false",
        headers={"LWB_RESTRICT_LIFECYCLE": "false"},
        extra={"LWB_RESTRICT_LIFECYCLE": False},
    )
    body = asyncio.run(req.json())
    body["prompt"]["1"]["inputs"]["LWB_RESTRICT_LIFECYCLE"] = False
    with pytest.raises(web.HTTPForbidden):
        asyncio.run(middleware(req, AsyncMock()))


def test_default_mode_does_not_override_comfy_denial():
    async def deny(req):
        raise web.HTTPForbidden(text="ComfyUI access denied")

    with pytest.raises(web.HTTPForbidden) as error:
        asyncio.run(create_lifecycle_middleware()(request("/prompt"), deny))
    assert error.value.text == "ComfyUI access denied"


@pytest.mark.parametrize("path", ["/prompt", "/api/prompt"])
def test_strict_mode_preserves_local_token_and_origin_requirements(monkeypatch, path):
    monkeypatch.setenv("LWB_RESTRICT_LIFECYCLE", "true")
    monkeypatch.setenv("LWB_API_TOKEN", "secret")
    middleware = create_lifecycle_middleware()
    handler = AsyncMock(return_value=web.Response(status=202))
    with pytest.raises(web.HTTPForbidden):
        asyncio.run(middleware(request(path, peer="127.0.0.1"), handler))
    with pytest.raises(web.HTTPForbidden):
        asyncio.run(
            middleware(
                request(
                    path,
                    peer="127.0.0.1",
                    headers={
                        "Authorization": "Bearer secret",
                        "Origin": "http://untrusted.example",
                    },
                ),
                handler,
            )
        )
    assert (
        asyncio.run(
            middleware(
                request(
                    path,
                    peer="127.0.0.1",
                    headers={
                        "Authorization": "Bearer secret",
                        "Host": "localhost:8188",
                        "Origin": "http://localhost:8188",
                    },
                ),
                handler,
            )
        ).status
        == 202
    )
