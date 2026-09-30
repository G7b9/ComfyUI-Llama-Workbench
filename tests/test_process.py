from __future__ import annotations

import pytest

import lwb.process as process_module
from lwb.process import OwnedServer, ServerLaunchConfig, ServerLaunchError, _parse_extra_args


def test_command_is_argv_and_keeps_quoted_custom_arguments(tmp_path):
    model = tmp_path / "model with spaces.gguf"
    projector = tmp_path / "projector.gguf"
    model.write_bytes(b"gguf")
    projector.write_bytes(b"gguf")
    config = ServerLaunchConfig(
        binary_path="unused",
        model_path=str(model),
        context_size=4096,
        gpu_layers=-1,
        mmproj_path=str(projector),
        extra_args='--chat-template "my template" --flash-attn on',
    )
    command = OwnedServer()._command_for("C:/tools/llama-server.exe", config)
    assert command[0].endswith("llama-server.exe")
    assert "--mmproj" in command
    assert command[-4:] == ["--chat-template", "my template", "--flash-attn", "on"]


def test_status_never_adopts_a_process():
    server = OwnedServer()
    status = server.status()
    assert status["owner"] == "ComfyUI-Llama-Workbench"
    assert status["running"] is False
    assert server.stop() is False


def test_extra_args_reports_unmatched_quote():
    try:
        _parse_extra_args('"unterminated')
    except Exception as error:
        assert "quoting" in str(error)
    else:
        raise AssertionError("expected malformed quoting failure")


def test_same_owned_process_is_rechecked_after_a_startup_timeout(monkeypatch, tmp_path):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"gguf")
    binary = str(tmp_path / "llama-server")
    config = ServerLaunchConfig(binary_path=binary, model_path=str(model))
    server = OwnedServer()

    class RunningProcess:
        pid = 42

        @staticmethod
        def poll():
            return None

    server._process = RunningProcess()
    server._config_fingerprint = config.fingerprint(binary)
    monkeypatch.setattr(process_module, "_resolve_binary", lambda _: binary)
    calls = []
    monkeypatch.setattr(server, "_wait_ready", lambda current, seconds: calls.append((current, seconds)) or "http://127.0.0.1:8080")

    backend = server.start(config, 123, timeout_seconds=456)

    assert backend.server_url == "http://127.0.0.1:8080"
    assert backend.timeout_seconds == 456.0
    assert calls == [(config, 123.0)]


def test_timed_out_owned_process_is_stopped_and_releases_its_port(monkeypatch, tmp_path):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"gguf")
    binary = str(tmp_path / "llama-server")
    config = ServerLaunchConfig(binary_path=binary, model_path=str(model))
    server = OwnedServer()

    class RunningProcess:
        pid = 42

        def __init__(self):
            self.running = True
            self.terminated = False

        def poll(self):
            return None if self.running else 0

        def terminate(self):
            self.terminated = True
            self.running = False

        def wait(self, timeout):
            return 0

    process = RunningProcess()
    server._process = process
    server._config_fingerprint = config.fingerprint(binary)
    monkeypatch.setattr(process_module, "_resolve_binary", lambda _: binary)
    monkeypatch.setattr(server, "_wait_ready", lambda current, seconds: (_ for _ in ()).throw(ServerLaunchError("timed out")))

    with pytest.raises(ServerLaunchError, match="was stopped"):
        server.start(config, 123)

    assert process.terminated is True
    assert server.is_running is False


def test_no_process_discovery_or_port_based_cleanup():
    assert not hasattr(OwnedServer, "_cleanup_conflicting_servers")


def test_start_refuses_a_preexisting_unowned_server(monkeypatch, tmp_path):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"gguf")
    config = ServerLaunchConfig(binary_path="unused", model_path=str(model), port=50123)
    server = OwnedServer()

    monkeypatch.setattr(process_module, "_resolve_binary", lambda _: "llama-server")
    monkeypatch.setattr(server, "_endpoint_responds", lambda current: current is config)

    with pytest.raises(ServerLaunchError, match="not owned by this Workbench session"):
        server.start(config)


def test_startup_failure_diagnostics_are_drained_while_lifecycle_lock_held(tmp_path):
    import os
    import sys
    if os.name == "nt":
        pytest.skip("test executable uses a POSIX shebang")
    binary = tmp_path / "fake-llama-server"
    binary.write_text(f"#!{sys.executable}\nimport sys\nif '--version' in sys.argv: sys.exit(0)\nprint('CUDA allocation failed: model traceback', flush=True)\nsys.exit(1)\n")
    binary.chmod(0o700)
    model = tmp_path / "model.gguf"
    model.write_bytes(b"gguf")
    server = OwnedServer()
    # No service is adopted or terminated; avoid any dependency on a free port.
    server._endpoint_responds = lambda config: False
    with pytest.raises(ServerLaunchError, match="CUDA allocation failed"):
        server.start(ServerLaunchConfig(str(binary), str(model), port=65534), wait_seconds=3)
    assert not server.is_running


def test_startup_probes_ignore_inherited_socks_proxy(monkeypatch):
    monkeypatch.setenv("ALL_PROXY", "socks5://invalid.invalid:1080")
    monkeypatch.setenv("HTTP_PROXY", "socks5://invalid.invalid:1080")
    observed = []
    def get(session, url, **kwargs):
        observed.append(session.trust_env)
        return type("Response", (), {"status_code": 200})()
    monkeypatch.setattr(process_module.requests.Session, "get", get)
    config = ServerLaunchConfig("unused", "unused")
    assert OwnedServer._endpoint_responds(config)
    server = OwnedServer()
    server._process = type("Process", (), {"poll": lambda self: None})()
    assert server._wait_ready(config, 1).startswith("http://127.0.0.1")
    assert observed == [False, False]


def test_models_200_does_not_override_loading_health(monkeypatch):
    server = OwnedServer()
    server._process = type("Process", (), {"poll": lambda self: None})()
    calls = []
    def get(url, **kwargs):
        calls.append(url)
        return type("Response", (), {"status_code": 503 if url.endswith("/health") else 200})()
    monkeypatch.setattr(server._session, "get", get)
    with pytest.raises(ServerLaunchError, match="Timed out"):
        server._wait_ready(ServerLaunchConfig("unused", "unused"), 0.01)
    assert all(url.endswith("/health") for url in calls)
