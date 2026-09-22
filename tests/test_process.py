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


def test_stale_cleanup_only_targets_the_configured_binary_and_port():
    server = OwnedServer()
    assert server._is_conflicting_server_argv(["/opt/llama-server", "--port", "50003"], "/opt/llama-server", 50003)
    assert not server._is_conflicting_server_argv(["/opt/llama-server", "--port", "8080"], "/opt/llama-server", 50003)
    assert not server._is_conflicting_server_argv(["/opt/other-server", "--port", "50003"], "/opt/llama-server", 50003)


def test_start_refuses_a_preexisting_unowned_server(monkeypatch, tmp_path):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"gguf")
    config = ServerLaunchConfig(binary_path="unused", model_path=str(model), port=50123)
    server = OwnedServer()

    monkeypatch.setattr(process_module, "_resolve_binary", lambda _: "llama-server")
    monkeypatch.setattr(server, "_endpoint_responds", lambda current: current is config)

    with pytest.raises(ServerLaunchError, match="not owned by this Workbench session"):
        server.start(config)
