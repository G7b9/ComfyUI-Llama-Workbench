"""Safely own a single external llama-server process for this node package."""

from __future__ import annotations

import hashlib
import os
import signal
import shlex
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from .backend import ServerBackend, normalize_server_url


class ServerLaunchError(RuntimeError):
    """The configured local llama-server could not be started."""


@dataclass(frozen=True, slots=True)
class ServerLaunchConfig:
    binary_path: str
    model_path: str
    host: str = "127.0.0.1"
    port: int = 8080
    context_size: int = 8192
    gpu_layers: int = -1
    mmproj_path: str = ""
    extra_args: str = ""

    def __post_init__(self) -> None:
        if not str(self.model_path).strip():
            raise ValueError("model_path is required")
        if not 1 <= int(self.port) <= 65535:
            raise ValueError("port must be between 1 and 65535")
        if int(self.context_size) < 512:
            raise ValueError("context_size must be at least 512")
        if not str(self.host).strip():
            raise ValueError("host is required")

    def fingerprint(self, resolved_binary: str) -> str:
        fields = "\0".join(
            [
                str(Path(resolved_binary).resolve()),
                str(Path(self.model_path).resolve()),
                self.host,
                str(self.port),
                str(self.context_size),
                str(self.gpu_layers),
                str(Path(self.mmproj_path).resolve()) if self.mmproj_path else "",
                self.extra_args,
            ]
        )
        return hashlib.sha256(fields.encode("utf-8")).hexdigest()


def _resolve_binary(value: str) -> str:
    requested = str(value or "").strip() or os.environ.get("LWB_LLAMA_SERVER_BINARY", "").strip()
    if requested:
        path = Path(requested).expanduser()
        if not path.is_file():
            raise ServerLaunchError(f"llama-server binary was not found: {path}")
        return str(path.resolve())
    for name in ("llama-server", "llama-server.exe"):
        found = shutil.which(name)
        if found:
            return found
    raise ServerLaunchError(
        "Cannot find llama-server. Set binary_path, LWB_LLAMA_SERVER_BINARY, or add llama-server to PATH."
    )


def _parse_extra_args(value: str) -> list[str]:
    try:
        # Widgets use the same double-quote syntax on every platform.  The
        # non-POSIX shlex mode keeps the quote characters, which would turn a
        # quoted Windows value into a different argv item.
        return shlex.split(str(value or ""), posix=True)
    except ValueError as exc:
        raise ServerLaunchError(f"Invalid extra_args quoting: {exc}") from exc


class OwnedServer:
    """One positively-owned server process, never discovered or killed by name."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._process: subprocess.Popen[str] | None = None
        self._config_fingerprint = ""
        self._config: ServerLaunchConfig | None = None
        self._binary = ""
        self._command: list[str] = []
        self._started_at: float | None = None
        self._logs: list[str] = []
        self._log_thread: threading.Thread | None = None

    @property
    def is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def is_running_for(self, config: ServerLaunchConfig) -> bool:
        """Return whether this exact launch configuration is already live."""

        with self._lock:
            return self.is_running and self._config == config

    def _record_logs(self, process: subprocess.Popen[str]) -> None:
        if process.stdout is None:
            return
        for line in iter(process.stdout.readline, ""):
            with self._lock:
                self._logs.append(line.rstrip())
                if len(self._logs) > 200:
                    del self._logs[:-200]

    def _command_for(self, binary: str, config: ServerLaunchConfig) -> list[str]:
        model = Path(config.model_path).expanduser()
        if not model.is_file():
            raise ServerLaunchError(f"model_path was not found: {model}")
        args = [
            binary,
            "-m",
            str(model.resolve()),
            "--host",
            config.host.strip(),
            "--port",
            str(config.port),
            "-c",
            str(config.context_size),
            "-ngl",
            str(config.gpu_layers),
        ]
        if config.mmproj_path.strip():
            projector = Path(config.mmproj_path).expanduser()
            if not projector.is_file():
                raise ServerLaunchError(f"mmproj_path was not found: {projector}")
            args.extend(["--mmproj", str(projector.resolve())])
        args.extend(_parse_extra_args(config.extra_args))
        return args

    @staticmethod
    def _probe(binary: str) -> None:
        """Give forked binaries an early, readable failure without imposing a version pin."""
        try:
            completed = subprocess.run(
                [binary, "--version"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise ServerLaunchError(f"Cannot execute llama-server binary: {exc}") from exc
        if completed.returncode != 0 and not (completed.stdout or "").strip():
            raise ServerLaunchError("The configured binary did not respond to --version")

    @staticmethod
    def _endpoint(config: ServerLaunchConfig) -> str:
        host = config.host.strip()
        if host in {"0.0.0.0", "::"}:
            host = "127.0.0.1"
        return normalize_server_url(f"http://{host}:{config.port}")

    @classmethod
    def _endpoint_responds(cls, config: ServerLaunchConfig) -> bool:
        """Return whether an already-running compatible server answers here.

        This check is deliberately used only before spawning a new process.
        A server found this way is *not* ours: adopting it would make a later
        auto-unload claim to release VRAM while leaving that process alive.
        """

        url = cls._endpoint(config)
        for path in ("/health", "/v1/models", "/props"):
            try:
                response = requests.get(url + path, timeout=0.75)
                if response.status_code < 500:
                    return True
            except requests.RequestException:
                pass
        return False

    def _wait_ready(self, config: ServerLaunchConfig, seconds: float) -> str:
        url = self._endpoint(config)
        deadline = time.monotonic() + seconds
        probe_paths = ("/health", "/v1/models", "/props")
        while time.monotonic() < deadline:
            if not self.is_running:
                tail = "\n".join(self._logs[-20:]) or "no server output"
                raise ServerLaunchError(f"llama-server exited during startup.\n{tail}")
            for path in probe_paths:
                try:
                    response = requests.get(url + path, timeout=1.5)
                    if response.status_code < 500:
                        return url
                except requests.RequestException:
                    pass
            time.sleep(0.25)
        tail = "\n".join(self._logs[-20:]) or "no server output"
        raise ServerLaunchError(f"Timed out waiting for {url}.\n{tail}")

    @staticmethod
    def _process_argv(pid: int) -> list[str] | None:
        """Read argv on Linux without guessing from ``ps`` output.

        The package only uses this to remove a stale server that was started
        through Start Server.  Returning ``None`` on other systems keeps the
        cleanup conservative rather than trying to discover arbitrary
        llama.cpp processes by name.
        """

        cmdline = Path(f"/proc/{pid}/cmdline")
        try:
            parts = cmdline.read_bytes().split(b"\0")
        except OSError:
            return None
        return [os.fsdecode(part) for part in parts if part]

    @staticmethod
    def _argv_uses_port(argv: list[str], port: int) -> bool:
        target = str(int(port))
        return any(arg == "--port" and index + 1 < len(argv) and argv[index + 1] == target for index, arg in enumerate(argv))

    @staticmethod
    def _same_binary(first: str, second: str) -> bool:
        try:
            return Path(first).resolve() == Path(second).resolve()
        except OSError:
            return first == second

    @classmethod
    def _is_conflicting_server_argv(cls, argv: list[str] | None, binary: str, port: int) -> bool:
        return bool(argv and cls._same_binary(argv[0], binary) and cls._argv_uses_port(argv, port))

    @staticmethod
    def _pid_is_alive(pid: int) -> bool:
        try:
            os.kill(pid, 0)
            return True
        except (ProcessLookupError, PermissionError, OSError):
            return False

    def _terminate_pid(self, pid: int) -> bool:
        """Terminate a positively matched stale child server on Unix."""

        if pid <= 0 or pid == os.getpid() or os.name == "nt" or not self._pid_is_alive(pid):
            return False
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            return False
        deadline = time.monotonic() + 8.0
        while time.monotonic() < deadline:
            if not self._pid_is_alive(pid):
                return True
            time.sleep(0.1)
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            return not self._pid_is_alive(pid)
        return True

    def _cleanup_conflicting_servers(self, binary: str, port: int) -> list[int]:
        """Remove stale Start Server processes matching the binary and port.

        A timeout used to intentionally leave the child process alive.  That
        is unsafe for a GPU-shared workflow: the orphan retains VRAM, RAM and
        the port.  Match the exact executable and ``--port`` from a Linux
        argv record, never a loose process-name search.  Users who intentionally
        run a server outside Workbench should use the Connection node instead
        of enabling this lifecycle management path.
        """

        proc_root = Path("/proc")
        if os.name == "nt" or not proc_root.is_dir():
            return []
        stopped: list[int] = []
        try:
            entries = list(proc_root.iterdir())
        except OSError:
            return stopped
        for entry in entries:
            if not entry.name.isdecimal():
                continue
            pid = int(entry.name)
            argv = self._process_argv(pid)
            if not self._is_conflicting_server_argv(argv, binary, port):
                continue
            if self._terminate_pid(pid):
                stopped.append(pid)
        return stopped

    def _raise_after_failed_start(self, error: ServerLaunchError) -> None:
        """Do not leave a failed launch holding the GPU or its TCP port."""

        stopped = self.stop()
        suffix = (
            "The owned llama-server was stopped to release its GPU memory, system memory, and port. "
            "Increase wait_seconds before trying again."
            if stopped
            else "The owned llama-server is no longer running."
        )
        raise ServerLaunchError(f"{error}\n{suffix}") from error

    def request_diagnostics(self) -> str:
        """Return a compact, safe diagnostic attached to an owned request failure."""

        with self._lock:
            process = self._process
            pid = process.pid if process is not None else None
            returncode = process.poll() if process is not None else None
            state = "running" if self.is_running else "stopped"
            tail = "\n".join(self._logs[-30:]) or "no server output captured"
        return f"owned llama-server diagnostics (pid={pid}, state={state}, returncode={returncode}):\n{tail}"

    def _make_backend(
        self,
        url: str,
        config: ServerLaunchConfig,
        timeout_seconds: float = 120.0,
    ) -> ServerBackend:
        return ServerBackend(
            url,
            timeout_seconds=float(timeout_seconds),
            label="owned llama-server",
            owned_by_workbench=True,
            context_size=config.context_size,
            diagnostic_provider=self.request_diagnostics,
        )

    def start(
        self,
        config: ServerLaunchConfig,
        wait_seconds: float = 60.0,
        cleanup_previous_server: bool = True,
        timeout_seconds: float = 120.0,
    ) -> ServerBackend:
        if wait_seconds <= 0:
            raise ValueError("wait_seconds must be positive")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        with self._lock:
            binary = _resolve_binary(config.binary_path)
            fingerprint = config.fingerprint(binary)
            if self.is_running and self._config_fingerprint == fingerprint:
                try:
                    url = self._wait_ready(config, float(wait_seconds))
                except ServerLaunchError as exc:
                    self._raise_after_failed_start(exc)
                return self._make_backend(url, config, timeout_seconds)
            if self.is_running:
                self.stop()
            if cleanup_previous_server:
                self._cleanup_conflicting_servers(binary, config.port)
            endpoint = self._endpoint(config)
            if self._endpoint_responds(config):
                raise ServerLaunchError(
                    f"A llama-server is already responding at {endpoint}, but it is not owned by this "
                    "Workbench session. Refusing to adopt it because auto_unload could not safely stop it. "
                    "Stop the existing server that owns this port, or use Llama Workbench Connection if it "
                    "is intentionally managed outside Workbench."
                )
            self._probe(binary)
            command = self._command_for(binary, config)
            self._logs = []
            creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) if os.name == "nt" else 0
            try:
                process = subprocess.Popen(
                    command,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    shell=False,
                    start_new_session=os.name != "nt",
                    creationflags=creationflags,
                )
            except OSError as exc:
                raise ServerLaunchError(f"Failed to start llama-server: {exc}") from exc
            self._process = process
            self._config = config
            self._config_fingerprint = fingerprint
            self._binary = binary
            self._command = command
            self._started_at = time.time()
            self._log_thread = threading.Thread(target=self._record_logs, args=(process,), daemon=True)
            self._log_thread.start()
            try:
                url = self._wait_ready(config, float(wait_seconds))
            except ServerLaunchError as exc:
                self._raise_after_failed_start(exc)
            return self._make_backend(url, config, timeout_seconds)

    def stop(self) -> bool:
        """Stop only the exact process started by this object."""
        with self._lock:
            process = self._process
            if process is None:
                return False
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=4)
            self._process = None
            self._config_fingerprint = ""
            self._config = None
            self._binary = ""
            self._command = []
            self._started_at = None
            return True

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "owner": "ComfyUI-Llama-Workbench",
                "running": self.is_running,
                "pid": self._process.pid if self._process is not None else None,
                "returncode": self._process.poll() if self._process is not None else None,
                "started_at": self._started_at,
                "binary": self._binary or None,
                "command": list(self._command),
                "log_tail": self._logs[-30:],
            }


OWNED_SERVER = OwnedServer()
