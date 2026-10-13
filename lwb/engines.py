"""Administrator-selected lifecycle adapters; no launcher or arbitrary Docker argv.

The caller holds ExecutionGate before any lifecycle or GPU operation. A resident
lease pins its flock, excluding cooperating external launchers between requests.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
import re
import socket
import subprocess
import threading
import time
import uuid

import requests

from .backend import ServerBackend
from .process import ServerLaunchError, ServerStartupTimeout


def digest_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fingerprint(config):
    return hashlib.sha256(json.dumps(asdict(config), sort_keys=True).encode()).hexdigest()


def container_process(pid, command):
    try:
        root = Path(f"/proc/{pid}")
        argv = root.joinpath("cmdline").read_bytes().rstrip(b"\0").split(b"\0")
        stat = root.joinpath("stat").read_text().rsplit(")", 1)[1].split()
        if [value.decode() for value in argv] != command or stat[0] == "Z":
            raise ServerLaunchError("Container process command differs; preserved")
        return {
            "pid": pid,
            "start_ticks": stat[19],
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        }
    except (OSError, IndexError, UnicodeError) as exc:
        raise ServerLaunchError("Container process generation unconfirmed; preserved") from exc


class ExecutionGate:
    """Order: in-process RLock -> shared deployment flock -> adapter lock.

    Never wait for an external launcher while owning the native worker. Fail
    closed instead. Keep flock for the resident model and unknown outcomes.
    """

    def __init__(self, filename=None):
        self.lock = threading.RLock()
        self.filename = filename
        self.file = None
        self.depth = 0
        self.pinned = False

    def acquire(self, blocking=True):
        if not self.lock.acquire(blocking=blocking):
            return False
        try:
            if self.filename and self.file is None:
                import fcntl

                handle = open(self.filename, "a")
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    handle.close()
                    if not blocking:
                        self.lock.release()
                        return False
                    raise
                except BaseException:
                    handle.close()
                    raise
                self.file = handle
            self.depth += 1
            return True
        except BaseException:
            self.lock.release()
            raise

    def pin(self):
        self.pinned = True

    def unpin(self):
        self.pinned = False

    def release(self):
        self.depth -= 1
        if self.depth == 0 and not self.pinned and self.file:
            self.file.close()
            self.file = None
        self.lock.release()

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *args):
        self.release()


@dataclass(frozen=True, slots=True)
class NInferProfile:
    engine: str
    model_path: str
    model_sha256: str
    runtime_directory: str
    binary_sha256: str
    image_id: str
    toolkit_directory: str
    switch_lock: str
    host: str = "127.0.0.1"
    port: int = 18200
    context_size: int = 262144
    kvmem_budget: int = 98304
    kvmem_gen_reserve: int = 16384
    kvmem_host_mib: int = 12288
    startup_timeout_seconds: float = 600
    model_alias: str = "ninfer-gsq-27b"
    api_key_env: str = ""
    adopt_container_id: str = ""

    def __post_init__(self):
        if self.engine != "ninfer":
            raise ValueError("NInfer engine must be ninfer")
        for name in ("model_sha256", "binary_sha256"):
            if not re.fullmatch(r"[a-f0-9]{64}", getattr(self, name)):
                raise ValueError(f"{name} must be a pinned SHA256")
        if not re.fullmatch(r"sha256:[a-f0-9]{64}", self.image_id):
            raise ValueError("image_id must be immutable sha256, not a Docker tag")
        if self.adopt_container_id and not re.fullmatch(r"[a-f0-9]{64}", self.adopt_container_id):
            raise ValueError("adoption requires a complete administrator-authorized container ID")
        for name in ("model_path", "runtime_directory", "toolkit_directory", "switch_lock"):
            if not Path(getattr(self, name)).is_absolute() or "," in getattr(self, name):
                raise ValueError(f"{name} must be an absolute path without mount separators")
        if Path(self.model_path).suffix != ".ninfer":
            raise ValueError("NInfer model_path must be .ninfer")
        if self.host != "127.0.0.1":
            raise ValueError("NInfer Docker publish must bind 127.0.0.1")
        for name, low, high in (
            ("port", 1, 65535),
            ("context_size", 512, 262144),
            ("kvmem_budget", 512, 262144),
            ("kvmem_gen_reserve", 1, 262144),
            ("kvmem_host_mib", 1, 12288),
        ):
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"invalid {name}")
        if self.kvmem_budget + self.kvmem_gen_reserve > self.context_size:
            raise ValueError("physical KV pool must not exceed logical context")
        value = self.startup_timeout_seconds
        if type(value) not in (int, float) or not math.isfinite(value) or not 1 <= value <= 1800:
            raise ValueError("startup_timeout_seconds must be in [1, 1800]")
        if not self.switch_lock or not self.model_alias or self.model_alias.startswith("-"):
            raise ValueError("switch_lock and model_alias are required")

    def command(self):
        return [
            "/runtime/ninfer-serve",
            "/models/" + Path(self.model_path).name,
            "--host",
            "0.0.0.0",
            "--port",
            "8080",
            "--model-id",
            self.model_alias,
            "--max-context",
            str(self.context_size),
            "--kv-dtype",
            "rk8v4",
            "--gdn-state-fp16",
            "--max-concurrency",
            "1",
            "--spec",
            "mtp",
            "--draft-tokens",
            "3",
            "--ngram-draft-tokens",
            "0",
            "--prefill-chunk",
            "256",
            "--structured-output",
            "--vision",
            "--vision-residency",
            "resident",
            "--vision-max-merged",
            "4096",
            "--kvmem-budget",
            str(self.kvmem_budget),
            "--kvmem-gen-reserve",
            str(self.kvmem_gen_reserve),
            "--kvmem-host-mib",
            str(self.kvmem_host_mib),
            "--kvmem-sessions",
            "1",
            "--kvmem-recent-tokens",
            "1024",
            "--kvmem-score-agg",
            "mean",
            "--kvmem-action-lrfu",
            "--kvmem-lrfu-lambda",
            "0.50",
            "--kvmem-lrfu-top-frac",
            "0.90",
        ]


def docker(*args):
    result = subprocess.run(["docker", *args], capture_output=True, text=True, timeout=90)
    if result.returncode:
        raise ServerLaunchError("Docker operation failed: " + result.stderr[:600])
    return result.stdout.strip()


def inspect_container(cid):
    # Only a definitive not-found from a functioning daemon proves absence.
    result = subprocess.run(["docker", "inspect", cid], capture_output=True, text=True, timeout=30)
    if result.returncode:
        if "No such object:" in result.stderr or "No such container:" in result.stderr:
            return None
        raise ServerLaunchError("Docker identity query unavailable")
    return json.loads(result.stdout)[0]


def container_dead(identity):
    try:
        current = inspect_container(identity["container_id"])
        return current is None or (
            current["Id"] == identity["container_id"]
            and not current["State"]["Running"]
            and current["State"]["Status"]
            in (
                {"exited", "dead"}
                if identity.get("startup_pending")
                else {"exited", "dead", "created"}
            )
            and not current["State"].get("Restarting")
        )
    except (OSError, ValueError, KeyError, subprocess.SubprocessError, ServerLaunchError):
        return False


def gpu_clear(allowed_pid=None):
    """Reject unknown compute occupants; allow only ComfyUI's bounded CUDA context."""
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        timeout=20,
    )
    if result.returncode:
        raise ServerLaunchError("GPU occupancy unavailable; refusing model load/handoff")
    for line in result.stdout.strip().splitlines():
        pid, used = [int(value.strip()) for value in line.split(",")]
        if pid != (allowed_pid or os.getpid()) or used > 2048:
            raise ServerLaunchError("GPU contains unconfirmed or unreleased compute allocation")
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        timeout=20,
    )
    if result.returncode or any(int(v) > 2048 for v in result.stdout.strip().splitlines()):
        raise ServerLaunchError("GPU memory handoff incomplete; refusing model load")


class NInferDocker:
    def __init__(self):
        self.config = None
        self.cid = None
        self.generation = None
        self.owner = None
        self.active = 0
        self.provenance = None
        self.pid_generation = None

    def verify(self):
        c = inspect_container(self.cid)
        p = self.config
        if c is None:
            raise ServerLaunchError("Managed container disappeared")
        mounts = {m["Destination"]: (m["Source"], m["RW"]) for m in c["Mounts"]}
        expected = {
            "/runtime": (p.runtime_directory, False),
            "/models": (str(Path(p.model_path).parent), False),
            "/toolkit": (p.toolkit_directory, False),
        }
        host = c["HostConfig"]
        ports = host.get("PortBindings", {})
        environment = dict(value.split("=", 1) for value in c["Config"].get("Env", []))
        if (
            c["Id"] != self.cid
            or c["Image"] != p.image_id
            or c["Config"]["Cmd"] != p.command()
            or c["Config"].get("Entrypoint") not in (None, [])
            or mounts != expected
            or host.get("Privileged")
            or host.get("CapAdd")
            or host.get("Devices")
            or host.get("Runtime") != "nvidia"
            or host.get("NetworkMode") != "bridge"
            or host.get("VolumesFrom")
            or host.get("SecurityOpt")
            or set(environment)
            - {"PATH", "NVIDIA_VISIBLE_DEVICES", "KVMEM_WINDOW_TRACE", "LD_LIBRARY_PATH"}
            or environment.get("NVIDIA_VISIBLE_DEVICES") != "0"
            or environment.get("KVMEM_WINDOW_TRACE") != "1"
            or environment.get("LD_LIBRARY_PATH") != "/toolkit/lib64"
            or host.get("PidMode")
            or host.get("IpcMode") == "host"
            or host.get("RestartPolicy", {}).get("Name") not in ("", "no")
            or ports != {"8080/tcp": [{"HostIp": p.host, "HostPort": str(p.port)}]}
            or c["Config"]["User"] != f"{os.getuid()}:{os.getgid()}"
        ):
            raise ServerLaunchError("Container ID/image/configuration mismatch; preserved")
        if self.generation is not None and c["State"]["StartedAt"] != self.generation:
            raise ServerLaunchError("Container generation changed; preserved")
        if c["State"]["Running"] and self.pid_generation is not None:
            if container_process(c["State"]["Pid"], p.command()) != self.pid_generation:
                raise ServerLaunchError("Container PID generation changed; preserved")
        return c

    def identity(self):
        c = self.verify()
        return {
            "kind": "docker",
            "container_id": self.cid,
            "image_id": c["Image"],
            "started_at": c["State"]["StartedAt"],
            "pid": c["State"]["Pid"],
            "config_fingerprint": fingerprint(self.config),
            "ownership": self.provenance,
            "startup_pending": self.generation is None,
            "pid_generation": self.pid_generation,
        }

    def is_running_for(self, config):
        return self.config == config and bool(self.cid) and self.verify()["State"]["Running"]

    def backend(self, timeout=3600):
        p = self.config
        return ServerBackend(
            f"http://{p.host}:{p.port}",
            api_key_env=p.api_key_env,
            timeout_seconds=timeout,
            model_name=p.model_alias,
            model_path=p.model_path,
            context_size=p.context_size,
            label="managed NInfer",
            strict_transport=True,
        )

    def idle(self):
        r = self.backend()._session.get(
            self.backend().server_url + "/stats", timeout=15, headers=self.backend()._headers()
        )
        r.raise_for_status()
        stats = r.json()
        depth = stats.get("queue", {}).get("depth")
        state = stats.get("scheduler")
        if (
            type(depth) is not int
            or depth != 0
            or not isinstance(state, dict)
            or not state
            or any(type(v) is not int or v != 0 for v in state.values())
        ):
            raise ServerLaunchError("NInfer idle state unconfirmed; lease retained")

    def start(self, config, lease_owner, on_spawn=None, on_identity=None, **kwargs):
        if self.cid:
            if self.owner != lease_owner or not self.is_running_for(config):
                raise ServerLaunchError("NInfer lease/configuration mismatch")
            self.idle()
            return self.backend(kwargs.get("timeout_seconds", 3600))
        for path, sha in (
            (config.model_path, config.model_sha256),
            (str(Path(config.runtime_directory) / "ninfer-serve"), config.binary_sha256),
        ):
            if digest_file(path) != sha:
                raise ServerLaunchError("NInfer artifact SHA256 mismatch")
        self.config, self.owner = config, lease_owner
        if config.adopt_container_id:
            self.cid = config.adopt_container_id
            self.provenance = "administrator_adopted"
            c = self.verify()
            if not c["State"]["Running"]:
                raise ServerLaunchError(
                    "Adoption requires a live instance; stopped containers preserved"
                )
            self.generation = c["State"]["StartedAt"]
            self.pid_generation = container_process(c["State"]["Pid"], config.command())
            self.idle()
            if on_identity:
                on_identity(self.identity())
        else:
            gpu_clear()
            with socket.socket() as sock:
                if sock.connect_ex((config.host, config.port)) == 0:
                    raise ServerLaunchError("Endpoint occupied; external instance preserved")
            name = "lwb-ninfer-" + uuid.uuid4().hex
            self.cid = docker(
                "create",
                "--name",
                name,
                "--label",
                "lwb.managed=" + name,
                "--runtime=nvidia",
                "--env",
                "NVIDIA_VISIBLE_DEVICES=0",
                "--env",
                "KVMEM_WINDOW_TRACE=1",
                "--env",
                "LD_LIBRARY_PATH=/toolkit/lib64",
                "--user",
                f"{os.getuid()}:{os.getgid()}",
                "--ulimit",
                "memlock=-1",
                "--publish",
                f"{config.host}:{config.port}:8080",
                "--log-opt",
                "max-size=20m",
                "--log-opt",
                "max-file=3",
                "--mount",
                f"type=bind,src={config.runtime_directory},dst=/runtime,readonly",
                "--mount",
                f"type=bind,src={Path(config.model_path).parent},dst=/models,readonly",
                "--mount",
                f"type=bind,src={config.toolkit_directory},dst=/toolkit,readonly",
                config.image_id,
                *config.command(),
            )
            self.provenance = "workbench_created"
            self.verify()
            # Durable container ID precedes GPU start, including daemon timeout.
            if on_identity:
                on_identity(self.identity())
            try:
                docker("start", self.cid)
            except Exception as exc:
                raise ServerStartupTimeout(
                    "Docker start outcome unknown; identity retained"
                ) from exc
            self.generation = self.verify()["State"]["StartedAt"]
            self.pid_generation = container_process(self.verify()["State"]["Pid"], config.command())
            if on_identity:
                on_identity(self.identity())
        deadline = time.monotonic() + kwargs.get("wait_seconds", config.startup_timeout_seconds)
        backend = self.backend(kwargs.get("timeout_seconds", 3600))
        while time.monotonic() < deadline:
            if not self.verify()["State"]["Running"]:
                raise ServerLaunchError("NInfer exited during startup; reconcile exact container")
            try:
                r = backend._session.get(
                    backend.server_url + "/health", timeout=3, headers=backend._headers()
                )
                if r.ok and r.json().get("status") == "ok":
                    return backend
            except (requests.RequestException, ValueError):
                pass
            time.sleep(0.25)
        raise ServerStartupTimeout("NInfer readiness timeout; container and GPU lease retained")

    def begin_request(self, owner):
        if owner != self.owner or not self.verify()["State"]["Running"]:
            raise ServerLaunchError("NInfer execution lease mismatch")
        self.idle()
        self.active += 1

    def end_request(self, owner):
        if owner != self.owner or self.active != 1:
            raise ServerLaunchError("NInfer execution lease mismatch")
        self.active -= 1

    def stop(self, lease_owner):
        if self.owner != lease_owner or self.active:
            raise ServerLaunchError("Cannot stop foreign or active NInfer lease")
        c = self.verify()
        if c["State"]["Running"]:
            self.idle()
            self.verify()
            docker("stop", "--time", "60", self.cid)
        if not container_dead(self.identity()):
            raise ServerLaunchError("Container stop unconfirmed; lease retained")
        gpu_clear()
        # Never remove an administrator-adopted container or its data.
        if self.provenance == "workbench_created":
            docker("rm", self.cid)
        self.cid = self.config = self.generation = self.owner = None
        self.pid_generation = None
        return True

    def acknowledge_dead(self, owner):
        if self.owner != owner or not container_dead(
            {
                "container_id": self.cid,
                "startup_pending": self.generation is None,
            }
        ):
            raise ServerLaunchError("Cannot acknowledge live/foreign container")
        self.active = 0
        self.cid = self.config = self.generation = self.owner = None
        self.pid_generation = None

    def status(self):
        if not self.cid:
            return {"pid": None, "running": False, "engine": "ninfer"}
        c = self.verify()
        return {
            "pid": c["State"]["Pid"],
            "running": c["State"]["Running"],
            "engine": "ninfer",
            "instance": self.identity(),
            "active_requests": self.active,
        }


class EngineManager:
    """Keep the scheduler independent of launch arguments and ownership mechanisms."""

    def __init__(self, llama):
        self.llama = llama
        self.ninfer = NInferDocker()
        self.current = llama
        self.config = None

        self.coordinate_gpu = False

    def coordinated(self):
        return self.coordinate_gpu or bool(getattr(self.config, "switch_lock", ""))

    def llama_idle(self):
        if self.current is not self.llama or not self.coordinated():
            return
        if not self.llama.status().get("pid"):
            return
        backend = ServerBackend(
            server_url=self.llama._endpoint(self.config),
            api_key_env=getattr(self.config, "api_key_env", ""),
        )
        try:
            response = backend._session.get(backend.server_url + "/slots", timeout=10)
            response.raise_for_status()
            slots = response.json()
            if not isinstance(slots, list) or not slots or any(
                not isinstance(slot, dict) or slot.get("is_processing") is not False
                for slot in slots
            ):
                raise ServerLaunchError("llama.cpp slots busy or idle state unconfirmed")
        except Exception as exc:
            raise ServerLaunchError("Cannot confirm llama.cpp idle state") from exc
        finally:
            backend._session.close()

    def begin_request(self, lease_owner):
        self.llama_idle()
        return self.current.begin_request(lease_owner)

    @staticmethod
    def prepare(config):
        if isinstance(config, NInferProfile):
            return config
        alias = getattr(config, "model_alias", "")
        return replace(
            config,
            extra_args=config.extra_args
            + " --no-context-shift --parallel 1"
            + (" --alias " + alias if alias else ""),
        )

    def adapter(self, config):
        return self.ninfer if isinstance(config, NInferProfile) else self.llama

    def is_running_for(self, config):
        return self.current is self.adapter(config) and self.current.is_running_for(config)

    def start(self, config, **kwargs):
        self.current = self.adapter(config)
        self.config = config
        if self.current is self.llama:
            kwargs.pop("on_identity", None)
            if self.coordinated() and not self.current.is_running_for(config):
                gpu_clear()
            for field, path in (
                ("model_sha256", config.model_path),
                ("binary_sha256", config.binary_path),
                ("mmproj_sha256", config.mmproj_path),
            ):
                expected = getattr(config, field, "")
                if expected and digest_file(path) != expected:
                    raise ServerLaunchError(f"{field} mismatch")
        backend = self.current.start(config, **kwargs)
        if isinstance(backend, ServerBackend):
            backend.strict_transport = True
            backend.api_key_env = getattr(config, "api_key_env", "")
            backend._session.headers.update(backend._headers())
        return backend

    def stop(self, lease_owner):
        self.llama_idle()
        result = self.current.stop(lease_owner=lease_owner)
        if self.coordinated() and self.current is self.llama:
            gpu_clear()
        return result

    def identity(self, process_identity, boot_id):
        if self.current is self.ninfer:
            return self.ninfer.identity()
        pid = self.current.status()["pid"]
        return process_identity(pid) or {"pid": pid, "boot_id": boot_id()}

    def snapshot(self, profile, identity):
        config = self.config
        return {
            "engine": getattr(config, "engine", "llama.cpp"),
            "profile": profile,
            "config_fingerprint": fingerprint(config),
            "logical_context": config.context_size,
            "weight_identity": {
                "filename": Path(config.model_path).name,
                "sha256": getattr(config, "model_sha256", None) or None,
            },
            "binary_identity": {"sha256": getattr(config, "binary_sha256", None) or None},
            "instance": identity,
            "source": "runtime_probe",
        }

    def no_instance_created(self):
        # Docker create never starts CUDA work. start is issued only after the
        # full returned ID is bound and durably reported. Unrecorded starts are
        # still quarantined across worker restart, not inferred from this flag.
        if self.current is self.ninfer:
            return self.ninfer.cid is None
        return self.current.status().get("pid") is None

    def __getattr__(self, name):
        return getattr(self.current, name)
