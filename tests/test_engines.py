from dataclasses import asdict, replace
import base64
from io import BytesIO
import json
from types import SimpleNamespace

from PIL import Image
import pytest

from lwb import engines
from lwb.backend import ServerBackend
from lwb.engines import ExecutionGate, NInferDocker, NInferProfile, container_dead
from lwb.multimodal import ninfer_count_body
from lwb.process import ServerLaunchError
from lwb.scheduler import Journal, Scheduler, definitely_dead, guard_executor, load_profiles


@pytest.fixture
def profile():
    return NInferProfile(
        engine="ninfer",
        model_path="/models/test.ninfer",
        model_sha256="a" * 64,
        runtime_directory="/runtime",
        binary_sha256="b" * 64,
        image_id="sha256:" + "c" * 64,
        toolkit_directory="/toolkit",
        switch_lock="/tmp/lwb-test.lock",
    )


def container(profile, cid="d" * 64):
    import os

    return {
        "Id": cid,
        "Image": profile.image_id,
        "State": {"Running": True, "Status": "running", "StartedAt": "generation-1", "Pid": 123},
        "Mounts": [
            {"Destination": dest, "Source": source, "RW": False}
            for dest, source in [
                ("/models", "/models"),
                ("/runtime", "/runtime"),
                ("/toolkit", "/toolkit"),
            ]
        ],
        "Config": {
            "Cmd": profile.command(),
            "Entrypoint": None,
            "User": f"{os.getuid()}:{os.getgid()}",
            "Env": [
                "NVIDIA_VISIBLE_DEVICES=0",
                "KVMEM_WINDOW_TRACE=1",
                "LD_LIBRARY_PATH=/toolkit/lib64",
            ],
        },
        "HostConfig": {
            "Runtime": "nvidia",
            "NetworkMode": "bridge",
            "PortBindings": {"8080/tcp": [{"HostIp": "127.0.0.1", "HostPort": "18200"}]},
            "RestartPolicy": {"Name": "no"},
        },
    }


def test_profile_validation_and_old_profile_compatibility(tmp_path, monkeypatch, profile):
    path = tmp_path / "profiles.json"
    path.write_text(
        json.dumps(
            {"new": asdict(profile), "old": {"binary_path": "llama", "model_path": "m.gguf"}}
        )
    )
    monkeypatch.setenv("LWB_PROFILES_FILE", str(path))
    parsed = load_profiles()
    assert parsed["old"].engine == "llama.cpp"
    assert parsed["new"].context_size == 262144
    assert "--parallel" not in parsed["new"].command()
    assert parsed["new"].command()[parsed["new"].command().index("--max-concurrency") + 1] == "1"
    for change in [
        {"image_id": "image:latest"},
        {"context_size": True},
        {"adopt_container_id": "familiar-name"},
        {"engine": "turbo"},
        {"model_path": "/m.gguf"},
        {"switch_lock": ""},
        {"kvmem_host_mib": 12289},
    ]:
        with pytest.raises(ValueError):
            replace(profile, **change)
    path.write_text(json.dumps({"new": {**asdict(profile), "extra_args": "--privileged"}}))
    with pytest.raises(TypeError):
        load_profiles()


def test_gate_pins_between_requests_and_through_nested_graph(tmp_path):
    import fcntl

    path = tmp_path / "switch.lock"
    gate = ExecutionGate(path)
    with gate:
        gate.pin()
        with gate:
            pass
    external = path.open("a")
    with pytest.raises(BlockingIOError):
        fcntl.flock(external, fcntl.LOCK_EX | fcntl.LOCK_NB)
    with gate:
        gate.unpin()
        with pytest.raises(BlockingIOError):
            fcntl.flock(external, fcntl.LOCK_EX | fcntl.LOCK_NB)
    fcntl.flock(external, fcntl.LOCK_EX | fcntl.LOCK_NB)
    external.close()


def test_external_lock_failure_does_not_kill_native_worker(tmp_path):
    import fcntl

    path = tmp_path / "switch.lock"
    external = path.open("a")
    fcntl.flock(external, fcntl.LOCK_EX | fcntl.LOCK_NB)
    scheduler = SimpleNamespace(gpu=ExecutionGate(path))
    called = []
    guarded = guard_executor(scheduler, lambda *args: called.append(True))
    executor = SimpleNamespace()
    guarded(executor, {"1": {"class_type": "KSampler"}})
    assert not called and not executor.success
    external.close()


@pytest.mark.parametrize("change", ["id", "image", "command", "generation", "mount", "privileged"])
def test_identity_mismatch_never_stops_container(monkeypatch, profile, change):
    c = container(profile)
    obj = NInferDocker()
    obj.cid, obj.config, obj.owner, obj.generation = c["Id"], profile, "owner", "generation-1"
    if change == "id":
        c["Id"] = "e" * 64
    if change == "image":
        c["Image"] = "sha256:" + "e" * 64
    if change == "command":
        c["Config"]["Cmd"] = ["wrong"]
    if change == "generation":
        c["State"]["StartedAt"] = "generation-2"
    if change == "mount":
        c["Mounts"][0]["RW"] = True
    if change == "privileged":
        c["HostConfig"]["Privileged"] = True
    monkeypatch.setattr(engines, "inspect_container", lambda cid: c)
    calls = []
    monkeypatch.setattr(engines, "docker", lambda *a: calls.append(a))
    with pytest.raises(ServerLaunchError):
        obj.stop("owner")
    assert not calls


def test_container_recovery_uses_daemon_not_host_pid(monkeypatch, profile):
    c = container(profile)
    identity = {"kind": "docker", "container_id": c["Id"], "pid": 99999999, "boot_id": "old"}
    monkeypatch.setattr(engines, "inspect_container", lambda cid: c)
    assert not definitely_dead(identity)
    c["State"].update(Running=False, Status="exited")
    assert definitely_dead(identity)
    c["State"]["Status"] = "created"
    assert not definitely_dead({**identity, "startup_pending": True})

    def unavailable(cid):
        raise ServerLaunchError("daemon unreachable")

    monkeypatch.setattr(engines, "inspect_container", unavailable)
    assert not container_dead(identity)


def test_adopted_stop_targets_full_id_and_preserves_container(monkeypatch, profile):
    c = container(profile)
    obj = NInferDocker()
    obj.cid, obj.config, obj.owner, obj.generation = c["Id"], profile, "owner", "generation-1"
    obj.provenance = "administrator_adopted"
    monkeypatch.setattr(engines, "inspect_container", lambda cid: c)
    calls = []

    def run(*args):
        calls.append(args)
        c["State"].update(Running=False, Status="exited")

    monkeypatch.setattr(engines, "docker", run)
    monkeypatch.setattr(engines, "gpu_clear", lambda: None)
    obj.idle = lambda: None
    assert obj.stop("owner")
    assert calls == [("stop", "--time", "60", c["Id"])]


def test_active_container_cannot_stop(monkeypatch, profile):
    obj = NInferDocker()
    obj.owner, obj.active = "owner", 1
    with pytest.raises(ServerLaunchError):
        obj.stop("owner")


def test_strict_backend_no_constraint_dropping_or_endpoint_retry(monkeypatch):
    body = []

    class Session:
        def post(self, url, **kwargs):
            body.append((url, kwargs["json"]))
            return SimpleNamespace(status_code=500, ok=False, text="unsupported schema")

    b = ServerBackend("http://127.0.0.1:1", strict_transport=True)
    b._session = Session()
    monkeypatch.setattr(ServerBackend, "_probe_external_server", lambda self: "")
    with pytest.raises(Exception, match="unsupported schema"):
        b.chat_response(
            [{"role": "user", "content": "x"}],
            max_tokens=10,
            enable_thinking=False,
            chat_template_kwargs={"custom": 1},
            response_format={"type": "json_schema", "json_schema": {"name": "x", "schema": {}}},
        )
    assert len(body) == 1
    assert body[0][1]["chat_template_kwargs"] == {"custom": 1, "enable_thinking": False}
    assert body[0][1]["response_format"]["type"] == "json_schema"


def test_managed_image_boundaries_and_count_translation():
    image = BytesIO()
    Image.new("RGB", (32, 32)).save(image, format="PNG")
    uri = "data:image/png;base64," + base64.b64encode(image.getvalue()).decode()
    payload = {
        "messages": [
            {"role": "system", "content": "system"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe"},
                    {"type": "image_url", "image_url": {"url": uri}},
                ],
            },
        ],
        "chat_template_kwargs": {"enable_thinking": False, "custom": "kept"},
    }
    counted = ninfer_count_body(payload)
    assert counted["input"][1]["content"][1]["image_url"] == uri
    assert counted["chat_template_kwargs"] == payload["chat_template_kwargs"]
    assert counted["truncation"] == "disabled"
    payload["response_format"] = {
        "type": "json_schema",
        "json_schema": {"name": "visual", "schema": {"type": "object"}, "strict": True},
    }
    assert ninfer_count_body(payload)["text"]["format"]["schema"] == {"type": "object"}
    payload["messages"][1]["content"][1]["image_url"]["url"] = "http://private/image.png"
    with pytest.raises(ValueError, match="data URI"):
        ninfer_count_body(payload)


def test_stop_failure_quarantines_handoff(tmp_path):
    class Server:
        def stop(self, **kwargs):
            raise RuntimeError("stop unconfirmed")

    scheduler = Scheduler(Journal(tmp_path / "j.db"), {}, server=Server())
    scheduler.lease_owner = "a"
    scheduler.journal.meta("lease", {"identity": {"boot_id": "unknown"}})
    with pytest.raises(RuntimeError):
        scheduler.before_graph({"1": {"class_type": "KSampler"}})
    assert scheduler.blocked and scheduler.lease_owner == "a"


def test_removed_container_reconciles_without_status_or_pid_guess(tmp_path, monkeypatch, profile):
    p = replace(profile, switch_lock=str(tmp_path / "switch.lock"))
    scheduler = Scheduler(Journal(tmp_path / "j.db"), {"ninfer": p})
    obj = scheduler.engine.ninfer
    obj.cid, obj.config, obj.owner, obj.generation = "d" * 64, p, "a", "old-generation"
    scheduler.engine.current = obj
    scheduler.lease_owner, scheduler.blocked = "a", True
    scheduler.journal.meta("lease", {"identity": {"kind": "docker", "container_id": obj.cid}})
    monkeypatch.setattr(engines, "inspect_container", lambda cid: None)
    assert not scheduler.reconcile()["blocked"]
    assert scheduler.lease_owner is None and scheduler.journal.meta("lease") is None


def test_prelaunch_refusal_does_not_create_phantom_lease(tmp_path, monkeypatch, profile):
    from types import SimpleNamespace

    p = replace(profile, switch_lock=str(tmp_path / "switch.lock"))
    scheduler = Scheduler(Journal(tmp_path / "j.db"), {"ninfer": p})
    scheduler.queue = SimpleNamespace(
        mutex=__import__("threading").RLock(), queue=[], put=lambda item: None
    )
    monkeypatch.setattr(engines, "digest_file", lambda path: "0" * 64)
    scheduler.submit(
        "a", "ninfer", {"messages": [{"role": "user", "content": "x"}], "max_tokens": 1}, "refusal"
    )
    result = scheduler.run("refusal")
    assert result["state"] == "failed"
    assert scheduler.journal.meta("lease") is None
    assert scheduler.lease_owner is None and not scheduler.gpu.pinned


def test_managed_connection_keeps_ipv6_and_independent_auth():
    from lwb.backend import ManagedProfileBackend

    config = SimpleNamespace(
        host="::1",
        port=50003,
        context_size=8192,
        model_path="model.gguf",
        api_key_env="PRIVATE_MODEL_TOKEN",
    )
    scheduler = SimpleNamespace(profiles={"old": config}, release_workflow=lambda: False)
    backend = ManagedProfileBackend(scheduler, "old")
    assert backend.server_url == "http://[::1]:50003"
    assert backend.api_key_env == "PRIVATE_MODEL_TOKEN"


def test_upgrade_preserves_legacy_profile_idempotency_without_rewriting_journal(tmp_path):
    import hashlib
    from lwb.scheduler import ManagedServerProfile, canonical, profile_digest_config

    config = ManagedServerProfile(binary_path="llama", model_path="m.gguf")
    payload = {"messages": [{"role": "user", "content": "x"}], "max_tokens": 1}
    legacy = profile_digest_config(config)
    assert "engine" not in legacy and "switch_lock" not in legacy
    old_digest = hashlib.sha256(canonical(["old", legacy, payload]).encode()).hexdigest()
    journal = Journal(tmp_path / "j.db")
    journal.create("stable", "a", old_digest)
    journal.update("stable", state="completed", response={"preserved": True})
    scheduler = Scheduler(journal, {"old": config})
    assert scheduler.submit("a", "old", payload, "stable")["response"] == {"preserved": True}
    assert (
        journal.db.execute("select digest from requests where id=?", ("stable",)).fetchone()[0]
        == old_digest
    )
    scheduler.profiles["old"] = replace(config, context_size=16384)
    with pytest.raises(ValueError, match="different"):
        scheduler.submit("a", "old", payload, "stable")

@pytest.mark.parametrize('slots', [[{'is_processing': True}], [{}], [], {'slots': []}])
def test_coordinated_llama_requires_observed_idle(monkeypatch, slots):
    from lwb.engines import EngineManager
    from lwb.scheduler import ManagedServerProfile
    stopped = []
    llama = SimpleNamespace(
        status=lambda: {'pid': 42},
        _endpoint=lambda config: 'http://127.0.0.1:1234',
        stop=lambda **kwargs: stopped.append(True),
        begin_request=lambda owner: stopped.append('begin'),
    )
    manager = EngineManager(llama)
    manager.config = ManagedServerProfile(binary_path='llama', model_path='m.gguf')
    manager.coordinate_gpu = True
    response = SimpleNamespace(raise_for_status=lambda: None, json=lambda: slots)
    session = SimpleNamespace(get=lambda *a, **k: response, close=lambda: None)
    monkeypatch.setattr(engines, 'ServerBackend', lambda **kwargs: SimpleNamespace(
        server_url=kwargs['server_url'], _session=session))
    with pytest.raises(ServerLaunchError, match='idle'):
        manager.stop('owner')
    with pytest.raises(ServerLaunchError, match='idle'):
        manager.begin_request('owner')
    assert not stopped


def test_mixed_group_checks_gpu_for_legacy_llama(monkeypatch):
    from lwb.engines import EngineManager
    from lwb.scheduler import ManagedServerProfile
    checked = []
    llama = SimpleNamespace(is_running_for=lambda c: False, start=lambda c, **k: 'backend')
    manager = EngineManager(llama)
    manager.coordinate_gpu = True
    monkeypatch.setattr(engines, 'gpu_clear', lambda: checked.append(True))
    assert manager.start(ManagedServerProfile(binary_path='llama', model_path='m.gguf')) == 'backend'
    assert checked == [True]


def test_idle_observer_ignores_cooperating_external_lock(tmp_path):
    import fcntl
    path = tmp_path / 'switch.lock'
    gate = ExecutionGate(path)
    with path.open('a') as external:
        fcntl.flock(external, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert gate.acquire(blocking=False) is False
        scheduler = Scheduler(Journal(':memory:'), {})
        scheduler.gpu = gate
        scheduler.idle_release()
        assert not scheduler.blocked
    with gate:
        assert gate.depth == 1
