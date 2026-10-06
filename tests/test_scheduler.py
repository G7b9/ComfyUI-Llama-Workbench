from __future__ import annotations

import heapq
import json
import math
import threading
from types import SimpleNamespace

import pytest
import requests

from lwb.nodes import (
    LlamaWorkbenchStartServer,
    LlamaWorkbenchStopServer,
    LlamaWorkbenchServerStatus,
)
from lwb.process import OwnedServer, ServerLaunchConfig, ServerLaunchError
from lwb.scheduler import Journal, Scheduler, authorize, load_profiles, validate_payload


PAYLOAD = {"messages": [{"role": "user", "content": "private prompt"}], "max_tokens": 20}


class Queue:
    def __init__(self):
        self.queue = []
        self.mutex = threading.RLock()

    def put(self, item):
        with self.mutex:
            heapq.heappush(self.queue, item)

    def get(self, timeout=None):
        with self.mutex:
            return (heapq.heappop(self.queue), 0) if self.queue else None

    def delete_queue_item(self, predicate):
        with self.mutex:
            self.queue[:] = [item for item in self.queue if not predicate(item)]
            heapq.heapify(self.queue)


class Server:
    def __init__(self):
        self.config = None
        self.pid = 10
        self.active = 0
        self.stops = 0

    def is_running_for(self, config):
        return self.config == config

    def start(self, config, **kwargs):
        if self.config != config:
            self.pid += 1
        self.config = config
        return SimpleNamespace(context_size=config.context_size)

    def status(self):
        return {"pid": self.pid}

    def begin_request(self, owner):
        self.active += 1

    def end_request(self, owner):
        self.active -= 1

    def stop(self, lease_owner):
        assert self.active == 0
        self.stops += 1
        self.config = None


class Response:
    ok = True
    status_code = 200
    text = ""

    def json(self):
        return {
            "choices": [{"message": {"content": '{"scene": "forest"}'}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 6},
        }


@pytest.fixture
def scheduler(tmp_path):
    server = Server()
    obj = Scheduler(
        Journal(tmp_path / "jobs.db"),
        {"text": ServerLaunchConfig("llama-server", "model.gguf")},
        server=server,
    )
    obj.queue = Queue()
    obj.install_queue_fairness()

    def infer(backend, payload, preflight):
        info = {"model_id": "test", "n_ctx": 8192, "input_tokens": 5}
        preflight(info)
        return Response(), info

    obj.infer = infer
    return obj


def test_text_text_image_text_reuses_pid_and_releases_before_image(scheduler):
    for rid in ("a", "b"):
        scheduler.submit("director", "text", PAYLOAD, rid)
        scheduler.run(rid)
    a, b = [scheduler.journal.get(rid) for rid in ("a", "b")]
    assert a["pid"] == b["pid"]
    assert scheduler.server.stops == 0
    assert a["response"]["choices"][0]["message"]["content"] == '{"scene": "forest"}'
    assert a["usage"]["prompt_tokens"] == 5
    scheduler.before_graph({"1": {"class_type": "KSampler"}})
    assert scheduler.server.config is None
    scheduler.submit("director", "text", PAYLOAD, "c")
    assert scheduler.run("c")["pid"] != a["pid"]


def test_idempotency_conflict_and_foreign_cancel(scheduler):
    first = scheduler.submit("alice", "text", PAYLOAD, "a")
    assert scheduler.submit("alice", "text", PAYLOAD, "a") == first
    assert len(scheduler.queue.queue) == 1
    with pytest.raises(ValueError, match="different"):
        scheduler.submit("alice", "text", dict(PAYLOAD, temperature=0), "a")
    with pytest.raises(KeyError):
        scheduler.cancel("bob", "a")
    scheduler.run("a")
    assert scheduler.run("a")["state"] == "completed"
    assert scheduler.server.pid == 11


def test_target_queue_cancel_and_dequeue_race(scheduler):
    for rid in ("a", "b"):
        scheduler.submit("alice", "text", PAYLOAD, rid)
    result = scheduler.cancel("alice", "a")
    assert result["cancellation"] == "queue"
    assert scheduler.run("a")["state"] == "cancelled"
    assert len(scheduler.queue.queue) == 1
    assert scheduler.run("b")["state"] == "completed"


def test_running_cancel_drains_without_global_interrupt(scheduler):
    entered, finish = threading.Event(), threading.Event()
    original = scheduler.infer

    def infer(*args):
        result = original(*args)  # preflight finishes before generation is in flight
        entered.set()
        assert finish.wait(3)
        return result

    scheduler.infer = infer
    scheduler.submit("alice", "text", PAYLOAD, "a")
    thread = threading.Thread(target=scheduler.run, args=("a",))
    thread.start()
    assert entered.wait(3)
    assert scheduler.cancel("alice", "a")["cancellation"] == "drain_in_flight"
    scheduler.idle_release()
    assert scheduler.server.stops == 0
    assert scheduler.server.active == 1
    finish.set()
    thread.join(3)
    assert not thread.is_alive()
    result = scheduler.journal.get("a")
    assert result["cancel_requested"] and result["state"] == "completed"


def test_unknown_transport_holds_lease_blocks_image_and_retry(scheduler):
    def fail(*args):
        raise requests.ReadTimeout("private transport detail")

    scheduler.infer = fail
    scheduler.submit("alice", "text", PAYLOAD, "a")
    assert scheduler.run("a")["state"] == "result_unknown"
    assert scheduler.server.active == 1
    scheduler.last_used = 0
    scheduler.idle_release()
    assert scheduler.server.stops == 0
    with pytest.raises(RuntimeError, match="quarantined"):
        scheduler.before_graph({"1": {"class_type": "KSampler"}})
    assert scheduler.submit("alice", "text", PAYLOAD, "a")["state"] == "result_unknown"
    assert scheduler.run("a")["state"] == "result_unknown"


def test_restart_retains_completed_results_and_never_replays(tmp_path):
    path = tmp_path / "jobs.db"
    journal = Journal(path)
    journal.create("completed", "a", "digest")
    journal.update("completed", state="completed", response={"answer": 1})
    journal.create("running", "a", "digest")
    journal.update("running", state="running")
    from lwb.scheduler import boot_id

    journal.meta("lease", {"identity": {"boot_id": boot_id()}})
    recovered = Scheduler(Journal(path), {})
    assert recovered.blocked
    assert recovered.journal.get("running")["state"] == "result_unknown"
    assert recovered.journal.get("completed")["response"] == {"answer": 1}
    assert not recovered.payloads


def test_idle_release_and_fairness(scheduler):
    scheduler.queue.put((0, "image-before", {}, {}, []))
    scheduler.submit("a", "text", PAYLOAD, "a")
    # Even repeated front-of-queue image requests cannot starve queued text.
    scheduler.queue.put((-100, "image-after", {}, {}, []))
    scheduler.submit("a", "text", PAYLOAD, "b")
    assert [heapq.heappop(scheduler.queue.queue)[1] for _ in range(4)] == [
        "image-before",
        "lwb-a",
        "image-after",
        "lwb-b",
    ]
    scheduler.run("a")
    scheduler.last_used = 0
    scheduler.idle_release()
    assert scheduler.server.stops == 1


def test_distinct_caller_lease_is_released_only_after_completion(scheduler):
    scheduler.submit("a", "text", PAYLOAD, "a")
    scheduler.run("a")
    scheduler.submit("b", "text", PAYLOAD, "b")
    scheduler.run("b")
    assert scheduler.server.stops == 1


@pytest.mark.parametrize(
    "change",
    [
        {"max_tokens": -1},
        {"temperature": float("nan")},
        {"temperature": True},
        {"stream": True},
        {"messages": [{"role": "tool", "content": "x"}]},
        {"chat_template_kwargs": []},
        {"response_format": {"type": "wrong"}},
        {
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "x", "schema": {"type": "invalid"}},
            }
        },
    ],
)
def test_validation_rejects_unsupported_payloads(change):
    with pytest.raises(ValueError):
        validate_payload(dict(PAYLOAD, **change))


def test_structured_payload_preserves_schema_and_template():
    payload = dict(
        PAYLOAD,
        chat_template_kwargs={"enable_thinking": False},
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "scene",
                "strict": True,
                "schema": {"type": "object", "properties": {"scene": {"type": "string"}}},
            },
        },
    )
    assert validate_payload(payload) == payload


def test_controls_are_never_cached():
    for cls in (LlamaWorkbenchStartServer, LlamaWorkbenchStopServer, LlamaWorkbenchServerStatus):
        assert math.isnan(cls.IS_CHANGED())


def test_owned_process_cannot_be_stopped_or_reconfigured_by_foreign_lease():
    server = OwnedServer()
    server._process = SimpleNamespace(poll=lambda: None)
    server._lease_owner = "alice"
    with pytest.raises(ServerLaunchError, match="another"):
        server.stop("bob")
    with pytest.raises(ServerLaunchError, match="another"):
        server.start(ServerLaunchConfig("unused", "unused"), lease_owner="bob")
    server.begin_request("alice")
    with pytest.raises(ServerLaunchError, match="active"):
        server.stop("alice")


def test_auth_loopback_lan_proxy_and_origin(monkeypatch):
    monkeypatch.delenv("LWB_API_TOKEN", raising=False)

    def request(ip, headers=None):
        return SimpleNamespace(
            headers=headers or {}, transport=SimpleNamespace(get_extra_info=lambda key: (ip, 123))
        )

    assert authorize(request("127.0.0.1")) == "local"
    with pytest.raises(PermissionError):
        authorize(request("192.168.1.10", {"X-Forwarded-For": "127.0.0.1"}))
    with pytest.raises(PermissionError):
        authorize(request("127.0.0.1", {"Origin": "https://evil.example"}))
    monkeypatch.setenv("LWB_API_TOKEN", "secret")
    assert authorize(request("192.168.1.10", {"Authorization": "Bearer secret"})) != "local"
    with pytest.raises(PermissionError):
        authorize(request("127.0.0.1"))


def test_profiles_refuse_remote_binding_and_unsafe_flags(tmp_path, monkeypatch):
    path = tmp_path / "profiles.json"
    monkeypatch.setenv("LWB_PROFILES_FILE", str(path))
    for changes in (
        {"host": "0.0.0.0"},
        {"extra_args": "--port 6666"},
        {"extra_args": "--verbose-prompt"},
        {"extra_args": "--api-key secret"},
    ):
        path.write_text(
            json.dumps({"text": dict(binary_path="unused", model_path="unused", **changes)})
        )
        with pytest.raises(ValueError):
            load_profiles()


def test_model_metadata_token_budget_and_raw_passthrough(scheduler):
    calls = []

    class Session:
        def get(self, url, **kwargs):
            if url.endswith("/props"):
                value = {"default_generation_settings": {"n_ctx": 8192}}
            else:
                value = {"data": [{"id": "actual-model"}]}
            return SimpleNamespace(ok=True, json=lambda: value)

        def post(self, url, json, **kwargs):
            calls.append((url, json))
            if url.endswith("/apply-template"):
                value = {"prompt": "templated text"}
            elif url.endswith("/tokenize"):
                value = {"tokens": [1, 2, 3]}
            else:
                return Response()
            return SimpleNamespace(ok=True, json=lambda: value)

    backend = SimpleNamespace(
        _session=Session(),
        server_url="http://192.168.1.2:8080",
        context_size=8192,
        timeout_seconds=120,
    )
    payload = dict(
        PAYLOAD,
        temperature=0,
        input_tokens=3,
        chat_template_kwargs={"enable_thinking": False},
        response_format={"type": "json_object"},
    )
    response, info = Scheduler.infer(scheduler, backend, payload)
    assert info["n_ctx"] == 8192 and info["model_id"] == "actual-model"
    assert info["input_tokens"] == 3
    assert calls[-1][1]["response_format"] == payload["response_format"]
    assert calls[-1][1]["chat_template_kwargs"] == payload["chat_template_kwargs"]
    assert response.json()["usage"]["prompt_tokens"] == 5
    calls.clear()
    with pytest.raises(ValueError, match="overflow"):
        Scheduler.infer(scheduler, backend, dict(PAYLOAD, max_tokens=8192))
    assert not any(url.endswith("/v1/chat/completions") for url, _ in calls)
    backend.context_size = 16384
    with pytest.raises(ValueError, match="refusing fallback"):
        Scheduler.infer(scheduler, backend, PAYLOAD)


def test_native_execution_guard_serializes_image_and_handles_failures(scheduler):
    from lwb.scheduler import guard_executor

    entered, finish, image_entered = threading.Event(), threading.Event(), threading.Event()
    executor = SimpleNamespace()

    def original(executor, prompt, *args, **kwargs):
        image_entered.set()
        assert scheduler.server.config is None

    guarded = guard_executor(scheduler, original)
    original_infer = scheduler.infer

    def infer(*args):
        entered.set()
        assert finish.wait(3)
        return original_infer(*args)

    scheduler.infer = infer
    scheduler.submit("a", "text", PAYLOAD, "a")
    text_thread = threading.Thread(target=scheduler.run, args=("a",))
    text_thread.start()
    assert entered.wait(3)
    image_thread = threading.Thread(
        target=guarded, args=(executor, {"1": {"class_type": "KSampler"}}, "image")
    )
    image_thread.start()
    assert not image_entered.wait(0.05)
    finish.set()
    text_thread.join(3)
    image_thread.join(3)
    assert image_entered.is_set()
    scheduler.blocked = True
    guarded(executor, {"1": {"class_type": "KSampler"}}, "blocked")
    assert executor.success is False
    assert executor.status_messages[0][0] == "execution_error"


def test_native_node_failure_does_not_leave_request_queued(scheduler):
    from lwb.scheduler import guard_executor, NODE

    scheduler.submit("a", "text", PAYLOAD, "a")

    def fails(*args):
        raise RuntimeError("private message must not leak")

    executor = SimpleNamespace()
    guard_executor(scheduler, fails)(
        executor, {"1": {"class_type": NODE, "inputs": {"request_id": "a"}}}, "lwb-a"
    )
    assert scheduler.journal.get("a")["state"] == "failed"
    assert "private" not in str(executor.status_messages)


def test_restart_reconcile_only_after_positive_process_death(tmp_path):
    import os
    from lwb.scheduler import process_identity

    journal = Journal(tmp_path / "restart.db")
    journal.meta("lease", {"identity": process_identity(os.getpid())})
    recovered = Scheduler(journal, {})
    assert recovered.blocked
    assert recovered.reconcile()["blocked"]
    identity = process_identity(os.getpid())
    identity["start_ticks"] = "different-process-generation"
    journal.meta("lease", {"identity": identity})
    assert not recovered.reconcile()["blocked"]


def test_request_journal_does_not_store_prompt_or_credentials(scheduler):
    scheduler.submit("a", "text", PAYLOAD, "a")
    rows = scheduler.journal.db.execute("SELECT * FROM requests").fetchall()
    assert "private prompt" not in str(rows)
    assert "private prompt" not in str(scheduler.queue.queue)


def test_remote_callers_have_distinct_authenticated_leases(monkeypatch):
    monkeypatch.setenv("LWB_API_TOKENS_JSON", '{"director-a":"token-a","director-b":"token-b"}')
    req = SimpleNamespace(headers={"Authorization": "Bearer token-a"})
    assert authorize(req) == "director-a"
    req.headers["Authorization"] = "Bearer token-b"
    assert authorize(req) == "director-b"


@pytest.mark.parametrize("lifecycle_mode", ["false", "true"])
def test_http_routes_submit_query_cancel_and_enforce_auth(tmp_path, monkeypatch, lifecycle_mode):
    import asyncio
    import sys
    from aiohttp import web
    from aiohttp.test_utils import TestClient, TestServer
    import lwb.scheduler as module

    monkeypatch.setenv("LWB_RESTRICT_LIFECYCLE", lifecycle_mode)
    monkeypatch.setenv("LWB_STATE_DIR", str(tmp_path / "state"))
    profiles = tmp_path / "profiles.json"
    profiles.write_text(json.dumps({"text": {"binary_path": "unused", "model_path": "unused"}}))
    monkeypatch.setenv("LWB_PROFILES_FILE", str(profiles))
    monkeypatch.setenv("LWB_API_TOKEN", "test-secret")
    monkeypatch.delenv("LWB_API_TOKENS_JSON", raising=False)
    instance = SimpleNamespace(
        routes=web.RouteTableDef(), app=web.Application(), prompt_queue=Queue()
    )
    fake_execution = SimpleNamespace(
        PromptExecutor=type("Executor", (), {"execute": lambda *a, **kw: None})
    )
    monkeypatch.setitem(sys.modules, "execution", fake_execution)
    monkeypatch.setitem(
        sys.modules, "server", SimpleNamespace(PromptServer=SimpleNamespace(instance=instance))
    )
    monkeypatch.setitem(
        sys.modules, "folder_paths", SimpleNamespace(get_user_directory=lambda: str(tmp_path))
    )
    monkeypatch.setattr(module, "SCHEDULER", None)
    # Test startup installation without leaving a permanent idle daemon.
    monkeypatch.setattr(module.threading.Thread, "start", lambda self: None)
    module.install()
    assert fake_execution.PromptExecutor.execute._lwb_guard
    instance.app.add_routes(instance.routes)

    async def check():
        async with TestClient(TestServer(instance.app)) as client:
            response = await client.get("/lwb/v1/status")
            assert response.status == 403
            headers = {"Authorization": "Bearer test-secret"}
            body = {"request_id": "http-1", "profile": "text", "payload": PAYLOAD}
            response = await client.post("/lwb/v1/requests", json=body, headers=headers)
            assert response.status == 202
            assert (await response.json())["state"] == "queued"
            assert (
                len(instance.prompt_queue.queue[0]) == 6
            )  # native current worker reads sensitive item[5]
            response = await client.post("/lwb/v1/requests", json=body, headers=headers)
            assert response.status == 202 and len(instance.prompt_queue.queue) == 1
            response = await client.get("/lwb/v1/requests/http-1", headers=headers)
            assert (await response.json())["state"] == "queued"
            response = await client.post("/lwb/v1/requests/http-1/cancel", headers=headers)
            assert (await response.json())["state"] == "cancelled"
            assert not instance.prompt_queue.queue
            response = await client.get("/lwb/v1/requests/missing", headers=headers)
            assert response.status == 404
            response = await client.post(
                "/lwb/v1/requests", json=dict(body, binary_path="/bin/sh"), headers=headers
            )
            assert response.status == 400
            response = await client.get(
                "/lwb/v1/status", headers=dict(headers, Origin="https://evil.example")
            )
            assert response.status == 403
            # Internal job nodes cannot be injected via generic /prompt, even
            # by another authenticated client who learned an ID from history.
            response = await client.post(
                "/prompt",
                json={
                    "prompt": {
                        "1": {
                            "class_type": "LlamaWorkbench_QueuedText",
                            "inputs": {"request_id": "http-1"},
                        }
                    }
                },
                headers=headers,
            )
            assert response.status == 403

    asyncio.run(check())


def test_real_owned_http_process_text_text_image_text(tmp_path, monkeypatch):
    """Exercise real PIDs/HTTP without needing a CUDA model or restarting Comfy."""
    import os
    import socket
    import sys
    from lwb.scheduler import guard_executor

    if os.name == "nt":
        pytest.skip("fixture binary uses a POSIX executable shebang")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    binary = tmp_path / "llama-test-server"
    binary.write_text(
        f"#!{sys.executable}\n"
        + """
import sys, json
from http.server import HTTPServer, BaseHTTPRequestHandler
if '--version' in sys.argv:
    print('test llama server')
    sys.exit(0)
port = int(sys.argv[sys.argv.index('--port') + 1])
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args): pass
    def send(self, value):
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(value).encode())
    def do_GET(self):
        if self.path == '/v1/models': self.send({'data': [{'id': 'actual-test-model'}]})
        elif self.path == '/props': self.send({'default_generation_settings': {'n_ctx': 8192}})
        else: self.send({'status': 'ok'})
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        if self.path == '/apply-template': self.send({'prompt': 'template'})
        elif self.path == '/tokenize': self.send({'tokens': [1,2,3]})
        else:
            print('PRIVATE inference log must be discarded', flush=True)
            self.send({'choices': [{'message': {'content': '{"scene":"forest"}'}, 'finish_reason': 'stop'}], 'usage': {'prompt_tokens': 3, 'completion_tokens': 6}, 'custom_build_field': 42})
HTTPServer(('127.0.0.1', port), Handler).serve_forever()
"""
    )
    binary.chmod(0o700)
    model = tmp_path / "test.gguf"
    model.write_bytes(b"test")
    monkeypatch.setenv("ALL_PROXY", "socks5://invalid.invalid:1080")
    monkeypatch.setenv("HTTP_PROXY", "socks5://invalid.invalid:1080")
    server = OwnedServer()
    scheduler = Scheduler(
        Journal(tmp_path / "real.db"),
        {"text": ServerLaunchConfig(str(binary), str(model), port=port)},
        server=server,
    )
    scheduler.queue = Queue()
    pid_before = os.getpid()
    try:
        results = []
        for rid in ("a", "b"):
            scheduler.submit("director", "text", PAYLOAD, rid)
            results.append(scheduler.run(rid))
        assert all(result["state"] == "completed" for result in results)
        assert results[0]["pid"] == results[1]["pid"]
        assert results[0]["response"]["custom_build_field"] == 42
        assert results[0]["model"]["n_ctx"] == 8192

        def image(executor, prompt, *args):
            assert not server.is_running
            executor.success = True

        executor = SimpleNamespace()
        guard_executor(scheduler, image)(executor, {"1": {"class_type": "KSampler"}}, "image-1")
        assert executor.success
        scheduler.submit("director", "text", PAYLOAD, "c")
        last = scheduler.run("c")
        assert last["state"] == "completed"
        assert last["pid"] != results[0]["pid"]
        assert "PRIVATE inference" not in str(server.status()["log_tail"])
        assert os.getpid() == pid_before
    finally:
        scheduler.release()


def test_backlogged_text_and_images_have_bounded_batches(scheduler):
    for index in range(5):
        scheduler.submit("a", "text", PAYLOAD, f"text-{index}")
    first = scheduler.queue.get()[0]
    assert first[1] == "lwb-text-0"
    scheduler.queue.put((100, "image-0", {}, {}, []))
    scheduler.queue.put((101, "image-1", {}, {}, []))
    assert scheduler.queue.get()[0][1] == "image-0"
    assert scheduler.queue.get()[0][1] == "lwb-text-1"
    assert scheduler.queue.get()[0][1] == "image-1"


def test_explicit_empty_request_id_is_rejected(scheduler):
    with pytest.raises(ValueError, match="request_id"):
        scheduler.submit("a", "text", PAYLOAD, "")


def test_http_errors_are_preserved_without_retry(scheduler):
    calls = []

    class Failure(Response):
        ok = False
        status_code = 422

        def json(self):
            return {"error": {"message": "unsupported schema", "code": "unsupported"}}

    def infer(*args):
        calls.append(True)
        return Failure(), {"n_ctx": 8192}

    scheduler.infer = infer
    scheduler.submit("a", "text", PAYLOAD, "a")
    result = scheduler.run("a")
    assert result["error"]["status"] == 422
    assert result["error"]["body"]["error"]["code"] == "unsupported"
    scheduler.run("a")
    assert len(calls) == 1
    assert scheduler.server.active == 0


def test_cancel_during_start_skips_generation(scheduler):
    entered, finish = threading.Event(), threading.Event()
    original_start = scheduler.server.start

    def start(*args, **kwargs):
        entered.set()
        assert finish.wait(3)
        return original_start(*args, **kwargs)

    scheduler.server.start = start
    scheduler.submit("a", "text", PAYLOAD, "a")
    thread = threading.Thread(target=scheduler.run, args=("a",))
    thread.start()
    assert entered.wait(3)
    assert scheduler.cancel("a", "a")["cancellation"] == "wait_for_start"
    finish.set()
    thread.join(3)
    result = scheduler.journal.get("a")
    assert result["state"] == "cancelled"
    assert result["cancellation"] == "before_inference"
    assert scheduler.server.active == 0
    assert "response" not in result


MTP_ARGS = (
    "--cache-type-k turbo4 --cache-type-v turbo3 --flash-attn on "
    "--spec-type draft-mtp --spec-draft-n-max 4 "
    "--spec-draft-type-k turbo4 --spec-draft-type-v turbo3 --fit off"
)


def write_managed_profile(tmp_path, monkeypatch, **overrides):
    path = tmp_path / "profiles.json"
    value = dict(
        binary_path="custom-llama",
        model_path="model.gguf",
        context_size=163840,
        gpu_layers=65,
        mmproj_path="",
        extra_args=MTP_ARGS,
    )
    value.update(overrides)
    path.write_text(json.dumps({"director": value}))
    monkeypatch.setenv("LWB_PROFILES_FILE", str(path))
    return load_profiles()["director"]


def test_verified_mtp_profile_preserves_every_argument(tmp_path, monkeypatch, scheduler):
    profile = write_managed_profile(tmp_path, monkeypatch, startup_timeout_seconds=900)
    scheduler.profiles["director"] = profile
    calls = []
    original = scheduler.server.start

    def start(config, **kwargs):
        calls.append((config, kwargs))
        return original(config, **kwargs)

    scheduler.server.start = start
    scheduler.submit("a", "director", PAYLOAD, "mtp")
    assert scheduler.run("mtp")["state"] == "completed"
    config, kwargs = calls[0]
    assert config.context_size == 163840 and config.gpu_layers == 65
    assert config.mmproj_path == ""
    assert config.extra_args == MTP_ARGS + " --no-context-shift --parallel 1"
    assert kwargs["wait_seconds"] == 900
    assert kwargs["retain_on_timeout"] and kwargs["validate_custom_options"]


@pytest.mark.parametrize(
    "arguments",
    [
        "--spec-type",
        "--spec-type draft",
        "--spec-draft-n-max",
        "--spec-draft-n-max 0",
        "--spec-draft-n-max 65",
        "--spec-draft-n-max 4.5",
        "--spec-draft-n-max -1",
        "--spec-draft-type-k turbo5",
        "--spec-draft-type-v invalid",
        "--spec-draft-type-v",
        "--fit",
        "--fit maybe",
        "--fit off extra",
        "--fit off --fit on",
        "--parallel 1",
        "--spec-type draft-mtp --port 9000",
    ],
)
def test_mtp_profile_rejects_wrong_arity_values_and_lifecycle_overrides(
    tmp_path, monkeypatch, arguments
):
    with pytest.raises(ValueError):
        write_managed_profile(tmp_path, monkeypatch, extra_args=arguments)


@pytest.mark.parametrize("timeout", [0, 1801, -10, True, "900", float("nan"), float("inf")])
def test_profile_timeout_is_bounded_numeric_admin_policy(tmp_path, monkeypatch, timeout):
    with pytest.raises(ValueError, match="startup_timeout_seconds"):
        write_managed_profile(tmp_path, monkeypatch, startup_timeout_seconds=timeout)


def test_old_profile_defaults_to_600_seconds(tmp_path, monkeypatch):
    assert write_managed_profile(tmp_path, monkeypatch).startup_timeout_seconds == 600


def test_startup_timeout_retains_lease_and_never_retries(scheduler):
    from lwb.process import ServerStartupTimeout

    calls = []

    def start(config, **kwargs):
        calls.append(config)
        raise ServerStartupTimeout("Still loading the model")

    scheduler.server.start = start
    scheduler.submit("a", "text", PAYLOAD, "slow")
    result = scheduler.run("slow")
    assert result["state"] == "result_unknown"
    assert result["error"]["kind"] == "startup_timeout"
    assert result["error"]["lease_retained"]
    assert scheduler.blocked and scheduler.journal.meta("lease")
    scheduler.last_used = 0
    scheduler.idle_release()
    assert scheduler.server.stops == 0
    assert scheduler.cancel("a", "slow")["state"] == "result_unknown"
    assert scheduler.submit("a", "text", PAYLOAD, "slow")["state"] == "result_unknown"
    with pytest.raises(RuntimeError, match="quarantined"):
        scheduler.before_graph({"1": {"class_type": "KSampler"}})
    assert len(calls) == 1
