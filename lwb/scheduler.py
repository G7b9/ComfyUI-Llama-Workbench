"""Durable text jobs executed by the *native ComfyUI prompt worker*.

No HTTP handler loads a model. A job is one output node, so the native queue
provides FIFO arbitration with image graphs for the entire inference lifetime.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import heapq
import ipaddress
import json
import math
import logging
import os
from pathlib import Path
import re
import secrets
import sqlite3
import threading
import time
import uuid

import requests

from .process import OWNED_SERVER, ServerLaunchConfig, ServerStartupTimeout

NODE = "LlamaWorkbench_QueuedText"
TERMINAL = {"completed", "failed", "cancelled", "result_unknown"}


class CancelledBeforeInference(Exception):
    """Targeted cancellation observed before any generation submission."""


class CapabilityError(ValueError):
    def __init__(self, path, response):
        super().__init__(f"Capability probe {path}: HTTP {response.status_code}")
        try:
            body = response.json()
        except ValueError:
            body = response.text
        self.detail = {
            "kind": "capability_http",
            "endpoint": path,
            "status": response.status_code,
            "body": body,
        }


def canonical(value):
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def validate_payload(payload):
    if not isinstance(payload, dict):
        raise ValueError("body must be an object")
    allowed = {
        "messages",
        "temperature",
        "max_tokens",
        "chat_template_kwargs",
        "response_format",
        "input_tokens",
    }
    if set(payload) - allowed:
        raise ValueError("unsupported fields: " + ", ".join(sorted(set(payload) - allowed)))
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a nonempty array")
    for message in messages:
        if not isinstance(message, dict) or set(message) - {"role", "content", "name"}:
            raise ValueError("supported message fields: role, content, name")
        if message.get("role") not in {"system", "user", "assistant"}:
            raise ValueError("supported roles: system, user, assistant")
        content = message.get("content")
        if isinstance(content, list):
            if not content or any(
                not isinstance(p, dict)
                or set(p) != {"type", "text"}
                or p["type"] != "text"
                or not isinstance(p["text"], str)
                for p in content
            ):
                raise ValueError("text endpoint supports only text content parts")
        elif not isinstance(content, str):
            raise ValueError("message content must be text or text parts")
        if "name" in message and not isinstance(message["name"], str):
            raise ValueError("message name must be a string")
    temperature = payload.get("temperature", 0.7)
    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or not math.isfinite(temperature)
        or not 0 <= temperature <= 2
    ):
        raise ValueError("temperature must be finite and in [0, 2]")
    maximum = payload.get("max_tokens")
    if type(maximum) is not int or maximum <= 0:
        raise ValueError("max_tokens must be an explicit positive integer")
    if "input_tokens" in payload and (
        type(payload["input_tokens"]) is not int or payload["input_tokens"] < 0
    ):
        raise ValueError("input_tokens must be a nonnegative integer assertion")
    if "chat_template_kwargs" in payload and not isinstance(payload["chat_template_kwargs"], dict):
        raise ValueError("chat_template_kwargs must be an object")
    fmt = payload.get("response_format")
    if fmt is not None:
        if not isinstance(fmt, dict) or fmt.get("type") not in {
            "text",
            "json_object",
            "json_schema",
        }:
            raise ValueError("unsupported response_format")
        if fmt["type"] == "json_schema":
            schema = fmt.get("json_schema")
            if (
                set(fmt) != {"type", "json_schema"}
                or not isinstance(schema, dict)
                or set(schema) - {"name", "description", "schema", "strict"}
            ):
                raise ValueError("invalid json_schema envelope")
            if (
                not isinstance(schema.get("name"), str)
                or not schema["name"]
                or not isinstance(schema.get("schema"), dict)
            ):
                raise ValueError("json_schema requires name and schema object")
            if "strict" in schema and type(schema["strict"]) is not bool:
                raise ValueError("json_schema.strict must be boolean")
            from jsonschema import Draft202012Validator

            try:
                Draft202012Validator.check_schema(schema["schema"])
            except Exception as exc:
                raise ValueError("invalid JSON schema") from exc
        elif set(fmt) != {"type"}:
            raise ValueError("unexpected response_format fields")
    canonical(payload)  # reject NaN and non-JSON template values
    return payload


@dataclass(frozen=True, slots=True)
class ManagedServerProfile(ServerLaunchConfig):
    """Administrator-only scheduling policy; never read from an HTTP payload."""

    startup_timeout_seconds: float = 600.0

    def __post_init__(self):
        ServerLaunchConfig.__post_init__(self)
        value = self.startup_timeout_seconds
        if type(value) not in (int, float) or not math.isfinite(value) or not 1 <= value <= 1800:
            raise ValueError("startup_timeout_seconds must be a finite number in [1, 1800]")


# Each option takes exactly one value. These are the explicitly supported
# managed-profile variants, not a claim that every llama.cpp build supports them.
CUSTOM_OPTION_VALUES = {
    "--spec-type": {"draft-mtp"},
    "--spec-draft-type-k": {"turbo3", "turbo4"},
    "--spec-draft-type-v": {"turbo3", "turbo4"},
    "--fit": {"on", "off"},
}


def load_profiles():
    """Only administrator-installed profiles can choose executable or argv."""
    filename = os.environ.get("LWB_PROFILES_FILE")
    if not filename:
        return {}
    raw = json.loads(Path(filename).read_text())
    if not isinstance(raw, dict):
        raise ValueError("LWB_PROFILES_FILE must contain an object")
    profiles = {}
    from .process import _parse_extra_args

    for name, value in raw.items():
        config = ManagedServerProfile(**value)
        if config.host not in {"127.0.0.1", "::1", "localhost"}:
            raise ValueError("managed llama-server profiles must bind loopback")
        # Explicit build flags with fixed arity; no lifecycle/context override,
        # verbose prompt logging, external URL loading, or executable injection.
        args = _parse_extra_args(config.extra_args)
        allowed = {
            "--flash-attn",
            "--cache-type-k",
            "--cache-type-v",
            "--batch-size",
            "--ubatch-size",
            "--threads",
            "--threads-batch",
            "--chat-template",
            "--reasoning-format",
            "--spec-draft-n-max",
            *CUSTOM_OPTION_VALUES,
        }
        seen = set()
        while args:
            flag = args.pop(0)
            if flag not in allowed or not args or args[0].startswith("--"):
                raise ValueError(f"unsupported or incomplete explicit build option: {flag}")
            if flag in seen:
                raise ValueError(f"duplicate explicit build option: {flag}")
            seen.add(flag)
            argument = args.pop(0)
            if flag in CUSTOM_OPTION_VALUES and argument not in CUSTOM_OPTION_VALUES[flag]:
                raise ValueError(
                    f"unsupported value for {flag}; allowed: {sorted(CUSTOM_OPTION_VALUES[flag])}"
                )
            if flag == "--spec-draft-n-max" and (
                not re.fullmatch(r"[0-9]+", argument) or not 1 <= int(argument) <= 64
            ):
                raise ValueError("--spec-draft-n-max must be an integer in [1, 64]")
        profiles[name] = config
    return profiles


def boot_id():
    try:
        return Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    except OSError:
        return "unknown"


def process_identity(pid):
    try:
        # comm may contain spaces and parentheses.
        stat = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return {"pid": pid, "start_ticks": stat[19], "boot_id": boot_id()}
    except (OSError, IndexError):
        return None


def definitely_dead(identity):
    if not identity:
        return False
    boot = boot_id()
    if boot != "unknown" and identity.get("boot_id") != boot:
        return True
    if identity.get("pid") and identity.get("start_ticks"):
        now = process_identity(identity["pid"])
        return (
            now is not None
            and now != identity
            or (
                now is None
                and not Path(f"/proc/{identity['pid']}").exists()
                and Path("/proc/self/stat").exists()
            )
        )
    return False


class Journal:
    def __init__(self, path):
        self.lock = threading.RLock()
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS requests (id TEXT PRIMARY KEY, owner TEXT, digest TEXT, record TEXT)"
        )
        self.db.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT)")
        self.db.commit()

    def get(self, request_id, owner=None):
        with self.lock:
            row = self.db.execute(
                "SELECT owner,record FROM requests WHERE id=?", (request_id,)
            ).fetchone()
            if not row or (owner is not None and row[0] != owner):
                raise KeyError(request_id)
            return json.loads(row[1])

    def update(self, request_id, **changes):
        with self.lock, self.db:
            record = self.get(request_id)
            record.update(changes, updated_at=time.time())
            self.db.execute(
                "UPDATE requests SET record=? WHERE id=?", (canonical(record), request_id)
            )
            return record

    def create(self, request_id, owner, digest):
        with self.lock, self.db:
            row = self.db.execute(
                "SELECT owner,digest,record FROM requests WHERE id=?", (request_id,)
            ).fetchone()
            if row:
                if row[0] != owner or row[1] != digest:
                    raise ValueError("request_id already exists with different owner or payload")
                return json.loads(row[2]), False
            record = dict(
                request_id=request_id,
                state="accepted",
                created_at=time.time(),
                updated_at=time.time(),
                cancel_requested=False,
            )
            self.db.execute(
                "INSERT INTO requests VALUES (?,?,?,?)",
                (request_id, owner, digest, canonical(record)),
            )
            return record, True

    def meta(self, key, value=...):
        with self.lock, self.db:
            if value is not ...:
                self.db.execute(
                    "INSERT OR REPLACE INTO metadata VALUES (?,?)", (key, canonical(value))
                )
                return value
            row = self.db.execute("SELECT value FROM metadata WHERE key=?", (key,)).fetchone()
            return json.loads(row[0]) if row else None

    def recover(self):
        with self.lock:
            for (rid,) in self.db.execute("SELECT id FROM requests").fetchall():
                record = self.get(rid)
                if record["state"] not in TERMINAL:
                    self.update(
                        rid,
                        state="result_unknown",
                        error={
                            "kind": "worker_restart",
                            "message": "Not replayed; original execution outcome is unknown",
                        },
                    )


class Scheduler:
    def __init__(
        self, journal, profiles, server=OWNED_SERVER, release_models=None, idle_seconds=30
    ):
        self.journal = journal
        self.profiles = profiles
        self.server = server
        self.release_models = release_models or (lambda: None)
        self.idle_seconds = idle_seconds
        self.gpu = threading.RLock()
        self.lock = threading.RLock()
        self.payloads = {}
        self.queue = None
        self.last_used = 0.0
        self.lease_owner = None
        self.blocked = bool(journal.meta("lease"))
        journal.recover()
        self.reconcile()

    def reconcile(self):
        with self.gpu:
            lease = self.journal.meta("lease")
            if self.lease_owner:
                self.server.status()  # poll/reap an exited child before reading /proc
            if self.blocked and lease and definitely_dead(lease.get("identity")):
                if self.lease_owner:
                    self.server.acknowledge_dead(self.lease_owner)
                    self.lease_owner = None
                self.journal.meta("lease", None)
                self.blocked = False
            return {"blocked": self.blocked, "lease": self.journal.meta("lease")}

    def submit(self, owner, profile, payload, request_id=None):
        validate_payload(payload)
        if profile not in self.profiles:
            raise ValueError("unknown administrator profile")
        rid = str(uuid.uuid4()) if request_id is None else request_id
        if not isinstance(rid, str) or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", rid):
            raise ValueError("invalid request_id")
        digest = hashlib.sha256(
            canonical([profile, asdict(self.profiles[profile]), payload]).encode()
        ).hexdigest()
        with self.lock:
            record, created = self.journal.create(rid, owner, digest)
            if not created:
                return record
            if self.blocked:
                return self.journal.update(rid, state="failed", error={"kind": "gpu_quarantined"})
            if self.queue is None:
                return self.journal.update(rid, state="failed", error={"kind": "queue_unavailable"})
            self.payloads[rid] = (owner, profile, json.loads(canonical(payload)))
            prompt = {"1": {"class_type": NODE, "inputs": {"request_id": rid}}}
            # Append after ALL currently pending native jobs, even explicit priorities.
            # One text job per queue item bounds a continuous text batch at one
            # once an image is waiting. Never monopolize the worker with a batch.
            with self.queue.mutex:
                priority = max([0] + [item[0] + 1 for item in self.queue.queue])
                self.journal.update(rid, state="queued", prompt_id="lwb-" + rid)
                try:
                    self.queue.put((priority, "lwb-" + rid, prompt, {}, ["1"], {}))
                except Exception:
                    self.payloads.pop(rid, None)
                    self.journal.update(
                        rid, state="result_unknown", error={"kind": "enqueue_uncertain"}
                    )
                    raise
            return self.journal.get(rid)

    def cancel(self, owner, rid):
        with self.lock:
            record = self.journal.get(rid, owner)
            if record["state"] in TERMINAL:
                return record
            # A dequeued node checks the journal before starting, closing the race.
            if record["state"] in {"accepted", "queued"}:
                self.queue.delete_queue_item(lambda item: item[1] == "lwb-" + rid)
                self.payloads.pop(rid, None)
                return self.journal.update(
                    rid, state="cancelled", cancellation="queue", cancel_requested=True
                )
            mode = "wait_for_start" if record["state"] == "starting" else "drain_in_flight"
            return self.journal.update(rid, cancel_requested=True, cancellation=mode)

    def release(self):
        if self.blocked:
            raise RuntimeError(
                "GPU lease quarantined: reconcile the unknown request before any model load"
            )
        if self.lease_owner:
            self.server.stop(lease_owner=self.lease_owner)
            self.lease_owner = None
            self.journal.meta("lease", None)

    def before_graph(self, prompt):
        if self.blocked:
            raise RuntimeError("GPU lease quarantined; no automatic replay or model load")
        # All other graphs, including hand-submitted image graphs, must release
        # the managed resident model before executing their first node.
        has_job = any(n.get("class_type") == NODE for n in prompt.values())
        if has_job and len(prompt) != 1:
            raise RuntimeError("Managed text jobs must be standalone queue items")
        if not has_job:
            self.release()

    def install_queue_fairness(self):
        original_put = self.queue.put

        def fair_put(item):
            with self.queue.mutex:
                # Preserve native ordering except while managed text is queued.
                # During that window append arrivals, preventing either class
                # from starving the other through repeated front submissions.
                pending_text = any(str(job[1]).startswith("lwb-") for job in self.queue.queue)
                if pending_text or str(item[1]).startswith("lwb-"):
                    item = (max([item[0]] + [job[0] + 1 for job in self.queue.queue]), *item[1:])
                return original_put(item)

        self.queue.put = fair_put
        original_get = self.queue.get
        last_kind = None

        def fair_get(*args, **kwargs):
            nonlocal last_kind
            with self.queue.mutex:
                # Existing text backlog must not delay an arriving image by an
                # unbounded batch. Alternate classes when both are waiting.
                text_jobs = [j for j in self.queue.queue if str(j[1]).startswith("lwb-")]
                image_jobs = [j for j in self.queue.queue if not str(j[1]).startswith("lwb-")]
                if text_jobs and image_jobs:
                    if last_kind is None:
                        candidates = [min(self.queue.queue)]
                    else:
                        candidates = image_jobs if last_kind == "text" else text_jobs
                    selected = min(candidates)
                    self.queue.queue.remove(selected)
                    priority = min(j[0] for j in self.queue.queue) - 1
                    self.queue.queue.append((priority, *selected[1:]))
                    heapq.heapify(self.queue.queue)
                result = original_get(*args, **kwargs)
                if result is not None:
                    last_kind = "text" if str(result[0][1]).startswith("lwb-") else "image"
                return result

        self.queue.get = fair_get

    def idle_release(self):
        # Called by a daemon; never stop an in-flight job or unknown execution.
        if not self.gpu.acquire(blocking=False):
            return
        try:
            if (
                not self.blocked
                and self.lease_owner
                and time.monotonic() - self.last_used >= self.idle_seconds
            ):
                self.release()
        finally:
            self.gpu.release()

    @staticmethod
    def _json(session, base, path, body=None):
        response = (
            session.get(base + path, timeout=30)
            if body is None
            else session.post(base + path, json=body, timeout=30)
        )
        if not response.ok:
            raise CapabilityError(path, response)
        return response.json()

    def infer(self, backend, payload, on_preflight=lambda info: None):
        session = backend._session  # trust_env=False, including SOCKS proxies
        base = backend.server_url
        models = self._json(session, base, "/v1/models")
        props = self._json(session, base, "/props")
        model_id = models["data"][0]["id"]
        actual_ctx = props.get("default_generation_settings", {}).get("n_ctx")
        if type(actual_ctx) is not int or actual_ctx <= 0:
            raise ValueError("server did not disclose actual n_ctx")
        if actual_ctx < backend.context_size:
            raise ValueError(
                f"actual n_ctx={actual_ctx} is smaller than requested {backend.context_size}; refusing fallback"
            )
        template_body = {"messages": payload["messages"]}
        if "chat_template_kwargs" in payload:
            template_body["chat_template_kwargs"] = payload["chat_template_kwargs"]
        rendered = self._json(session, base, "/apply-template", template_body)
        tokens = self._json(
            session,
            base,
            "/tokenize",
            {"content": rendered["prompt"], "add_special": True, "parse_special": True},
        )["tokens"]
        if not isinstance(tokens, list) or any(type(token) is not int for token in tokens):
            raise ValueError("server returned an invalid token array")
        count = len(tokens)
        if "input_tokens" in payload and payload["input_tokens"] != count:
            raise ValueError(f"input_tokens assertion mismatch: server counted {count}")
        if count + payload["max_tokens"] > actual_ctx:
            raise ValueError(
                f"context overflow: {count} input + {payload['max_tokens']} output > actual n_ctx {actual_ctx}; no truncation"
            )
        info = {
            "model_id": model_id,
            "n_ctx": actual_ctx,
            "input_tokens": count,
            "token_counting": {"available": True, "method": "apply-template + tokenize"},
        }
        on_preflight(info)
        body = {k: v for k, v in payload.items() if k != "input_tokens"}
        body.update(model=model_id, stream=False)
        # One submission, one endpoint, no compatibility retry under any error.
        response = session.post(
            base + "/v1/chat/completions", json=body, timeout=backend.timeout_seconds
        )
        return response, info

    def run(self, rid):
        with self.gpu:
            with self.lock:
                record = self.journal.get(rid)
                if record["state"] in TERMINAL:
                    return record
                if record["state"] != "queued" or rid not in self.payloads:
                    raise RuntimeError("request is not eligible for execution")
                owner, profile, payload = self.payloads.pop(rid)
                self.journal.update(rid, state="starting")
            lease_owner = "api:" + owner
            active = False
            try:
                if self.blocked:
                    raise RuntimeError("GPU lease quarantined")
                original_config = self.profiles[profile]
                config = replace(
                    original_config,
                    extra_args=original_config.extra_args + " --no-context-shift --parallel 1",
                )
                if self.lease_owner and (
                    self.lease_owner != lease_owner or not self.server.is_running_for(config)
                ):
                    self.release()
                if not self.server.is_running_for(config):
                    self.release_models()
                self.lease_owner = lease_owner
                # Write BEFORE spawn: a crash in the spawn window is quarantined.
                prior_identity = (
                    process_identity(self.server.status()["pid"])
                    if self.server.is_running_for(config)
                    else None
                )
                self.journal.meta(
                    "lease",
                    {
                        "owner": owner,
                        "request_id": rid,
                        "identity": prior_identity or {"boot_id": boot_id()},
                    },
                )

                def spawned(pid):
                    self.journal.update(rid, pid=pid)
                    self.journal.meta(
                        "lease",
                        {
                            "owner": owner,
                            "request_id": rid,
                            "identity": process_identity(pid) or {"pid": pid, "boot_id": boot_id()},
                        },
                    )

                budget = gpu_budget()
                self.journal.update(rid, gpu_budget=budget)
                backend = self.server.start(
                    config,
                    wait_seconds=getattr(original_config, "startup_timeout_seconds", 600),
                    retain_on_timeout=True,
                    validate_custom_options=True,
                    timeout_seconds=3600,
                    lease_owner=lease_owner,
                    on_spawn=spawned,
                )
                pid = self.server.status()["pid"]
                self.journal.meta(
                    "lease",
                    {
                        "owner": owner,
                        "request_id": rid,
                        "identity": process_identity(pid) or {"pid": pid, "boot_id": boot_id()},
                    },
                )
                if self.journal.get(rid)["cancel_requested"]:
                    raise CancelledBeforeInference()
                self.server.begin_request(lease_owner)
                active = True
                self.journal.update(rid, state="running", pid=pid)

                def preflight(info):
                    if self.journal.get(rid)["cancel_requested"]:
                        raise CancelledBeforeInference()
                    self.journal.update(rid, model=info)
                    self.journal.meta("model", info)

                response, info = self.infer(backend, payload, preflight)
                # Raw response belongs only to the authenticated result endpoint,
                # never to logs, ComfyUI history or websocket messages.
                try:
                    raw = response.json()
                except ValueError:
                    raw = response.text
                if not response.ok:
                    return self.journal.update(
                        rid,
                        state="failed",
                        model=info,
                        error={"kind": "backend_http", "status": response.status_code, "body": raw},
                    )
                if not isinstance(raw, dict) or not raw.get("choices"):
                    return self.journal.update(
                        rid,
                        state="failed",
                        model=info,
                        error={"kind": "invalid_response", "body": raw},
                    )
                return self.journal.update(
                    rid,
                    state="completed",
                    model=info,
                    response=raw,
                    usage=raw.get("usage"),
                    finish_reason=[c.get("finish_reason") for c in raw["choices"]],
                )
            except CancelledBeforeInference:
                return self.journal.update(rid, state="cancelled", cancellation="before_inference")
            except ServerStartupTimeout as exc:
                self.blocked = True
                return self.journal.update(
                    rid,
                    state="result_unknown",
                    error={"kind": "startup_timeout", "message": str(exc), "lease_retained": True},
                )
            except CapabilityError as exc:
                return self.journal.update(rid, state="failed", error=exc.detail)
            except requests.RequestException as exc:
                self.blocked = True
                return self.journal.update(
                    rid,
                    state="result_unknown",
                    error={
                        "kind": type(exc).__name__,
                        "message": "Transport failed; not replayed; GPU lease retained",
                        "detail": str(exc),
                    },
                )
            except Exception as exc:
                # Failures before HTTP generation have no uncertain inference.
                return self.journal.update(
                    rid, state="failed", error={"kind": type(exc).__name__, "message": str(exc)}
                )
            finally:
                if active and not self.blocked:
                    self.server.end_request(lease_owner)
                self.last_used = time.monotonic()


def gpu_budget():
    try:
        import torch

        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            return {
                "free_bytes": free,
                "total_bytes": total,
                "note": "Live free VRAM includes resident ComfyUI CUDA overhead; no automatic context expansion",
            }
    except Exception:
        pass
    return {
        "free_bytes": None,
        "note": "VRAM measurement unavailable; configured context is never expanded",
    }


SCHEDULER: Scheduler | None = None


class LlamaWorkbenchQueuedText:
    CATEGORY = "Llama Workbench / Backend"
    RETURN_TYPES = ("STRING",)
    FUNCTION = "execute"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"request_id": ("STRING", {})}}

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def execute(self, request_id):
        if SCHEDULER is None:
            raise RuntimeError("Managed text API is not configured")
        result = SCHEDULER.run(request_id)
        return (canonical({"request_id": request_id, "state": result["state"]}),)


def authorize(request, allow_local_ui=False):
    """Reject browser-origin calls; token required off loopback, no proxy trust."""
    if request.headers.get("Origin") and not allow_local_ui:
        raise PermissionError("browser-origin lifecycle requests are disabled")
    if allow_local_ui and request.headers.get("Origin"):
        from urllib.parse import urlparse

        origin = urlparse(request.headers["Origin"])
        if origin.netloc != request.host or origin.hostname not in {
            "localhost",
            "127.0.0.1",
            "::1",
        }:
            raise PermissionError("untrusted lifecycle origin")
    token = os.environ.get("LWB_API_TOKEN", "")
    clients = json.loads(os.environ.get("LWB_API_TOKENS_JSON", "{}"))
    supplied = request.headers.get("Authorization", "")
    if clients:
        for owner, client_token in clients.items():
            if client_token and secrets.compare_digest(supplied, "Bearer " + client_token):
                return owner
        raise PermissionError("authentication required")
    if token:
        if not secrets.compare_digest(supplied, "Bearer " + token):
            raise PermissionError("authentication required")
        return hashlib.sha256(token.encode()).hexdigest()[:24]
    peer = request.transport.get_extra_info("peername") if request.transport else None
    try:
        local = bool(peer) and ipaddress.ip_address(peer[0]).is_loopback
    except ValueError:
        local = False
    if not local:
        raise PermissionError("loopback or bearer authentication required")
    return "local"


def guard_executor(scheduler, original):
    def guarded(executor, prompt, *args, **kwargs):
        rid = None
        jobs = [n for n in prompt.values() if n.get("class_type") == NODE]
        if len(jobs) == 1:
            rid = jobs[0].get("inputs", {}).get("request_id")
        with scheduler.gpu:
            try:
                scheduler.before_graph(prompt)
                return original(executor, prompt, *args, **kwargs)
            except Exception:
                # A native prompt worker may not catch exceptions from execute.
                # Report a failed graph without killing the worker or leaking data.
                executor.success = False
                executor.history_result = {"outputs": {}}
                executor.status_messages = [
                    (
                        "execution_error",
                        {
                            "prompt_id": args[0] if args else kwargs.get("prompt_id"),
                            "exception_message": "LWB guarded execution failed or GPU lease quarantined",
                            "exception_type": "GPUExecutionBlocked",
                        },
                    )
                ]
                return None
            finally:
                if rid:
                    try:
                        record = scheduler.journal.get(rid)
                        if record["state"] == "queued":
                            scheduler.payloads.pop(rid, None)
                            scheduler.journal.update(
                                rid, state="failed", error={"kind": "node_not_executed"}
                            )
                        elif record["state"] in {"starting", "running"}:
                            scheduler.blocked = bool(scheduler.journal.meta("lease"))
                            scheduler.journal.update(
                                rid, state="result_unknown", error={"kind": "node_execution_lost"}
                            )
                    except KeyError:
                        pass

    guarded._lwb_guard = True
    return guarded


def install():
    """Fail closed if the expected native-worker integration cannot be installed."""
    global SCHEDULER
    import execution
    from server import PromptServer
    from aiohttp import web
    import folder_paths

    if getattr(execution.PromptExecutor.execute, "_lwb_guard", False):
        raise RuntimeError("LWB scheduler already installed; hot reload is unsafe")
    instance = PromptServer.instance
    original = execution.PromptExecutor.execute

    def installation_failed(executor, prompt, *args, **kwargs):
        executor.success = False
        executor.history_result = {"outputs": {}}
        executor.status_messages = [
            (
                "execution_error",
                {
                    "exception_type": "LWBInstallationFailed",
                    "exception_message": "LWB scheduler initialization failed; refusing uncoordinated GPU execution",
                },
            )
        ]

    execution.PromptExecutor.execute = installation_failed
    state_dir = Path(
        os.environ.get("LWB_STATE_DIR", str(Path(folder_paths.get_user_directory()) / "lwb"))
    )
    state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    state_dir.chmod(0o700)
    from .nodes import _release_comfy_model_cache

    def release_models():
        if not _release_comfy_model_cache():
            raise RuntimeError(
                "ComfyUI model cache release failed; refusing to load a second GPU model"
            )

    SCHEDULER = Scheduler(
        Journal(state_dir / "requests.sqlite3"), load_profiles(), release_models=release_models
    )
    SCHEDULER.queue = instance.prompt_queue
    SCHEDULER.install_queue_fairness()
    execution.PromptExecutor.execute = guard_executor(SCHEDULER, original)

    async def dispatch(request):
        try:
            owner = authorize(request)
            rid = request.match_info.get("request_id")
            if request.method == "GET":
                if rid:
                    return web.json_response(SCHEDULER.journal.get(rid, owner))
                return web.json_response(
                    {
                        "profiles": list(SCHEDULER.profiles),
                        "worker_pid": os.getpid(),
                        "model": SCHEDULER.journal.meta("model"),
                        "gpu": {"blocked": SCHEDULER.blocked},
                        "policy": {
                            "max_text_batch_when_images_wait": 1,
                            "idle_seconds": SCHEDULER.idle_seconds,
                            "cancel_running": "drain_in_flight",
                        },
                    }
                )
            if rid:
                return web.json_response(SCHEDULER.cancel(owner, rid))
            body = await request.json()
            if not isinstance(body, dict) or set(body) - {"request_id", "profile", "payload"}:
                raise ValueError("supported submission fields: request_id, profile, payload")
            record = SCHEDULER.submit(
                owner, body["profile"], body["payload"], body.get("request_id")
            )
            return web.json_response(record, status=202)
        except PermissionError:
            raise web.HTTPForbidden(text="authentication required")
        except KeyError:
            raise web.HTTPNotFound(text="unknown request or missing required field")
        except (ValueError, TypeError) as exc:
            raise web.HTTPBadRequest(text=str(exc))

    async def reconcile(request):
        try:
            authorize(request)
        except PermissionError:
            raise web.HTTPForbidden(text="authentication required")
        # Never wait on the GPU mutex from the event loop.
        import asyncio

        return web.json_response(await asyncio.to_thread(SCHEDULER.reconcile))

    @web.middleware
    async def protect_lifecycle(request, handler):
        if request.method == "POST" and request.path in {"/prompt", "/api/prompt"}:
            body = await request.json()
            prompt = body.get("prompt", {})
            sensitive_nodes = {"LlamaWorkbench_StartServer", "LlamaWorkbench_StopServer", NODE}
            if isinstance(prompt, dict) and any(
                isinstance(n, dict) and n.get("class_type") == NODE for n in prompt.values()
            ):
                raise web.HTTPForbidden(
                    text="QueuedText is internal; submit through /lwb/v1/requests"
                )
            if isinstance(prompt, dict) and any(
                isinstance(n, dict) and n.get("class_type") in sensitive_nodes
                for n in prompt.values()
            ):
                try:
                    authorize(request, allow_local_ui=True)
                except PermissionError:
                    raise web.HTTPForbidden(
                        text="LWB lifecycle nodes require local or authenticated access"
                    )
                # Even authenticated API clients cannot select arbitrary executables.
                # Legacy loopback UI retains its explicitly configured binary widget.
                peer = request.transport.get_extra_info("peername") if request.transport else None
                local = bool(peer) and ipaddress.ip_address(peer[0]).is_loopback
                if not local:
                    raise web.HTTPForbidden(
                        text="Remote lifecycle nodes are disabled; use administrator profiles at /lwb/v1/requests"
                    )
        return await handler(request)

    instance.app.middlewares.append(protect_lifecycle)

    instance.routes.get("/lwb/v1/status")(dispatch)
    instance.routes.post("/lwb/v1/requests")(dispatch)
    instance.routes.get("/lwb/v1/requests/{request_id}")(dispatch)
    instance.routes.post("/lwb/v1/requests/{request_id}/cancel")(dispatch)
    instance.routes.post("/lwb/v1/reconcile")(reconcile)

    def idle():
        while True:
            time.sleep(1)
            try:
                SCHEDULER.idle_release()
            except Exception:
                # Fail closed if the owned process could not be stopped.
                SCHEDULER.blocked = True

    threading.Thread(target=idle, name="lwb-idle-release", daemon=True).start()
    logging.info(
        "[Llama Workbench] /lwb/v1 routes registered; %d administrator profiles; GPU quarantined=%s",
        len(SCHEDULER.profiles),
        SCHEDULER.blocked,
    )
