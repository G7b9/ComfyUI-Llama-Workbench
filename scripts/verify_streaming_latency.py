"""Isolated streaming measurements using the managed adapters and shared GPU lock.

Requires an idle ComfyUI with no resident lease, verified by its private journal.
No replay, context fallback, global interrupt or group stop. Unknown outcomes
preserve the exact instance and report; stop further measurements.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sqlite3
import sys
import time
import uuid

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lwb import engines  # noqa: E402
from lwb.engines import EngineManager, ExecutionGate  # noqa: E402
from lwb.process import OwnedServer  # noqa: E402
from lwb.scheduler import Journal, Scheduler, boot_id, load_profiles, process_identity  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles-file", type=Path, required=True)
    parser.add_argument("--profile", action="append", required=True)
    parser.add_argument("--comfy-url", default="http://127.0.0.1:8188")
    parser.add_argument("--comfy-journal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite existing measurements")
    os.environ["LWB_PROFILES_FILE"] = str(args.profiles_file.resolve())
    profiles = load_profiles()
    locks = {profiles[p].switch_lock for p in args.profile}
    if len(locks) != 1 or not next(iter(locks)):
        parser.error("all selected profiles must share the existing deployment switch_lock")
    client = requests.Session()
    client.trust_env = False
    if os.environ.get("LWB_API_TOKEN"):
        client.headers["Authorization"] = "Bearer " + os.environ["LWB_API_TOKEN"]

    def api(path):
        r = client.get(args.comfy_url.rstrip("/") + path, timeout=20)
        r.raise_for_status()
        return r.json()

    status = api("/lwb/v1/status")
    pid = status["worker_pid"]
    expected_worker = process_identity(pid)
    if not expected_worker or status["gpu"]["blocked"]:
        raise RuntimeError("ComfyUI generation unavailable/quarantined")
    report = {
        "started_at": time.time(),
        "worker": expected_worker,
        "source": "this_run_streaming",
        "requests": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        with os.fdopen(
            os.open(args.output, os.O_CREAT | os.O_TRUNC | os.O_WRONLY, 0o600), "w"
        ) as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

    def idle():
        if process_identity(pid) != expected_worker:
            raise RuntimeError("ComfyUI PID generation changed")
        q = api("/queue")
        if q["queue_running"] or q["queue_pending"]:
            raise RuntimeError("Native jobs present; refusing isolated measurement")
        with sqlite3.connect(
            "file:" + str(args.comfy_journal.resolve()) + "?mode=ro", uri=True
        ) as db:
            row = db.execute("SELECT value FROM metadata WHERE key='lease'").fetchone()
            if row and json.loads(row[0]) is not None:
                raise RuntimeError(
                    "Workbench resident/unknown lease present; wait for normal release"
                )

    idle()
    save()
    # The verified native worker may retain only its <=2 GiB CUDA context. Its
    # GPU executions are excluded by this same flock for the measurement span.
    original_clear = engines.gpu_clear
    engines.gpu_clear = lambda: original_clear(allowed_pid=pid)
    manager = EngineManager(OwnedServer())
    preflight = Scheduler(Journal(":memory:"), {}, server=OwnedServer())
    schema = {
        "type": "object",
        "properties": {"status": {"const": "READY"}},
        "required": ["status"],
        "additionalProperties": False,
    }
    with ExecutionGate(next(iter(locks))):
        idle()
        for profile in args.profile:
            config = manager.prepare(profiles[profile])
            report["lease"] = {
                "profile": profile,
                "identity": {"boot_id": boot_id()},
                "state": "starting",
            }
            save()

            def identity(value):
                report["lease"]["identity"] = value
                save()

            try:
                start = time.monotonic()
                backend = manager.start(
                    config,
                    lease_owner="stream-acceptance",
                    retain_on_timeout=True,
                    validate_custom_options=True,
                    wait_seconds=config.startup_timeout_seconds,
                    timeout_seconds=1800,
                    on_identity=identity,
                    on_spawn=lambda p: identity(
                        process_identity(p) or {"pid": p, "boot_id": boot_id()}
                    ),
                )
                identity(manager.identity(process_identity, boot_id))
                report["lease"]["startup_seconds"] = time.monotonic() - start
                for target in [32, 32768]:
                    messages = [{"role": "user", "content": "Return status READY."}]
                    payload = {
                        "messages": messages,
                        "max_tokens": 96,
                        "temperature": 0,
                        "chat_template_kwargs": {"enable_thinking": False},
                        "response_format": {
                            "type": "json_schema",
                            "json_schema": {"name": "latency", "schema": schema},
                        },
                    }
                    if target > 32:
                        repeats = target // 10
                        for _ in range(6):
                            messages[0]["content"] = (
                                " Ordinary ledger line has no new facts.\n" * repeats
                                + "\nReturn status READY."
                            )
                            info = preflight.preflight(backend, payload)
                            if abs(info["input_tokens"] - target) < 64:
                                break
                            repeats = max(1, int(repeats * target / info["input_tokens"]))
                    info = preflight.preflight(backend, payload)
                    rid = "stream-acceptance-" + uuid.uuid4().hex
                    item = {
                        "request_id": rid,
                        "profile": profile,
                        "target_input": target,
                        "startup_seconds": report["lease"]["startup_seconds"],
                        "preflight": info,
                        "runtime": manager.snapshot(
                            profile, manager.identity(process_identity, boot_id)
                        ),
                        "state": "running",
                        "raw_events": [],
                    }
                    report["requests"].append(item)
                    report["lease"]["state"] = "running"
                    save()
                    manager.begin_request("stream-acceptance")
                    started = time.monotonic()
                    body = {
                        **payload,
                        "model": info["model_id"],
                        "stream": True,
                        "stream_options": {"include_usage": True},
                    }
                    response = backend._session.post(
                        backend.server_url + "/v1/chat/completions",
                        json=body,
                        stream=True,
                        timeout=1800,
                    )
                    item["http_status"] = response.status_code
                    if not response.ok:
                        item.update(state="failed", error_body=response.text)
                    else:
                        texts, reasons, usage, done = [], [], None, False
                        for line in response.iter_lines(chunk_size=1, decode_unicode=True):
                            if not line or not line.startswith("data:"):
                                continue
                            event = line[5:].strip()
                            item["raw_events"].append(event)
                            if event == "[DONE]":
                                done = True
                                break
                            chunk = json.loads(event)
                            if chunk.get("usage"):
                                usage = chunk["usage"]
                            for choice in chunk.get("choices", []):
                                delta = choice.get("delta", {})
                                content = delta.get("content") or ""
                                reasoning = delta.get("reasoning_content") or ""
                                if (content or reasoning) and "first_token_seconds" not in item:
                                    item["first_token_seconds"] = time.monotonic() - started
                                if content:
                                    texts.append(content)
                                if choice.get("finish_reason"):
                                    reasons.append(choice["finish_reason"])
                        if not done:
                            raise RuntimeError("Stream ended without DONE; outcome unknown")
                        item.update(
                            state="completed",
                            elapsed_seconds=time.monotonic() - started,
                            content="".join(texts),
                            finish_reason=reasons,
                            usage=usage,
                            usage_matches_preflight=bool(usage)
                            and usage["prompt_tokens"] == info["input_tokens"],
                        )
                        item["schema_valid"] = json.loads(item["content"]) == {"status": "READY"}
                        if usage and item.get("first_token_seconds") is not None:
                            duration = item["elapsed_seconds"] - item["first_token_seconds"]
                            item["observed_decode_tokens_per_second"] = (
                                (usage["completion_tokens"] - 1) / duration
                                if duration > 0
                                else None
                            )
                    response.close()
                    manager.end_request("stream-acceptance")
                    report["lease"]["state"] = "idle"
                    save()
                    print(
                        profile,
                        target,
                        item["state"],
                        "TTFT",
                        item.get("first_token_seconds"),
                        flush=True,
                    )
                manager.stop(lease_owner="stream-acceptance")
                report["lease"] = None
                save()
            except Exception as exc:
                report["error"] = {
                    "kind": type(exc).__name__,
                    "message": str(exc),
                    "instance_retained": True,
                    "no_replay": True,
                }
                if report["requests"] and report["requests"][-1]["state"] == "running":
                    report["requests"][-1]["state"] = "result_unknown"
                save()
                raise
    report["finished_at"] = time.time()
    report["worker_unchanged"] = process_identity(pid) == expected_worker
    save()


if __name__ == "__main__":
    main()
