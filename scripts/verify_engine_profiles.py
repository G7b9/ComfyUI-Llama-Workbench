"""Real native-worker profile acceptance; private reports, one submission per ID.

Requires configured ComfyUI and optional administrator-exported director schemas.
No service stopping, retries, context changes, launcher calls or failure injection.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import threading
import time
import uuid
import subprocess

import requests
from jsonschema import Draft202012Validator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8188")
    parser.add_argument("--model-url", required=True)
    parser.add_argument("--profile", action="append", required=True)
    parser.add_argument("--schema-dir", type=Path)
    parser.add_argument("--schema-fixture-dir", type=Path)
    parser.add_argument(
        "--schemas-only",
        action="store_true",
        help="Probe schema files after a short warmup; skip long-input cases",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--long-profile", default="gsq")
    parser.add_argument("--timeout", type=float, default=1800)
    args = parser.parse_args()
    session = requests.Session()
    session.trust_env = False
    if os.environ.get("LWB_API_TOKEN"):
        session.headers["Authorization"] = "Bearer " + os.environ["LWB_API_TOKEN"]
    model = requests.Session()
    model.trust_env = False
    if os.environ.get("LWB_MODEL_API_TOKEN"):
        model.headers["Authorization"] = "Bearer " + os.environ["LWB_MODEL_API_TOKEN"]
    report = {"started_at": time.time(), "source": "this_run", "requests": [], "profiles": {}}
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        temporary = args.output.with_suffix(".tmp")
        with os.fdopen(os.open(temporary, os.O_CREAT | os.O_TRUNC | os.O_WRONLY, 0o600), "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        temporary.replace(args.output)

    def api(method, path, body=None):
        response = session.request(method, args.base_url.rstrip("/") + path, json=body, timeout=30)
        response.raise_for_status()
        return response.json()

    report["initial_status"] = api("GET", "/lwb/v1/status")
    stop = threading.Event()
    samples = args.output.with_suffix(".samples.jsonl")

    def monitor():
        with samples.open("a") as stream:
            while not stop.is_set():
                sample = {"timestamp": time.time(), "phase": report.get("phase")}
                try:
                    result = subprocess.run(
                        [
                            "nvidia-smi",
                            "--query-gpu=memory.used,utilization.gpu,power.draw",
                            "--format=csv,noheader,nounits",
                        ],
                        capture_output=True,
                        text=True,
                        timeout=10,
                    )
                    sample["gpu"] = result.stdout.strip() if result.returncode == 0 else None
                    result = subprocess.run(
                        [
                            "nvidia-smi",
                            "--query-compute-apps=pid,used_gpu_memory",
                            "--format=csv,noheader,nounits",
                        ],
                        capture_output=True,
                        text=True,
                        timeout=10,
                    )
                    sample["compute"] = result.stdout.strip()
                    pids = [
                        int(line.split(",")[0])
                        for line in result.stdout.strip().splitlines()
                        if line
                    ]
                    sample["memory_kib"] = {}
                    for pid in pids:
                        try:
                            fields = Path(f"/proc/{pid}/smaps_rollup").read_text().splitlines()
                            sample["memory_kib"][str(pid)] = {
                                line.split(":")[0]: int(line.split()[1])
                                for line in fields
                                if line.startswith(("Rss:", "Pss:"))
                            }
                        except (OSError, ValueError):
                            sample["memory_kib"][str(pid)] = None
                except (OSError, subprocess.SubprocessError):
                    sample["sampling_error"] = True
                stream.write(json.dumps(sample) + "\n")
                stream.flush()
                stop.wait(0.5)

    worker = threading.Thread(target=monitor, daemon=True)
    worker.start()

    def submit(profile, label, payload, schema=None):
        rid = "engine-acceptance-" + uuid.uuid4().hex
        item = {"request_id": rid, "profile": profile, "case": label, "submitted_at": time.time()}
        report["requests"].append(item)
        report["phase"] = profile + ":" + label
        save()  # ID durable before HTTP side effect; never replay if acknowledgement is lost.
        started = time.monotonic()
        record = api(
            "POST", "/lwb/v1/requests", {"request_id": rid, "profile": profile, "payload": payload}
        )
        seen = {}
        while record["state"] not in {"completed", "failed", "cancelled", "result_unknown"}:
            seen.setdefault(record["state"], time.monotonic() - started)
            if time.monotonic() - started > args.timeout:
                raise TimeoutError(f"{rid}: result unknown; no replay")
            time.sleep(0.25)
            record = api("GET", "/lwb/v1/requests/" + rid)
        item.update(result=record, elapsed_seconds=time.monotonic() - started, observed_states=seen)
        item["first_token_seconds"] = None
        item["first_token_source"] = "not_observable_in_nonstream_queue_response"
        if record["state"] == "completed":
            raw = record["response"]
            item["server_timings"] = raw.get("timings")
            if schema:
                try:
                    parsed = json.loads(raw["choices"][0]["message"]["content"])
                    Draft202012Validator(schema).validate(parsed)
                    item["schema_valid"] = True
                except Exception as exc:
                    item["schema_valid"] = False
                    item["schema_error"] = str(exc)[:1000]
            item["usage_matches_preflight"] = (
                raw.get("usage", {}).get("prompt_tokens") == record["model"]["input_tokens"]
            )
        save()
        print(
            profile,
            label,
            record["state"],
            round(item["elapsed_seconds"], 2),
            "schema=" + str(item.get("schema_valid")),
            flush=True,
        )
        if record["state"] == "result_unknown":
            raise RuntimeError("Unknown inference outcome; matrix stopped, GPU lease preserved")
        return item

    def payload(text, fmt=None, maximum=256):
        result = {
            "messages": [{"role": "user", "content": text}],
            "temperature": 0,
            "max_tokens": maximum,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if fmt:
            result["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "acceptance", "schema": fmt},
            }
        return result

    def counted_document(target):
        evidence = [
            ("FACT-001", "The cedar archive opened on 14 March 2041."),
            ("FACT-002", "The copper key belonged to Mira Chen."),
            ("FACT-003", "The final ledger recorded exactly 73 lanterns."),
        ]
        instruction = (
            "Extract the three FACT identifiers and their exact verbatim sentences, in order. "
            "Return only a JSON object with facts array, each containing id and evidence.\n"
        )

        def text(repetitions):
            third = repetitions // 3
            filler = " Ordinary archival filler contains no factual evidence.\n"
            return instruction + "".join(
                f"\n{key}: {sentence}\n" + filler * third for key, sentence in evidence
            )

        def count(content):
            r = model.post(
                args.model_url.rstrip("/") + "/apply-template",
                json={
                    "messages": [{"role": "user", "content": content}],
                    "chat_template_kwargs": {"enable_thinking": False},
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {"name": "acceptance", "schema": schema},
                    },
                },
                timeout=30,
            )
            r.raise_for_status()
            r = model.post(
                args.model_url.rstrip("/") + "/tokenize",
                json={"content": r.json()["prompt"], "add_special": True, "parse_special": True},
                timeout=30,
            )
            r.raise_for_status()
            return len(r.json()["tokens"])

        repeats = target // 10
        for _ in range(6):
            content = text(repeats)
            n = count(content)
            if abs(n - target) < 64:
                return content, n, evidence
            repeats = max(3, int(repeats * target / n))
        return content, n, evidence

    schema = {
        "type": "object",
        "properties": {
            "facts": {
                "type": "array",
                "minItems": 3,
                "maxItems": 3,
                "items": {
                    "type": "object",
                    "properties": {"id": {"type": "string"}, "evidence": {"type": "string"}},
                    "required": ["id", "evidence"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["facts"],
        "additionalProperties": False,
    }
    try:
        for profile in args.profile:
            short = submit(profile, "short-text", payload("Reply with exactly: READY", maximum=64))
            if short["result"]["state"] != "completed":
                report["profiles"][profile] = {"startup": "failed"}
                continue
            report["profiles"][profile] = {"runtime": short["result"].get("runtime")}
            probes = {}
            for path in ("/health", "/v1/models", "/props", "/stats", "/slots"):
                r = model.get(args.model_url.rstrip("/") + path, timeout=20)
                try:
                    data = r.json()
                except ValueError:
                    data = r.text[:200]
                probes[path] = {"status": r.status_code, "body": data}
            report["profiles"][profile]["runtime_probes"] = probes
            if not args.schemas_only:
                submit(
                    profile,
                    "standard-schema",
                    payload("FACT-001: A. FACT-002: B. FACT-003: C. Extract facts.", schema),
                    schema,
                )
                # Count long input while its server is resident, before other cases can idle it out.
                for target in [32768, 130048] if profile == args.long_profile else [32768]:
                    content, tokens, evidence = counted_document(target)
                    p = payload(content, schema, maximum=512)
                    p["input_tokens"] = tokens
                    item = submit(profile, f"input-{target}", p, schema)
                    if item["result"]["state"] == "completed":
                        try:
                            facts = json.loads(
                                item["result"]["response"]["choices"][0]["message"]["content"]
                            )["facts"]
                            item["verbatim_facts_valid"] = [
                                (f["id"], f["evidence"]) for f in facts
                            ] == evidence
                        except (KeyError, ValueError, TypeError):
                            item["verbatim_facts_valid"] = False
                    save()
            if args.schema_dir:
                for file in sorted(args.schema_dir.glob("*.json")):
                    supplied = json.loads(file.read_text())
                    instruction = (
                        "Synthetic acceptance only: Mira opens a cedar archive, finds a copper key, and records "
                        "73 lanterns. Produce a minimal object matching the supplied schema. No unrelated facts."
                    )
                    if args.schema_fixture_dir:
                        fixture = json.loads((args.schema_fixture_dir / file.name).read_text())
                        Draft202012Validator(supplied).validate(fixture)
                        instruction = (
                            "Return exactly this JSON object, no additions:\n"
                            + json.dumps(fixture, ensure_ascii=False)
                        )
                    submit(
                        profile,
                        "director-schema-" + file.stem,
                        payload(
                            instruction,
                            supplied,
                            maximum=2048,
                        ),
                        supplied,
                    )
        report["final_status"] = api("GET", "/lwb/v1/status")
        report["finished_at"] = time.time()
        save()
    finally:
        stop.set()
        worker.join(15)
        save()


if __name__ == "__main__":
    main()
