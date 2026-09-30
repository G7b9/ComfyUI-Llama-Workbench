"""Live acceptance: two text jobs -> supplied real image graph -> text.

Never resubmits a generation. IDs are written before each POST so a lost HTTP
response can be reconciled manually with the status API.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
import uuid

import requests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8188")
    parser.add_argument("--profile", required=True)
    parser.add_argument(
        "--image-prompt",
        type=Path,
        required=True,
        help="Actual diffusion graph exported in ComfyUI API format",
    )
    parser.add_argument("--output", type=Path, default=Path("queue-api-acceptance-results.json"))
    parser.add_argument("--timeout", type=float, default=1800)
    args = parser.parse_args()
    session = requests.Session()
    session.trust_env = False
    if os.environ.get("LWB_API_TOKEN"):
        session.headers["Authorization"] = "Bearer " + os.environ["LWB_API_TOKEN"]
    base = args.base_url.rstrip("/")
    report = {"text_ids": [], "image_id": str(uuid.uuid4()), "results": []}

    def save():
        # Output contains model responses; keep the acceptance report private.
        with os.fdopen(
            os.open(args.output, os.O_CREAT | os.O_TRUNC | os.O_WRONLY, 0o600), "w"
        ) as stream:
            json.dump(report, stream, indent=2, ensure_ascii=False)

    def call(method, path, body=None):
        response = session.request(method, base + path, json=body, timeout=30)
        response.raise_for_status()
        return response.json()

    def poll_text(rid, early=False):
        deadline = time.monotonic() + args.timeout
        while time.monotonic() < deadline:
            result = call("GET", "/lwb/v1/requests/" + rid)
            if result["state"] in {"failed", "cancelled", "result_unknown"}:
                report["failure"] = result
                save()
                raise RuntimeError(f"{rid}: {result['state']}; not replayed")
            if result["state"] == "completed" or (
                early and result["state"] in {"starting", "running"}
            ):
                return result
            time.sleep(0.25)
        raise TimeoutError(
            f"{rid}: poll deadline exceeded; execution may continue; do not resubmit"
        )

    def submit_text():
        rid = "acceptance-" + str(uuid.uuid4())
        report["text_ids"].append(rid)
        save()
        call(
            "POST",
            "/lwb/v1/requests",
            {
                "request_id": rid,
                "profile": args.profile,
                "payload": {
                    "messages": [
                        {
                            "role": "user",
                            "content": 'Return a JSON object with a short scene description under key "description".',
                        }
                    ],
                    "max_tokens": 256,
                    "temperature": 0,
                    "response_format": {"type": "json_object"},
                },
            },
        )
        return rid

    original_status = call("GET", "/lwb/v1/status")
    report["worker_pid"] = original_status["worker_pid"]
    first = poll_text(submit_text())
    report["results"].append(first)
    second_id = submit_text()
    poll_text(second_id, early=True)
    image = json.loads(args.image_prompt.read_text())
    graph = image.get("prompt", image)
    save()
    image_submission = call("POST", "/prompt", {"prompt_id": report["image_id"], "prompt": graph})
    report["image_id"] = image_submission["prompt_id"]
    save()
    second = poll_text(second_id)
    report["results"].append(second)
    assert first["pid"] == second["pid"], "consecutive text did not reuse PID"
    deadline = time.monotonic() + args.timeout
    while time.monotonic() < deadline:
        history = call("GET", "/history/" + report["image_id"])
        if report["image_id"] in history:
            report["image_history"] = history[report["image_id"]]
            save()
            assert report["image_history"].get("status", {}).get("completed"), "image graph failed"
            break
        time.sleep(0.5)
    else:
        raise TimeoutError("Image result unknown; not resubmitted")
    third = poll_text(submit_text())
    report["results"].append(third)
    assert third["pid"] != second["pid"], "text PID did not change after image graph"
    assert call("GET", "/lwb/v1/status")["worker_pid"] == report["worker_pid"], (
        "ComfyUI worker restarted"
    )
    report["passed"] = True
    save()
    print(f"Passed. Private results: {args.output.resolve()}")


if __name__ == "__main__":
    main()
