"""Bounded local GPU/PSS sampler; no prompt, environment, command line or credentials."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=1800)
    parser.add_argument("--interval", type=float, default=0.5)
    args = parser.parse_args()
    if args.duration <= 0 or args.interval < 0.25:
        parser.error("positive duration and interval >= .25 required")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with os.fdopen(
        os.open(args.output, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600), "w"
    ) as stream:
        deadline = time.monotonic() + args.duration
        while time.monotonic() < deadline:
            sample = {"timestamp": time.time(), "source": "live_sample"}
            for label, query in [
                ("gpu", "--query-gpu=memory.used,memory.total,utilization.gpu,power.draw"),
                ("compute", "--query-compute-apps=pid,used_gpu_memory"),
            ]:
                try:
                    result = subprocess.run(
                        ["nvidia-smi", query, "--format=csv,noheader,nounits"],
                        capture_output=True,
                        text=True,
                        timeout=15,
                    )
                    sample[label] = result.stdout.strip() if result.returncode == 0 else None
                except (OSError, subprocess.SubprocessError):
                    sample[label] = None
            sample["process_memory_kib"] = {}
            sample["cgroup_memory_bytes"] = {}
            for row in (sample["compute"] or "").splitlines():
                try:
                    pid = int(row.split(",")[0])
                    fields = Path(f"/proc/{pid}/smaps_rollup").read_text().splitlines()
                    sample["process_memory_kib"][str(pid)] = {
                        line.split(":")[0]: int(line.split()[1])
                        for line in fields
                        if line.startswith(("Rss:", "Pss:"))
                    }
                    # cgroup v2 memory includes the complete container/group,
                    # distinct from this CUDA process's proportional RSS.
                    for line in Path(f"/proc/{pid}/cgroup").read_text().splitlines():
                        hierarchy, controllers, relative = line.split(":", 2)
                        if hierarchy == "0" and not controllers:
                            group = Path("/sys/fs/cgroup") / relative.lstrip("/")
                            if group.resolve().is_relative_to(Path("/sys/fs/cgroup")):
                                sample["cgroup_memory_bytes"][str(pid)] = {
                                    name: int((group / name).read_text())
                                    for name in ("memory.current", "memory.peak")
                                    if (group / name).is_file()
                                }
                except (OSError, ValueError):
                    continue
            stream.write(json.dumps(sample) + "\n")
            stream.flush()
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
