"""Validate administrator profiles without restarting ComfyUI or loading models."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lwb.process import OwnedServer, _resolve_binary  # noqa: E402
from lwb.scheduler import load_profiles  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profiles", type=Path)
    parser.add_argument(
        "--check-build",
        action="store_true",
        help="Also check paths and run the administrator-selected binary with --version/--help; no model launch",
    )
    args = parser.parse_args()
    os.environ["LWB_PROFILES_FILE"] = str(args.profiles.resolve())
    try:
        profiles = load_profiles()
        for profile in profiles.values():
            if args.check_build:
                binary = _resolve_binary(profile.binary_path)
                OwnedServer()._command_for(binary, profile)
                OwnedServer._probe(binary)
                OwnedServer._check_custom_options(binary, profile.extra_args)
    except Exception as exc:
        parser.exit(1, f"Profile validation failed: {exc}\n")
    print(
        json.dumps(
            {
                "valid": True,
                "build_checked": args.check_build,
                "profiles": {
                    name: {
                        "context_size": p.context_size,
                        "gpu_layers": p.gpu_layers,
                        "mmproj_enabled": bool(p.mmproj_path),
                        "startup_timeout_seconds": p.startup_timeout_seconds,
                    }
                    for name, p in profiles.items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
