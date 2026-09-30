"""Compatibility entry point; see the generic queue API tools."""
from pathlib import Path
import runpy

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("check_profiles.py")), run_name="__main__")
