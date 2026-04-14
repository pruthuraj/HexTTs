"""Wrapper that keeps old validation entrypoint available from scripts/."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "validate_dataset.py"

if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    runpy.run_path(str(TARGET), run_name="__main__")
