"""Thin wrapper for package-based dataset validation."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Ensure package imports work when running this file directly.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hextts.data.validation import main

if __name__ == "__main__":
    # Delegate execution to package-owned validator entrypoint.
    raise SystemExit(main())
