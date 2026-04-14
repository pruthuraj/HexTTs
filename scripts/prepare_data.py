"""Thin wrapper for package-based dataset preprocessing."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Ensure package imports work when running this file directly.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hextts.data.preprocessing import cli_main


if __name__ == "__main__":
    # Delegate all CLI parsing/logic to package-owned implementation.
    raise SystemExit(cli_main())
