"""Warning configuration helpers."""

from __future__ import annotations

import warnings


def configure_warning_filters() -> None:
    """Enable default visibility for user warnings during local runs."""
    warnings.filterwarnings("default", category=UserWarning)