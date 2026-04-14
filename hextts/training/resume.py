"""Resume helpers for training."""

from __future__ import annotations

from typing import Optional


def normalize_checkpoint_arg(checkpoint: Optional[str]) -> Optional[str]:
    """Normalize CLI checkpoint argument by trimming blanks to None."""
    if checkpoint is None:
        return None
    checkpoint = checkpoint.strip()
    return checkpoint or None
