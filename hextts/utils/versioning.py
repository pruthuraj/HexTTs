"""Version helpers for checkpoints and reports."""

from __future__ import annotations

import subprocess


MODEL_VERSION = "v0.5.0"


def get_model_version() -> str:
    """Return static project model version tag."""
    return MODEL_VERSION


def get_git_commit(default: str = "unknown") -> str:
    """Resolve current git commit SHA, returning default when unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return default
    return result.stdout.strip() or default