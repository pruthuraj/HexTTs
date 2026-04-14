"""Small file I/O helpers."""

from __future__ import annotations

from pathlib import Path


def ensure_parent_dir(path: str | Path) -> Path:
    """Create parent directories for a target file path and return Path object."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def read_text_file(path: str | Path, encoding: str = "utf-8") -> str:
    """Read UTF-8 text content from disk."""
    return Path(path).read_text(encoding=encoding)


def write_text_file(path: str | Path, text: str, encoding: str = "utf-8") -> Path:
    """Write text to disk, creating parent folders when needed."""
    target = ensure_parent_dir(path)
    target.write_text(text, encoding=encoding)
    return target