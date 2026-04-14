"""Config loading utilities for HexTTs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from .schema import validate_config


DEFAULT_CONFIG_PATH = Path("configs/base.yaml")
LEGACY_CONFIG_PATH = Path("vits_config.yaml")


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a mapping: {path}")
    return data


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load and validate config, preferring package configs when available."""
    if config_path:
        path = Path(config_path)
    elif DEFAULT_CONFIG_PATH.exists():
        path = DEFAULT_CONFIG_PATH
    elif LEGACY_CONFIG_PATH.exists():
        path = LEGACY_CONFIG_PATH
    else:
        raise FileNotFoundError(
            "No config file found. Checked configs/base.yaml and vits_config.yaml"
        )

    config = _load_yaml(path)
    config.setdefault("config_path", str(path))
    return validate_config(config)
