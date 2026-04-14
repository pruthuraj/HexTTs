"""Single public dataloader factory for raw vs cached features."""

from __future__ import annotations

from typing import Callable, Dict, Tuple

from torch.utils.data import DataLoader

from . import raw_dataset as vits_data
from . import cached_dataset as vits_data_cached


def _select_backend(config: Dict) -> Tuple[Callable, object]:
    """Choose raw or cached dataloader backend from one config flag."""
    use_cached = bool(config.get("use_cached_features", False))
    if use_cached:
        return vits_data_cached.create_dataloaders, vits_data_cached
    return vits_data.create_dataloaders, vits_data


def create_dataloaders(config: Dict):
    """Create dataloaders using one authoritative API."""
    factory, _ = _select_backend(config)
    # Keep worker and batch defaults centralized here for both backends.
    batch_size = int(config.get("batch_size", 1))
    num_workers = int(config.get("num_workers", 0))
    return factory(config, batch_size=batch_size, num_workers=num_workers)


def get_warning_summary(config: Dict):
    """Expose backend-specific warning counters through a stable facade."""
    _, backend = _select_backend(config)
    return backend.get_warning_summary()


def reset_warning_summary(config: Dict) -> None:
    """Reset backend-specific warning counters through a stable facade."""
    _, backend = _select_backend(config)
    backend.reset_warning_summary()
