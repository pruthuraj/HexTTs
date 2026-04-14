"""Single public dataloader factory for raw vs cached features."""

from __future__ import annotations

from typing import Callable, Dict, Tuple

from torch.utils.data import DataLoader

import vits_data
import vits_data_cached


def _select_backend(config: Dict) -> Tuple[Callable, object]:
    use_cached = bool(config.get("use_cached_features", False))
    if use_cached:
        return vits_data_cached.create_dataloaders, vits_data_cached
    return vits_data.create_dataloaders, vits_data


def create_dataloaders(config: Dict):
    """Create dataloaders using one authoritative API."""
    factory, _ = _select_backend(config)
    batch_size = int(config.get("batch_size", 1))
    num_workers = int(config.get("num_workers", 0))
    return factory(config, batch_size=batch_size, num_workers=num_workers)


def get_warning_summary(config: Dict):
    _, backend = _select_backend(config)
    return backend.get_warning_summary()


def reset_warning_summary(config: Dict) -> None:
    _, backend = _select_backend(config)
    backend.reset_warning_summary()
