"""Shared VITS model build path for training and inference."""

from __future__ import annotations

from typing import Dict, Optional

import torch

from vits_model import VITS
from vits_data import VOCAB_SIZE


def get_vocab_size() -> int:
    """Return authoritative vocabulary size."""
    return VOCAB_SIZE


def build_vits_model(config: Dict, device: Optional[torch.device] = None) -> VITS:
    """Build VITS model with normalized config."""
    model_config = dict(config)
    model_config["vocab_size"] = get_vocab_size()
    model = VITS(model_config)
    if device is not None:
        model = model.to(device)
    return model
