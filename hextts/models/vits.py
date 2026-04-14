"""Shared VITS model build path for training and inference."""

from __future__ import annotations

from typing import Dict, Optional

import torch

from .vits_impl import VITS
from hextts.data.raw_dataset import VOCAB_SIZE


def get_vocab_size() -> int:
    """Return authoritative vocabulary size."""
    # Keep one source of truth for token count across train/infer/checkpoint code.
    return VOCAB_SIZE


def build_vits_model(config: Dict, device: Optional[torch.device] = None) -> VITS:
    """Build VITS model with normalized config."""
    # Copy input config so this helper does not mutate caller-owned dictionaries.
    model_config = dict(config)
    model_config["vocab_size"] = get_vocab_size()
    model = VITS(model_config)
    if device is not None:
        model = model.to(device)
    return model
