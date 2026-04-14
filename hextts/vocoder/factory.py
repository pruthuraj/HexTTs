"""Vocoder factory helpers."""

from __future__ import annotations

import torch

from .hifigan import HiFiGANVocoder


def build_vocoder(checkpoint_path: str, config_path: str, device: torch.device) -> HiFiGANVocoder:
    """Construct the default neural vocoder used by inference scripts."""
    return HiFiGANVocoder(checkpoint_path=checkpoint_path, config_path=config_path, device=device)