"""High-level synthesis convenience function."""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from .pipeline import VITSInferencePipeline


def synthesize_text(
    checkpoint_path: str,
    config: dict,
    text: str,
    device: torch.device,
    duration_scale: float = 1.0,
    noise_scale: float = 0.3,
    vocoder_checkpoint: Optional[str] = None,
    vocoder_config: Optional[str] = None,
) -> Tuple:
    """Create a pipeline and synthesize one utterance with optional vocoder override."""
    pipeline = VITSInferencePipeline(
        checkpoint_path=checkpoint_path,
        config=config,
        device=device,
        vocoder_checkpoint=vocoder_checkpoint,
        vocoder_config=vocoder_config,
    )
    return pipeline.synthesize(text, duration_scale=duration_scale, noise_scale=noise_scale)
