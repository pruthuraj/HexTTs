"""Shared inference pipeline."""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from inference_vits import VITSInference


class VITSInferencePipeline:
    """Wrapper over the existing inference engine."""

    def __init__(
        self,
        checkpoint_path: str,
        config: dict,
        device: torch.device,
        vocoder_checkpoint: Optional[str] = None,
        vocoder_config: Optional[str] = None,
    ):
        self._engine = VITSInference(
            checkpoint_path=checkpoint_path,
            config=config,
            device=device,
            vocoder_checkpoint=vocoder_checkpoint,
            vocoder_config=vocoder_config,
        )

    def synthesize(
        self,
        text: str,
        duration_scale: float = 1.0,
        noise_scale: float = 0.3,
    ) -> Tuple:
        return self._engine.synthesize(
            text=text,
            duration_scale=duration_scale,
            noise_scale=noise_scale,
        )
