"""Training runner built on the existing trainer implementation."""

from __future__ import annotations

from typing import Dict, Optional

import torch

from train_vits import VITSTrainer


def run_training(config: Dict, device: torch.device, checkpoint: Optional[str] = None) -> None:
    """Run the training loop using the existing stable trainer."""
    trainer = VITSTrainer(config, device)
    if checkpoint:
        trainer.load_checkpoint(checkpoint)
    trainer.train()
