"""Training loss helpers for HexTTs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class LossBreakdown:
    """Named loss components for reporting and logging."""

    total: float
    reconstruction: float = 0.0
    duration: float = 0.0
    kl: float = 0.0

    def as_dict(self) -> Dict[str, float]:
        return {
            "total": self.total,
            "reconstruction": self.reconstruction,
            "duration": self.duration,
            "kl": self.kl,
        }


def combine_losses(*, reconstruction: float = 0.0, duration: float = 0.0, kl: float = 0.0) -> LossBreakdown:
    total = reconstruction + duration + kl
    return LossBreakdown(total=total, reconstruction=reconstruction, duration=duration, kl=kl)


class MultiScaleMelLoss(nn.Module):
    """L1 mel loss at multiple temporal resolutions.

    avg_pool1d at scales (1, 2, 4) forces the model to get coarse temporal
    structure right in addition to fine-grained per-frame detail.  The result
    is the mean over scales so its total magnitude stays comparable to a plain
    L1 loss and existing loss weights need no retuning.
    """

    def __init__(self, scales: Tuple[int, ...] = (1, 2, 4)):
        super().__init__()
        self.scales = scales

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred, target: (batch, mel_channels, time)
        Returns:
            scalar mean L1 loss across scales
        """
        total = pred.new_zeros(())
        for scale in self.scales:
            if scale > 1:
                p = F.avg_pool1d(pred, kernel_size=scale, stride=scale)
                t = F.avg_pool1d(target, kernel_size=scale, stride=scale)
            else:
                p, t = pred, target
            total = total + F.l1_loss(p, t)
        return total / len(self.scales)