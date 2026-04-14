"""Training loss helpers for HexTTs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


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