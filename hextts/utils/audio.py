"""Lightweight audio helpers."""

from __future__ import annotations

import numpy as np


def ensure_mono(audio: np.ndarray) -> np.ndarray:
    """Convert stereo-like audio to mono by channel averaging."""
    if audio.ndim == 1:
        return audio
    if audio.ndim != 2:
        raise ValueError(f"Expected 1D or 2D audio array, got shape {audio.shape}")
    return audio.mean(axis=0)


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """Peak-normalize audio to [-1, 1] while preserving silent inputs."""
    peak = float(np.max(np.abs(audio)))
    if peak <= 0.0:
        return audio.astype(np.float32, copy=False)
    return (audio / peak).astype(np.float32, copy=False)


def clamp_audio(audio: np.ndarray, limit: float = 1.0) -> np.ndarray:
    """Hard-clip audio amplitude to a symmetric limit."""
    return np.clip(audio, -limit, limit).astype(np.float32, copy=False)