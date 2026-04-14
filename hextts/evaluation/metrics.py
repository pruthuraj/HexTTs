"""Objective audio metrics used by evaluation workflows."""

from __future__ import annotations

import librosa
import numpy as np


def safe_float(x):
    arr = np.asarray(x)
    return float(arr.item()) if arr.size == 1 else float(np.mean(arr))


def spectral_flatness(audio: np.ndarray, n_fft: int = 1024, hop_length: int = 256) -> float:
    if len(audio) == 0:
        return 0.0

    flatness = librosa.feature.spectral_flatness(
        y=audio.astype(np.float32),
        n_fft=n_fft,
        hop_length=hop_length,
    )
    return float(np.mean(flatness))


def compute_silence_ratio(audio: np.ndarray, threshold: float = 0.01) -> float:
    if len(audio) == 0:
        return 1.0
    return float(np.mean(np.abs(audio) < threshold))
