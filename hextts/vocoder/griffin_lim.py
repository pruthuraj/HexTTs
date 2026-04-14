"""Fallback Griffin-Lim vocoder helpers."""

from __future__ import annotations

import numpy as np
import librosa


def mel_to_audio(
    mel_spec: np.ndarray,
    *,
    n_iter: int = 32,
    hop_length: int = 256,
    win_length: int = 1024,
    sample_rate: int = 22050,
    n_fft: int = 1024,
) -> np.ndarray:
    """Convert magnitude/mel-like spectrogram input to waveform with Griffin-Lim.

    This is used as a fallback path when neural vocoder assets are unavailable.
    """
    return librosa.griffinlim(
        mel_spec,
        n_iter=n_iter,
        hop_length=hop_length,
        win_length=win_length,
        n_fft=n_fft,
    ).astype(np.float32)