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
    fmin: float = 0.0,
    fmax: float = 8000.0,
) -> np.ndarray:
    """Convert a dB-scale mel spectrogram to waveform with Griffin-Lim.

    Pipeline: dB → power → mel_to_stft (pseudo-inverse) → griffinlim.
    Matches the Griffin-Lim fallback path in hextts/inference/pipeline.py.
    """
    mel_power = np.power(10.0, mel_spec / 10.0)
    linear = librosa.feature.inverse.mel_to_stft(
        mel_power,
        sr=sample_rate,
        n_fft=n_fft,
        fmin=fmin,
        fmax=fmax,
    )
    return librosa.griffinlim(
        linear,
        n_iter=n_iter,
        hop_length=hop_length,
        win_length=win_length,
    ).astype(np.float32)
