"""Audio-level evaluation pipeline for generated wav files."""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from .metrics import compute_silence_ratio, safe_float, spectral_flatness


def collect_audio_files(input_path: Path) -> list[Path]:
    """Resolve one input path into a validated list of wav files to evaluate."""
    if not input_path.exists():
        raise FileNotFoundError(f"Path not found: {input_path}")

    if input_path.is_file():
        if input_path.suffix.lower() != ".wav":
            raise ValueError(f"Not a .wav file: {input_path}")
        return [input_path]

    audio_files = sorted(input_path.glob("*.wav"))
    if not audio_files:
        raise FileNotFoundError(f"No .wav files found in directory: {input_path}")
    return audio_files


def evaluate_audio(audio_path: str, sr_target: int | None = None) -> dict:
    """Compute objective quality metrics and rule-based verdicts for one wav file."""
    audio, sr = sf.read(audio_path)

    # Downmix to mono so all metrics are comparable across stereo/mono files.
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    # Optional resample keeps cross-run comparisons on the same sample-rate basis.
    if sr_target is not None and sr != sr_target:
        audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=sr_target)
        sr = sr_target

    duration_sec = len(audio) / sr if sr > 0 else 0.0
    peak = np.max(np.abs(audio)) if len(audio) else 0.0
    rms = safe_float(np.sqrt(np.mean(audio ** 2))) if len(audio) else 0.0
    silence_ratio = compute_silence_ratio(audio, threshold=0.01)
    zcr = (
        safe_float(
            librosa.feature.zero_crossing_rate(
                audio,
                frame_length=1024,
                hop_length=256,
            ).mean()
        )
        if len(audio)
        else 0.0
    )
    spectral_flatness_val = spectral_flatness(audio)

    mel = librosa.feature.melspectrogram(
        y=audio.astype(np.float32),
        sr=sr,
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        n_mels=80,
        fmin=0,
        fmax=8000,
        power=2.0,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_frames = mel.shape[1]
    mel_mean_db = safe_float(mel_db.mean())
    mel_std_db = safe_float(mel_db.std())

    # Heuristic verdicts are intentionally simple and interpretable for quick triage.
    verdicts = []

    if duration_sec < 0.7:
        verdicts.append("Too short: duration predictor likely still weak.")
    elif duration_sec < 1.2:
        verdicts.append("Short, but may still contain early speech structure.")
    else:
        verdicts.append("Duration is in a more realistic range.")

    if silence_ratio > 0.80:
        verdicts.append("Mostly silence or near-silence.")
    elif silence_ratio > 0.50:
        verdicts.append("Contains a lot of silence.")
    else:
        verdicts.append("Silence level looks acceptable.")

    if rms < 0.01:
        verdicts.append("Very low energy output.")
    elif rms < 0.03:
        verdicts.append("Low energy, likely weak/soft speech.")
    else:
        verdicts.append("Energy level looks reasonable.")

    if zcr > 0.25:
        verdicts.append("Possibly noisy/buzzy output.")
    else:
        verdicts.append("Waveform is not excessively noisy by ZCR.")

    flat = spectral_flatness_val
    if flat > 0.2:
        verdicts.append("Spectrum looks buzzy/noisy.")
    elif flat > 0.05:
        verdicts.append("Spectrum has mild noise present.")
    else:
        verdicts.append("Spectrum has more speech-like structure.")

    return {
        "file": str(audio_path),
        "sample_rate": sr,
        "num_samples": len(audio),
        "duration_sec": round(duration_sec, 4),
        "peak_amplitude": round(float(peak), 6),
        "rms_energy": round(float(rms), 6),
        "silence_ratio": round(float(silence_ratio), 6),
        "zero_crossing_rate": round(float(zcr), 6),
        "spectral_flatness": round(float(spectral_flatness_val), 6),
        "mel_frames": int(mel_frames),
        "mel_mean_db": round(float(mel_mean_db), 4),
        "mel_std_db": round(float(mel_std_db), 4),
        "verdict": verdicts,
    }
