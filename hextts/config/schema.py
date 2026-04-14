"""Configuration schema checks for HexTTs."""

from __future__ import annotations

from typing import Any, Dict

REQUIRED_KEYS = {
    "data_dir": str,
    "audio_dir": str,
    "log_dir": str,
    "checkpoint_dir": str,
    "sample_rate": int,
    "n_mel_channels": int,
    "batch_size": int,
    "learning_rate": (int, float),
    "num_epochs": int,
}


def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate required config keys and basic invariants."""
    missing = [k for k in REQUIRED_KEYS if k not in config]
    if missing:
        raise ValueError(f"Missing required config keys: {missing}")

    for key, expected_type in REQUIRED_KEYS.items():
        if not isinstance(config[key], expected_type):
            raise TypeError(
                f"Invalid type for '{key}'. Expected {expected_type}, got {type(config[key])}."
            )

    if config["sample_rate"] <= 0:
        raise ValueError("sample_rate must be > 0")
    if config["n_mel_channels"] <= 0:
        raise ValueError("n_mel_channels must be > 0")
    if config["batch_size"] <= 0:
        raise ValueError("batch_size must be > 0")
    if float(config["learning_rate"]) <= 0:
        raise ValueError("learning_rate must be > 0")

    # Runtime invariants used across training/inference/checkpoint compatibility.
    if "mel_n_fft" in config and int(config["mel_n_fft"]) <= 0:
        raise ValueError("mel_n_fft must be > 0")
    if "mel_hop_length" in config and int(config["mel_hop_length"]) <= 0:
        raise ValueError("mel_hop_length must be > 0")
    if "mel_win_length" in config and int(config["mel_win_length"]) <= 0:
        raise ValueError("mel_win_length must be > 0")

    mel_n_fft = int(config.get("mel_n_fft", 1))
    mel_hop_length = int(config.get("mel_hop_length", 1))
    mel_win_length = int(config.get("mel_win_length", 1))

    if mel_hop_length > mel_win_length:
        raise ValueError("mel_hop_length must be <= mel_win_length")
    if mel_win_length > mel_n_fft:
        raise ValueError("mel_win_length must be <= mel_n_fft")

    if "mel_f_max" in config and config["mel_f_max"] is not None:
        nyquist = float(config["sample_rate"]) / 2.0
        if float(config["mel_f_max"]) > nyquist:
            raise ValueError("mel_f_max must be <= sample_rate / 2")

    if "vocab_size" in config and int(config["vocab_size"]) <= 0:
        raise ValueError("vocab_size must be > 0")

    return config
