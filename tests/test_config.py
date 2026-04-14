from pathlib import Path

import pytest

from hextts.config.load import load_config
from hextts.config.schema import validate_config


def test_load_base_config_explicit_path():
    cfg = load_config("configs/base.yaml")
    assert cfg["sample_rate"] == 22050
    assert cfg["n_mel_channels"] == 80


def test_validate_config_missing_key_raises():
    with pytest.raises(ValueError):
        validate_config({"sample_rate": 22050})


def test_config_file_exists():
    assert Path("configs/base.yaml").exists()


def test_validate_config_rejects_mel_f_max_above_nyquist():
    cfg = {
        "data_dir": "./data",
        "audio_dir": "./audio",
        "log_dir": "./logs",
        "checkpoint_dir": "./checkpoints",
        "sample_rate": 22050,
        "n_mel_channels": 80,
        "batch_size": 4,
        "learning_rate": 1e-4,
        "num_epochs": 1,
        "mel_f_max": 12000,
    }
    with pytest.raises(ValueError):
        validate_config(cfg)


def test_validate_config_rejects_invalid_mel_lengths():
    cfg = {
        "data_dir": "./data",
        "audio_dir": "./audio",
        "log_dir": "./logs",
        "checkpoint_dir": "./checkpoints",
        "sample_rate": 22050,
        "n_mel_channels": 80,
        "batch_size": 4,
        "learning_rate": 1e-4,
        "num_epochs": 1,
        "mel_n_fft": 256,
        "mel_win_length": 512,
        "mel_hop_length": 128,
    }
    with pytest.raises(ValueError):
        validate_config(cfg)
