import pytest
import torch

from hextts.models.checkpointing import (
    load_checkpoint,
    save_checkpoint,
    validate_checkpoint_compatibility,
)


def test_checkpoint_compatibility_happy_path():
    config = {"vocab_size": 40, "sample_rate": 22050, "n_mel_channels": 80}
    checkpoint = {"vocab_size": 40, "sample_rate": 22050, "n_mels": 80}
    validate_checkpoint_compatibility(checkpoint, config)


def test_checkpoint_compatibility_mismatch_raises():
    config = {"vocab_size": 40, "sample_rate": 22050, "n_mel_channels": 80}
    checkpoint = {"vocab_size": 41, "sample_rate": 22050, "n_mels": 80}
    with pytest.raises(ValueError):
        validate_checkpoint_compatibility(checkpoint, config)


def test_checkpoint_architecture_flags_compatible():
    config = {
        "vocab_size": 40,
        "sample_rate": 22050,
        "n_mel_channels": 80,
        "use_postnet": True,
        "max_duration_value": 20.0,
    }
    checkpoint = {
        "vocab_size": 40,
        "sample_rate": 22050,
        "n_mels": 80,
        "architecture_flags": {
            "use_postnet": True,
            "duration_clamp": 20.0,
        },
    }
    validate_checkpoint_compatibility(checkpoint, config)


def test_checkpoint_architecture_flags_mismatch_raises():
    config = {
        "vocab_size": 40,
        "sample_rate": 22050,
        "n_mel_channels": 80,
        "use_postnet": True,
        "max_duration_value": 20.0,
    }
    checkpoint = {
        "vocab_size": 40,
        "sample_rate": 22050,
        "n_mels": 80,
        "architecture_flags": {
            "use_postnet": False,
            "duration_clamp": 20.0,
        },
    }

    with pytest.raises(ValueError):
        validate_checkpoint_compatibility(checkpoint, config)


def test_checkpoint_roundtrip_save_load(tmp_path):
    config = {
        "vocab_size": 40,
        "sample_rate": 22050,
        "n_mel_channels": 80,
        "max_duration_value": 20.0,
    }

    model = torch.nn.Linear(4, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    ckpt_path = tmp_path / "roundtrip.pt"

    save_checkpoint(
        path=str(ckpt_path),
        model=model,
        optimizer=optimizer,
        scheduler=None,
        scaler=None,
        epoch=3,
        global_step=42,
        config=config,
        model_version="v0.5.0",
        git_commit="deadbeef",
    )

    loaded = load_checkpoint(str(ckpt_path), device=torch.device("cpu"))

    assert loaded["epoch"] == 3
    assert loaded["global_step"] == 42
    assert loaded["model_version"] == "v0.5.0"
    assert loaded["git_commit"] == "deadbeef"
    assert loaded["sample_rate"] == 22050
    assert loaded["n_mels"] == 80
    assert loaded["architecture_flags"]["use_postnet"] is True
    assert loaded["architecture_flags"]["duration_clamp"] == 20.0
    assert "model_state_dict" in loaded
    assert "optimizer_state_dict" in loaded
