"""Minimal end-to-end inference smoke tests for model API stability."""

import torch

from hextts.models.vits import build_vits_model, get_vocab_size


def _tiny_config():
    """Return a tiny model config to keep tests fast on CPU."""
    return {
        "vocab_size": get_vocab_size(),
        "encoder_hidden_size": 64,
        "encoder_num_layers": 1,
        "encoder_num_heads": 2,
        "encoder_kernel_size": 3,
        "encoder_dropout": 0.1,
        "duration_predictor_filters": 32,
        "duration_predictor_kernel_sizes": [3, 3],
        "duration_predictor_dropout": 0.1,
        "n_mel_channels": 80,
        "decoder_hidden_size": 64,
        "decoder_num_layers": 1,
        "latent_dim": 32,
    }


def test_inference_smoke_runs():
    """Inference should produce a non-empty mel with expected tensor rank/layout."""
    config = _tiny_config()
    model = build_vits_model(config)
    model.eval()

    phonemes = torch.randint(0, get_vocab_size(), (1, 8))
    lengths = torch.tensor([8], dtype=torch.long)

    with torch.no_grad():
        mel = model.inference(phonemes, lengths=lengths, duration_scale=1.0, noise_scale=0.1)

    assert mel.dim() == 3
    assert mel.shape[0] == 1
    assert mel.shape[1] == config["n_mel_channels"]
    assert mel.shape[2] > 0
