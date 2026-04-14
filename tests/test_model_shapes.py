import torch

from hextts.models.vits import build_vits_model, get_vocab_size


def _tiny_config():
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


def test_model_forward_shapes():
    config = _tiny_config()
    model = build_vits_model(config)
    model.train()

    batch_size = 2
    seq_len = 6
    mel_len = 12

    phonemes = torch.randint(0, get_vocab_size(), (batch_size, seq_len))
    lengths = torch.tensor([seq_len, seq_len - 1], dtype=torch.long)
    mel = torch.randn(batch_size, config["n_mel_channels"], mel_len)

    out = model(phonemes, mel_spec=mel, lengths=lengths)

    assert out["predicted_mel"].dim() == 3
    assert out["predicted_mel"].shape[0] == batch_size
    assert out["predicted_mel"].shape[1] == config["n_mel_channels"]
    assert out["duration"].shape[:2] == (batch_size, seq_len)
