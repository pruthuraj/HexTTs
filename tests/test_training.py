"""Training-loop correctness tests.

Validates loss computation, duration regulation shape, and skip-counter behavior
using toy inputs so tests run fast on CPU without a real dataset.
"""

import torch
import pytest

from hextts.models.vits import build_vits_model, get_vocab_size


def _tiny_config():
    return {
        "vocab_size": get_vocab_size(),
        "encoder_hidden_size": 64,
        "encoder_num_layers": 1,
        "encoder_num_heads": 2,
        "encoder_kernel_size": 3,
        "encoder_dropout": 0.0,
        "duration_predictor_filters": 32,
        "duration_predictor_kernel_sizes": [3, 3],
        "duration_predictor_dropout": 0.0,
        "n_mel_channels": 80,
        "decoder_hidden_size": 64,
        "decoder_num_layers": 1,
        "latent_dim": 32,
        # loss / trainer knobs
        "loss_weight_reconstruction": 1.0,
        "loss_weight_kl": 0.1,
        "loss_weight_duration": 0.1,
        "loss_weight_pre_postnet": 0.5,
        "loss_weight_stft": 0.1,
        "kl_warmup_steps": 0,
        "duration_token_alpha": 1.0,
        "duration_sum_beta": 0.2,
        "max_duration_value": 20.0,
        "max_skipped_ratio": 0.5,
    }


class TestDurationRegulation:
    """Duration predictor output → repeat_interleave expansion contracts."""

    def test_forward_produces_positive_durations(self):
        """Duration predictor must output values > 0 (softplus + clamp enforces this)."""
        config = _tiny_config()
        model = build_vits_model(config)
        model.eval()

        phonemes = torch.randint(0, get_vocab_size(), (2, 5))
        lengths = torch.tensor([5, 4], dtype=torch.long)
        mel = torch.randn(2, 80, 12)

        with torch.no_grad():
            out = model(phonemes, mel_spec=mel, lengths=lengths)

        assert out["duration"].min().item() > 0.0

    def test_inference_mel_batch_dim(self):
        """Inference must return a 3-D tensor (batch=1, mel_channels, time)."""
        config = _tiny_config()
        model = build_vits_model(config)
        model.eval()

        phonemes = torch.randint(0, get_vocab_size(), (1, 5))
        lengths = torch.tensor([5], dtype=torch.long)

        with torch.no_grad():
            mel = model.inference(phonemes, lengths=lengths)

        assert mel.dim() == 3
        assert mel.shape[0] == 1
        assert mel.shape[1] == config["n_mel_channels"]

    def test_duration_clamp_respected(self):
        """No predicted duration should exceed max_duration_value."""
        config = _tiny_config()
        model = build_vits_model(config)
        model.eval()

        phonemes = torch.randint(0, get_vocab_size(), (2, 10))
        lengths = torch.tensor([10, 8], dtype=torch.long)
        mel = torch.randn(2, 80, 25)

        with torch.no_grad():
            out = model(phonemes, mel_spec=mel, lengths=lengths)

        assert out["duration"].max().item() <= config["max_duration_value"]


class TestLossComputation:
    """compute_loss output contracts."""

    def _make_trainer(self):
        import torch.optim as optim
        from hextts.training.trainer import VITSTrainer

        config = _tiny_config()
        config.update({
            "learning_rate": 1e-4,
            "adam_beta1": 0.9,
            "adam_beta2": 0.999,
            "adam_eps": 1e-9,
            "scheduler_type": "exponential",
            "scheduler_gamma": 0.999,
            "warmup_steps": 0,
            "use_amp": False,
            "log_dir": "/tmp/hextts_test_logs",
            "checkpoint_dir": "/tmp/hextts_test_ckpts",
            "data_dir": ".",
            "audio_dir": ".",
            "batch_size": 2,
            "num_workers": 0,
        })
        # Avoid creating a real trainer (needs dataloaders); test compute_loss directly.
        return config

    def test_loss_dict_keys_present(self):
        """compute_loss must return all expected metric keys."""
        from hextts.training.trainer import VITSTrainer
        from hextts.training.losses import MultiScaleMelLoss

        config = self._make_trainer()
        device = torch.device("cpu")
        model = build_vits_model(config, device=device)
        model.train()

        # Manually wire up the minimum trainer state needed for compute_loss.
        trainer = object.__new__(VITSTrainer)
        trainer.config = config
        trainer.device = device
        trainer.global_step = 0
        trainer.ms_mel_loss = MultiScaleMelLoss(scales=(1, 2, 4))

        batch_size, seq_len, mel_len = 2, 5, 12
        phonemes = torch.randint(0, get_vocab_size(), (batch_size, seq_len))
        lengths = torch.tensor([seq_len, seq_len - 1], dtype=torch.long)
        mel = torch.randn(batch_size, 80, mel_len)

        out = model(phonemes, mel_spec=mel, lengths=lengths)
        loss_dict = trainer.compute_loss(out, mel, lengths, lengths, config)

        required_keys = {
            "total_loss", "recon_loss", "kl_loss", "duration_loss",
            "token_duration_mae", "sum_error_mean", "length_mismatch_frames",
        }
        missing = required_keys - loss_dict.keys()
        assert not missing, f"compute_loss missing keys: {missing}"

    def test_total_loss_is_finite(self):
        """Total loss must be finite for a well-formed batch."""
        from hextts.training.trainer import VITSTrainer
        from hextts.training.losses import MultiScaleMelLoss

        config = self._make_trainer()
        device = torch.device("cpu")
        model = build_vits_model(config, device=device)
        model.train()

        trainer = object.__new__(VITSTrainer)
        trainer.config = config
        trainer.device = device
        trainer.global_step = 0
        trainer.ms_mel_loss = MultiScaleMelLoss(scales=(1, 2, 4))

        batch_size, seq_len, mel_len = 2, 5, 12
        phonemes = torch.randint(0, get_vocab_size(), (batch_size, seq_len))
        lengths = torch.tensor([seq_len, seq_len - 1], dtype=torch.long)
        mel = torch.randn(batch_size, 80, mel_len)

        out = model(phonemes, mel_spec=mel, lengths=lengths)
        loss_dict = trainer.compute_loss(out, mel, lengths, lengths, config)

        loss = loss_dict["total_loss"]
        assert torch.isfinite(loss), f"total_loss is not finite: {loss.item()}"

    def test_pseudo_duration_targets_sum_matches_mel_length(self):
        """Pseudo duration targets must sum exactly to mel_length for each sample."""
        from hextts.training.trainer import VITSTrainer

        phoneme_lengths = torch.tensor([5, 7])
        mel_lengths = torch.tensor([20, 35])

        targets = VITSTrainer._build_pseudo_duration_targets(
            phoneme_lengths=phoneme_lengths,
            mel_lengths=mel_lengths,
            max_seq_len=10,
            device=torch.device("cpu"),
        )

        for i in range(2):
            n = phoneme_lengths[i].item()
            expected_sum = mel_lengths[i].item()
            actual_sum = targets[i, :n].sum().item()
            assert actual_sum == expected_sum, (
                f"Sample {i}: target sum {actual_sum} != mel_length {expected_sum}"
            )
