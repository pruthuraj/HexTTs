"""Shared inference pipeline."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional, Tuple

import librosa
import numpy as np
import torch

from hextts.models.checkpointing import load_checkpoint, validate_checkpoint_compatibility
from hextts.models.vits import build_vits_model, get_vocab_size
from hextts.vocoder import HiFiGANVocoder
from hextts.data.raw_dataset import PHONEME_TO_ID


class VITSInference:
    """Inference engine for VITS model."""

    def __init__(
        self,
        checkpoint_path: str,
        config: dict,
        device: torch.device,
        vocoder_checkpoint: Optional[str] = None,
        vocoder_config: Optional[str] = None,
    ):
        # Keep runtime artifacts on the chosen device and retain config for conversions.
        self.device = device
        self.config = config
        self.vocoder = None

        # Force vocabulary size from shared dataset mapping to avoid config drift.
        config["vocab_size"] = get_vocab_size()
        self.model = build_vits_model(config, device=device)
        self.model.eval()

        # Load and validate checkpoint before partial/non-strict state_dict load.
        checkpoint = load_checkpoint(checkpoint_path, device=device)
        validate_checkpoint_compatibility(checkpoint, config)

        missing_keys, unexpected_keys = self.model.load_state_dict(
            checkpoint["model_state_dict"],
            strict=False,
        )

        non_postnet_missing_keys = [k for k in missing_keys if not k.startswith("postnet.")]
        if non_postnet_missing_keys:
            warnings.warn(
                f"Unexpected missing keys in checkpoint (non-PostNet): {non_postnet_missing_keys}",
                UserWarning,
            )
        if unexpected_keys:
            warnings.warn(
                f"Unexpected keys found in checkpoint (not in model): {unexpected_keys}",
                UserWarning,
            )

        if vocoder_checkpoint or vocoder_config:
            if not vocoder_checkpoint or not vocoder_config:
                raise ValueError("Both --vocoder_checkpoint and --vocoder_config are required to enable HiFi-GAN")

            if not Path(vocoder_checkpoint).exists():
                raise FileNotFoundError(f"Vocoder checkpoint not found: {vocoder_checkpoint}")
            if not Path(vocoder_config).exists():
                raise FileNotFoundError(f"Vocoder config not found: {vocoder_config}")

            self.vocoder = HiFiGANVocoder(vocoder_checkpoint, vocoder_config, device)

        # Cache audio-related config values to keep conversion code explicit.
        self.sample_rate = config["sample_rate"]
        self.n_mel_channels = config["n_mel_channels"]
        self.mel_n_fft = config["mel_n_fft"]
        self.mel_hop_length = config["mel_hop_length"]
        self.mel_win_length = config["mel_win_length"]
        self.mel_f_min = config["mel_f_min"]
        self.mel_f_max = config["mel_f_max"]
        self.ref_level_db = config["ref_level_db"]
        self.min_level_db = config["min_level_db"]

    def text_to_phonemes(self, text: str) -> str:
        """Convert raw text to a whitespace-separated phoneme string.

        Raises RuntimeError if g2p_en is unavailable or fails — continuing
        with the raw text would produce garbage phoneme IDs and silent/noisy output.
        """
        try:
            from g2p_en import G2p
            g2p = G2p()
        except ImportError as exc:
            raise RuntimeError(
                "g2p_en is not installed. Install it with: pip install g2p_en"
            ) from exc

        try:
            phoneme_list = g2p(text)
        except Exception as exc:
            raise RuntimeError(
                f"g2p_en failed to convert text to phonemes: {exc!r}"
            ) from exc

        phonemes = [p.rstrip("012") for p in phoneme_list]
        phonemes = [p for p in phonemes if p and p != " "]

        if not phonemes:
            raise RuntimeError(
                f"g2p_en returned no valid phonemes for text: {text!r}"
            )

        return " ".join(phonemes)

    def phonemes_to_ids(self, phoneme_str: str) -> torch.Tensor:
        """Map phoneme tokens to model IDs and return shape (1, seq_len) tensor."""
        phonemes = phoneme_str.split()
        ids = []

        for phoneme in phonemes:
            p = phoneme.strip().upper()
            if p in PHONEME_TO_ID:
                ids.append(PHONEME_TO_ID[p])
            else:
                print(f"Warning: Unknown phoneme '{p}', skipping")

        if not ids:
            # Always provide at least one token so downstream inference can proceed.
            ids = [PHONEME_TO_ID["PAD"]]

        return torch.LongTensor(ids).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def generate_mel_spectrogram(
        self,
        phoneme_ids: torch.Tensor,
        duration_scale: float = 1.0,
        noise_scale: float = 0.3,
    ) -> np.ndarray:
        """Generate model mel output and map it back to dB-like scale used by vocoder path."""
        if phoneme_ids.dim() != 2 or phoneme_ids.size(0) != 1:
            raise ValueError(f"Expected phoneme_ids of shape (1, seq_len), got {phoneme_ids.shape}")

        lengths = torch.LongTensor([phoneme_ids.size(1)]).to(self.device)

        mel_spec = self.model.inference(
            phoneme_ids,
            lengths=lengths,
            duration_scale=duration_scale,
            noise_scale=noise_scale,
        )

        if torch.isnan(mel_spec).any() or torch.isinf(mel_spec).any():
            raise ValueError("predicted_mel contains NaN or Inf")

        mel_spec = mel_spec.squeeze(0).cpu().numpy()
        # Inverse of training normalization: norm = (mel_db - min_level_db) / -min_level_db
        # → mel_db = norm * -min_level_db + min_level_db
        # Output is in [min_level_db, 0] dB.
        mel_spec = mel_spec * -self.min_level_db + self.min_level_db
        return mel_spec

    def mel_spectrogram_to_audio(self, mel_spec: np.ndarray) -> np.ndarray:
        """Convert mel to waveform via HiFi-GAN when present, else Griffin-Lim fallback."""
        if self.vocoder is not None:
            # Re-normalize dB-scale mel back to [0, 1] using the same convention as the dataset.
            # norm = (mel_db - min_level_db) / -min_level_db → [0, 1] for mel_db ∈ [min_level_db, 0]
            mel_for_vocoder = np.clip(
                (mel_spec - self.min_level_db) / -self.min_level_db, 0.0, 1.0
            )
            return self.vocoder(mel_for_vocoder.astype(np.float32))

        # Griffin-Lim fallback path for environments without neural vocoder assets.
        mel_spec = np.power(10.0, mel_spec / 10.0)
        spectrogram = librosa.feature.inverse.mel_to_stft(
            mel_spec,
            sr=self.sample_rate,
            n_fft=self.mel_n_fft,
            fmin=self.mel_f_min,
            fmax=self.mel_f_max,
        )

        audio = librosa.griffinlim(
            spectrogram,
            n_iter=256,
            hop_length=self.mel_hop_length,
            win_length=self.mel_win_length,
            momentum=0.99,
            init="random",
        )

        audio = audio / (np.abs(audio).max() + 1e-7)
        return audio

    def synthesize(
        self,
        text: str,
        duration_scale: float = 1.0,
        noise_scale: float = 0.3,
    ) -> Tuple[np.ndarray, int]:
        """Run end-to-end text-to-audio synthesis and return waveform + sample rate."""
        phoneme_str = self.text_to_phonemes(text)
        phoneme_ids = self.phonemes_to_ids(phoneme_str)
        mel_spec = self.generate_mel_spectrogram(
            phoneme_ids,
            duration_scale=duration_scale,
            noise_scale=noise_scale,
        )

        audio = self.mel_spectrogram_to_audio(mel_spec)
        output_sample_rate = self.vocoder.sample_rate if self.vocoder is not None else self.sample_rate
        return audio, output_sample_rate


class VITSInferencePipeline:
    """Simple reusable wrapper over VITSInference."""

    def __init__(
        self,
        checkpoint_path: str,
        config: dict,
        device: torch.device,
        vocoder_checkpoint: Optional[str] = None,
        vocoder_config: Optional[str] = None,
    ):
        # Keep this wrapper lightweight so scripts and services can share one call surface.
        self._engine = VITSInference(
            checkpoint_path=checkpoint_path,
            config=config,
            device=device,
            vocoder_checkpoint=vocoder_checkpoint,
            vocoder_config=vocoder_config,
        )

    def synthesize(
        self,
        text: str,
        duration_scale: float = 1.0,
        noise_scale: float = 0.3,
    ) -> Tuple[np.ndarray, int]:
        """Delegate to the internal engine while exposing a stable interface."""
        return self._engine.synthesize(
            text=text,
            duration_scale=duration_scale,
            noise_scale=noise_scale,
        )
