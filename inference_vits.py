"""
Inference Script for VITS TTS
Generates speech from text using trained model
"""

import warnings
from pathlib import Path

import torch
import numpy as np
import librosa
import soundfile as sf
import argparse
from typing import Tuple, Optional

from hextts.config import load_config
from hextts.models.vits import build_vits_model, get_vocab_size
from hextts.models.checkpointing import load_checkpoint, validate_checkpoint_compatibility
from vits_data import PHONEME_TO_ID
from vocoder import HiFiGANVocoder


class VITSInference:
    """Inference class for VITS model"""

    def __init__(
        self,
        checkpoint_path: str,
        config: dict,
        device: torch.device,
        vocoder_checkpoint: Optional[str] = None,
        vocoder_config: Optional[str] = None,
    ):
        """
        Initialize inference

        Args:
            checkpoint_path: path to trained model checkpoint
            config: configuration dict
            device: torch device
        """
        self.device = device
        self.config = config
        self.vocoder = None

        # Load model
        print(f"Loading model from {checkpoint_path}...")
        config["vocab_size"] = get_vocab_size()
        self.model = build_vits_model(config, device=device)
        self.model.eval()

        checkpoint = load_checkpoint(checkpoint_path, device=device)
        validate_checkpoint_compatibility(checkpoint, config)

        # Allow loading older checkpoints that do not contain PostNet weights
        missing_keys, unexpected_keys = self.model.load_state_dict(
            checkpoint["model_state_dict"],
            strict=False
        )

        # Validate that any missing keys are only the expected PostNet keys
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

        print("Model loaded successfully!")

        if vocoder_checkpoint or vocoder_config:
            if not vocoder_checkpoint or not vocoder_config:
                raise ValueError("Both --vocoder_checkpoint and --vocoder_config are required to enable HiFi-GAN")

            if not Path(vocoder_checkpoint).exists():
                raise FileNotFoundError(f"Vocoder checkpoint not found: {vocoder_checkpoint}")
            if not Path(vocoder_config).exists():
                raise FileNotFoundError(f"Vocoder config not found: {vocoder_config}")

            self.vocoder = HiFiGANVocoder(vocoder_checkpoint, vocoder_config, device)

        # Audio parameters
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
        """
        Convert text to phoneme sequence using g2p_en

        Args:
            text: input text

        Returns:
            phoneme string (space-separated)
        """
        try:
            from g2p_en import G2p
            g2p = G2p()
            phoneme_list = g2p(text)

            # Remove stress markers and filter
            phonemes = [p.rstrip("012") for p in phoneme_list]
            phonemes = [p for p in phonemes if p and p != " "]

            return " ".join(phonemes)
        except Exception as e:
            print(f"Error converting text to phonemes: {e}")
            return text

    def phonemes_to_ids(self, phoneme_str: str) -> torch.Tensor:
        """
        Convert phoneme string to ID tensor

        Args:
            phoneme_str: space-separated phonemes

        Returns:
            tensor of shape (1, seq_len)
        """
        phonemes = phoneme_str.split()
        ids = []

        for p in phonemes:
            p = p.strip().upper()
            if p in PHONEME_TO_ID:
                ids.append(PHONEME_TO_ID[p])
            else:
                print(f"Warning: Unknown phoneme '{p}', skipping")

        if not ids:
            # Return silence if no valid phonemes
            ids = [PHONEME_TO_ID["PAD"]]

        return torch.LongTensor(ids).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def generate_mel_spectrogram(
        self,
        phoneme_ids: torch.Tensor,
        duration_scale: float = 1.0,
        noise_scale: float = 0.3
    ) -> np.ndarray:
        """
        Generate mel-spectrogram from phoneme IDs

        Args:
            phoneme_ids: tensor of shape (1, seq_len)
            duration_scale: scale for predicted durations
            noise_scale: scale for latent noise

        Returns:
            mel-spectrogram of shape (n_mel_channels, time_steps)
        """
        if phoneme_ids.dim() != 2 or phoneme_ids.size(0) != 1:
            raise ValueError(f"Expected phoneme_ids of shape (1, seq_len), got {phoneme_ids.shape}")

        lengths = torch.LongTensor([phoneme_ids.size(1)]).to(self.device)

        mel_spec = self.model.inference(
            phoneme_ids,
            lengths=lengths,
            duration_scale=duration_scale,
            noise_scale=noise_scale
        )

        # Check for NaN or Inf values in the predicted mel spectrogram
        if torch.isnan(mel_spec).any() or torch.isinf(mel_spec).any():
            raise ValueError("predicted_mel contains NaN or Inf")

        # Remove batch dimension and move to CPU
        mel_spec = mel_spec.squeeze(0).cpu().numpy()

        # Convert from model output scale
        mel_spec = mel_spec * -self.ref_level_db + self.ref_level_db

        return mel_spec

    def mel_spectrogram_to_audio(self, mel_spec: np.ndarray) -> np.ndarray:
        """
        Convert mel-spectrogram to audio.

        Args:
            mel_spec: mel-spectrogram of shape (n_mel_channels, time_steps)

        Returns:
            audio waveform
        """
        if self.vocoder is not None:
            mel_for_vocoder = np.clip((self.ref_level_db - mel_spec) / self.ref_level_db, 0.0, 1.0)
            return self.vocoder(mel_for_vocoder.astype(np.float32))

        # Convert from dB to power
        mel_spec = np.power(10.0, mel_spec / 10.0)

        # Invert mel spectrogram
        spectrogram = librosa.feature.inverse.mel_to_stft(
            mel_spec,
            sr=self.sample_rate,
            n_fft=self.mel_n_fft,
            fmin=self.mel_f_min,
            fmax=self.mel_f_max
        )

        # Griffin-Lim to convert spectrogram to waveform
        audio = librosa.griffinlim(
            spectrogram,
            n_iter=256, # Increase iterations for better quality (default is 32)
            hop_length=self.mel_hop_length,
            win_length=self.mel_win_length,
            momentum=0.99,
            init="random"
        )

        # Normalize
        audio = audio / (np.abs(audio).max() + 1e-7)

        return audio

    def synthesize(
        self,
        text: str,
        duration_scale: float = 1.0,
        noise_scale: float = 0.3
    ) -> Tuple[np.ndarray, int]:
        """
        Synthesize speech from text

        Args:
            text: input text
            duration_scale: scale for predicted durations
            noise_scale: scale for latent noise

        Returns:
            audio waveform, sample rate
        """
        print(f"Input text: {text}")

        # Text → Phonemes
        phoneme_str = self.text_to_phonemes(text)
        print(f"Phonemes: {phoneme_str}")

        # Phonemes → IDs
        phoneme_ids = self.phonemes_to_ids(phoneme_str)
        print(f"Phoneme IDs shape: {phoneme_ids.shape}")

        # Mel-spectrogram generation
        mel_spec = self.generate_mel_spectrogram(
            phoneme_ids,
            duration_scale=duration_scale,
            noise_scale=noise_scale
        )
        print(f"Mel-spectrogram shape: {mel_spec.shape}")

        # Mel-spectrogram → Audio
        audio = self.mel_spectrogram_to_audio(mel_spec)
        output_sample_rate = self.vocoder.sample_rate if self.vocoder is not None else self.sample_rate
        print(f"Audio shape: {audio.shape}, duration: {len(audio) / output_sample_rate:.2f}s")

        return audio, output_sample_rate


def main():
    parser = argparse.ArgumentParser(description="VITS TTS Inference")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--config", type=str, default=None, help="Path to config file")
    parser.add_argument("--vocoder_checkpoint", type=str, default=None, help="Optional HiFi-GAN generator checkpoint")
    parser.add_argument("--vocoder_config", type=str, default=None, help="Optional HiFi-GAN config file")
    parser.add_argument("--duration_scale", type=float, default=1.0, help="Scale predicted durations (higher = slower speech)")
    parser.add_argument("--noise_scale", type=float, default=0.3, help="Latent noise scale (lower = cleaner / less varied)")
    parser.add_argument("--text", type=str, required=True, help="Text to synthesize")
    parser.add_argument("--output", type=str, default="output.wav", help="Output audio file")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda or cpu)")
    args = parser.parse_args()

    # Load config (prefers configs/base.yaml, falls back to vits_config.yaml)
    config = load_config(args.config)

    # Set device
    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Initialize inference
    inference = VITSInference(
        args.checkpoint,
        config,
        device,
        vocoder_checkpoint=args.vocoder_checkpoint,
        vocoder_config=args.vocoder_config,
    )

    # Synthesize
    audio, sr = inference.synthesize(
        args.text,
        duration_scale=args.duration_scale,
        noise_scale=args.noise_scale,
    )

    # Save audio
    sf.write(args.output, audio, sr)
    print(f"\nAudio saved to {args.output}")


if __name__ == "__main__":
    main()