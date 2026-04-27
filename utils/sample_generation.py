"""Utility for writing quick qualitative sample WAVs during training."""

from pathlib import Path
from typing import Optional
import warnings
import numpy as np
import torch
import torchaudio

from hextts.vocoder.griffin_lim import mel_to_audio


@torch.no_grad()
def generate_samples(
    model,
    texts,
    text_to_sequence_fn,
    output_dir: str,
    epoch: int,
    device: str = "cuda",
    sample_rate: int = 22050,
    config: Optional[dict] = None,
):
    """Run model inference for each prompt, convert mel-spectrograms to waveforms, and save numbered WAV files."""
    model.eval()
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Extract audio processing params from config or use defaults
    if config is None:
        config = {}
    
    ref_level_db = config.get("ref_level_db", 20.0)
    mel_hop_length = config.get("mel_hop_length", 256)
    mel_win_length = config.get("mel_win_length", 1024)
    mel_n_fft = config.get("mel_n_fft", 1024)

    for i, text in enumerate(texts, start=1):
        # Convert text into model input IDs using caller-provided tokenizer.
        seq = text_to_sequence_fn(text)
        x = torch.LongTensor(seq).unsqueeze(0).to(device)
        x_lengths = torch.LongTensor([x.size(1)]).to(device)

        # Model returns mel-spectrogram (batch_size, n_mels, time_steps)
        mel_spec = model.inference(x, x_lengths)

        if isinstance(mel_spec, tuple):
            mel_spec = mel_spec[0]

        # Verify no NaN/Inf values in mel-spectrogram
        if torch.isnan(mel_spec).any() or torch.isinf(mel_spec).any():
            warnings.warn(f"Skipping sample {i} at epoch {epoch}: mel-spectrogram contains NaN or Inf")
            continue

        # Convert to numpy and denormalize
        mel_spec_np = mel_spec.squeeze(0).cpu().numpy()  # (n_mels, time_steps)
        mel_spec_np = mel_spec_np * -ref_level_db + ref_level_db
        mel_spec_np = np.clip(mel_spec_np, a_min=-100, a_max=100)

        # Convert mel-spectrogram to waveform using Griffin-Lim (no external vocoder needed)
        try:
            audio = mel_to_audio(
                mel_spec_np,
                n_iter=32,
                hop_length=mel_hop_length,
                win_length=mel_win_length,
                sample_rate=sample_rate,
                n_fft=mel_n_fft,
            )
            
            # Normalize audio to prevent clipping
            max_val = np.abs(audio).max()
            if max_val > 0:
                audio = audio / (max_val + 1e-7)

            # Convert numpy array to tensor for torchaudio.save
            audio_tensor = torch.FloatTensor(audio).unsqueeze(0)  # (1, num_samples)
            
            save_path = out_dir / f"epoch_{epoch:03d}_sample_{i}.wav"
            torchaudio.save(str(save_path), audio_tensor, sample_rate)
            
        except Exception as e:
            warnings.warn(f"Failed to generate audio for sample {i} at epoch {epoch}: {e}")
            continue

    # Restore training mode so caller can continue optimization immediately.
    model.train()