"""Utility for writing quick qualitative sample WAVs during training."""

from pathlib import Path
import torch
import torchaudio


@torch.no_grad()
def generate_samples(
    model,
    texts,
    text_to_sequence_fn,
    output_dir: str,
    epoch: int,
    device: str = "cuda",
    sample_rate: int = 22050,
):
    """Run model inference for each prompt and save numbered WAV files."""
    model.eval()
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, text in enumerate(texts, start=1):
        # Convert text into model input IDs using caller-provided tokenizer.
        seq = text_to_sequence_fn(text)
        x = torch.LongTensor(seq).unsqueeze(0).to(device)
        x_lengths = torch.LongTensor([x.size(1)]).to(device)

        # Model returns audio-like output in project-specific layout.
        audio = model.inference(x, x_lengths)[0]

        if isinstance(audio, tuple):
            audio = audio[0]

        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        save_path = out_dir / f"epoch_{epoch:03d}_sample_{i}.wav"
        torchaudio.save(str(save_path), audio.cpu(), sample_rate)

    # Restore training mode so caller can continue optimization immediately.
    model.train()