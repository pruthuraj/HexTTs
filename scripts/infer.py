"""Thin CLI wrapper for HexTTs inference."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hextts.config import load_config
from hextts.inference import VITSInferencePipeline


def _pick_device(device_name: str) -> torch.device:
    if device_name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main() -> None:
    parser = argparse.ArgumentParser(description="HexTTs inference wrapper")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--output", type=str, default="output.wav")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--duration_scale", type=float, default=1.0)
    parser.add_argument("--noise_scale", type=float, default=0.3)
    parser.add_argument("--vocoder_checkpoint", type=str, default=None)
    parser.add_argument("--vocoder_config", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    device = _pick_device(args.device)

    pipeline = VITSInferencePipeline(
        checkpoint_path=args.checkpoint,
        config=config,
        device=device,
        vocoder_checkpoint=args.vocoder_checkpoint,
        vocoder_config=args.vocoder_config,
    )

    audio, sr = pipeline.synthesize(
        text=args.text,
        duration_scale=args.duration_scale,
        noise_scale=args.noise_scale,
    )

    sf.write(args.output, audio, sr)
    print(f"Audio saved to {args.output}")


if __name__ == "__main__":
    main()
