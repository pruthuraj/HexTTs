"""Thin CLI wrapper for HexTTs inference."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
# Allow running this file directly from the repo root without package install.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hextts.config import load_config
from hextts.inference import VITSInferencePipeline


def _pick_device(device_name: str) -> torch.device:
    """Resolve the requested device with a safe CUDA fallback to CPU."""
    if device_name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main() -> None:
    # Keep the wrapper small: parse args, load config, call package pipeline.
    parser = argparse.ArgumentParser(description="HexTTs inference wrapper")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--output", type=str, default="output.wav")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--duration_scale",
        type=float,
        default=None,
        help="Override duration scale. Defaults to config inference_duration_scale or 1.0.",
    )
    parser.add_argument("--noise_scale", type=float, default=0.3)
    parser.add_argument("--vocoder_checkpoint", type=str, default=None)
    parser.add_argument("--vocoder_config", type=str, default=None)
    args = parser.parse_args()

    # `load_config` already applies repository defaults when config is omitted.
    config = load_config(args.config)
    device = _pick_device(args.device)

    # Prefer explicit CLI override; otherwise use config default for practical speech pace.
    duration_scale = (
        float(args.duration_scale)
        if args.duration_scale is not None
        else float(config.get("inference_duration_scale", 1.0))
    )

    pipeline = VITSInferencePipeline(
        checkpoint_path=args.checkpoint,
        config=config,
        device=device,
        vocoder_checkpoint=args.vocoder_checkpoint,
        vocoder_config=args.vocoder_config,
    )

    audio, sr = pipeline.synthesize(
        text=args.text,
        duration_scale=duration_scale,
        noise_scale=args.noise_scale,
    )

    # Persist final waveform as a standard WAV file.
    sf.write(args.output, audio, sr)
    print(f"Audio saved to {args.output}")


if __name__ == "__main__":
    main()
