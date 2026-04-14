"""Thin CLI wrapper for HexTTs training."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
# Allow direct script execution without requiring editable installation.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hextts.config import load_config
from hextts.training import run_training


def _pick_device(device_name: str) -> torch.device:
    """Resolve a runtime device with graceful CUDA fallback."""
    if device_name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main() -> None:
    # This wrapper intentionally keeps orchestration minimal and delegates to package code.
    parser = argparse.ArgumentParser(description="HexTTs train wrapper")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument("--checkpoint", type=str, default=None, help="Resume checkpoint")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    args = parser.parse_args()

    # Centralized config loader keeps train/infer behavior consistent.
    config = load_config(args.config)
    device = _pick_device(args.device)

    # Trainer implementation and checkpoint semantics are package-owned.
    run_training(config=config, device=device, checkpoint=args.checkpoint)


if __name__ == "__main__":
    main()
