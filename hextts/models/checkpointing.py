"""Checkpoint save/load and compatibility validation helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import torch


def _safe_torch_load(path: str, map_location: torch.device, weights_only: bool) -> Dict[str, Any]:
    """Load checkpoint with best-effort compatibility across torch versions."""
    try:
        return torch.load(path, map_location=map_location, weights_only=weights_only)
    except TypeError:
        # Backward compatibility for older torch that does not support weights_only.
        return torch.load(path, map_location=map_location)


def build_checkpoint_metadata(config: Dict[str, Any], model_version: str) -> Dict[str, Any]:
    """Build normalized metadata persisted with every checkpoint."""
    return {
        "model_version": model_version,
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "vocab_size": config.get("vocab_size"),
        "sample_rate": config.get("sample_rate"),
        "n_mels": config.get("n_mel_channels"),
        "architecture_flags": {
            "use_postnet": bool(config.get("use_postnet", True)),
            "duration_clamp": float(config.get("max_duration_value", 20.0)),
        },
    }


def validate_checkpoint_compatibility(checkpoint: Dict[str, Any], config: Dict[str, Any]) -> None:
    """Validate critical checkpoint metadata against runtime config."""
    # These fields directly affect tensor shapes and audio contracts.
    expected = {
        "vocab_size": config.get("vocab_size"),
        "sample_rate": config.get("sample_rate"),
        "n_mels": config.get("n_mel_channels"),
    }

    for key, exp in expected.items():
        got = checkpoint.get(key)
        if got is not None and exp is not None and got != exp:
            raise ValueError(f"Checkpoint mismatch for {key}: expected {exp}, got {got}")

    # Validate known architecture flags when present.
    # This protects against silent runtime behavior drift after refactors.
    ckpt_flags = checkpoint.get("architecture_flags")
    if not isinstance(ckpt_flags, dict):
        return

    expected_flags = {
        "use_postnet": bool(config.get("use_postnet", True)),
        "duration_clamp": float(config.get("max_duration_value", 20.0)),
    }

    for key, exp in expected_flags.items():
        got = ckpt_flags.get(key)
        if got is not None and got != exp:
            raise ValueError(
                f"Checkpoint mismatch for architecture flag '{key}': expected {exp}, got {got}"
            )


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[Any],
    scaler: Optional[Any],
    epoch: int,
    global_step: int,
    config: Dict[str, Any],
    model_version: str = "v0.5.0",
    git_commit: str = "unknown",
) -> None:
    """Persist model and runtime state as a resumable training checkpoint."""
    payload: Dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "config": config,
        "git_commit": git_commit,
    }

    # Store reproducibility metadata alongside model weights.
    payload.update(build_checkpoint_metadata(config, model_version))

    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    if scaler is not None:
        payload["scaler_state_dict"] = scaler.state_dict()

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)


def load_checkpoint(path: str, device: torch.device, allow_legacy_pickle: bool = True) -> Dict[str, Any]:
    """Load checkpoint with safer default behavior."""
    # Prefer weights-only loading first to reduce pickle surface area.
    try:
        ckpt = _safe_torch_load(path, map_location=device, weights_only=True)
    except Exception:
        if not allow_legacy_pickle:
            raise
        # Fallback supports older checkpoints that contain optimizer/scheduler objects.
        ckpt = _safe_torch_load(path, map_location=device, weights_only=False)

    if not isinstance(ckpt, dict):
        raise ValueError(f"Invalid checkpoint payload type: {type(ckpt)}")

    return ckpt
