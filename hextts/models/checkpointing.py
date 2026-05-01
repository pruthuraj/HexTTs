"""Checkpoint save/load and compatibility validation helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import torch




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
    """Validate critical checkpoint metadata against runtime config.

    Returns None on success. Architecture-flag validation is silently skipped
    when the checkpoint pre-dates the flags block — older checkpoints will load
    but won't get the extra safety net.

    Raises:
        ValueError: if vocab_size, sample_rate, n_mels, or any architecture
            flag in the checkpoint disagrees with the active config.
    """
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
    """Persist model and runtime state as a resumable training checkpoint.

    Creates parent directories as needed. Optimizer/scheduler/scaler state are
    written only when supplied, so the same function serves both periodic
    resumable saves and weights-only export.
    """
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


def load_checkpoint(path: str, device: torch.device) -> Dict[str, Any]:
    """Load checkpoint using safe weights-only deserialization.

    Requires torch >= 1.13. If loading fails, the checkpoint may have been saved
    with an older torch version. Re-save it using save_checkpoint() before loading.
    """
    try:
        ckpt = torch.load(path, map_location=device, weights_only=True)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load checkpoint '{path}' safely.\n"
            "This usually means the checkpoint was saved with an older torch version "
            "that pickled non-tensor objects.\n"
            "To migrate: load with weights_only=False once in a trusted environment, "
            "then re-save using save_checkpoint()."
        ) from exc

    if not isinstance(ckpt, dict):
        raise ValueError(f"Invalid checkpoint payload type: {type(ckpt)}")

    return ckpt
