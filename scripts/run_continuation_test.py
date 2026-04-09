r"""
run_continuation_test.py
Automates the continuation test pipeline:
1) Build a temporary continuation config
2) Resume training from a checkpoint
3) Read latest duration diagnostics from TensorBoard logs
4) Run HiFi-GAN inference on a fixed sentence
5) Evaluate generated audio and print key metrics

Usage:
  venv\Scripts\python.exe scripts\run_continuation_test.py

Example:
  venv\Scripts\python.exe scripts\run_continuation_test.py \
    --epochs 3 \
    --resume-checkpoint checkpoints_sanity/checkpoint_step_003000.pt \
    --output tts_output/hifigan_continue_auto.wav
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, Tuple

import yaml
from tensorboard.backend.event_processing import event_accumulator


ROOT = Path(__file__).resolve().parents[1]


def run_cmd(cmd: list[str]) -> str:
    """Run a command from repo root and return combined stdout/stderr."""
    print("\n$", " ".join(cmd))
    result = subprocess.run(
        cmd,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=True,
    )
    print(result.stdout)
    return result.stdout


def build_config(args: argparse.Namespace) -> Path:
    base_config_path = ROOT / args.base_config
    out_config_path = ROOT / args.out_config

    with open(base_config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg["num_epochs"] = args.epochs
    cfg["log_dir"] = args.log_dir
    cfg["checkpoint_dir"] = args.checkpoint_dir
    cfg["duration_token_alpha"] = args.alpha
    cfg["duration_sum_beta"] = args.beta

    with open(out_config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    return out_config_path


def latest_event_file(log_dir: Path) -> Path:
    event_files = sorted(log_dir.rglob("events.out.tfevents.*"), key=lambda p: p.stat().st_mtime)
    if not event_files:
        raise FileNotFoundError(f"No TensorBoard event files found under {log_dir}")
    return event_files[-1]


def read_latest_scalars(log_dir: Path) -> Dict[str, Tuple[int, float]]:
    event_file = latest_event_file(log_dir)
    ea = event_accumulator.EventAccumulator(str(event_file))
    ea.Reload()

    tags = [
        "train/token_duration_mae",
        "val/token_duration_mae",
        "train/sum_error_mean",
        "val/sum_error_mean",
        "train/pred_speech_rate_proxy",
        "val/pred_speech_rate_proxy",
    ]

    out: Dict[str, Tuple[int, float]] = {}
    for tag in tags:
        items = ea.Scalars(tag)
        if not items:
            continue
        last = items[-1]
        out[tag] = (int(last.step), float(last.value))

    return out


def parse_eval_metrics(eval_output: str) -> Dict[str, str]:
    patterns = {
        "duration": r"Duration\s*:\s*([0-9.]+\s*s)",
        "zcr": r"Zero crossing rate\s*:\s*([0-9.]+)",
        "flatness": r"Spectral flatness\s*:\s*([0-9.]+)",
    }

    metrics: Dict[str, str] = {}
    for key, pattern in patterns.items():
        m = re.search(pattern, eval_output)
        if m:
            metrics[key] = m.group(1)

    verdict_lines = re.findall(r"^\s*•\s*(.+)$", eval_output, flags=re.MULTILINE)
    if verdict_lines:
        metrics["verdict"] = " | ".join(verdict_lines)

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run continuation train+infer+eval test pipeline")
    parser.add_argument("--base-config", default="vits_config.yaml")
    parser.add_argument("--out-config", default="vits_config.continue_auto.yaml")
    parser.add_argument("--resume-checkpoint", default="checkpoints_sanity/checkpoint_step_003000.pt")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--device", default="cuda")

    parser.add_argument("--log-dir", default="./logs/continue_auto")
    parser.add_argument("--checkpoint-dir", default="./checkpoints_continue_auto")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.2)

    parser.add_argument("--text", default="we are at present concerned")
    parser.add_argument("--output", default="tts_output/hifigan_continue_auto.wav")
    parser.add_argument("--sample-rate", type=int, default=22050)
    parser.add_argument("--vocoder-checkpoint", default="hifigan/generator_v1")
    parser.add_argument("--vocoder-config", default="hifigan/config_v1.json")

    args = parser.parse_args()

    resume_ckpt = ROOT / args.resume_checkpoint
    if not resume_ckpt.exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {resume_ckpt}")

    if not (ROOT / args.vocoder_checkpoint).exists():
        raise FileNotFoundError(f"Vocoder checkpoint not found: {ROOT / args.vocoder_checkpoint}")
    if not (ROOT / args.vocoder_config).exists():
        raise FileNotFoundError(f"Vocoder config not found: {ROOT / args.vocoder_config}")

    out_cfg = build_config(args)

    # 1) Resume training
    train_cmd = [
        sys.executable,
        "train_vits.py",
        "--config",
        str(out_cfg.relative_to(ROOT)),
        "--checkpoint",
        args.resume_checkpoint,
        "--device",
        args.device,
    ]
    train_output = run_cmd(train_cmd)

    # 2) Read latest diagnostics
    scalars = read_latest_scalars(ROOT / args.log_dir)

    # 3) Run HiFi-GAN inference
    best_ckpt = Path(args.checkpoint_dir) / "best_model.pt"
    infer_cmd = [
        sys.executable,
        "inference_vits.py",
        "--checkpoint",
        str(best_ckpt).replace("\\", "/"),
        "--config",
        str(out_cfg.relative_to(ROOT)),
        "--vocoder_checkpoint",
        args.vocoder_checkpoint,
        "--vocoder_config",
        args.vocoder_config,
        "--text",
        args.text,
        "--output",
        args.output,
        "--device",
        "cpu",
    ]
    run_cmd(infer_cmd)

    # 4) Evaluate generated audio
    eval_cmd = [
        sys.executable,
        "scripts/evaluate_tts_output.py",
        "--audio",
        args.output,
        "--sample_rate",
        str(args.sample_rate),
    ]
    eval_output = run_cmd(eval_cmd)
    eval_metrics = parse_eval_metrics(eval_output)

    # 5) Print compact summary
    print("\n" + "=" * 70)
    print("CONTINUATION TEST SUMMARY")
    print("=" * 70)

    mid_line = "(not found)"
    for line in train_output.splitlines():
        if "loss=" in line and "Epoch" in line:
            mid_line = line.strip()
            break

    val_line = "(not found)"
    for line in train_output.splitlines():
        if "Validation loss" in line:
            val_line = line.strip()

    print(f"Mid-run line: {mid_line}")
    print(f"Validation line: {val_line}")
    print("\nLatest duration diagnostics:")
    for tag in [
        "train/token_duration_mae",
        "val/token_duration_mae",
        "train/sum_error_mean",
        "val/sum_error_mean",
        "train/pred_speech_rate_proxy",
        "val/pred_speech_rate_proxy",
    ]:
        if tag in scalars:
            step, value = scalars[tag]
            print(f"- {tag}: step={step}, value={value:.10f}")
        else:
            print(f"- {tag}: not found")

    print("\nFinal HiFi-GAN metrics:")
    print(f"- Duration: {eval_metrics.get('duration', 'n/a')}")
    print(f"- ZCR: {eval_metrics.get('zcr', 'n/a')}")
    print(f"- Spectral flatness: {eval_metrics.get('flatness', 'n/a')}")
    print(f"- Verdict: {eval_metrics.get('verdict', 'n/a')}")


if __name__ == "__main__":
    main()
