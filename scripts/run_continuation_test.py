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
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import yaml
from tensorboard.backend.event_processing import event_accumulator


ROOT = Path(__file__).resolve().parents[1]


def run_cmd(cmd: list[str]) -> str:
    """Run a command from repo root and stream combined stdout/stderr."""
    print("\n$", " ".join(cmd))
    child_env = os.environ.copy()
    child_env["PYTHONUNBUFFERED"] = "1"

    process = subprocess.Popen(
        cmd,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=child_env,
        text=False,
        bufsize=0,
    )

    output_parts = bytearray()
    assert process.stdout is not None

    # Byte-level passthrough keeps carriage-return tqdm updates smooth.
    while True:
        chunk = process.stdout.read(1)
        if not chunk:
            break
        output_parts.extend(chunk)
        try:
            sys.stdout.buffer.write(chunk)
            sys.stdout.buffer.flush()
        except Exception:
            sys.stdout.write(chunk.decode("utf-8", errors="replace"))
            sys.stdout.flush()

    output_text = output_parts.decode("utf-8", errors="replace")

    return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, cmd, output=output_text)

    return output_text


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
    cfg["duration_debug_checks"] = bool(args.duration_debug_checks)

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


def extract_training_snapshots(train_output: str) -> List[str]:
    """Extract lines that contain tqdm loss/recon/kl/dur snapshots."""
    snapshots: List[str] = []
    for line in train_output.splitlines():
        if "loss=" in line and "recon=" in line and "kl=" in line and "dur=" in line:
            snapshots.append(line.strip())
    return snapshots


def build_summary_text(
    mid_line: str,
    val_line: str,
    scalars: Dict[str, Tuple[int, float]],
    eval_metrics: Dict[str, str],
) -> str:
    lines: List[str] = []
    lines.append("=" * 70)
    lines.append("CONTINUATION TEST SUMMARY")
    lines.append("=" * 70)
    lines.append(f"Mid-run line: {mid_line}")
    lines.append(f"Validation line: {val_line}")
    lines.append("")
    lines.append("Latest duration diagnostics:")
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
            lines.append(f"- {tag}: step={step}, value={value:.10f}")
        else:
            lines.append(f"- {tag}: not found")

    lines.append("")
    lines.append("Final HiFi-GAN metrics:")
    lines.append(f"- Duration: {eval_metrics.get('duration', 'n/a')}")
    lines.append(f"- ZCR: {eval_metrics.get('zcr', 'n/a')}")
    lines.append(f"- Spectral flatness: {eval_metrics.get('flatness', 'n/a')}")
    lines.append(f"- Verdict: {eval_metrics.get('verdict', 'n/a')}")
    return "\n".join(lines)


def write_report(
    report_path: Path,
    train_output: str,
    eval_output: str,
    summary_text: str,
) -> None:
    """Write a plain-text continuation report with training/eval/summary sections."""
    report_path.parent.mkdir(parents=True, exist_ok=True)

    snapshots = extract_training_snapshots(train_output)

    report_lines: List[str] = []
    report_lines.append("HexTTs Continuation Test Report")
    report_lines.append("=" * 70)
    report_lines.append("")
    report_lines.append("TRAINING SNAPSHOTS (loss, recon, kl, dur)")
    report_lines.append("-" * 70)

    if snapshots:
        report_lines.append("First snapshot:")
        report_lines.append(snapshots[0])
        report_lines.append("")
        report_lines.append("Last snapshot:")
        report_lines.append(snapshots[-1])
        report_lines.append("")
        report_lines.append("Recent snapshots:")
        for line in snapshots[-10:]:
            report_lines.append(line)
    else:
        report_lines.append("No tqdm snapshots found in training output.")

    report_lines.append("")
    report_lines.append("HexTTS Output Evaluation Report")
    report_lines.append("-" * 70)
    report_lines.append(eval_output.strip())
    report_lines.append("")
    report_lines.append(summary_text)
    report_lines.append("")

    report_path.write_text("\n".join(report_lines), encoding="utf-8")


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
    parser.add_argument("--duration-debug-checks", action="store_true")

    parser.add_argument("--text", default="we are at present concerned")
    parser.add_argument("--output", default="tts_output/hifigan_continue_auto.wav")
    parser.add_argument("--sample-rate", type=int, default=22050)
    parser.add_argument("--vocoder-checkpoint", default="hifigan/generator_v1")
    parser.add_argument("--vocoder-config", default="hifigan/config_v1.json")
    parser.add_argument("--report-file", default="reports/continuation_test_report.txt")

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
    summary_text = build_summary_text(mid_line, val_line, scalars, eval_metrics)
    print(summary_text)

    report_path = ROOT / args.report_file
    write_report(
        report_path=report_path,
        train_output=train_output,
        eval_output=eval_output,
        summary_text=summary_text,
    )
    print(f"\nReport saved to {report_path}")


if __name__ == "__main__":
    main()
