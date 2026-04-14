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
  venv\Scripts\python.exe scripts\run_continuation_test.py ^
    --epochs 3 ^
    --resume-checkpoint checkpoints_sanity/checkpoint_step_003000.pt ^
    --output tts_output/hifigan_continue_auto.wav
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import yaml
from tensorboard.backend.event_processing import event_accumulator


ROOT = Path(__file__).resolve().parents[1]

SCALAR_TAGS: tuple[str, ...] = (
    "train/token_duration_mae",
    "val/token_duration_mae",
    "train/sum_error_mean",
    "val/sum_error_mean",
    "train/pred_speech_rate_proxy",
    "val/pred_speech_rate_proxy",
)


@dataclass(frozen=True)
class ScalarPoint:
    step: int
    value: float


@dataclass(frozen=True)
class EvalMetrics:
    duration: str = "n/a"
    zcr: str = "n/a"
    flatness: str = "n/a"
    verdict: str = "n/a"


@dataclass(frozen=True)
class PipelineArtifacts:
    config_path: Path
    report_path: Path
    best_checkpoint_path: Path
    output_audio_path: Path


def repo_path(path_str: str) -> Path:
    """Resolve a user-provided path relative to repo root."""
    return (ROOT / path_str).resolve()


def require_file(path: Path, label: str) -> None:
    """Raise a helpful error if a required file does not exist."""
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"{label} is not a file: {path}")


def ensure_parent_dir(path: Path) -> None:
    """Create parent directory for a file path if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)


def path_for_cli(path: Path) -> str:
    """
    Convert a path into a stable CLI-friendly repo-relative path when possible.
    Falls back to absolute path if relative conversion is not possible.
    """
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def run_cmd(cmd: list[str], *, cwd: Path = ROOT) -> str:
    """
    Run a command, stream stdout/stderr live, and return full captured output.
    Raises CalledProcessError on non-zero exit.
    """
    print("\n$", " ".join(cmd))

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    process = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        text=False,
        bufsize=0,
    )

    if process.stdout is None:
        raise RuntimeError("Failed to capture subprocess stdout.")

    output_buffer = bytearray()

    while True:
        chunk = process.stdout.read(1)
        if not chunk:
            break

        output_buffer.extend(chunk)

        try:
            sys.stdout.buffer.write(chunk)
            sys.stdout.buffer.flush()
        except Exception:
            sys.stdout.write(chunk.decode("utf-8", errors="replace"))
            sys.stdout.flush()

    output_text = output_buffer.decode("utf-8", errors="replace")
    return_code = process.wait()

    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, cmd, output=output_text)

    return output_text


def load_yaml(path: Path) -> dict:
    require_file(path, "Base config")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"YAML config must load to a dictionary: {path}")

    return data


def save_yaml(data: dict, path: Path) -> None:
    ensure_parent_dir(path)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def build_config(args: argparse.Namespace) -> Path:
    """Create a temporary continuation config derived from the base config."""
    base_config_path = repo_path(args.base_config)
    out_config_path = repo_path(args.out_config)

    cfg = load_yaml(base_config_path)

    cfg_updates = {
        "num_epochs": args.epochs,
        "log_dir": args.log_dir,
        "checkpoint_dir": args.checkpoint_dir,
        "duration_token_alpha": args.alpha,
        "duration_sum_beta": args.beta,
        "duration_debug_checks": bool(args.duration_debug_checks),
    }
    cfg.update(cfg_updates)

    save_yaml(cfg, out_config_path)
    return out_config_path


def find_latest_event_file(log_dir: Path) -> Path:
    """Return the most recently modified TensorBoard event file under log_dir."""
    if not log_dir.exists():
        raise FileNotFoundError(f"Log directory not found: {log_dir}")

    event_files = sorted(
        log_dir.rglob("events.out.tfevents.*"),
        key=lambda p: p.stat().st_mtime,
    )
    if not event_files:
        raise FileNotFoundError(f"No TensorBoard event files found under: {log_dir}")

    return event_files[-1]


def read_latest_scalars(log_dir: Path, tags: Iterable[str] = SCALAR_TAGS) -> Dict[str, ScalarPoint]:
    """Read the most recent scalar value for each requested TensorBoard tag."""
    event_file = find_latest_event_file(log_dir)
    accumulator = event_accumulator.EventAccumulator(str(event_file))
    accumulator.Reload()

    available_tags = set(accumulator.Tags().get("scalars", []))
    results: Dict[str, ScalarPoint] = {}

    for tag in tags:
        if tag not in available_tags:
            continue

        values = accumulator.Scalars(tag)
        if not values:
            continue

        last = values[-1]
        results[tag] = ScalarPoint(step=int(last.step), value=float(last.value))

    return results


def parse_eval_metrics(eval_output: str) -> EvalMetrics:
    """Parse key metrics from evaluate_tts_output.py console output."""
    patterns = {
        "duration": r"Duration\s*:\s*([0-9.]+\s*s)",
        "zcr": r"Zero crossing rate\s*:\s*([0-9.]+)",
        "flatness": r"Spectral flatness\s*:\s*([0-9.]+)",
    }

    extracted: dict[str, str] = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, eval_output)
        if match:
            extracted[key] = match.group(1)

    verdict_lines = re.findall(r"^\s*•\s*(.+)$", eval_output, flags=re.MULTILINE)
    if verdict_lines:
        extracted["verdict"] = " | ".join(verdict_lines)

    return EvalMetrics(
        duration=extracted.get("duration", "n/a"),
        zcr=extracted.get("zcr", "n/a"),
        flatness=extracted.get("flatness", "n/a"),
        verdict=extracted.get("verdict", "n/a"),
    )


def extract_training_snapshots(train_output: str) -> List[str]:
    """Extract tqdm-style training snapshot lines containing loss breakdowns."""
    snapshots: List[str] = []

    for raw_line in train_output.splitlines():
        line = raw_line.strip()
        if all(token in line for token in ("loss=", "recon=", "kl=", "dur=")):
            snapshots.append(line)

    return snapshots


def find_first_matching_line(text: str, required_tokens: Iterable[str]) -> str:
    """Return the first line containing all required tokens, or '(not found)'."""
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if all(token in line for token in required_tokens):
            return line
    return "(not found)"


def find_last_matching_line(text: str, required_tokens: Iterable[str]) -> str:
    """Return the last line containing all required tokens, or '(not found)'."""
    found = "(not found)"
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if all(token in line for token in required_tokens):
            found = line
    return found


def build_summary_text(
    mid_line: str,
    val_line: str,
    scalars: Dict[str, ScalarPoint],
    eval_metrics: EvalMetrics,
) -> str:
    """Build the compact text summary shown in console and report."""
    lines: List[str] = [
        "=" * 70,
        "CONTINUATION TEST SUMMARY",
        "=" * 70,
        f"Mid-run line: {mid_line}",
        f"Validation line: {val_line}",
        "",
        "Latest duration diagnostics:",
    ]

    for tag in SCALAR_TAGS:
        if tag in scalars:
            point = scalars[tag]
            lines.append(f"- {tag}: step={point.step}, value={point.value:.10f}")
        else:
            lines.append(f"- {tag}: not found")

    lines.extend(
        [
            "",
            "Final HiFi-GAN metrics:",
            f"- Duration: {eval_metrics.duration}",
            f"- ZCR: {eval_metrics.zcr}",
            f"- Spectral flatness: {eval_metrics.flatness}",
            f"- Verdict: {eval_metrics.verdict}",
        ]
    )

    return "\n".join(lines)


def write_report(
    report_path: Path,
    train_output: str,
    eval_output: str,
    summary_text: str,
) -> None:
    """Write a plain-text report containing training snapshots, eval output, and summary."""
    ensure_parent_dir(report_path)

    snapshots = extract_training_snapshots(train_output)

    report_lines: List[str] = [
        "HexTTs Continuation Test Report",
        "=" * 70,
        "",
        "TRAINING SNAPSHOTS (loss, recon, kl, dur)",
        "-" * 70,
    ]

    if snapshots:
        report_lines.extend(
            [
                "First snapshot:",
                snapshots[0],
                "",
                "Last snapshot:",
                snapshots[-1],
                "",
                "Recent snapshots:",
                *snapshots[-10:],
            ]
        )
    else:
        report_lines.append("No tqdm snapshots found in training output.")

    report_lines.extend(
        [
            "",
            "HexTTS Output Evaluation Report",
            "-" * 70,
            eval_output.strip(),
            "",
            summary_text,
            "",
        ]
    )

    report_path.write_text("\n".join(report_lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run continuation train + infer + eval pipeline."
    )

    parser.add_argument("--base-config", default="vits_config.yaml")
    parser.add_argument("--out-config", default="vits_config.continue_auto.yaml")
    parser.add_argument(
        "--resume-checkpoint",
        default="checkpoints_sanity/checkpoint_step_003000.pt",
    )
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

    return parser.parse_args()


def validate_inputs(args: argparse.Namespace) -> PipelineArtifacts:
    resume_checkpoint = repo_path(args.resume_checkpoint)
    vocoder_checkpoint = repo_path(args.vocoder_checkpoint)
    vocoder_config = repo_path(args.vocoder_config)

    require_file(resume_checkpoint, "Resume checkpoint")
    require_file(vocoder_checkpoint, "Vocoder checkpoint")
    require_file(vocoder_config, "Vocoder config")

    report_path = repo_path(args.report_file)
    output_audio_path = repo_path(args.output)
    best_checkpoint_path = repo_path(str(Path(args.checkpoint_dir) / "best_model.pt"))
    config_path = repo_path(args.out_config)

    return PipelineArtifacts(
        config_path=config_path,
        report_path=report_path,
        best_checkpoint_path=best_checkpoint_path,
        output_audio_path=output_audio_path,
    )


def run_training(args: argparse.Namespace, config_path: Path) -> str:
    train_cmd = [
        sys.executable,
        "train_vits.py",
        "--config",
        path_for_cli(config_path),
        "--checkpoint",
        args.resume_checkpoint,
        "--device",
        args.device,
    ]
    return run_cmd(train_cmd)


def run_inference(args: argparse.Namespace, config_path: Path, best_checkpoint_path: Path) -> None:
    infer_cmd = [
        sys.executable,
        "inference_vits.py",
        "--checkpoint",
        path_for_cli(best_checkpoint_path),
        "--config",
        path_for_cli(config_path),
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


def run_evaluation(args: argparse.Namespace) -> str:
    eval_cmd = [
        sys.executable,
        "scripts/evaluate_tts_output.py",
        "--audio",
        args.output,
        "--sample_rate",
        str(args.sample_rate),
    ]
    return run_cmd(eval_cmd)


def main() -> None:
    args = parse_args()

    artifacts = validate_inputs(args)
    config_path = build_config(args)

    train_output = run_training(args, config_path)
    scalars = read_latest_scalars(repo_path(args.log_dir))

    require_file(artifacts.best_checkpoint_path, "Best checkpoint after continuation run")
    run_inference(args, config_path, artifacts.best_checkpoint_path)

    eval_output = run_evaluation(args)
    eval_metrics = parse_eval_metrics(eval_output)

    mid_line = find_first_matching_line(train_output, ("Epoch", "loss="))
    val_line = find_last_matching_line(train_output, ("Validation loss",))
    summary_text = build_summary_text(mid_line, val_line, scalars, eval_metrics)

    print("\n" + summary_text)

    write_report(
        report_path=artifacts.report_path,
        train_output=train_output,
        eval_output=eval_output,
        summary_text=summary_text,
    )
    print(f"\nReport saved to {artifacts.report_path}")


if __name__ == "__main__":
    main()