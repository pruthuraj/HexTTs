"""
main_flow.py
Simple command wrappers for the common HexTTs workflow.

Usage examples:
  python scripts/main_flow.py train --device cuda
  python scripts/main_flow.py infer --text "hello world" --hifigan
  python scripts/main_flow.py eval --audio tts_output/hifigan_test.wav
    python scripts/main_flow.py audit --dry-run
  python scripts/main_flow.py compare --text "we are at present concerned"
    python scripts/main_flow.py continuation-test --epochs 3
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run_cmd(args: list[str]) -> None:
    """Run a command from repository root and stream output."""
    print("\n$", " ".join(args))
    subprocess.run(args, cwd=ROOT, check=True)


def cmd_train(args: argparse.Namespace) -> None:
    cmd = [
        sys.executable,
        "train_vits.py",
        "--config",
        args.config,
        "--device",
        args.device,
    ]

    if args.checkpoint:
        cmd.extend(["--checkpoint", args.checkpoint])

    run_cmd(cmd)


def cmd_infer(args: argparse.Namespace) -> None:
    cmd = [
        sys.executable,
        "inference_vits.py",
        "--checkpoint",
        args.checkpoint,
        "--config",
        args.config,
        "--text",
        args.text,
        "--output",
        args.output,
        "--device",
        args.device,
        "--duration_scale",
        str(args.duration_scale),
        "--noise_scale",
        str(args.noise_scale),
    ]

    if args.hifigan:
        cmd.extend(["--vocoder_checkpoint", args.vocoder_checkpoint])
        cmd.extend(["--vocoder_config", args.vocoder_config])

    run_cmd(cmd)


def cmd_eval(args: argparse.Namespace) -> None:
    cmd = [
        sys.executable,
        "scripts/evaluate_tts_output.py",
        "--audio",
        args.audio,
        "--sample_rate",
        str(args.sample_rate),
    ]
    run_cmd(cmd)


def cmd_audit(args: argparse.Namespace) -> None:
    cmd = [
        sys.executable,
        "scripts/audit_dataset.py",
        "--config",
        args.config,
    ]

    if args.dry_run:
        cmd.append("--dry-run")
    if args.min_duration is not None:
        cmd.extend(["--min-duration", str(args.min_duration)])
    if args.max_duration is not None:
        cmd.extend(["--max-duration", str(args.max_duration)])
    if args.min_rms_db is not None:
        cmd.extend(["--min-rms-db", str(args.min_rms_db)])
    if args.max_silence is not None:
        cmd.extend(["--max-silence", str(args.max_silence)])
    if args.no_clip_check:
        cmd.append("--no-clip-check")

    run_cmd(cmd)


def cmd_compare(args: argparse.Namespace) -> None:
    gl_output = args.gl_output
    hifigan_output = args.hifigan_output

    # Griffin-Lim baseline
    cmd_infer(
        argparse.Namespace(
            checkpoint=args.checkpoint,
            config=args.config,
            text=args.text,
            output=gl_output,
            device=args.device,
            duration_scale=args.duration_scale,
            noise_scale=args.noise_scale,
            hifigan=False,
            vocoder_checkpoint=args.vocoder_checkpoint,
            vocoder_config=args.vocoder_config,
        )
    )

    # HiFi-GAN path
    cmd_infer(
        argparse.Namespace(
            checkpoint=args.checkpoint,
            config=args.config,
            text=args.text,
            output=hifigan_output,
            device=args.device,
            duration_scale=args.duration_scale,
            noise_scale=args.noise_scale,
            hifigan=True,
            vocoder_checkpoint=args.vocoder_checkpoint,
            vocoder_config=args.vocoder_config,
        )
    )

    # Evaluate both outputs
    cmd_eval(argparse.Namespace(audio=gl_output, sample_rate=args.sample_rate))
    cmd_eval(argparse.Namespace(audio=hifigan_output, sample_rate=args.sample_rate))


def cmd_continuation_test(args: argparse.Namespace) -> None:
    cmd = [
        sys.executable,
        "scripts/run_continuation_test.py",
        "--base-config",
        args.base_config,
        "--out-config",
        args.out_config,
        "--resume-checkpoint",
        args.resume_checkpoint,
        "--epochs",
        str(args.epochs),
        "--device",
        args.device,
        "--log-dir",
        args.log_dir,
        "--checkpoint-dir",
        args.checkpoint_dir,
        "--alpha",
        str(args.alpha),
        "--beta",
        str(args.beta),
        "--text",
        args.text,
        "--output",
        args.output,
        "--sample-rate",
        str(args.sample_rate),
        "--vocoder-checkpoint",
        args.vocoder_checkpoint,
        "--vocoder-config",
        args.vocoder_config,
    ]
    run_cmd(cmd)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="HexTTs simplified workflow runner",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # train
    p_train = subparsers.add_parser("train", help="Run model training")
    p_train.add_argument("--config", default="vits_config.yaml")
    p_train.add_argument("--device", default="cuda")
    p_train.add_argument("--checkpoint", default=None)
    p_train.set_defaults(func=cmd_train)

    # infer
    p_infer = subparsers.add_parser("infer", help="Run text-to-speech inference")
    p_infer.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    p_infer.add_argument("--config", default="vits_config.yaml")
    p_infer.add_argument("--text", required=True)
    p_infer.add_argument("--output", default="tts_output/output.wav")
    p_infer.add_argument("--device", default="cpu")
    p_infer.add_argument("--duration_scale", type=float, default=1.0)
    p_infer.add_argument("--noise_scale", type=float, default=0.3)
    p_infer.add_argument("--hifigan", action="store_true", help="Enable HiFi-GAN vocoder")
    p_infer.add_argument("--vocoder_checkpoint", default="hifigan/generator_v1")
    p_infer.add_argument("--vocoder_config", default="hifigan/config_v1.json")
    p_infer.set_defaults(func=cmd_infer)

    # eval
    p_eval = subparsers.add_parser("eval", help="Evaluate generated audio")
    p_eval.add_argument("--audio", default="tts_output")
    p_eval.add_argument("--sample_rate", type=int, default=22050)
    p_eval.set_defaults(func=cmd_eval)

    # audit
    p_audit = subparsers.add_parser("audit", help="Audit and filter dataset metadata")
    p_audit.add_argument("--config", default="vits_config.yaml")
    p_audit.add_argument("--dry-run", action="store_true")
    p_audit.add_argument("--min-duration", type=float, default=None)
    p_audit.add_argument("--max-duration", type=float, default=None)
    p_audit.add_argument("--min-rms-db", type=float, default=None)
    p_audit.add_argument("--max-silence", type=float, default=None)
    p_audit.add_argument("--no-clip-check", action="store_true")
    p_audit.set_defaults(func=cmd_audit)

    # compare
    p_compare = subparsers.add_parser(
        "compare",
        help="Generate and evaluate Griffin-Lim vs HiFi-GAN on same text",
    )
    p_compare.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    p_compare.add_argument("--config", default="vits_config.yaml")
    p_compare.add_argument("--text", required=True)
    p_compare.add_argument("--device", default="cpu")
    p_compare.add_argument("--duration_scale", type=float, default=1.0)
    p_compare.add_argument("--noise_scale", type=float, default=0.3)
    p_compare.add_argument("--vocoder_checkpoint", default="hifigan/generator_v1")
    p_compare.add_argument("--vocoder_config", default="hifigan/config_v1.json")
    p_compare.add_argument("--gl_output", default="tts_output/gl_test.wav")
    p_compare.add_argument("--hifigan_output", default="tts_output/hifigan_test.wav")
    p_compare.add_argument("--sample_rate", type=int, default=22050)
    p_compare.set_defaults(func=cmd_compare)

    # continuation-test
    p_test = subparsers.add_parser(
        "continuation-test",
        help="Run continuation train + diagnostics + HiFi-GAN evaluation",
    )
    p_test.add_argument("--base-config", default="vits_config.yaml")
    p_test.add_argument("--out-config", default="vits_config.continue_auto.yaml")
    p_test.add_argument("--resume-checkpoint", default="checkpoints_sanity/checkpoint_step_003000.pt")
    p_test.add_argument("--epochs", type=int, default=3)
    p_test.add_argument("--device", default="cuda")
    p_test.add_argument("--log-dir", default="./logs/continue_auto")
    p_test.add_argument("--checkpoint-dir", default="./checkpoints_continue_auto")
    p_test.add_argument("--alpha", type=float, default=1.0)
    p_test.add_argument("--beta", type=float, default=0.2)
    p_test.add_argument("--text", default="we are at present concerned")
    p_test.add_argument("--output", default="tts_output/hifigan_continue_auto.wav")
    p_test.add_argument("--sample-rate", type=int, default=22050)
    p_test.add_argument("--vocoder-checkpoint", default="hifigan/generator_v1")
    p_test.add_argument("--vocoder-config", default="hifigan/config_v1.json")
    p_test.set_defaults(func=cmd_continuation_test)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
