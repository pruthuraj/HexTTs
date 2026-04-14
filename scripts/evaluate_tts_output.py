"""CLI utility for objective evaluation of generated audio files."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hextts.evaluation import (
    collect_audio_files,
    evaluate_audio,
    print_batch_summary,
    print_report,
)


def main():
    """CLI entrypoint for single-file or batch wav evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate HexTTS generated wav file(s)")
    parser.add_argument(
        "--audio",
        type=str,
        default=".",
        help="Path to wav file or directory containing wav files (default: current directory)",
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=None,
        help="Optional resample target (applies to all files)",
    )
    args = parser.parse_args()

    # Resolve one file or many files from an input directory.
    input_path = Path(args.audio)
    audio_files = collect_audio_files(input_path)

    if input_path.is_dir():
        print(f"Found {len(audio_files)} .wav file(s) to evaluate\n")

    all_reports = []
    for audio_path in audio_files:
        try:
            # Evaluate each file independently so one failure does not kill the full batch.
            report = evaluate_audio(str(audio_path), sr_target=args.sample_rate)
            all_reports.append(report)
            print_report(report)
        except Exception as e:
            print(f"\n Error evaluating {audio_path}: {e}\n")

    print_batch_summary(all_reports)


if __name__ == "__main__":
    main()