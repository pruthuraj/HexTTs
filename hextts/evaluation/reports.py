"""Report rendering helpers for audio evaluation."""

from __future__ import annotations

from pathlib import Path


def print_report(report: dict):
    """Render a detailed single-file evaluation report to stdout."""
    print("\n" + "=" * 60)
    print(" HexTTS Output Evaluation Report ")
    print("=" * 60)
    print(f"File               : {report['file']}")
    print(f"Sample rate        : {report['sample_rate']} Hz")
    print(f"Samples            : {report['num_samples']:,}")
    print(f"Duration           : {report['duration_sec']:>6.4f} s")
    print(f"Peak amplitude     : {report['peak_amplitude']:>6.6f}")
    print(f"RMS energy         : {report['rms_energy']:>6.6f}")
    print(f"Silence ratio      : {report['silence_ratio']:>6.6f}")
    print(f"Zero crossing rate : {report['zero_crossing_rate']:>6.6f}")
    print(f"Spectral flatness  : {report['spectral_flatness']:>6.6f}")
    print(f"Mel frames         : {report['mel_frames']:>6d}")
    print(f"Mel mean dB        : {report['mel_mean_db']:>6.4f} dB")
    print(f"Mel std dB         : {report['mel_std_db']:>6.4f} dB")
    print("\n Verdict:")
    for v in report["verdict"]:
        print(f"  - {v}")
    print("-" * 60)


def print_batch_summary(all_reports: list[dict]):
    """Render compact comparison table when multiple files were evaluated."""
    if len(all_reports) <= 1:
        return

    print("\n" + "=" * 60)
    print("BATCH EVALUATION SUMMARY")
    print("=" * 60)
    for i, report in enumerate(all_reports, 1):
        # Keep summary one-line per file for quick side-by-side scanning.
        filename = Path(report["file"]).name
        duration = report["duration_sec"]
        rms = report["rms_energy"]
        zcr = report["zero_crossing_rate"]
        flatness = report["spectral_flatness"]
        print(f"{i}. {filename:40s} | {duration:6.3f}s | RMS: {rms:.4f} | ZCR: {zcr:.4f} | Flat: {flatness:.4f}")
