"""
audit_dataset.py
Dataset audit and filtering tool for HexTTs / LJSpeech pipeline.

Reads train.txt and val.txt from data_dir, loads each audio file from audio_dir,
scores it on five quality axes, and writes filtered output files in exactly the
same format so no other code needs to change.

Usage
-----
    python scripts/audit_dataset.py --config vits_config.yaml

  # Dry-run: see report without writing filtered files
    python scripts/audit_dataset.py --config vits_config.yaml --dry-run

  # Adjust thresholds from the command line
    python scripts/audit_dataset.py --config vits_config.yaml --min-duration 0.4 --max-duration 12.0

Output
------
  data/ljspeech_prepared/train_filtered.txt   (same format as train.txt)
  data/ljspeech_prepared/val_filtered.txt
  data/ljspeech_prepared/audit_report.txt     (full per-sample scores)
"""

import os
import sys
import argparse
import yaml
import numpy as np
import librosa
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from tqdm import tqdm


# ── Per-sample result ──────────────────────────────────────────────────────

@dataclass
class SampleAudit:
    filename: str
    duration_sec: float       = 0.0
    rms_db: float             = -999.0
    silence_ratio: float      = 1.0
    peak_amplitude: float     = 0.0
    is_clipped: bool          = False
    phoneme_count: int        = 0
    phonemes_per_sec: float   = 0.0
    load_error: str           = ""
    kept: bool                = False
    reject_reasons: List[str] = field(default_factory=list)


# ── Thresholds (all overridable via CLI) ───────────────────────────────────

@dataclass
class AuditThresholds:
    # Duration
    min_duration_sec: float = 0.5    # shorter clips are usually truncated/broken
    max_duration_sec: float = 10.0   # longer clips cause extreme padding

    # Energy
    min_rms_db: float = -45.0        # below this is effectively silence
    max_silence_ratio: float = 0.60  # fraction of frames that are near-silent

    # Clipping
    clip_threshold: float = 0.99     # peak amplitude fraction triggering clip flag

    # Transcript-to-audio ratio
    # LJSpeech averages ~10-15 phonemes/sec for normal speech
    min_phonemes_per_sec: float = 3.0   # below = probably mismatched transcript
    max_phonemes_per_sec: float = 30.0  # above = probably mismatched transcript

    # Silence detection energy threshold (fraction of peak RMS per frame)
    silence_energy_threshold: float = 0.02


# ── Core analysis ──────────────────────────────────────────────────────────

def analyse_sample(
    audio_path: str,
    phoneme_str: str,
    thresholds: AuditThresholds,
    sr_target: int = 22050,
) -> SampleAudit:
    """Load one audio file and compute all quality metrics."""

    filename = Path(audio_path).name
    result = SampleAudit(filename=filename)

    # ── Load audio ────────────────────────────────────────────────────
    try:
        audio, sr = librosa.load(audio_path, sr=sr_target, mono=True)
    except Exception as e:
        result.load_error = str(e)
        result.reject_reasons.append(f"load_error: {e}")
        return result

    if len(audio) == 0:
        result.load_error = "empty file"
        result.reject_reasons.append("empty file")
        return result

    # ── Duration ──────────────────────────────────────────────────────
    result.duration_sec = len(audio) / sr

    # ── Peak amplitude / clipping ─────────────────────────────────────
    result.peak_amplitude = float(np.abs(audio).max())
    result.is_clipped = result.peak_amplitude >= thresholds.clip_threshold

    # ── RMS energy ───────────────────────────────────────────────────
    rms_linear = float(np.sqrt(np.mean(audio ** 2)) + 1e-9)
    result.rms_db = 20.0 * np.log10(rms_linear)

    # ── Silence ratio ────────────────────────────────────────────────
    # Frame-level energy, then count how many frames fall below threshold
    frame_length = 512
    hop_length = 256
    rms_frames = librosa.feature.rms(
        y=audio, frame_length=frame_length, hop_length=hop_length
    )[0]
    peak_frame_rms = float(rms_frames.max()) + 1e-9
    silent_frames = (rms_frames < (peak_frame_rms * thresholds.silence_energy_threshold)).sum()
    result.silence_ratio = float(silent_frames) / (len(rms_frames) + 1e-9)

    # ── Phoneme density ───────────────────────────────────────────────
    phonemes = [p for p in phoneme_str.strip().split() if p]
    result.phoneme_count = len(phonemes)
    if result.duration_sec > 0:
        result.phonemes_per_sec = result.phoneme_count / result.duration_sec

    # ── Apply thresholds ──────────────────────────────────────────────
    reasons = []

    if result.duration_sec < thresholds.min_duration_sec:
        reasons.append(
            f"too_short: {result.duration_sec:.2f}s < {thresholds.min_duration_sec}s"
        )
    if result.duration_sec > thresholds.max_duration_sec:
        reasons.append(
            f"too_long: {result.duration_sec:.2f}s > {thresholds.max_duration_sec}s"
        )
    if result.rms_db < thresholds.min_rms_db:
        reasons.append(
            f"too_quiet: {result.rms_db:.1f} dB < {thresholds.min_rms_db} dB"
        )
    if result.silence_ratio > thresholds.max_silence_ratio:
        reasons.append(
            f"too_much_silence: {result.silence_ratio:.2%} > {thresholds.max_silence_ratio:.0%}"
        )
    if result.is_clipped:
        reasons.append(
            f"clipped: peak={result.peak_amplitude:.4f}"
        )
    if result.phoneme_count > 0:
        if result.phonemes_per_sec < thresholds.min_phonemes_per_sec:
            reasons.append(
                f"phoneme_rate_low: {result.phonemes_per_sec:.1f}/s "
                f"(likely transcript mismatch)"
            )
        if result.phonemes_per_sec > thresholds.max_phonemes_per_sec:
            reasons.append(
                f"phoneme_rate_high: {result.phonemes_per_sec:.1f}/s "
                f"(likely transcript mismatch)"
            )

    result.reject_reasons = reasons
    result.kept = (len(reasons) == 0)
    return result


# ── File parsing ───────────────────────────────────────────────────────────

def parse_metadata_line(line: str) -> Tuple[str, str, str]:
    """
    Parse one line from train.txt / val.txt.
    Format: filename|phoneme sequence|optional extra fields

    Some datasets store the first field as:
      LJ001-0001
    instead of:
      LJ001-0001.wav

    Returns:
        (filename_with_extension, phoneme_str, raw_line_without_newline)
    """
    line = line.rstrip("\n")
    parts = line.split("|")

    filename = parts[0].strip() if len(parts) > 0 else ""
    phoneme_str = parts[1].strip() if len(parts) > 1 else ""

    # Add .wav automatically if missing
    if filename and not filename.lower().endswith(".wav"):
        filename += ".wav"

    return filename, phoneme_str, line

# ── Report writing ─────────────────────────────────────────────────────────

def write_report(
    audits: List[SampleAudit],
    split_name: str,
    report_lines: List[str],
) -> None:
    """Append per-sample audit rows to the shared report buffer."""

    kept   = [a for a in audits if a.kept]
    removed = [a for a in audits if not a.kept]

    report_lines.append(f"\n{'='*70}")
    report_lines.append(f"  {split_name}  —  {len(audits)} samples total")
    report_lines.append(f"  Kept:    {len(kept):,}  ({len(kept)/max(1,len(audits)):.1%})")
    report_lines.append(f"  Removed: {len(removed):,}  ({len(removed)/max(1,len(audits)):.1%})")
    report_lines.append(f"{'='*70}")

    # Rejection reason summary
    from collections import Counter
    reason_counts: Counter = Counter()
    for a in removed:
        for r in a.reject_reasons:
            # Bucket by the label before the colon
            label = r.split(":")[0]
            reason_counts[label] += 1

    if reason_counts:
        report_lines.append("\nRejection reasons (samples can fail multiple checks):")
        for label, count in reason_counts.most_common():
            report_lines.append(f"  {label:<35} {count:>6,}")

    # Duration stats for kept samples
    if kept:
        durations = [a.duration_sec for a in kept]
        report_lines.append(
            f"\nDuration of kept samples (s):"
            f"  min={min(durations):.2f}"
            f"  median={np.median(durations):.2f}"
            f"  mean={np.mean(durations):.2f}"
            f"  max={max(durations):.2f}"
        )

    # Load errors (separate from filter failures)
    errors = [a for a in audits if a.load_error]
    if errors:
        report_lines.append(f"\nLoad errors ({len(errors)}):")
        for a in errors[:20]:
            report_lines.append(f"  {a.filename}: {a.load_error}")
        if len(errors) > 20:
            report_lines.append(f"  ... and {len(errors)-20} more")

    # Removed sample list
    if removed:
        report_lines.append(f"\nRemoved samples:")
        for a in removed:
            reasons_str = "; ".join(a.reject_reasons)
            report_lines.append(f"  {a.filename:<25}  {reasons_str}")


# ── Main ──────────────────────────────────────────────────────────────────

def audit_split(
    metadata_path: str,
    audio_dir: str,
    thresholds: AuditThresholds,
    sample_rate: int,
    report_lines: List[str],
    split_name: str,
) -> Tuple[List[str], List[SampleAudit]]:
    """
    Audit one split (train or val).
    Returns (kept_lines, all_audits).
    kept_lines are the raw metadata lines for samples that passed all checks.
    """
    with open(metadata_path, "r", encoding="utf-8") as f:
        raw_lines = f.readlines()

    kept_lines = []
    audits = []

    for line in tqdm(raw_lines, desc=f"Auditing {split_name}", unit="sample"):
        line = line.strip()
        if not line:
            continue

        filename, phoneme_str, raw = parse_metadata_line(line)
        if not filename:
            continue

        audio_path = str(Path(audio_dir) / filename)

        if not os.path.exists(audio_path):
            # File missing entirely — always reject
            a = SampleAudit(filename=filename)
            a.reject_reasons.append("file_not_found")
            audits.append(a)
            continue

        a = analyse_sample(audio_path, phoneme_str, thresholds, sr_target=sample_rate)
        audits.append(a)

        if a.kept:
            kept_lines.append(raw + "\n")

    write_report(audits, split_name, report_lines)
    return kept_lines, audits


def main():
    parser = argparse.ArgumentParser(
        description="Audit and filter LJSpeech-style dataset for TTS training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard run using config defaults
    python scripts/audit_dataset.py --config vits_config.yaml

  # Dry-run: see what would be removed without writing files
    python scripts/audit_dataset.py --config vits_config.yaml --dry-run

  # Stricter duration filter
    python scripts/audit_dataset.py --config vits_config.yaml --min-duration 0.8 --max-duration 8.0

  # Allow more silence (useful for datasets with natural pauses)
    python scripts/audit_dataset.py --config vits_config.yaml --max-silence 0.75
        """,
    )
    parser.add_argument("--config", default="vits_config.yaml",
                        help="Path to vits_config.yaml")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print report but do not write filtered files")
    parser.add_argument("--min-duration", type=float, default=None)
    parser.add_argument("--max-duration", type=float, default=None)
    parser.add_argument("--min-rms-db",   type=float, default=None)
    parser.add_argument("--max-silence",  type=float, default=None,
                        help="Max silence ratio, e.g. 0.60 for 60%%")
    parser.add_argument("--no-clip-check", action="store_true",
                        help="Disable clipping detection")
    args = parser.parse_args()

    # ── Load config ───────────────────────────────────────────────────
    if not os.path.exists(args.config):
        print(f"Config not found: {args.config}")
        sys.exit(1)

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    data_dir   = config.get("data_dir",  "./data/ljspeech_prepared")
    audio_dir  = config.get("audio_dir", "./data/LJSpeech-1.1/wavs")
    sample_rate = config.get("sample_rate", 22050)

    # ── Build thresholds ──────────────────────────────────────────────
    t = AuditThresholds()
    if args.min_duration  is not None: t.min_duration_sec     = args.min_duration
    if args.max_duration  is not None: t.max_duration_sec     = args.max_duration
    if args.min_rms_db    is not None: t.min_rms_db           = args.min_rms_db
    if args.max_silence   is not None: t.max_silence_ratio    = args.max_silence
    if args.no_clip_check:             t.clip_threshold       = 1.01  # effectively disabled

    print(f"\nAudit thresholds:")
    print(f"  Duration      : {t.min_duration_sec}s – {t.max_duration_sec}s")
    print(f"  RMS floor     : {t.min_rms_db} dB")
    print(f"  Max silence   : {t.max_silence_ratio:.0%}")
    print(f"  Clip threshold: {t.clip_threshold}")
    print(f"  Phoneme rate  : {t.min_phonemes_per_sec} – {t.max_phonemes_per_sec} /s")
    print()

    report_lines: List[str] = ["HexTTs Dataset Audit Report", "=" * 70]

    # ── Audit each split ──────────────────────────────────────────────
    results = {}
    for split in ("train", "val"):
        metadata_path = str(Path(data_dir) / f"{split}.txt")
        if not os.path.exists(metadata_path):
            print(f"  Skipping {split}.txt (not found at {metadata_path})")
            continue

        kept_lines, audits = audit_split(
            metadata_path=metadata_path,
            audio_dir=audio_dir,
            thresholds=t,
            sample_rate=sample_rate,
            report_lines=report_lines,
            split_name=split,
        )
        results[split] = (kept_lines, audits)

    # ── Print report to console ───────────────────────────────────────
    print("\n" + "\n".join(report_lines))

    if args.dry_run:
        print("\n[dry-run] No files written.")
        return

    # ── Write filtered metadata files ─────────────────────────────────
    for split, (kept_lines, _) in results.items():
        out_path = str(Path(data_dir) / f"{split}_filtered.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.writelines(kept_lines)
        print(f"\nWrote {len(kept_lines):,} samples → {out_path}")

    # ── Write full report to disk ─────────────────────────────────────
    report_path = str(Path(data_dir) / "audit_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"Full report saved → {report_path}")

    # ── Usage reminder ────────────────────────────────────────────────
    print("""
To use the filtered dataset, update vits_config.yaml:

  data_dir: "./data/ljspeech_prepared"   # stays the same

Then in train_vits.py (or vits_data.py), point to *_filtered.txt instead of
*train.txt / val.txt.  The easiest way is to rename them once you are
satisfied with the filter settings:

  copy data\\ljspeech_prepared\\train_filtered.txt  data\\ljspeech_prepared\\train.txt
  copy data\\ljspeech_prepared\\val_filtered.txt    data\\ljspeech_prepared\\val.txt

Or keep both and pass the path explicitly if your dataloader accepts it.
""")


if __name__ == "__main__":
    main()