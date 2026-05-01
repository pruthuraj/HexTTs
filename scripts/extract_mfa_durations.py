"""
Extract per-phoneme duration targets from MFA TextGrid output.

Usage
-----
1. Install and run Montreal Forced Aligner on LJSpeech:

       mfa align data/LJSpeech-1.1/wavs  \\
                  english_us_arpa         \\
                  english_us_arpa         \\
                  data/mfa_alignments

2. Run this script to convert TextGrid files to frame-count arrays:

       python scripts/extract_mfa_durations.py \\
           --textgrid_dir data/mfa_alignments    \\
           --metadata_dir data/ljspeech_prepared  \\
           --output_dir   data/ljspeech_prepared/durations \\
           --hop_length   256                     \\
           --sample_rate  22050

3. Set `duration_dir` in configs/base.yaml:

       duration_dir: ./data/ljspeech_prepared/durations

After this, the trainer uses real MFA durations instead of pseudo-uniform targets.
Files with no matching TextGrid fall back to pseudo-targets automatically.
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Arpabet helpers (must match raw_dataset.py)
# ---------------------------------------------------------------------------

SILENCE_LABELS = {"", "sp", "sil", "SIL", "spn", "SPN", "<eps>"}

PHONEME_TO_ID = {
    'AA': 0, 'AE': 1, 'AH': 2, 'AO': 3, 'AW': 4, 'AY': 5, 'B': 6, 'CH': 7,
    'D': 8, 'DH': 9, 'EH': 10, 'ER': 11, 'EY': 12, 'F': 13, 'G': 14, 'HH': 15,
    'IH': 16, 'IY': 17, 'JH': 18, 'K': 19, 'L': 20, 'M': 21, 'N': 22, 'NG': 23,
    'OW': 24, 'OY': 25, 'P': 26, 'R': 27, 'S': 28, 'SH': 29, 'T': 30, 'TH': 31,
    'UH': 32, 'UW': 33, 'V': 34, 'W': 35, 'Y': 36, 'Z': 37, 'ZH': 38,
    'PAD': 39,
}


def _strip_stress(phoneme: str) -> str:
    return phoneme.strip().upper().rstrip("012")


# ---------------------------------------------------------------------------
# TextGrid parser (Praat "full" format — default MFA output)
# ---------------------------------------------------------------------------

def parse_textgrid_phones(path: Path) -> list[tuple[str, float, float]]:
    """
    Parse the 'phones' IntervalTier from a Praat full-format TextGrid.

    Returns a list of (phoneme_label, xmin, xmax) tuples in time order.
    Stress numbers are stripped from the label.
    """
    text = path.read_text(encoding="utf-8")

    # Find the phones tier block
    phones_match = re.search(
        r'name\s*=\s*"phones"\s*(.*?)(?=item\s*\[\d+\]|$)',
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if phones_match is None:
        raise ValueError(f"No 'phones' tier found in {path}")

    tier_text = phones_match.group(1)

    # Extract all intervals: xmin / xmax / text triples
    interval_pattern = re.compile(
        r'xmin\s*=\s*([\d.e+\-]+)\s*'
        r'xmax\s*=\s*([\d.e+\-]+)\s*'
        r'text\s*=\s*"([^"]*)"',
        re.DOTALL,
    )

    intervals = []
    for m in interval_pattern.finditer(tier_text):
        xmin = float(m.group(1))
        xmax = float(m.group(2))
        label = _strip_stress(m.group(3))
        intervals.append((label, xmin, xmax))

    return intervals


# ---------------------------------------------------------------------------
# Duration extraction
# ---------------------------------------------------------------------------

def intervals_to_frame_durations(
    intervals: list[tuple[str, float, float]],
    expected_phonemes: list[str],
    hop_length: int,
    sample_rate: int,
) -> np.ndarray | None:
    """
    Convert time-aligned phoneme intervals to mel-frame duration counts.

    Strategy for silence intervals:
      - Leading silence  → added to the first phoneme's duration
      - Trailing silence → added to the last phoneme's duration
      - Inter-word silence → split evenly between the preceding and following phoneme

    Returns None if the non-silence phoneme count doesn't match expected_phonemes.
    """
    def to_frame(t: float) -> int:
        return int(round(t * sample_rate / hop_length))

    # Separate speech and silence intervals, tracking position
    speech: list[tuple[str, float, float]] = []
    for label, xmin, xmax in intervals:
        if label in SILENCE_LABELS:
            continue
        if label not in PHONEME_TO_ID:
            continue  # unknown phoneme — skip
        speech.append((label, xmin, xmax))

    if len(speech) != len(expected_phonemes):
        return None

    # Base frame durations from MFA timing
    durations = np.zeros(len(speech), dtype=np.int64)
    for i, (label, xmin, xmax) in enumerate(speech):
        durations[i] = max(1, to_frame(xmax) - to_frame(xmin))

    # Distribute silence frames
    all_intervals = [(label, xmin, xmax) for label, xmin, xmax in intervals]
    speech_indices = [
        i for i, (label, _, _) in enumerate(all_intervals)
        if label not in SILENCE_LABELS and label in PHONEME_TO_ID
    ]

    sil_intervals = [
        (label, xmin, xmax) for label, xmin, xmax in all_intervals
        if label in SILENCE_LABELS
    ]

    for sil_label, sil_xmin, sil_xmax in sil_intervals:
        sil_frames = to_frame(sil_xmax) - to_frame(sil_xmin)
        if sil_frames <= 0:
            continue

        # Determine position: leading / trailing / inter-word
        sil_start = sil_xmin
        preceding = None
        following = None
        for i, (label, xmin, xmax) in enumerate(all_intervals):
            if label in SILENCE_LABELS or label not in PHONEME_TO_ID:
                continue
            if xmax <= sil_start:
                preceding = label
            elif xmin >= sil_xmax and following is None:
                following = label

        speech_count = len(speech)
        if preceding is None:
            # Leading silence → first phoneme
            durations[0] += sil_frames
        elif following is None:
            # Trailing silence → last phoneme
            durations[-1] += sil_frames
        else:
            # Inter-word → split evenly
            half = sil_frames // 2
            # Find indices of preceding / following in the speech list
            pre_idx = next(
                (i for i, (l, _, xmax) in enumerate(speech) if xmax <= sil_start),
                speech_count - 1,
            )
            fol_idx = next(
                (i for i, (l, xmin, _) in enumerate(speech) if xmin >= sil_xmax),
                0,
            )
            durations[pre_idx] += half
            durations[fol_idx] += sil_frames - half

    # Guarantee every phoneme has at least 1 frame
    durations = np.maximum(durations, 1)
    return durations.astype(np.int32)


# ---------------------------------------------------------------------------
# Metadata reader
# ---------------------------------------------------------------------------

def read_metadata(metadata_file: Path) -> dict[str, list[str]]:
    """Read train.txt or val.txt and return {filename: [stripped_phonemes]}."""
    result = {}
    with open(metadata_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("|")
            if len(parts) < 2:
                continue
            filename = parts[0]
            phoneme_str = parts[1]
            phonemes = [
                _strip_stress(p)
                for p in phoneme_str.split()
                if _strip_stress(p) and _strip_stress(p) in PHONEME_TO_ID
                   and _strip_stress(p) != "PAD"
            ]
            result[filename] = phonemes
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Extract MFA duration targets for HexTTs")
    parser.add_argument("--textgrid_dir", required=True,
                        help="Directory containing MFA TextGrid output files")
    parser.add_argument("--metadata_dir", required=True,
                        help="Directory containing train.txt and val.txt")
    parser.add_argument("--output_dir", required=True,
                        help="Output directory for .npy duration arrays")
    parser.add_argument("--hop_length", type=int, default=256,
                        help="Mel spectrogram hop length (default: 256)")
    parser.add_argument("--sample_rate", type=int, default=22050,
                        help="Audio sample rate (default: 22050)")
    args = parser.parse_args()

    textgrid_dir = Path(args.textgrid_dir)
    metadata_dir = Path(args.metadata_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read all metadata
    all_samples: dict[str, list[str]] = {}
    for split in ("train.txt", "val.txt"):
        meta_path = metadata_dir / split
        if meta_path.exists():
            all_samples.update(read_metadata(meta_path))

    print(f"Total samples in metadata: {len(all_samples)}")
    print(f"Searching for TextGrid files in: {textgrid_dir}")
    print(f"Writing duration arrays to: {output_dir}")
    print()

    saved = 0
    skipped_no_tg = 0
    skipped_mismatch = 0
    skipped_parse_error = 0

    for filename, expected_phonemes in all_samples.items():
        tg_path = textgrid_dir / f"{filename}.TextGrid"
        if not tg_path.exists():
            # MFA may organise by speaker subdirectory (LJSpeech has none, but handle it)
            candidates = list(textgrid_dir.rglob(f"{filename}.TextGrid"))
            if not candidates:
                skipped_no_tg += 1
                continue
            tg_path = candidates[0]

        try:
            intervals = parse_textgrid_phones(tg_path)
        except Exception as e:
            print(f"  [parse error] {filename}: {e}")
            skipped_parse_error += 1
            continue

        durations = intervals_to_frame_durations(
            intervals,
            expected_phonemes,
            args.hop_length,
            args.sample_rate,
        )

        if durations is None:
            # Count mismatch between TextGrid and metadata phoneme sequences
            tg_phones = [l for l, _, _ in intervals if l not in SILENCE_LABELS and l in PHONEME_TO_ID]
            print(
                f"  [mismatch] {filename}: "
                f"TextGrid has {len(tg_phones)} phones, metadata has {len(expected_phonemes)}"
            )
            skipped_mismatch += 1
            continue

        out_path = output_dir / f"{filename}.npy"
        np.save(str(out_path), durations)
        saved += 1

    total = len(all_samples)
    print(f"\nResults:")
    print(f"  Saved   : {saved}/{total} ({100*saved/max(1,total):.1f}%)")
    print(f"  No TextGrid : {skipped_no_tg}")
    print(f"  Seq mismatch: {skipped_mismatch}")
    print(f"  Parse error : {skipped_parse_error}")

    if saved == 0:
        print("\nNo durations saved. Check that --textgrid_dir points to MFA output.")
        return 1

    coverage = saved / max(1, total)
    if coverage < 0.8:
        print(f"\nWARNING: Only {coverage:.0%} coverage. "
              f"Samples without a duration file will use pseudo-targets during training.")
    else:
        print(f"\nDone. Set in configs/base.yaml: duration_dir: {output_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
