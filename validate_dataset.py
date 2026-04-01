"""
Production-style dataset validator for HexTTs / VITS pipeline.

Validates:
1. Raw LJSpeech dataset
2. Prepared train.txt / val.txt
3. Phoneme quality
4. Train/val overlap leakage
5. Cached mel and phoneme-id features
6. Shape consistency between cached files and metadata
"""

import os
import csv
from pathlib import Path
from collections import Counter

import librosa
import numpy as np
from tqdm import tqdm


# Keep this aligned with your training vocabulary
PHONEME_TO_ID = {
    'AA': 0, 'AE': 1, 'AH': 2, 'AO': 3, 'AW': 4, 'AY': 5, 'B': 6, 'CH': 7,
    'D': 8, 'DH': 9, 'EH': 10, 'ER': 11, 'EY': 12, 'F': 13, 'G': 14, 'HH': 15,
    'IH': 16, 'IY': 17, 'JH': 18, 'K': 19, 'L': 20, 'M': 21, 'N': 22, 'NG': 23,
    'OW': 24, 'OY': 25, 'P': 26, 'R': 27, 'S': 28, 'SH': 29, 'T': 30, 'TH': 31,
    'UH': 32, 'UW': 33, 'V': 34, 'W': 35, 'Y': 36, 'Z': 37, 'ZH': 38,
    'PAD': 39,
}


def print_header(title: str):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def validate_raw_dataset(dataset_path: str | Path, sample_limit: int = 200) -> dict:
    print_header("1. VALIDATING RAW LJSPEECH DATASET")

    dataset_path = Path(dataset_path)
    wavs_dir = dataset_path / "wavs"
    metadata_path = dataset_path / "metadata.csv"

    results = {
        "ok": True,
        "total_metadata_rows": 0,
        "checked_audio_files": 0,
        "missing_audio": [],
        "sample_rates": Counter(),
        "durations": [],
        "load_errors": [],
    }

    if not wavs_dir.exists():
        print(f"❌ Missing directory: {wavs_dir}")
        results["ok"] = False
        return results

    if not metadata_path.exists():
        print(f"❌ Missing file: {metadata_path}")
        results["ok"] = False
        return results

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = list(csv.reader(f, delimiter="|"))

    results["total_metadata_rows"] = len(metadata)
    print(f"✓ metadata.csv rows: {len(metadata)}")

    rows_to_check = metadata[:sample_limit] if sample_limit else metadata

    for row in tqdm(rows_to_check, desc="Checking raw audio"):
        if len(row) < 3:
            results["load_errors"].append(f"Malformed metadata row: {row}")
            continue

        filename = row[0]
        wav_path = wavs_dir / f"{filename}.wav"

        if not wav_path.exists():
            results["missing_audio"].append(filename)
            continue

        try:
            audio, sr = librosa.load(wav_path, sr=None)
            duration = len(audio) / sr
            results["sample_rates"][sr] += 1
            results["durations"].append(duration)
            results["checked_audio_files"] += 1
        except Exception as e:
            results["load_errors"].append(f"{filename}: {e}")

    print(f"✓ Checked audio files: {results['checked_audio_files']}")
    print(f"⚠ Missing audio files: {len(results['missing_audio'])}")
    print(f"⚠ Audio load errors: {len(results['load_errors'])}")

    if results["sample_rates"]:
        print("\nSample rates found:")
        for sr, count in sorted(results["sample_rates"].items()):
            print(f"  {sr} Hz: {count}")

    if results["durations"]:
        durations = np.array(results["durations"])
        print("\nDuration stats:")
        print(f"  Min:    {durations.min():.2f}s")
        print(f"  Max:    {durations.max():.2f}s")
        print(f"  Mean:   {durations.mean():.2f}s")
        print(f"  Median: {np.median(durations):.2f}s")

    if results["missing_audio"] or results["load_errors"]:
        results["ok"] = False

    return results


def read_prepared_file(path: Path):
    entries = []
    malformed = []

    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            parts = line.split("|", 1)
            if len(parts) != 2:
                malformed.append((i, line))
                continue

            filename, phoneme_str = parts
            entries.append((filename.strip(), phoneme_str.strip()))

    return entries, malformed


def validate_prepared_dataset(prepared_path: str | Path) -> dict:
    print_header("2. VALIDATING PREPARED PHONEME DATASET")

    prepared_path = Path(prepared_path)
    train_file = prepared_path / "train.txt"
    val_file = prepared_path / "val.txt"

    results = {
        "ok": True,
        "train_count": 0,
        "val_count": 0,
        "train_malformed": [],
        "val_malformed": [],
        "empty_phoneme_rows": [],
        "unknown_phonemes": Counter(),
        "train_val_overlap": set(),
        "sample_entry": None,
    }

    if not train_file.exists():
        print(f"❌ Missing file: {train_file}")
        results["ok"] = False
        return results

    if not val_file.exists():
        print(f"❌ Missing file: {val_file}")
        results["ok"] = False
        return results

    train_entries, train_malformed = read_prepared_file(train_file)
    val_entries, val_malformed = read_prepared_file(val_file)

    results["train_count"] = len(train_entries)
    results["val_count"] = len(val_entries)
    results["train_malformed"] = train_malformed
    results["val_malformed"] = val_malformed

    print(f"✓ Train samples: {len(train_entries)}")
    print(f"✓ Val samples:   {len(val_entries)}")
    print(f"⚠ Train malformed rows: {len(train_malformed)}")
    print(f"⚠ Val malformed rows:   {len(val_malformed)}")

    all_entries = [("train", *x) for x in train_entries] + [("val", *x) for x in val_entries]

    for split, filename, phoneme_str in all_entries:
        if not phoneme_str.strip():
            results["empty_phoneme_rows"].append((split, filename))
            continue

        for p in phoneme_str.split():
            p_norm = p.strip().upper().rstrip("012")
            if not p_norm:
                continue
            if p_norm not in PHONEME_TO_ID:
                results["unknown_phonemes"][p_norm] += 1

    train_names = {x[0] for x in train_entries}
    val_names = {x[0] for x in val_entries}
    results["train_val_overlap"] = train_names.intersection(val_names)

    if train_entries:
        results["sample_entry"] = train_entries[0]

    if results["sample_entry"]:
        filename, phoneme_str = results["sample_entry"]
        print(f"\nExample entry:\n  {filename}|{phoneme_str[:120]}")

    print(f"\n⚠ Empty phoneme rows: {len(results['empty_phoneme_rows'])}")
    print(f"⚠ Train/val overlap:  {len(results['train_val_overlap'])}")

    if results["unknown_phonemes"]:
        print("\nTop unknown phonemes:")
        for phoneme, count in results["unknown_phonemes"].most_common(10):
            print(f"  {phoneme}: {count}")
    else:
        print("\n✓ No unknown phonemes found")

    if (
        train_malformed
        or val_malformed
        or results["empty_phoneme_rows"]
        or results["train_val_overlap"]
        or results["unknown_phonemes"]
    ):
        results["ok"] = False

    return results


def validate_cached_features(prepared_path: str | Path) -> dict:
    print_header("3. VALIDATING CACHED FEATURES")

    prepared_path = Path(prepared_path)
    cache_dir = prepared_path / "cache"
    mel_dir = cache_dir / "mels"
    ids_dir = cache_dir / "ids"
    train_file = prepared_path / "train.txt"
    val_file = prepared_path / "val.txt"

    results = {
        "ok": True,
        "mel_count": 0,
        "ids_count": 0,
        "missing_mels": [],
        "missing_ids": [],
        "bad_mel_shapes": [],
        "bad_id_shapes": [],
        "nan_mels": [],
        "sample_mel_shape": None,
        "sample_id_shape": None,
    }

    if not mel_dir.exists():
        print(f"❌ Missing mel cache directory: {mel_dir}")
        results["ok"] = False
        return results

    if not ids_dir.exists():
        print(f"❌ Missing id cache directory: {ids_dir}")
        results["ok"] = False
        return results

    mel_files = list(mel_dir.glob("*.npy"))
    ids_files = list(ids_dir.glob("*.npy"))

    results["mel_count"] = len(mel_files)
    results["ids_count"] = len(ids_files)

    print(f"✓ Cached mel files: {len(mel_files)}")
    print(f"✓ Cached id files:  {len(ids_files)}")

    expected_names = set()
    for file_path in [train_file, val_file]:
        if file_path.exists():
            entries, _ = read_prepared_file(file_path)
            expected_names.update(filename for filename, _ in entries)

    for filename in tqdm(sorted(expected_names), desc="Checking cache consistency"):
        mel_path = mel_dir / f"{filename}.npy"
        ids_path = ids_dir / f"{filename}.npy"

        if not mel_path.exists():
            results["missing_mels"].append(filename)
            continue

        if not ids_path.exists():
            results["missing_ids"].append(filename)
            continue

        try:
            mel = np.load(mel_path)
            ids = np.load(ids_path)

            if results["sample_mel_shape"] is None:
                results["sample_mel_shape"] = mel.shape
            if results["sample_id_shape"] is None:
                results["sample_id_shape"] = ids.shape

            if mel.ndim != 2 or mel.shape[0] != 80:
                results["bad_mel_shapes"].append((filename, mel.shape))

            if ids.ndim != 1 or len(ids) == 0:
                results["bad_id_shapes"].append((filename, ids.shape))

            if np.isnan(mel).any() or np.isinf(mel).any():
                results["nan_mels"].append(filename)

        except Exception as e:
            results["bad_mel_shapes"].append((filename, str(e)))

    print(f"\n⚠ Missing mel files: {len(results['missing_mels'])}")
    print(f"⚠ Missing id files:  {len(results['missing_ids'])}")
    print(f"⚠ Bad mel shapes:    {len(results['bad_mel_shapes'])}")
    print(f"⚠ Bad id shapes:     {len(results['bad_id_shapes'])}")
    print(f"⚠ NaN/Inf mels:      {len(results['nan_mels'])}")

    if results["sample_mel_shape"] is not None:
        print(f"\nExample mel shape: {results['sample_mel_shape']}")
    if results["sample_id_shape"] is not None:
        print(f"Example id shape:  {results['sample_id_shape']}")

    if (
        results["missing_mels"]
        or results["missing_ids"]
        or results["bad_mel_shapes"]
        or results["bad_id_shapes"]
        or results["nan_mels"]
    ):
        results["ok"] = False

    return results


def print_summary(raw_results: dict, prepared_results: dict, cache_results: dict):
    print_header("FINAL SUMMARY")

    checks = {
        "Raw dataset": raw_results["ok"],
        "Prepared dataset": prepared_results["ok"],
        "Cached features": cache_results["ok"],
    }

    all_ok = True
    for name, ok in checks.items():
        mark = "✓ PASS" if ok else "❌ FAIL"
        print(f"{mark}: {name}")
        if not ok:
            all_ok = False

    print("\nOverall status:")
    if all_ok:
        print("✓ Dataset pipeline looks healthy and ready for training.")
    else:
        print("⚠ Dataset pipeline has problems. Fix them before training.")

    print("\nMost common next actions:")
    print("- Re-run prepare_data.py if phoneme rows are bad")
    print("- Re-run precompute_features.py if cache files are missing")
    print("- Check train/val split if overlap is detected")
    print("- Remove or fix malformed metadata rows")


def main():
    raw_path = "./data/LJSpeech-1.1"
    prepared_path = "./data/ljspeech_prepared"

    raw_results = validate_raw_dataset(raw_path, sample_limit=200)
    prepared_results = validate_prepared_dataset(prepared_path)
    cache_results = validate_cached_features(prepared_path)

    print_summary(raw_results, prepared_results, cache_results)


if __name__ == "__main__":
    main()