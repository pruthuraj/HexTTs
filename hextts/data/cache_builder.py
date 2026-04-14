"""Cached feature precompute pipeline for HexTTs."""

from __future__ import annotations

import argparse
from pathlib import Path

import librosa
import numpy as np

from hextts.config import load_config
from .raw_dataset import PHONEME_TO_ID


def phonemes_to_ids(phoneme_str: str) -> np.ndarray:
    phonemes = phoneme_str.strip().split()
    ids = []

    for p in phonemes:
        p = p.strip().upper().rstrip("012")
        if not p:
            continue
        if p in PHONEME_TO_ID:
            ids.append(PHONEME_TO_ID[p])
        else:
            ids.append(PHONEME_TO_ID["PAD"])

    if not ids:
        ids = [PHONEME_TO_ID["PAD"]]

    return np.asarray(ids, dtype=np.int64)


def audio_to_mel(audio: np.ndarray, sample_rate: int, config: dict) -> np.ndarray:
    audio = audio / (np.abs(audio).max() + 1e-7)

    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_fft=config["mel_n_fft"],
        hop_length=config["mel_hop_length"],
        win_length=config["mel_win_length"],
        fmin=config["mel_f_min"],
        fmax=config["mel_f_max"],
        n_mels=config["n_mel_channels"],
    )

    mel_spec = np.maximum(mel_spec, 1e-8)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    mel_spec_norm = (mel_spec_db - config["ref_level_db"]) / -config["ref_level_db"]
    mel_spec_norm = np.clip(mel_spec_norm, 0, 1)

    return mel_spec_norm.astype(np.float32)


def process_split(split_name: str, config: dict):
    from tqdm import tqdm

    data_dir = Path(config["data_dir"])
    audio_dir = Path(config["audio_dir"])
    metadata_file = data_dir / f"{split_name}.txt"

    cache_dir = data_dir / "cache"
    mel_dir = cache_dir / "mels"
    ids_dir = cache_dir / "ids"
    mel_dir.mkdir(parents=True, exist_ok=True)
    ids_dir.mkdir(parents=True, exist_ok=True)

    if not metadata_file.exists():
        raise FileNotFoundError(f"Missing metadata file: {metadata_file}")

    lines = metadata_file.read_text(encoding="utf-8").splitlines()
    total = 0
    skipped = 0

    for line in tqdm(lines, desc=f"Precomputing {split_name}"):
        line = line.strip()
        if not line:
            continue

        parts = line.split("|", 1)
        if len(parts) != 2:
            skipped += 1
            continue

        filename, phoneme_str = parts
        wav_path = audio_dir / f"{filename}.wav"

        if not wav_path.exists():
            skipped += 1
            continue

        mel_path = mel_dir / f"{filename}.npy"
        ids_path = ids_dir / f"{filename}.npy"

        if mel_path.exists() and ids_path.exists():
            total += 1
            continue

        audio, _ = librosa.load(wav_path, sr=config["sample_rate"])
        mel = audio_to_mel(audio, config["sample_rate"], config)
        ids = phonemes_to_ids(phoneme_str)

        np.save(mel_path, mel)
        np.save(ids_path, ids)

        total += 1

    print(f"{split_name}: cached {total} samples, skipped {skipped}")


def cli_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args(argv)

    config = load_config(args.config)

    process_split("train", config)
    process_split("val", config)

    print("\\nDone.")
    print(f"Cache created under: {Path(config['data_dir']) / 'cache'}")
    return 0
