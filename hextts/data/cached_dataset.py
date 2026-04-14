"""
Cached feature dataloader for faster VITS training.

Patched features:
- loads precomputed mel spectrograms and phoneme IDs
- filters long samples using max_seq_length
- sorts kept samples by mel length to reduce padding waste
- keeps warning summary support

Expected cache layout:
<data_dir>/cache/mels/<filename>.npy
<data_dir>/cache/ids/<filename>.npy
"""

import os
from pathlib import Path
from collections import defaultdict
from typing import Tuple, List

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

PHONEME_TO_ID = {
    'AA': 0, 'AE': 1, 'AH': 2, 'AO': 3, 'AW': 4, 'AY': 5, 'B': 6, 'CH': 7,
    'D': 8, 'DH': 9, 'EH': 10, 'ER': 11, 'EY': 12, 'F': 13, 'G': 14, 'HH': 15,
    'IH': 16, 'IY': 17, 'JH': 18, 'K': 19, 'L': 20, 'M': 21, 'N': 22, 'NG': 23,
    'OW': 24, 'OY': 25, 'P': 26, 'R': 27, 'S': 28, 'SH': 29, 'T': 30, 'TH': 31,
    'UH': 32, 'UW': 33, 'V': 34, 'W': 35, 'Y': 36, 'Z': 37, 'ZH': 38,
    'PAD': 39,
}
ID_TO_PHONEME = {v: k for k, v in PHONEME_TO_ID.items()}
VOCAB_SIZE = len(PHONEME_TO_ID)

WARNING_STATS = {
    "missing_cache": defaultdict(int),
    "skipped_too_long": defaultdict(int),
}

# Functions to record and summarize warnings during dataset loading
def record_warning(category: str, key: str):
    if category in WARNING_STATS:
        WARNING_STATS[category][key] += 1

# Function to get a summary of warnings encountered during dataset loading
def get_warning_summary() -> dict:
    return {k: dict(v) for k, v in WARNING_STATS.items()}

# Function to reset warning statistics (useful between epochs or runs)
def reset_warning_summary():
    for category in WARNING_STATS:
        WARNING_STATS[category].clear()

# Dataset class that loads precomputed mel spectrograms and phoneme IDs, with filtering and sorting
class TTSCachedDataset(Dataset):
    def __init__(self, metadata_file: str, data_dir: str, max_seq_length=None):
        self.data_dir = Path(data_dir)
        self.cache_mels = self.data_dir / "cache" / "mels"
        self.cache_ids = self.data_dir / "cache" / "ids"
        self.max_seq_length = max_seq_length

        self.samples = []
        skipped_missing = 0
        skipped_too_long = 0

        with open(metadata_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split("|", 1)
                if len(parts) != 2:
                    continue

                filename, _ = parts

                mel_path = self.cache_mels / f"{filename}.npy"
                ids_path = self.cache_ids / f"{filename}.npy"

                if not mel_path.exists() or not ids_path.exists():
                    skipped_missing += 1
                    record_warning("missing_cache", filename)
                    continue

                try:
                    # Load mel spectrogram in memory-mapped mode to get its length without loading the entire array
                    mel = np.load(mel_path, mmap_mode="r")
                    mel_len = int(mel.shape[1])

                    if self.max_seq_length is not None and mel_len > self.max_seq_length:
                        skipped_too_long += 1
                        record_warning("skipped_too_long", filename)
                        continue

                    self.samples.append({
                        "filename": filename,
                        "mel_len": mel_len,
                    })
                except Exception:
                    skipped_missing += 1
                    record_warning("missing_cache", filename)

        self.samples.sort(key=lambda x: x["mel_len"])

        print(f"Loaded cached dataset from {metadata_file}")
        print(f"  Kept samples: {len(self.samples)}")
        print(f"  Skipped missing/bad cache: {skipped_missing}")
        print(f"  Skipped too long: {skipped_too_long}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        filename = sample["filename"]

        mel_path = self.cache_mels / f"{filename}.npy"
        ids_path = self.cache_ids / f"{filename}.npy"

        mel_spec = np.load(mel_path)
        phoneme_ids = np.load(ids_path)

        return {
            "filename": filename,
            "mel_spec": torch.FloatTensor(mel_spec),
            "phoneme_ids": torch.LongTensor(phoneme_ids),
        }


def collate_fn_vits(batch: List[dict]) -> dict:
    filenames = [item["filename"] for item in batch]
    phoneme_ids = [item["phoneme_ids"] for item in batch]
    mel_specs = [item["mel_spec"] for item in batch]

    phoneme_lengths = torch.LongTensor([len(p) for p in phoneme_ids])
    mel_lengths = torch.LongTensor([m.size(1) for m in mel_specs])

    max_phoneme_len = int(phoneme_lengths.max().item())
    phoneme_padded = torch.zeros(len(batch), max_phoneme_len, dtype=torch.long)
    for i, p in enumerate(phoneme_ids):
        phoneme_padded[i, :len(p)] = p

    max_mel_len = int(mel_lengths.max().item())
    mel_padded = torch.zeros(len(batch), mel_specs[0].size(0), max_mel_len)
    for i, m in enumerate(mel_specs):
        mel_padded[i, :, :m.size(1)] = m

    return {
        "filenames": filenames,
        "phoneme_ids": phoneme_padded,
        "phoneme_lengths": phoneme_lengths,
        "mel_spec": mel_padded,
        "mel_lengths": mel_lengths,
    }


def create_dataloaders(config: dict, batch_size: int, num_workers: int = 0) -> Tuple[DataLoader, DataLoader]:
    data_dir = config["data_dir"]
    max_seq_length = config.get("max_seq_length", None)

    train_set = TTSCachedDataset(
        os.path.join(data_dir, "train.txt"),
        data_dir,
        max_seq_length=max_seq_length,
    )
    val_set = TTSCachedDataset(
        os.path.join(data_dir, "val.txt"),
        data_dir,
        max_seq_length=max_seq_length,
    )

    print(f"Training set size: {len(train_set)}")
    print(f"Validation set size: {len(val_set)}")

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn_vits,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn_vits,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader
