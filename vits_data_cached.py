"""
Cached feature dataloader for faster VITS training.

This version loads:
- precomputed mel spectrograms from .npy
- precomputed phoneme IDs from .npy

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
}

def record_warning(category: str, key: str):
    if category in WARNING_STATS:
        WARNING_STATS[category][key] += 1

def get_warning_summary() -> dict:
    return {k: dict(v) for k, v in WARNING_STATS.items()}

def reset_warning_summary():
    for category in WARNING_STATS:
        WARNING_STATS[category].clear()


class TTSCachedDataset(Dataset):
    def __init__(self, metadata_file: str, data_dir: str):
        self.data_dir = Path(data_dir)
        self.cache_mels = self.data_dir / "cache" / "mels"
        self.cache_ids = self.data_dir / "cache" / "ids"

        self.metadata = []
        with open(metadata_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("|", 1)
                if len(parts) == 2:
                    filename, _ = parts
                    self.metadata.append(filename)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx: int):
        filename = self.metadata[idx]
        mel_path = self.cache_mels / f"{filename}.npy"
        ids_path = self.cache_ids / f"{filename}.npy"

        if not mel_path.exists() or not ids_path.exists():
            record_warning("missing_cache", filename)
            raise FileNotFoundError(f"Missing cached feature for {filename}")

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

    train_set = TTSCachedDataset(os.path.join(data_dir, "train.txt"), data_dir)
    val_set = TTSCachedDataset(os.path.join(data_dir, "val.txt"), data_dir)

    print(f"Training set size: {len(train_set)}")
    print(f"Validation set size: {len(val_set)}")

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
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
