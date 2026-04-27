"""
Data Loading Utilities for VITS TTS
Handles audio processing, mel-spectrogram computation, and batching
"""

import os
import numpy as np
import librosa
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Optional, Dict
import pickle
from pathlib import Path
from collections import defaultdict


# Phoneme to ID mapping (Arpabet)
PHONEME_TO_ID = {
    'AA': 0, 'AE': 1, 'AH': 2, 'AO': 3, 'AW': 4, 'AY': 5, 'B': 6, 'CH': 7,
    'D': 8, 'DH': 9, 'EH': 10, 'ER': 11, 'EY': 12, 'F': 13, 'G': 14, 'HH': 15,
    'IH': 16, 'IY': 17, 'JH': 18, 'K': 19, 'L': 20, 'M': 21, 'N': 22, 'NG': 23,
    'OW': 24, 'OY': 25, 'P': 26, 'R': 27, 'S': 28, 'SH': 29, 'T': 30, 'TH': 31,
    'UH': 32, 'UW': 33, 'V': 34, 'W': 35, 'Y': 36, 'Z': 37, 'ZH': 38,
    # Add PAD token
    'PAD': 39,
}

# WARNING: this dict is module-level and not multiprocessing-safe.
# Each DataLoader worker (num_workers > 0) runs in a separate process with its own
# copy — warnings emitted by workers are never aggregated in the main process.
# Counts here reflect only the main-process view (useful for num_workers=0 debugging).
WARNING_STATS = {
    "unknown_phoneme": defaultdict(int),
    "audio_load_error": defaultdict(int),
}

ID_TO_PHONEME = {v: k for k, v in PHONEME_TO_ID.items()}
VOCAB_SIZE = len(PHONEME_TO_ID)


class TTSDataset(Dataset):
    """
    Dataset for VITS TTS training
    Loads audio files and converts to mel-spectrograms
    """
    
    def __init__(self, metadata_file: str, audio_dir: str, config: dict):
        """
        Args:
            metadata_file: path to train.txt or val.txt
            audio_dir: path to audio directory
            config: configuration dict with audio parameters
        """
        self.audio_dir = audio_dir
        self.config = config
        self.sample_rate = config['sample_rate']
        self.n_mel_channels = config['n_mel_channels']
        self.mel_n_fft = config['mel_n_fft']
        self.mel_hop_length = config['mel_hop_length']
        self.mel_win_length = config['mel_win_length']
        self.mel_f_min = config['mel_f_min']
        self.mel_f_max = config['mel_f_max']
        self.ref_level_db = config['ref_level_db']
        self.min_level_db = config['min_level_db']
        
        # Read metadata
        self.metadata = []
        with open(metadata_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split('|')
                    if len(parts) == 2:
                        filename, phoneme_str = parts
                        self.metadata.append({
                            'filename': filename,
                            'phoneme_str': phoneme_str
                        })
    
    def __len__(self) -> int:
        return len(self.metadata)
    
    def __getitem__(self, idx: int) -> dict:
        """Load and process one sample"""
        item = self.metadata[idx]
        filename = item['filename']
        phoneme_str = item['phoneme_str']
        
        # Load audio
        wav_path = os.path.join(self.audio_dir, f"{filename}.wav")
        try:
            audio, sr = librosa.load(wav_path, sr=self.sample_rate)
        except Exception as exc:
            record_warning("audio_load_error", filename)
            raise RuntimeError(f"Failed to load audio: {wav_path}") from exc
        
        # Convert to mel-spectrogram
        mel_spec = self._audio_to_mel(audio)
        
        # Convert phonemes to IDs
        phoneme_ids = self._phonemes_to_ids(phoneme_str)
        
        return {
            'filename': filename,
            'audio': torch.FloatTensor(audio),
            'mel_spec': torch.FloatTensor(mel_spec),
            'phoneme_ids': torch.LongTensor(phoneme_ids),
            'phoneme_str': phoneme_str,
        }
    
    def _audio_to_mel(self, audio: np.ndarray) -> np.ndarray:
        """Convert raw audio to mel-spectrogram"""
        
        # Normalize audio
        audio = audio / (np.abs(audio).max() + 1e-7)
        
        # Compute mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.mel_n_fft,
            hop_length=self.mel_hop_length,
            win_length=self.mel_win_length,
            fmin=self.mel_f_min,
            fmax=self.mel_f_max,
            n_mels=self.n_mel_channels
        )
        
        # Convert to dB scale
        mel_spec = np.maximum(mel_spec, 1e-8)  # avoid log of zero
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        # mel_spec_db is in (-inf, 0] dB relative to the per-utterance maximum.

        # Map [min_level_db, 0] → [0, 1].
        # Dividing by -min_level_db (positive) gives 1.0 at 0 dB and 0.0 at min_level_db.
        # Previously used ref_level_db (20) here, which produced (mel_db - 20) / -20 ≥ 1.0
        # for all valid mel_db ≤ 0 — i.e. every training target was clipped to constant 1.0.
        mel_spec_norm = (mel_spec_db - self.min_level_db) / -self.min_level_db
        mel_spec_norm = np.clip(mel_spec_norm, 0, 1)
        
        return mel_spec_norm
    
    @staticmethod
    def _phonemes_to_ids(phoneme_str: str) -> List[int]:
        """Convert phoneme string to IDs safely"""
        phoneme_str = phoneme_str.strip()
        phonemes = phoneme_str.split()

        ids = []
        for p in phonemes:
            p = p.strip().upper().rstrip("012")  # normalize like inference
            if not p:
                continue

            if p in PHONEME_TO_ID:
                ids.append(PHONEME_TO_ID[p])
            else:
                record_warning("unknown_phoneme", p)
                # skip unknowns instead of padding every time
                ids.append(PHONEME_TO_ID['PAD'])
        
        if not ids:
            ids = [PHONEME_TO_ID['PAD']]

        return ids


def collate_fn_vits(batch: List[dict]) -> dict:
    """
    Custom collate function for VITS
    Handles variable-length sequences with padding
    """
    
    # Get batch data
    filenames = [item['filename'] for item in batch]
    phoneme_ids = [item['phoneme_ids'] for item in batch]
    mel_specs = [item['mel_spec'] for item in batch]
    
    # Get lengths
    phoneme_lengths = torch.LongTensor([len(p) for p in phoneme_ids])
    mel_lengths = torch.LongTensor([m.size(1) for m in mel_specs])
    
    # Pad phoneme sequences
    max_phoneme_len = phoneme_lengths.max().item()
    phoneme_padded = torch.zeros(len(batch), int(max_phoneme_len), dtype=torch.long)
    for i, p in enumerate(phoneme_ids):
        phoneme_padded[i, :len(p)] = p
    
    # Pad mel-spectrograms
    max_mel_len = mel_lengths.max().item()
    mel_padded = torch.zeros(len(batch), mel_specs[0].size(0), int(max_mel_len))
    for i, m in enumerate(mel_specs):
        mel_padded[i, :, :m.size(1)] = m
    
    return {
        'filenames': filenames,
        'phoneme_ids': phoneme_padded,
        'phoneme_lengths': phoneme_lengths,
        'mel_spec': mel_padded,
        'mel_lengths': mel_lengths,
    }


def create_dataloaders(config: dict, batch_size: int, num_workers: int = 0) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders
    
    Args:
        config: configuration dict
        batch_size: batch size
        num_workers: number of workers for data loading
    
    Returns:
        (train_loader, val_loader)
    """
    
    data_dir = config['data_dir']
    audio_dir = config['audio_dir']
    
    # Create datasets
    train_set = TTSDataset(
        os.path.join(data_dir, 'train.txt'),
        audio_dir,
        config
    )
    
    val_set = TTSDataset(
        os.path.join(data_dir, 'val.txt'),
        audio_dir,
        config
    )
    
    print(f"Training set size: {len(train_set)}")
    print(f"Validation set size: {len(val_set)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn_vits,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn_vits,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader

# Warning tracking utilities
def record_warning(category: str, key: str):
    """Increment warning count for a given category/key."""
    if category in WARNING_STATS:
        WARNING_STATS[category][key] += 1

# Utility to get a summary of warnings (for logging or analysis)
def get_warning_summary() -> dict:
    """Return a normal dict copy of current warning stats."""
    summary = {}
    for category, values in WARNING_STATS.items():
        summary[category] = dict(values)
    return summary

# Utility to reset warning stats (e.g., at the start of a new epoch)
def reset_warning_summary():
    """Reset all warning counters."""
    for category in WARNING_STATS:
        WARNING_STATS[category].clear()


if __name__ == "__main__":
    # Test the dataset
    config = {
        'data_dir': './data/ljspeech_prepared',
        'audio_dir': './data/LJSpeech-1.1/wavs',
        'sample_rate': 22050,
        'n_mel_channels': 80,
        'mel_n_fft': 1024,
        'mel_hop_length': 256,
        'mel_win_length': 1024,
        'mel_f_min': 0,
        'mel_f_max': 11025,
        'ref_level_db': 20,
        'min_level_db': -100,
    }
    
    print("Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(config, batch_size=4, num_workers=0)
    
    print(f"\nBatch from training set:")
    batch = next(iter(train_loader))
    print(f"  Phoneme IDs shape: {batch['phoneme_ids'].shape}")
    print(f"  Mel-spec shape: {batch['mel_spec'].shape}")
    print(f"  Filenames: {batch['filenames'][:2]}")
