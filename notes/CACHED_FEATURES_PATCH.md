# Cached Feature Training Patch

## Goal
Speed up VITS training by precomputing expensive features once instead of recomputing them every batch.

## Files
- `precompute_features.py`
- `vits_data_cached.py`

## Why this is faster
original dataloader computes these for every sample on every epoch:
- `librosa.load(...)`
- mel spectrogram
- dB conversion
- phoneme ID conversion

That is CPU-heavy and slows GPU training.

With cached training:
- mel spectrograms are saved once as `.npy`
- phoneme IDs are saved once as `.npy`
- training only loads arrays from disk

## New cache layout
```text
data/ljspeech_prepared/cache/
  mels/
    LJ001-0001.npy
  ids/
    LJ001-0001.npy
```

## How to use

### 1. Precompute features
```bash
python precompute_features.py --config vits_config.yaml
```

### 2. Switch training to cached loader
In `train_vits.py`, replace:
```python
from vits_data import create_dataloaders, get_warning_summary, reset_warning_summary
```

with:
```python
from vits_data_cached import create_dataloaders, get_warning_summary, reset_warning_summary
```

### 3. Train normally
```bash
python train_vits.py --config vits_config.yaml --device cuda
```

## Expected result
Training should become much faster, with better GPU utilization and shorter epoch time.
