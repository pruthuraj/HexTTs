# Phase 2: Setup, Data Preparation, and Cache Pipeline

## VITS TTS Project

This phase prepares the full training pipeline for the current version of the project.

It covers:

- environment setup
- GPU verification
- dataset download and validation
- phoneme-clean metadata generation
- cached feature generation for faster training
- sanity checks before training

This guide matches the current project state, including:

- clean phoneme metadata
- vocabulary consistency
- cached mel / ID features
- RTX 3050 Ti friendly settings

---

## 1. System Requirements

Recommended setup:

| Component  | Requirement                           |
| ---------- | ------------------------------------- |
| OS         | Windows 10 / 11                       |
| Python     | 3.9 or 3.10                           |
| GPU        | NVIDIA RTX 3050 Ti (4GB VRAM)         |
| CUDA       | 11.8 or compatible PyTorch CUDA build |
| Disk Space | ~50 GB free                           |
| Dataset    | LJSpeech 1.1                          |

Notes:

- LJSpeech contains **13,100** audio clips.
- Cached training needs extra disk space because mel spectrograms and phoneme IDs are stored as `.npy` files.

---

## 2. Verify GPU Setup

Open Command Prompt or PowerShell and run:

```bash
nvidia-smi
```

You should see your NVIDIA GPU and driver information.

Then verify PyTorch CUDA later with:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

Expected output:

```text
True
```

---

## 3. Create Project Structure

Use this structure:

```text
VITS_TTS/
├── data/
│   ├── LJSpeech-1.1/
│   └── ljspeech_prepared/
│       └── cache/
│           ├── mels/
│           └── ids/
├── logs/
├── checkpoints/
├── scripts/
├── prepare_data.py
├── validate_dataset.py
├── precompute_features.py
├── vits_model.py
├── vits_data.py
├── vits_data_cached.py
├── train_vits.py
├── inference_vits.py
├── tts_app.py
└── vits_config.yaml
```

Create folders:

```bash
mkdir VITS_TTS
cd VITS_TTS
mkdir data
mkdir logs
mkdir checkpoints
mkdir scripts
```

---

## 4. Create Virtual Environment

Create:

```bash
python -m venv venv
```

Activate in Command Prompt:

```bash
venv\Scripts\activate
```

Activate in PowerShell:

```bash
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run:

```bash
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

## 5. Install PyTorch and Core Dependencies

For CUDA 11.8:

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Then install dependencies:

```bash
pip install "numpy<2"
pip install librosa soundfile g2p_en scipy pyyaml tqdm tensorboard matplotlib scikit-learn
```

Why `numpy<2`:

- it avoids compatibility issues with older audio tooling used in the project

Verify setup:

```bash
python -c "import torch, librosa, g2p_en; print(torch.cuda.is_available())"
```

---

## 6. Download and Place the Dataset

Download **LJSpeech 1.1** and extract it into:

```text
VITS_TTS/data/LJSpeech-1.1
```

Expected contents:

```text
data/LJSpeech-1.1/
├── wavs/
├── metadata.csv
└── README
```

---

## 7. Validate the Dataset

Run:

```bash
python validate_dataset.py ./data/LJSpeech-1.1
```

You want to confirm:

- audio files are present
- metadata is readable
- no obvious corruption exists

Typical goal:

```text
13100 utterances found
13100 valid audio files
```

---

## 8. Prepare Clean Phoneme Metadata

Run:

```bash
python prepare_data.py ./data/LJSpeech-1.1 ./data/ljspeech_prepared
```

This step is important.

The data preparation patch changed the metadata generation so the training files now contain **clean phoneme tokens**, not raw words. The preparation step now:

- uses `g2p_en`
- removes stress markers like `AH0 -> AH`
- uppercases tokens
- removes empty and invalid tokens
- keeps alphabetic phoneme tokens only
- skips empty phoneme results
- shuffles before train/validation split

Expected output files:

```text
data/ljspeech_prepared/
├── train.txt
├── val.txt
└── metadata.json
```

A correct line should look like:

```text
LJ001-0001|DH AH P R AA JH EH K T G UW T AH N B ER G
```

A wrong line would look like raw text:

```text
LJ001-0001|THE PROJECT GUTENBERG
```

If you still see raw words in `train.txt`, stop and fix preprocessing before training.

---

## 9. Precompute Cached Features

Run:

```bash
python precompute_features.py --config vits_config.yaml
```

This computes expensive features once and stores them on disk.

Cached layout:

```text
data/ljspeech_prepared/cache/
├── mels/
│   └── LJ001-0001.npy
└── ids/
    └── LJ001-0001.npy
```

Why this matters:

Without caching, the loader repeatedly computes:

- `librosa.load(...)`
- mel spectrograms
- dB conversion
- phoneme ID conversion

With caching:

- mel spectrograms are saved once
- phoneme IDs are saved once
- training mostly loads arrays from disk

This should improve epoch speed and GPU utilization.

---

## 10. Use the Cached Data Loader

In `train_vits.py`, make sure you import the cached loader:

```python
from vits_data_cached import create_dataloaders, get_warning_summary, reset_warning_summary
```

Instead of:

```python
from vits_data import create_dataloaders, get_warning_summary, reset_warning_summary
```

The cached loader includes improvements such as:

- filtering samples longer than `max_seq_length`
- sorting by mel length to reduce padding
- no shuffle so sorting is preserved
- memory-efficient cache inspection with `mmap`
- tracking skipped samples like missing cache and too-long samples
- better initialization logs

---

## 11. Confirm Vocabulary Consistency

The model should **not** use a hardcoded vocabulary size.

Correct approach in `vits_model.py`:

```python
from vits_data import VOCAB_SIZE
```

This keeps the model embedding size aligned with the phoneme mapping defined by the dataset loader.

Do not keep old code like:

```python
VOCAB_SIZE = 149
```

The vocabulary size must come from the dataset source of truth.

---

## 12. Recommended Config for RTX 3050 Ti

Use settings close to this in `vits_config.yaml`:

```yaml
batch_size: 4
num_workers: 0
use_amp: true
grad_clip_val: 0.5
max_seq_length: 300
pin_memory: true
```

Notes:

- `batch_size: 4` is safer for 4 GB VRAM
- `num_workers: 0` is slower than higher values, but it can reduce Windows multiprocessing issues
- `use_amp: true` usually helps performance on NVIDIA GPUs
- `max_seq_length` helps prevent long outliers from blowing up memory
- if cached loader skips too many samples, you may need to raise `max_seq_length`

---

## 13. Sanity Checks Before Training

Before starting Phase 3, confirm all of the following:

- `torch.cuda.is_available()` returns `True`
- `train.txt` and `val.txt` contain phonemes, not words
- cache files exist in `data/ljspeech_prepared/cache/`
- `train_vits.py` imports `vits_data_cached`
- `vits_model.py` imports `VOCAB_SIZE` instead of hardcoding it
- config uses RTX 3050 Ti friendly settings

---

## 14. What You Have After Phase 2

At this point you should have:

- working Python environment
- CUDA-enabled PyTorch
- validated LJSpeech dataset
- clean phoneme metadata
- cached mel and ID features
- vocabulary aligned between dataset and model
- training-ready project layout

Next phase: **train the VITS model, monitor it, and generate speech**.
