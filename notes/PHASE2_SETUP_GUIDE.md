# Phase 2: Setup & Dataset Preparation
## Complete Windows Guide

---

## Prerequisites
- Windows 10/11
- NVIDIA GPU with CUDA support
- Python 3.9 or 3.10 (3.11+ may have compatibility issues)
- ~50 GB free disk space (LJSpeech is 24 GB)
- Command Prompt or PowerShell

---

## Step 1: Verify GPU Setup (5 minutes)

### Check NVIDIA Driver
Open Command Prompt and run:
```bash
nvidia-smi
```

You should see:
```
NVIDIA-SMI 555.00    Driver Version: 555.00
CUDA Version: 12.5
```

**If you don't see this:**
1. Download NVIDIA drivers: https://www.nvidia.com/Download/driverDetails.aspx
2. Install them and restart
3. Try `nvidia-smi` again

**If `nvidia-smi` is not found:**
- You don't have CUDA installed
- Download CUDA Toolkit: https://developer.nvidia.com/cuda-downloads
- Choose Windows, your GPU type, and latest version
- Install and restart

---

## Step 2: Create Project Structure (5 minutes)

Open Command Prompt and create your project:

```bash
# Navigate to a convenient location
cd C:\Users\YourUsername\Documents

# Create project folder
mkdir VITS_TTS
cd VITS_TTS

# Create subdirectories
mkdir data
mkdir logs
mkdir checkpoints
mkdir scripts
```

Your structure should look like:
```
VITS_TTS/
├── data/
├── logs/
├── checkpoints/
├── scripts/
└── (project files go here)
```

---

## Step 3: Python Virtual Environment (10 minutes)

A virtual environment isolates your dependencies and prevents conflicts.

### Create Virtual Environment
```bash
# In your VITS_TTS folder
python -m venv venv
```

This creates a `venv/` folder with isolated Python.

### Activate Virtual Environment

**Option A: Command Prompt**
```bash
venv\Scripts\activate
```

**Option B: PowerShell**
```bash
.\venv\Scripts\Activate.ps1
```

**If PowerShell says "execution policy":**
```bash
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then try again.

You should see `(venv)` at the start of your command line:
```
(venv) C:\Users\YourUsername\VITS_TTS>
```

---

## Step 4: Install PyTorch with CUDA (20-30 minutes)

**Important:** PyTorch must match your CUDA version!

### Check Your CUDA Version
From Step 1, note your CUDA version:
- CUDA 12.5 → Install cu121
- CUDA 12.1-12.4 → Install cu121
- CUDA 11.8 → Install cu118
- CUDA 11.6-11.7 → Install cu117

### Install PyTorch

**For CUDA 12.x (most common):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**For CUDA 11.8:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

This takes 5-10 minutes. It's downloading PyTorch binaries.

### Verify Installation
```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

Output should show:
```
2.1.0+cu121
True
```

⚠️ **If `torch.cuda.is_available()` returns `False`:**
- Your PyTorch doesn't have CUDA support
- Uninstall: `pip uninstall torch torchvision torchaudio`
- Check your CUDA version with `nvidia-smi`
- Reinstall with the correct cu version

---

## Step 5: Install Dependencies (10 minutes)

### Download requirements.txt
Download the `requirements.txt` file (provided separately) and save it to your VITS_TTS folder.

### Install All Dependencies
```bash
# Make sure (venv) is activated
pip install -r requirements.txt
```

This installs:
- librosa (audio processing)
- soundfile (read/write audio)
- g2p_en (text→phoneme conversion)
- numpy<2 (CRITICAL for compatibility)
- scipy, pyyaml, tqdm (utilities)

⚠️ **Important:** The `numpy<2` pin prevents NumPy 2.x which breaks librosa.

### Verify Installation
```bash
python -c "import librosa; import g2p_en; print('All imports OK!')"
```

---

## Step 6: Download LJSpeech Dataset (30-60 minutes)

LJSpeech is 13,100 audio clips of a single female speaker.

### Download via Browser (Recommended)

1. Go to: https://keithito.com/LJ-Speech-Dataset/
2. Click the download link (3.2 GB file)
3. Wait for download to complete
4. Extract the `.tar.bz2` file:
   - Windows 10/11: Right-click → Extract all
   - Or use 7-Zip: https://www.7-zip.org/
5. You'll get a folder named `LJSpeech-1.1`

### Move Dataset to Project

Move the extracted `LJSpeech-1.1` folder to:
```
VITS_TTS/data/LJSpeech-1.1
```

Your structure should now be:
```
VITS_TTS/
├── data/
│   └── LJSpeech-1.1/
│       ├── wavs/                    (13,100 .wav files)
│       ├── metadata.csv             (transcript + phoneme text)
│       └── README
├── logs/
├── checkpoints/
└── scripts/
```

### Verify Dataset
```bash
cd data\LJSpeech-1.1
dir wavs /s | find /c ".wav"
```

You should see: **13100**

---

## Step 7: Validate Dataset Quality (5 minutes)

Run the validation script to check dataset integrity:

### Download validate_dataset.py
Save the `validate_dataset.py` script to your VITS_TTS folder.

### Run Validation
```bash
python validate_dataset.py ./data/LJSpeech-1.1
```

Expected output:
```
✓ Found 13100 utterances in metadata.csv
✓ Valid audio files: 13100
✓ No issues found!

Audio Duration Statistics:
  Min: 1.00 seconds
  Max: 11.00 seconds
  Mean: 6.50 seconds
  Total: 24.30 hours
```

---

## Step 8: Prepare Data with Phonemes (20-30 minutes)

LJSpeech has text, but VITS needs phonemes (the sounds).

### Download prepare_data.py
Save the `prepare_data.py` script to your VITS_TTS folder.

### Run Data Preparation
```bash
python prepare_data.py ./data/LJSpeech-1.1 ./data/ljspeech_prepared
```

This script:
1. Reads all 13,100 transcripts
2. Converts text → phonemes using g2p_en
3. Splits into train (95%) and validation (5%)
4. Saves VITS-compatible metadata

Output will be:
```
data/ljspeech_prepared/
├── train.txt      (12,445 utterances)
├── val.txt        (655 utterances)
└── metadata.json  (detailed info)
```

**This takes 10-20 minutes because it's converting thousands of texts to phonemes.**

---

## Step 9: Verify Everything Works (5 minutes)

### Download test_setup.py
Save the `test_setup.py` script to your VITS_TTS folder.

### Run Full Test
```bash
python test_setup.py
```

Expected output:
```
SETUP SUMMARY
✓ PASS: PyTorch
✓ PASS: Audio Libraries
✓ PASS: Phoneme Conversion
✓ PASS: NumPy Version
✓ PASS: Data Loading
✓ PASS: Prepared Metadata

✓ ALL TESTS PASSED!
You're ready for Phase 3: Training Setup
```

---

## Troubleshooting

### "CUDA is not available"
```bash
# Check your CUDA version
nvidia-smi

# Reinstall PyTorch with correct version
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### "No module named 'librosa'"
```bash
# Reinstall dependencies
pip install -r requirements.txt
```

### "AttributeError: module 'numpy' has no attribute 'float_'"
```bash
# Fix NumPy 2.x incompatibility
pip install 'numpy<2'
```

### "LJSpeech-1.1 folder not found"
```bash
# Make sure you extracted the dataset to the correct location
# Should be: VITS_TTS/data/LJSpeech-1.1
dir data
```

### Dataset download is stuck
- Download manually from: https://keithito.com/LJ-Speech-Dataset/
- Use a download manager (IDM, Aria2, etc.)
- Try again at a different time

---

## What We've Done

You now have:

1. ✅ **Environment**: Python venv with PyTorch + CUDA
2. ✅ **Dataset**: 13,100 audio files from LJSpeech
3. ✅ **Prepared Data**: Text converted to phonemes, train/val splits
4. ✅ **Verified Setup**: All dependencies working, data loading OK

---

## Next Steps: Phase 3

Phase 3 will cover:
1. Downloading/understanding VITS model architecture
2. Creating VITS training configuration
3. Setting up the training loop
4. Training the model on your GPU
5. Monitoring progress with TensorBoard

**You're now ready to build and train the actual TTS model!**
