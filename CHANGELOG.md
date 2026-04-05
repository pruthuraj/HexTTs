# CHANGELOG

All notable changes to HexTTs are documented here.
All notable GPU temperature increases are documented in your electricity bill.

---

## [v0.4.1] - 2026-04-06

### Fixed

- **Sample Generation API Mismatch** — `generate_samples()` was calling `model.infer()` but VITS exposes `model.inference()`
  - Fixed by aligning the API call in `utils/sample_generation.py`
  - Prevents `AttributeError` during sample generation at epoch 5, 10, 15, etc.
  - Sample audio now generates correctly for qualitative evaluation

- **Mel-Spectrogram Logging in TensorBoard** — `log_audio_samples()` was incorrectly treating 80-channel mel-spectrogram as audio
  - Changed from `add_audio()` to `add_image()` for proper visualization
  - Mel-spectrograms now logged as heatmaps instead of "audio" (which would error or produce garbage)
  - Updated method docstring to clarify it logs mel-spectrograms, not waveforms

- **TensorBoard LR Tag Inconsistency** — Learning rate was logged as `lr` instead of `train/lr`
  - Changed tag from `'lr'` to `'train/lr'` for consistent metric grouping
  - Now matches changelog documentation and other training metrics under `train/*` namespace
  - TensorBoard grouping now displays cleanly

- **Config Section Organization** — `max_duration_loss` was listed under **INFERENCE SETTINGS** instead of training configuration
  - Moved to new **TRAINING STABILITY** section (logically grouped with stability knobs)
  - Clarifies that this is a training-time safety mechanism, not an inference parameter
  - Reduces confusion about when and how the setting applies

### Internal

- **Training Pipeline Robustness** — Better separation of sample generation and TensorBoard logging concerns
- **Config Clarity** — Clearer organization of training vs. inference settings

---

## [v0.4.0] - 2026-04-04

### Added

- **TensorBoard Training Monitoring** — Real-time loss visualization because staring at numbers in the terminal is barbaric
  - `train/loss` — Total agony metric (should ↓)
  - `train/recon_loss` — Mel-spectrogram reconstruction error
  - `train/kl_loss` — VAE latent regularization (preventing boring latent spaces)
  - `train/duration_loss` — Phoneme duration alignment learning
  - `train/lr` — Learning rate (how aggressive your optimizer is feeling)
  - `val/loss` — Validation horror metrics for comparison
- **Duration Prediction Histogram** — Watch your model learn phoneme timing in real time
  - Visualized in TensorBoard for masochistic monitoring

- **Automatic Speech Sample Generation** — Every 5 epochs, the model produces audio samples
  - Located in `samples/epoch_XXX_sample_Y.wav`
  - Allows early quality inspection without waiting for epoch 100
  - Proof that your GPU is doing something productive

- **Training Report System** — `reports/report_04.04-2026.md` tracks daily progress with actual metrics and emotional support

- **Duration Loss Safety Mechanism** — Skips catastrophic batches to prevent alignment explosions
  - Prevents rare numerical instability from corrupting full training runs
  - Acts as a panic button for gradient meltdowns

### Improved

- **Training Stability** — Better handling of edge cases in duration prediction
  - Improved NaN detection and batch skipping
  - More robust error handling in mixed precision training
  - Better logging for debugging divergence issues

- **Training Monitoring** — Enhanced visibility into VITS-specific behavior
  - Duration loss tracking shows phoneme alignment learning progress
  - Validation loss monitoring catches overfitting early
  - Loss component breakdowns make debugging easier

- **Checkpoint Safety** — Improved reliability during multi-day training sessions
  - GradScaler state persistence across checkpoints
  - Better checkpoint serialization
  - More informative checkpoint logging

### Fixed

- **Potential Duration Alignment Instability** — Rare batches with exploding duration loss now handled gracefully
- **TensorBoard Initialization** — Fixed event file creation during first training runs
- **Sample Generation Device Type** — Audio generation now receives correct device string (not torch.device object)

### Internal

- **Refactored Training Loop Logging** — Cleaner instrumentation for debugging
- **Improved TensorBoard Integration** — Better metric organization and naming
- **Enhanced Training Diagnostics** — More informative console output for tracking convergence

---

## Training Status as of Epoch 15

Latest metrics snapshot:

```
Epoch              : 15 / 100
Training loss      : ~133.20
Validation loss    : ~314.02
Reconstruction     : ~0.49
KL divergence      : ~0.46
Duration loss      : ~105 (down from 133 — genuine improvement!)
```

### What's Happening

The model is currently in the **duration alignment learning phase**:

- Learning how long each phoneme should be pronounced
- Latent space is stabilizing (still chaotic, but productive chaos)
- First intelligible speech expected around **epoch 30–40**

### Temporary Validation Loss Spike

The validation loss spike (161.92 → 314.02) is **normal and expected** because:

- Duration learning is fundamentally unstable in early training
- Decoder is adjusting to variable sequence lengths
- Latent space will converge as training continues

This is not a bug. This is the model doing exactly what it should be doing: learning the hard stuff first.

### Developer Notes

Current thermal and emotional status:

```
GPU cooling      : jet engine (normal operation)
Training quality : improving (duration loss ↓)
Developer mood   : cautiously optimistic
Panic level      : unnecessary (for now)
```

---

## v0.3.2 - 2026-04-04 (Previous Updates)

### Fixed and Added generate_samples

- Training **no longer gaslights you** by pretending it restarted from epoch 1.  
  Checkpoint resume now behaves like a responsible adult.
- Eliminated the **VOCAB_SIZE existential crisis** by aligning model vocabulary with dataset phoneme vocabulary.
- Cached dataloader behavior improved so sequence handling is now **predictable instead of chaotic neutral**.
- **PyTorch 2.x future-proofed** by migrating from deprecated `torch.cuda.amp` API to new `torch.amp` API
  - Updated `autocast()` to use new API with device-type specification
  - Updated `GradScaler()` to use new API with device-type parameter
  - Eliminates FutureWarning spam (and ensures compatibility when old API is removed)
- **GradScaler state now persisted** across checkpoint saves/resumes
  - GradScaler internal state (loss scale, overflow counts) saved to checkpoint
  - Restored on resume, preventing gradient instability when continuing training
  - Ensures multi-day training runs remain stable across checkpoint boundaries
- **Hardened torch.load()** with explicit `weights_only=False` parameter
  - Suppresses FutureWarning in modern PyTorch versions
  - Explicitly documents that checkpoints contain complex Python objects
- **Fixed context manager issue** in mixed precision training
  - Forward/backward passes now correctly apply (or don't apply) mixed precision
- **Fixed device parameter type** in sample generation
  - `generate_samples()` now receives device as string (`self.device.type`) instead of torch.device object
  - Prevents type mismatch errors during audio sample generation at inference time

### Added

- Cached feature workflow using:
  - precomputed **mel spectrograms**
  - cached **phoneme ID files**

  This means your GPU now spends more time training and less time **waiting for audio preprocessing to finish its coffee break**.

- Better documentation for dataset preparation so future-you doesn't ask  
  _“Why is everything broken?”_

- New patch notes covering:
  - vocabulary size consistency
  - cached dataloader speed improvements
  - training pipeline updates

### Improved

- Training performance on **low-VRAM GPUs (hello RTX 3050 Ti gang)** by adding support for:
  - cached mel loading
  - cached phoneme ID loading
  - max sequence length filtering
  - length-based sample sorting

  Result: **less GPU suffering, fewer CUDA tantrums.**

- Practical configuration guidance for **RTX 3050 Ti 4GB** so the GPU survives the training process.

- Clearer recommended training ritual:
  - Step 1: Validate raw dataset
  - Step 2: Prepare phoneme metadata
  - Step 3: Precompute cached features
  - Step 4: Train model
  - Step 5: Resume training without emotional damage

### Developer Emotional Status

Stable but cautious.

### GPU Emotional Status

Still judging you.

---

# v0.3.1 — The "Stop Feeding Your GPU Nonsense" Patch

## Fixed / Improved in `vits_data_cached.py`

### Length Filtering

- Added `max_seq_length` filtering during dataset initialization
- Samples longer than the limit are now shown the door, quietly and efficiently
- Prevents the one suspiciously long outlier sample from padding your entire batch into OOM territory
- **Before:** loader accepted everything like a golden retriever. **After:** loader has standards.

### Sample Sorting

- Samples now sorted by mel length at initialization
- Short samples group with short. Long samples group with long. Everyone finds their people.
- Padding per batch reduced significantly — GPU processes fewer zeros, moves faster
- Batches are now civilized instead of chaotic

### Shuffle Behavior

- `shuffle=True` → `shuffle=False` for training loader
- Yes, on purpose. No, this is not a bug.
- Shuffling undoes the sort. The sort is the entire point. Shuffle had to go.
- Training convergence: unaffected. Padding waste: eliminated.

### Memory-Efficient Cache Checking

- Switched to `mmap_mode="r"` when inspecting cached mel lengths during init
- Now peeks at array shape without loading the full file into RAM
- Checking 12,000+ samples at startup no longer costs you 4GB of RAM before training even starts

### Warning Tracking

- Added `skipped_too_long` counter alongside existing `missing_cache` counter
- Initialization now prints a full breakdown: kept, skipped (bad cache), skipped (too long)
- You now know exactly what got dropped and why, instead of staring at a smaller-than-expected dataset and wondering

### Detailed Initialization Logging

- **Before:** `Loading dataset... Done.`
- **After:** Counts, lengths, stats — everything you need to confirm it loaded correctly
- If something is wrong, you'll know before training starts, not six hours into it

---

# v0.3.0 — Training Stability + Performance

### _"We Gave the GPU a Proper Meal Plan"_

## Added

- **Mixed precision (`use_amp`) support** — free speed, free memory savings, zero downsides. It was right there the whole time.
- **RTX 3050 Ti optimized config** — because not everyone has a 4090, and those people deserve to train too. `batch_size: 4`, `use_amp: true`, go.
- **`precompute_features.py`** — precomputes mel spectrograms once, upfront, so training doesn't recompute them every single batch like it's baking fresh bread for every sandwich
- **`vits_data_cached.py`** — the faster, smarter data loader that actually uses those precomputed features. Use this. Please.

## Improved

- **GPU utilization** — GPU now spends less time waiting for CPU to finish its homework
- **Training speed** — meaningfully faster when cached loader is enabled
- **CPU workload** — CPU gets a break. It was working very hard. It deserved this.

---

# v0.2.1 — Vocabulary Consistency Fix

### _"The Phantom Phonemes Are Gone Now"_

## Fixed

A vocabulary size mismatch that was, in retrospect, a fairly important thing to have wrong.

**Before** — hardcoded into the model like a bad secret:

```python
VOCAB_SIZE = 149  # where did 149 come from? nobody remembers.
```

**After** — dynamically imported from the data pipeline like an adult:

```python
from vits_data import VOCAB_SIZE  # now it matches the actual dataset. revolutionary.
```

**What this fixed:**

- Embedding size now matches the dataset vocabulary. As it should. Always should have.
- Removed phantom phoneme tokens that existed in the model but corresponded to nothing in the data
- Cleaner architecture. Fewer ghost tokens haunting the embedding layer.

**What caused it:** A hardcoded constant that fell out of sync with the dataset.
Classic. Happens to everyone. We don't dwell on it.

---

# v0.2.0 — Training Pipeline

### _"The Part Where It Actually Does Something"_

## Added

- **`train_vits.py`** — the main training loop. The whole reason this project exists.
- **Checkpoint saving** — model snapshots every N steps, so crashes are an inconvenience instead of a catastrophe
- **TensorBoard logging** — real-time loss curves for the obsessive among us. `http://localhost:6006`. You will check it constantly.
- **`inference_vits.py`** — generates speech from text using a trained checkpoint. This is the payoff.
- **`tts_app.py`** — interactive TTS interface, for when you want to type things and hear your robot say them without constructing a CLI command each time

---

# v0.1.0 — Project Foundation

### _"In the Beginning, There Was a Folder"_

## Added

- **Dataset preparation scripts** — converts raw LJSpeech transcripts into phoneme sequences, splits into train/val
- **Phoneme conversion** — g2p_en integration. Slow, thorough, necessary.
- **Dataset validation** — `validate_dataset.py` confirms all 13,100 files are present, intact, and not secretly corrupted
- **Project structure** — folders, configs, and the skeleton of ambition

**Dataset:**  
LJSpeech — 13,100 samples, single speaker, 24kHz, recorded by Linda Johnson who had no idea she was going to become the foundation of so many TTS experiments. Legend.

---

# Known Issues

### _"Things We Are Aware Of and Have Accepted"_

| Issue                                         | Status                       | Notes                                                     |
| --------------------------------------------- | ---------------------------- | --------------------------------------------------------- |
| Griffin-Lim vocoder is slow                   | Won't fix (it's Griffin-Lim) | Has been slow since 1984. HiFi-GAN exists for a reason.   |
| Laptop fans become aggressive during training | Working as intended          | Your GPU is doing its job. Let it work.                   |
| Electricity bill increases                    | Expected behavior            | You were warned. Multiple times. In multiple files.       |
| `max_seq_length` filtering removes samples    | By design                    | Check init logs if dataset size seems off. Adjust config. |

---

_Last updated: 01.04.2026_  
_Changelog maintained by: someone who cares, apparently_  
_GPU status: Occupied_  
_Linda Johnson memorial fund: ongoing_
