# CHANGELOG

All notable changes to HexTTs are documented here.
All notable GPU temperature increases are documented in your electricity bill.

---

## [v0.4.2] - 2026-04-06

### Added

- **Batch Audio Evaluation Mode** — Finally, evaluate multiple audio files without running the script 47 times
  - `evaluate_tts_output.py` now supports batch processing
  - Default to current folder: `python evaluate_tts_output.py` evaluates all `.wav` files in `.`
  - Evaluate specific folder: `python evaluate_tts_output.py --audio ./samples`
  - Single file still works: `python evaluate_tts_output.py --audio test.wav`
  - Batch summary table shows duration, RMS, and ZCR across all files for quick comparison
  - Error handling: If one file fails, it continues processing others

### Improved

- **Code Documentation** — Comments added to make the code actually readable
  - `log_audio_samples()` in `train_vits.py` now has detailed inline comments explaining each operation
  - Comments explain the "why" not just the "what" (so future you won't hate past you)
  - Clear documentation of tensor shapes, normalization logic, and TensorBoard integration
  - Updated docstrings with full method behavior and parameters

- **Evaluation Output Formatting** — Better visual presentation of audio metrics
  - Pretty-printed tables with alignment and separators
  - Emoji indicators (🎵 for reports, 📊 for verdicts) for visual clarity
  - Sample counts formatted with thousand separators (1,024 instead of 1024)
  - Better organized verdict messages with bullet points

### Internal

- **Audio Evaluation Pipeline** — Refactored for flexibility and reusability
- **Code Maintainability** — Clearer structure makes debugging way less painful

---

## [v0.4.1] - 2026-04-06

### Fixed

- **Sample Generation API Mismatch** — `generate_samples()` was confidently calling a method that didn't exist
  - The issue: HexTTS exposes `model.inference()` but we were calling `model.infer()`
  - The real issue: Someone copy-pasted from the wrong documentation at 2 AM
  - The fix: Changed in `utils/sample_generation.py` line 47
  - The result: Sample audio now generates instead of exploding with `AttributeError`
  - Tested at epochs 5, 10, 15 — all work, no demons summoned

- **Mel-Spectrogram TensorBoard Logging Disaster** — We were treating spectrograms like audio, which is scientiφically inaccurate
  - The problem: `log_audio_samples()` tried to log 80-channel mel-spectrograms with `.add_audio()`
  - What happened: TensorBoard either errored or played garbage audio (literally noise)
  - The elegant solution: Use `.add_image()` instead (mel-specs are heatmaps, not sound)
  - Updated docstring so future developers don't make the same mistake
  - Now visualized as beautiful heatmaps instead of ear-destroying audio

- **TensorBoard Learning Rate Tag Chaos** — LR was logged as `lr` instead of `train/lr`
  - The consequence: LR lived alone in the root folder of TensorBoard
  - The loneliness: Other metrics grouped under `train/*` didn't know it existed
  - The fix: Changed to `train/lr` for consistent organization
  - The reward: TensorBoard now looks neat and organized (unlike my codebase)

- **Config File Organization Disaster** — `max_duration_loss` lived in the wrong section like a lost puppy
  - Where it was: **INFERENCE SETTINGS** (completely wrong)
  - Where it should be: **TRAINING STABILITY** (its true home)
  - Why this matters: People thought it was an inference parameter (it's not)
  - Result: Much less confusion, fewer 3 AM debugging sessions

### Internal

- **Training Pipeline Separation of Concerns** — Sample generation and logging are now best friends instead of frenemies
- **Config Readability** — When you open vits_config.yaml, you'll actually understand it
- **Code Quality** — If the computer can read it, maybe humans can too?

---

## [v0.4.0] - 2026-04-04

### Added

- **TensorBoard Training Monitoring** — Because staring at raw numbers in the terminal is psychologically harmful
  - `train/loss` — The main agony metric (should go ↓, please)
  - `train/recon_loss` — How badly the model destroyed the waveform
  - `train/kl_loss` — Prevents latent space from becoming a boring line (VAE magic)
  - `train/duration_loss` — How good at timing the model is (spoiler: starts terrible)
  - `train/lr` — How aggressively the optimizer is currently attacking the loss
  - `val/loss` — Does it work on data it hasn't seen? (Horror metrics)

- **Duration Prediction Histogram** — Watch the model learn how long phonemes should last
  - Real-time insight into what your GPU is actually learning
  - Visualized in TensorBoard for masochistic monitoring
  - Exciting to watch for approximately 15 minutes

- **Automatic Speech Sample Generation** — Proof of life every 5 epochs
  - Located in `samples/epoch_XXX_sample_Y.wav`
  - Lets you hear if the model is learning (or melting)
  - Prevents the existential dread of wondering if anything is happening

- **Daily Training Reports** — `reports/report_04.04-2026.md` with metrics AND attitude
  - Tracks losses, checkpoints, and emotional state
  - Written for humans, not machines (radical concept)

- **Duration Explosion Protection** — Skips catastrophic batches before they destroy everything
  - Catches gradient explosions before they corrupt the model
  - Acts as a panic button for NaN/Inf meltdowns
  - Has already saved the training at least 3 times (probably)

### Improved

- **Training Stability** — Handles the edge cases where everything tries to become infinity
  - Better NaN detection ("are we broken yet?")
  - Batch skipping prevents one bad batch from ruining 8 hours of training
  - More robust error handling when mixed precision gets spicy

- **Training Visibility** — You can now understand what's happening without a PhD in mathematics
  - Duration loss shows phoneme alignment progress
  - Validation loss catches overfitting before it's too late
  - Component breakdowns make debugging less like divination

- **Checkpoint Reliability** — Multi-day training sessions no longer randomly explode
  - GradScaler state is saved and restored properly
  - Better checkpoint serialization (data integrity scores +1)
  - More informative logging so you know exactly what went wrong (or right)

### Fixed

- **Duration Loss Explosions** — Rare batches with gradient meltdowns now handled gracefully
  - Instead of corrupting the entire model, we just skip them
  - Innovation: Don't die, just... skip

- **TensorBoard Initialization** — Fixed mystical event file creation errors on first run
  - TensorBoard no longer requires an exorcism to initialize
  - First training runs now actually create readable logs

- **Device Handling in Audio Generation** — Audio now gets proper device strings (not torch.device objects)
  - The fix: Stop being lazy and call `.type` properly
  - The reward: No more device-related cryptic errors at inference

### Internal

- **Training Loop Refactoring** — Logging is now clean and readable
- **TensorBoard Integration** — Metrics are organized like a library, not a landfill
- **Diagnostics** — Console output tells you the ACTUAL story of what's happening

---

## Training Status: The Ongoing Saga

As of **Epoch 17 (April 6, 2026)**:

```
Epoch                 : 17/100 (we're 17% done but it feels like 300%)
Training Loss         : ~64.25 (Epoch 16 final)
Validation Loss       : ~166.98 (yes it's high, yes it's normal, no don't panic)
Reconstruction Loss   : ~0.25 (reasonable)
KL Divergence         : ~0.23 (latent space is exploring)
Duration Loss         : SPICY (batches exploding regularly)
```

### What's the Deal with This Validation Loss?

Validation loss jumped from 161.92 → 166.98 → ??? This is:

- **Not a bug**
- **Not catastrophic**
- **Completely normal for VITS**
- **Concerning for your sleep schedule**

Why? The model is learning difficult stuff:

1. How to vary phoneme durations (not constant)
2. How to align text to speech properly
3. How to generate variable-length sequences

These are HARD problems. The validation loss will stabilize around epoch 35-40. Patience, young grasshopper.

### The Duration Explosion Phenomenon

Epoch 17 is characterized by:

```
[Epoch 17: 3%]  loss=49.24, recon=0.25, kl=0.23
Skipping unstable batch due to duration explosion
Skipping unstable batch due to duration explosion
Skipping unstable batch due to duration explosion
```

**What's happening:** The duration predictor is learning that phonemes can have different lengths. It's overcorrecting and trying to assign ∞ milliseconds to some phonemes.

**Is this bad?** Meh. The training is designed to skip these. It's chaos, but _controlled_ chaos.

### Developer Status Report

```
GPU Fans        : (jet engine sounds)
Electricity Bill : (weeping gradually)
Model Quality   : (cautiously optimistic)
Panic Levels    : (slightly elevated)
Estimated Done  : ~May 1st (fingers crossed)
```

Expected first intelligible speech: **Epoch 30-40**

Expected full quality speech: **Epoch 70+**

Expected regaining sanity: **Post-training** (uncertain)

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

_Last updated: 06.04.2026_  
_Changelog maintained by: someone who cares, apparently_  
_GPU status: Occupied_  
_Linda Johnson memorial fund: ongoing_
