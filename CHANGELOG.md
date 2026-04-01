# CHANGELOG

All notable changes to HexTTs are documented here.
All notable GPU temperature increases are documented in your electricity bill.

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
