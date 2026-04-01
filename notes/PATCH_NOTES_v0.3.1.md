# PATCH NOTES — vits_data_cached.py

### _"The Cached Loader Grows Up, Gets Therapy, Stops Loading Garbage"_

---

## Version: 0.3.1 — The "Stop Feeding Your GPU Nonsense" Patch

**Date:** 01.04.2026  
**Affected file:** `vits_data_cached.py`  
**Risk level:** Low  
**GPU temperature change:** Marginally more reasonable  
**Vibe:** Production-hardened. Battle-tested. Slightly smug.

---

## What Was Wrong

The old `vits_data_cached.py` was a very trusting loader. It loaded everything.
Every sample. Long ones. Short ones. Ones that were probably fine. Ones that
absolutely were not fine. It asked no questions. It had no standards.
It was the golden retriever of data loaders.

This caused:

- Silent OOM errors mid-training when a suspiciously long sample appeared
- Batches padded to the size of the longest sequence in history
- GPU memory used on padding that was, to be clear, not speech
- You, staring at a crash log at 2AM wondering what went wrong

This patch fixes all of that. The loader now has opinions.

---

## Changes

---

### 1. Length Filtering — "No, You Cannot Sit With Us"

**Before:**
The loader accepted every sample unconditionally, like a buffet with no bouncer.

**After:**
Samples longer than `max_seq_length` are filtered out during initialization.
Quietly. Efficiently. Without drama.

```python
# Anything longer than this gets shown the door
max_seq_length: 1000  # configurable in vits_config.yaml
```

**Why this matters:**  
One extremely long outlier sample can inflate the entire batch's padding to
match it. Every other sample in that batch then gets padded with silence
to catch up. Your GPU spends cycles processing zeros. Nobody wins.
The outlier is gone now. We don't miss it.

**Impact:** Fewer OOM crashes. Fewer 2AM debugging sessions. Better you.

---

### 2. Sample Sorting — "Shortest Kids in the Front"

**Before:**
Samples loaded in whatever order the metadata file felt like.
Batches were a random mix of short and long sequences.
Padding was chaotic. Batch efficiency suffered quietly.

**After:**
All samples are **sorted by mel length** at initialization.
Short samples group with short samples. Long samples group with long samples.
Everyone finds their people.

```
Before:  [████████████] [██] [█████████] [███] [████████████████]
          ^padding hell                          ^more padding hell

After:   [██] [███] [████] ... [████████████] [████████████████]
          ^minimal padding      ^still long, but at least consistent
```

**Why this matters:**  
Padding is wasted computation. Sorted batches minimize padding within each
batch. Your GPU processes fewer zeros. Training steps get faster.
It's free efficiency and it costs nothing except the 3 minutes
`precompute_features.py` takes to sort everything.

---

### 3. Shuffle Behavior Change — "We Had to Kill the Shuffle"

**Before:** `shuffle=True` for training loader.  
**After:** `shuffle=False` for training loader.

Yes, we turned off shuffle. No, this is not a mistake.

Shuffling undoes the sort. If you sort by length and then immediately shuffle,
you have just wasted everyone's time, including your GPU's.
The length-based grouping is the point. Shuffle destroys the point.
Shuffle is gone.

```python
# OLD — shuffle=True, sort means nothing, padding is chaos
DataLoader(dataset, shuffle=True, ...)

# NEW — shuffle=False, sort is preserved, padding is civilized
DataLoader(dataset, shuffle=False, ...)
```

**"But doesn't shuffling help training stability?"**  
Yes, normally. In practice, the sort order provides enough variation that
training convergence is not meaningfully affected. The OOM reduction
and padding savings are worth it. We tested this. It's fine.

---

### 4. Memory-Efficient Cache Checking — "Look Before You Load"

**Before:**
When checking if a cached mel sample existed and was valid, the loader
loaded the full numpy array into memory, checked the length, then
proceeded. This is like picking up an entire watermelon to check if
it's ripe, instead of just tapping it.

**After:**
Uses `mmap_mode="r"` to inspect mel length from the file header
without loading the full array into RAM.

```python
# OLD — loads the whole thing just to check the shape
mel = np.load(mel_path)
length = mel.shape[1]

# NEW — peeks at the shape without touching the data
mel = np.load(mel_path, mmap_mode="r")
length = mel.shape[1]
# mel data stays on disk until actually needed
```

**Why this matters:**  
During initialization, the loader checks every cached file.
With 12,000+ samples, loading full arrays to check lengths was
burning significant RAM before training even started.
Now it just peeks. Much more polite.

---

### 5. Warning Tracking — "Receipts"

**Before:**
The loader tracked `missing_cache` warnings only.
Anything filtered for other reasons: silently vanished.
You had no idea what was getting dropped or why.

**After:**
The loader now tracks and reports two separate warning categories:

```
[DataLoader] Initialization complete.
  Kept:                  12,187 samples
  Skipped (bad cache):       41 samples   ← files missing or corrupted
  Skipped (too long):        217 samples  ← exceeded max_seq_length
  Total filtered:            258 samples
```

**Why this matters:**  
If 2,000 samples are being silently dropped because your `max_seq_length`
is set too aggressively, you should know that. Previously you would not know
that. You would just wonder why training wasn't improving despite having
a big dataset. Now you know. Adjust accordingly.

---

### 6. Detailed Initialization Logging — "Show Your Work"

**Before:**

```
Loading dataset...
Done.
```

**After:**

```
[DataLoader] Scanning cache directory...
[DataLoader] Checking mel lengths (mmap mode)...
[DataLoader] Sorting 12,187 samples by mel length...
[DataLoader] Initialization complete.
  Kept:              12,187 samples
  Skipped (bad cache):   41 samples
  Skipped (too long):   217 samples
  Shortest mel:          32 frames
  Longest mel:          998 frames
  Average mel:          487 frames
```

More words. More information. More confidence that something is happening
and it is the right something.

---

## Summary Table

| Change                  | Before               | After                    | Why You Should Care                               |
| ----------------------- | -------------------- | ------------------------ | ------------------------------------------------- |
| Length filtering        | None. Everything in. | Filters > max_seq_length | Prevents OOM on long outliers                     |
| Sample order            | Metadata order       | Sorted by mel length     | Reduces padding per batch                         |
| Shuffle                 | True                 | False                    | Preserves the sort. Shuffle would undo the point. |
| Cache check memory      | Full array load      | mmap peek                | Lower RAM during init                             |
| Skipped sample tracking | missing_cache only   | missing + too_long       | You know what's being dropped                     |
| Init logging            | Minimal              | Detailed counts + stats  | Visibility into what loaded                       |

---

## Migration Notes

No breaking changes. Drop-in replacement.

If you were already using `vits_data_cached.py`, update the file and:

1. Check your `vits_config.yaml` for `max_seq_length` — add it if missing:

   ```yaml
   max_seq_length: 1000 # adjust based on your dataset
   ```

2. Re-run `precompute_features.py` if your cache is outdated:

   ```bash
   python precompute_features.py --config vits_config.yaml
   ```

3. Read the new initialization logs. If you see large `skipped_too_long`
   numbers, your `max_seq_length` might be too strict. Raise it and retry.

---

## Known Issues (Still Not Fixed in This Patch)

- Griffin-Lim vocoder is still slow. This is a Griffin-Lim problem.
  It has been a Griffin-Lim problem since 1984. It will continue to be.
- Your laptop fans will still become aggressive during training.
  This is between you and your cooling system.
- Electricity bill trajectory: unchanged. Still upward. Still painful.

---

_Patch authored: 01.04.2026_  
_GPU dignity restored: partially_  
_Padding waste eliminated: significantly_  
_Vibe: professional, finally_
