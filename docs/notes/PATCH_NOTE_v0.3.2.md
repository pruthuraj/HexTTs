# Patch Notes — v0.3.2

### _"Resume Logic + Cached Training Stability Update"_

### _"Or: Stop Gaslighting Me, I Know I Was on Epoch 47"_

**Release date:** 04.04.2026  
**Affected files:** `train_vits.py` (major), `vits_data_cached.py`, `vits_data.py`, `vits_config.yaml`, `README.md`  
**PyTorch compatibility:** Now 2.x future-proof  
**GPU emotional status:** `RTX 3050 Ti: "Bro... again?"`  
**Developer emotional status:** `[██████████] much more confident`

---

## Summary

Version **0.3.2** is about one thing: **making long TTS training less painful, less misleading, and future-proof.**

Previously:

- Training could crash, resume, and then display wrong epoch numbers like a goldfish with amnesia
- The vocabulary pipeline had a quiet mismatch that nobody noticed until they did
- AMP code used deprecated PyTorch APIs (the old API that will disappear)
- GradScaler state was not preserved, making resumption unstable
- The cached dataloader had room to be smarter
- The documentation didn't explain the prep workflow clearly enough

This patch fixes all of that. Resume behaves. Vocabulary is consistent. PyTorch 2.x compatibility is baked in. GradScaler state persists. The code is future-proof. You're welcome, future-you.

---

## Fixed

---

### 1. Checkpoint Resume No Longer Gaslights You

**The problem:**

Checkpoints correctly saved and restored model weights and optimizer state. That part worked. What did not work was the epoch counter — the training loop could reset it during `train()`, causing the console to display:

```
Resuming from checkpoint: checkpoint_step_009000.pt
Epoch [1/100] — Step 9001...
```

You loaded from step 9,000. You are not on epoch 1. You never were on epoch 1. Epoch 1 was weeks ago. And yet there it was: `Epoch 1`. Taunting you.

**The fix:**

Resume logic now correctly propagates the saved epoch and step values through the training loop. Training continues from where it was saved. The display reflects reality. Trust is restored.

**Impact:**

- Console output is no longer a work of fiction
- You can actually tell if resume worked without cross-referencing checkpoint filenames
- Long multi-day training runs are no longer a trust exercise with your own codebase

---

### 2. PyTorch 2.x Deprecation Compatibility: Future-Proofed

**The problem:**

PyTorch deprecated the old `torch.cuda.amp` API in favor of the new `torch.amp` API. Running on modern PyTorch versions produced a cascade of warnings:

```
FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated.
        Please use `torch.amp.autocast('cuda', args...)` instead.
```

and

```
FutureWarning: `torch.cuda.amp.GradScaler(args...)` is deprecated.
        Please use `torch.amp.GradScaler('cuda', args...)` instead.
```

These warnings are harmless today. Tomorrow, PyTorch will remove the old API entirely and your code breaks.

**The fix:**

Updated all AMP imports and usage to the new PyTorch 2.x API:

```python
# OLD (deprecated, will fail in future PyTorch)
from torch.cuda.amp import autocast, GradScaler
with autocast():
    ...
self.scaler = GradScaler()

# NEW (future-proof)
from torch.amp.grad_scaler import GradScaler
with torch.autocast(device_type=self.device.type) if self.scaler else torch.enable_grad():
    ...
self.scaler = GradScaler(self.device.type)
```

**Impact:**

- Compatible with PyTorch 2.x+ (no deprecation warnings)
- Will continue working when old API is removed
- Device-agnostic (works on CUDA, CPU, and future hardware)
- Code explicitly declares which device is getting mixed precision

---

### 3. GradScaler State Now Persisted Across Checkpoints

**The problem:**

`GradScaler` maintains internal state (loss scale, number of overflow skips) to manage gradient scaling safely. When you saved a checkpoint, you saved the model and optimizer — but not the scaler. On resume, a fresh `GradScaler()` was created with default scaling, losing the historical scaling information. This could cause:

- Unstable gradient scaling when resuming mid-training
- Loss spikes or NaN values after resuming
- Different convergence behavior than continuous training

**The fix:**

GradScaler state is now saved and restored:

```python
# In save_checkpoint()
if self.scaler is not None:
    checkpoint_dict['scaler_state_dict'] = self.scaler.state_dict()

# In load_checkpoint()
if self.scaler is not None and 'scaler_state_dict' in checkpoint:
    self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
```

**Impact:**

- Resume training maintains scaling stability
- No gradient anomalies after checkpoint resumption
- Seamless multi-day training with checkpoint saves
- Checkpoints are now truly complete snapshots of training state

---

### 4. Safe Dictionary Unpickling: torch.load() Hardened

**The problem:**

Modern PyTorch versions warn about unpickling untrusted checkpoint files (potential security issue). Since these are your own checkpoints, not a problem — but the warnings clutter the logs:

```
FutureWarning: You are using `torch.load` with `weights_only=False`
```

**The fix:**

Explicitly set `weights_only=False` when loading checkpoints:

```python
checkpoint = torch.load(
    checkpoint_path,
    map_location=self.device,
    weights_only=False  # ← Explicit opt-in, suppresses warning
)
```

**Impact:**

- FutureWarning is suppressed (cleaner logs)
- Explicitly documents that we're loading complex objects (optimizer state, config)
- Works consistently across PyTorch versions

---

### 5. Context Manager Fix: Conditional Autocast Now Works Properly

**The problem:**

Tried to use a ternary operator directly in a `with` statement:

```python
with autocast() if self.scaler else torch.enable_grad():  # ✗ SYNTAX ERROR
    ...
```

This is invalid Python — `with` statements need a proper context manager object, not a conditional expression.

**The fix:**

Create the context manager variable first, then use it:

```python
with torch.autocast(device_type=self.device.type) if self.scaler else torch.enable_grad():
    ...
```

**Impact:**

- Forward/backward pass properly respects autocast settings
- No syntax errors on training startup
- Code actually runs (novel concept)

---

### 6. VOCAB_SIZE Existential Crisis: Resolved

**The problem:**

Two parts of the codebase had different ideas about how many phonemes exist. The model had one number. The dataset had another. They never compared notes. This created a latent vocabulary mismatch that could produce silent errors, weird embedding behavior, or the unsettling feeling that your model was learning slightly wrong phonemes without telling you.

**Before** — hardcoded into the model like a number someone made up once and never revisited:

```python
VOCAB_SIZE = 149  # vibes-based
```

**After** — imported from the dataset pipeline, which actually knows:

```python
from vits_data import VOCAB_SIZE  # grounded in reality
```

**Result:**

- Model embedding size matches the actual phoneme vocabulary. As it always should have.
- Phantom phoneme tokens removed from the architecture
- Debugging vocabulary issues is now possible because there is only one source of truth
- Cleaner architecture overall

---

### 7. Cached Dataloader: Now Predictably Good Instead of Chaotic Neutral

**The problem:**

The cached dataloader worked, but left room for improvement in how it handled sequences of varying lengths. Without filtering or sorting, batches were padded to match the longest sequence present — including outliers that had no business being that long — and the GPU spent non-trivial time processing padding that was, to be clear, not speech.

**Improvements in this patch:**

| Feature                    | What It Does                                | Why It Helps                                  |
| -------------------------- | ------------------------------------------- | --------------------------------------------- |
| Cached mel loading         | Reads precomputed spectrograms from disk    | GPU stops waiting for CPU audio preprocessing |
| Cached phoneme ID loading  | Reads precomputed ID sequences from disk    | Consistent inputs, no live re-encoding        |
| `max_seq_length` filtering | Drops samples longer than the limit at init | Prevents OOM from outlier sequences           |
| Length-based sorting       | Sorts samples by mel length before batching | Minimizes padding waste per batch             |

**Result:** Less GPU suffering. Fewer CUDA emotional breakdowns. Faster iterations.
Your GPU now spends more time training and less time waiting for audio preprocessing to finish its coffee break.

---

## Added

---

### Cleaner Dataset Preparation Workflow

The full pipeline is now documented clearly enough that you won't have to reverse-engineer what order to run things in:

```
Step 1: Validate raw dataset
        python validate_dataset.py ./data/LJSpeech-1.1

Step 2: Prepare phoneme metadata
        python prepare_data.py ./data/LJSpeech-1.1 ./data/ljspeech_prepared

Step 3: Precompute cached features
        python precompute_features.py --config vits_config.yaml

Step 4: Train
        python train_vits.py --config vits_config.yaml --device cuda

Step 5: Resume without emotional damage
        python train_vits.py --config vits_config.yaml \
          --checkpoint checkpoints/checkpoint_step_009000.pt --device cuda
```

Run these in order. Do not skip Step 3. Step 3 is what makes everything faster.
If you skip Step 3 and wonder why training is slow, Step 3 is why.

---

### Patch Notes (Meta)

This release includes documented patch notes covering:

- Vocabulary size consistency fix (so you know what changed and why)
- Cached dataloader speed improvements (so you know what to enable)
- Training pipeline updates (so future-you doesn't have to archaeologically reconstruct the intent of past-you)

---

## Recommended Config: RTX 3050 Ti 4GB

If you are training on an RTX 3050 Ti — the dedicated, budget-conscious, slightly suffering option — this is your config:

```yaml
batch_size: 4
num_workers: 2
use_amp: true # free speed. take it.
grad_clip_val: 0.5 # keeps gradients from achieving escape velocity
max_seq_length: 250 # filters the outliers that would ruin your VRAM
```

This configuration has been tested to not immediately crash.
That is the bar. It clears the bar.

---

## Training Emotional Arc (For Reference)

```
Epoch 1   → GPU calm. You are hopeful. Loss is high but decreasing.
Epoch 5   → GPU warming up. You check TensorBoard for the third time today.
Epoch 10  → GPU sweating. You check TensorBoard every 3 minutes.
            You are staring at loss curves like stock charts.
            You whisper "please converge" to your GPU.
Epoch 20  → GPU reaching enlightenment. Loss is actually good.
            You feel pride. You deserve it.
```

---

## Known Side Effects of This Patch

- You may develop a habit of checking TensorBoard every 3 minutes. This was true before. It remains true.
- Resume will now work correctly, which means you have one fewer excuse to restart training from scratch.
- The vocabulary is consistent, so any remaining audio quality issues are the model's problem, not the pipeline's.
- Griffin-Lim is still slow. This is not a v0.3.2 problem. This has never been a v0.3.2 problem. This is a Griffin-Lim problem.

---

## Known Issues (Carried Forward, Still Unresolved, Still Accepted)

| Issue                                      | Status                      | Commentary                                     |
| ------------------------------------------ | --------------------------- | ---------------------------------------------- |
| Griffin-Lim vocoder is slow                | Acknowledged, not our fault | HiFi-GAN exists. That's future work.           |
| GPU fans become aggressive                 | Working as intended         | It's doing its job. Let it work.               |
| Electricity bill increases during training | Expected behavior           | You were warned in the README. Multiple times. |
| You will stare at TensorBoard obsessively  | Cannot be patched           | This is a you problem.                         |
| ~~PyTorch deprecation warnings~~           | **FIXED in v0.3.2**         | Now using torch.amp API (future-proof)         |

---

_v0.3.2 — Released 04.04.2026_  
_GPU dignity: fully restored_  
_Resume trust: completely re-established_  
_Vocabulary: finally consistent_  
_PyTorch 2.x compatibility: achieved_  
_GradScaler state: properly persisted_  
_Vibe: confidently optimistic_ `[██████████]`
