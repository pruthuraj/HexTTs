# HexTTs Patch Note v0.5.3

## "The Model Was Doing Everything Wrong Architecturally And We Fixed It"

**Date:** 2026-04-28
**Tag:** v0.5.3
**Status:** Complete
**Urgency:** High — several architectural decisions were silently limiting model quality from the ground up

---

## Overview

v0.5.3 is an **architecture and training pipeline improvement** patch.

v0.5.2 made the training pipeline honest — the reconstruction loss actually worked for the first time.
v0.5.3 makes the model worthy of that honest supervision.

Seven model-level changes. Two pipeline bug fixes. Three training speed improvements. One config tuning pass.
Everything requires a fresh training run. The v0.5.2 checkpoints were only ~5% through training (step ~16k of ~306k).
The loss from starting over is approximately 64 minutes. The gain is a model that is no longer fighting itself.

---

## Bug Fixes

---

### Bug 1 — Griffin-Lim Was Called on a Mel Spectrogram

**Severity:** High — all sample audio generation was broken

**Where it lived:**
- `hextts/vocoder/griffin_lim.py` — `mel_to_audio()`
- `utils/sample_generation.py` — caller

**What was wrong:**

`librosa.griffinlim()` expects a **linear STFT magnitude spectrogram** of shape `(n_fft//2 + 1, T)` — 513 bins for n_fft=1024.
The function was being called with an 80-bin mel spectrogram.

librosa tried to reconstruct: it internally computed a linear spectrogram of shape `(513, T)` then attempted to broadcast it back into `(80, T)`. It could not. The error was caught and swallowed by the `warnings.warn` fallback in the sample generation loop, so training continued silently with zero valid audio samples at every epoch.

Every qualitative training diagnostic since the audio generation was added has been garbage.

**Fix:**

`griffin_lim.py` now follows the correct pipeline used by `hextts/inference/pipeline.py`:
1. `mel_db → power`: `np.power(10.0, mel / 10.0)`
2. `mel_power → linear STFT`: `librosa.feature.inverse.mel_to_stft()`
3. `linear → waveform`: `librosa.griffinlim()`

`fmin` and `fmax` parameters added so the mel filterbank inversion matches the training filterbank.

---

### Bug 2 — Sample Generation Denormalized with the Wrong Formula

**Severity:** High — all sample waveforms were scaled incorrectly even before Griffin-Lim

**Where it lived:**
- `utils/sample_generation.py` — denormalization step

**What was wrong:**

```python
# What was there (broken):
mel_spec_np = mel_spec_np * -ref_level_db + ref_level_db
# = mel_spec_np * (-20) + 20
```

The v0.5.2 normalization formula maps `[min_level_db, 0] → [0, 1]` where `min_level_db = -100`.
The correct inverse is `norm * 100 - 100`, not `norm * (-20) + 20`.

The sample generation was using the pre-v0.5.2 broken formula even after v0.5.2 fixed it everywhere else.
The mel values fed into Griffin-Lim were wrong. The audio was wrong. Nothing crashed.

This is the same class of bug that broke all of v0.5.1 training — silent formula mismatch, plausible-looking output.

**Fix:**

```python
# What it is now (correct):
mel_spec_np = mel_spec_np * (-min_level_db) + min_level_db
# = mel_spec_np * 100 - 100
```

`min_level_db` is now read from config, consistent with `pipeline.py`.

---

## Architecture Changes

All changes are in `hextts/models/vits_impl.py`. All require fresh training.

---

### Change 1 — Decoder Output: `tanh` → `sigmoid`

**Impact:** High

The v0.5.2 normalization maps mel dB values to `[0, 1]`. Every training target is in `[0, 1]`.

The Decoder was outputting `tanh(x)` which has range `[-1, 1]`. Half the output range — everything below zero — corresponds to values that never appear in the training targets. The reconstruction loss pushed the model to never use that half, but the gradient signal for outputs near 1.0 was weak because `tanh` saturates there.

`sigmoid` maps the full real line to `[0, 1]`, matching the target range exactly. No wasted capacity. Better gradient signal near the target maximum.

This is directly downstream of the v0.5.2 normalization fix. The normalization was corrected but the decoder output range was never updated to match. The model has been fighting a range mismatch on every training step.

---

### Change 2 — TransformerBlock: Post-norm → Pre-norm

**Impact:** Medium-High

Standard post-norm (normalize after residual) places LayerNorm outside the residual path. During backprop, gradients flow through the full unnormalized residual — useful early in training, increasingly unstable as the model deepens or the loss landscape sharpens.

Pre-norm (normalize before each sub-layer) keeps LayerNorm inside the residual bypass. The main path always has normalized inputs to attention and FFN. Gradients through the skip connection are unaffected by the normalization. This is why all modern transformer architectures (GPT-2 onward, PaLM, LLaMA) use pre-norm.

For a 4-layer encoder and 4-layer decoder trained from scratch, the difference is moderate but consistent: more stable loss curves, less sensitivity to learning rate choice.

Same parameters, different ordering in forward pass. Checkpoint-breaking.

---

### Change 3 — TransformerBlock FFN: `ReLU` → `GELU`

**Impact:** Low-Medium

GELU (Gaussian Error Linear Unit) is a smooth approximation of ReLU that gates activation by input magnitude. For transformer FFNs, this is now the standard — it produces slightly better representations across a wide range of language and speech tasks at zero parameter cost.

ReLU hard-zeros negative activations. GELU softly gates them. The difference is small per layer but compounds across 8 transformer blocks (4 encoder + 4 decoder).

---

### Change 4 — PosteriorEncoder: 1×1 Convolutions → 3×1 + GroupNorm

**Impact:** Medium-High

The posterior encoder previously used two `Conv1d(..., kernel_size=1)` layers — both are point-wise linear projections. Each mel frame was mapped to a latent vector independently. The encoder had no temporal context: it could not look at neighboring frames when estimating the posterior distribution.

Speech does not work this way. Phoneme coarticulation, pitch contours, and energy patterns all span multiple frames. A posterior that cannot see ±1 frame cannot capture any of this structure. The latent representations it produced were frame-local at best.

`kernel_size=3, padding=1` gives each frame ±1 neighboring frame at each conv layer — approximately ±23 ms at 22050 Hz / 256 hop. Two stacked 3×1 convs give an effective receptive field of ±2 frames.

`GroupNorm(1, hidden_size)` is instance normalization applied channel-wise. It normalises each channel across time, stable with small batches and fully compatible with AMP, unlike BatchNorm1d which degrades with batch_size < 8.

---

### Change 5 — DurationPredictor: Fixed 3rd Context Conv (kernel=5)

**Impact:** Medium

The duration predictor previously had two configurable conv layers (`kernel_sizes: [3, 3]`). Each layer sees ±1 phoneme context. After two layers, the effective receptive field is ±2 phonemes.

For a phoneme like /AH/ in the middle of a word, the predictor was estimating duration from a window of ±2 phonemes — too narrow to capture prosodic patterns that span phrases.

A fixed third conv layer with `kernel_size=5, padding=2` extends the receptive field to ±4 phonemes at the deepest layer, without adding a configurable parameter or changing the existing stack. This layer is always present regardless of `duration_predictor_kernel_sizes` in config.

Duration prediction is the component with the highest ceiling for improvement given real alignment targets. This patch makes the predictor architecturally capable of learning broader patterns from pseudo-targets in the meantime.

---

### Change 6 — Prior Projection: Linear → 2-Layer MLP

**Impact:** Medium

The text-conditioned latent prior previously mapped encoder output to latent space via a single linear layer:
```python
self.prior_proj = nn.Linear(encoder_hidden_size, latent_dim)
# 384 → 192: can only learn a linear relationship
```

A single linear layer can only learn linear combinations of phoneme embeddings. The relationship between text context and the latent distribution of speech is not linear — prosody, rhythm, and phoneme-specific variation all involve non-linear interactions between neighboring phoneme representations.

A 2-layer MLP with a GELU hidden layer adds one non-linear transformation:
```
encoder_hidden → encoder_hidden (GELU) → latent_dim
```

The hidden-layer size matches `encoder_hidden_size` (384), keeping the parameter count reasonable (~590K for the MLP vs ~74K for the linear layer — a 516K increase on a 22M param model).

---

### Change 7 — Multi-Scale Mel Loss: Add Scale=8

**Impact:** Low-Medium

The multi-scale mel loss previously computed L1 at temporal downsampling factors of 1×, 2×, and 4×.

Scale=4 averages over ~4 frames (~47 ms). This is sufficient to enforce local phoneme-level structure but blind to sentence-level prosody patterns that span 20–40 frames (230–465 ms).

Scale=8 (~93 ms averaging window) forces the model to get coarse temporal structure right — the rise and fall of energy across syllables and phrases, not just frame-by-frame accuracy.

One additional `avg_pool1d` + `l1_loss` per forward pass. Negligible VRAM and compute cost.

---

## Training Speed Changes (base.yaml)

| Setting | Before | After | Reason |
|---------|--------|-------|--------|
| `use_amp` | `false` | `true` | Float16 tensor cores: ~1.5–2× faster per step on RTX 3050 Ti |
| `batch_size` | `4` | `6` | AMP freed ~40% VRAM; larger batches = more stable gradients |
| `checkpoint_interval` | `500` | `1000` | 273 MB write every 500 steps was measurable I/O overhead |
| `loss_weight_duration` | `0.1` | `0.15` | Duration predictor was under-supervised relative to its improvement potential |
| `loss_weight_stft` | `0.1` | `0.15` | Multi-scale spectral loss was under-weighted at its original value |

---

## Compatibility

**Checkpoint-breaking.** Architecture changes (pre-norm, sigmoid, PosteriorEncoder convs, MLP prior) change parameter semantics and add parameters. No v0.5.2 checkpoint can be loaded.

The v0.5.2 training run was at step ~16,000 / ~306,000 (≈5%) when this patch was applied. Training restart cost: approximately 64 minutes of GPU time.

---

## What Changed, File by File

| File | Changes |
|------|---------|
| `hextts/vocoder/griffin_lim.py` | Correct mel→linear→audio pipeline; added `fmin`/`fmax` params |
| `utils/sample_generation.py` | Fixed denormalization formula; added `fmin`/`fmax` passthrough |
| `hextts/models/vits_impl.py` | Pre-norm, GELU FFN, sigmoid output, PosteriorEncoder, duration context conv, MLP prior |
| `hextts/training/trainer.py` | MultiScaleMelLoss scales (1,2,4) → (1,2,4,8) |
| `configs/base.yaml` | AMP, batch_size, checkpoint_interval, loss weights |

---

## What v0.5.3 Is Not

- Not a vocoder change. Griffin-Lim is still the fallback. HiFi-GAN is still the goal.
- Not a duration alignment fix. Pseudo-uniform targets are still in use. MFA is the next frontier.
- Not a vocabulary change. Arpabet, 40 tokens, unchanged.

---

## Developer Status

```
Architectural sins corrected    : 7
Sample audio bugs finally fixed : 2 (they were both broken, simultaneously, silently)
Checkpoints invalidated         : all of them (again)
GPU minutes lost to fresh start : ~64
GPU minutes saved by AMP        : ~400 (over full 100-epoch run)
Net GPU time saved              : +336 minutes
Regrets                         : fewer than last time
GPU fan RPM                     : unchanged (already at max, as always)
```

---

## Up Next (v0.5.4 Target)

- `duration_predictor_dropout: 0.5 → 0.3` — high dropout is actively preventing the predictor from learning
- `adam_beta1: 0.9 → 0.8`, `adam_beta2: 0.999 → 0.99` — VITS paper defaults for TTS convergence
- **Montreal Forced Aligner (MFA)** integration for real phoneme-level duration targets
  - Current pseudo-targets distribute frames uniformly across phonemes
  - MFA gives actual measured durations from forced alignment of audio to transcript
  - This is the single largest remaining quality ceiling
  - Requires: MFA installation, alignment run on LJSpeech, new data pipeline path
