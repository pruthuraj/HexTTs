# HexTTs Patch Note v0.5.1

## "Deep Dive Code Review: Everything That Was Wrong and Why We're Fixing It"

**Date:** 2026-04-27
**Tag:** v0.5.1
**Status:** In Progress — fixes staged from full codebase audit
**Urgency:** High — several issues silently corrupt training signals

---

## Overview

v0.5.1 is a **correctness and transparency release** triggered by a full codebase review performed after the v0.5.0 restructure.

No new features. No architecture changes. No vocoder swaps.

This patch is about fixing problems that were hiding in plain sight:

- silent normalization mismatches corrupting vocoder input
- batch skipping that makes loss curves look clean when they are not
- checkpoint loading that quietly runs unsafe Python deserialization
- a sample generation system that synthesizes nonsense and calls it "diagnostics"

In plain English: the codebase was structurally clean but had several places where failures were being swallowed instead of surfaced. This patch surfaces them.

---

## Audit Scope

Full read-through of:

- `hextts/config/` — load.py, schema.py
- `hextts/models/` — vits_impl.py, vits.py, checkpointing.py
- `hextts/data/` — raw_dataset.py, cached_dataset.py, dataloaders.py, collate.py
- `hextts/training/` — trainer.py, losses.py, callbacks.py, logging.py
- `hextts/inference/` — pipeline.py, synthesize.py, text_processing.py
- `hextts/vocoder/` — hifigan.py, griffin_lim.py, factory.py
- `configs/base.yaml`
- `tests/` — all test files

---

## Issues Found and Fixes Planned

### HIGH PRIORITY

---

#### Issue 1 — Mel Normalization Pipeline Is Inconsistent

**Severity:** High — silently degrades vocoder output quality

**Where it lives:**
- `hextts/data/raw_dataset.py` — mel output clipped to `[0, 1]`
- `hextts/models/vits_impl.py` Decoder — outputs `tanh()`, range `[-1, 1]`
- `hextts/inference/pipeline.py` — inverse-normalizes as `mel * -ref_level_db + ref_level_db`

**What is wrong:**

Three components in the same pipeline disagree on what range the mel-spectrogram is supposed to be in.

1. The dataset normalizes mel to `[0, 1]` via clip.
2. The decoder produces mel in `[-1, 1]` via tanh.
3. The inference pipeline inverse-normalizes using `ref_level_db` arithmetic that assumes a different convention.
4. The HiFi-GAN vocoder receives whatever comes out of step 3 — it was trained to expect a specific range and that range is not guaranteed here.

The failure mode is silent. The audio comes out wrong, the metrics look plausible, and nothing crashes.

**Fix applied:**

Root cause confirmed via math trace: `(mel_db - 20) / -20` produces values ≥ 1.0 for ALL valid mel_db ≤ 0, so every training target was clipped to constant 1.0. The reconstruction loss was trivially minimized; only KL and duration losses provided real gradient structure.

Changes:
- `raw_dataset.py`: formula changed to `(mel_db - min_level_db) / -min_level_db` — maps `[min_level_db, 0]` → `[0, 1]` correctly
- `pipeline.py`: inverse normalization updated to `norm * -min_level_db + min_level_db`; vocoder re-normalization updated to match
- `tests/test_normalization.py`: 12 tests added covering round-trip identity, range bounds, and a regression guard that confirms the old formula was broken

**Migration notice:** This change is checkpoint-breaking. All checkpoints trained before v0.5.1 used the broken normalization. The model must be retrained from scratch to benefit from the corrected reconstruction objective.

---

#### Issue 2 — Batch Skipping Is Opaque

**Severity:** High — makes loss curves misleading during instability

**Where it lives:**
- `hextts/training/trainer.py` — `train_epoch` method

**What is wrong:**

When a batch produces NaN/Inf losses or extreme duration values, the trainer silently skips it. The skipped batch is excluded from the loss average. From the outside, TensorBoard shows a clean loss curve.

This means:
- training can be deeply broken while metrics look stable
- there is no way to tell how many batches were skipped without adding debug prints
- the loss curve denominators are inconsistent step-to-step

**Fix:**

- Log a `train/skipped_batches` counter to TensorBoard every log interval.
- Emit a console warning with the skip reason (NaN loss / Inf loss / extreme duration).
- Do not hide the skip from the step count — log the skip as a step with a flagged metric.
- Add a configurable `max_skipped_ratio` threshold: if more than X% of batches in an epoch are skipped, stop training with a clear error instead of producing a useless checkpoint.

---

#### Issue 3 — Length Handling Differs Between Model and Trainer

**Severity:** High — can silently produce shape mismatches

**Where it lives:**
- `hextts/models/vits_impl.py` — uses `repeat_interleave` for duration expansion
- `hextts/training/trainer.py` — uses `F.interpolate` to align mel lengths

**What is wrong:**

`repeat_interleave` and `F.interpolate` are not the same operation. `repeat_interleave` copies frames according to integer counts. `F.interpolate` does bilinear or nearest-neighbor resampling. When lengths differ by even one frame, the interpolation path silently changes the signal. This creates a discrepancy between what the model produces and what the trainer compares it against.

**Fix:**

- Unify length handling: pick one strategy and use it everywhere.
- Preferred: use `repeat_interleave` (model path) as the canonical expansion, and handle any trailing frame differences by trimming/padding with zero frames.
- Remove the interpolation path from the trainer unless a specific justification exists.
- Add a shape assertion in the loss computation to catch mismatches early.

---

#### Issue 4 — Checkpoint Loading Falls Back to Unsafe Deserialization

**Severity:** High — loads arbitrary Python objects from checkpoint files

**Where it lives:**
- `hextts/models/checkpointing.py` — `load_checkpoint()`, lines 105–119

**What is wrong:**

The loader tries `weights_only=True` first (safe). If that fails — due to an older torch version or an incompatible checkpoint — it falls back to `weights_only=False`.

`weights_only=False` allows arbitrary Python object execution during unpickling. This is a well-documented PyTorch security issue. The fallback was added for compatibility, but it means any checkpoint file can execute arbitrary code on load.

**Fix:**

- Remove the `weights_only=False` fallback entirely.
- If loading fails with `weights_only=True`, raise a clear error with instructions:
  - upgrade torch to a compatible version, or
  - use the migration utility to re-save the checkpoint in the new format.
- Document the minimum torch version required in `pyproject.toml` and `CLAUDE.md`.

---

### MEDIUM PRIORITY

---

#### Issue 5 — Duration Clamping Has No Justification

**Severity:** Medium — caps speech rate control silently

**Where it lives:**
- `hextts/models/vits_impl.py` — DurationPredictor, clamp to `[1.0, 20.0]`

**What is wrong:**

Durations are clamped to a maximum of 20 frames per phoneme with no comment explaining why. At 22050 Hz with 256-frame hop, 20 frames = ~233 ms per phoneme. This is plausible for slow speech but the limit is not derived from any dataset analysis.

Additionally, the clamp interacts with `duration_scale` in inference. If `duration_scale = 1.5`, the effective maximum becomes 30 frames, but the model never learned durations above 20, so the clamp silently truncates the intended scale.

**Fix:**

- Add a comment explaining the derivation of the clamp bounds.
- Log a `train/clamped_duration_count` metric to TensorBoard when durations hit the ceiling.
- Consider making the clamp bounds configurable (`duration_min_frames`, `duration_max_frames` in `base.yaml`).

---

#### Issue 6 — Silent Failure in Text-to-Phonemes

**Severity:** Medium — inference continues with garbage phoneme sequences

**Where it lives:**
- `hextts/inference/pipeline.py` — `text_to_phonemes()`

**What is wrong:**

If `g2p_en` fails (missing dependency, unsupported input, internal error), the method prints a warning to stdout and returns the original text string. The pipeline then tries to look up each character as a phoneme, maps most of them to PAD, and synthesizes silence or noise without raising an error.

**Fix:**

- Raise `RuntimeError` on g2p failure with the original exception chained.
- Add a pre-check for `g2p_en` availability at inference pipeline initialization time.
- Return a typed `Optional[list[str]]` and handle `None` explicitly in the caller.

---

#### Issue 7 — Sample Generation in Trainer Uses Garbage Phoneme IDs

**Severity:** Medium — TensorBoard audio samples are not meaningful

**Where it lives:**
- `hextts/training/trainer.py` — sample generation, lines 36–39

**What is wrong:**

The trainer generates audio samples every N epochs for TensorBoard visualization. The text-to-ID conversion does `ord(char) % vocab_size`, which maps characters to random phoneme IDs. The generated audio is synthesized from meaningless phoneme sequences. It looks like a training diagnostic but provides no information about how the model is actually learning.

**Fix:**

- Use the real g2p pipeline (same as inference) to convert sample texts to phoneme IDs.
- If g2p is unavailable at training time, skip sample generation with a logged warning rather than generating noise.
- Make the sample texts configurable in `base.yaml` under `training.sample_texts`.

---

#### Issue 8 — Warning Counter Is Thread-Unsafe

**Severity:** Medium — warnings can be lost or double-counted with multiple workers

**Where it lives:**
- `hextts/data/raw_dataset.py` — module-level `defaultdict` for unknown phoneme tracking

**What is wrong:**

The warning counter is a module-level mutable dict. When `num_workers > 0`, each DataLoader worker runs in a separate process with its own copy of the dict. Warnings emitted by workers never reach the main process counter. The main process sees zero unknown phoneme warnings even when workers are logging many.

**Fix:**

- Move warning counting into a per-epoch summary logged from the trainer.
- Or use a multiprocessing-safe counter (e.g., `multiprocessing.Value`) if real-time counts are needed.
- At minimum, document the limitation: "warning counts are per-worker and not aggregated in multi-worker mode."

---

#### Issue 9 — Audio Error Handling Returns Silent Zeros

**Severity:** Medium — silently injects 5s of silence into training batches

**Where it lives:**
- `hextts/data/raw_dataset.py` — `__getitem__()` error handling

**What is wrong:**

If a wav file fails to load, the dataset returns a 5-second tensor of zeros in place of the real audio. The training loop receives this silently as a valid batch. This contaminates loss computation and produces misleading gradient updates with no visible error.

**Fix:**

- Raise an exception on load failure and let the DataLoader's error handling surface it.
- Or: return `None` and add a filter in the collate function that drops None items and logs the skip.
- Do not silently inject zero tensors into training data.

---

### LOW PRIORITY

---

#### Issue 10 — VITSTrainer Is Too Long

**Severity:** Low — maintenance concern

**Where it lives:**
- `hextts/training/trainer.py` — 793 lines

**What is wrong:**

The trainer class handles model optimization, loss computation, TensorBoard logging, checkpointing, sample generation, and early stopping all in one class. Individual methods are up to 150 lines. This makes isolated testing of any one concern very difficult.

**Fix (non-breaking):**

- Extract `LossComputer` into `hextts/training/losses.py` alongside `MultiScaleMelLoss`.
- Extract `CheckpointManager` into `hextts/training/checkpoint_manager.py`.
- Keep trainer as the orchestrator, not the implementor.
- No behavior changes required — this is structure only.

---

#### Issue 11 — Dead Code in Trainer

**Severity:** Low — clutter

**Where it lives:**
- `hextts/training/trainer.py` — lines ~350, ~454 (commented-out blocks)

**Fix:**

Remove the commented-out blocks. They predate the current hybrid duration path and are no longer relevant. Git history preserves them if needed.

---

#### Issue 12 — Test Coverage Is Minimal

**Severity:** Low — risk surface for future changes

**What is missing:**

- No test for NaN/Inf propagation through the forward pass
- No test for duration regulation correctness (repeat_interleave output shape)
- No test for loss computation with known inputs and expected outputs
- No test for checkpoint round-trip (save → load → weights identical)
- No gradient flow test

**Fix:**

Add targeted tests in priority order:
1. `test_training.py` — loss computation correctness with toy inputs
2. `test_checkpointing.py` — round-trip save/load weight equality
3. `test_duration.py` — repeat_interleave output shape matches mel_length
4. `test_normalization.py` — mel range at dataset output and decoder output boundaries

---

## Risk Assessment

### What these fixes touch

| Fix | Files | Checkpoint risk | VRAM risk |
|---|---|---|---|
| Mel normalization | raw_dataset.py, pipeline.py | None (no architecture change) | None |
| Batch skip logging | trainer.py | None | None |
| Length unification | trainer.py, vits_impl.py | Low (shape change in loss path) | None |
| Checkpoint security | checkpointing.py | Low (load path change) | None |
| Duration clamp docs | vits_impl.py | None | None |
| g2p failure raise | pipeline.py | None | None |
| Sample generation fix | trainer.py | None | None |
| Warning counter | raw_dataset.py | None | None |
| Silent zero fix | raw_dataset.py | None | None |

### What these fixes do NOT touch

- Model architecture (encoder, decoder, postnet, duration predictor weights)
- Vocoder (HiFi-GAN architecture or weights)
- Loss weight values
- Audio configuration (sample rate, n_mels, FFT parameters)
- Vocabulary or phoneme mapping
- Existing checkpoints

No checkpoint migration required for any of these fixes.

---

## Implementation Order and Status

| # | Issue | File(s) | Status |
|---|---|---|---|
| 1 | Checkpoint loading security | `checkpointing.py` | ✅ Done |
| 2 | Batch skip logging | `trainer.py` | ✅ Done |
| 3 | Silent audio zero fix | `raw_dataset.py` | ✅ Done |
| 4 | g2p failure raise | `inference/pipeline.py` | ✅ Done |
| 5 | Sample generation fix | `trainer.py` | ✅ Done |
| 6 | Duration clamp documentation | `vits_impl.py` | ✅ Done |
| 7 | Warning counter note | `raw_dataset.py` | ✅ Done |
| 8 | Mel normalization fix | `raw_dataset.py`, `pipeline.py` | ✅ Done — **checkpoint-breaking, retraining required** |
| 9 | Length mismatch logging | `trainer.py` | ✅ Done — structural unification deferred |
| 10 | Tests | `tests/` | ✅ Done — 20 new tests added |
| 11 | Trainer refactor | `trainer.py` | Deferred |

---

## Validation Plan

After each fix:

```bash
python -m pytest tests/test_config.py
python -m pytest tests/test_model_shapes.py
python -m pytest tests/test_checkpointing.py
python -m pytest -q
```

After mel normalization fix:

```bash
python scripts/infer.py --checkpoint checkpoints/best_model.pt --config configs/base.yaml \
  --text "hello world" --output tts_output/output_v051.wav --device cpu
python scripts/evaluate_tts_output.py --audio tts_output/output_v051.wav --sample_rate 22050
```

Compare output duration, ZCR, and spectral flatness against the v0.4.7 baseline:

- Duration: `1.8576 s` (v0.4.7 reference)
- ZCR: `0.114986`
- Spectral flatness: `0.015726`

Any large regression in these metrics after a "non-functional" fix is a red flag.

---

## What v0.5.1 Is Not

- Not a model improvement patch.
- Not a vocoder change.
- Not a feature release.
- Not a performance optimization.

This patch makes the existing system behave the way it was supposed to. The model quality ceiling is unchanged. The floor is higher because fewer silent failures can drag it down.

---

## Final Words

v0.5.0 gave the project a clean structure and a proper package layout.

v0.5.1 makes that structure honest.

The goal is a codebase where a failure is a crash, a warning is logged, and a loss curve means what it says. That is the foundation every future training run depends on.
