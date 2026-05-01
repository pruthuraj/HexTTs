# CHANGELOG

All notable changes to HexTTs are documented here.
All notable GPU temperature increases are documented in your electricity bill.

Migration note (v0.5.x): older entries may mention legacy files such as `train_vits.py`,
`inference_vits.py`, and `vits_config.yaml`. Current active entrypoints/configs are
`scripts/train.py`, `scripts/infer.py`, and `configs/base.yaml` (plus other profiles in `configs/`).

---

## [v0.5.4] - 2026-05-01

### _"The Duration Predictor Finally Has Real Targets To Learn From (And We Didn't Mess Up The Imports This Time)"_

### Summary

One new script, 12,884 duration files, zero architectural changes, and one import path that was quietly
failing for three hours because Python doesn't shout about missing packages inside exception handlers.

The headline: phoneme-level duration alignment is complete. All 12,884 LJSpeech utterances now have real,
ground-truth duration targets extracted via torchaudio forced alignment to WAV2VEC2_ASR_BASE_960H.
The duration predictor can now learn what it is supposed to predict instead of guessing from reconstruction
loss gradients and hoping for the best.

The subplot: the alignment script spent six hours producing 0% real alignments (100% fallback) despite
98% success rate in isolation diagnostics. Root cause: `hextts` package import inside exception handler.
Silent failure. Hidden by catch. Fixed once sys.path was corrected. Moral: don't catch exceptions without
logging them first.

### New

- **Windows-Native Phoneme Alignment** — `scripts/align_torchaudio.py`
  - Single-script alternative to MFA (Linux/conda, external tools, manual setup)
  - Uses `torchaudio.functional.forced_align()` with pretrained WAV2VEC2_ASR_BASE_960H model (960 hours LibriSpeech)
  - Inputs: audio directory, metadata CSV, prepared data directory
  - Outputs: per-file duration arrays saved as `.npy` (int32, frame-level frame counts per phoneme)
  - Batched inference with DataLoader + configurable workers, prefetch, and AMP for GPU acceleration
  - GPU throughput: **6.22 files/sec** on RTX 3050 Ti (batch_size=8, num_workers=4, AMP, TF32)
  - Fallback strategy for sparse alignments: proportional frame redistribution across phonemes (handles edge cases)
  - Real alignment coverage: **100%** (12,884 real, 0 fallback) on full LJSpeech after import fix

- **Duration Directory Configuration** — `configs/base.yaml`, `configs/continue_auto.yaml`, `configs/continue3.yaml`, `configs/debug.yaml`, `configs/sanity.yaml`
  - Added `duration_dir: ./data/ljspeech_prepared/durations` to all config profiles
  - Trainer now loads real duration targets by default if directory exists; graceful fallback to pseudo-uniform if missing
  - All training modes (base, continuation, debug, sanity) now use real targets

### Fixed

- **Silent hextts Import Failure in align_torchaudio.py** — `scripts/align_torchaudio.py` line 47
  - `phonemes_per_word()` from `hextts.data.preprocessing` was being imported inside exception handler
  - When called in batch mode, import error was silently swallowed → PPW calculation returned None → all alignments fell back
  - Diagnostic tests worked because they ran outside the batched pipeline
  - Fix: added `sys.path.insert(0, str(Path(__file__).parent.parent))` at module top
  - Added exception logging in `phonemes_per_word()` to expose hidden errors
  - Result: real alignment coverage restored from 0% to 100%

### Improved

- **Alignment Robustness** — `scripts/align_torchaudio.py`
  - Flexible span grouping: if word count from WAV2VEC2 alignment mismatches text g2p,
    proportionally redistribute available phonemes across alignment frames instead of rejecting batch
  - Covers edge cases: merged contractions, tokenizer quirks, pronunciation variations
  - Logs mismatch summary (real vs fallback) at end

- **GPU Optimization Defaults** — `scripts/align_torchaudio.py` (CLI options)
  - `--device cuda` (or `cpu`; GPU by default if CUDA available)
  - `--batch_size 8` (GPU batching, default 1 for CPU)
  - `--num_workers 4` (parallel audio loading, Windows-safe)
  - `--prefetch_factor 4` (reduced to 1 on macOS for stability)
  - `--disable_amp` flag to turn off AMP if needed (default: AMP enabled on CUDA)
  - Non-blocking GPU transfers: `to(device, non_blocking=(device.type == "cuda"))`
  - TF32 + cuDNN benchmark mode for tensor cores: `torch.backends.cuda.matmul.allow_tf32 = True`

- **TensorBoard Duration Diagnostics** — `hextts/training/trainer.py` (unchanged API, inherited in v0.5.3)
  - Trainer already logs `train/using_real_duration_targets` to confirm real vs pseudo mode
  - Now configs send real targets by default; log value changes from 0.0 to 1.0

### Developer Status

```
Duration files generated           : 12,884 (100% real alignment)
Files processed                    : 12,884
Alignment fallback rate            : 0% (initial 100% before import fix)
GPU throughput achieved            : 6.22 files/sec
Invisible bugs fixed               : 1 (hextts import + exception handler)
Config files updated               : 5 (all training profiles)
Time spent debugging why 0% worked : 3 hours (worth it)
Moral of the story                 : catch exceptions, log them, move on
```

### Expected Impact

- Duration predictor loss should converge faster (real targets vs proxy-learned targets)
- Inference duration predictions should be more realistic (trained on ground truth)
- No model architecture changes; resumable from any v0.5.3 checkpoint
- Fresh training with real durations recommended for best quality

### Compatibility

- **Checkpoint-safe.** No architecture or model state changes. All v0.5.3 checkpoints remain valid.
- **Config-compatible.** The `duration_dir` key is optional; if missing or empty, trainer falls back to pseudo-uniform durations.
- **Backward-compatible.** Script runs on CPU if CUDA unavailable; defaults are conservative.

---

## [v0.5.3] - 2026-04-28

### _"The Model Was Doing Everything Wrong Architecturally And We Fixed It"_

### Summary

Seven architecture improvements, two silent pipeline bugs fixed, and a training speed pass. v0.5.2
made the reconstruction loss actually work. v0.5.3 makes the model actually worth training.

The headline bugs: Griffin-Lim was being called directly on a mel spectrogram (crashes, swallowed by
a warn handler, zero valid audio samples since audio generation was introduced). The sample denormalization
was still using the pre-v0.5.2 broken formula. Both were silent. Both have been wrong for a long time.

On the architecture side: the decoder was outputting `tanh` (range [-1,1]) while all targets are in [0,1]
after the v0.5.2 normalization fix. The transformer blocks were post-norm when they should be pre-norm.
The posterior encoder had 1×1 convolutions with no temporal context. The prior was a single linear layer.
All fixed. Requires fresh training — the v0.5.2 run was only ~5% complete so the restart cost is small.

### Fixed

- **Griffin-Lim Called on Mel Spectrogram** — `hextts/vocoder/griffin_lim.py`, `utils/sample_generation.py`
  - `librosa.griffinlim()` expects a linear STFT (513 bins), not a mel spectrogram (80 bins)
  - Crash was caught by `warnings.warn`, so training continued with zero valid audio samples at every epoch
  - Fix: correct pipeline — `mel_db → power → mel_to_stft (pseudo-inverse) → griffinlim`
  - Added `fmin`/`fmax` parameters so the filterbank inversion matches the training filterbank

- **Sample Generation Denormalization Bug** — `utils/sample_generation.py`
  - Was using `mel * (-ref_level_db) + ref_level_db` = `mel * (-20) + 20` — the pre-v0.5.2 broken formula
  - Correct inverse of v0.5.2 normalization: `mel * (-min_level_db) + min_level_db` = `mel * 100 - 100`
  - v0.5.2 fixed this everywhere except sample generation; now consistent across the full pipeline

### Architecture Changes (all checkpoint-breaking, fresh training required)

- **Decoder output: `tanh` → `sigmoid`** — `hextts/models/vits_impl.py`
  - Training targets are in [0, 1]; `tanh` output [-1, 1] was wasting half the output range
  - `sigmoid` maps the full real line to [0, 1], matching targets exactly, better gradient at saturation

- **TransformerBlock: post-norm → pre-norm** — `hextts/models/vits_impl.py`
  - Pre-norm places LayerNorm before each sub-layer, keeping the skip path clean for gradients
  - More stable training curves, less LR sensitivity — standard in all modern transformer architectures

- **TransformerBlock FFN: `ReLU` → `GELU`** — `hextts/models/vits_impl.py`
  - GELU softly gates negative activations rather than hard-zeroing them
  - Standard upgrade for transformer FFNs, no parameter change

- **PosteriorEncoder: 1×1 convs → 3×1 convs + GroupNorm** — `hextts/models/vits_impl.py`
  - 1×1 convolutions are point-wise projections with zero temporal context
  - `kernel_size=3, padding=1` gives each frame ±1 neighboring frame (~23 ms per layer)
  - `GroupNorm(1, hidden_size)` (instance norm) — stable with small batches, AMP-compatible

- **DurationPredictor: fixed 3rd context conv (kernel=5)** — `hextts/models/vits_impl.py`
  - Previous 2-layer stack with kernel=3 had effective receptive field of ±2 phonemes
  - Fixed third conv (kernel=5) extends receptive field to ±4 phonemes for broader prosodic context

- **Prior projection: linear → 2-layer MLP with GELU** — `hextts/models/vits_impl.py`
  - Single linear prior could only learn linear phoneme→latent mappings
  - 2-layer MLP (hidden dim = encoder_hidden_size, GELU) adds non-linear expressive capacity

- **Multi-scale mel loss: add scale=8** — `hextts/training/trainer.py`
  - Previous scales (1, 2, 4) captured up to ~47 ms temporal structure
  - Scale=8 (~93 ms) enforces sentence-level prosodic structure, not just local frame accuracy

### Training Speed (base.yaml)

- `use_amp: true` — float16 tensor cores: ~1.5–2× faster per step on RTX 3050 Ti
- `batch_size: 6` — AMP freed VRAM headroom; larger batches, more stable gradients
- `checkpoint_interval: 1000` — reduced 273 MB checkpoint write frequency
- `loss_weight_duration: 0.15`, `loss_weight_stft: 0.15` — both were under-weighted at 0.1

### Compatibility

- **Checkpoint-breaking.** Architecture changes add parameters and change forward-pass semantics.
  No v0.5.2 checkpoint can be loaded under v0.5.3. Retrain from scratch.

### Developer Status

```
Architectural sins corrected    : 7
Sample audio bugs fixed         : 2 (both broken simultaneously, both silent)
Checkpoints invalidated         : all of them (again) (the v0.5.2 run was only 5% done)
Net GPU time saved vs continuing: +336 minutes (AMP savings over full run > restart cost)
GPU fan RPM                     : unchanged (already at max, as always)
```

---

## [v0.5.2] - 2026-04-27

### _"The Model Was Lying To You The Entire Time And We Fixed It"_

### Summary

Correctness and transparency audit of the full v0.5.x codebase. No new features. No architecture changes.
Just a systematic elimination of places where the training pipeline was failing silently, swallowing errors,
or producing numbers that looked fine and meant nothing.

The headline finding: the mel normalization formula was mathematically broken. Every training target was
being clipped to a constant 1.0. The reconstruction loss had been useless since the beginning.
The model was learning entirely from KL and duration loss, with the reconstruction objective doing nothing.
This is checkpoint-breaking. All prior checkpoints are invalid under v0.5.2. Retraining required.

Everything else in this patch is about making failures loud, not silent.

### Fixed

- **Mel Normalization Formula** — `hextts/data/raw_dataset.py`, `hextts/inference/pipeline.py`
  - The old formula: `(mel_db - ref_level_db) / -ref_level_db` with `ref_level_db = 20`
  - For any valid mel_db ≤ 0 (which is all of them), this produces a value ≥ 1.0
  - Every training target was clipped to constant 1.0. The reconstruction loss minimized itself
    trivially and provided zero gradient signal about actual mel structure.
    The model was optimizing KL and duration loss into a pretend mel landscape.
  - The new formula: `(mel_db - min_level_db) / -min_level_db` with `min_level_db = -100`
  - This correctly maps `[min_level_db, 0]` → `[0, 1]`. The reconstruction objective now works.
  - Inference inverse normalization and vocoder re-normalization updated to match.
  - **Migration notice: all checkpoints trained before v0.5.2 used the broken normalization.
    Retrain from scratch.**

- **Checkpoint Loading Security** — `hextts/models/checkpointing.py`
  - Removed the `weights_only=False` fallback that ran when `weights_only=True` failed.
  - `weights_only=False` allows arbitrary Python object execution during unpickling.
    Any checkpoint file could execute arbitrary code. The fallback existed for "compatibility."
    Compatibility with arbitrary code execution is not a feature.
  - If loading fails, a clear error is raised with instructions to re-save or upgrade torch.

- **Silent Audio Zero Injection** — `hextts/data/raw_dataset.py`
  - If a wav file failed to load, the dataset returned 5 seconds of zero-tensor silence.
  - The training loop received this as a valid batch. The loss computed. The gradient updated.
    The model learned something, probably about silence.
  - Now raises an exception. The error surfaces. You fix the file.

- **g2p Silent Passthrough** — `hextts/inference/pipeline.py`
  - If `g2p_en` failed, `text_to_phonemes()` printed a warning and returned the raw input string.
  - The pipeline then mapped each character to PAD or garbage phoneme IDs and synthesized noise.
    The inference completed. The audio was meaningless. Nothing crashed.
  - Now raises `RuntimeError` with the original exception chained. The failure is visible.

- **Garbage Sample Generation in Trainer** — `hextts/training/trainer.py`
  - The trainer's TensorBoard sample generation used `ord(char) % vocab_size` to convert text to IDs.
    This maps characters to arbitrary phoneme IDs. The synthesized audio was nonsense.
    It was logged as a training diagnostic. It was not a diagnostic. It was noise with a label.
  - Now uses the real g2p pipeline (same as inference). If g2p is unavailable, sample generation
    is skipped with a logged warning instead of producing meaningless audio.

### Improved

- **Batch Skip Logging** — `hextts/training/trainer.py`
  - Previously, skipped batches were excluded from loss averages with no record.
    TensorBoard showed a clean curve. The curve was a lie.
  - Now logs `train/skipped_batches` to TensorBoard at every log interval.
  - Emits a console warning with the skip reason (NaN loss / Inf loss / extreme duration).
  - New config key: `max_skipped_ratio` — if more than X% of batches in an epoch are skipped,
    training stops with a clear error instead of producing a checkpoint trained on nothing.

- **Length Mismatch Visibility** — `hextts/training/trainer.py`
  - Added `train/length_mismatch_frames` TensorBoard metric to surface frame-count mismatches
    between predicted mel and ground truth mel.
  - The structural unification of `repeat_interleave` vs `F.interpolate` paths is deferred
    to a follow-up patch.

- **Duration Clamp Documentation** — `hextts/models/vits_impl.py`
  - The `[1.0, 20.0]` frame clamp now has a comment explaining the derivation:
    20 frames × 256 hop / 22050 Hz ≈ 233 ms per phoneme. Previously a magic number.

- **Warning Counter Limitation Documented** — `hextts/data/raw_dataset.py`
  - The module-level unknown phoneme counter is per-worker and not aggregated in multi-worker mode.
    Workers run in separate processes; their counts never reach the main process.
    This limitation is now documented in the code so nobody spends an afternoon debugging it.

### New TensorBoard Scalars

| Scalar                         | What It Tells You                                                      |
| ------------------------------ | ---------------------------------------------------------------------- |
| `train/skipped_batches`        | How many batches were silently excluded from this epoch's loss average |
| `train/length_mismatch_frames` | Frame-count delta between predicted and target mel                     |

### Tests Added

- `tests/test_normalization.py` — 12 tests covering round-trip identity, range bounds,
  and a regression guard that confirms the old formula (`ref_level_db=20`) was broken
- `tests/test_training.py` — 8 tests covering batch skip logic, trainer construction,
  and loss computation

**Test suite: 40/40 passing.**

### Compatibility

- **Checkpoint-breaking.** The mel normalization fix changes what the model is trained to predict.
  All checkpoints from v0.5.1 and earlier produced targets clamped to constant 1.0.
  Those checkpoints cannot be fine-tuned; the reconstruction objective has changed.
  Start fresh. The model will actually learn this time.
- No model architecture changes. No vocab changes. No config format changes beyond `max_skipped_ratio`.
- `max_skipped_ratio` defaults to a safe value; existing training runs are unaffected unless explicitly set.

### Developer Status

```
Silent failures eliminated   : 6
Reconstruction loss fixed    : yes (it was broken)
Checkpoints invalidated      : all of them (they were wrong)
Regrets about not finding this sooner : substantial
GPU fan RPM                  : unchanged (it was already at max, as always)
```

---

## [v0.5.1] - 2026-04-27

### _"The Model Can Finally Feel Shame About Its Mistakes"_

### Summary

Six training quality improvements in one patch. The model was previously optimizing with the
enthusiasm of a student who never checks their answers. Now it gets feedback at multiple levels,
its attention mechanism stops hallucinating over padding, and the learning rate no longer sprints
off a cliff on step one. Whether any of this makes the audio sound better is left as an exercise
for your ears and your TensorBoard dashboard.

### Added

- **KL Annealing** — `configs/base.yaml`, `hextts/training/trainer.py`
  - New config key: `kl_warmup_steps: 10000`
  - KL weight now ramps linearly from 0 → `loss_weight_kl` over the first 10k steps
  - Previously, KL pressure was applied from step 1, which encouraged the posterior encoder
    to give up immediately and output `N(0,1)` regardless of the input mel
  - Technically this is called posterior collapse. Emotionally it is called the model lying to you.
  - New TensorBoard scalar `train/kl_anneal_factor` so you can watch the ramp in real time
    and feel like you're in control of something

- **Dual Mel Supervision (Pre + Post PostNet)** — `hextts/models/vits_impl.py`, `hextts/training/trainer.py`
  - New config key: `loss_weight_pre_postnet: 0.5`
  - `forward()` now returns `decoder_mel` (pre-PostNet) alongside `predicted_mel` (post-PostNet)
  - Both are supervised with L1 loss against the ground truth mel
  - Previously, PostNet had no idea what it was supposed to fix — it just received gradient from
    the final combined output and had to figure it out telepathically
  - Now the decoder is forced to produce a usable coarse mel independently, and PostNet learns
    an actual residual instead of compensating for everything simultaneously
  - Logged as `train/pre_postnet_loss`

- **Multi-Scale Spectral Mel Loss** — `hextts/training/losses.py`, `hextts/training/trainer.py`
  - New config key: `loss_weight_stft: 0.1`
  - `MultiScaleMelLoss` computes L1 at 1×, 2×, and 4× temporal downsampling of the mel
  - Plain per-frame L1 loss is blind to coarse temporal structure — a predicted mel that is
    shifted by a few frames can score perfectly fine while sounding completely wrong
  - Multi-scale supervision forces the model to be correct at multiple time granularities,
    not just locally
  - Logged as `train/ms_mel_loss`

- **LR Warmup** — `hextts/training/trainer.py`, `configs/base.yaml`
  - `warmup_steps: 1000` — was in `base.yaml` but completely ignored by the trainer.
    It was sitting there the whole time, doing nothing, like a fire extinguisher with no pin.
  - LR now linearly ramps from ~0 → `learning_rate` over the first 1000 steps
  - Main exponential scheduler is held back until warmup completes
  - Transformer attention heads are sensitive to large gradient updates at initialization —
    without warmup, early steps can permanently damage attention patterns before
    the model has any idea what it is doing

- **`inference_noise_scale` Wired to Config** — `hextts/models/vits_impl.py`, `configs/base.yaml`
  - New config key: `inference_noise_scale: 0.3`
  - Previously hardcoded as `* 0.3` in `vits_impl.py`, invisible to config, tunable by no one
  - Now readable, overridable, and documented like a responsible hyperparameter
  - Controls how much random variation is injected into the latent at inference time.
    Higher = more expressive and unpredictable. Lower = more consistent and slightly robotic.
    0.3 is the polite middle ground that nobody will complain about.

- **Decoder Padding Masks** — `hextts/models/vits_impl.py`
  - `Decoder.forward()` now accepts a `mask` parameter passed down to each `TransformerBlock`
  - Training: mask built from ground-truth `mel_lengths` (actual frame counts per batch item)
  - Inference: mask built from `regulated_lengths` (sum of predicted per-phoneme durations)
  - Previously, padding positions at the end of shorter batch items were attending to — and being
    attended to by — real tokens in every decoder layer on every training step.
    The model learned this. We are sorry.

### Improved

- **`_length_regulate` Vectorization** — `hextts/models/vits_impl.py`
  - Replaced nested Python for-loop (with per-frame `.item()` CPU-GPU sync) with
    `torch.repeat_interleave`, a single vectorized CUDA kernel per batch item
  - The old code triggered approximately 600 CPU-GPU round-trips per forward pass on a
    typical sequence. On Windows, where CUDA sync overhead is generously high, this was
    quietly degrading throughput on every single training step. It had been doing this
    the entire time. Nobody asked how it was doing. Nobody checked.

- **Decoder Config Params** — `hextts/models/vits_impl.py`
  - `Decoder.__init__` now reads `num_heads`, `kernel_size`, and `dropout` from config
    instead of hardcoded values (`2`, `3`, `0.1`)
  - Previously, changing `decoder_num_heads` in `base.yaml` had exactly zero effect on anything.
    The config key existed. It was validated. It was logged. It was silently ignored.
  - Checkpoint-safe: defaults match the old hardcoded values, so existing checkpoints load cleanly

### New TensorBoard Scalars

| Scalar                   | What It Tells You                                       |
| ------------------------ | ------------------------------------------------------- |
| `train/kl_anneal_factor` | KL ramp progress (0.0 → 1.0 over `kl_warmup_steps`)     |
| `train/pre_postnet_loss` | How bad the decoder mel is before PostNet covers for it |
| `train/ms_mel_loss`      | Multi-scale temporal alignment quality                  |

### Compatibility

- No checkpoint format changes. All improvements are training-time only.
- Existing checkpoints resume correctly. `global_step` is restored from checkpoint, so
  the KL annealing ramp knows how far along it already is and won't start over.
- `VITS.forward()` gains an optional `mel_lengths` parameter (default `None`).
  Callers that omit it train without decoder masks — identical to previous behaviour.
  The trainer now passes it.

### Known Risks

- If `kl_warmup_steps` is too short relative to total training budget, KL pressure ramps up
  while the decoder is still learning what a mel spectrogram is supposed to look like.
  If `train/kl_loss` spikes after the ramp completes, try `loss_weight_kl: 0.05`
  or extend `kl_warmup_steps: 20000`.

### Developer Status

```
New losses added         : 3
Hardcoded values removed : 2
CPU-GPU syncs eliminated : ~600 per forward pass
Regrets                  : mild
GPU fan RPM              : unchanged (already at max, as always)
```

---

## [v0.5.0] - 2026-04-15

### Summary

- Continued package-first refactor cleanup with focus on docs/runtime alignment.
- Kept active wrappers stable for existing workflows (`scripts/train.py`, `scripts/infer.py`).

### Updated

- Refined active markdown guidance to match current script-first command paths and config profiles.
- Clarified architecture transition messaging for `hextts/` package layout usage.
- Improved consistency between README quick notes and changelog release notes.

### Compatibility

- No breaking CLI path changes in this patch line.
- Existing training/inference commands remain valid under the current wrapper flow.

### Detailed Notes

- For full refactor scope and phased breakdown, see `REFACTOR_PLAN.md`.

---

## [v0.4.7] - 2026-04-09

### Added

- **Duration Debug Verification Hooks** — `train_vits.py`
  - Optional config flag: `duration_debug_checks`
  - Prints one train/val sample with:
    - `phoneme_length`
    - `mel_length`
    - `target_duration` vector and sum
    - `predicted_duration` vector and sum
    - proxy formula readout (`pred_sum / phoneme_length`)
  - Purpose: verify scale consistency before trying new duration weighting ideas again

- **Continuation Text Report Output** — `scripts/run_continuation_test.py`
  - New CLI option: `--report-file` (default: `reports/continuation_test_report.txt`)
  - Report now includes:
    - training snapshots (`loss`, `recon`, `kl`, `dur`)
    - full `HexTTS Output Evaluation Report`
    - final `CONTINUATION TEST SUMMARY`

### Improved

- **Continuation Streaming UX** — progress output now streams in larger chunks
  - Keeps `tqdm` progress display more readable in wrapper output
  - Reduces repeated-line noise from carriage-return redraws

### Fixed

- **Duration Supervision Regression Recovery**
  - Rolled back the failed phoneme-aware pseudo-duration experiment
  - Restored previous working hybrid supervision path (uniform sum-preserving token targets + token/sum loss)
  - Removed phoneme-weighted target path from active training logic

### Validation Snapshot

- Debug continuation command:
  - `venv\Scripts\python.exe scripts/run_continuation_test.py --epochs 1 --duration-debug-checks --report-file reports/continuation_test_report_debug.txt`
- Confirmed target sums matched mel lengths in debug samples
- Confirmed proxy returned to healthy range (~8.26 to ~8.35)
- Output duration recovered to realistic value (`1.8576 s`) with clean waveform metrics:
  - `ZCR: 0.114986`
  - `Spectral flatness: 0.015726`

---

## [v0.4.6] - 2026-04-09

### Added

- **Continuation Test Automation Script** — `scripts/run_continuation_test.py`
  - Builds a temporary continuation config from base config
  - Resumes training from a checkpoint
  - Extracts latest TensorBoard duration diagnostics
  - Runs HiFi-GAN inference on fixed sentence
  - Runs objective evaluation and prints compact summary

- **New Main Flow Subcommand** — `continuation-test`
  - Added to `scripts/main_flow.py`
  - One command for the full continuation experiment:
    - `python scripts/main_flow.py continuation-test --epochs 3`

### Improved

- **README Workflow Coverage**
  - Updated patch notes version label to include latest automation update
  - Added continuation-test command in simplified main-flow section
  - Added dedicated continuation automation section with usage and behavior

### Internal

- Reduced manual command chaining for continuation experiments
- Lowered risk of missing diagnostics during short resume runs

---

## [v0.4.5] - 2026-04-09

### Added

- **Simplified Main Flow Wrapper** — new `scripts/main_flow.py` so you can stop typing command-line essays
  - `train`: wraps `train_vits.py`
  - `infer`: wraps `inference_vits.py` (supports `--hifigan`)
  - `eval`: wraps `evaluate_tts_output.py`
  - `audit`: wraps dataset audit/filter flow with common threshold flags
  - `compare`: runs Griffin-Lim vs HiFi-GAN and evaluates both in one go

### Improved

- **Main Flow Usability** — common tasks are now one command away instead of six flags away
  - Added sensible defaults for checkpoint, config, output paths, and sample rate
  - Keeps the original scripts untouched for power users and command-line gladiators

- **Repository Root Cleanup** — moved utility tools from root into `scripts/`
  - `audit_dataset.py` → `scripts/audit_dataset.py`
  - `evaluate_tts_output.py` → `scripts/evaluate_tts_output.py`
  - `view_spectrogram.py` → `scripts/view_spectrogram.py`
  - `test_setup.py` → `scripts/test_setup.py`
  - Updated references in workflow docs and wrappers to match new paths

- **README Organization** — documented the new simplified flow prominently
  - Added "Simplified Main Flow" section with concrete examples
  - Updated version/footer references to v0.4.5
  - Updated project structure description for `scripts/`

### Internal

- Reduced copy-paste risk in day-to-day workflow commands
- Reduced chance of typo-driven spiritual damage

---

## [v0.4.4] - 2026-04-09

### Added

- **Real HiFi-GAN Inference Path** — The neural vocoder is now a real option, not a motivational poster
  - `inference_vits.py` accepts `--vocoder_checkpoint` and `--vocoder_config`
  - If both are provided, inference uses HiFi-GAN
  - If omitted, it falls back to Griffin-Lim (for nostalgia and mild suffering)

- **HiFi-GAN Checkpoint Compatibility** — Official `generator_v1` now loads without throwing a novel-length traceback
  - Updated vocoder architecture to match official `resblocks.*.convs1/convs2` key layout
  - Added support for resblock type selection from vocoder config
  - Weight norm removal aligned with the updated module structure

- **README Patch Notes & Navigation** — Documentation now easier to scan without scrolling through your entire lifespan
  - Added quick navigation block
  - Added explicit v0.4.4 patch notes in README
  - Added clear HiFi-GAN usage examples and A/B comparison commands

### Improved

- **Buzz Detection Guidance** — Spectral flatness is now documented as a practical metric instead of mysterious academic decoration
  - Added interpretation bands (speech-like vs mild noise vs buzzy/noisy)
  - Included side-by-side Griffin-Lim vs HiFi-GAN example workflow

- **Inference Output Consistency** — Audio duration reporting now respects active backend sample rate
  - Prevents misleading duration printouts when vocoder sample rate differs from base config

### Fixed

- **State Dict Mismatch Crash** — `RuntimeError: Missing key(s)/Unexpected key(s)` when loading HiFi-GAN checkpoint
  - Root cause: local wrapper used a simplified ResBlock layout incompatible with official checkpoint format
  - Resolution: updated vocoder internals to official-compatible module naming and structure
  - Result: checkpoint loads and synthesis completes successfully

### Validation Snapshot

- Generated `tts_output/gl_test.wav` (Griffin-Lim) and `tts_output/hifigan_test.wav` (HiFi-GAN)
- Evaluation on same sentence showed lower ZCR and lower spectral flatness for HiFi-GAN
- Translation: less metallic buzz, more speech-like waveform structure

---

## [v0.4.3] - 2026-04-07

### Added

- **PostNet for Mel-Spectrogram Refinement** — Neural network that learns to refine decoder output
  - New `PostNet` class that predicts mel-scale residuals
  - Added to both training and inference pipelines
  - Reduces artifacts in generated spectrograms by adding: `refined_mel = decoder_mel + postnet(decoder_mel)`
  - Helps reduce buzzy/metallic artifacts in audio output

- **Enhanced Audio Evaluation Metrics** — `evaluate_tts_output.py` now includes spectral analysis
  - New `spectral_flatness` metric using librosa's `spectral_flatness()` function
  - Evaluates whether audio spectrum resembles white noise (flat) vs speech (structured)
  - Verdicts added: noise-like (>0.6), somewhat noisy (>0.4), speech-like (<0.4)
  - Batch summary now includes spectral flatness alongside existing metrics
  - More comprehensive audio quality understanding

- **Comprehensive Duration Diagnostics** — Detailed TensorBoard logging for duration prediction
  - New metrics: `train/pred_duration_sum_mean`, `train/target_duration_sum_mean`, `train/relative_duration_error_mean`
  - Duration extremes tracked: `train/duration_max`, `train/duration_min`
  - Mel-spectrogram range monitoring: `train/predicted_mel_max`, `train/predicted_mel_min`
  - Identical validation metrics for cross-phase comparison (`val/duration_max`, `val/duration_min`, etc.)
  - Helps identify when duration predictor diverges or produces extreme values

### Improved

- **Duration Loss Supervision** — Switched to relative error with robust outlier handling
  - Old method: Simple L1 loss between predicted and target duration sums
  - New method: Relative absolute error `|pred - target| / target` clamped at 5.0
  - Clamping prevents single exploding batches from dominating loss
  - Better stability during training, especially with smaller batch sizes

- **Duration Predictor Stability** — Hard clamping prevents extreme values
  - Duration output range: now clamped to `[1.0, 20.0]` frames per phoneme
  - Previously only had soft constraint via softplus
  - Prevents duration explosion that caused NaN failures in length regulation
  - Additional validation check: skip batches if `duration.max() >= 20.0`

- **Mixed Precision Handling** — Fixed context manager for autocast
  - Old: Awkward context with `if self.scaler else torch.enable_grad()`
  - New: Explicit `torch.autocast(device_type='cuda', enabled=False)` when not using AMP
  - Cleaner, more explicit control over mixed precision regions
  - Prevents edge cases with gradient computation

- **Gradient NaN Protection** — Validation loop now checks gradient health
  - Skips batches if gradients contain NaN/Inf values
  - Prevents gradient stack corruption mid-training
  - Additional safeguard beyond loss checking

- **Inference Function Documentation** — Comprehensive comments on latent code generation
  - Detailed step-by-step explanation of inference pipeline
  - Clarifies role of duration scaling, noise scale, text prior, and stochasticity
  - Better for future maintenance and understanding

### Configuration Updates

- **Learning Rate Reduction** — More conservative training
  - `learning_rate: 0.0002 → 0.0001`
  - Reduces risk of unstable updates to duration and decoder
  - Slower convergence but more stable training trajectory

- **Duration Loss Weight Options** — Multiple recommended values provided
  - Current: `loss_weight_duration: 0.1` (balanced)
  - Alternatives documented: `0.05` (very conservative) and higher for stability
  - Prevents duration component from overwhelming other losses

- **AMP Disabled by Default** — Mixed precision now off for stability
  - `use_amp: false` (changed from `true`)
  - Reason: NaN instability observed with smaller batch sizes and float16
  - Can re-enable if using larger batches or with more stable checkpoint

- **Reduced Max Sequence Length** — Smaller sequences for stability
  - `max_seq_length: 500 → 300` frames (≈3 seconds)
  - Reduces GPU memory pressure
  - Decreases likelihood of extreme duration predictions

### Fixed

- **Length Regulation Edge Cases** — Clamped repeat counts to prevent runaway expansion
  - Changed: `repeat_count = max(1, int(duration[i, j].item()))`
  - To: `repeat_count = min(20, max(1, int(duration[i, j].item())))`
  - Prevents single phoneme from expanding to 1000+ frames due to prediction error

- **NaN Batch Skipping Logic** — Removed overly aggressive duration loss threshold
  - Old check: `if loss_dict["duration_loss"] > 300: skip_batch`
  - Issue: Relative error supervision makes absolute thresholds unreliable
  - New approach: Rely on clamping + extremes check (`duration.max() >= 20.0`)
  - More robust across different training regimes

### Internal

- **Training Stability** — Better protection against outlier batches and gradient explosion
- **Diagnostics** — Much richer TensorBoard telemetry for debugging duration issues
- **Code Quality** — Clearer separation of concerns in duration vs mel prediction monitoring

### Known Issues

- **Audio Quality Still Buzzy** — High zero-crossing rate (~0.49) indicates metallic artifacts
  - Root causes: Simplified VITS architecture, basic Griffin-Lim vocoder
  - Not a bug; architectural limitation at current training stage
  - Improves with longer training and better vocoder (HiFi-GAN upgrade path exists)

- **Duration Prediction Learning Slow** — Model struggles with phoneme timing alignment
  - Current approach matches total length, not frame-by-frame alignment
  - Future improvement: Implement Monotonic Alignment Search (MAS)

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

_Last updated: 28.04.2026_  
_Changelog maintained by: someone who cares, apparently_  
_GPU status: Occupied_  
_Linda Johnson memorial fund: ongoing_  
_Normalization formula correctness: still correct_  
_Architectural decisions correctness: finally_
