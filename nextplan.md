# HexTTs — Next Plan

_Last updated: 2026-05-01 — added torchaudio aligner (scripts/align_torchaudio.py)_

This file tracks what is done, what is pending, and what comes after.
Update it after each work session so the next session starts with full context.

---

## Status at end of 2026-04-28

- **Current version:** v0.5.3
- **Training state:** v0.5.2 run was stopped at step ~16,000. v0.5.3 fresh training run to be started.
- **Test suite:** 40/40 passing
- **Checkpoint state:** All pre-v0.5.3 checkpoints invalid (architecture changed)

---

## What was completed today (v0.5.3)

### Bug fixes

- [x] Griffin-Lim was called on a mel spectrogram — `librosa.griffinlim` expects linear STFT (513 bins), not mel (80 bins). Fixed: `dB → power → mel_to_stft → griffinlim` pipeline.
- [x] Sample generation denormalization used pre-v0.5.2 broken formula (`ref_level_db=20`). Fixed to use `min_level_db=-100`.

### Architecture improvements (`hextts/models/vits_impl.py`)

- [x] Decoder output: `tanh` → `sigmoid` — targets are [0,1], tanh output [-1,1] was wasting half the output range
- [x] TransformerBlock: post-norm → pre-norm — better gradient flow through encoder and decoder stacks
- [x] TransformerBlock FFN: `ReLU` → `GELU` — smoother activations, standard in modern transformers
- [x] PosteriorEncoder: 1×1 convs → 3×1 convs + GroupNorm — added temporal context (±1 frame/layer) and normalisation
- [x] DurationPredictor: fixed 3rd context conv (kernel=5) — extends receptive field to ±4 phonemes
- [x] Prior projection: linear → 2-layer MLP with GELU — more expressive text-to-latent mapping

### Loss improvement (`hextts/training/trainer.py`)

- [x] MultiScaleMelLoss: added scale=8 — enforces sentence-level prosody structure (~93 ms window)

### Config tuning (`configs/base.yaml`)

- [x] `use_amp: true` — ~1.5–2× faster per step on RTX 3050 Ti
- [x] `batch_size: 6` — AMP freed VRAM headroom; was 4
- [x] `checkpoint_interval: 1000` — reduced checkpoint write overhead; was 500
- [x] `loss_weight_duration: 0.15` — was 0.1
- [x] `loss_weight_stft: 0.15` — was 0.1
- [x] `duration_predictor_dropout: 0.3` — was 0.5 (0.5 was preventing reliable duration learning)
- [x] `adam_beta1: 0.8`, `adam_beta2: 0.99` — VITS paper defaults for TTS; was 0.9 / 0.999

### MFA integration (code complete, needs MFA to run)

- [x] `scripts/extract_mfa_durations.py` — parses MFA TextGrid files → per-file `.npy` duration arrays
- [x] `hextts/data/raw_dataset.py` — loads real duration targets from `duration_dir` when present
- [x] `hextts/data/cached_dataset.py` — same
- [x] `hextts/training/trainer.py` — uses real targets in `compute_loss` when available; pseudo-uniform fallback otherwise
- [x] `configs/base.yaml` — `duration_dir: ""` placeholder ready to fill in
- [x] TensorBoard: `train/using_real_duration_targets` scalar — confirms real vs pseudo targets per step

---

## Immediate next actions (do before next training run)

### 1. Start v0.5.3 training

```bash
venv\Scripts\activate
python scripts/train.py --config configs/base.yaml --device cuda
```

Expected speed with AMP + batch_size=6: ~4–5 it/s, ~10–11 min/epoch, ~17–18 hours total.

### 2. Monitor first epoch closely

Check TensorBoard after first 500 steps:

- `train/recon_loss` should be decreasing (not stuck near constant value)
- `train/skipped_batches` should be low (< 5% per epoch)
- `train/duration_max` should stay below 20.0
- No NaN/Inf in loss

If `train/recon_loss` is erratic in the first 200 steps: warmup is working, this is normal.
If `train/skipped_batches` > 20% in epoch 1: check `max_duration_value` and duration predictor stability.

---

## Forced alignment (activate real duration targets)

**Status:** Code complete. Use Option A on Windows — no conda required.

### Why this matters

Pseudo-uniform targets distribute mel frames equally across phonemes.
Real speech has highly unequal durations: vowels last 3–5× longer than stops, unstressed syllables are shorter.
With real targets the duration predictor can learn actual phoneme timing — the single largest remaining quality ceiling.

---

### Option A — torchaudio aligner (Windows-native, recommended)

Uses `WAV2VEC2_ASR_BASE_960H` + `torchaudio.functional.forced_align`. No Kaldi, no conda required.

```bash
pip install g2p_en   # skip if already installed

python scripts/align_torchaudio.py ^
    --audio_dir    data/LJSpeech-1.1/wavs ^
    --metadata_csv data/LJSpeech-1.1/metadata.csv ^
    --prepared_dir data/ljspeech_prepared ^
    --output_dir   data/ljspeech_prepared/durations ^
    --device cpu
```

Expected: ~1–2 hours on CPU for 13,100 files. Add `--device cuda` for ~3–4× speedup.
Add `--limit 100` for a quick dry-run first.
Coverage goal: ≥ 80% real alignment (rest saves uniform-fallback .npy, same quality as pseudo-targets).

---

### Option B — MFA (Linux/conda only)

MFA requires Kaldi C++ binaries — available only via conda, not pip.
`pip install montreal-forced-aligner` fails on Windows venv (`_kalpy` missing).

```bash
# conda environment required
mfa model download acoustic english_us_arpa
mfa model download dictionary english_us_arpa
mfa align data/LJSpeech-1.1/wavs english_us_arpa english_us_arpa data/mfa_alignments --num_jobs 4

python scripts/extract_mfa_durations.py ^
    --textgrid_dir data/mfa_alignments ^
    --metadata_dir data/ljspeech_prepared ^
    --output_dir   data/ljspeech_prepared/durations ^
    --hop_length   256 ^
    --sample_rate  22050
```

---

### After either option

Update `configs/base.yaml`:

```yaml
duration_dir: ./data/ljspeech_prepared/durations
```

Restart training. Confirm `train/using_real_duration_targets = 1` in TensorBoard within the first log interval.

---

## Upcoming model improvements (v0.5.4 targets)

### A. Evaluate v0.5.3 audio quality first

At epochs 10, 20, 30 — listen to `samples/epoch_XXX_sample_Y.wav` and check:

- Is the rhythm recognisable speech?
- Is there phoneme structure (not just noise)?
- Is the energy envelope natural?

Use `scripts/evaluate_tts_output.py` for objective metrics:

```bash
python scripts/evaluate_tts_output.py --audio samples/epoch_030_sample_1.wav --sample_rate 22050
```

Targets at epoch 30: ZCR < 0.3, spectral flatness < 0.2.

### B. HiFi-GAN vocoder (quality ceiling above Griffin-Lim)

Griffin-Lim is a reconstruction algorithm — it cannot generate waveform quality above a hard limit.
HiFi-GAN is a neural vocoder trained on speech and produces dramatically better audio.

Two options:

1. **Use pre-trained HiFi-GAN** (fastest) — download official checkpoint and use it for inference
   - Inference: `scripts/infer.py --vocoder_checkpoint <path> --vocoder_config <path>`
   - No training required, but the vocoder was not trained on HexTTs mel output (may mismatch)

2. **Fine-tune HiFi-GAN on HexTTs mels** (best quality)
   - Requires HiFi-GAN training code and ~100 GPU-hours
   - Input: HexTTs mel spectrograms, Target: LJSpeech waveforms
   - Output: a vocoder tuned to this exact mel normalization and filterbank

Current status: pre-trained HiFi-GAN path already wired in `hextts/vocoder/hifigan.py` and `inference/pipeline.py`.

### C. Increase model capacity (if quality stalls after epoch 50)

Current: 22.12M parameters with hidden sizes 384/512 and 4 encoder/decoder layers.
If the model quality has not saturated but loss is still decreasing, these changes help:

| Config key            | Current | Proposed | VRAM impact |
| --------------------- | ------- | -------- | ----------- |
| `encoder_num_layers`  | 4       | 6        | +~15%       |
| `decoder_num_layers`  | 4       | 6        | +~15%       |
| `encoder_hidden_size` | 384     | 512      | +~30%       |
| `latent_dim`          | 192     | 256      | +~5%        |

**Do not change these mid-training** — all are checkpoint-breaking architecture changes.

### D. Monotonic Alignment Search (MAS) — long-term

MFA gives real durations for LJSpeech specifically, but MAS would let the model learn its own alignment without an external aligner, which generalises to new speakers and datasets.
MAS is the training-time alignment mechanism used in the original VITS paper.

This is a significant architectural addition (requires a separate alignment network and training procedure). Not a priority until the current architecture is producing intelligible speech.

---

## Known issues to watch during training

| Issue                   | What to check                                  | Action if triggered                                                                       |
| ----------------------- | ---------------------------------------------- | ----------------------------------------------------------------------------------------- |
| Duration explosion      | `train/duration_max` > 18                      | Reduce `loss_weight_duration` to 0.1 or increase `duration_predictor_dropout` back to 0.4 |
| KL posterior collapse   | `train/kl_loss` near 0 after step 10k          | Extend `kl_warmup_steps` to 20000                                                         |
| Length mismatch growing | `train/length_mismatch_frames` > 10 on average | Duration predictor diverging — check duration metrics                                     |
| Batch skip rate > 20%   | `train/epoch_skip_ratio`                       | Reduce `batch_size` back to 4, check VRAM                                                 |
| AMP NaN                 | Training aborts with NaN and AMP enabled       | Set `use_amp: false`, restart from last checkpoint                                        |

---

## Inference tuning (after training)

When generating audio, these knobs control output character:

- `inference_duration_scale: 4.0` — controls speech speed. Higher = slower. Tune down if speech is too slow.
- `inference_noise_scale: 0.3` — controls expressiveness. Higher = more varied but less stable.

After MFA targets are active, `inference_duration_scale` can likely be reduced to 2.0–3.0 (pseudo-targets were over-uniform, requiring more stretching to sound natural).

---

## Reference commands

```bash
# Start training
python scripts/train.py --config configs/base.yaml --device cuda

# Resume training
python scripts/train.py --config configs/base.yaml --checkpoint checkpoints/checkpoint_step_XXXXXX.pt --device cuda

# Monitor
tensorboard --logdir=./logs

# Infer from checkpoint
python scripts/infer.py --checkpoint checkpoints/best_model.pt --config configs/base.yaml --text "Hello, this is a test." --output tts_output/test.wav --device cpu

# Evaluate audio
python scripts/evaluate_tts_output.py --audio tts_output/test.wav --sample_rate 22050

# Run tests
python -m pytest -q
```
