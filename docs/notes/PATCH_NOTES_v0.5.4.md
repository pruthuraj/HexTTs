# HexTTs v0.5.4 Patch Notes

**Release Date:** 2026-05-01  
**Headline:** Real phoneme-level duration targets are live, and the docs finally know what the project is doing.

---

## TL;DR for Users

### What Changed

- Phoneme-level duration alignment complete: 12,884 / 12,884 LJSpeech utterances
- All training profiles now reference `duration_dir: ./data/ljspeech_prepared/durations`
- Training uses real duration targets by default
- No model architecture changes: all v0.5.3 checkpoints remain compatible
- Documentation overhaul shipped under `docs/` with index-first navigation
- Mermaid system and architecture diagrams added to core docs
- Portfolio summary document added: `docs/HexTTs_AI_Project_Document.docx`

### What to Do

```bash
python scripts/train.py --config configs/base.yaml --device cuda
```

Real durations auto-load. Monitor TensorBoard for:

```text
train/using_real_duration_targets = 1.0
```

For docs, start here:

```text
docs/index.md
```

---

## What Got Fixed

### The Big One: Import Path in Alignment Script

The alignment script (`scripts/align_torchaudio.py`) produced 0% real alignment while diagnostics reported ~98% success in isolation.

- Root cause: `from hextts.data.preprocessing import phonemes_per_word` failed inside an exception path, and the error was swallowed.
- Fix: add repo root to `sys.path` at module load.
- Result: real alignment coverage restored to 100% (12,884 files).

### Config Coverage

Added `duration_dir: ./data/ljspeech_prepared/durations` to:

- `configs/base.yaml`
- `configs/continue_auto.yaml`
- `configs/continue3.yaml`
- `configs/debug.yaml`
- `configs/sanity.yaml`

All training modes now point at real targets by default.

---

## New Features

### Windows-Native Forced Alignment (No MFA Required)

```bash
python scripts/align_torchaudio.py ^
    --audio_dir data/LJSpeech-1.1/wavs ^
    --metadata_csv data/LJSpeech-1.1/metadata.csv ^
    --prepared_dir data/ljspeech_prepared ^
    --output_dir data/ljspeech_prepared/durations ^
    --device cuda --batch_size 8 --num_workers 4
```

Performance snapshot:

- GPU throughput: 6.22 files/sec (RTX 3050 Ti, batch_size=8, num_workers=4, AMP)
- Coverage: 100% real alignment
- Fallback strategy: proportional redistribution for sparse edge cases

Optional flags:

- `--device cpu`
- `--disable_amp`
- `--batch_size N`

### Documentation Architecture Refresh

New maintained docs:

- `docs/index.md`
- `docs/project-rationale.md`
- `docs/system-flow.md`
- `docs/data-pipeline.md`
- `docs/model-training.md`
- `docs/inference-evaluation.md`
- `docs/operations-troubleshooting.md`

Updated summary pages:

- `docs/architecture.md`
- `docs/training.md`
- `docs/inference.md`
- `docs/troubleshooting.md`

New portfolio deliverable:

- `docs/HexTTs_AI_Project_Document.docx`

Mermaid diagrams added in:

- `docs/system-flow.md`
- `docs/architecture.md`
- `docs/model-training.md`

---

## Expected Training Improvements

| Metric | Expected Behavior |
| --- | --- |
| `train/duration_loss` | Faster and cleaner convergence with real targets |
| Inference timing | More realistic duration prediction behavior |
| Duration stability | `train/duration_max` should stay under guardrails |
| Skip ratio | Should remain low with valid duration arrays |

---

## Backward Compatibility

Fully backward-compatible with v0.5.3:

- No model architecture changes
- No checkpoint format changes
- Existing v0.5.3 checkpoints can load and resume
- If `duration_dir` is missing, trainer falls back to pseudo-uniform durations

---

## Files Modified

```text
scripts/align_torchaudio.py
configs/base.yaml
configs/continue_auto.yaml
configs/continue3.yaml
configs/debug.yaml
configs/sanity.yaml
docs/index.md
docs/project-rationale.md
docs/system-flow.md
docs/data-pipeline.md
docs/model-training.md
docs/inference-evaluation.md
docs/operations-troubleshooting.md
docs/architecture.md
docs/training.md
docs/inference.md
docs/troubleshooting.md
docs/HexTTs_AI_Project_Document.docx
CHANGELOG.md
README.md
readme.long.md
```

---

## Next Steps

1. Start or resume training with real durations:
```bash
python scripts/train.py --config configs/base.yaml --device cuda
```
2. Monitor duration diagnostics in TensorBoard:
```bash
tensorboard --logdir=./logs
```
3. Use the new docs index instead of searching old note files:
```text
docs/index.md
```

---

## The Funny Part

Three hours of debugging told us the alignment math was "fine" while the import path was silently on fire.  
Then we wrote a full documentation system about it so future-us can panic with better navigation.

---

## Credits

Forced-alignment workflow inspired by modern speech alignment practice, with pragmatically aggressive debugging after the silent-import incident.
