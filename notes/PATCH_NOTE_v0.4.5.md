# HexTTs v0.4.5 Patch Note

## "We Cleaned the Root Folder and Gave You a Proper Control Panel"

**Release Date:** April 9, 2026  
**Tag:** v0.4.5  
**Status:** Stable  
**Urgency:** Medium (recommended if your command history currently looks like a crime scene)

---

## Overview

v0.4.5 is a **workflow simplification + repo hygiene release**.

No magical architecture leap here. This patch is about making day-to-day operations less cursed:

- fewer giant commands
- fewer root-level utility files
- more consistent script paths
- cleaner docs

In plain English: we moved from "manual ritual" to "repeatable process."

---

## Why This Release Happened

Before v0.4.5, the main flow required long command chains with many flags and too many opportunities for typo-based sadness.

Also, several helper scripts sat in the project root like random tools on a kitchen floor.

v0.4.5 fixes both problems:

1. introduces a single wrapper for common operations
2. reorganizes utility scripts under `scripts/`
3. updates documentation so the command paths actually match reality

---

## Major Features

### 1) Simplified Main Flow Wrapper

**File:** `scripts/main_flow.py`

A central command runner for common operations.

Supported commands in v0.4.5:

- `train`
- `infer`
- `eval`
- `audit`
- `compare`

#### Why this matters

You can now run common TTS tasks as short, predictable commands, instead of retyping a full command novel every time.

#### Example usage

```bash
# Train
python scripts/main_flow.py train --device cuda

# Inference
python scripts/main_flow.py infer --text "hello world" --hifigan --output tts_output/hello_hifigan.wav

# Evaluate
python scripts/main_flow.py eval --audio tts_output/hello_hifigan.wav

# Compare Griffin-Lim vs HiFi-GAN
python scripts/main_flow.py compare --text "we are at present concerned"
```

---

### 2) Root Cleanup: Utility Scripts Moved to `scripts/`

To reduce root-folder chaos, utility files were moved from root to `scripts/`.

#### File moves

- `audit_dataset.py` → `scripts/audit_dataset.py`
- `evaluate_tts_output.py` → `scripts/evaluate_tts_output.py`
- `view_spectrogram.py` → `scripts/view_spectrogram.py`
- `test_setup.py` → `scripts/test_setup.py`

#### Why this matters

- root folder now focuses on core pipeline files
- utility tooling is grouped in one place
- easier onboarding for humans who are not telepathic

---

### 3) Integrated Dataset Audit Flow

`main_flow.py` got an `audit` command that wraps dataset quality filtering.

#### Supported options

- `--dry-run`
- `--min-duration`
- `--max-duration`
- `--min-rms-db`
- `--max-silence`
- `--no-clip-check`

#### Example

```bash
python scripts/main_flow.py audit --dry-run
```

This removes another manual step where users had to remember separate script names and argument styles.

---

## Documentation Updates

### README

Updated to reflect v0.4.5 workflow changes:

- added/expanded simplified main-flow section
- updated command examples to moved script paths
- documented new `audit` usage path
- aligned footer/version metadata

### CHANGELOG

Added a v0.4.5 entry with:

- main flow wrapper details
- root cleanup mapping
- documentation update summary

---

## Validation Snapshot (v0.4.5)

Sanity checks performed after changes:

- `python scripts/main_flow.py -h` works
- `python scripts/main_flow.py infer -h` works
- `python scripts/main_flow.py compare -h` works
- `python scripts/main_flow.py audit -h` works
- `python scripts/main_flow.py eval --audio tts_output/hifigan_test.wav --sample_rate 22050` runs successfully

No static errors reported in:

- `scripts/main_flow.py`
- `scripts/audit_dataset.py`
- `README.md`
- `CHANGELOG.md`

---

## Impact

### What got better

- Less command repetition
- Lower chance of path mistakes
- Cleaner project structure
- Better docs-to-code path consistency

### What did not change

- Core model architecture
- Vocoder internals
- Dataset content itself
- Training objective design

---

## Known Limitations

- This patch improves workflow ergonomics, not model intelligence.
- If your model still hallucinates robot poetry, this patch did not cause that.
- You can still write broken commands manually if you insist.

---

## Recommendation

Treat `scripts/main_flow.py` as the default control surface for daily work.

Use direct script calls only when you need edge-case control or enjoy unnecessary typing for cardio.

---

## Final Words

v0.4.5 does not make your TTS model smarter.

It does make your workflow less annoying.

And honestly, that is how real progress often looks.
