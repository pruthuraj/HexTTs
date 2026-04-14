# HexTTs v0.4.6 Patch Note

## "One-Command Continuation Testing, Because We Have Better Things To Do Than Manually Typing 17 Flags"

**Release Date:** April 9, 2026  
**Tag:** v0.4.6  
**Status:** Stable  
**Urgency:** Medium-High (recommended if you are running duration/alignment experiments)

---

## Overview

v0.4.6 is a **workflow automation + experiment consistency release**.

In short:

- We already proved HiFi-GAN is cleaner than Griffin-Lim.
- We already patched duration supervision.
- The next bottleneck became "human patience" and "command-line typo RNG".

So this release introduces a single automation script that runs the **full continuation test pipeline** end-to-end.

Translation: less copy-paste, fewer mistakes, more actual science.

---

## Why This Exists

Before v0.4.6, continuation testing looked like this:

1. manually create/adjust continuation config
2. manually resume training from checkpoint
3. manually parse TensorBoard diagnostics
4. manually run HiFi-GAN inference
5. manually run evaluation script
6. manually summarize everything without forgetting a metric

That flow works. It also invites chaos.

v0.4.6 turns that into one command.

---

## Major Additions

### 1) New Automation Script

**File:** `scripts/run_continuation_test.py`

This script runs the full continuation experiment pipeline:

1. Builds a continuation config from your base config
2. Resumes training from a checkpoint
3. Extracts latest duration diagnostics from TensorBoard event files
4. Runs HiFi-GAN inference on the fixed sentence
5. Evaluates output (`duration`, `ZCR`, `spectral flatness`, verdict)
6. Prints a compact summary block

Yes, all of that in one run.

#### Default behavior

- base config: `vits_config.yaml`
- output config: `vits_config.continue_auto.yaml`
- resume checkpoint: `checkpoints_sanity/checkpoint_step_003000.pt`
- epochs: `3`
- alpha/beta: `1.0 / 0.2`
- vocoder: `hifigan/generator_v1` + `hifigan/config_v1.json`
- output wav: `tts_output/hifigan_continue_auto.wav`

#### Example

```bash
venv\Scripts\python.exe scripts/run_continuation_test.py --epochs 3
```

---

### 2) main_flow Integration

**File:** `scripts/main_flow.py`

Added a new subcommand:

- `continuation-test`

So now your "do the whole experiment" command is:

```bash
python scripts/main_flow.py continuation-test --epochs 3
```

In other words, the old workflow was a wall of arguments. The new workflow is a sentence.

---

## Documentation Updates

### README

- version markers updated to v0.4.6
- `continuation-test` added to simplified workflow commands
- new dedicated "Continuation Test (3-Epoch Automation)" section
- examples added for direct script run and main flow subcommand

### CHANGELOG

- new v0.4.6 entry added
- captures script addition + wrapper integration + README updates

---

## Validation Notes

`scripts/main_flow.py continuation-test -h` parses successfully.

`scripts/run_continuation_test.py -h` parses successfully.

No static errors reported for:

- `scripts/run_continuation_test.py`
- `scripts/main_flow.py`
- `README.md`
- `CHANGELOG.md`

---

## Impact

### What gets better

- Fewer manual mistakes in continuation experiments
- Easier apples-to-apples comparisons across short runs
- Standardized metric extraction for:
  - `train/val token_duration_mae`
  - `train/val sum_error_mean`
  - `train/val pred_speech_rate_proxy`
  - output duration, `ZCR`, `spectral flatness`

### What does not change

- Core model architecture
- Vocoder weights
- Dataset contents
- Fundamental TTS laws of physics

---

## Known Limitations

- The script still depends on valid checkpoints/log folders and installed Python deps.
- If your GPU decides to become decorative art mid-run, this script cannot negotiate with it.
- It automates the experiment, not your life choices.

---

## Recommendation

Use `continuation-test` as the default way to run 3-epoch checkpoint continuations while tuning duration behavior.

If this continues to show stable speech-rate proxy + improving sum error while keeping low-noise HiFi-GAN output, move to longer controlled runs with the same metric set.

---

## Final Words

v0.4.6 does not make your model magically perfect.

It does make your workflow less cursed.

And honestly, at this stage, that is a legitimate win.
