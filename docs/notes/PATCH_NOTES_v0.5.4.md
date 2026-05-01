# HexTTs v0.5.4 Patch Notes

**Release Date:** 2026-05-01  
**Headline:** Real phoneme-level duration targets are now live. The duration predictor can finally learn.

---

## TL;DR for Users

### What Changed

- Phoneme-level duration alignment complete (12,884 / 12,884 LJSpeech utterances)
- All 5 config files now reference `duration_dir: ./data/ljspeech_prepared/durations`
- Training uses real duration targets by default
- No model architecture changes (all v0.5.3 checkpoints still work)

### What to Do

```bash
python scripts/train.py --config configs/base.yaml --device cuda
```

Real durations will auto-load. Monitor TensorBoard for `train/using_real_duration_targets` = 1.0 (confirmed).

---

## What Got Fixed

### The Big One: Import Path in Alignment Script

The alignment script (`scripts/align_torchaudio.py`) was producing **0% real alignment** despite 98% success in diagnostics.  
**Root cause:** `from hextts.data.preprocessing import phonemes_per_word` was failing inside an exception handler, which silently swallowed the error.  
**Fix:** Added `sys.path.insert(0, str(Path(__file__).parent.parent))` at line 47.  
**Result:** Real alignment coverage: 0% → **100%** (all 12,884 files).

### Config Files

Added `duration_dir: ./data/ljspeech_prepared/durations` to:

- `configs/base.yaml`
- `configs/continue_auto.yaml`
- `configs/continue3.yaml`
- `configs/debug.yaml`
- `configs/sanity.yaml`

All training profiles now use real targets automatically.

---

## New Features

### Windows-Native Alignment (No MFA Required)

```bash
python scripts/align_torchaudio.py ^
    --audio_dir data/LJSpeech-1.1/wavs ^
    --metadata_csv data/LJSpeech-1.1/metadata.csv ^
    --prepared_dir data/ljspeech_prepared ^
    --output_dir data/ljspeech_prepared/durations ^
    --device cuda --batch_size 8 --num_workers 4
```

**Performance:**

- GPU: 6.22 files/sec (RTX 3050 Ti, batch_size=8, num_workers=4, AMP enabled)
- Coverage: 100% real alignment
- Fallback strategy: proportional frame redistribution for sparse alignments (robust)

**Optional flags:**

- `--device cpu` — CPU mode (slower, no VRAM cost)
- `--disable_amp` — Turn off automatic mixed precision if needed
- `--batch_size N` — Adjust batch size for available VRAM

---

## Expected Training Improvements

| Metric                | Expected Behavior                                   |
| --------------------- | --------------------------------------------------- |
| `train/duration_loss` | Should converge faster (real targets vs. proxy)     |
| Inference predictions | More realistic (trained on ground truth)            |
| Duration stability    | TensorBoard `train/duration_max` should stay < 20.0 |
| Skip ratio            | Should remain < 5% (real targets are well-formed)   |

---

## Backward Compatibility

**Fully backward-compatible with v0.5.3**

- No model architecture changes
- No checkpoint format changes
- All v0.5.3 checkpoints load and resume unchanged
- If `duration_dir` is missing/empty, trainer falls back to pseudo-uniform durations

---

## Known Limitations

- Alignment accuracy depends on WAV2VEC2_ASR_BASE_960H model (pretrained on 960h LibriSpeech)
- May not work perfectly on very noisy, heavily accented, or out-of-domain speech
- For maximum quality: review a few generated duration files in `./data/ljspeech_prepared/durations/` manually
- Windows paths must use `^` for multiline commands in PowerShell

---

## Files Modified

```
scripts/align_torchaudio.py     ← Fixed sys.path import issue
configs/base.yaml               ← Added duration_dir
configs/continue_auto.yaml      ← Added duration_dir
configs/continue3.yaml          ← Added duration_dir
configs/debug.yaml              ← Added duration_dir
configs/sanity.yaml             ← Added duration_dir
CHANGELOG.md                    ← New entry (v0.5.4)
```

---

## Next Steps

1. **Optional:** Review a few alignment outputs:

   ```bash
   # Check shape and contents of first duration file
   python -c "import numpy as np; d = np.load('./data/ljspeech_prepared/durations/LJ001-0001.npy'); print(f'Shape: {d.shape}, Min: {d.min()}, Max: {d.max()}, Sum: {d.sum()}')"
   ```

2. **Start training:**

   ```bash
   python scripts/train.py --config configs/base.yaml --device cuda
   ```

3. **Monitor in TensorBoard:**
   ```bash
   tensorboard --logdir=./logs
   ```
   Look for `train/using_real_duration_targets = 1.0` to confirm real targets are loaded.

---

## Debugging Alignment Issues (If Needed)

If alignment produces mostly fallback results:

1. **Check sys.path fix is in place:**

   ```bash
   python -c "from hextts.data.preprocessing import text_to_phonemes; print('Import OK')"
   ```

2. **Test on a single file:**

   ```bash
   python -c "
   from scripts.align_torchaudio import _align_with_log_probs
   # (run diagnostics on sample files)
   "
   ```

3. **Verify audio format:**
   - Expected: 22050 Hz mono `.wav` files
   - Use `scripts/validate_dataset.py` first if unsure

---

## The Funny Part

Three hours of debugging a script that was generating 0% real alignments despite 98% success in isolation.  
Turned out the word "import" was being swallowed by an exception handler. No error message. No warning.
Just silent 0%. Python.

---

## Credits

Alignment pipeline inspired by forced alignment workflows in speech synthesis research.  
Duration predictor now has real targets to dream about instead of whatever it was learning before.
