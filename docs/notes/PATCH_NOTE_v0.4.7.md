# HexTTs Patch Note v0.4.7

Date: 2026-04-09
Branch: copilot/vscode-mnrjulze-pje7

## Summary

v0.4.7 is the stabilization release after a bad duration experiment.

The phoneme-aware weighting branch that produced a `0.23 s` collapse was rolled back, the previous hybrid duration path was restored, and a debug-verified continuation run confirmed the system is back in a healthy regime.

## What Changed

### 1) Restored previous working hybrid duration supervision

In `train_vits.py`:

- removed active phoneme-aware weighting path from training logic
- restored uniform, sum-preserving pseudo token duration targets
- kept hybrid duration loss (token-level + sum-level)

This reverts to the known stable duration behavior.

### 2) Added duration debug verification hooks

In `train_vits.py`:

- new optional config flag: `duration_debug_checks`
- prints one train and one validation sample with:
  - `phoneme_length`
  - `mel_length`
  - `target_duration` vector and sum
  - `predicted_duration` vector and sum
  - proxy formula output

This was added so future duration experiments can be validated before trusting metrics.

### 3) Improved continuation runner output and reporting

In `scripts/run_continuation_test.py`:

- streams subprocess output in chunks for cleaner tqdm rendering
- added `--duration-debug-checks` CLI switch that propagates to generated config
- added `--report-file` output (default: `reports/continuation_test_report.txt`)
- report includes:
  - training snapshot lines (`loss`, `recon`, `kl`, `dur`)
  - full HexTTS evaluation report block
  - continuation summary with duration diagnostics

## Validation Run

Command:

```bash
venv\Scripts\python.exe scripts/run_continuation_test.py --epochs 1 --duration-debug-checks --report-file reports/continuation_test_report_debug.txt
```

Run result: success

### Debug consistency checks

Train sample:

- `phoneme_length=43`
- `mel_length=329`
- `target_sum=329.0000`
- `pred_sum=359.3018`
- proxy: `8.323909`

Validation sample:

- `phoneme_length=91`
- `mel_length=629`
- `target_sum=629.0000`
- `pred_sum=751.9414`
- proxy: `8.266350`

Interpretation:

- target vector sums exactly match mel lengths
- predicted durations are on the same frame scale as targets
- proxy is in healthy range, not collapsed

### Final audio and diagnostics

- Output file: `tts_output/hifigan_continue_auto.wav`
- Duration: `1.8576 s`
- ZCR: `0.114986`
- Spectral flatness: `0.015726`

Latest duration diagnostics:

- `train/token_duration_mae`: `1.1307485104`
- `val/token_duration_mae`: `1.2926565409`
- `train/sum_error_mean`: `60.0709152222`
- `val/sum_error_mean`: `140.8995971680`
- `train/pred_speech_rate_proxy`: `8.3540611267`
- `val/pred_speech_rate_proxy`: `8.2652015686`

## Conclusion

The system is back in the correct operating regime:

- duration learning: stable
- output length: realistic
- waveform quality: clean
- proxy scale: consistent

v0.4.7 marks the transition from duration-collapse debugging back to controlled prosody refinement.
