# How To Run All HexTTs Steps

This file is the quick path for the full HexTTs workflow.
It shows the commands and the output you should expect when things are working.

## 1. Train

```bash
venv\Scripts\python.exe train_vits.py --config vits_config.yaml --device cuda
```

Expected output:

- model initialization messages
- dataloader counts
- tqdm progress bar for each epoch
- TensorBoard logging lines
- checkpoint saves such as `checkpoints/best_model.pt`
- validation loss at the end of the epoch

Healthy signs:

- training loss moves down over time
- validation loss stays finite
- no NaN or Inf batch skips dominate the run

## 2. Run Inference

Griffin-Lim fallback:

```bash
venv\Scripts\python.exe inference_vits.py --checkpoint checkpoints/best_model.pt --config vits_config.yaml --text "we are at present concerned" --output tts_output/output.wav --device cpu
```

HiFi-GAN preferred:

```bash
venv\Scripts\python.exe inference_vits.py --checkpoint checkpoints/best_model.pt --config vits_config.yaml --vocoder_checkpoint hifigan/generator_v1 --vocoder_config hifigan/config_v1.json --text "we are at present concerned" --output tts_output/output.wav --device cpu
```

Expected output:

- model load confirmation
- phoneme list for the text
- mel-spectrogram shape
- audio shape and duration
- `Audio saved to ...`

Healthy signs:

- HiFi-GAN loads successfully
- output duration is not absurdly short
- audio file is written to `tts_output/`

## 3. Evaluate Audio

```bash
venv\Scripts\python.exe scripts/evaluate_tts_output.py --audio tts_output/output.wav --sample_rate 22050
```

Expected output:

- `HexTTS Output Evaluation Report`
- file path
- duration
- peak amplitude
- RMS energy
- silence ratio
- zero crossing rate
- spectral flatness
- verdict bullets

Healthy signs:

- duration is roughly in the 1.5 to 3 second range for the fixed continuation sentence
- zero crossing rate stays low
- spectral flatness stays low
- silence ratio is small but not zero

## 4. Run The Full Continuation Test

```bash
venv\Scripts\python.exe scripts/run_continuation_test.py --epochs 1 --duration-debug-checks --report-file reports/continuation_test_report_debug.txt
```

Expected output:

- resumed training command
- tqdm progress during training
- duration debug prints for one train sample and one val sample
- HiFi-GAN inference output
- evaluation report output
- `CONTINUATION TEST SUMMARY`
- report saved to `reports/continuation_test_report_debug.txt`

Healthy signs:

- target duration sum matches mel length
- predicted duration scale is frame-like, not tiny fractions
- speech-rate proxy stays around the healthy range seen in working runs
- output duration stays near the prior stable result around 1.86 s

## 5. Read TensorBoard

```bash
tensorboard --logdir=./logs
```

Expected output:

- TensorBoard startup message
- local URL, usually `http://localhost:6006`

What to watch:

- `train/token_duration_mae`
- `val/token_duration_mae`
- `train/sum_error_mean`
- `val/sum_error_mean`
- `train/pred_speech_rate_proxy`
- `val/pred_speech_rate_proxy`
- `train/loss`
- `val/loss`

## 6. What A Good Run Looks Like

- training completes without crashing
- validation loss is finite
- duration proxy stays around the same scale as the working runs
- HiFi-GAN output is clean
- evaluation report says duration is realistic

## 7. What A Bad Run Looks Like

- duration collapses to a tiny value
- proxy drops toward zero
- output becomes too short
- the report says the duration predictor is weak
- audio sounds rushed, smeared, or buzzy

## 8. Useful Files

- [README.md](README.md)
- [CHANGELOG.md](CHANGELOG.md)
- [reports/evaluation_09.04.2026.md](reports/evaluation_09.04.2026.md)
- [scripts/run_continuation_test.py](scripts/run_continuation_test.py)
- [train_vits.py](train_vits.py)

## Short Version

If you only want the important commands:

```bash
venv\Scripts\python.exe train_vits.py --config vits_config.yaml --device cuda
venv\Scripts\python.exe inference_vits.py --checkpoint checkpoints/best_model.pt --config vits_config.yaml --vocoder_checkpoint hifigan/generator_v1 --vocoder_config hifigan/config_v1.json --text "we are at present concerned" --output tts_output/output.wav --device cpu
venv\Scripts\python.exe scripts/evaluate_tts_output.py --audio tts_output/output.wav --sample_rate 22050
venv\Scripts\python.exe scripts/run_continuation_test.py --epochs 1 --duration-debug-checks --report-file reports/continuation_test_report_debug.txt
```
