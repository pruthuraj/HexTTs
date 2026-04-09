# HexTTS Evaluation Report: Duration Supervision Sanity Run, or "Can the Robot Stop Speedrunning the Sentence Without Falling Down the Stairs?" (09.04.2026)

**Date:** 09.04.2026  
**Branch:** `copilot/vscode-mnrjulze-pje7`  
**Goal:** Validate new token-level pseudo duration supervision + hybrid duration loss before the model gets cocky, starts freelancing, and we waste a full training run.

---

## Why This Run Happened

We replaced the old "roughly get the total length right and pray to the loss function" duration supervision with a more serious hybrid objective:

- token-level pseudo duration targets (sum-preserving allocation)
- SmoothL1 token loss (masked by valid phoneme lengths)
- L1 sum loss against mel lengths
- weighted as: `alpha=1.0`, `beta=0.2`

This sanity run checks whether timing improves without turning the waveform back into angry electrical confetti or a microwave with opinions.

---

## Sanity Run Setup

**Base config:** `vits_config.yaml`  
**Sanity config:** `vits_config.sanity.yaml`

Overrides used:

- `num_epochs: 1`
- `log_dir: ./logs/sanity_run`
- `checkpoint_dir: ./checkpoints_sanity`
- `checkpoint_interval: 1000`
- `validation_interval: 500`
- `duration_token_alpha: 1.0`
- `duration_sum_beta: 0.2`

Dataset status during run:

- `train.txt` and `val.txt` matched filtered counts, which is excellent because mystery data is how horror movies begin and spreadsheets become cursed
- `train.txt = 12238`
- `val.txt = 646`
- `train_filtered.txt = 12238`
- `val_filtered.txt = 646`

---

## Training Snapshot (Short Run)

Training command:

```bash
venv\Scripts\python.exe train_vits.py --config vits_config.sanity.yaml --device cuda
```

Result: **success** (CUDA / RTX 3050 Ti used, which is a polite way of saying the GPU survived another round of emotional arithmetic).

One mid-run training line (captured from the trenches, where the coffee is hot and the gradients are not):

```text
Epoch 1: 82% ... 2499/3060 ... loss=1.5 ...
```

Validation line:

```text
Epoch 1/1 - Validation loss: 1.9076
```

Best checkpoint saved:

- `checkpoints_sanity/best_model.pt`

---

## New Duration Diagnostics (TensorBoard)

Latest scalar values from `logs/sanity_run`:

| Metric                       | Step | Value          |
| ---------------------------- | ---- | -------------- |
| train/token_duration_mae     | 3000 | 1.2630032301   |
| train/sum_error_mean         | 3000 | 79.4142761230  |
| train/pred_speech_rate_proxy | 3000 | 8.1854972839   |
| val/token_duration_mae       | 3060 | 1.3781827688   |
| val/sum_error_mean           | 3060 | 150.2218933105 |
| val/pred_speech_rate_proxy   | 3060 | 8.3506851196   |

Quick interpretation, translated from machine pain into human words and small lies of optimism:

- Token MAE is finite and stable, which is the bar, and the bar did not need medical attention.
- Speech-rate proxy is consistent between train/val (~8.2 to ~8.35), so the model is not wildly improvising tempo like a jazz drummer who lost the chart.
- Sum error is still high (expected in early stage), but the run stayed stable and completed instead of launching the logs into orbit.

---

## HiFi-GAN Inference + Evaluation (Post-Sanity)

Inference command:

```bash
venv\Scripts\python.exe inference_vits.py --checkpoint checkpoints_sanity/best_model.pt --config vits_config.sanity.yaml --vocoder_checkpoint hifigan/generator_v1 --vocoder_config hifigan/config_v1.json --text "we are at present concerned" --output tts_output/hifigan_sanity.wav --device cpu
```

Evaluation command:

```bash
venv\Scripts\python.exe scripts/evaluate_tts_output.py --audio tts_output/hifigan_sanity.wav --sample_rate 22050
```

Key metrics:

- **Duration:** `1.8576 s`
- **ZCR:** `0.115707`
- **Spectral flatness:** `0.006294`

Evaluator verdict summary:

- Duration is in a realistic range, so the robot no longer sounds like it blacked out halfway through the sentence.
- Silence level looks acceptable.
- Energy level looks reasonable.
- Waveform is not excessively noisy by ZCR, which is a nice break from the usual robot-bee situation and a small victory for civilization.
- Spectrum has speech-like structure.

---

## Comparison Against Previous Short Outputs (~0.45 s)

Previous short outputs were around:

- `0.4412 s`
- `0.4528 s`

Sanity output after patch:

- `1.8576 s`

This is a major shift toward realistic utterance duration while keeping the waveform calmer than Griffin-Lim after three coffees.

---

## Conclusion

The patch passes short-run sanity checks:

- training remained stable and did not attempt a dramatic exit through the nearest wall
- new duration diagnostics are populated and meaningful
- output duration increased substantially
- HiFi-GAN output retained low-noise profile (low ZCR/flatness), which is what we wanted and not just a lucky accident

**Next step:** run a 3 to 5 epoch continuation with the same diagnostics and fixed sentence set to confirm the model is learning timing instead of just becoming confidently wrong in a fancier font.

---

## What I’d Do Next

Run the next experiment for 3 to 5 epochs, not 1, because the model has earned a slightly longer probation period.

Keep the setup unchanged for now:

- HiFi-GAN as the default output path
- filtered dataset
- same fixed sentence list
- current alpha/beta values (`alpha=1.0`, `beta=0.2`)

Track these every epoch so the robot cannot lie its way out of accountability:

- `train/token_duration_mae`
- `val/token_duration_mae`
- `train/sum_error_mean`
- `val/sum_error_mean`
- `pred_speech_rate_proxy`
- output duration
- `ZCR`
- `spectral flatness`

What good continuation should look like:

- output duration stays around the `1.5–2.5 s` range, or improves naturally without sprinting off a cliff
- `val/sum_error_mean` starts drifting downward
- `token_duration_mae` trends down slowly instead of throwing a tantrum
- speech-rate proxy stays stable
- `ZCR` stays low
- `flatness` stays low

Warning signs, otherwise known as "the robot is having a rough day":

- duration suddenly shoots much higher
- `sum_error_mean` gets worse every epoch
- speech-rate proxy swings wildly
- audio becomes longer but slurred or repetitive
- buzz returns like an uninvited sequel

Bottom line:

This sanity run is a green light. The patch is good enough to move from "sanity check" to a short real run of 3–5 epochs. If that run keeps the clean HiFi-GAN profile and starts pulling validation sum error down, then this duration patch is a real improvement and not just a lucky laboratory accident.

---

## Continuation Run (3 Epochs From Checkpoint)

Because we enjoy evidence more than wishful thinking, we resumed from the sanity checkpoint and ran a short continuation.

Resume command:

```bash
venv\Scripts\python.exe train_vits.py --config vits_config.continue3.yaml --checkpoint checkpoints_sanity/checkpoint_step_003000.pt --device cuda
```

Run status: **success**  
Continuation outputs:

- `checkpoints_continue_3e/best_model.pt`
- checkpoints through `checkpoint_step_012000.pt`
- TensorBoard events in `logs/continue_3e`

One mid-run training line (captured):

```text
Epoch 1: 16% ... 499/3060 ... loss=1.3084, recon=0.1926 ...
```

Validation completion: successful; latest validation scalars logged at step `12180`.

### Latest Continuation Diagnostics

| Metric                       | Step  | Value          |
| ---------------------------- | ----- | -------------- |
| train/token_duration_mae     | 12100 | 1.0136495829   |
| val/token_duration_mae       | 12180 | 1.3615362644   |
| train/sum_error_mean         | 12100 | 43.1375541687  |
| val/sum_error_mean           | 12180 | 148.4074096680 |
| train/pred_speech_rate_proxy | 12100 | 8.2769231796   |
| val/pred_speech_rate_proxy   | 12180 | 8.3342504501   |

Quick read:

- train token MAE improved versus the 1-epoch sanity run.
- train sum error improved substantially.
- speech-rate proxy stayed stable between train and val.
- val sum error remains high, but did not catastrophically diverge.

### HiFi-GAN Check After Continuation

Inference command:

```bash
venv\Scripts\python.exe inference_vits.py --checkpoint checkpoints_continue_3e/best_model.pt --config vits_config.continue3.yaml --vocoder_checkpoint hifigan/generator_v1 --vocoder_config hifigan/config_v1.json --text "we are at present concerned" --output tts_output/hifigan_continue3.wav --device cpu
```

Evaluation command:

```bash
venv\Scripts\python.exe scripts/evaluate_tts_output.py --audio tts_output/hifigan_continue3.wav --sample_rate 22050
```

Key metrics:

- **Duration:** `1.8460 s`
- **ZCR:** `0.115656`
- **Spectral flatness:** `0.010673`

Verdict summary:

- Duration remained realistic.
- Waveform stayed low-noise by ZCR.
- Spectrum remained speech-like.

Translation for tired humans:

The continuation did not break the good stuff. Timing stayed in the useful range, buzz did not return, and the duration patch still looks like a real improvement instead of a one-off lucky punch.
