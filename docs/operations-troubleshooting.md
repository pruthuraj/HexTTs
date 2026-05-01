# Operations And Troubleshooting

This page lists common operational issues in HexTTs and the shortest useful diagnostic path for each.

## First Checks

Before debugging model behavior, confirm the basics:

```bash
python scripts/validate_dataset.py ./data/LJSpeech-1.1
python scripts/test_setup.py
python scripts/precompute_features.py --config configs/base.yaml
```

Then run a minimal inference smoke test:

```bash
python scripts/infer.py --checkpoint checkpoints/best_model.pt --config configs/base.yaml --text "hello world" --output tts_output/smoke.wav
```

## Config Validation Failure

Symptoms:

- startup fails before training or inference
- error mentions missing config keys or invalid mel settings

Check:

- config path exists
- required keys exist
- `sample_rate` is positive
- `mel_hop_length <= mel_win_length <= mel_n_fft`
- `mel_f_max <= sample_rate / 2`

Start from `configs/base.yaml` and copy it for experiments instead of creating configs from scratch.

## Dataset Or Cache Mismatch

Symptoms:

- dataloader fails
- batch items have unexpected shapes
- cache files are missing
- training starts failing after preprocessing changes

Check:

```bash
python scripts/validate_dataset.py ./data/LJSpeech-1.1
python scripts/precompute_features.py --config configs/base.yaml
```

If metadata, mel parameters, audio paths, or phoneme behavior changed, rebuild cached features.

## Checkpoint Compatibility Error

Symptoms:

- inference or resume fails when loading a checkpoint
- error mentions architecture/config mismatch

Cause:

The checkpoint metadata does not match the active config or model expectations.

Fix:

- use the config that produced the checkpoint
- do not mix checkpoints across incompatible architecture changes
- use `checkpoints/best_model.pt` only with matching project settings
- keep stable checkpoints before major experiments

## NaNs During Training

Symptoms:

- repeated batch skipping
- duration loss increases sharply
- training aborts due to skipped batch ratio
- predicted mel contains NaN or Inf

Likely causes:

- unstable duration predictor behavior
- bad or very long samples
- too aggressive learning rate
- AMP instability
- duration targets that do not match phoneme sequences

Actions:

- reduce learning rate
- disable AMP temporarily
- lower `max_seq_length`
- inspect skipped batches or dataset outliers
- confirm duration targets are valid
- resume from a stable checkpoint before the instability began

Relevant config controls:

```yaml
learning_rate: 0.0001
use_amp: true
grad_clip_val: 0.6
max_seq_length: 300
max_duration_value: 20.0
max_skipped_ratio: 0.5
duration_debug_checks: false
```

## Buzzy Or Metallic Audio

Symptoms:

- output is speech-like but noisy
- zero-crossing rate is high
- Griffin-Lim output sounds harsh
- HiFi-GAN improves output but does not fully fix it

Check:

```bash
python scripts/main_flow.py compare --text "we are at present concerned"
python scripts/evaluate_tts_output.py --audio tts_output --sample_rate 22050
```

Interpretation:

- if both Griffin-Lim and HiFi-GAN are poor, predicted mel quality or timing is likely the issue
- if HiFi-GAN is much better, waveform reconstruction is a major factor
- if duration feels wrong, tune `duration_scale` and inspect duration training behavior

## HiFi-GAN Setup Problem

Symptoms:

- inference fails when `--vocoder_checkpoint` is supplied
- error says one vocoder path is missing
- error says vocoder checkpoint or config does not exist

Fix:

Provide both paths:

```bash
python scripts/infer.py --checkpoint checkpoints/best_model.pt --config configs/base.yaml --vocoder_checkpoint hifigan/generator_v1 --vocoder_config hifigan/config_v1.json --text "hello world" --output tts_output/hifigan.wav
```

Confirm:

```text
hifigan/generator_v1
hifigan/config_v1.json
```

## Slow Training

Likely causes:

- raw feature extraction is happening during training
- batch size is too high for available hardware
- CPU dataloader workers are too low or too high for the machine
- GPU is falling back to CPU

Actions:

- set `use_cached_features: true`
- run `scripts/precompute_features.py`
- confirm `--device cuda` and CUDA availability
- use `configs/debug.yaml` or `configs/sanity.yaml` for short verification runs

## Documentation And Reports

Historical notes and reports exist under:

```text
docs/notes/
docs/reports/
```

Treat those as development history. For external review, use the maintained docs linked from [Documentation Index](index.md).
