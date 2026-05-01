# Inference And Evaluation

Inference turns text into waveform audio using a trained checkpoint. Evaluation measures generated audio so runs can be compared instead of judged only by memory or subjective listening.

## Basic Inference

Run:

```bash
python scripts/infer.py --checkpoint checkpoints/best_model.pt --config configs/base.yaml --text "hello world" --output tts_output/hello.wav
```

The inference path:

1. loads and validates the config
2. builds the VITS-style model
3. loads checkpoint weights
4. converts text to phonemes with `g2p_en`
5. maps phonemes to IDs
6. predicts a mel spectrogram
7. converts the mel spectrogram to waveform audio
8. writes the `.wav` output

## Griffin-Lim Fallback

If no neural vocoder paths are provided, HexTTs uses Griffin-Lim reconstruction. This is useful because it works without extra vocoder assets, but it is not expected to produce the best speech quality.

Use Griffin-Lim for:

- smoke testing model inference
- confirming checkpoint compatibility
- debugging mel prediction shape and scale
- running on minimal environments

Do not treat Griffin-Lim quality as the final ceiling for the model.

## HiFi-GAN Path

HiFi-GAN is the preferred waveform path when assets are available:

```bash
python scripts/infer.py --checkpoint checkpoints/best_model.pt --config configs/base.yaml --vocoder_checkpoint hifigan/generator_v1 --vocoder_config hifigan/config_v1.json --text "we are at present concerned" --output tts_output/output_hifigan.wav
```

Both vocoder arguments are required together:

```text
--vocoder_checkpoint hifigan/generator_v1
--vocoder_config hifigan/config_v1.json
```

The vocoder improves waveform synthesis, but it cannot fully repair poor predicted mel features. If mel predictions are noisy or badly timed, HiFi-GAN may still produce unnatural audio.

## Inference Parameters

Important inference controls:

```yaml
inference_duration_scale: 4.0
inference_noise_scale: 0.3
```

Command-line inference also exposes duration and noise scaling through the wrapper:

```bash
python scripts/main_flow.py infer --text "hello world" --output tts_output/output.wav --duration_scale 3.0 --noise_scale 0.3 --hifigan
```

Practical interpretation:

- higher duration scale slows speech down
- lower duration scale speeds speech up
- higher noise scale can add variation but may increase artifacts
- lower noise scale can make output more stable but less expressive

## Side-By-Side Comparison

The simplified workflow can generate Griffin-Lim and HiFi-GAN outputs for the same text and evaluate both:

```bash
python scripts/main_flow.py compare --text "we are at present concerned"
```

This is useful for separating model issues from vocoder issues. If both outputs are poor, the mel prediction or timing may be the main problem. If HiFi-GAN is much better, waveform reconstruction was likely the limiting factor.

## Objective Evaluation

Run:

```bash
python scripts/evaluate_tts_output.py --audio tts_output/output_hifigan.wav --sample_rate 22050
```

The evaluation pipeline reports metrics such as:

- duration
- RMS energy
- silence ratio
- zero-crossing rate
- spectral flatness

These metrics help detect obvious problems:

- very high silence ratio can indicate missing or weak output
- very high zero-crossing rate can indicate buzzy or noisy audio
- abnormal RMS energy can indicate weak, clipped, or unstable waveform levels
- high spectral flatness can indicate noise-like output

## How To Use Metrics Responsibly

Objective metrics do not prove that speech sounds natural. They are diagnostic signals. The best evaluation loop is:

1. generate audio from a fixed test phrase
2. listen to the output
3. run waveform metrics
4. compare against earlier checkpoints
5. record whether changes improved timing, clarity, and noise

The repository keeps prior reports under `docs/reports/` for historical context, but future reports should use professional language and focus on reproducible observations.

## Common Inference Failures

Common inference blockers include:

- missing checkpoint file
- checkpoint metadata incompatible with config
- `g2p_en` unavailable
- missing HiFi-GAN checkpoint or config
- one vocoder path provided without the other
- NaN or Inf in predicted mel output
- output directory missing or not writable

When inference fails, confirm the checkpoint/config pair first, then isolate text processing, mel generation, and vocoder conversion separately.
