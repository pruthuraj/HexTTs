# HexTTS Training & Evaluation Report

**Project:** HexTTS – Neural Text-to-Speech System
**Architecture:** Simplified VITS-based model
**Framework:** PyTorch
**Hardware:** NVIDIA RTX 3050 Ti (Laptop GPU)
**Dataset:** LJSpeech-style speech dataset
**Sample Rate:** 22050 Hz

---

# 1. Objective

The goal of this experiment was to train a **neural text-to-speech system using a simplified VITS architecture** that converts text into natural-sounding speech.

The training pipeline consists of:

Text → Tokenization → Encoder → Duration Modeling → Latent Representation → Decoder → Mel Spectrogram → Waveform Audio

The system was trained to learn the mapping between text tokens and corresponding speech audio using supervised speech data.

---

# 2. Training Configuration

## Hardware

GPU: NVIDIA RTX 3050 Ti
VRAM: 4GB

## Key Training Parameters

| Parameter           | Value    |
| ------------------- | -------- |
| Batch size          | 4        |
| Mixed precision     | Enabled  |
| Gradient clipping   | 0.5      |
| Max sequence length | 300      |
| Optimizer           | Adam     |
| Sample rate         | 22050 Hz |

---

# 3. Training Behavior

The model was trained for **25 epochs**.

During early experiments, the training process exhibited **instability**, including:

- NaN losses
- duration loss explosions
- skipped batches

These issues were mitigated by:

- gradient clipping
- improved duration supervision
- training stability checks

After stabilization, the training losses converged to reasonable ranges.

### Final Training Loss Statistics

| Metric              | Approximate Range |
| ------------------- | ----------------- |
| Total loss          | ~0.27             |
| Reconstruction loss | ~0.27             |
| KL divergence       | ~0.002            |
| Duration loss       | ~0.21             |

This indicates the **latent space remained stable** and the model learned reasonable spectrogram reconstruction.

---

# 4. Audio Output Evaluation

Generated audio samples were evaluated using a custom **HexTTS evaluation script**.

Metrics computed include:

- Duration
- RMS energy
- Silence ratio
- Zero Crossing Rate (ZCR)
- Spectral flatness
- Mel spectrogram statistics

---

# 5. Evaluation Results

## Sample 1

| Metric             | Value  |
| ------------------ | ------ |
| Duration           | 0.87 s |
| Peak amplitude     | 1.000  |
| RMS energy         | 0.201  |
| Silence ratio      | 0.044  |
| Zero crossing rate | 0.494  |
| Spectral flatness  | 0.886  |

---

## Sample 2

| Metric             | Value  |
| ------------------ | ------ |
| Duration           | 1.13 s |
| Peak amplitude     | 0.999  |
| RMS energy         | 0.159  |
| Silence ratio      | 0.054  |
| Zero crossing rate | 0.489  |
| Spectral flatness  | 0.885  |

---

## Sample 3

| Metric             | Value  |
| ------------------ | ------ |
| Duration           | 1.13 s |
| Peak amplitude     | 1.000  |
| RMS energy         | 0.154  |
| Silence ratio      | 0.055  |
| Zero crossing rate | 0.474  |
| Spectral flatness  | 0.892  |

---

# 6. Metric Interpretation

### Duration

Generated audio durations ranged between **0.87s and 1.13s**.

This indicates the model is producing **complete audio segments**, but they remain shorter than typical natural speech for the given text.

---

### Energy Metrics

RMS energy values ranged between **0.15 and 0.20**, indicating that the output waveform has **adequate amplitude and is not silent or collapsed**.

---

### Silence Ratio

Silence ratios around **0.04–0.05** indicate the model is not producing excessive silence.

---

### Zero Crossing Rate (ZCR)

Observed values:

```
ZCR ≈ 0.47 – 0.49
```

High ZCR values often indicate:

- noisy signals
- metallic artifacts
- unstable waveform structure

Natural speech typically has significantly lower ZCR.

---

### Spectral Flatness

Observed values:

```
Spectral Flatness ≈ 0.88 – 0.89
```

Spectral flatness measures how noise-like a signal is.

Typical ranges:

| Signal Type | Flatness  |
| ----------- | --------- |
| Speech      | 0.1 – 0.3 |
| Music       | 0.1 – 0.4 |
| Noise       | 0.6 – 1.0 |

The high values observed here strongly indicate the generated waveform is **dominated by noise-like spectral characteristics**.

---

# 7. Audio Quality Assessment

Subjective listening and objective metrics both indicate that the generated audio exhibits:

- buzzing artifacts
- metallic tone
- weak speech structure

Although the model produces audible signals with correct duration and energy levels, the waveform lacks the harmonic structure typical of natural speech.

---

# 8. Training Outcome

The training process achieved:

✔ numerical stability
✔ consistent loss convergence
✔ non-silent audio generation

However, the model did **not reach natural speech quality**.

---

# 9. Likely Causes

Several factors may contribute to the observed output quality limitations.

### Simplified architecture

The implemented model is a simplified version of the full VITS architecture and lacks several components present in production systems.

### Weak duration supervision

Duration modeling is simplified compared with the monotonic alignment mechanisms used in full VITS.

### Decoder limitations

The mel decoder may not learn sufficiently structured spectrogram representations.

### Waveform generation stage

The final waveform generation step may not provide enough spectral refinement.

---

# 10. Conclusion

The HexTTS experiment successfully demonstrated:

- stable training of a neural TTS system
- end-to-end text-to-audio generation
- evaluation of speech quality using objective metrics

However, the generated speech remains **noisy and lacks natural phonetic structure**, indicating that further architectural improvements are required.

---

# 11. Future Improvements

Potential improvements include:

### Architectural improvements

- improved mel decoder
- better duration supervision
- stronger alignment mechanisms

### Post-processing improvements

- mel refinement networks
- advanced waveform generation modules

### Training improvements

- larger datasets
- longer training schedules
- improved hyperparameter tuning

---

# 12. Final Assessment

The current system represents a **stable but early-stage neural TTS prototype**.

While the model successfully learns the basic mapping between text and audio signals, further development is necessary to produce **high-quality natural speech**.
