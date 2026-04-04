# HexTTS Architecture Graphs (Mermaid)

This file contains Mermaid diagrams for the HexTTS project.
Paste these directly into GitHub README, documentation, or Mermaid Live Editor.

---

## 1. HexTTS System Overview

```mermaid
flowchart LR
    A[Raw Dataset wav + metadata] --> B[Text Processing clean normalize phonemes]
    B --> C[Feature Preparation mel spectrograms cache]
    C --> D[Training VITS model + checkpoints]
    D --> E[Inference load model + generate speech]

    D --> F[Training Outputs checkpoints + logs]
    E --> G[Runtime Controls play pause speed pitch]
    C --> H[Optimizations cached mel cached phonemes]
```
---

## 2. Dataset Preparation Flow

```mermaid
flowchart LR
    A[Collect Audio] --> B[Validate Dataset]
    B --> C[Normalize Text]
    C --> D[Generate Phonemes]
    D --> E[Create Metadata]
    E --> F[Precompute Cache]

    B --> B1[Checks missing files sample rate duplicates]
    D --> D1[Vocab checks align dataset and model]
    F --> F1[Cache mel files phoneme ids]
```
---

## 3. Training Pipeline

```mermaid
flowchart LR
    A[Batch Loader] --> B[Text Encoder]
    A --> C[Posterior Encoder]
    B --> D[Flow Module]
    C --> D
    D --> E[Decoder / Vocoder]

    E --> F[Reconstruction Loss]
    D --> G[KL Loss]
    B --> H[Duration Loss]

    F --> I[Total Loss]
    G --> I
    H --> I

    I --> J[Backpropagation]
    J --> K[Optimizer Step]
    K --> L[Checkpoint Save]
```
---

## 4. Inference Pipeline

```mermaid
flowchart LR
    A[Input Text] --> B[Text Cleanup]
    B --> C[Phoneme IDs]
    C --> D[Load Trained VITS Model]
    D --> E[Waveform Output]
    E --> F[Playback or Save WAV]

    D --> G[Synthesis Controls noise scale speed]
```
---

## 5. Training Metrics Dashboard

```mermaid
flowchart TD
    A[Training Metrics] --> B[Total Loss]
    A --> C[Reconstruction Loss]
    A --> D[KL Loss]
    A --> E[Learning Rate]

    B --> F[Model Convergence]
    C --> F
    D --> F
    E --> F
```
---

## 6. Deployment Flow

```mermaid
flowchart LR
    A[Development configs scripts model code] --> B[Artifacts checkpoints logs cache]
    B --> C[Application Layer CLI or app]
    C --> D[User Output speech playback wav export]
```
---

## 7. VITS Architecture

```mermaid
flowchart LR
    A[Input Text] --> B[Text Encoder]
    B --> C[Prior Distribution]

    D[Ground Truth Audio] --> E[Posterior Encoder]
    E --> F[Latent Representation]

    C --> G[Normalizing Flow]
    F --> G

    G --> H[Decoder]
    H --> I[Generated Waveform]

    B --> J[Duration Predictor]
    J --> H
```
---

## 8. TTS Architecture Comparison

```mermaid
flowchart TD
    A[TTS Models]

    A --> B[Tacotron]
    A --> C[FastSpeech]
    A --> D[VITS]

    B --> B1[Autoregressive]
    C --> C1[Non-autoregressive]
    D --> D1[End-to-end]

    D --> D2[Integrated vocoder]
```
---

## 9. Mel Spectrogram Pipeline

```mermaid
flowchart LR
    A[Raw Audio] --> B[Frame Signal]
    B --> C[Fourier Transform]
    C --> D[Power Spectrum]
    D --> E[Mel Filter Bank]
    E --> F[Log Mel Spectrogram]
    F --> G[Model Input Feature]
```
---

## 10. Checkpoint Resume Logic

```mermaid
flowchart TD
    A[Start Training] --> B{Checkpoint exists}
    B -- No --> C[Start from scratch]
    B -- Yes --> D[Load checkpoint]

    D --> E[Restore weights]
    E --> F[Restore optimizer]
    F --> G[Restore epoch]
    G --> H[Continue training]

    C --> H
```
---

## 11. Cached Training Flow

```mermaid
flowchart LR
    A[Raw wav + text] --> B[Compute mel spectrogram]
    A --> C[Generate phoneme ids]

    B --> D[Cached mel files]
    C --> E[Cached phoneme ids]

    D --> F[Fast DataLoader]
    E --> F

    F --> G[Training Loop]
```
---

## 12. Vocabulary Consistency Fix

```mermaid
flowchart TD
    A[Dataset phoneme mapping] --> B[PHONEME_TO_ID]
    B --> C[VOCAB_SIZE = len mapping]
    C --> D[Model embedding layer]

    E[Hardcoded vocab] -. causes mismatch .-> D
    D --> F[Consistent training]
```
