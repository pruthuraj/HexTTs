# Project Rationale

## What HexTTs Is

HexTTs is a local neural text-to-speech project built with Python, PyTorch, and a VITS-style architecture. It takes text, converts it into phoneme IDs, predicts mel spectrograms, and converts those spectrograms into waveform audio through either Griffin-Lim or an optional HiFi-GAN vocoder.

The purpose is not to wrap a cloud TTS API. The purpose is to expose the engineering work behind speech synthesis: data preparation, model design, training stability, inference behavior, checkpoint compatibility, and quality evaluation.

## Why Build A Local TTS System

Local TTS development forces the project to solve problems that managed APIs hide:

- Is the dataset structurally valid and consistent?
- Are text, phonemes, audio files, and duration targets aligned?
- Can the model train on limited GPU memory?
- Can long runs be resumed safely?
- Do generated waveforms contain speech-like structure or only noisy audio?
- Which parts of the pipeline are responsible for poor output quality?

That makes HexTTs useful as an AI engineering portfolio project. It shows how the system behaves under real training constraints rather than only showing a polished inference endpoint.

## Why A VITS-Style Design

VITS is a relevant design target because it connects text encoding, duration modeling, latent acoustic representation, and waveform generation concepts. HexTTs implements a practical VITS-style path around the pieces that are most useful for local experimentation:

- phoneme-based text representation
- transformer-style text encoding
- duration prediction and regulation
- mel spectrogram prediction
- checkpointed training and inference
- external vocoder support

The implementation is deliberately treated as VITS-style rather than full VITS. A full production-grade VITS stack would require stronger alignment learning, posterior/flow modeling, adversarial waveform losses, and more extensive vocoder integration. HexTTs documents that gap instead of hiding it.

## Why LJSpeech

LJSpeech is a practical single-speaker dataset for local TTS experiments. It provides paired utterance text and audio, making it suitable for proving the end-to-end pipeline before expanding to harder settings such as multi-speaker training, expressive prosody, or domain-specific voices.

The repository expects the raw dataset at:

```text
data/LJSpeech-1.1
```

The prepared metadata is written to:

```text
data/ljspeech_prepared
```

## Design Goals

HexTTs prioritizes:

- reproducibility through config files and deterministic preprocessing
- debuggability through validation, reports, warnings, and explicit errors
- iteration speed through cached feature loading
- recoverability through checkpoint metadata and resume workflows
- honest quality assessment through objective waveform metrics
- maintainability through package code under `hextts/` and thin CLI wrappers under `scripts/`

## Current Limitations

The current project can generate audio and support training/evaluation experiments, but the synthesized quality is still a research target. Known limitations include:

- duration prediction can become unstable during training
- generated audio can be buzzy or metallic depending on checkpoint and vocoder path
- Griffin-Lim is useful as a fallback but is not expected to produce high-fidelity speech
- HiFi-GAN improves waveform synthesis but still depends on the quality of predicted mel features
- the architecture does not yet represent every component of full VITS

These limitations are part of the project's value: they identify concrete next engineering steps instead of presenting the model as finished.

## Success Criteria

For this stage, success means the project can:

- validate and prepare a known speech dataset
- train a model from config-driven entrypoints
- resume or reject checkpoints predictably
- synthesize audio from text using a saved checkpoint
- compare vocoder paths and inference settings
- report objective audio diagnostics
- explain its architecture, tradeoffs, and failure modes clearly
