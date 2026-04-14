# HexTTs Project Description

## Short Description

HexTTs is a research-driven neural Text-to-Speech (TTS) project focused on building, training, and improving a VITS-based speech synthesis system from scratch on local hardware. The project explores the full TTS pipeline, from text preprocessing and phoneme conversion to mel-spectrogram generation, neural inference, vocoding, evaluation, and training stabilization.

Unlike API-based speech tools, HexTTs is designed as a hands-on engineering and research project to understand how modern neural speech synthesis works internally. The system is trained on the LJSpeech dataset and developed with practical constraints in mind, including limited GPU memory, checkpoint compatibility, training instability, inference quality, and reproducible evaluation.

---

## Research-Oriented Description

HexTTs investigates how a compact but practical VITS-style TTS system can be implemented, trained, debugged, and improved in a local development environment. The project combines deep learning, speech processing, model experimentation, and software engineering to study both the generation quality and the operational reliability of a neural speech synthesis pipeline.

The work goes beyond simple model training by addressing real-world challenges that appear in custom TTS development, such as:

- phoneme and vocabulary consistency
- dataset validation and preprocessing
- cached feature generation for faster training
- duration prediction instability
- checkpoint and architecture compatibility
- inference quality issues such as buzz and unnatural timing
- evaluation using waveform-based metrics such as duration, RMS energy, zero-crossing rate, and spectral flatness

The broader goal of the project is to move from an experimental prototype toward a more maintainable and reproducible TTS framework with stronger configuration management, shared runtime paths, checkpoint validation, and structured testing.

---

## Technical Description

HexTTs is built around the VITS architecture and includes modules for:

- text preprocessing and phoneme sequence generation
- dataset preparation and validation
- raw and cached data loading workflows
- model training and checkpointing
- inference with configurable duration and noise scaling
- optional HiFi-GAN vocoder integration
- audio quality evaluation and continuation testing

The project is implemented in Python using PyTorch and related speech/audio tooling. A major engineering focus of HexTTs is making TTS experimentation feasible on modest consumer hardware, especially laptop GPUs with limited VRAM. This has led to optimizations such as cached feature pipelines, mixed precision training, resume-safe checkpoints, and targeted debugging tools for unstable duration behavior.

---

## Portfolio / Resume Version

HexTTs is an end-to-end neural Text-to-Speech project built with PyTorch and the VITS architecture. The project covers the full speech synthesis workflow: dataset validation, phoneme-based preprocessing, feature caching, model training, checkpoint management, inference, vocoder integration, and audio quality evaluation. A key focus of the project is training stability and practical deployment on limited local GPU hardware, including debugging duration prediction issues, reducing noisy outputs, and improving reproducibility through structured tooling and documentation.

---

## One-Line Version

HexTTs is a VITS-based neural Text-to-Speech research project that explores end-to-end speech synthesis, training stability, inference quality, and practical TTS engineering on local hardware.
