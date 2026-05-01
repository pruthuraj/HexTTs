# HexTTs

HexTTs is a local neural text-to-speech project built around a VITS-style pipeline, cached feature loading, and optional HiFi-GAN vocoder inference.

It is designed as a practical AI engineering portfolio project for training, evaluating, and iterating on TTS systems end-to-end on consumer hardware.

---

## Overview

HexTTs covers the full text-to-speech workflow:

- dataset validation and preprocessing
- phoneme conversion and metadata generation
- cached feature extraction for faster training
- model training and checkpoint management
- inference with Griffin-Lim or optional HiFi-GAN
- waveform-based evaluation and diagnostics

The project emphasizes reproducibility, debuggability, and real local training workflows rather than API-only black-box synthesis.

---

## Portfolio Signals

This repository demonstrates practical AI engineering skills across:

- deep learning model training and inference pipelines
- data quality validation and preprocessing automation
- performance optimization through cached feature workflows
- experiment continuity via checkpoint-safe resume logic
- objective audio quality measurement and A/B comparison tooling
- package-first refactor discipline with thin CLI wrappers

---

## Project Status

Current capabilities include:

- VITS-style training and inference
- cached dataset pipeline for faster iteration
- resume-safe checkpoint workflows
- optional HiFi-GAN vocoder integration
- objective waveform evaluation metrics
- simplified CLI flow through `scripts/main_flow.py`

This repository is actively structured for incremental quality improvement and maintainable experimentation.

---

## Features

- VITS-style neural TTS training and inference
- raw and cached dataset loading pipelines
- optional HiFi-GAN vocoder support for cleaner synthesis
- dataset validation and preparation utilities
- waveform evaluation using:
  - duration
  - RMS energy
  - silence ratio
  - zero-crossing rate
  - spectral flatness
- checkpoint-based resume workflow for interrupted training
- package-first project structure with script wrappers

---

## Repository Layout

```text
HexTTs/
|- scripts/
|  |- train.py
|  |- infer.py
|  |- main_flow.py
|  |- prepare_data.py
|  |- precompute_features.py
|  |- validate_dataset.py
|  \- evaluate_tts_output.py
|
|- hextts/
|  |- data/
|  |- models/
|  |- inference/
|  |- training/
|  |- vocoder/
|  \- evaluation/
|
|- configs/
|  \- base.yaml
|
|- checkpoints/
|- logs/
|- docs/
|- diagram/
\- data/
```

---

## Requirements

- Python 3.9+
- PyTorch with CUDA support for training
- LJSpeech dataset
- dependencies listed in `requirements.txt`

For practical local training, a GPU is strongly recommended.

---

## Setup

Create and activate a virtual environment:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Place the LJSpeech dataset in:

```text
data/LJSpeech-1.1
```

Validate the dataset:

```bash
python scripts/validate_dataset.py ./data/LJSpeech-1.1
```

---

## Quick Start

Prepare the dataset, precompute features, train, and run inference:

```bash
python scripts/prepare_data.py ./data/LJSpeech-1.1 ./data/ljspeech_prepared
python scripts/precompute_features.py --config configs/base.yaml
python scripts/train.py --config configs/base.yaml --device cuda
python scripts/infer.py --checkpoint checkpoints/best_model.pt --config configs/base.yaml --text "hello world" --output tts_output/hello.wav
```

---

## Simplified Workflow

The project also provides a simplified CLI wrapper:

```bash
python scripts/main_flow.py <command> [options]
```

Example usage:

```bash
python scripts/main_flow.py train --device cuda
python scripts/main_flow.py infer --text "hello world" --output tts_output/hello.wav
python scripts/main_flow.py eval --audio tts_output/hello.wav
```

---

## Training

Recommended workflow:

1. Validate the dataset
2. Prepare phoneme metadata
3. Precompute cached features
4. Train from the prepared dataset
5. Monitor logs and checkpoints
6. Resume from checkpoints when needed

Example:

```bash
python scripts/train.py --config configs/base.yaml --device cuda
```

For faster iteration, use the cached pipeline by running feature precomputation first.

---

## Inference

Run text-to-speech inference from a trained checkpoint:

```bash
python scripts/infer.py \
	--checkpoint checkpoints/best_model.pt \
	--config configs/base.yaml \
	--text "hello world" \
	--output tts_output/hello.wav
```

By default, the project can use Griffin-Lim as a fallback reconstruction path.

If HiFi-GAN assets are available, inference can use the neural vocoder path for better output quality.

Example with HiFi-GAN:

```bash
python scripts/infer.py \
	--checkpoint checkpoints/best_model.pt \
	--config configs/base.yaml \
	--vocoder_checkpoint hifigan/generator_v1 \
	--vocoder_config hifigan/config_v1.json \
	--text "we are at present concerned" \
	--output tts_output/hifigan_test.wav
```

---

## Evaluation

Generated audio can be evaluated with the repository's waveform analysis tools:

```bash
python scripts/evaluate_tts_output.py --audio tts_output/hello.wav --sample_rate 22050
```

Common metrics:

- duration
- RMS energy
- silence ratio
- zero-crossing rate
- spectral flatness

These help compare runs and diagnose noisy, unstable, or overly buzzy output.

---

## Results Snapshot

The repository is already structured to support:

- objective comparison between Griffin-Lim and HiFi-GAN outputs
- faster training through cached mel features
- checkpoint continuation experiments
- training stability diagnostics for duration-related behavior

This makes HexTTs useful not only as a model implementation, but also as a debugging and evaluation environment for TTS improvement.

---

## Documentation

- `docs/index.md` - A-to-Z documentation entrypoint
- `docs/project-rationale.md` - what/why, goals, constraints, and limitations
- `docs/system-flow.md` - end-to-end pipeline flow
- `docs/model-training.md` - architecture, losses, checkpoints, stability guardrails
- `docs/inference-evaluation.md` - inference paths and objective metrics
- `docs/operations-troubleshooting.md` - operational diagnostics and failure handling
- `docs/HexTTs_AI_Project_Document.docx` - portfolio-ready project summary
- `readme.long.md` - full long-form version with deeper explanations and project personality
- `PROJECT_DESCRIPTION.md` - broader technical description
- `diagram/` - visual architecture and flow references

---

## Notes

- back up `checkpoints/best_model.pt` before major experiments
- use the cached pipeline for faster training whenever possible
- review `configs/base.yaml` before long runs
- keep training and inference code aligned with checkpoint architecture changes

---

## Roadmap

Potential next steps:

- improve output naturalness and prosody
- strengthen training stability further
- expand HiFi-GAN integration and evaluation workflows
- add cleaner experiment tracking and result comparison
- extend toward advanced vocoder or multi-speaker experiments

---

## License

MIT

---

## Extended Readme

For the full detailed version, including troubleshooting, expected outputs, and the more expressive project write-up, see:

```text
readme.long.md
```
