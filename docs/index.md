# HexTTs Documentation Index

HexTTs is a local neural text-to-speech project built around a VITS-style PyTorch pipeline. The repository documents the full AI engineering path: validating LJSpeech, converting text to phonemes, caching mel features, training a compact speech model, resuming from checkpoints, synthesizing audio, and evaluating generated waveforms.

This documentation set is written for portfolio reviewers first. It explains both what the project does and why the engineering choices were made.

## Recommended Reading Path

1. [Project Rationale](project-rationale.md) - the problem, goals, constraints, and why a local VITS-style system was chosen.
2. [System Flow](system-flow.md) - the end-to-end flow from raw dataset to synthesized waveform.
3. [Architecture](architecture.md) - package layout and component responsibilities.
4. [Data Pipeline](data-pipeline.md) - dataset validation, phoneme metadata, duration targets, and cached features.
5. [Model and Training](model-training.md) - model structure, losses, checkpointing, and training stability decisions.
6. [Inference and Evaluation](inference-evaluation.md) - synthesis paths, vocoders, tuning parameters, and objective metrics.
7. [Operations and Troubleshooting](operations-troubleshooting.md) - common failures and how to diagnose them.

## Project Status

HexTTs is an active research and engineering project, not a production TTS service. It demonstrates an end-to-end local speech synthesis workflow with practical constraints:

- consumer-hardware training
- reproducible config-driven runs
- checkpoint-safe continuation
- cached feature pipelines for faster iteration
- explicit debugging around duration instability and noisy output
- objective waveform evaluation for generated audio

The current implementation is best described as a VITS-style or VITS-lite system. It includes the core text-to-mel training and inference workflow, but it is intentionally documented with its limitations: output quality is still under active improvement, duration prediction remains a key risk area, and neural vocoder quality strongly affects final audio.

## Quick Workflow

```bash
python scripts/validate_dataset.py ./data/LJSpeech-1.1
python scripts/prepare_data.py ./data/LJSpeech-1.1 ./data/ljspeech_prepared
python scripts/precompute_features.py --config configs/base.yaml
python scripts/train.py --config configs/base.yaml --device cuda
python scripts/infer.py --checkpoint checkpoints/best_model.pt --config configs/base.yaml --text "hello world" --output tts_output/hello.wav
python scripts/evaluate_tts_output.py --audio tts_output/hello.wav --sample_rate 22050
```

The same workflow can be accessed through the simplified command wrapper:

```bash
python scripts/main_flow.py train --device cuda
python scripts/main_flow.py infer --text "hello world" --output tts_output/output.wav --hifigan
python scripts/main_flow.py eval --audio tts_output/output.wav
python scripts/main_flow.py compare --text "we are at present concerned"
```

## Source Documents

The Markdown files in this folder are the source of truth. A polished portfolio DOCX summary is generated from this material at:

```text
docs/HexTTs_AI_Project_Document.docx
```
