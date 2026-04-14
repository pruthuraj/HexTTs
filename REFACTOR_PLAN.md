# HexTTs Refactor Plan

## Goal

Refactor HexTTs from a script-centered experimental TTS repo into a package-based, config-validated, checkpoint-versioned project with cleaner interfaces, better maintainability, and basic automated tests.

---

## _Refactoring is done by copilot_

## Main Diagnosis

HexTTs already has strong functional pieces:

- VITS model implementation
- training pipeline
- inference pipeline
- cached feature workflow
- dataset validation and preprocessing
- evaluation scripts
- continuation/debug tooling

The main issue is not missing functionality.
The main issue is that the repo is currently **patch-driven and script-centric**.

That creates long-term risks:

- training and inference can drift apart
- checkpoint compatibility becomes fragile
- configs can become partially implicit in code
- scripts may duplicate logic
- testing and reuse become harder

---

## Target Architecture

```text
HexTTs/
├── hextts/
│   ├── __init__.py
│   ├── config/
│   │   ├── __init__.py
│   │   ├── schema.py
│   │   ├── load.py
│   │   └── defaults.yaml
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── raw_dataset.py
│   │   ├── cached_dataset.py
│   │   ├── collate.py
│   │   ├── preprocessing.py
│   │   └── dataloaders.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── vits.py
│   │   ├── modules.py
│   │   └── checkpointing.py
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py
│   │   ├── losses.py
│   │   ├── callbacks.py
│   │   ├── logging.py
│   │   └── resume.py
│   │
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── synthesize.py
│   │   ├── text_processing.py
│   │   └── pipeline.py
│   │
│   ├── vocoder/
│   │   ├── __init__.py
│   │   ├── griffin_lim.py
│   │   ├── hifigan.py
│   │   └── factory.py
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   ├── audio_eval.py
│   │   └── reports.py
│   │
│   └── utils/
│       ├── __init__.py
│       ├── io.py
│       ├── audio.py
│       ├── warnings.py
│       └── versioning.py
│
├── scripts/
│   ├── train.py
│   ├── infer.py
│   ├── main_flow.py
│   ├── precompute_features.py
│   ├── validate_dataset.py
│   ├── evaluate_tts_output.py
│   ├── run_continuation_test.py
│   └── test_setup.py
│
├── tests/
│   ├── test_config.py
│   ├── test_tokenizer.py
│   ├── test_dataloaders.py
│   ├── test_model_shapes.py
│   ├── test_checkpointing.py
│   └── test_inference_smoke.py
│
├── configs/
│   ├── base.yaml
│   ├── sanity.yaml
│   ├── continue3.yaml
│   └── debug.yaml
│
├── docs/
│   ├── architecture.md
│   ├── training.md
│   ├── inference.md
│   ├── troubleshooting.md
│   └── experiments/
│
├── notes/
├── checkpoints/
├── data/
├── samples/
├── logs/
├── README.md
├── pyproject.toml
└── requirements.txt
```

---

## File-by-File Refactor Mapping

### `vits_model.py`

Move into:

- `hextts/models/vits.py`
- optionally split reusable submodules into `hextts/models/modules.py`

### `train_vits.py`

Split into:

- `scripts/train.py` for CLI only
- `hextts/training/trainer.py`
- `hextts/training/losses.py`
- `hextts/training/resume.py`
- `hextts/training/callbacks.py`

### `inference_vits.py`

Split into:

- `scripts/infer.py`
- `hextts/inference/pipeline.py`
- `hextts/inference/synthesize.py`
- `hextts/inference/text_processing.py`

### `vits_data.py` and `vits_data_cached.py`

Refactor into:

- `hextts/data/raw_dataset.py`
- `hextts/data/cached_dataset.py`
- `hextts/data/dataloaders.py`
- `hextts/data/collate.py`

External code should call one public API only:

```python
create_dataloaders(config)
```

That factory should decide whether to use raw or cached mode.

### `vits_config.yaml`

Refactor into:

- `configs/base.yaml`
- `hextts/config/schema.py`
- `hextts/config/load.py`

### `prepare_data.py`

Move core logic into:

- `hextts/data/preprocessing.py`

Keep the script as a thin wrapper.

### `precompute_features.py`

Keep as a script wrapper and move internal logic into:

- `hextts/data/preprocessing.py`
- or `hextts/data/cache_builder.py`

### `tts_app.py`

Make it call the shared inference pipeline in:

- `hextts/inference/pipeline.py`

### `scripts/evaluate_tts_output.py`

Move metric logic into:

- `hextts/evaluation/metrics.py`
- `hextts/evaluation/audio_eval.py`

### `vocoder.py`

Split into:

- `hextts/vocoder/griffin_lim.py`
- `hextts/vocoder/hifigan.py`
- `hextts/vocoder/factory.py`

---

## Refactor Principles

### 1. One authoritative config system

All scripts should load the same validated config object.

### 2. One authoritative model build path

Training and inference must build the model through the same code path.

### 3. One checkpoint contract

Checkpoint metadata must be explicit and validated before loading.

### 4. One public dataloader API

No manual import swapping for cached vs raw mode.

### 5. Scripts should be thin wrappers

CLI scripts should parse args and call library code, not contain core business logic.

---

## Recommended Refactor Order

### Phase 1 — Stabilize core interfaces

- create `hextts/` package
- add config loader and schema validation
- move model builder into shared module
- create shared checkpoint save/load utilities
- create shared dataloader factory

### Phase 2 — Thin out scripts

- make train and infer scripts wrappers only
- move logic into package modules
- keep old script names temporarily if needed for compatibility

### Phase 3 — Add runtime invariants

Add hard validation for:

- vocab size
- mel dimensions
- sample rate
- architecture flags
- checkpoint/config compatibility

### Phase 4 — Add tests

Start with smoke tests and shape tests.

### Phase 5 — Clean docs

- shorten README
- move deep notes into `docs/` and `notes/`
- keep humor, but separate official docs from dev-journal content

---

## Checkpoint Format Recommendation

Each checkpoint should save at least:

```python
{
    "model_state_dict": ...,
    "optimizer_state_dict": ...,
    "scheduler_state_dict": ...,
    "scaler_state_dict": ...,
    "epoch": ...,
    "global_step": ...,
    "config": full_config_dict,
    "model_version": "v0.5.0",
    "git_commit": "...",
    "vocab_size": ...,
    "sample_rate": ...,
    "n_mels": ...,
    "architecture_flags": {
        "use_postnet": True,
        "duration_clamp": 30.0
    }
}
```

The load path should validate compatibility before `load_state_dict`.

---

## First Tests to Add

### `tests/test_config.py`

- loads `base.yaml`
- validates required fields
- fails on invalid values

### `tests/test_tokenizer.py`

- checks deterministic text-to-token behavior
- checks empty or noisy input handling

### `tests/test_dataloaders.py`

- raw dataloader returns expected keys and shapes
- cached dataloader returns expected keys and shapes

### `tests/test_model_shapes.py`

- fake mini-batch runs through model
- output tensor shapes are checked

### `tests/test_checkpointing.py`

- save checkpoint
- reload checkpoint
- validate metadata checks

### `tests/test_inference_smoke.py`

- run one tiny inference path
- verify output artifact exists

---

## README Cleanup Plan

The top-level README should be shorter and more professional.

Suggested structure:

1. Project summary
2. Features
3. Architecture overview
4. Quickstart
5. Training
6. Inference
7. Evaluation
8. Known limitations
9. Roadmap
10. Links to deeper docs

Humorous or diary-style content should live in:

- `notes/`
- `docs/experiments/`
- optional separate fun markdown files

---

## Best Next Milestone

Recommended next release goal:

**v0.5.0 — Package Refactor and Runtime Stabilization**

Scope:

- package-based structure
- unified config loader
- unified dataloader factory
- checkpoint metadata validation
- smoke tests
- cleaned README

---

## Final Summary

HexTTs already shows real ML engineering effort.
The next leap is not another model tweak.
The next leap is **stronger boundaries, stable contracts, and cleaner interfaces**.

That is what will move the repo from an impressive experimental project to a serious, maintainable TTS engineering repo.
