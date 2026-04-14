# HexTTs Architecture (Refactor Transition)

This repository is in transition from script-first to package-first design.

## Current Direction

- Shared runtime logic lives under `hextts/`.
- Root scripts are being converted into thin wrappers.
- Config, checkpoint, and dataloader contracts are unified.

## Package Layout

```text
hextts/
├── config/
│   ├── load.py            # shared config loader
│   └── schema.py          # runtime invariants and validation
├── data/
│   ├── dataloaders.py     # single raw/cached dataloader API
│   ├── preprocessing.py   # LJSpeech preprocessing pipeline
│   └── cache_builder.py   # precompute mel/ids cache pipeline
├── models/
│   ├── vits.py            # model build path shared by train/infer
│   └── checkpointing.py   # save/load + compatibility checks
├── training/
│   └── trainer.py         # training runner wrapper
├── inference/
│   ├── pipeline.py        # shared inference pipeline wrapper
│   └── text_processing.py # token-id helpers
└── evaluation/
    ├── metrics.py         # objective metric functions
    ├── audio_eval.py      # batch/single evaluation pipeline
    └── reports.py         # report printing
```

## Compatibility Strategy

- Legacy entrypoints (`train_vits.py`, `inference_vits.py`, `prepare_data.py`, `precompute_features.py`) remain available.
- New script wrappers under `scripts/` call package modules.
- Existing workflows continue to work while internals migrate.

## Next Migration Targets

- Expand test coverage for checkpoint and inference contracts.
- Continue moving remaining business logic out of script files.
- Keep docs aligned with package interfaces, not ad-hoc script internals.
