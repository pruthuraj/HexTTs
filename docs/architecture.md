# HexTTs Architecture

This document replaces the old archived copy in [deprecated/doc_history/architecture.md](../deprecated/doc_history/architecture.md).

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
│   ├── checkpointing.py   # save/load + compatibility checks
│   └── modules.py         # reusable model building blocks
├── training/
│   ├── trainer.py         # training runner wrapper
│   ├── losses.py          # named loss breakdown helpers
│   ├── callbacks.py       # callback protocol and no-op implementation
│   └── logging.py         # logger setup helper
├── inference/
│   ├── pipeline.py        # shared inference pipeline wrapper
│   └── text_processing.py # token-id helpers
├── vocoder/
│   ├── hifigan.py         # pretrained HiFi-GAN wrapper
│   ├── griffin_lim.py     # Griffin-Lim fallback helper
│   └── factory.py         # vocoder construction helper
├── utils/
│   ├── io.py              # text and path helpers
│   ├── audio.py           # waveform helpers
│   ├── warnings.py        # warning configuration
│   └── versioning.py      # checkpoint/version metadata helpers
└── evaluation/
    ├── metrics.py         # objective metric functions
    ├── audio_eval.py      # batch/single evaluation pipeline
    └── reports.py         # report printing
```

## Compatibility Strategy

- Runtime entrypoints are script wrappers under `scripts/`.
- Root legacy wrappers have been removed after migration.
- Existing workflows should use `scripts/train.py`, `scripts/infer.py`, `scripts/prepare_data.py`, and `scripts/precompute_features.py`.

## Next Migration Targets

- Expand test coverage for checkpoint and inference contracts.
- Continue moving remaining business logic out of script files.
- Keep docs aligned with package interfaces, not ad-hoc script internals.
