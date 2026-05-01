# HexTTs Training

This page is the quick training entrypoint. For the full explanation of model design, losses, checkpoints, and stability guardrails, see [Model and Training](model-training.md).

## Standard Run

```bash
python scripts/train.py --config configs/base.yaml --device cuda
```

## Short Profiles

```bash
python scripts/train.py --config configs/debug.yaml --device cuda
python scripts/train.py --config configs/sanity.yaml --device cuda
python scripts/train.py --config configs/continue3.yaml --device cuda
```

## Resume

```bash
python scripts/train.py --config configs/base.yaml --checkpoint checkpoints/checkpoint_step_080000.pt --device cuda
```

## Before Training

```bash
python scripts/validate_dataset.py ./data/LJSpeech-1.1
python scripts/prepare_data.py ./data/LJSpeech-1.1 ./data/ljspeech_prepared
python scripts/precompute_features.py --config configs/base.yaml
```

## Related Docs

- [Data Pipeline](data-pipeline.md)
- [Model and Training](model-training.md)
- [Operations and Troubleshooting](operations-troubleshooting.md)
