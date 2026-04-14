# HexTTs Training

Current training entrypoint:

```bash
venv\Scripts\python.exe scripts\train.py --config configs/base.yaml --device cuda
```

Quick profile variants:

```bash
# Fast debug smoke run
venv\Scripts\python.exe scripts\train.py --config configs/debug.yaml --device cuda

# One-epoch sanity run
venv\Scripts\python.exe scripts\train.py --config configs/sanity.yaml --device cuda

# 3-epoch continuation profile
venv\Scripts\python.exe scripts\train.py --config configs/continue3.yaml --device cuda
```

Resume from checkpoint:

```bash
venv\Scripts\python.exe scripts\train.py --config configs/base.yaml --checkpoint checkpoints/checkpoint_step_080000.pt --device cuda
```

## Topics

- config-driven training setup
- raw vs cached dataloaders
- checkpoint save and resume flow
- training stability and runtime invariants
