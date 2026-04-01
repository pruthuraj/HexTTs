# CHANGELOG

All notable changes to this project are documented here.

---

# v0.3.0 — Training Stability + Performance

## Added

- Mixed precision (`use_amp`) support
- RTX 3050 Ti optimized config
- Cached feature training pipeline
- `precompute_features.py`
- `vits_data_cached.py` loader

## Improved

- GPU utilization
- Training speed
- CPU workload

---

# v0.2.1 — Vocabulary Consistency Fix

## Fixed

Model vocabulary mismatch.

Before:

```
VOCAB_SIZE = 149
```

After:

```
from vits_data import VOCAB_SIZE
```

Result:

- embedding size matches dataset
- removed phantom phoneme tokens
- cleaner architecture

---

# v0.2.0 — Training Pipeline

## Added

- `train_vits.py`
- checkpoint saving
- tensorboard logging
- inference script
- interactive TTS app

---

# v0.1.0 — Project Foundation

## Added

- dataset preparation scripts
- phoneme conversion
- dataset validation
- project structure

Dataset used:

- LJSpeech (13,100 samples)

---

# Known Issues

- Griffin-Lim vocoder is slow
- laptop fans become aggressive
- electricity bill may increase

---

Last updated: 01.04.2026
