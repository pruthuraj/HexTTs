# Data Pipeline

The HexTTs data pipeline turns LJSpeech text/audio pairs into model-ready phoneme IDs, mel spectrograms, and optional duration targets.

## Dataset Layout

Expected raw dataset:

```text
data/LJSpeech-1.1/
  metadata.csv
  wavs/
    LJ001-0001.wav
    ...
```

Expected prepared dataset:

```text
data/ljspeech_prepared/
  train.txt
  val.txt
  metadata.json
  durations/
```

The config points to the active data paths:

```yaml
data_dir: ./data/ljspeech_prepared
audio_dir: ./data/LJSpeech-1.1/wavs
duration_dir: ./data/ljspeech_prepared/durations
```

## Validation

Validation should be the first step before training:

```bash
python scripts/validate_dataset.py ./data/LJSpeech-1.1
```

Validation protects against common expensive failures:

- missing `metadata.csv`
- missing audio files
- malformed prepared metadata
- cache files that do not match prepared splits
- unusable audio paths

## Metadata Preparation

Run:

```bash
python scripts/prepare_data.py ./data/LJSpeech-1.1 ./data/ljspeech_prepared
```

The preparation step:

- reads LJSpeech rows
- uses normalized text as the phoneme source
- converts text to phonemes with `g2p_en`
- removes empty or invalid phoneme conversions
- verifies referenced `.wav` files exist
- writes train/validation split files
- writes a small `metadata.json` summary

The model trains on phoneme sequences instead of raw characters because phonemes are a more direct representation of pronunciation.

## Phoneme IDs

The prepared metadata stores readable phoneme symbols. Runtime code maps those phonemes to integer IDs through the dataset vocabulary. The shared model builder uses one authoritative vocabulary size so training, inference, and checkpoint compatibility stay aligned.

Unknown phonemes are skipped with warnings. If all phonemes are skipped during inference, the inference path falls back to the `PAD` token so the model receives a valid tensor shape, but that output should be treated as a diagnostic result rather than useful speech.

## Duration Targets

TTS quality depends heavily on timing. HexTTs supports duration targets under:

```text
data/ljspeech_prepared/durations
```

Duration files provide per-phoneme frame supervision. They help the duration predictor learn how long each phoneme should last in the output mel sequence.

When duration supervision is weak or missing, speech timing can become unstable. That is why the project includes duration-specific config controls and training guardrails.

## Raw Versus Cached Loading

HexTTs supports two data loading paths:

- raw loading: compute features during training
- cached loading: precompute mel/id features before training

The active mode is selected in config:

```yaml
use_cached_features: true
```

Cached loading is recommended for normal runs because it removes repeated feature extraction from the training loop. Raw loading remains useful for debugging preprocessing behavior and validating changes to feature generation.

## Precomputing Features

Run:

```bash
python scripts/precompute_features.py --config configs/base.yaml
```

This stage should be rerun when any of these change:

- prepared metadata
- audio preprocessing parameters
- mel extraction settings
- phoneme vocabulary behavior
- source dataset location

## Dataset Audit

The simplified flow exposes dataset auditing:

```bash
python scripts/main_flow.py audit --dry-run
```

Useful audit thresholds include duration limits, RMS limits, silence ratio limits, and clip checking. Auditing is especially important because bad samples can cause clustered training instability and misleading loss behavior.

## Practical Failure Modes

The most common data-related issues are:

- prepared metadata references missing audio
- `use_cached_features` is true but cache files are missing
- duration targets do not match phoneme sequence lengths
- audio sample rate or mel settings changed after cache generation
- text normalization produced empty phoneme sequences
- very long or unusual samples destabilize batches

When these happen, fix the data artifact first. Training code should not be expected to compensate for inconsistent dataset inputs.
