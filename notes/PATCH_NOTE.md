# Patch Note

## File
`prepare_data.py`

## Problem
Training produced huge "unknown phoneme" warnings such as:

- `THE`
- `OF`
- `AND`
- `TO`

That means the metadata being written for training was still too close to raw text tokens instead of clean ARPAbet phoneme tokens.

## Root Cause
The training loader expects:

```text
filename|PHONEME PHONEME PHONEME
```

but the prepared metadata could still contain non-phoneme tokens or inconsistent conversion output.

## Fix Applied
The data preparation step was patched so that it now:

1. uses `g2p_en` to convert text into phoneme tokens
2. removes stress markers like `AH0 -> AH`
3. uppercases all phoneme tokens
4. removes empty tokens and spaces
5. keeps only alphabetic phoneme tokens
6. skips samples whose phoneme result is empty
7. shuffles data before train/validation split
8. stores clean phoneme-only metadata in:
   - `train.txt`
   - `val.txt`

## Expected Result
Metadata lines now look like:

```text
LJ001-0001|DH AH P R AA JH EH K T G UW T AH N B ER G
```

instead of raw-word style lines such as:

```text
LJ001-0001|THE PROJECT GUTENBERG
```

This should drastically reduce unknown phoneme warnings during training.

## What To Run
Regenerate the prepared dataset:

```bash
python prepare_data.py ./data/LJSpeech-1.1 ./data/ljspeech_prepared
```

Then validate a few lines manually in `train.txt` and confirm they contain phonemes, not words.

After that, restart training from scratch:

```bash
python train_vits.py --config vits_config.yaml --device cuda
```

## Important Note
Do **not** continue training from an old checkpoint produced using bad metadata.  
Start a fresh run after regenerating `train.txt` and `val.txt`.
