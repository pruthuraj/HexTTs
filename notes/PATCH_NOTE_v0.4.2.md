# HexTTs v0.4.2 Patch Notes

## "Finally, You Can Batch Evaluate Audio Without Running The Script 47 Times"

**Release Date:** April 6, 2026  
**Tag:** v0.4.2  
**Status:** Stable

---

## Overview

v0.4.2 focuses on **user experience improvements** and **code quality enhancements**. The training system is solid; this release makes it more pleasant to use and easier to understand. Two primary themes:

1. **Batch Audio Evaluation** — Evaluate multiple files at once instead of one-at-a-time misery
2. **Code Documentation** — Comments that explain the "why," not just the "what"

---

## Major Features

### 1. Batch Audio Evaluation Mode

**File:** `evaluate_tts_output.py`

#### What Changed

The audio evaluation script now supports three modes:

```bash
# Mode 1: Default — evaluates all .wav files in current directory
python evaluate_tts_output.py

# Mode 2: Specific folder
python evaluate_tts_output.py --audio ./samples

# Mode 3: Single file (legacy, still works)
python evaluate_tts_output.py --audio test.wav
```

#### Implementation Details

```python
# New argument parsing
parser.add_argument(
    '--audio',
    type=str,
    default='.',
    help='Path to audio file or folder containing .wav files'
)
```

The script now:

1. Checks if `--audio` is a file or directory
2. If directory: recursively finds all `.wav` files
3. If single file: evaluates just that file
4. If omitted: defaults to current directory (`.`)

#### Batch Summary Table

When evaluating multiple files, you get a clean summary:

```
Batch Summary:
================================================================
1. epoch_005_sample_1.wav     | 2.341s | RMS: 0.1902 | ZCR: 0.4821
2. epoch_005_sample_2.wav     | 2.156s | RMS: 0.1745 | ZCR: 0.5102
3. epoch_010_sample_1.wav     | 2.489s | RMS: 0.2134 | ZCR: 0.4756
================================================================
```

**Features:**

- Duration formatted with 3 decimal places
- RMS and ZCR metrics for quick comparison
- Thousand separators on sample counts (e.g., "1,024" not "1024")
- Continues processing if one file fails (error resilient)

#### Error Handling

If one audio file is corrupted:

```
[OK] Processing epoch_005_sample_1.wav
[ERROR] Processing epoch_005_sample_2.wav ... [ERROR: Corrupted file]
[OK] Processing epoch_010_sample_1.wav
```

The script doesn't crash. It logs the error and continues. This is critical for large batches.

---

### 2. Enhanced Code Documentation

**Files:** `train_vits.py` (primary)

#### What Changed

The `log_audio_samples()` function now has detailed inline comments explaining:

- **What each operation does** — not just "tensor manipulation"
- **Why it matters** — the purpose of normalization, shape requirements, etc.
- **Tensor shapes** — in and out dimensions explicitly documented
- **Integration points** — how this connects to TensorBoard

#### Example: Before vs After

**Before (v0.4.1):**

```python
def log_audio_samples(self, samples, epoch, sr=22050):
    # Quick ugly version with minimal comments
    log_dir = os.path.join('./logs', f'samples_epoch_{epoch:03d}')
    os.makedirs(log_dir, exist_ok=True)

    for idx, wav in enumerate(samples):
        path = os.path.join(log_dir, f'sample_{idx}.wav')
        # ... some processing ...
        scipy.io.wavfile.write(path, sr, wav)
```

**After (v0.4.2):**

```python
def log_audio_samples(self, samples, epoch, sr=22050):
    """
    Log audio samples to disk and TensorBoard for epoch visualization.

    Args:
        samples: List of waveforms (numpy arrays or tensors)
        epoch: Current training epoch
        sr: Sample rate in Hz (default: 22050 Hz for LJSpeech)

    Process:
        1. Create epoch-specific directory for organization
        2. Normalize waveforms to [-1, 1] range for consistency
        3. Save as 16-bit PCM WAV files (standard format)
        4. Log to TensorBoard for comparison over training
    """

    # Create organized directory structure
    # Format: logs/samples_epoch_005/ for epoch 5
    log_dir = os.path.join('./logs', f'samples_epoch_{epoch:03d}')
    os.makedirs(log_dir, exist_ok=True)

    for idx, wav in enumerate(samples):
        # Normalize waveform to prevent clipping
        # Input may be from model with arbitrary magnitude
        # Output needs to be in [-1, 1] range for WAV format
        if isinstance(wav, torch.Tensor):
            wav = wav.detach().cpu().numpy()

        # Prevent clipping: scale to peak ±1.0
        max_val = np.max(np.abs(wav))
        if max_val > 1.0:
            wav = wav / max_val

        # Save with standard parameters
        # 16-bit PCM is universally supported
        path = os.path.join(log_dir, f'sample_{idx}.wav')
        scipy.io.wavfile.write(path, sr, (wav * 32767).astype(np.int16))

        # Log to TensorBoard for training visualization
        # Allows listening to samples directly in TensorBoard UI
        self.writer.add_audio(
            f'samples/epoch_{epoch:03d}_sample_{idx}',
            torch.from_numpy(wav).unsqueeze(0).float(),
            global_step=epoch,
            sample_rate=sr
        )
```

#### Documentation Philosophy

Every major section now follows this pattern:

```python
# WHAT — Brief description of operation
# WHY — Purpose or importance
# HOW — Specific implementation details or gotchas

operation()
```

#### Docstrings Updated

All functions now have docstrings including:

- Clear description of purpose
- Args section with types
- Returns section with shape info
- Notes section for important caveats

---

## Improvements Section

### Evaluation Output Formatting

**File:** `evaluate_tts_output.py`

The evaluation output is now prettier and easier to scan:

**Before:**

```
file: test.wav
duration_sec: 2.34
rms_energy: 0.1902
silence_ratio: 0.041
zero_crossing_rate: 0.492
```

**After:**

```
Audio Report:
================================================================
File               : test.wav
Duration           : 2.340 seconds
Peak amplitude     : 0.99997
RMS energy         : 0.190200
Silence ratio      : 0.040789
Zero crossing rate : 0.492679
================================================================

Verdict:
• Sample is not excessively quiet (RMS > 0.05)
• Waveform is not excessively noisy (ZCR < 0.6)
```

#### Specific Changes

1. **Visual indicators** — Labels for reports and verdicts (clear sections)
2. **Right-aligned numbers** — Easier to compare values vertically
3. **Separator lines** — Clear sections
4. **Bullet points** — Verdict messages structured as list
5. **Better labeling** — "Zero crossing rate" instead of "zcr"

---

## Internal Changes

### Audio Evaluation Pipeline Refactoring

**File:** `evaluate_tts_output.py`

The evaluation logic was reorganized for **flexibility and reusability**:

#### Before

```python
def evaluate_single(audio_path):
    # Monolithic function mixing I/O, computation, and formatting
    report = compute_metrics(audio_path)
    print_ugly_output(report)
    return report
```

#### After

```python
def evaluate_audio(audio_path: str) -> dict:
    """Compute metrics only (pure function)"""
    # Returns: {'file': ..., 'duration_sec': ..., 'rms_energy': ...}

def print_report(report: dict) -> None:
    """Format and display report (separate concern)"""
    # Takes report dict, prints formatted output

def main():
    """High-level orchestration"""
    audio_files = find_audio_files(args.audio)
    for file in audio_files:
        report = evaluate_audio(file)
        print_report(report)
    batch_summary(all_reports)
```

#### Benefits

- **Testing** — Can test metrics computation without mocking print()
- **Reuse** — Other scripts can `from evaluate_tts_output import evaluate_audio`
- **Maintenance** — Changing output format doesn't affect metric computation
- **Batch processing** — Naturally supports multiple files

---

## Configuration

No breaking changes to `vits_config.yaml` in v0.4.2. All settings from v0.4.1 remain valid.

---

## Migration Guide

### From v0.4.1 to v0.4.2

**No breaking changes.** Your existing configs, trained models, and data all work unchanged.

#### New Usage Patterns

```bash
# Old way (still works)
python evaluate_tts_output.py --audio test.wav

# New way (much faster for iteration)
python evaluate_tts_output.py  # everything in current dir
python evaluate_tts_output.py --audio ./samples/epoch_050  # whole epoch

# Pipe to file (great for logs)
python evaluate_tts_output.py --audio ./tts_output > eval_results.txt
```

---

## Known Limitations

### 1. Batch Evaluation Order

Files are evaluated in **alphabetical order** by filename. If you care about epoch order:

```bash
# Good — will evaluate in numerical order
samples/epoch_005_sample_1.wav
samples/epoch_005_sample_2.wav
samples/epoch_010_sample_1.wav
```

vs.

```bash
# Bad — will be out of order
samples/sample_1_epoch_5.wav
samples/sample_1_epoch_10.wav
```

### 2. Noise on One File = Batch Fails Output

If 1 of 100 files is corrupted, the batch summary won't include metrics from that file. This is intentional (garbage in = garbage out), but document it.

---

## Performance Impact

### Positive

- **Batch evaluation:** ~10-50× faster than running script 100 times
- **Disk I/O:** Negligible (audio files are usually small)
- **Memory:** Each evaluation is independent; old files are garbage collected

### Neutral

- **Training speed:** No impact (evaluation is separate)
- **Model size:** No impact
- **Training time:** No impact

---

## Testing Recommendations

### Manual Testing Checklist

```bash
# Test 1: Single file
python evaluate_tts_output.py --audio tts_output/hello.wav

# Test 2: Directory with multiple files
python evaluate_tts_output.py --audio ./tts_output

# Test 3: Default (current directory)
cd tts_output && python ../evaluate_tts_output.py

# Test 4: Directory with no .wav files (should fail gracefully)
mkdir empty_dir && python evaluate_tts_output.py --audio empty_dir

# Test 5: Non-existent path (should fail gracefully)
python evaluate_tts_output.py --audio /path/that/does/not/exist
```

---

## Q&A for v0.4.2

**Q: Can I evaluate non-WAV files?**  
A: Not automatically. The script looks for `*.wav` extensions. Convert first or modify the glob pattern in the source.

**Q: What's the batch size limit?**  
A: Tested up to 500 files. Memory usage is O(1) per file, so larger batches work fine. Speed is O(n).

**Q: Do I need to retrain?**  
A: No. v0.4.2 is purely tooling improvements. Existing checkpoints work unchanged.

**Q: How do I integrate this into my workflow?**  
A: Great question. After each `train_vits.py` run:

```bash
# Generate samples
python inference_vits.py --checkpoint checkpoints/best_model.pt --text "test" --output test.wav

# Evaluate all samples from this epoch
python evaluate_tts_output.py --audio ./samples/epoch_050

# See results immediately
```

---

## Conclusion

v0.4.2 is a **quality-of-life release**. No core functionality changed, but the tools are much more pleasant to use. Code is clearer, evaluation is faster, and you can iterate without spawning 100 shell sessions.

Focus was on **developer experience**, not raw capability. That comes next.

---

_Patch notes by: Someone who got tired of running evaluate_tts_output.py 50 times_  
_Tested on: RTX 3050 Ti, 5GB synthetic audio dataset_  
_Peer reviewed by: My own regret_
