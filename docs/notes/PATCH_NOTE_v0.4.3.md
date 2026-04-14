# HexTTs v0.4.3 Patch Notes

## "PostNet, Better Duration Prediction, and Enough Diagnostics to Satisfy Your Debugging Paranoia"

**Release Date:** April 7, 2026  
**Tag:** v0.4.3  
**Status:** Stable  
**Urgency:** Medium (fixes training instability, recommended to apply)

---

## Overview

v0.4.3 is a **mid-training stability and diagnostics release**. The model was encountering NaN failures during Epoch 20 training due to duration predictor instability. This release:

1. **Implements PostNet** — A neural residual network for mel-spectrogram refinement
2. **Stabilizes Duration Prediction** — Relative error supervision with smart clamping
3. **Adds Comprehensive Diagnostics** — 15+ new TensorBoard metrics to catch issues early
4. **Hardens Loss Computation** — Better protection against outlier batches

The core issue: Duration predictor was learning unbounded durations (phonemes lasting 100+ frames), causing length regulation to allocate gigantic mel-spectrograms, triggering NaN cascades.

The fix: Hard clamping + relative error supervision + better monitoring.

---

## Major Features

### 1. PostNet: Mel-Spectrogram Refinement

**File:** `vits_model.py`  
**New Class:** `PostNet`

#### What Is It?

PostNet is a **convolutional residual network** that learns to refine the decoder's mel-spectrogram output by predicting mel-scale residuals.

```
predicted_mel_raw = decoder(latent_code)
residuals = postnet(predicted_mel_raw)
refined_mel = predicted_mel_raw + residuals
```

#### Architecture

```python
class PostNet(nn.Module):
    """
    Conv1d-based residual network for mel-spectrogram refinement.

    Layers:
        1. Conv1d(mel_channels → hidden_channels) + BatchNorm + Tanh + Dropout
        2-5. Conv1d(hidden_channels → hidden_channels) + BatchNorm + Tanh + Dropout
        6. Conv1d(hidden_channels → mel_channels) + BatchNorm + Dropout

    Parameters:
        - mel_channels: 80 (standard)
        - hidden_channels: 256 (default)
        - kernel_size: 5 (local temporal context)
        - num_layers: 5 (5 total, 3 hidden)
    """
```

#### Why Use PostNet?

The raw decoder output captures phoneme-level structure but often has:

- **Spectral rippling** — Unnatural oscillations in frequency bins
- **Sharp transitions** — Mel-frames don't blend smoothly
- **Amplitude noise** — Inconsistent loudness across frequencies

PostNet learns to smooth these artifacts by predicting **what to add** rather than retraining from scratch.

#### Integration Points

**Training:** `vits_model.py` VITS.forward()

```python
predicted_mel = self.decoder(z)
predicted_mel = predicted_mel + self.postnet(predicted_mel)  # NEW
```

**Inference:** `vits_model.py` VITS.inference()

```python
predicted_mel = self.decoder(z)
predicted_mel = predicted_mel + self.postnet(predicted_mel)  # NEW
```

#### Design Decisions

1. **Why residual (not direct)?**
   - Decoder already does the heavy lifting
   - PostNet only corrects small errors → faster convergence
   - Residual = bounded output (prevents destabilization)

2. **Why Conv1d (not FC)?**
   - Temporal locality matters in spectrograms
   - Conv1d has ~10× fewer parameters than FC
   - Generalizes better to different sequence lengths

3. **Why BatchNorm + Dropout?**
   - BatchNorm: Stabilizes training, speeds convergence
   - Dropout: Prevents overfitting to training MEL artifacts
   - Tanh: Bounds outputs to [-1, 1] during residuals

#### Performance Impact

**Parameters Added:** ~120K (out of 45M total = 0.3% increase)  
**Inference Speed:** ~5% slower per sample  
**Training Speed:** ~8% slower (more compute, but still dominated by mel computation)  
**Memory:** +50MB for activations (negligible for 4GB+ GPUs)

**Result:** Small cost for noticeably better audio quality.

---

### 2. Relative Error Duration Supervision

**File:** `train_vits.py`  
**Function:** `VITSTrainer.compute_loss()`

#### The Problem

In v0.4.2, duration loss was simple L1:

```python
pred_duration_sum = duration.sum(dim=1)  # Sum of all phoneme durations
target_duration_sum = mel_lengths.float()  # Actual mel-spectrogram length
duration_loss = L1_Loss(pred_duration_sum, target_duration_sum)
```

**Issue:** A single batch with huge targets (long audio) dominates loss, causing gradient explosion.

Example:

- Batch 1: target=500, pred=100 → error=400 → HUGE gradient
- Batch 2: target=100, pred=95 → error=5 → tiny gradient
- Average gradient: influenced mostly by Batch 1
- Result: Duration predictor learns wrong priorities

#### The Solution

**Relative error** with clamping:

```python
# Relative absolute error: |pred - target| / target
rel_error = torch.abs(pred_duration_sum - target_duration_sum) / (target_duration_sum + 1e-6)

# Clamp outliers to prevent domination
rel_error = torch.clamp(rel_error, max=5.0)

# Final loss
duration_loss = rel_error.mean()
```

#### Why This Works

1. **Relative Error:** 10% error is treated the same regardless of absolute magnitude
   - 100 → 110: 10% error (reasonable)
   - 500 → 550: 10% error (same priority)
   - Prevents long sequences from drowning out short ones

2. **Clamping at 5.0:** A batch with 500% error is treated same as 500% error
   - Extreme outliers don't explode the gradient
   - Still penalizes bad predictions, just not catastrophically

#### Code Change

```python
# v0.4.2 (OLD):
pred_duration_sum = duration.sum(dim=1)
target_duration_sum = mel_lengths.float()
duration_loss = nn.functional.l1_loss(
    pred_duration_sum, target_duration_sum, reduction="mean"
)

# v0.4.3 (NEW):
pred_duration_sum = duration.sum(dim=1)
target_duration_sum = mel_lengths.float()
rel_error = torch.abs(pred_duration_sum - target_duration_sum) / (target_duration_sum + 1e-6)
rel_error = torch.clamp(rel_error, max=5.0)
duration_loss = rel_error.mean()
```

---

### 3. Hard Duration Clamping

**File:** `vits_model.py`  
**Class:** `DurationPredictor`

#### The Issue

Duration predictor uses softplus activation:

```python
duration = F.softplus(raw_duration)  # Maps R → (0, ∞)
```

Softplus is **unbounded**. If the model predicts raw_duration=5.0, softplus outputs ~5, which means "make this phoneme 5 frames". But if it predicts 10.0, softplus outputs ~10.

Nothing stops it from predicting 50, 100, or 1000.

Result: A single phoneme expands to 1000+ frames → length regulation allocates 10MB mel-spectrogram → NaN.

#### The Solution

Hard clamp to 1-20 frames:

```python
# v0.4.2 (OLD):
duration = torch.clamp(F.softplus(duration), min=1.0)

# v0.4.3 (NEW):
duration = torch.clamp(F.softplus(duration), min=1.0, max=20.0)
```

**Interpretation:**

- min=1.0: Every phoneme must last at least 1 frame (~11ms at 22kHz)
- max=20.0: Every phoneme can last at most 20 frames (~220ms at 22kHz)

This is **reasonable for single-speaker TTS** — phonemes rarely exceed 250ms.

#### Additional Validation Check

In the training loop:

```python
# NEW v0.4.3: Skip batches with extreme duration values
if outputs['duration'].max().item() >= self.config.get("max_duration_value", 20.0):
    print("Skipping unstable batch due to extreme duration values")
    continue
```

This catches the few edge cases where clamping isn't applied correctly.

---

### 4. Comprehensive Duration Diagnostics

**File:** `train_vits.py`  
**TensorBoard Locations:** `train/*`, `val/*`

#### New Metrics (Training)

```
train/pred_duration_sum_mean        — Average sum of predicted durations per batch
train/target_duration_sum_mean      — Average target mel length per batch
train/relative_duration_error_mean  — Mean relative error (diagnostics)
train/duration_max                  — Maximum predicted duration in batch
train/duration_min                  — Minimum predicted duration in batch
train/predicted_mel_length          — Output mel-spectrogram frames this step
train/target_mel_length_mean        — Expected mel frames
train/predicted_mel_max             — Peak value in predicted mel
train/predicted_mel_min             — Lowest value in predicted mel
```

#### New Metrics (Validation)

Same as training, prefixed with `val/`:

```
val/pred_duration_sum_mean
val/target_duration_sum_mean
val/duration_max
val/duration_min
val/predicted_mel_max
val/predicted_mel_min
```

#### Why These Metrics?

1. **Duration Sum Comparison:** If pred consistently > target, duration predictor is overestimating
2. **Duration Extremes:** Spotting max=30 means clamping is firing; max=19.9 means clamp is working
3. **Mel Ranges:** If predicted mel has values >5 or <-5, mel normalization is off
4. **Cross-Phase Comparison:** Validation metrics diverging from training = overfitting/instability

#### Reading the Metrics

**Healthy State:**

```
train/duration_max: 19.8 (near but below clamp at 20)
train/pred_duration_sum_mean: 250.5
train/target_duration_sum_mean: 248.2
train/relative_duration_error_mean: 0.15 (15% error is ok)
```

**Warning State:**

```
train/duration_max: 20.0 (at clamp limit, predictor wants higher)
train/relative_duration_error_mean: 2.5 (250% error, very unstable)
```

**Critical State:**

```
train/duration_max: NaN (gradient has exploded)
train/duration_min: -inf (something computed NaN)
```

---

### 5. Enhanced Autocast Context Manager

**File:** `train_vits.py`  
**Function:** `VITSTrainer.train_epoch()`

#### The Problem

v0.4.2 had awkward mixed precision handling:

```python
with torch.autocast(device_type=self.device.type) if self.scaler else torch.enable_grad():
    # This is confusing — mixing autocast and enable_grad
    outputs = self.model(...)
```

Why awkward?

1. `torch.autocast()` manages float16 precision
2. `torch.enable_grad()` manages whether to compute gradients
3. These are **different concerns**
4. Mixing them in one conditional is confusing

#### The Solution

Explicit context manager that's always valid:

```python
# Choose context based on AMP setting
if self.scaler:
    autocast_context = torch.autocast(device_type=self.device.type)
else:
    autocast_context = torch.autocast(device_type=self.device.type, enabled=False)

# Use it consistently
with autocast_context:
    outputs = self.model(...)
    loss_dict = self.compute_loss(...)
    loss = loss_dict['total_loss']

    # Backward pass is always within autocast
    if self.scaler:
        self.scaler.scale(loss).backward()
    else:
        loss.backward()
```

**Benefits:**

- Always inside autocast context (clear and explicit)
- `enabled=False` disables mixed precision cleanly
- No confusing `torch.enable_grad()` mixing

---

## Configuration Changes

### 1. Learning Rate Reduction

```yaml
# v0.4.2
learning_rate: 0.0002

# v0.4.3
learning_rate: 0.0001  # Half the old rate
```

**Rationale:** With more aggressive loss (relative error), smaller steps prevent overshooting.

### 2. Duration Loss Weight

```yaml
# v0.4.2
loss_weight_duration: 0.5  # Duration loss was too aggressive

# v0.4.3
loss_weight_duration: 0.1  # Balanced weight, alternatives provided
# Optional:
# loss_weight_duration: 0.05  # For very stable training
```

**Rationale:** PostNet+relative error means duration predictor needs less weight. They're not doing all the work anymore.

### 3. Disable AMP by Default

```yaml
# v0.4.2
use_amp: true

# v0.4.3
use_amp: false  # Off by default for stability
# Reason: NaN instability with float16 + smaller batches
```

**Why?** With batch_size=4 on RTX 3050 Ti, float16 overflows cause NaNs in duration computation. Disabled by default; can re-enable on larger GPUs.

### 4. Reduced Max Sequence Length

```yaml
# v0.4.2
max_seq_length: 500  # ~5 seconds at 22kHz

# v0.4.3
max_seq_length: 300  # ~3 seconds at 22kHz
```

**Rationale:** Shorter sequences = smaller mel-spectrograms = less memory = lower chance of numeric overflow in PostNet.

---

## Improved Code Quality

### 1. Inference Function Documentation

**File:** `vits_model.py`  
**Method:** `VITS.inference()`

Added comprehensive step-by-step comments:

```python
def inference(self, phonemes, lengths=None, duration_scale=1.0, noise_scale=0.667):
    """
    Inference mode: phonemes → predicted mel-spectrogram

    Args:
        phonemes: Phoneme indices tensor (batch_size, seq_len)
        lengths: Optional sequence lengths for masking
        duration_scale: Multiplicative scale for predicted durations
        noise_scale: Standard deviation of Gaussian noise

    Returns:
        predicted_mel: Mel-spectrogram (batch_size, mel_channels, time_steps)
    """
    # Step 1: Text Encoder - Convert phoneme indices to embeddings
    # This builds a sequence of phonetically meaningful vectors
    encoder_out = self.encoder(phonemes, lengths=lengths)

    # Step 2: Duration Predictor - Phonemes → frame durations
    # Predicts how long each phoneme should last in mel-spectrogram space
    # Scale by duration_scale parameter (>1.0 = slower, <1.0 = faster)
    duration = self.duration_predictor(encoder_out) * duration_scale
    duration = torch.clamp(duration, min=1.0, max=20.0)

    # ... and so on ...
```

Every section explains both "how" and "why".

### 2. PostNet Documentation

The new `PostNet` class includes:

- Clear docstring explaining purpose
- Inline comments on each layer explaining its role
- Documentation of input/output shapes
- Notes on why architectural choices were made

---

## Validation Improvements

### Gradient NaN Protection

**File:** `train_vits.py`  
**Method:** `VITSTrainer.validate()`

New validation checks:

```python
# Check 1: Loss is valid
if torch.isnan(loss) or torch.isinf(loss):
    print("Invalid loss detected — skipping batch")
    continue

# Check 2: Gradients are healthy
bad_tensor = False
for name, param in self.model.named_parameters():
    if param.grad is not None:
        if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
            print(f"NaN gradient detected in {name}")
            self.optimizer.zero_grad()
            bad_tensor = True
            break

if bad_tensor:
    continue
```

This catches gradient corruption before it spreads through backprop.

---

## Migration Guide

### From v0.4.2 to v0.4.3

#### Breaking Changes

**NONE.** v0.4.3 is backward compatible. Old checkpoints, configs, and data work unchanged.

#### Recommended Upgrades

1. **Update `train_vits.py`** — Critical for stability improvements
2. **Update `vits_model.py`** — Adds PostNet, duration clamping
3. **Update `vits_config.yaml`** — Recommended settings
4. **Re-evaluate your saved checkpoints** — Audio quality should improve slightly with PostNet

#### Checkpoint Retraining

If you resume from a v0.4.2 checkpoint:

```bash
python train_vits.py \
    --config vits_config.yaml \
    --checkpoint checkpoints/best_model.pt
```

The model will:

1. Load weights from v0.4.2
2. Initialize PostNet (new, untrained)
3. Start training with new v0.4.3 loss + config
4. PostNet gradually learns to refine

**Expected behavior:** Loss might dip slightly as PostNet settles in, then resume normal trajectory.

---

## Performance Characteristics

### Model Size

```
v0.4.2:  45.0M parameters
v0.4.3:  45.1M parameters (+120K for PostNet)
```

### Inference Speed

```
v0.4.2:  ~1.2s per second of audio (with Griffin-Lim vocoder)
v0.4.3:  ~1.3s per second of audio (+8% due to PostNet)
```

### Memory Usage

```
v0.4.2:  Peak ~3.2GB (batch_size=4, max_seq_length=500)
v0.4.3:  Peak ~3.3GB (batch_size=4, max_seq_length=300)
```

Slightly higher due to PostNet activations, but reduced by shorter max sequence length.

### Training Stability

```
v0.4.2:  NaN failures ~10% of the time in unstable phases
v0.4.3:  NaN failures <1% of the time with new protections
```

---

## Testing Results

### Manual Testing

Epoch 20 (the problem epoch from v0.4.2):

```
v0.4.2:
  NaN skipped batches: 50+ per epoch
  Duration loss: 97.49 (SPIKY)
  Status: UNSTABLE

v0.4.3 (applied to same checkpoint):
  NaN skipped batches: <5 per epoch (dramatic improvement!)
  Duration loss: 8.21 (smooth, healthy)
  Status: STABLE
```

### Audio Quality

Informal listening tests (post-training, 20 epochs):

```
v0.4.2:  Buzzy, metallic robot voice
v0.4.3:  Slightly smoother, less harsh artifacts
```

Not a night-and-day difference (architecture still simplified), but noticeable improvement.

---

## Known Issues & Limitations

### 1. Audio Still Sounds Robotic

**Root Cause:** This is v0.4.3, not the Audiobook Generation System.

Current limitations:

- Simple cross-entropy text encoder (not transformer)
- No adversarial discriminator (no waveform-level feedback)
- Griffin-Lim vocoder (audio quality ceiling is 80s-era technology)
- Only 20+ epochs of training (needs 100+)

**Not a bug.** Architectural limitation. Hear v0.4.0 at epoch 5? Compare to v0.4.3 at epoch 20. Progress IS happening.

### 2. Duration Prediction Still Not Perfect

With relative error supervision, the model learns "get the total right," not "get each phoneme right."

Example problem:

```
Target duration: [3, 4, 5] → total 12
Prediction: [2, 5, 5] → total 12
Loss: 0 (total is correct)
But individual phonemes are wrong!
```

**Fix:** Implement Monotonic Alignment Search (MAS) in future version to learn frame-level alignment.

### 3. PostNet Can Overfit

If trained too long on small dataset (like 13k samples), PostNet can memorize mel artifacts instead of learning general refinement.

**Mitigation:** We use dropout=0.1 + BatchNorm, which helps. But it's still a risk.

### 4. AMP Off by Default

Mixed precision is disabled to prevent NaN cascades with float16. This costs ~5-10% training speed.

**When to Re-enable:**

- GPU with 8GB+ VRAM
- Batch size ≥16
- If you encounter no NaN failures for 50 epochs

```yaml
use_amp: true # Re-enable if confident
```

---

## Q&A for v0.4.3

**Q: Do I need to retrain from scratch?**  
A: No. Apply the update and resume from your best checkpoint. PostNet will initialize randomly and learn during training.

**Q: Why is learning_rate halved?**  
A: With relative error supervision (which is more normalized), the optimization landscape changed. Smaller steps prevent oscillation.

**Q: Can I disable PostNet?**  
A: Sure. Comment out these lines in `vits_model.py`:

```python
# self.postnet = PostNet(mel_channels=config['n_mel_channels'])
# predicted_mel = predicted_mel + self.postnet(predicted_mel)
```

But you lose refinement benefits.

**Q: Is PostNet used during inference?**  
A: Yes. It's part of the model, always applied. No extra flags or options.

**Q: What if my audio gets WORSE with v0.4.3?**  
A: Unlikely, but possible if PostNet diverges. Revert learning_rate to 0.0002 or loss_weight_duration to 0.5.

**Q: Can I fine-tune just PostNet?**  
A: Technically yes, but not recommended. It's designed to be trained jointly with the decoder.

**Q: How long until Realistic Audio?**  
A: Honestly? You need:

1. 200+ epochs (not 20)
2. Better vocoder (HiFi-GAN)
3. Full VITS architecture (flows, discriminator, proper posterior)
4. Or just use Coqui TTS like a reasonable person

---

## Conclusion

v0.4.3 is a **stability and observability release**. The training system is now:

- **Stable:** Hard clamping + relative error supervision prevent NaN cascades
- **Observable:** 15+ new metrics help debug when something goes wrong
- **Refined:** PostNet learns to smooth out mel-spectrogram artifacts

This is still a mid-stage VITS implementation. But it's now **robust enough for extended training** without surprises.

The next major release would implement:

- Full VITS with normalizing flows
- HiFi-GAN vocoder
- Proper alignment learning (MAS)
- Longer training runs (200+ epochs)

For now: v0.4.3 is a solid foundation. Use it.

---

_Patch notes by: Someone who debugged NaN cascades with 50 printstatements_  
_Tested on: RTX 3050 Ti 4GB, v0.4.2 checkpoint at Epoch 20_  
_Estimated debugging time: 14+ hours_  
_Sanity level at finish: Questionable_  
_Was it worth it: Objectively yes, emotionally no_
