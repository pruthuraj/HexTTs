# HexTTs Patch Notes v0.4.1

**Release Date:** April 6, 2026  
**Previous Version:** v0.4.0 (April 4, 2026)  
**Severity:** Critical (sample generation was completely broken)

---

## The Dramatic Tagline

> "v0.4.0 said 'let me log audio samples' and v0.4.1 said 'actually, let me NOT throw an AttributeError'"

---

## What Got Broken and How We Fixed It (A Tragic Tale)

### 🔧 **The Great API Name Confusion** (CRITICAL)

**The Crime:** `utils/sample_generation.generate_samples()` was calling `model.infer()` but VITS model only knows `model.inference()`.

**The Victim:** Every 5 epochs of training, a beautiful `AttributeError` would bloom:

```
AttributeError: 'VITS' object has no attribute 'infer'
```

**Why This Matters:**

- Sample generation would crash and burn faster than a GPU with thermal paste made of butter
- TensorBoard logging would silently give up (no error, just... nothing)
- You'd see your model training, losing hope, and you'd get ZERO audio samples to evaluate
- "Is my model learning or just hallucinating?" — you'd never know

**The Fix:** Renamed one method call. That's it. Honest.

```python
# What v0.4.0 tried (and failed spectacularly):
audio = model.infer(x, x_lengths)[0]

# What works (because VITS actually exposes this):
audio = model.inference(x, x_lengths)[0]
```

**Files Destroyed and Rebuilt:**

- `utils/sample_generation.py` (line 27) — one character away from losing your mind

---

### 🎨 **The Mel-Spectrogram Audio Apocalypse**

**The Stupidity:** We were logging 80-channel mel-spectrograms to TensorBoard as if they were audio waveforms.

**Why This Was Bad:**

- An 80-channel "audio" file isn't audio — it's a sick cry for help
- TensorBoard would either choke trying to process it or log complete garbage
- You'd open TensorBoard expecting to hear your model learning to speak... and hear nothing but digital screaming

**The Reality Check:**

```python
# v0.4.0: "Let's pass a (80, 1500) tensor to add_audio()"
# TensorBoard: "That's... not audio. That's a war crime."
# User: "Why is my TensorBoard empty?"
```

**The Redemption:** We switched to visualization that actually works.

```python
# Before (criminal negligence):
self.writer.add_audio(
    tag=f"sample_audio_{i}",
    snd_tensor=audio,  # Surprise! It's actually a mel-spectrogram
    global_step=epoch,
    sample_rate=self.config.get("sample_rate", 22050),
)

# After (respectable engineering):
self.writer.add_image(
    tag=f"sample_mel_{i}",
    img_tensor=mel,  # Nice heatmap visualization
    global_step=epoch,
)
```

**What Changed:**

- Method still called `log_audio_samples()` (naming is hard, okay?)
- Switched from `add_audio()` to `add_image()` (because mel-spectrograms are... images)
- Renamed variable from cryptic `audio` to honest `mel`
- Updated docstring so you know what you're actually looking at

**Files That Needed Therapy:**

- `train_vits.py` (lines 303-331) — the source of much confusion

---

### 📊 **The TensorBoard Namespace Chaos**

**The Embarrassment:** Learning rate was logged as just `'lr'` instead of fitting in nicely with `'train/lr'`.

**The Consequence:**

- TensorBoard groups metrics by namespace (the part before the `/`)
- LR was hanging out alone in the _other group_ like a friendless metric
- Documentation said `train/lr` but code said `lr`
- Users opened their dashboard and asked "where'd the learning rate go?"

**The Petty Fix:**

```python
# Before (antisocial):
self.writer.add_scalar('lr', self.optimizer.param_groups[0]['lr'], self.global_step)

# After (teamwork):
self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], self.global_step)
```

Adding **one forward slash** fixed the family reunion. That's it.

**Files Touched:**

- `train_vits.py` (line 240) — the fix that changed everything by changing nothing much

---

### 📝 **The Config File's Existential Crisis**

**The Problem:** `max_duration_loss` was sitting under **INFERENCE SETTINGS** pretending to be an inference parameter.

**The Confusion Matrix:**

- Users: "Do I set this for inference?"
- Documentation: "This prevents training NaN explosions"
- Users: "...So only during training?"
- Config: "I'm chilling in the INFERENCE section tho"
- Everyone: _confused screaming_

**The Reality:**

- `max_duration_loss` **only matters during training**
- It's a safety guard, not an inference knob
- It shouldn't live with sampling temperature and inference steps

**The Relocation:**

- Moved from: **INFERENCE SETTINGS** (where it didn't belong)
- Moved to: New **TRAINING STABILITY** section (where it makes sense)
- Same value: `300` (still works, just organized better)

**Files Reorganized:**

- `vits_config.yaml` (lines 169-180) — Marie Kondo would be proud

---

## How to Get This Fix Into Your Life

### Option 1: The Lazy Way

```bash
# Just pull the latest
git pull origin main
```

No retraining. No data preparation. Just pull and run.

### Option 2: The Paranoid Way

```bash
# Download fresh
git clone https://github.com/...  # (if this was on GitHub)
```

Still no retraining needed. Code fixes only.

---

## Verify It Actually Works

```bash
# Start training (sample generation happens at epoch 5)
python train_vits.py \
    --config vits_config.yaml \
    --num-epochs 10

# Expect:
# ✅ At epoch 5: No AttributeError
# ✅ No silent failures in sample logging
# ✅ Sample generation prints "Saved audio samples for epoch 5"
```

**Check TensorBoard:**

```bash
tensorboard --logdir=logs
```

**Look for:**

1. **IMAGES tab** — `sample_mel_*` entries (pretty heatmaps, not error messages)
2. **SCALARS tab** — `train/lr` grouped with other `train/*` metrics (no lonely `lr` tag)
3. **No crash at epoch 5** — the bar we lowered until we could finally limbo under it

---

## Known Issues Left Unfixed

None. We broke everything and fixed it all.

(If you find more bugs, send them to the GitHub issues, or just suffer silently like the rest of us.)

---

## What's Next (v0.5.0 Fantasy Checklist)

- **Vocoder Integration** — Convert mel-spectrograms to actual audio without Griffin-Lim torture
- **Real TensorBoard Audio** — Hear your model's predictions, for better or worse
- **Better Dashboards** — Graphs that don't make you cry

---

## TL;DR (The Suffering in One Paragraph)

v0.4.0 broke sample logging in two ways: (1) called a non-existent method (`model.infer` instead of `model.inference`), and (2) treated 80-channel mel-spectrograms as audio. We also logged learning rate under the wrong namespace and put a training-only parameter in the inference section. v0.4.1 fixes all four with minimal changes because the underlying design was right — the engineer just had a bad day. Pull the latest, run training, and enjoy seeing your TensorBoard actually work.

---

## Refer to These If You're Still Confused

- `CHANGELOG.md` — The official record of your poor decisions
- `PHASE3_QUICKSTART.md` — Getting started (hopefully with v0.4.1)
- `vits_config.yaml` — Now organized way better if you look closely
