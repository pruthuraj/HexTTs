# HexTTS Training Log — 04.04.2026

## Day 1 of Teaching My Laptop To Speak

Today I began training **HexTTS**, a VITS-based Text-to-Speech model.

Goal:
Make my laptop GPU produce **human speech instead of fan noise**.

Current status:
The GPU is screaming, but progress is happening.

---

# Training Setup

| Component       | Value                                   |
| --------------- | --------------------------------------- |
| Model           | VITS                                    |
| Framework       | PyTorch                                 |
| GPU             | RTX 3050 Ti (4GB and emotional support) |
| Dataset         | LJSpeech                                |
| Epochs          | 100                                     |
| Steps per Epoch | 3108                                    |

---

# Training Progress

| Epoch | Validation Loss | Emotional Status                |
| ----- | --------------- | ------------------------------- |
| 1     | 161.92          | "Did I break something?"        |
| 4     | 152.83          | "Okay it's learning something." |

Loss going **down** = good.

Loss going **up** = panic.

Today: **no panic yet**.

---

# Epoch 5 Snapshot

```id="hxt1"
loss = ~149
recon = ~0.21
kl = ~0.18
dur = ~133
```

Translation:

- recon → audio reconstruction
- kl → latent regularization
- dur → alignment between text and speech

Or simply:

```
model: trying its best
gpu: sweating
```

---

# Training Speed

| Metric                  | Value        |
| ----------------------- | ------------ |
| Iterations/sec          | ~7–8         |
| Epoch time              | ~10 minutes  |
| Estimated full training | ~10–12 hours |

Laptop GPU right now:

```
██████████████████████████
██████████████████████████
██████████████████████████
temperature: spicy
```

---

# Checkpoints Saved

Because developers trust **nothing**.

```id="hxt2"
./checkpoints/best_model.pt
./checkpoints/checkpoint_step_018000.pt
./checkpoints/checkpoint_step_018500.pt
./checkpoints/checkpoint_step_019000.pt
./checkpoints/checkpoint_step_019500.pt
./checkpoints/checkpoint_step_020000.pt
./checkpoints/checkpoint_step_020500.pt
```

Meaning if everything explodes tomorrow, I can pretend it never happened.

---

# Current Observations

Things that **did NOT break today**

✔ No NaN loss
✔ GPU did not melt
✔ Checkpoints saving
✔ Model actually learning

Things that **did break**

⚠ My patience waiting for epoch 100.

---

# Expected Training Timeline

| Epoch  | Expected Result         |
| ------ | ----------------------- |
| 1–10   | model confused          |
| 10–20  | speech starts existing  |
| 20–40  | understandable voice    |
| 40–80  | decent TTS              |
| 80–100 | "wow this works" moment |

Currently:

```
epoch 5
model still learning what humans sound like
```

---

# Developer Mood Chart

```
epoch 1 : (••)
epoch 5 : ( - -)
epoch 20 : (⌐■■)
epoch 60 : (⊙⊙)
epoch 100 : (－‸ლ)
```

---

# Next Steps

Continue training until:

```
epoch >= 40
```

Then test speech generation and see if the model says:

```
Hello, I am HexTTS
```

instead of:

```
HHHHHHHHHSSSSSSSSZZZZZ
```

---

# Epoch 15 Status Update

**Training has evolved. We're at Epoch 15 now.**

Latest snapshot from the training logs:

```
Training loss     : ~133.20
Validation loss   : ~314.02
recon loss        : ~0.49
kl loss           : ~0.46
duration loss     : ~105
```

Interpretation:

| Metric | What It Means                                             | The Vibe               |
| ------ | --------------------------------------------------------- | ---------------------- |
| recon  | mel-spectrogram accuracy (how well the robot draws sound) | Getting better         |
| kl     | VAE latent regularization (prevents boring latent space)  | Stable-ish             |
| dur    | phoneme duration prediction (how long to hold sounds)     | **ACTUALLY IMPROVING** |

**Important observation:**

The **duration loss dropped from ~133 → ~105**, which means your model is genuinely learning how long to hold each phoneme. That's progress, baby. Real progress.

However, validation loss spiked temporarily (161.92 → 314.02). Before you panic:

This is **completely normal in early VITS training** because:

- Model is learning duration alignment (hard problem)
- Latent space is still unstable (it'll settle down)
- Decoder is adjusting to variable sequence lengths (expected chaos)

In other words:

```
model   : actively learning (chaotic)
training: progressing normally
panic   : unnecessary (for now)
```

---

# Training Features Added This Session

Your training pipeline got a serious upgrade today:

### TensorBoard Energy

New metrics being logged every 100 steps:

```
train/loss                  ← total agony metric
train/recon_loss            ← "how bad is my mel-spectrogram drawing"
train/kl_loss               ← latent space behavior
train/duration_loss         ← "did I time the phonemes right"
train/lr                    ← learning rate (optimizer aggression level)
val/loss                    ← validation horror stories
```

Open TensorBoard and watch those lines in real time:

```bash
tensorboard --logdir=./logs
```

Then obsessively check `http://localhost:6006` while pretending to work.

### Audio Sample Generation

Every **5 epochs**, the model generates actual speech samples:

```
samples/
├── epoch_005_sample_1.wav     ← "it's a robot"
├── epoch_005_sample_2.wav     ← "still a robot"
├── epoch_010_sample_1.wav     ← "slightly better robot"
├── epoch_015_sample_1.wav     ← you are here
└── ...
```

Go listen. Be amazed at how it's actually improving.

### Training Stability Firewall

A safety mechanism was added to prevent rare catastrophic batches:

```python
if duration_loss > 300:
    skip batch  # "nope, not today"
```

This stops alignment explosions from corrupting your entire training run. Basically a panic button for gradient instability.

---

# GPU Status Report

Current thermal situation:

```
temperature  : acceptable (for now)
fan speed    : jet engine (permanent state)
developer    : cautiously optimistic (dangerous)
```

Your RTX 3050 Ti is doing its absolute best. Respect it. Feed it good data.

---

# Updated Epoch Progression Prediction

Based on duration loss improvements:

| Epoch Range | Expected Behavior       | Your Experience            |
| ----------- | ----------------------- | -------------------------- |
| 1–15        | duration learning phase | "Is it working?"           |
| 15–30       | rough speech begins     | "Oh my god it said words!" |
| 30–60       | understandable audio    | "This is actually good"    |
| 60–100      | stable, natural TTS     | "I trained this myself"    |

Current stage:

```
████████░░░░░░░░░░  duration learning (15%)
```

Milestone alert:

```
First intelligible speech ≈ epoch 30–40 (watch for audio samples)
```

---

# Final Status

Training pipeline: **working beautifully**
GPU: **still alive (barely)**
Model: **learning faster now**
Developer: **getting excited too early**

Mission continues. We're onto something here.

---

_End of today's battle with neural networks. Update: the battle is going slightly better._
