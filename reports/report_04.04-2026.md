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

# Final Status

Training pipeline: **working**
GPU: **still alive**
Model: **learning**

Mission continues tomorrow.

---

_End of today's battle with neural networks._
