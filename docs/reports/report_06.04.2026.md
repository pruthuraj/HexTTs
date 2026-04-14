# HexTTS Training Log — 06.04.2026

## Epoch 17: When Physics Fights Back

After two days of relentless GPU abuse, we've reached **Epoch 17 of 100**.

Current status:
The model is learning. The duration predictions are **exploding**. Literally.

---

# Training Overview

| Component       | Value                                 |
| --------------- | ------------------------------------- |
| Current Epoch   | 17/100                                |
| Training Loss   | 64.2558 (Epoch 16)                    |
| Validation Loss | 166.9800 (Epoch 16)                   |
| Hours Trained   | ~27+ hours                            |
| Emotional State | "Why is it so angry about durations?" |

---

# The Duration Explosion Phenomenon ™

```
Batch 104/3108:
[Epoch 17: 3%] loss=49.2426, recon=0.2526, kl=0.2309
Skipping unstable batch due to duration explosion
Skipping unstable batch due to duration explosion
Skipping unstable batch due to duration explosion
```

**What's happening?**

- The model's duration predictor is going full **Hiroshima**
- Instead of predicting "this phoneme takes 100ms", it predicts "∞"
- GPU says: "Nope."
- Training says: "Skip that one, buddy."

**Technical explanation:** Duration predictions are hitting NaN/Inf values, causing gradient explosions.

**Emotional explanation:** The model is having a bad day.

---

# Epoch 16 Summary (The Good One)

Before things got spicy, Epoch 16 completed successfully:

| Metric              | Value      |
| ------------------- | ---------- |
| Training Loss       | 64.2558    |
| Validation Loss     | 166.9800   |
| Validation Accuracy | 100%       |
| Batches Processed   | 164/164    |
| Duration            | 12h00m00s  |
| Speed               | 13.48 it/s |

**Plot twist:** Validation loss went **up** after reaching 166.9800. This is either:

A) Normal regularization behavior ✓
B) The model is learning to overfit ✗
C) The universe is testing us ??

We're going with **A** because it feels better.

---

# Checkpoint Graveyard

We now have checkpoints every 500 steps. Because paranoia:

```
./checkpoints/best_model.pt              ← The golden child
./checkpoints/checkpoint_step_064000.pt  ← Latest save
./checkpoints/checkpoint_step_064500.pt  ← Brand new baby
```

Plus ~128 other checkpoints from previous epochs.

**Storage:** Taking up half the hard drive at this point.

**Justification:** "What if we need to go back?"

---

# Warnings of the Day

```
Skipping unstable batch due to duration explosion
Skipping unstable batch due to duration explosion
Skipping unstable batch due to duration explosion
```

**Frequency:** Getting spicy in Epoch 17

**Cause:** Duration predictor is learning the concept of "infinity"

**Solution:**

- Reduce learning rate? Maybe.
- Adjust duration prediction loss weight? Perhaps.
- Sacrifice a small GPU to the training gods? Definitely.

---

# GPU Status Report

```
Temperature: (Spicy™)
VRAM: 3.9GB / 4GB (Living on the edge)
Fans: (Sounding like a small aircraft)
Thermals: ACCEPT FATE
```

---

# Next Steps

1. Monitor Epoch 17 vs 18 for stability
2. Watch for more "duration explosion" warnings
3. Consider adjusting duration loss weight if things don't settle
4. Drink more coffee
5. Contemplate the universe

---

# Mood Board

```
Model confidence:  (sort of)
Validation performance: (confusing)
Duration prediction sanity: (concerning)
My sanity: (roller coaster)
```

Stay spicy, HexTTS.
