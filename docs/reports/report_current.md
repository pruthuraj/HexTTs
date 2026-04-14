# HexTTS Training Report — The "We Nuked 80GB Of Checkpoints" Edition

## Day N: When Storage Space Became The Real Enemy

Today I performed the digital equivalent of throwing out 99% of my checkpoints.

**Storage freed:** 80GB
**Emotional impact:** Immense
**Regret factor:** Moderate to severe

Current status:
Epoch 19 completed successfully. Epoch 20 said "hold my beer" and went absolutely **unhinged**.

---

# The NaN Invasion of Epoch 20

## A Brief Timeline of Chaos

**Epoch 19:** Model goes BRRRR with speed. Loss = 64.5293, duration = 80ms, everything cool and chill.

**Early Epoch 20:** Model enters its chaotic phase like a teenager discovering ChatGPT.

**The Pattern:**

```
NaN detected — skipping batch
NaN detected — skipping batch
NaN detected — skipping batch
```

Repeat this **AT LEAST 50 TIMES IN ONE EPOCH**.

**Frequency:** Every 3-4 batches, like clockwork.

**Duration loss:** Now at 97.4914 (compare to Epoch 19's civilized 80.0699)

**Translation:** The duration predictor discovered infinity and forgot how to math.

---

## The Good News vs The Oh-Crap News

### What The Model Still Has Going For It

- Model weights: **Numerically still a thing**
- Training loop: **Hasn't exploded yet**
- Epoch completion: **Actually finished**
- KL divergence: **Chilling at 0.23 (still healthy)**
- Losses: **Still finite, not NaN across the board**

### What's Actively On Fire

- Duration predictor: **Discovered rocket fuel**
- Batch discard rate: **Getting spicy**
- Learning efficiency: **Speedrunning into a wall**
- Model confidence: **In the dumpster**
- Developer's sanity: **questionable**

---

## The Validation Assessment (The Silver Lining)

**Validation loss:** 141.2224

This is honestly not terrible considering how chaotic training was.

**What this tells us:**

- Model still learned something useful
- Network isn't completely destroyed
- The insanity is mostly during training, not validation
- The model might actually produce usable inference (surprising!)

**Important note:** Just because the model CAN produce audio doesn't mean it SHOULD be allowed to continue training unsupervised.

---

# The Doctor's Verdict

| Symptom          | Diagnosis              |
| ---------------- | ---------------------- |
| Model Survival   | YES (but barely)       |
| Training Health  | WEAK (coffee-fueled)   |
| Duration Module  | BROKEN (like really)   |
| Usefulness Today | QUESTIONABLE AT BEST   |
| Can It Speak?    | YES BUT WITH A STUTTER |

**Final diagnosis:** This is a surviving-but-unstable training run.

**Not dead. Not healthy. Somewhere in between.**

---

# The Action Plan (AKA Getting Out Of This Mess)

## Step 1: STOP THE BLEEDING

Stop Epoch 20 immediately. The repeated NaN skipping combined with duration loss going BALLISTIC (97.49 WHAT) means continuing is just throwing GPU time into the void.

**Decision:** Funeral for this run. RIP Epoch 20.

## Step 2: Triage The Checkpoints

Keep only:

- `best_model.pt` (the favorite child)
- `checkpoint_step_075500.pt` (the last remotely stable one)

Delete the rest. Free up that glorious 80GB (oh wait, already done).

## Step 3: Test The Survivor

Run inference on the saved checkpoint. Check if the speech is still understandable. If it's 50% intelligible, that's a win.

## Step 4: Resume Strategy

**Do NOT resume from the unstable checkpoint.**

Resume from:

- The latest stable checkpoint BEFORE the NaN storm
- Or just use `best_model.pt` for inference and restart from an actually safe checkpoint

## Step 5: Prevent This From Happening Again

### The TL;DR Recipe For Stability

1. **Lower the learning rate by half** (0.0002 → 0.0001. Or lower.)
2. **Disable AMP temporarily** (mixed precision was too spicy)
3. **Clamp those durations harder:** `torch.clamp(F.softplus(duration), min=1.0, max=20.0)`
4. **Strengthen gradient clipping:** Keep it at 0.5 or even lower
5. **Filter the weird samples:** No 10-minute audiobooks in the training set
6. **Lower max sequence length:** 500 tokens → 300 tokens

## Step 6: Find The Culprit Samples

The NaNs are clustered, which means specific batches are toxic. Log which ones cause skips. You're looking for:

- Text that's absurdly long
- Audio that doesn't match text
- Corrupted wav files
- Silence storms
- Whatever makes the duration predictor cry

## Step 7: The Watchdog Number

**Monitor duration loss closely during next training.**

If it exceeds ~100 and NaN skipping returns frequently, **STOP AGAIN**.

Your duration predictor is the weakest link right now.

---

# The Inference Test Results

Time to see if the model learned anything useful.

**Test audio:** "we are at present concerned"

### Running The Model

```bash
python inference_vits.py --checkpoint checkpoints/best_model.pt \
  --config vits_config.yaml --text "we are at present concerned" \
  --output test1.wav --duration_scale 2.0 --noise_scale 0.10

python inference_vits.py --checkpoint checkpoints/best_model.pt \
  --config vits_config.yaml --text "we are at present concerned" \
  --output test2.wav --duration_scale 3.0 --noise_scale 0.10

python inference_vits.py --checkpoint checkpoints/best_model.pt \
  --config vits_config.yaml --text "we are at present concerned" \
  --output test3.wav --duration_scale 3.0 --noise_scale 0.05
```

### The Evaluation Gauntlet

```bash
python evaluate_tts_output.py --sample_rate 22050
```

**Found 3 .wav file(s) to evaluate**

---

## Audio Quality Report Card

### test1.wav (Duration Scale: 2.0, Noise: 0.10)

| Metric             | Value      |
| ------------------ | ---------- |
| Duration           | 1.6022 s   |
| Peak Amplitude     | 0.999969   |
| RMS Energy         | 0.197007   |
| Silence Ratio      | 0.040789   |
| Zero Crossing Rate | 0.492679   |
| Mel Mean dB        | -2.8007 dB |

**Verdict:** If you don't listen too carefully, this sounds like speech. If you DO listen carefully... it's still speech, just... buzzy. Like an angry bee with a voice box.

---

### test2.wav (Duration Scale: 3.0, Noise: 0.10)

| Metric             | Value      |
| ------------------ | ---------- |
| Duration           | 2.5078 s   |
| Peak Amplitude     | 0.999969   |
| RMS Energy         | 0.138344   |
| Silence Ratio      | 0.058377   |
| Zero Crossing Rate | 0.495612   |
| Mel Mean dB        | -2.7920 dB |

**Verdict:** Slightly less energetic than test1. Duration is more realistic. Still has the metallic hum of 10,000 angry bees having a rave.

---

### test3.wav (Duration Scale: 3.0, Noise: 0.05)

| Metric             | Value      |
| ------------------ | ---------- |
| Duration           | 2.5078 s   |
| Peak Amplitude     | 1.000000   |
| RMS Energy         | 0.148271   |
| Silence Ratio      | 0.055990   |
| Zero Crossing Rate | 0.491742   |
| Mel Mean dB        | -2.9268 dB |

**Verdict:** Lower noise scale, but still sounds like a swarm of robot crickets had a concert inside your ear.

---

## The Batch Summary

| File      | Duration | RMS   | ZCR   | Vibe                   |
| --------- | -------- | ----- | ----- | ---------------------- |
| test1.wav | 1.602s   | 0.197 | 0.493 | Angry metallic robot   |
| test2.wav | 2.508s   | 0.138 | 0.496 | Sleepy angry robot     |
| test3.wav | 2.508s   | 0.148 | 0.492 | Sleepy angry robot Jr. |

---

## The Ugly Truth About The Audio

### High Zero Crossing Rate (ZCR ≈ 0.49)

This is VERY high for human speech. Natural voices sit around 0.10-0.15. Ours is approaching 0.5.

**What this means:** The waveform oscillates rapidly between positive and negative values.

**In English:** BZZZZZZZZZZZZZZZZZZZZ

### Low Spectral Variance

```
Mel std ≈ 0.57–0.77
```

The spectrum is flatter than your laptop keyboard.

**What this means:** All frequencies are equally loud (which is not how human speech works).

**In English:** Sounds like white noise shaped vaguely like words.

---

# Why Is It Still Buzzy? (The Mystery Deepens)

After implementing safety measures (duration clamping, reduced loss weight, lower learning rate) the buzz persists.

**Three main culprits:**

## A: The Alignment Problem

The model predicts duration but doesn't actually LEARN true phoneme timing. It just matches total duration to mel length. That's like learning to cook by throwing ingredients at the wall until something sticks.

**Solution:** Implement proper alignment (Monotonic Alignment Search - MAS).

## B: Your Architecture Is Simplified

This isn't a full VITS with:

- Normalizing flows
- Adversarial waveform discriminator
- Full posterior encoding

It's VITS-lite. Like store-brand cereal.

**Solution:** Implement proper VITS components or give up and use someone else's model (coward's choice).

## C: The Vocoder Is The Real Problem

You're generating mel spectrograms then converting to waveform. The vocoder (waveform converter) is likely introducing most of the noise.

**Solution:** Use a better vocoder (HiFi-GAN, UnivNet, etc.) or train your own.

---

# What I Changed To Try To Fix It

## Config Modifications

```yaml
# Duration Predictor Clamping
from: duration = torch.clamp(F.softplus(duration), min=1.0)
to: duration = torch.clamp(F.softplus(duration), min=1.0, max=20.0)

# Loss weight reduction (duration was too aggressive)
loss_weight.duration: 0.05 → 0.005

# Learning rate cut in half
learning_rate: 0.0002 → 0.0001

# Sequence length limit
max_seq_length: 500 → 300
```

## Checkpoint Resume

Resumed from `checkpoint_step_064000.pt` instead of continuing from the unstable region.

**Result:** Training continued, NaNs reduced, but the buzz remained.

**Conclusion:** The buzz is architectural, not numerical.

---

# The Big Picture Assessment

## What's Actually Working

- Text to phoneme pipeline: Check
- Phoneme encoding: Check
- VITS forward pass: Check
- Mel spectrogram generation: Check
- Basic audio output: Check

## What's Broken-ish

- Natural acoustic quality: Nope
- Realistic duration prediction: Trying its best
- Waveform synthesis: Sounds like angry aliens
- Overall audio realism: 4/10 (generous)

---

# The Bottom Line

**This model IS operational, but NOT high quality YET.**

Status report:

- Generates complete, consistent audio: YES
- Maintains numerically stable training: KIND OF
- Produces intelligible speech: MAYBE
- Sounds natural: LOL NO

The HexTTS system is in **early beta stage**. It's past the "making noise" phase and into the "making recognizable sounds" phase.

Next target: "making sounds that don't trigger the uncanny valley."

---

# The Real Next Steps

1. **Implement proper VITS** (flows, posterior, full discriminator)
2. **Use a better vocoder** (HiFi-GAN minimum)
3. **Train for way longer** (100k-200k steps minimum)
4. **Improve dataset quality** (clean up LJSpeech preprocessing)
5. **Implement MAS** (proper alignment learning)

Or just accept that local TTS is hard and use Coqui TTS like a normal person.

---

# Conclusion

The HexTTS training saga continues. We've successfully:

- Found the problem (duration explosion)
- Fixed the symptom (clamping and reduced learning)
- Exposed the real problem (architectural limitations)
- Generated audio that is technically sound waves

Not bad for a Wednesday.

**Next episode:** "Can we make it sound like actual speech instead of an angry synthesizer?"

Stay tuned.
