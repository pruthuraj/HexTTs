# HexTTS Inference Evaluation Report: "Why Does My Robot Sound Like a Broken Kazoo?"

**Date:** 04.04.2026  
**Model:** HexTTS (VITS Architecture with Vibes and Attitude)  
**Checkpoint:** `checkpoints/best_model.pt` (a.k.a. the "best" one, jury still out)  
**Test Subject:** A sentence that took way too long to pronounce correctly

---

## The Grand Objective

We shoved a number into our neural network baby and asked: "Can you say this without sounding like a robot having an existential crisis?"

The ancient knobs we turned to torture our model:

- `duration_scale` — How long should each sound be? (_Answer: WRONG, but less wrong_)
- `noise_scale` — How much randomness should we add? (_Answer: sometimes a little helps, sometimes it's chaos_)

| Parameter      | What It Actually Does                                      |
| -------------- | ---------------------------------------------------------- |
| duration_scale | Makes the speech faster or slower (spoiler: we want slow)  |
| noise_scale    | Controls how "creative" the model gets (creativity = bugs) |

---

## The Unlucky Test Sentence

```
we are at present concerned
```

Translation: **"I am deeply concerned about your life choices"**

Why this sentence?

- It's got **20 phonemes** (plenty of suffering opportunities)
- It's long enough to hear the model panic
- It's relatable (we're all concerned about something)

---

## The Experimental Torture Chamber

We tested three configurations, essentially asking: "If we twist the knobs differently, will it sound less like a chainsaw?"

| Test  | duration_scale | noise_scale | Optimism Level  |
| ----- | -------------- | ----------- | --------------- |
| test1 | 2.0            | 0.10        | Cautiously High |
| test2 | 3.0            | 0.10        | Medium Hope     |
| test3 | 3.0            | 0.05        | This Is It      |

---

## The Results (Brace Yourself)

### Test 1: "The Auctioneering Approach"

| Metric             | Value                                |
| ------------------ | ------------------------------------ |
| Duration           | 1.149 s (way too fast)               |
| Mel Frames         | 100 (compressed like a panic attack) |
| RMS Energy         | 0.192 (barely there)                 |
| Silence Ratio      | 0.044 (mostly noise, barely silence) |
| Zero Crossing Rate | 0.480 (BZZZZZZZZZZ)                  |

**What Happened:**

- The model rushed through the sentence like it was running late for an appointment
- Duration predictor said "I'll just yeet this out real quick"
- Buzzed like a confused hornet trapped in a speaker
- Sounded like a fast-talking game show host who lost his mind

**The Brutal Honest Verdict:**

```
"It spoke! ...Too fast. Fix it."
```

**Emotional Status:** Disappointment with a side of buzzing tinnitus

---

### Test 2: "The Slightly Less Chaotic Comedy"

| Metric             | Value                                      |
| ------------------ | ------------------------------------------ |
| Duration           | 1.846 s (finally, a real sentence!)        |
| Mel Frames         | 160 (adequate for human consumption)       |
| RMS Energy         | 0.181 (it's trying, I swear)               |
| Silence Ratio      | 0.045 (almost like real speech!)           |
| Zero Crossing Rate | 0.488 (still buzzing but with more poetry) |

**What Happened:**

- The model finally took a breath and said the actual sentence
- Duration prediction worked! Sort of! Ish!
- Still has that metallic Griffin-Lim robot voice (Griffin-Lim: when your vocoder went to film school instead of engineering school)
- Sounds like a robot who learned English but still has an accent

**The Honest Verdict:**

```
"This is actually... listenable? (with earplugs)"
```

**Emotional Status:** Cautious optimism mixed with ambient buzzing

---

### Test 3: "The Golden Goose (Or So We Thought)"

| Metric             | Value                                     |
| ------------------ | ----------------------------------------- |
| Duration           | 1.846 s (consistency! growth! evolution!) |
| Mel Frames         | 160 (same rhythm as Test 2, but cleaner)  |
| RMS Energy         | 0.141 (quieter... or is it just tired?)   |
| Silence Ratio      | 0.055 (embracing actual silence)          |
| Zero Crossing Rate | 0.482 (less angry than Test 2)            |

**What Happened:**

- We turned `noise_scale` down and the model chill'd out
- Fewer random artifacts = less chaotic
- RMS Energy dropped (model sounds less aggressive, more depressed)
- The buzzing is still there, but it's... _poetic buzzing_

**The Enlightened Verdict:**

```
"This is the least-worst option. Declare victory."
```

**Emotional Status:** "This is fine" _ambient fire in background_

---

## The Brutal Comparison (Side-by-Side Suffering)

| Metric        | Test 1 (Speed Racer) | Test 2 (The Buzzer)   | Test 3 (The Winner)  |
| ------------- | -------------------- | --------------------- | -------------------- |
| Duration      | **Way too short**    | **Perfect**           | **Perfect**          |
| Mel Frames    | **Panic mode**       | **Good**              | **Good**             |
| Noise Level   | **Moderate chaos**   | **High chaos**        | **Acceptable chaos** |
| Voice Quality | **Auctioneer**       | **Robot on steroids** | **Tired robot**      |
| Overall Vibe  | **"NO"**             | **"Maybe?"**          | **"Fine. I guess."** |

---

## The "Best" Configuration (We Promise This Works)

```
duration_scale = 3.0
noise_scale = 0.05
```

If you want that exact suffering experience, run this:

```bash
python inference_vits.py \
  --checkpoint checkpoints/best_model.pt \
  --config vits_config.yaml \
  --text "we are at present concerned" \
  --output best_test.wav \
  --duration_scale 3.0 \
  --noise_scale 0.05
```

_(Try at your own risk. Keep headphones volume low.)_

---

## What We Actually Learned (The Silver Linings)

**The Good News:**

- Model actually produces sound (groundbreaking)
- Sentence duration is realistic-ish (it's not just 0.5 seconds anymore)
- The audio isn't completely silent (we have energy!)
- Duration predictor is learning (slowly, painfully, but learning)
- Mel-spectrogram alignment makes sense

**The Suffering Remains:**

- **Zero Crossing Rate = 0.48** = "There's still a swarm of angry bees in this audio"
- **Griffin-Lim vocoder** = "We couldn't afford a neural vocoder so we used 2005 technology"
- Model sounds like a robot who went to voice acting class and _failed_
- Still very much in "training wheels" phase

---

## Current Model Status: Early Synthesizer Baby

```
Stage: "Learning how to talk but sounds like a detuned synthesizer"
```

Right Now:

- Knows what words are
- Knows roughly how long to say them
- Sounds like a sci-fi movie villain
- Can't pronounce anything without robotic artifacts
- Probably just wants to be left alone

---

## The Training Gauntlet (What's Coming)

| Epoch Range | What the Model Is Doing                             |
| ----------- | --------------------------------------------------- |
| 1–20        | "Wait, I'm supposed to learn DURATIONS too??"       |
| 20–40       | "Okay I might be saying actual words now"           |
| 40–80       | "You know what? This sounds almost... human-ish"    |
| 80–100      | "I am the voice of the future" (narrator: it's not) |

Current checkpoint is stuck in **"Duration Panic Mode"** (which is fine, this is normal, we swear).

---

## What's Next (The Suffering Must Continue)

1. **Keep training** until the model stops sounding like a Speak & Spell from 1987
2. **Monitor duration loss** to see if it stops exploding
3. **Generate samples periodically** and pretend they're getting better
4. **Replace Griffin-Lim** with an actual neural vocoder (so it sounds like 2024 instead of 2004)
5. **Therapy for listening to 100 robot voices** (available upon request)

---

## Final Damage Report

```
Inference pipeline:         WORKING (in the loosest sense)
Duration modeling:          SLOWLY IMPROVING
Speech quality:              ROBOTIC BUT PERSISTENT
User eardrum safety:         QUESTIONABLE
Overall vibes:              JAZZY (if by jazzy you mean "chaotic")
Training progress:          HAPPENING (we think)
```

**Conclusion:** HexTTS is successfully saying words in a length-appropriate timeframe. Whether those words are _pleasant_ is a question for philosophers and brave users with noise-canceling headphones.

---

## The Uncomfortable Truth

Your neural TTS is a baby. Right now it's a baby that can walk, kinda, but sounds like a robot stuck in a harmonic oscillation. Give it time. Or give it a better vocoder. Honestly, both would be nice.

---

_End of evaluation report. Therapeutic Tea Recommended._

**P.S.** — If you're listening to these audio files and your family asks what that sound is, you can now confidently say: "It's machine learning. No, it doesn't get better. Yes, I'm aware. Please leave me alone."
