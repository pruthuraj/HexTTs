# PATCH NOTE — Vocabulary Size Consistency Fix

### "The Case of the 149 Phantom Phonemes"

---

## The Problem

Once upon a time in `vits_model.py`, a mysterious line existed:

```
VOCAB_SIZE = 149
```

No one remembers why.
No one remembers who wrote it.
But it sat there… silently judging the rest of the code.

Meanwhile in `vits_data.py`, the **actual phoneme vocabulary** was defined like a responsible adult:

```
PHONEME_TO_ID = {...}
VOCAB_SIZE = len(PHONEME_TO_ID)
```

This meant the **model thought it knew 149 phonemes**,  
while the **dataset only knew ~40**.

So the neural network basically had **109 imaginary phonemes**.

---

## Root Cause

The embedding layer inside the model was built for **149 tokens**, but the dataset only used about **40 real phonemes**.

Nothing crashed.  
Nothing exploded.  
Training still ran.

But internally the model was like:

> "Cool cool cool… but where are the other 109 sounds??"

This caused:

- unused embedding rows
- confusing debugging
- inconsistent architecture definitions
- developers staring at the code like

# after 4 hours of training checked the best model

the output : trash not even 1 sec audio for "hello i am pruthu"
proof in tts_output/

---

## The Fix

We **removed the cursed hardcoded number** from `vits_model.py`.

Instead the model now imports the vocabulary size **directly from the dataset**, like a civilized program:

```
from vits_data import VOCAB_SIZE
```

Now the flow finally makes sense:

```
dataset phonemes
      ↓
dataset vocabulary
      ↓
model embedding size
```

No guessing.  
No phantom phonemes.  
No developer headaches.

---

## Result

After this fix:

✔ Model embedding matches dataset phoneme mapping  
✔ No unused embedding slots  
✔ Cleaner architecture definitions  
✔ Easier debugging  
✔ Slightly fewer existential crises

The neural network now knows **exactly how many sounds exist in its universe.**

---

## Files Modified

```
vits_model.py
```

---

## Training Impact

**None.**

You can still run:

```
python train_vits.py --config vits_config.yaml --device cuda
```

Your GPU will still suffer exactly the same amount.

---

## Recommendation

For speech models:

> **The dataset vocabulary is the source of truth.**

Never trust hardcoded values.  
They age like milk.

---

## Moral of the Story

If your neural network thinks there are **149 phonemes**,  
but your dataset only has **40**,

your model is basically **hallucinating sounds**.

And we already have LLMs doing that.  
Let's not teach TTS models to hallucinate too.

---

_Last updated: Patch Day_
