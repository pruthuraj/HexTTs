# HexTTs: The Robot That Finally Learned to Speak

_v0.4.7_

### _"Because Your GPU Wasn't Hot Enough Yet"_

---

## TL;DR (For Busy Humans)

- Want speech now: run inference with HiFi-GAN.
- Want less buzz: check `Spectral flatness` in `scripts/evaluate_tts_output.py`.
- Want speed: use `vits_data_cached.py` and precompute features.
- Want peace: keep backups of `checkpoints/best_model.pt` before experimenting.

---

## Quick Navigation (So You Don't Scroll Forever)

- [Installation](#installation-the-suffering-begins)
- [Simplified Main Flow](#simplified-main-flow-less-typing-more-talking)
- [Data Prep](#data-prep-okay-but-why-is-your-dataset-like-this)
- [Training](#training-lets-make-your-room-hot)
- [Inference](#expected-console-output-during-inference)
- [HiFi-GAN Flags](#optional-hifi-gan-vocoder-recommended)
- [Buzz Metric](#buzz-metric-spectral-flatness)
- [A/B Comparison](#quick-ab-griffin-lim-vs-hifi-gan)
- [Continuation Test](#continuation-test-3-epoch-automation)
- [Q&A](#qa-will-you-answer-my-questions)

---

## Patch Notes (v0.4.7)

### What's New

- **HiFi-GAN is now first-class in inference**: pass `--vocoder_checkpoint` + `--vocoder_config` and stop pretending Griffin-Lim is modern.
- **Official checkpoint compatibility fixed**: the local vocoder wrapper now matches the official HiFi-GAN state dict layout.
- **Better buzz diagnostics**: `Spectral flatness` is now a documented quality signal with practical thresholds.
- **A/B workflow documented**: run the same sentence through Griffin-Lim and HiFi-GAN and compare like an adult scientist.
- **Continuation test automation added**: `scripts/run_continuation_test.py` now runs resume-train + diagnostics + HiFi-GAN inference + evaluation in one go.
- **Hybrid duration path restored and verified**: rolled back the failed phoneme-aware branch and confirmed healthy duration behavior again.
- **Duration debug checks added**: optional `duration_debug_checks` prints target/pred vectors and sums for one train/val sample.
- **Continuation report export added**: `--report-file` writes training snapshots + full eval report + final summary to text.

### Real Talk Result

In current tests, the restored hybrid duration path produced stable timing again (`~1.8576 s`) with clean waveform metrics (`ZCR ~0.115`, `flatness ~0.016`).
Translation: less metallic mosquito energy, realistic timing, and no duration collapse drama.

---

## What Is This Monstrosity?

HexTTs is a **Text-to-Speech (TTS)** project that teaches an AI neural network to convert boring text into spoken words. It's basically teaching a computer to be a voice actor, except it won't complain about late night shoots or demand residuals.

- **Trained on**: 13,100 audio clips of a very patient woman (LJSpeech dataset)
- **Powers**: The ability to type "hello world" and actually hear your computer SAY it
- **Side effects**: Your GPU fans will sound like a jet engine, your electricity bill will cry, and you'll start explaining mel-spectrograms at parties.

If confusion reaches critical mass, check the `diagram/` and `doc/` folders before declaring the project haunted.

---

## The Project Structure (What's All This Junk?)

```
HexTTs/
├── train_vits.py              ← The actual sorcery happens here
├── inference_vits.py          ← "Please make sounds from my text"
├── tts_app.py                 ← Interactive mode (for people who hate command lines)
│
├── vits_model.py              ← The neural network brain (45 million parameters btw)
├── vits_data.py               ← Data loading (it's surprisingly boring)
├── vits_data_cached.py        ← Data loading, but make it fast (new and improved suffering)
├── scripts/view_spectrogram.py← Visualize mel spectrograms (stare at pretty graphs)
│
├── vits_config.yaml           ← "How angry should my GPU get?"
├── requirements.txt           ← All the suffering, listed as pip packages
├── CHANGELOG.md               ← A record of your poor decision-making over time
│
├── prepare_data.py            ← "Let me fix the phonemes because the dataset was messy"
├── validate_dataset.py        ← Quality control (spoiler: data is weird)
├── precompute_features.py     ← Computes mel spectrograms ahead of time so training doesn't die slowly
├── scripts/test_setup.py      ← Sanity check that all your dependencies installed correctly
├── scripts/audit_dataset.py   ← Dataset filtering and cleanup detective
├── scripts/evaluate_tts_output.py ← Audio quality analyzer (batch + single file)
│
├── checkpoints/               ← Model snapshots (save the good ones, delete the tragic ones)
├── logs/                      ← TensorBoard metrics (watch your loss go brrr)
├── tts_output/                ← The fruits of your GPU's labor
│
├── diagram/                   ← Visual documentation (pretty pictures of data flow)
├── doc/                       ← Extra documentation you'll read later (spoiler: you won't)
├── notes/                     ← Patch notes, setup guides, lessons learned the hard way
├── deprecated/                ← Old code graveyard (abandon hope, all ye who enter here)
├── scripts/                   ← Main-flow wrappers (less typing, fewer CLI faceplants)
│
└── data/
    ├── LJSpeech-1.1/          ← 13,100 voice samples (24GB of pure audio patience)
    └── ljspeech_prepared/     ← "The cleaned up version" (+ cached/ if you're smart)
```

---

## Installation: "The Suffering Begins"

### Step 1: Create Virtual Environment

```bash
# Build a protective bubble around yourself from the Python chaos
python -m venv venv
venv\Scripts\activate        # Windows (the suffering OS)
source venv/bin/activate     # Linux/Mac (the enlightened paths)
```

### Step 2: Install PyTorch (CUDA Edition — The ONLY Way)

```bash
# This is the way. The ONLY way. Don't use CPU PyTorch unless you hate:
# - Time
# - Productivity
# - Your GPU (which would be sad)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

VERIFY IT HAS CUDA POWERS:

```bash
python -c "import torch; print(torch.cuda.is_available())"
# Output: True     ← You are enlightened ✓
# Output: False    ← You installed the depression version, start over ✗
# Output: Error    ← Something has gone very wrong
```

### Step 3: Install The Entire Ecosystem

```bash
pip install -r requirements.txt
# This will:
# 1. Download 47 different things you've never heard of
# 2. Compile 3 of them from source (patience required)
# 3. Mysteriously resolve 2 circular dependencies
# 4. Ask "are you SURE?" in version conflicts (you are, just say yes)
```

The Dependency Grimoire:

- **librosa**: Understands audio better than you ever will. Also slower than continental drift.
- **g2p_en**: Translates "hello" → "HH EH L OW". Somehow that's relevant to TTS. Don't ask.
- **numpy**: Exists in every single Python project ever. Probably powers your dreams too.
- **matplotlib**: For beautiful loss curves that will make you weep. Especially the upward ones.
- **tensorboard**: Real-time suffering visualization. Watch your metrics live. Find out about failures in real-time instead of later.

---

## Simplified Main Flow (Less Typing, More Talking)

The old flow used long commands with lots of flags. That was technically correct and emotionally exhausting.

Now use:

```bash
python scripts/main_flow.py <command> [options]
```

Main commands:

- `train`: run training with default config
- `infer`: synthesize one sentence (optional `--hifigan`)
- `eval`: evaluate one file or a folder of wav files
- `audit`: filter and score dataset metadata files
- `compare`: run Griffin-Lim vs HiFi-GAN + evaluate both
- `continuation-test`: run the 3-epoch continuation workflow with diagnostics and final HiFi-GAN report

Examples:

```bash
# Train
python scripts/main_flow.py train --device cuda

# Inference (Griffin-Lim fallback)
python scripts/main_flow.py infer --text "hello world" --output tts_output/hello_gl.wav

# Inference (HiFi-GAN)
python scripts/main_flow.py infer --text "hello world" --hifigan --output tts_output/hello_hifigan.wav

# Evaluate one file
python scripts/main_flow.py eval --audio tts_output/hello_hifigan.wav

# Audit dataset without writing outputs
python scripts/main_flow.py audit --dry-run

# Full A/B comparison in one command
python scripts/main_flow.py compare --text "we are at present concerned"

# Full continuation test in one command
python scripts/main_flow.py continuation-test --epochs 3
```

If you still enjoy manually typing 400-character commands, the original scripts are still available.

### Continuation Test (3-Epoch Automation)

If you want the full checkpoint continuation experiment without babysitting terminals:

```bash
venv\Scripts\python.exe scripts/run_continuation_test.py --epochs 3
```

This script will:

- create an auto continuation config
- resume from a checkpoint
- extract latest duration diagnostics from TensorBoard
- run HiFi-GAN inference on the fixed sentence
- run objective evaluation (duration, ZCR, spectral flatness, verdict)

---

## Data Prep: "Okay But Why Is Your Dataset Like THIS?"

```bash
# Step 1: Download LJSpeech-1.1 (13GB of vocal patience)
# From: https://keithito.com/LJ-Speech-Dataset/
# (Warning: takes FOREVER. Go get coffee. A lot of coffee.)

# Step 2: Place it here (exactly):
# data/LJSpeech-1.1
# (The script is VERY particular about folder names)

# Step 3: Validate the dataset isn't cursed
python validate_dataset.py ./data/LJSpeech-1.1
# This checks for:
# ✓ Corrupted audio files
# ✓ Mismatched transcripts
# ✓ Ancient curses
# ✓ Digital demons (probably won't find any but who knows)

# Step 4: Convert words into phoneme sorcery
python prepare_data.py ./data/LJSpeech-1.1 ./data/ljspeech_prepared
# This is where the magic happens (or the chaos, depending on your perspective)
```

**What's Actually Happening Here:**

The script reads 13,100 voice transcripts and converts them into phoneme sequences. Why? Because neural networks think in phonemes, not words. It's like speaking in a secret code that only AI understands.

Output files:

```
train.txt        ← 12,445 training examples (the A-team)
val.txt          ← 655 validation examples (the B-team)
phoneme sequences ← Looks like: P R UW M O D EL IZ T R AY N D
                   Translation: ???
                   Actually Works: Yes, inexplicably
```

---

## Speed Optimization: "WAIT, This Can Train 3x FASTER?!"

**YES.** Here's the secret: By default, the training loop cooks mel-spectrograms fresh from raw audio _every single batch_. This is the computational equivalent of baking a new loaf of bread every time you want a sandwich.

Solution: Bake all the bread once, then just grab slices.

```bash
python precompute_features.py --config vits_config.yaml
# This will:
# 1. Load all 13,100 audio files (takes forever)
# 2. Convert to mel-spectrograms (takes more forever)
# 3. Cache them to disk (takes even MORE forever)
# 4. Be ABSOLUTELY WORTH IT (training speeds up 3x)
```

Creates:

```
data/ljspeech_prepared/cache/
    mels/    ← Your bread factory (computed once, used forever)
    ids/     ← Reference numbers (which slice is which)
```

### Activate the Speed Boost

In `train_vits.py`, make this ONE change:

```python
#  SLOW MODE (don't use this, it's suffering)
from vits_data import create_dataloaders

#  WARP SPEED (3x faster, enlightenment achieved)
from vits_data_cached import create_dataloaders
```

**Result:** Training becomes significantly faster. Your GPU actually gets to do GPU things instead of waiting for your CPU to finish its homework.

---

## Training: "Let's Make Your Room Hot"

```bash
# Start training (go get a date live your life or touch grass , it will take a while depending on your config)
python train_vits.py --config vits_config.yaml --device cuda
```

Recommended starting config (especially if you don't have a flagship GPU):

```yaml
batch_size: 4
num_workers: 2
use_amp: true # Automatic Mixed Precision — free speed, take it
grad_clip_val: 0.5 # Prevents your gradients from achieving escape velocity
```

### What Happens (A Dramatic Retelling):

1. Neural network receives a phoneme sequence
2. Network predicts a mel-spectrogram (fancy picture of sound)
3. Compares to the real spectrogram from the dataset
4. Cries about how wrong it was (loss calculation)
5. Updates 45 million weights to be slightly less wrong
6. Repeats 10,000+ times
7. **Eventually**: Makes acceptable robot sounds

### Expected Timeline:

| Time       | Audio Quality              | Your Mood                  |
| ---------- | -------------------------- | -------------------------- |
| Hour 0–1   | Garbage disposal simulator | Optimistic, possibly naive |
| Hour 4–8   | Vaguely human-shaped noise | Cautiously hopeful         |
| Hour 12–24 | Not bad, actually          | Smug                       |
| Hour 24+   | Genuinely good             | Please go to sleep         |

### Monitoring (The Loss Curve Staring Contest):

```bash
# In another terminal, watch the descent into correctness
tensorboard --logdir=./logs
# Open http://localhost:6006
```

**Key Metrics You'll Obsess Over:**

- **Total Loss**: Should go ↓ (good) not ↑ (alarming)
- **Reconstruction Loss**: How wrong the mel-spectrogram prediction is
- **KL Loss**: Prevents the network from totally cheating its way through training
- **Your Sanity**: Will also go ↓, but recovers post-training

---

## Inference: "Let Me Hear My Robot"

### Single Command:

```bash
python inference_vits.py `
  --checkpoint checkpoints/best_model.pt `
  --config vits_config.yaml `
  --text "Hello I am Pruthu" `
  --output hello.wav
```

Output lands in `tts_output/hello.wav`. Go listen to it. Marvel at your creation. Notice it sounds slightly robotic. Accept it. This is normal.

### Interactive Mode (For Impatient People):

```bash
python tts_app.py --checkpoint checkpoints/best_model.pt
```

Then type stuff at it:

```
> Hello world
> This is a neural text to speech system
> My voice is 45 million parameters of math
> exit
```

---

## Resume Training: "Oops, It Crashed"

If your training gets interrupted (power cut, GPU tantrum, cat walked on keyboard):

```bash
# See what checkpoints you have
dir checkpoints/

# Resume from the latest
python train_vits.py `
  --config vits_config.yaml `
  --checkpoint checkpoints/checkpoint_step_5000.pt
```

It's like saving in a video game. Except the game is teaching math to a robot and the save files are 200MB each.

---

## Common Problems & Completely Unsympathetic Solutions

### "CUDA out of memory"

```
Your GPU: "I'm full"
You: "But I only gave you one sentence!"
Your GPU: "TOO BAD. Reduce batch_size or leave me alone."
```

**Fix:** Open `vits_config.yaml`, set `batch_size` to 4, then 2, then 1 if you must. Yes, 1. We don't judge. (We judge a little.)

### "Loss is not decreasing"

Your learning rate is probably too aggressive. Like giving your neural network three espressos and telling it to do calculus.

```yaml
learning_rate: 0.001 # Decaf. Responsible. Adult.
```

### "The audio sounds robotic"

That's because it IS a robot. It needs more training. More training = more time = more electricity = more tears. Keep going.

### "ModuleNotFoundError: No module named 'vits_model'"

Did you download ALL the files? Did you put them in the SAME folder? Did you activate your virtual environment? Did you read the instructions even once?

### "My audio is 10 seconds but processing took a minute"

Welcome to Griffin-Lim. It converts spectrograms to audio using an iterative algorithm from 1984, and it shows. This is why HiFi-GAN exists (see: Future Improvements). For now: patience.

### "precompute_features.py ran but training is still slow"

Did you actually swap `vits_data` to `vits_data_cached` in `train_vits.py`? Go look. I'll wait.

---

## File Descriptions (TL;DR Edition)

| File                             | What It Actually Does (Honest Edition)                                     | When You'll Actually Care                                   |
| -------------------------------- | -------------------------------------------------------------------------- | ----------------------------------------------------------- |
| `train_vits.py`                  | 45M parameters of pure chaos learning to speak (with comments now!)        | Always. Forever. It haunts your nightmares                  |
| `inference_vits.py`              | "Please convert my text to robot sounds" (it tries its best, bless it)     | After training (or when you give up)                        |
| `tts_app.py`                     | GUI wrapper for people afraid of command lines (aka cowards, endearing)    | When you're tired of suffering in Terminal                  |
| `vits_model.py`                  | The actual neural network brain (touch it and it breaks, don't ask how)    | Never. Ever. Seriously. Don't.                              |
| `vits_data.py`                   | Original data loader (slow enough to watch paint dry)                      | During training (the masochist path)                        |
| `vits_data_cached.py`            | FAST data loader (seriously use this one instead of suffering)             | During training (the enlightened path)                      |
| `vits_config.yaml`               | "How much suffering do I want today?" (knobs to destroy your GPU)          | Before training (read it carefully)                         |
| `prepare_data.py`                | Converts Linda Johnson's words into phoneme soup (alphabet chaos)          | Once. Then pray it worked. Then run it again when it didn't |
| `validate_dataset.py`            | "Is my data broken?" (spoiler: yes, but in acceptable ways)                | When paranoid. Which is always.                             |
| `precompute_features.py`         | Pre-bakes mel spectrograms so training doesn't die (PLEASE RUN THIS)       | Before training (seriously, do it)                          |
| `scripts/view_spectrogram.py`    | Stare at pretty frequency heatmaps and pretend you understand them         | When debugging or procrastinating                           |
| `scripts/evaluate_tts_output.py` | Audio quality analyzer (single file or batch mode, now SUUUPER flexible)   | After inference (to judge your creation)                    |
| `scripts/test_setup.py`          | "Does CUDA actually exist or did I hallucinate it?" (verification script)  | On first setup (and sometimes at 3 AM)                      |
| `requirements.txt`               | All your dependencies (a Pandora's box of suffering)                       | During `pip install` (prepare for pain)                     |
| `CHANGELOG.md`                   | A historical record of your poor decisions over time                       | When mysteriously everything breaks                         |
| `checkpoints/`                   | Nuclear launch codes, but for neural networks (200MB each, delete wisely)  | Always. Treat these like your children.                     |
| `logs/`                          | TensorBoard metrics (watch your loss curve either ↓ or ↑ very ominously)   | During/after training (obsess over it)                      |
| `tts_output/`                    | Where your robot's voice lives (cherish it, it took $500 in electricity)   | After inference (go show your therapist)                    |
| `diagram/`                       | Pretty architecture diagrams (for impressing people who don't know better) | When explaining to non-AI people (futile)                   |
| `doc/`                           | Extended documentation (the README you never read, but should)             | When README isn't enough (it will be)                       |
| `notes/`                         | Patch notes, setup guides, lessons learned at 2 AM (your suffering diary)  | When troubleshooting (aka all the time)                     |
| `deprecated/`                    | Graveyard of old code (archaeology/horror museum combo)                    | Never. For. The. Love. Of. God. Never.                      |
| `scripts/`                       | Miscellaneous utility scripts (person who organized these: questionable)   | When you remember they exist (you won't)                    |

---

## Utils Directory & Module Breakdown

### _"The Junk Drawer of Helpful Functions"_

The `utils/` directory contains helper modules for the training & inference pipeline:

| Module                   | Purpose                                               | Key Functions                                                                                                                                             |
| ------------------------ | ----------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `sample_generation.py`   | Audio generation during training (the audition booth) | `generate_samples()` — produces test audio every N epochs so you can hear your robot learn without your GPU literally catching fire waiting for epoch 100 |
| `(other future modules)` | Additional utilities (TBD by future-you)              | Coming soon to a GitHub repo near you (maybe)                                                                                                             |

### What `sample_generation.py` Does

Every 5 epochs, the model gets a chance to perform:

1. Takes SAMPLE_TEXTS from `train_vits.py` (pre-written audition lines)
2. Converts text → phoneme IDs via `text_to_sequence_fn` (elegant alphabet soup)
3. Runs model inference to generate mel-spectrograms (draws pretty pictures of how sound should look)
4. Vocodes the mel-spectrogram to audio (Griffin-Lim slowly paints those pictures into actual sound)
5. Saves WAV files to `samples/` directory (your robot's karaoke collection)

Output format: `samples/epoch_005_sample_1.wav`, `samples/epoch_005_sample_2.wav`, etc.

**Why this matters:** You get live entertainment watching your model fail progressively less. Epoch 5 sounds like a malfunctioning alarm clock. Epoch 50 sounds like an alien attempting human speech. Epoch 100+ actually starts sounding human-ish. It's like watching a toddler learn to talk, except the toddler is 45 million parameters and costs $500 to train.

---

## Expected Outputs at Each Stage

### After Data Preparation

```
data/ljspeech_prepared/
├── train.txt                    ← ~12,445 lines (Linda Johnson's voice, phoneme-fied)
├── val.txt                      ← ~655 lines (the test set that judges your model)
└── metadata.json                ← Proof that you actually ran the scripts
```

**What train.txt looks like:**

```
LJ001-0001.wav|D IH S IZ AE N IH N T ER EH S T IH NG |...
LJ001-0002.wav|Y EH S T ER D EY W AZ AE W ES AH M ER...
...
```

Each line: `[audio_file_you_have]|[mysterious_phoneme_alphabet_soup]|[other stuff you don't care about]`

Pro tip: That first line says "This is an interesting" in the most complicated way possible. English phonetics are chaos.

### After Feature Precomputation

```
data/ljspeech_prepared/
└── cache/                       ← "The Pre-Baked Bread Aisle"
    ├── mels/                    ← ~13,100 .npy files (mel spectrograms, pre-computed)
    │   ├── LJ001-0001_mel.npy   ← Picture of sound #1
    │   ├── LJ001-0002_mel.npy   ← Picture of sound #2
    │   └── ...                  ← Pictures #3 through #13,100 (yes, all of them)
    └── ids/                     ← Bookmarks so you don't forget which picture is which
        ├── train_indices.npy
        └── val_indices.npy
```

**Total size:** ~2–4 GB (you just spent 30 minutes computing math so your GPU doesn't have to recompute it 100 times. This was the right call.)

### During Training

Outputs explode into existence across your hard drive:

```
checkpoints/                      ← "The Save File Museum"
├── checkpoint_step_001000.pt    ← Backup #1 (emergency life insurance)
├── checkpoint_step_002000.pt    ← Backup #2 (because I'm paranoid)
├── checkpoint_step_003000.pt    ← Backup #3 (someone's going to crash someday)
├── best_model.pt                ← The actual important one (gets lonely waiting)
└── ...                          ← 50+ more backups because hoarding = safety

logs/                            ← "The Obsession Records"
├── events.out.tfevents.*        ← TensorBoard files (quantified pain)
└── ...                          ← More metrics than any human can parse

samples/                         ← "The Robot's Karaoke Album"
├── epoch_005_sample_1.wav       ← Your robot's audition reel (adorably terrible)
├── epoch_005_sample_2.wav       ← Round 2 of adorable failure
├── epoch_010_sample_1.wav       ← Slightly less terrible now
└── ...                          ← A growing archive of improvement
```

**Checkpoint file size:** ~200 MB each (weight snapshots captured mid-training, each one a bet you might need it)

**Training logs:** Agonizing metrics logged every 100 steps:

- `train/loss` — Total failure metric (↓ means you're winning)
- `train/recon_loss` — "I drew a bad mel-spectrogram"
- `train/kl_loss` — Mathematical punishment for boring latent spaces
- `train/duration_loss` — "Your syllables are timing out"
- `lr` — How aggressively the optimizer is working (like espresso shots for your neural network)

**Sample audio quality progression (The Painful Timeline):**
| Epoch | Expected Quality | Your Reaction |
| ----- | --------------- | -------------- |
| 5 | White noise with structure | "Is it working?" |
| 20 | Vaguely human vowels | "Oh my god there's a voice!" |
| 50 | Recognizable speech (wobbly) | "It said actual words!" |
| 100+ | Clear speech (still robotic) | "Okay this is actually impressive" |

### After Full Training

```
checkpoints/best_model.pt          ← Your trophy (treat it well, it earned this)
logs/                              ← The complete tragedy and triumph timeline
samples/                           ← Audio snapshots showing your robot's journey from garbage to "not bad actually"
```

**Total training artifacts:** ~50–100 GB (you just bought yourself expensive storage for this achievement)

### After Inference

Running `inference_vits.py --text "Hello" --output hello.wav`:

```
tts_output/
├── hello.wav                   ← Proof your robot can talk (go show your friends)
└── hello_mel.npy               ← [Optional] Mel-spectrogram (for debugging your dreams)
```

**Audio specs:**

- Sample rate: 22,050 Hz (or whatever you configured during a moment of optimism)
- Duration: ~50–150 ms per phoneme (rough estimate, phonemes are surprisingly inconsistent)
- Channels: Mono (1 channel, because Linda only had one voice)
- Format: WAV (16-bit PCM, the most boring audio format ever invented)

**Expected file sizes:**

- Short sentence (5 words): ~50 KB (adorably tiny)
- Long sentence (20+ words): ~200 KB (still absurdly small for audio)

---

## Expected Console Output During Training

When you run `python train_vits.py --config vits_config.yaml --device cuda`, expect a wall of text like this:

```
Loading config from vits_config.yaml
Using GPU: NVIDIA RTX 3050 Ti

Initializing VITS model...
Using vocabulary size: 149
Total parameters: 45.2M              ← (Your GPU: awkward silence as it realizes what you're about to do)

Creating dataloaders...
Loading dataset from ./data/ljspeech_prepared
Loaded 12445 training samples        ← (Linda Johnson's voice, cloned 12,445 times)
Loaded 655 validation samples        ← (The samples that will judge your model harshly)

Starting training for 100 epochs...
Device: cuda
Total steps per epoch: 3111          ← (You will see this progress bar 3,111 times per epoch. Get comfortable.)

Initial warning summary before training:
--------------------------------------------------
(Silence means your data is probably okay. Probably.)
--------------------------------------------------

Epoch 1/100 - Training loss: 8.3421  ← (This is CATASTROPHICALLY bad. Celebrate. It gets better from here.)
Epoch 1/100 - Validation loss: 7.8932  ← (The test set agrees: this stinks)
Saved checkpoint: checkpoints/checkpoint_step_001000.pt  ← (First insurance policy. You'll need many more.)
...

Printing warning summary...
Unknown phoneme warnings: 0          ← (Good. No ghost phonemes haunting your job.)
Audio load errors: 0                 ← (Your files didn't corrupt. Praise be.)
--------------------------------------------------

Epoch 5/100 - Training loss: 5.2134  ← (Better! Still garbage, but quantifiably better)
Epoch 5/100 - Validation loss: 4.9876  ← (The robot is learning! Go listen to the audio samples!)
Saved audio samples for epoch 5      ← (Prepare yourself for adorable failure)
New best validation loss: 4.9876 (saved to checkpoints/best_model.pt)  ← (This is your current champion)
...
```

**Things to obsessively monitor:**

- Loss should go ↓ (winning), ↑ (disaster), = (you're stuck, lower the learning rate)
- If loss is NaN: congratulations, math broke. Reduce learning rate and try again.
- Warning summary should be silent and empty
- New best checkpoints appear every 5–10 epochs initially, then less frequently (diminishing returns, like life)

---

## Expected Console Output During Inference

When you run `python inference_vits.py --checkpoint checkpoints/best_model.pt --text "Hello world"`:

```
Loading checkpoint from checkpoints/best_model.pt
Model loaded successfully                    ← (Prayers answered. State exists.)
Config vocab_size: 149

Processing text: "Hello world"
Converting text to phoneme sequence...
Phoneme sequence: [29, 5, 13, 15, 12, 87, 92, 81, 4, 91, 81]  ← (Alphabet soup translation complete)

Running inference...
Generated mel-spectrogram shape: (80, 127)   ← (Your robot just drew a picture of sound)
Vocoding with Griffin-Lim (this will take 10–30 seconds)...  ← (Go read a book. Seriously. The algorithm is from 1984.)
Vocoding complete

Saved audio to tts_output/hello_world.wav
Audio duration: 2.34 seconds                 ← (Two seconds of robot voice is now your property)
```

**Important note:** Griffin-Lim vocoding is a 40-year-old algorithm and it SHOWS. Converting a 2-second mel-spectrogram can take 10–30 seconds because it iterates until convergence. This is not a bug, it's a _feature_ of using technology from when MTV was still playing music videos. Go upgrade to HiFi-GAN if you value your sanity.

### Optional HiFi-GAN Vocoder (Recommended)

`inference_vits.py` now supports an optional neural vocoder path.

**New flags:**

- `--vocoder_checkpoint`: path to HiFi-GAN generator checkpoint (for example `hifigan/generator_v1`)
- `--vocoder_config`: path to HiFi-GAN config (`.json` or `.yaml`)

If both flags are provided, inference uses HiFi-GAN.
If omitted, inference falls back to Griffin-Lim.

Example:

```bash
python inference_vits.py \
  --checkpoint checkpoints/best_model.pt \
  --config vits_config.yaml \
  --vocoder_checkpoint hifigan/generator_v1 \
  --vocoder_config hifigan/config_v1.json \
  --text "we are at present concerned" \
  --output tts_output/hifigan_test.wav \
  --device cpu
```

---

## Buzz Metric: Spectral Flatness

`scripts/evaluate_tts_output.py` now reports `Spectral flatness` directly for each file.

Interpretation guide:

- `0.00–0.05`: tonal / speech-like
- `0.05–0.20`: mild noise
- `>0.20`: buzzy / noisy

Example evaluation:

```bash
python scripts/evaluate_tts_output.py --audio tts_output/hifigan_test.wav --sample_rate 22050
```

---

## Quick A/B: Griffin-Lim vs HiFi-GAN

Use the same sentence for both paths:

```bash
# Griffin-Lim baseline
python inference_vits.py \
  --checkpoint checkpoints/best_model.pt \
  --config vits_config.yaml \
  --text "we are at present concerned" \
  --output tts_output/gl_test.wav \
  --device cpu

# HiFi-GAN
python inference_vits.py \
  --checkpoint checkpoints/best_model.pt \
  --config vits_config.yaml \
  --vocoder_checkpoint hifigan/generator_v1 \
  --vocoder_config hifigan/config_v1.json \
  --text "we are at present concerned" \
  --output tts_output/hifigan_test.wav \
  --device cpu

# Evaluate both
python scripts/evaluate_tts_output.py --audio tts_output/gl_test.wav --sample_rate 22050
python scripts/evaluate_tts_output.py --audio tts_output/hifigan_test.wav --sample_rate 22050
```

Recent sample run in this repo:

- `gl_test.wav`: spectral flatness `0.0663`, ZCR `0.3588`
- `hifigan_test.wav`: spectral flatness `0.0252`, ZCR `0.1294`

Lower values here indicate less buzz and a more speech-like waveform for the HiFi-GAN path.

---

## How Does It Actually Work?

### Step 1: Text → Phonemes

```
Input:  "The coffee is ready"
Output: "DH AH K AA F IY IH Z R EH D IY"
```

g2p_en converts spelling into actual sounds, because English spelling is a crime against phonetics and should not be trusted with anything important.

### Step 2: Phonemes → Mel Spectrogram

VITS takes phoneme sequences and predicts a mel-spectrogram — a 2D picture of sound where X = time, Y = frequency (mel scale), and brightness = loudness. The model learns to draw the right picture for each sentence. 45 million parameters worth of drawing lessons.

### Step 3: Mel Spectrogram → Audio

The vocoder converts the spectrogram picture back into actual audio waves. You can use Griffin-Lim (simple fallback, more robotic) or HiFi-GAN (neural vocoder, much cleaner output).

---

## GPU Requirements (Will You Suffer? A Guide)

| GPU                   | Training Time         | Recommended batch_size | Agony Level           |
| --------------------- | --------------------- | ---------------------- | --------------------- |
| RTX 4090 (24GB)       | ~3 hours              | 32                     | Tolerable             |
| RTX 3060 Ti (8GB)     | ~12 hours             | 16                     | Mild                  |
| RTX 3060 (6GB)        | ~24 hours             | 8                      | Modern                |
| **RTX 3050 Ti (4GB)** | **~1–2 days**         | **4**                  | **Dedicated**         |
| CPU                   | Several business days | 1 s                    | **Maximum Suffering** |

**If you have an RTX 3050 Ti:** Set `batch_size: 4`, enable `use_amp: true`, run `precompute_features.py` first, and accept that your GPU is doing its absolute best. Respect it.

---

## Model Architecture (For the Curious)

This implements **VITS (Variational Inference Text-to-Speech)** — a single model that goes from phonemes to audio in one shot.

Components, explained without a PhD:

- **Text Encoder**: Turns phonemes into number vectors the rest of the network can work with
- **Posterior Encoder**: During training, looks at real audio and says "here's what this should look like" (training only, retires at inference)
- **Flow Module**: Fancy math that makes the latent space behave properly
- **Decoder**: Turns latent codes into mel spectrograms
- **Adversarial Discriminator**: A second network that judges the output and tells the first network it's bad, which somehow makes the first network better. Neural peer review.

Total: **~21 million parameters**. Big, but not unreasonable. Not WaveNet. We don't talk about WaveNet.

---

## Future Improvements (Aspirational Section)

Once basic VITS is working, theoretically you could:

1. **Train HiFi-GAN**: A proper neural vocoder. Audio quality goes up significantly. Requires more training, more GPU, more courage.
2. **Multi-speaker TTS**: Train on multiple voices. Requires a different dataset and the willingness to suffer more.
3. **Emotion control**: Happy robot, sad robot, very serious robot.
4. **Your own voice**: Record ~10 hours of yourself speaking. Retrain. Become immortal (kind of).
5. **Diffusion-based TTS**: The fancy modern way. Requires reading several papers and feeling bad about yourself first.

Realistically: you'll be happy your computer can talk. That's enough. That's a victory.

---

## Q&A: "Will You Answer My Questions?"

**Q: Did i use AI?**
A: yes the were file you are read is create by AI ( 6:4 , AI:Me ) not the funny version

**Q: How long does training take?**  
A: 3 hours to 2 days, depending on GPU. Most people do 12–24 hours and call it done.

**Q: Can I use CPU?**  
A: Technically yes. Practically, no. Do not. For the love of all that is g00d, do not.

**Q: Why is data prep so slow?**  
A: g2p_en is converting 13,100 sentences to phonemes. It was not built for speed. It was built for accuracy. Go make a snack.

**Q: Can I interrupt training?**  
A: Yes! Checkpoints save every N steps. Resume anytime. This is not optional knowledge — your machine WILL suicide at some point.

**Q: Why does my audio sound weird?**  
A: Pick your poison: not enough training, learning rate issues, bad checkpoint, or Griffin-Lim being Griffin-Lim. Try more training first. Then adjust. Then blame the vocoder.

**Q: Will this work on Mac?**  
A: If you have an M-series chip with current PyTorch MPS support, maybe. Report back. We are curious and slightly concerned for you.

---

## Credits

- The smart researchers at KAIST who invented VITS (2021)
- Linda Johnson, for recording 13,100 sentences at professional quality. An absolute legend.
- PyTorch community, for making deep learning accessible to people like us
- Your GPU, for suffering silently through all of this
- GitHub Copilot, for helping debug device type mismatches and other "why isn't this working?" moments
- You, for reading this far. Truly. Go train something.

---

## License

MIT. Do whatever you want. Make a TTS alarm clock that wakes you up in a robot voice. Build a virtual assistant that sounds vaguely ominous. Train it on audiobooks and confuse your friends. The possibilities are yours.

---

## Final Words

Congratulations. You now have the tools to:

Train VITS on your own GPU  
Precompute features like someone who values their time  
Convert any text to speech (with mild robot energy)  
Convince people your computer is sentient  
Waste electricity in a deeply educational way  
Explain what a mel spectrogram is at a dinner party

**Now go forth and make your GPU suffer for science.**

## The total loss in HexTTS is not bounded between 0 and 1 because it is a weighted sum of reconstruction, KL, and duration losses. In particular, the duration term is based on frame-length differences, which can easily be much larger than 1. Therefore, the absolute loss value is less important than whether it decreases over time, whether validation remains stable, and whether generated audio improves.##

---

_Last updated: April 9, 2026_ (v0.4.7 — hybrid rollback validated + debug/report tooling)  
_GPU cooling status: CRITICAL_  
_Electricity bill status: DO NOT OPEN_  
_vits_data_cached.py status: Use it. Seriously._
