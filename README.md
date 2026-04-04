# HexTTs: The Robot That Finally Learned to Speak

_v0.3.2_

### _"Because Your GPU Wasn't Hot Enough Yet"_

---

## What Is This Monstrosity?

HexTTs is a **Text-to-Speech (TTS)** project that teaches an AI neural network to convert boring text into spoken words. It's basically teaching a computer to be a voice actor, except it won't complain about late night shoots or demand residuals.

- **Trained on**: 13,100 audio clips of a very patient woman (LJSpeech dataset)
- **Powers**: The ability to type "hello world" and actually hear your computer SAY it
- **Side effects**: Your GPU fans will sound like a jet engine, your electricity bill will cry, and you'll start explaining mel-spectrograms at parties.

Let's pivot our eyeballs toward the _\diagram and \doc_ to synergize our confusion into a cohesive misunderstanding

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
├── view_spectrogram.py        ← Visualize mel spectrograms (stare at pretty graphs)
│
├── vits_config.yaml           ← "How angry should my GPU get?"
├── requirements.txt           ← All the suffering, listed as pip packages
├── CHANGELOG.md               ← A record of your poor decision-making over time
│
├── prepare_data.py            ← "Let me fix the phonemes because the dataset was messy"
├── validate_dataset.py        ← Quality control (spoiler: data is weird)
├── precompute_features.py     ← Computes mel spectrograms ahead of time so training doesn't die slowly
├── test_setup.py              ← Sanity check that all your dependencies installed correctly
│
├── checkpoints/               ← Model snapshots (save the good ones, delete the tragic ones)
├── logs/                      ← TensorBoard metrics (watch your loss go brrr)
├── tts_output/                ← The fruits of your GPU's labor
│
├── diagram/                   ← Visual documentation (pretty pictures of data flow)
├── doc/                       ← Extra documentation you'll read later (spoiler: you won't)
├── notes/                     ← Patch notes, setup guides, lessons learned the hard way
├── deprecated/                ← Old code graveyard (abandon hope, all ye who enter here)
├── scripts/                   ← Miscellaneous utility scripts (your personal junk drawer)
│
└── data/
    ├── LJSpeech-1.1/          ← 13,100 voice samples (24GB of pure audio patience)
    └── ljspeech_prepared/     ← "The cleaned up version" (+ cached/ if you're smart)
```

---

## Installation: "The Suffering Begins"

### Step 1: Create Virtual Environment

```bash
# Isolate yourself from the chaos
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Linux/Mac
```

### Step 2: Install PyTorch (The REAL PyTorch, with CUDA)

```bash
# This is the CORRECT way (don't @ me, CPU users)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Verify it actually got CUDA support:

```bash
python -c "import torch; print(torch.cuda.is_available())"
# Should print: True
# If it prints False, you installed the sad version of PyTorch. Start over.
```

### Step 3: Install Everything Else

```bash
pip install -r requirements.txt
```

The magic incantations:

- **librosa**: For pretending you understand spectrograms
- **g2p_en**: Converts "the quick brown fox" into "DH AH K W IH K BR AW N F AA K S" (very slowly)
- **numpy**: Used everywhere. Breathes numpy.
- **matplotlib**: For when you want to stare at loss curves and feel something
- **tensorboard**: Real-time loss anxiety, now with graphs

---

## Data Prep: "Why Is Your Dataset Like This?"

```bash
# Download LJSpeech manually (13GB) from:
# https://keithito.com/LJ-Speech-Dataset/

# Place it in: data/LJSpeech-1.1

# Validate it's not corrupted or haunted
python validate_dataset.py ./data/LJSpeech-1.1

# Convert words to phonemes (this is important, I swear)
python prepare_data.py ./data/LJSpeech-1.1 ./data/ljspeech_prepared
```

**What happens:** The script reads 13,100 transcripts, turns them into phonemes, and creates train/val splits. It's less exciting than it sounds. The output is:

```
train.txt        ← ~12,445 phoneme entries (the main event)
val.txt          ← ~655 entries (the understudy)
phoneme sequences ← cryptic strings of capitalized nonsense that somehow become speech
```

---

## Speed Optimization: "Wait, Training Can Be FASTER?"

Yes. By default, mel spectrograms are computed from scratch every single batch during training. This is like baking the same bread every time you want a sandwich. `precompute_features.py` bakes all the bread once upfront.

```bash
python precompute_features.py --config vits_config.yaml
```

This creates:

```
data/ljspeech_prepared/cache/
    mels/    ← Precomputed mel spectrograms (the bread)
    ids/     ← File identifiers (the labels on the bread)
```

### Enable the Cached Loader

In `train_vits.py`, swap this one line:

```python
# SLOW (the old way, computing mels fresh every batch like a masochist)
from vits_data import create_dataloaders

# FAST (the new way, reading pre-baked mels like a rational person)
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

| File                     | What It Does                                     | When You Care                   |
| ------------------------ | ------------------------------------------------ | ------------------------------- |
| `train_vits.py`          | Trains the entire model                          | Always                          |
| `inference_vits.py`      | Makes audio from text                            | After training                  |
| `tts_app.py`             | Pretty interface for making audio                | When you're tired of CLI        |
| `vits_model.py`          | 45 million parameters of confusion               | Never (it's magic, don't touch) |
| `vits_data.py`           | Original data loader                             | During training (slow version)  |
| `vits_data_cached.py`    | Faster data loader                               | During training (smart version) |
| `vits_config.yaml`       | "How do I balance quality vs speed vs survival?" | Before training                 |
| `prepare_data.py`        | Converts transcripts to phonemes                 | Once, then forget it            |
| `validate_dataset.py`    | "Is my data okay?"                               | When paranoid                   |
| `precompute_features.py` | Pre-bakes mel spectrograms                       | Before training (do this)       |
| `view_spectrogram.py`    | Visualize mel spectrograms                       | When debugging audio issues     |
| `test_setup.py`          | Verify PyTorch/CUDA installed correctly          | On first setup                  |
| `requirements.txt`       | All your dependencies                            | During pip install              |
| `CHANGELOG.md`           | What changed between versions                    | When things break mysteriously  |
| `checkpoints/`           | Saved model states                               | Always (precious cargo)         |
| `logs/`                  | TensorBoard event files                          | During/after training           |
| `tts_output/`            | Generated audio files                            | After inference                 |
| `diagram/`               | Architecture diagrams and flowcharts             | When explaining to others       |
| `doc/`                   | Extended documentation                           | When README isn't enough        |
| `notes/`                 | Development notes, patches, and lessons learned  | When troubleshooting            |
| `deprecated/`            | Old code nobody knows how to delete              | Never (for archaeologists only) |
| `scripts/`               | Utility scripts and helper tools                 | When needed, god help you       |

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

The vocoder converts the spectrogram picture back into actual audio waves. We use Griffin-Lim (classic, a bit robotic) because it requires no extra training. HiFi-GAN is the upgrade path if you want to feel fancy later.

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

**Q: Did I use AI?**
A: Yes… but only for the serious document in the /docs folder and the connments.

The funny parts?
That was 100% human intelligence.

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

---

_Last updated: April 4, 2026_  
_GPU cooling status: CRITICAL_  
_Electricity bill status: DO NOT OPEN_  
_vits_data_cached.py status: Use it. Seriously._
