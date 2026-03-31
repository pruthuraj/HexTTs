# HexTTs: The Robot That Finally Learned to Speak

### _"Because Your GPU Wasn't Hot Enough Yet"_

---

## What Is This Monstrosity?

HexTTs is a **Text-to-Speech (TTS)** project that teaches an AI neural network to convert boring text into spoken words. It's basically teaching a computer to be a voice actor, except it won't complain about late nightshots or demand residuals.

- **Trained on**: 13,100 audio clips of a very patient woman (LJSpeech dataset)
- **Powers**: The ability to type "hello world" and actually hear your computer SAY it
- **Side effects**: Your GPU fans will sound like a jet engine, and your electricity bill will cry

---

## The Project Structure (What's All This Junk?)

```
HexTTs/
├── train_vits.py           ← The actual sorcery happens here
├── inference_vits.py       ← "Please make sounds from my text"
├── tts_app.py              ← Interactive mode (for people who hate command lines)
├── vits_model.py           ← The neural network brain (45 million parameters btw)
├── vits_data.py            ← Data loading (it's surprisingly boring)
├── vits_config.yaml        ← "How angry should my GPU get?"
├── prepare_data.py         ← "Let me fix the phonemes because the dataset was messy"
├── validate_dataset.py     ← Quality control (spoiler: data is weird)
├── requirements.txt        ← "Pip install your life away"
├── checkpoints/            ← Model snapshots (save the good ones!)
├── data/
│   ├── LJSpeech-1.1/      ← 13,100 voice samples (24GB of pure audio)
│   └── ljspeech_prepared/ ← "The cleaned up version"
├── logs/                   ← TensorBoard metrics (watch your loss go brrr)
└── tts_output/            ← The fruits of your GPU's labor
```

---

## Installation: "The Suffering Begins"

### Step 1: Set Up Your GPU (If You Have One)

```bash
# Check if your NVIDIA card is ready to suffer
nvidia-smi

# If this doesn't work, Google for 2 hours and cry a little
```

### Step 2: Create Virtual Environment

```bash
# Isolate yourself from the chaos
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac (jk, no one uses Mac for this)
```

### Step 3: Install PyTorch (The REAL PyTorch, with CUDA)

```bash
# This is the CORRECT way (don't @ me, CPU users)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Step 4: Install Everything Else

```bash
pip install -r requirements.txt
```

The magic incantations:

- **librosa**: For pretending you understand spectrograms
- **g2p_en**: Converts "the quick brown fox" into "DH AH K W IH K BR AW N F AA K S"
- **numpy<2**: Not numpy 2 (it breaks everything, trust me)

---

## Data Prep: "Why Is Your Dataset Like This?"

```bash
# Download LJSpeech manually (13GB) from:
# https://keithito.com/LJ-Speech-Dataset/

# Validate it's not corrupted
python validate_dataset.py ./data/LJSpeech-1.1

# Convert words to phonemes (this is important, I swear)
python prepare_data.py ./data/LJSpeech-1.1 ./data/ljspeech_prepared
```

**What happens:** The script reads 13,100 transcripts, turns them into phonemes, and creates train/val splits. It's less exciting than it sounds.

---

## Training: "Let's Make Your Room Hot"

```bash
# Start training (go get coffee, you'll be here a while)
python train_vits.py --config vits_config.yaml --device cuda
```

### What Happens:

1. Neural network gets a text → phoneme sequence
2. Network predicts a mel-spectrogram (fancy spectrogram)
3. Compares to real spectrogram
4. Cries about how wrong it was (loss calculation)
5. Updates weights to be less wrong
6. Repeats 10,000+ times
7. **Eventually**: Makes acceptable robot sounds

### Expected Timeline:

- **Hour 0-1**: "This is going to work!" → Audio sounds like garbage disposal
- **Hour 4-8**: "Okay, sounds slightly human-ish" → Still pretty robotic
- **Hour 12-24**: "Not bad!" → Decent speech synthesis
- **Hour 24+**: "This is actually good" → Congratulations, you wasted a day of electricity

### Monitoring:

```bash
# In another terminal, watch the loss curves in real-time
tensorboard --logdir=./logs
# Open http://localhost:6006
```

### Key Metrics You'll Obsess Over:

- **Total Loss**: Should go ↓ (good) not ↑ (bad)
- **Recon Loss**: How wrong the mel-spectrogram is
- **KL Loss**: Prevents the neural network from totally cheating
- **Your Sanity**: Will also go ↓

---

## Inference: "Let Me Hear My Robot" 

### Single Command:

```bash
python inference_vits.py \
  --checkpoint checkpoints/best_model.pt \
  --config vits_config.yaml \
  --text "Hello, I am a robot overlord" \
  --output hello.wav
```

### Interactive Mode (For Impatient People):

```bash
python tts_app.py --checkpoint checkpoints/best_model.pt
```

Then type stuff:

```
> Hello world
> Save this one
> My name is HexTTs and I sound weird
> exit
```

Output files go to `tts_output/` directory.

---

## Common Problems & Funny Solutions

### "CUDA out of memory"

```
Your GPU: "I'm full"
You: "But I only gave you one sentence!"
Your GPU: "TOO BAD"
```

**Fix:** Reduce `batch_size` in vits_config.yaml (8, 4, or 2)

### "Loss is not decreasing"

Your learning rate is probably too aggressive. Like giving your neural network espresso.

```yaml
learning_rate: 0.001 # More like decaf
```

### "The audio sounds robotic"

That's because it IS a robot. It needs more training. More training = More tears in your electricity bill.

### "ModuleNotFoundError: No module named 'vits_model'"

Did you download ALL the files? Did you put them in the SAME folder? Did you actually read the instructions?

### "My audio is 10 seconds long why is it processing for a minute?"

The Griffin-Lim vocoder is slower than your GPU. Welcome to life.

---

## File Descriptions (TL;DR Edition)

| File                  | What It Does                         | When You Care              |
| --------------------- | ------------------------------------ | -------------------------- |
| `train_vits.py`       | Trains the entire model              | Always                     |
| `inference_vits.py`   | Makes audio from text                | After training             |
| `tts_app.py`          | Pretty UI for making audio           | When you're tired of CLI   |
| `vits_model.py`       | 45 million parameters of confusion   | Never (it's magic)         |
| `vits_data.py`        | Loads data without crashing          | During training            |
| `vits_config.yaml`    | "How do I balance quality vs speed?" | Before training            |
| `prepare_data.py`     | Fixes the dataset                    | Once, then forget about it |
| `validate_dataset.py` | "Is my data okay?"                   | When paranoid              |

---

## The Deep Dive: "But How Does It Actually Work?" 

### Phase 1: Text → Phonemes

```
Input: "The coffee is ready"
Output: "DH AH K AA F I IH Z R EH D IH"
```

This is using g2p_en (Grapheme-to-Phoneme). It's like a really smart autocorrect, but for sounds.

### Phase 2: Phonemes → Mel-Spectrogram

Your neural network (VITS) takes phoneme sequences and predicts a mel-spectrogram. A mel-spectrogram is basically a fancy picture of sound. Spectrograms are cool because:

- They show time on the X-axis
- Frequency on the Y-axis (but in the "mel" scale, which matches how humans hear)
- Intensity as color brightness

### Phase 3: Mel-Spectrogram → Audio

Griffin-Lim algorithm reconstructs audio from the spectrogram. It's basically audio origami.

---

## GPU Requirements (Will You Suffer?)

| GPU               | Training Time | Batch Size | Agony Level |
| ----------------- | ------------- | ---------- | ----------- |
| RTX 3060 (6GB)    | ~24 hours     | 8          | Modern      |
| RTX 3060 Ti (8GB) | ~12 hours     | 16         | Mild        |
| RTX 4090 (24GB)   | ~3 hours      | 32         | Tolerable   |
| CPU               | ... days      | 1          | **Maximum** |

**Pro Tip:** If you use CPU mode, just let it training in the background for a month. Seriously.

---

## Resume Training: "Oops, It Crashed"

If your training gets interrupted:

```bash
# Find the latest checkpoint
dir checkpoints/

# Resume from where you left off
python train_vits.py --config vits_config.yaml --checkpoint checkpoints/checkpoint_step_005000.pt
```

It's like saving in a video game. Except the game is making AI sounds.

---

## What's Actually in This Project?

This is a **VITS (Variational Inference Text-to-Speech)** implementation because:

- VITS = Good quality audio
- VITS = Relatively fast inference
- VITS = 45 million parameters (it's big but not HUGE)
- VITS = Easier to train than WaveNet

It's not the latest fancy model (that would be something with Transformers and diffusion), but it actually works and isn't a nightmare to train.

---

## Advanced Stuff (For Masochists)

Once you have basic VITS working, you could:

1. **Train HiFi-GAN**: Better vocoder (Griffin-Lim sounds kinda bad)
2. **Multi-speaker TTS**: Train on multiple speakers at once
3. **Emotion control**: Make the robot angry, sad, or excited
4. **Your own voice**: Record yourself, retrain (good luck sounding better than LJ)
5. **FastSpeech2**: Make it faster (sacrificing quality)

But honestly, you probably won't. You'll be satisfied that your computer can finally talk.

---

## Real Talk (Actual Useful Info)

- **Paper**: VITS: A Single Shot Text-to-Speech with Conditional Adversarial Networks
- **Authors**: Seriously smart people from Korea
- **Year**: 2021 (still relevant!)
- **Dataset**: LJSpeech (Linda Johnson speaking 13,100 sentences)
- **Your rig**: Will work, probably

---

## QA: "Will You Answer My Stupid Questions?"

**Q: How long does training take?**
A: Depends on your GPU. 3 hours to 3 days. Most people do 12-24 hours and call it good.

**Q: Can I use CPU?**
A: Technically yes. Should you? No.

**Q: Why does my data preparation take forever?**
A: g2p_en is slow. It's converting thousands of words to phonemes. Patience, young grasshopper.

**Q: Can I interrupt training?**
A: Yes! It saves checkpoints every N steps. Just resume later.

**Q: Why is the audio weird after training?**
A: Either:

- Not enough training (do more epochs)
- Bad hyperparameters (adjust learning rate)
- Corrupted checkpoint (start over)
- Griffin-Lim vocoder limitations (use HiFi-GAN)

**Q: Will this work on Mac?**
A: Maybe! If you have an M-series chip with proper PyTorch support. Good luck though.

---

## Credits

- The actual smart researchers who invented VITS
- Linda Johnson for 13,100 recordings of sentences (what a legend)
- You, for having the patience to read this entire README
- Your GPU, for suffering through this with you

---

## License

MIT? Sure. Use it however you want. Make cursed TTS models. Make your alarm clock speak in a robot voice. Make a virtual assistant that sounds vaguely threatening.

---

## Final Words

Congratulations! You now have the tools to:

-Train VITS on your own GPU  
-Convert any text to speech (kind of)  
-Convince people your computer is sentient  
-Waste electricity impressively  
-Understand mel-spectrograms (approximately)

**Now go forth and make your GPU suffer for science!**

---

_Last updated: March 31, 2026_  
_GPU cooling status: CRITICAL_  
_Electricity bill status: DO NOT OPEN_
