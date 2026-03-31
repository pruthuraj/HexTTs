# Phase 3: Quick Start Guide

## Training VITS and Generating Speech

---

## 📁 Files You Need

Download these 8 files and place them in your `VITS_TTS/` folder:

**Configuration:**

- `vits_config.yaml` - Training parameters

**Model Code:**

- `vits_model.py` - VITS neural network
- `vits_data.py` - Data loading
- `train_vits.py` - Training script

**Inference:**

- `inference_vits.py` - Generate speech from text
- `tts_app.py` - Interactive TTS application

---

## 🚀 Quick Start: 3 Commands

### Command 1: Start Training

```bash
# Activate environment
venv\Scripts\activate

# Train the model
python train_vits.py --config vits_config.yaml --device cuda
```

**Expected output:**

```
Loading config from vits_config.yaml
Using GPU: NVIDIA GeForce RTX 3060
Initializing VITS model...
Total parameters: 45.32M

Training set size: 12445
Validation set size: 655

Starting training for 100 epochs...

Epoch 1/100 - Training loss: 2.3451
Epoch 1/100 - Validation loss: 2.1234
```

**Training time:**

- First results: 1-2 hours
- Good quality: 8-12 hours
- Excellent: 20-24 hours

---

### Command 2: Monitor Training

While training is running, open another command prompt:

```bash
# In a new terminal (with venv activated)
tensorboard --logdir=./logs
```

Then open http://localhost:6006 in your web browser to see:

- Loss curves (training vs validation)
- Loss breakdown (reconstruction, KL, duration)
- Learning rate changes

---

### Command 3: Generate Speech

After training completes, generate speech:

**Command-line mode:**

```bash
python inference_vits.py \
  --checkpoint checkpoints/best_model.pt \
  --config vits_config.yaml \
  --text "Hello, this is my AI voice" \
  --output hello.wav
```

**Interactive mode:**

```bash
python tts_app.py --checkpoint checkpoints/best_model.pt
```

Then type:

```
> hello world
> save My name is VITS
> exit
```

---

## 📊 Training Process

### What Happens

```
Epoch 1: Loss 2.34 → Model learning random patterns
Epoch 10: Loss 0.95 → Getting better
Epoch 50: Loss 0.25 → Good speech quality
Epoch 100: Loss 0.12 → Excellent speech quality
```

### Key Metrics

| Metric     | Good         | Bad           |
| ---------- | ------------ | ------------- |
| Train loss | Decreasing ↓ | Constant or ↑ |
| Val loss   | Decreasing ↓ | Increasing ↑  |
| Recon loss | 0.1-0.4      | > 1.0         |
| KL loss    | 0.001-0.01   | 0 or > 0.1    |

---

## 🛠️ Configuration: GPU Memory

**If you run out of memory (OOM error):**

Edit `vits_config.yaml`:

```yaml
# For 6 GB VRAM (RTX 3060):
batch_size: 8  # reduced from 16

# For 12 GB VRAM (RTX 3060 Ti):
batch_size: 16

# For 24+ GB VRAM (RTX 4090):
batch_size: 32
```

Save and restart training.

---

## 📂 Output Structure

After training:

```
VITS_TTS/
├── checkpoints/
│   ├── best_model.pt            ← Use this for inference
│   ├── checkpoint_step_001000.pt
│   └── checkpoint_step_002000.pt
├── logs/
│   └── events.out.tfevents.xxx  ← TensorBoard logs
└── tts_output/                  ← Generated audio files
    ├── tts_0.wav
    ├── tts_1.wav
    └── ...
```

---

## 🎯 Common Issues

### Issue: CUDA out of memory

```
RuntimeError: CUDA out of memory
```

**Fix:** Reduce `batch_size` in vits_config.yaml (8, 4, or 2)

### Issue: Loss not decreasing

**Cause:** Learning rate too high
**Fix:** Change in vits_config.yaml:

```yaml
learning_rate: 0.001 # reduced from 0.002
```

### Issue: Generated audio is robotic

**Cause:** Need more training
**Fix:** Increase num_epochs to 100-200

### Issue: "ModuleNotFoundError: No module named 'vits_model'"

**Fix:** Make sure all .py files are in the same folder

---

## ⏱️ Timeline

### Hour 1-2: First Training Iteration

- Model learns basic phoneme to sound mapping
- Audio will sound very robotic
- Loss drops rapidly

### Hour 6-8: Good Results

- Speech is intelligible
- Most phonemes correct
- Some artifacts remain

### Hour 12-24: Excellent Results

- Natural-sounding speech
- Minimal artifacts
- Good prosody and timing

---

## 🔄 Resume Training

If training crashes or is interrupted:

```bash
# List available checkpoints
dir checkpoints/

# Resume from checkpoint
python train_vits.py --config vits_config.yaml --checkpoint checkpoints/checkpoint_step_005000.pt
```

---

## 📝 Typical Training Session

```bash
# 1. Activate environment
venv\Scripts\activate

# 2. In Terminal 1: Start training
python train_vits.py --config vits_config.yaml --device cuda

# 3. In Terminal 2: Monitor with TensorBoard
tensorboard --logdir=./logs
# Open http://localhost:6006

# 4. After ~12 hours, training finishes
# Model saved to checkpoints/best_model.pt

# 5. In Terminal 3: Test the model
python tts_app.py --checkpoint checkpoints/best_model.pt
# Type: > hello world
# Audio saved to tts_output/tts_0.wav
```

---

## ✅ Verification Checklist

- [ ] All 8 files downloaded to VITS_TTS folder
- [ ] Training starts without errors
- [ ] GPU is being used (check with nvidia-smi)
- [ ] Loss decreases each epoch
- [ ] TensorBoard shows loss curves
- [ ] Training completes (or can be resumed)
- [ ] best_model.pt exists in checkpoints/
- [ ] Inference generates audio file
- [ ] Generated audio is intelligible

---

## 🎓 What's Actually Happening

**Training:**

1. Model gets phoneme sequence
2. Predicts mel-spectrogram
3. Compares to real mel-spectrogram
4. Calculates error (loss)
5. Updates weights to reduce error
6. Repeats 10,000+ times
7. Model learns to convert text → speech

**Inference:**

1. Your text → Phonemes
2. Phonemes → Mel-spectrogram (using trained model)
3. Mel-spectrogram → Audio (using Griffin-Lim)
4. Audio saved to .wav file

---

## 📞 Help & Troubleshooting

**Common Questions:**

Q: How long should I train?
A: Start with 10 epochs (1-2 hours), then decide. 50+ epochs for good quality.

Q: Can I stop and resume?
A: Yes, use --checkpoint with the latest checkpoint file.

Q: What GPU do I need?
A: 6GB minimum (RTX 3060). Faster with 12GB+.

Q: Can I use CPU?
A: Yes (--device cpu) but training takes 5-10x longer.

Q: Why is audio quality bad?
A: Model needs more training. Train for more epochs.

Q: Can I use my own voice?
A: Yes (advanced). Record ~1 hour, prepare metadata, retrain.

---

## 🚀 Next Level (Advanced)

Once you have basic VITS working:

1. **Better Vocoder:** Train HiFi-GAN instead of Griffin-Lim
2. **Multi-speaker:** Train on multiple speakers
3. **FastSpeech2:** For faster inference
4. **Emotion Control:** Add emotion tokens
5. **Your Own Voice:** Collect speaker data and retrain

---

## 📚 Key Files Reference

| File              | What to do              | When                  |
| ----------------- | ----------------------- | --------------------- |
| vits_config.yaml  | Edit hyperparameters    | Before training       |
| train_vits.py     | Run with --checkpoint   | Resume training       |
| inference_vits.py | Run with --text         | Generate single audio |
| tts_app.py        | Run for interactive use | After training        |
| tensorboard       | Point to ./logs         | While training        |

---

## ✨ You're Ready!

You now have everything needed to:

1. ✅ Train VITS on your GPU
2. ✅ Monitor training progress
3. ✅ Generate speech from text
4. ✅ Use interactive TTS app

**Start training now and check back in a few hours!** 🚀
