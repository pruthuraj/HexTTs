# My Config — RTX 3050 Ti (4GB)

### "How to Train VITS Without Cooking Your Laptop"

---

## The Sacred GPU Ritual

Inside `vits_config.yaml`:

```yaml
batch_size: 4
num_workers: 2
use_amp: true
grad_clip_val: 0.5
max_seq_length: 300
pin_memory: true
```

These values were scientifically determined by:

```
Me:   let's try this
GPU:  pls no
Me:   ok batch_size=4
GPU:  acceptable
```

---

# Meme Section

### When training starts

```
You:      python train_vits.py
GPU:      on fire
Laptop:   fan no they are engine
Neighbors: "Is that a jet engine??"
```

---

# Why These Settings Exist

## use_amp = true

Mixed precision means:

```
GPU calculations: 32 bit → 16 bit
GPU memory:       half
GPU happiness:    doubled
```

Real explanation:

> Less precision = faster training

Meme explanation:

```
GPU: "You want full precision math??"
GPU: "On THIS economy??"
```

If NaNs appear:

```
Model: NaN NaN NaN NaN
You:   (-_-)
```

Turn it off:

```yaml
use_amp: false
```

---

# Sequence Length (The VRAM Destroyer)

```yaml
max_seq_length: 300
```

Increasing this number results in:

```
CUDA out of memory
```

Which translates to:

```
GPU: "I am tired boss."
```

Lower values =

✔ less memory  
✔ faster training  
✔ fewer emotional breakdowns

---

# num_workers = 2

Workers control how fast the CPU feeds data to the GPU.

| Workers | Result         |
| ------- | -------------- |
| 0       | painfully slow |
| 2       | good           |
| 8       | Windows chaos  |

So we choose **2**, the peaceful option.

---

# batch_size = 4

my GPU has **4GB VRAM**.

Which means my options are:

```
batch_size = 4 → good
batch_size = 8 → crash
batch_size = 16 → instant death
```

Error message translation:

```
RuntimeError: CUDA out of memory
```

Actual meaning:

```
GPU: help me,HELP ME !!!
```

---

# The Real Bottleneck

My GPU is not the slowest thing.

This innocent villain is:

```
librosa
```

Right now the dataloader does this every sample:

```
load wav
compute mel
convert to dB
convert phonemes
send to GPU
```

CPU during training:

```
89 % used
```

---

# The Real Speed Upgrade

Precompute features.

Instead of computing mel spectrograms every batch:

```
dataset → preprocess once → save .npy
training → load arrays only
```

Result:

```
Epoch time: ↓↓↓
GPU usage:  ↑↑↑
CPU pain:   ↓↓↓
```

---

# Training Strategy

Start training:

```
python train_vits.py --config vits_config.yaml --device cuda
```

Then wait.

Typical timeline:

| Time     | Result                 |
| -------- | ---------------------- |
| 1 hour   | garbage robot noises   |
| 6 hours  | slightly human         |
| 12 hours | pretty good            |
| 24 hours | impressive robot voice |

---

# Hardware Target

Optimized for:

```
GPU: RTX 3050 Ti
VRAM: 4GB
Fan = jet engine
Room temperature: +5°C
```

---

# Final Wisdom

Training deep learning models requires:

✔ patience  
✔ correct configs  
✔ tolerance for loud laptop fans

Eventually may your computer say:

```
Hello world
```

And it only cost **one day of GPU suffering**.
