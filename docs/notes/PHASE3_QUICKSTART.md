# Phase 3: Training, Monitoring, and Inference

## VITS TTS Project

This phase covers the current training workflow for the project.

It includes:

- starting training
- using the cached loader
- monitoring progress
- resuming from checkpoints
- generating speech
- handling common failures based on the current patches

---

## 1. Files Required

Make sure these files exist in the project root:

```text
vits_config.yaml
vits_model.py
vits_data.py
vits_data_cached.py
train_vits.py
inference_vits.py
tts_app.py
precompute_features.py
```

Also make sure the prepared data exists:

```text
data/ljspeech_prepared/
├── train.txt
├── val.txt
├── metadata.json
└── cache/
```

---

## 2. Final Pre-Training Checklist

Before training, confirm:

- metadata contains phonemes, not raw words
- cached features are already generated
- `train_vits.py` imports `vits_data_cached`
- `vits_model.py` imports `VOCAB_SIZE` from dataset code
- your config is suitable for RTX 3050 Ti

Recommended config:

```yaml
batch_size: 4
num_workers: 0
use_amp: true
grad_clip_val: 0.5
max_seq_length: 300
pin_memory: true
```

---

## 3. Start Training

Activate environment:

```bash
venv\Scripts\activate
```

Start training:

```bash
python train_vits.py --config vits_config.yaml --device cuda
```

Typical startup flow:

```text
Loading config from vits_config.yaml
Using device: cuda
Initializing VITS model...
Loading cached dataset...
Starting training...
```

---

## 4. What the Cached Loader Changes

The current training flow should use `vits_data_cached.py`.

Benefits:

- faster epochs because mels and IDs are precomputed
- less CPU overhead
- better padding efficiency because samples are sorted by length
- fewer OOM crashes because very long samples can be filtered
- better visibility through initialization logs

Typical loader behavior now includes:

- keep valid cached samples
- skip corrupted or missing cache
- skip samples longer than `max_seq_length`
- report counts for kept and skipped samples

If the skipped-too-long count is very high, increase `max_seq_length` carefully.

---

## 5. Monitor Training with TensorBoard

Open a second terminal and run:

```bash
tensorboard --logdir=./logs
```

Then open:

```text
http://localhost:6006
```

Track:

- total loss
- reconstruction loss
- KL loss
- validation loss
- learning rate

Healthy training usually means:

- train loss decreases
- validation loss also decreases or stays stable
- metrics do not become NaN

---

## 6. Expected Training Timeline

Very rough expectation:

| Time        | Result                     |
| ----------- | -------------------------- |
| 1–2 hours   | very rough / robotic audio |
| 6–8 hours   | understandable speech      |
| 12–24 hours | much better quality        |
| beyond that | incremental improvements   |

With an **RTX 3050 Ti**, training will be slower than larger GPUs. Cached loading helps, but it does not magically remove the cost of training.

---

## 7. Checkpoints

Training should save checkpoints in:

```text
checkpoints/
├── best_model.pt
├── checkpoint_step_1000.pt
├── checkpoint_step_2000.pt
└── ...
```

Use `best_model.pt` for inference unless you want to test a specific checkpoint.

---

## 8. Resume Training

If training stops or crashes, resume from a checkpoint:

```bash
python train_vits.py --config vits_config.yaml --checkpoint checkpoints/checkpoint_step_5000.pt
```

Notes:

- resuming is fine for normal interruptions
- do **not** continue from old checkpoints created before the bad metadata fix
- if your earlier run was trained on raw-word metadata instead of clean phonemes, start a fresh run

---

## 9. Generate Speech from a Trained Model

Single command inference:

```bash
python inference_vits.py --checkpoint checkpoints/best_model.pt --config vits_config.yaml --text "Hello I am Pruthu" --output hello.wav
```

Output file:

```text
hello.wav
```

If your project writes outputs into `tts_output/`, check there as well.

---

## 10. Interactive TTS Mode

Run:

```bash
python tts_app.py --checkpoint checkpoints/best_model.pt
```

Example:

```text
> hello world
> save this is my neural voice
> exit
```

This is useful for quick manual tests without editing shell commands repeatedly.

---

## 11. Interpreting Bad Output

If the audio output is trash, extremely short, or not even close to speech, check these first:

### A. Was the metadata fixed?

Your `train.txt` must contain phonemes, not words.

Correct:

```text
LJ001-0001|DH AH ...
```

Wrong:

```text
LJ001-0001|THE PROJECT ...
```

### B. Was vocab size fixed?

The model must import vocabulary size from dataset code, not use a hardcoded number.

Correct:

```python
from vits_data import VOCAB_SIZE
```

Wrong:

```python
VOCAB_SIZE = 149
```

### C. Are you testing an old bad checkpoint?

If the checkpoint was produced before the metadata cleanup or vocabulary consistency fix, do not trust it.

### D. Has the model trained long enough?

Early checkpoints often sound very bad. That is normal.

---

## 12. Common Problems and Fixes

### CUDA out of memory

Reduce batch size:

```yaml
batch_size: 2
```

You can also lower `max_seq_length` a bit, but be careful not to discard too much useful data.

---

### Loss becomes NaN

Try:

```yaml
learning_rate: 0.001
grad_clip_val: 0.5
```

Also verify:

- cache is valid
- no corrupted samples slipped through
- metadata is clean

---

### Training is slow even after caching

Check all of the following:

- did you actually run `precompute_features.py`
- does `data/ljspeech_prepared/cache/` contain `.npy` files
- does `train_vits.py` import `vits_data_cached`
- is `use_amp: true`
- is the GPU being used

---

### Unknown phoneme warnings are still huge

That usually means preprocessing is still wrong.

Regenerate the prepared dataset:

```bash
python prepare_data.py ./data/LJSpeech-1.1 ./data/ljspeech_prepared
```

Then inspect `train.txt` manually.

---

### Too many samples are skipped

The cached loader can skip samples that exceed `max_seq_length`.

If the logs show a large skipped-too-long count, raise:

```yaml
max_seq_length: 500
```

or another careful value based on your data and VRAM.

---

## 13. Suggested Training Workflow

Use this order:

### Step 1

```bash
venv\Scripts\activate
```

### Step 2

```bash
python prepare_data.py ./data/LJSpeech-1.1 ./data/ljspeech_prepared
```

### Step 3

```bash
python precompute_features.py --config vits_config.yaml
```

### Step 4

Confirm:

- cached loader is imported
- vocab size is not hardcoded

### Step 5

```bash
python train_vits.py --config vits_config.yaml --device cuda
```

### Step 6

```bash
tensorboard --logdir=./logs
```

### Step 7

After a useful checkpoint exists:

```bash
python inference_vits.py --checkpoint checkpoints/best_model.pt --config vits_config.yaml --text "Hello I am Pruthu" --output hello.wav
```

---

## 14. What Success Looks Like

A good Phase 3 result means:

- training runs without crashes
- checkpoint files are created
- losses trend downward
- TensorBoard logs appear
- generated audio is at least intelligible
- later checkpoints sound better than earlier ones

---

## 15. Next Improvements After Basic Training

Once the current pipeline works reliably, future upgrades can include:

- better vocoder than Griffin-Lim
- multi-speaker support
- cleaner inference pipeline
- more robust evaluation scripts
- stronger README demos and sample outputs

For now, the goal is simple:

**stable training + correct data + usable speech output**.
