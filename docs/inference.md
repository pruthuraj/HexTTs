# HexTTs Inference

Primary inference entrypoint:

```bash
venv\Scripts\python.exe scripts\infer.py --checkpoint checkpoints/best_model.pt --config configs/base.yaml --text "hello world" --output tts_output/hello.wav --device cpu
```

HiFi-GAN inference (recommended quality path):

```bash
venv\Scripts\python.exe scripts\infer.py --checkpoint checkpoints/best_model.pt --config configs/base.yaml --vocoder_checkpoint hifigan/generator_v1 --vocoder_config hifigan/config_v1.json --text "we are at present concerned" --output tts_output/output_hifigan.wav --device cpu
```

Evaluate generated audio:

```bash
venv\Scripts\python.exe scripts/evaluate_tts_output.py --audio tts_output/output_hifigan.wav --sample_rate 22050
```

## Topics

- shared inference pipeline
- text normalization and tokenization
- Griffin-Lim fallback
- HiFi-GAN vocoder path
