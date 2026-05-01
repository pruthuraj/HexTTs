# HexTTs Inference

This page is the quick inference entrypoint. For the full synthesis and metric workflow, see [Inference and Evaluation](inference-evaluation.md).

## Basic Inference

```bash
python scripts/infer.py --checkpoint checkpoints/best_model.pt --config configs/base.yaml --text "hello world" --output tts_output/hello.wav --device cpu
```

## HiFi-GAN Inference

```bash
python scripts/infer.py --checkpoint checkpoints/best_model.pt --config configs/base.yaml --vocoder_checkpoint hifigan/generator_v1 --vocoder_config hifigan/config_v1.json --text "we are at present concerned" --output tts_output/output_hifigan.wav --device cpu
```

## Evaluate Output

```bash
python scripts/evaluate_tts_output.py --audio tts_output/output_hifigan.wav --sample_rate 22050
```

## Compare Vocoder Paths

```bash
python scripts/main_flow.py compare --text "we are at present concerned"
```

## Related Docs

- [System Flow](system-flow.md)
- [Inference and Evaluation](inference-evaluation.md)
- [Operations and Troubleshooting](operations-troubleshooting.md)
