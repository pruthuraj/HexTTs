# Deprecated Experiment Record: Phoneme-Aware Duration Targets

Date: 2026-04-09
Branch: copilot/vscode-mnrjulze-pje7

## What Was Tried

A phoneme-aware pseudo-duration target was added to `train_vits.py` so that:

- vowels received slightly more duration
- pause and silence tokens received the most duration
- consonants stayed at baseline
- exact total frame sum was preserved

The goal was to improve rhythm and reduce rushed, machine-gun-like speech.

## What Happened

A continuation smoke test was run with the patched duration logic.

### Training / validation outcome

- Training loss: 0.8752
- Validation loss: 0.7669
- Token duration MAE: 0.0494 train, 0.0354 val
- Sum error mean: 4.25 train, 4.00 val
- Pred speech rate proxy: 0.1255 train, 0.1487 val

### Audio outcome

- Output duration: 0.2322 s
- Zero crossing rate: 0.111282
- Spectral flatness: 0.045637
- Verdict: too short, duration predictor likely still weak

## Why It Was Rejected

The metrics were not internally consistent with a healthy duration model.

The output audio collapsed to a very short duration even though token MAE looked small. The speech-rate proxy also collapsed to an implausibly low value, which suggested the patch introduced a target or metric mismatch rather than a real improvement in prosody.

## Decision

This phoneme-aware duration patch was rolled back.

The restored version uses the earlier uniform, sum-preserving pseudo-duration targets, which had already produced a healthier result around 1.86 s with clean HiFi-GAN audio.

## Deprecated Code

```python
from vits_data import create_dataloaders, get_warning_summary, reset_warning_summary, PHONEME_TO_ID

VOWEL_PHONEMES = {
	"AA", "AE", "AH", "AO", "AW", "AY",
	"EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"
}

PAUSE_PHONEMES = {
	"PAD", "SP", "SIL", "PAUSE"
}

def build_pseudo_duration_targets(
	self,
	phoneme_ids: torch.Tensor,
	phoneme_lengths: torch.Tensor,
	mel_lengths: torch.Tensor,
) -> torch.Tensor:
	batch_size, max_tokens = phoneme_ids.shape
	targets = torch.zeros(batch_size, max_tokens, device=phoneme_ids.device, dtype=torch.float32)

	id_to_phoneme = {v: k for k, v in PHONEME_TO_ID.items()}

	for i in range(batch_size):
		n_tokens = min(int(phoneme_lengths[i].item()), max_tokens)
		total_frames = int(mel_lengths[i].item())

		if n_tokens <= 0 or total_frames <= 0:
			continue

		weights = []
		for j in range(n_tokens):
			pid = int(phoneme_ids[i, j].item())
			phoneme = id_to_phoneme.get(pid, "")

			if phoneme in PAUSE_PHONEMES:
				weight = 2.0
			elif phoneme in VOWEL_PHONEMES:
				weight = 1.5
			else:
				weight = 1.0

			weights.append(weight)

		weights = torch.tensor(weights, device=phoneme_ids.device, dtype=torch.float32)
		weights = weights / weights.sum().clamp_min(1e-6)

		raw = weights * total_frames
		base = torch.floor(raw)
		remainder = total_frames - int(base.sum().item())

		if remainder > 0:
			frac = raw - base
			topk = torch.topk(frac, k=remainder).indices
			base[topk] += 1

		targets[i, :n_tokens] = base

	return targets

def compute_loss(
	self,
	outputs: dict,
	mel_spec: torch.Tensor,
	mel_lengths: torch.Tensor,
	phoneme_lengths: torch.Tensor,
	phoneme_ids: torch.Tensor,
	config: dict,
) -> dict:
	pred_duration = outputs['duration'].squeeze(-1)
	max_seq_len = pred_duration.size(1)

	target_duration = self.build_pseudo_duration_targets(
		phoneme_ids=phoneme_ids,
		phoneme_lengths=phoneme_lengths,
		mel_lengths=mel_lengths,
	)

	token_mask = self._make_length_mask(phoneme_lengths, max_seq_len).float()

	token_loss = nn.functional.smooth_l1_loss(
		pred_duration * token_mask,
		target_duration * token_mask,
		reduction="sum",
	) / (token_mask.sum() + 1e-6)

	pred_sum = (pred_duration * token_mask).sum(dim=1)
	target_sum = mel_lengths.float()
	sum_loss = nn.functional.l1_loss(pred_sum, target_sum, reduction="mean")

	alpha = float(config.get("duration_token_alpha", 1.0))
	beta = float(config.get("duration_sum_beta", 0.2))
	duration_loss = alpha * token_loss + beta * sum_loss
```

## Notes For Future Attempts

If phoneme-aware weighting is revisited later, verify these first:

- print one target duration vector and its sum
- print one predicted duration vector and its sum
- verify the speech-rate proxy formula
- confirm training and inference duration handling use the same scale

## Verification Gate

Do not reintroduce phoneme-aware weighting unless the following all hold on the same run:

- target durations sum exactly to the mel length for a sample batch
- predicted duration vectors are in the same numeric scale as the targets
- speech-rate proxy stays in a plausible range, not collapsed toward zero
- inference output duration stays in the healthy range seen in the previous working hybrid setup

## Status

Deprecated experimental branch of the duration supervision work.
