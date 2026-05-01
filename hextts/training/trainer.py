"""
Training Script for VITS TTS Model
Handles the complete training pipeline
Training Script for patched VITS-like TTS model
Fixes:
1. real duration supervision
2. optional NaN protection
3. cleaner logging
"""

import os
from typing import Optional
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard.writer import SummaryWriter
from utils.sample_generation import generate_samples
from torch.amp.grad_scaler import GradScaler
from tqdm import tqdm
import argparse

from hextts.config import load_config
from hextts.data.dataloaders import (
    create_dataloaders as create_shared_dataloaders,
    get_warning_summary,
    reset_warning_summary,
)
from hextts.models.vits import build_vits_model, get_vocab_size
from hextts.models.checkpointing import (
    load_checkpoint as load_shared_checkpoint,
    save_checkpoint as save_shared_checkpoint,
    validate_checkpoint_compatibility,
)
from hextts.training.losses import MultiScaleMelLoss
from hextts.data.raw_dataset import PHONEME_TO_ID

# Sample texts for validation generation (can be customized)
SAMPLE_TEXTS = [
    "Hello, I am HexTTS. This is a training sample.",
    "Neural text to speech is learning step by step.",
]


def _text_to_phoneme_ids(text: str) -> list:
    """Convert text to phoneme IDs via g2p_en + PHONEME_TO_ID.

    Returns None if g2p_en is unavailable or produces empty output,
    so callers can skip sample generation gracefully instead of
    synthesizing garbage from ord(char) % vocab_size.
    """
    try:
        from g2p_en import G2p
        g2p = G2p()
        phonemes = g2p(text)
        ids = []
        for p in phonemes:
            p = p.strip().upper().rstrip("012")
            if not p or p == " ":
                continue
            if p in PHONEME_TO_ID:
                ids.append(PHONEME_TO_ID[p])
        return ids if ids else None
    except Exception:
        return None

class VITSTrainer:
    """Trainer class for VITS model patched with text-conditioned latent prior and improved logging/warning tracking."""
    
    def __init__(self, config: dict, device: torch.device):
        self.config = config
        self.device = device
        
        # Create output directories
        os.makedirs(config['log_dir'], exist_ok=True)
        os.makedirs(config['checkpoint_dir'], exist_ok=True)
        
        # Initialize model
        print("Initializing VITS model...")
        config['vocab_size'] = get_vocab_size()
        print(f"Using vocabulary size: {config['vocab_size']}")

        self.model = build_vits_model(config, device=device)
        
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Total parameters: {total_params / 1e6:.2f}M")
        
        # Create dataloaders
        print("\nCreating dataloaders...")
        self.train_loader, self.val_loader = create_shared_dataloaders(config)
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['learning_rate'],
            betas=(config['adam_beta1'], config['adam_beta2']),
            eps=config['adam_eps']
        )
        
        # Learning rate scheduler
        if config['scheduler_type'] == 'exponential':
            self.scheduler = optim.lr_scheduler.ExponentialLR(
                self.optimizer,
                gamma=config['scheduler_gamma']
            )
        else:
            self.scheduler = None

        # Per-step LR warmup (runs independently of the per-epoch main scheduler)
        self.warmup_total_steps = int(config.get('warmup_steps', 0))
        if self.warmup_total_steps > 0:
            self.warmup_scheduler = optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=1.0 / self.warmup_total_steps,
                end_factor=1.0,
                total_iters=self.warmup_total_steps,
            )
        else:
            self.warmup_scheduler = None

        # Multi-scale mel loss (no trainable params, created once).
        # scale=8 adds sentence-level prosody enforcement on top of frame/window detail.
        self.ms_mel_loss = MultiScaleMelLoss(scales=(1, 2, 4, 8))
        
        # AMP scaler (for mixed precision training)
        # Use new torch.amp.GradScaler API with device specification
        self.scaler = GradScaler(self.device.type) if config.get('use_amp', False) else None
        
        # TensorBoard
        self.writer = SummaryWriter(log_dir=config['log_dir'])
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0

    @staticmethod
    def _make_length_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
        """Create boolean mask where True marks valid (non-pad) token positions."""
        return torch.arange(max_len, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)

    @staticmethod
    def _build_pseudo_duration_targets(
        phoneme_lengths: torch.Tensor,
        mel_lengths: torch.Tensor,
        max_seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Build token-level pseudo duration targets with sum-preserving allocation.

        For each sample:
          base = T // N
          remainder = T % N
          first `remainder` tokens get base + 1, others get base

        This guarantees sum(target[:N]) == T exactly.
        """
        batch_size = phoneme_lengths.size(0)
        targets = torch.zeros(batch_size, max_seq_len, dtype=torch.long, device=device)

        for i in range(batch_size):
            n_tokens = max(1, int(phoneme_lengths[i].item()))
            n_tokens = min(n_tokens, max_seq_len)
            total_frames = max(0, int(mel_lengths[i].item()))

            base = total_frames // n_tokens
            remainder = total_frames % n_tokens

            targets[i, :n_tokens] = base
            if remainder > 0:
                targets[i, :remainder] += 1

        return targets
    
    def compute_loss(
        self,
        outputs: dict,
        mel_spec: torch.Tensor,
        phoneme_lengths: torch.Tensor,
        mel_lengths: torch.Tensor,
        config: dict,
        duration_targets: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Compute total loss
        
        Args:
            outputs: model output dict
            mel_spec: ground truth mel-spectrogram
            mel_lengths: lengths of the mel-spectrograms
            config: config dict with loss weights
        
        Returns:
            dict with individual losses and total loss
        """
        
        predicted_mel = outputs['predicted_mel']
        decoder_mel = outputs.get('decoder_mel')
        duration = outputs['duration'].squeeze(-1)
        mu = outputs['mu']
        logvar = outputs['logvar']

        # Align predicted lengths to target mel length when duration rounding causes a mismatch.
        # Ideally this fires rarely; large or frequent mismatches indicate duration supervision drift.
        # The mismatch magnitude is logged to TensorBoard as train/length_mismatch_frames.
        length_mismatch = abs(predicted_mel.size(2) - mel_spec.size(2))
        if predicted_mel.size(2) != mel_spec.size(2):
            predicted_mel = nn.functional.interpolate(
                predicted_mel, size=mel_spec.size(2), mode="linear", align_corners=False,
            )
        if decoder_mel is not None and decoder_mel.size(2) != mel_spec.size(2):
            decoder_mel = nn.functional.interpolate(
                decoder_mel, size=mel_spec.size(2), mode="linear", align_corners=False,
            )

        # 1. Reconstruction loss — post-PostNet (primary)
        recon_loss = nn.functional.l1_loss(predicted_mel, mel_spec, reduction='mean')

        # 2. Pre-PostNet mel loss — teaches the decoder to produce a good coarse mel
        #    independently of PostNet, giving PostNet a clear residual to learn.
        if decoder_mel is not None:
            pre_postnet_loss = nn.functional.l1_loss(decoder_mel, mel_spec, reduction='mean')
        else:
            pre_postnet_loss = predicted_mel.new_zeros(())

        # 3. Multi-scale spectral loss — L1 at temporal resolutions 1×, 2×, 4×
        ms_loss = self.ms_mel_loss(predicted_mel, mel_spec)

        # 4. KL divergence with annealing — ramps from 0 to loss_weight_kl over
        #    kl_warmup_steps to prevent posterior collapse in early training.
        if mu is not None and logvar is not None:
            kl_warmup = int(config.get('kl_warmup_steps', 0))
            kl_anneal = min(1.0, self.global_step / max(1, kl_warmup)) if kl_warmup > 0 else 1.0
            kl_raw = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.numel()
            kl_loss = kl_raw * kl_anneal
        else:
            kl_anneal = 0.0
            kl_loss = predicted_mel.new_zeros(())
        
        # 3. Duration loss: real MFA targets when available, pseudo-uniform fallback otherwise.
        pred = duration
        max_seq_len = pred.size(1)

        using_real_targets = (
            duration_targets is not None
            and duration_targets.size(0) == pred.size(0)
            and duration_targets.size(1) >= max_seq_len
        )

        if using_real_targets:
            target = duration_targets[:, :max_seq_len].to(pred.device)
        else:
            target = self._build_pseudo_duration_targets(
                phoneme_lengths=phoneme_lengths,
                mel_lengths=mel_lengths,
                max_seq_len=max_seq_len,
                device=pred.device,
            )

        token_mask = self._make_length_mask(phoneme_lengths, max_seq_len)
        token_mask_f = token_mask.float()

        # Token-level SmoothL1 over valid (non-pad) tokens only
        token_loss_raw = nn.functional.smooth_l1_loss(
            pred,
            target.float(),
            reduction='none',
        )
        token_loss = (token_loss_raw * token_mask_f).sum() / token_mask_f.sum().clamp_min(1.0)

        # Sum-level L1 encourages total predicted duration to match mel length
        pred_duration_sum = (pred * token_mask_f).sum(dim=1)
        target_duration_sum = mel_lengths.float()
        sum_loss = nn.functional.l1_loss(pred_duration_sum, target_duration_sum, reduction='mean')

        alpha = float(config.get('duration_token_alpha', 1.0))
        beta = float(config.get('duration_sum_beta', 0.2))
        duration_loss = alpha * token_loss + beta * sum_loss

        token_mae = (torch.abs(pred - target.float()) * token_mask_f).sum() / token_mask_f.sum().clamp_min(1.0)
        sum_abs_error = torch.abs(pred_duration_sum - target_duration_sum)
        speech_rate_proxy = pred_duration_sum / phoneme_lengths.float().clamp_min(1.0)
        
        
        # Weighted sum
        total_loss = (
            config['loss_weight_reconstruction'] * recon_loss +
            config.get('loss_weight_pre_postnet', 0.5) * pre_postnet_loss +
            config.get('loss_weight_stft', 0.1) * ms_loss +
            config['loss_weight_kl'] * kl_loss +
            config['loss_weight_duration'] * duration_loss
        )

        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss.item(),
            'pre_postnet_loss': pre_postnet_loss.item() if torch.is_tensor(pre_postnet_loss) else float(pre_postnet_loss),
            'ms_mel_loss': ms_loss.item(),
            'kl_loss': kl_loss.item(),
            'kl_anneal_factor': kl_anneal,
            'duration_loss': duration_loss.item(),
            'token_duration_loss': token_loss.item(),
            'sum_duration_loss': sum_loss.item(),
            'pred_duration_sum_mean': pred_duration_sum.mean().item(),
            'target_duration_sum_mean': target_duration_sum.mean().item(),
            'token_duration_mae': token_mae.item(),
            'sum_error_mean': sum_abs_error.mean().item(),
            'speech_rate_proxy_mean': speech_rate_proxy.mean().item(),
            'length_mismatch_frames': length_mismatch,
            'using_real_duration_targets': int(using_real_targets),
        }

    def log_duration_debug(
        self,
        prefix: str,
        batch: dict,
        outputs: dict,
        loss_dict: dict,
    ) -> None:
        """Print one duration sample for scale and proxy verification when debugging is enabled."""
        if not self.config.get('duration_debug_checks', False):
            return

        try:
            sample_idx = 0
            phoneme_lengths = batch['phoneme_lengths']
            mel_lengths = batch['mel_lengths']

            predicted_duration = outputs['duration'].squeeze(-1).detach()
            target_duration = self._build_pseudo_duration_targets(
                phoneme_lengths=phoneme_lengths,
                mel_lengths=mel_lengths,
                max_seq_len=predicted_duration.size(1),
                device=predicted_duration.device,
            )

            n_tokens = min(int(phoneme_lengths[sample_idx].item()), predicted_duration.size(1))
            pred_vec = predicted_duration[sample_idx, :n_tokens].cpu().tolist()
            target_vec = target_duration[sample_idx, :n_tokens].cpu().tolist()
            pred_sum = float(sum(pred_vec))
            target_sum = float(sum(target_vec))

            print(f"[{prefix}] duration debug sample={sample_idx}")
            print(f"[{prefix}] phoneme_length={int(phoneme_lengths[sample_idx].item())} mel_length={int(mel_lengths[sample_idx].item())}")
            print(f"[{prefix}] target_duration={target_vec}")
            print(f"[{prefix}] predicted_duration={pred_vec}")
            print(f"[{prefix}] target_sum={target_sum:.4f} pred_sum={pred_sum:.4f}")
            print(f"[{prefix}] proxy_formula=pred_sum / phoneme_length -> {loss_dict['speech_rate_proxy_mean']:.6f}")
        except Exception as exc:
            print(f"[{prefix}] duration debug failed: {exc}")
    
    def train_epoch(self) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        valid_batches = 0
        skipped_batches = 0
        epoch_skip_reasons: dict = {}

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch + 1}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            phoneme_ids = batch['phoneme_ids'].to(self.device)
            mel_spec = batch['mel_spec'].to(self.device)
            phoneme_lengths = batch['phoneme_lengths'].to(self.device)
            mel_lengths = batch['mel_lengths'].to(self.device)
            duration_targets = batch.get('duration_targets')
            if duration_targets is not None:
                duration_targets = duration_targets.to(self.device)
            
            # Forward pass with optional autocast for mixed precision
            try:
                if self.scaler:
                    autocast_context = torch.autocast(device_type=self.device.type)
                else:
                    autocast_context = torch.autocast(device_type=self.device.type, enabled=False)

                with autocast_context:
                    outputs = self.model(
                        phoneme_ids,
                        mel_spec=mel_spec,
                        lengths=phoneme_lengths,
                        mel_lengths=mel_lengths,
                    )
                    loss_dict = self.compute_loss(
                        outputs, mel_spec, phoneme_lengths, mel_lengths,
                        self.config, duration_targets=duration_targets,
                    )

                    if batch_idx == 0:
                        self.log_duration_debug("train", batch, outputs, loss_dict)
                    
                    max_duration_value = self.config.get("max_duration_value", 20.0)
                    if outputs['duration'].max().item() > max_duration_value:
                        reason = "extreme_duration"
                        epoch_skip_reasons[reason] = epoch_skip_reasons.get(reason, 0) + 1
                        skipped_batches += 1
                        print(f"[skip] batch {batch_idx}: extreme duration ({outputs['duration'].max().item():.1f} > {max_duration_value})")
                        continue

                    loss = loss_dict['total_loss']
                    if torch.isnan(loss):
                        reason = "nan_loss"
                        epoch_skip_reasons[reason] = epoch_skip_reasons.get(reason, 0) + 1
                        skipped_batches += 1
                        print(f"[skip] batch {batch_idx}: NaN loss")
                        continue
                    if torch.isinf(loss):
                        reason = "inf_loss"
                        epoch_skip_reasons[reason] = epoch_skip_reasons.get(reason, 0) + 1
                        skipped_batches += 1
                        print(f"[skip] batch {batch_idx}: Inf loss")
                        continue
                
                # Backward pass
                self.optimizer.zero_grad()
                
                if self.scaler:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                else:
                    loss.backward()
                
                # Gradient clipping
                if self.config.get('grad_clip_val', 0) > 0:
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['grad_clip_val']
                    )
                
                # Optimizer step
                if self.scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                # Per-step LR warmup (only active during warmup phase)
                if self.warmup_scheduler is not None and self.global_step < self.warmup_total_steps:
                    self.warmup_scheduler.step()

                # Log
                total_loss += loss.item()
                valid_batches += 1
                self.global_step += 1
                
                if self.global_step % self.config.get('log_interval', 100) == 0:
                    avg_loss = total_loss / max(1, valid_batches) # Avoid division by zero
                    pbar.set_postfix({
                        'loss': f'{avg_loss:.4f}',
                        'recon': f'{loss_dict["recon_loss"]:.4f}',
                        'kl': f'{loss_dict["kl_loss"]:.4f}',
                        'dur': f'{loss_dict["duration_loss"]:.4f}',
                        'skip': skipped_batches,
                    })
                    
                    # TensorBoard
                    self.writer.add_scalar('train/loss', avg_loss, self.global_step)
                    self.writer.add_scalar('train/recon_loss', loss_dict['recon_loss'], self.global_step)
                    self.writer.add_scalar('train/pre_postnet_loss', loss_dict['pre_postnet_loss'], self.global_step)
                    self.writer.add_scalar('train/ms_mel_loss', loss_dict['ms_mel_loss'], self.global_step)
                    self.writer.add_scalar('train/kl_loss', loss_dict['kl_loss'], self.global_step)
                    self.writer.add_scalar('train/kl_anneal_factor', loss_dict['kl_anneal_factor'], self.global_step)
                    self.writer.add_scalar('train/duration_loss', loss_dict['duration_loss'], self.global_step)
                    self.writer.add_scalar('train/duration_token_loss', loss_dict['token_duration_loss'], self.global_step)
                    self.writer.add_scalar('train/duration_sum_loss', loss_dict['sum_duration_loss'], self.global_step)
                    self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], self.global_step)
                    self.writer.add_scalar('train/skipped_batches', skipped_batches, self.global_step)
                    self.writer.add_scalar('train/length_mismatch_frames', loss_dict['length_mismatch_frames'], self.global_step)
                    self.writer.add_scalar('train/using_real_duration_targets', loss_dict['using_real_duration_targets'], self.global_step)

                    # Duration diagnostics
                    self.writer.add_scalar('train/pred_duration_sum_mean', loss_dict['pred_duration_sum_mean'], self.global_step)
                    self.writer.add_scalar('train/target_duration_sum_mean', loss_dict['target_duration_sum_mean'], self.global_step)
                    self.writer.add_scalar('train/token_duration_mae', loss_dict['token_duration_mae'], self.global_step)
                    self.writer.add_scalar('train/sum_error_mean', loss_dict['sum_error_mean'], self.global_step)
                    self.writer.add_scalar('train/pred_speech_rate_proxy', loss_dict['speech_rate_proxy_mean'], self.global_step)
                    self.writer.add_scalar('train/duration_max', outputs['duration'].max().item(), self.global_step)
                    self.writer.add_scalar('train/duration_min', outputs['duration'].min().item(), self.global_step)

                    # Mel/output diagnostics
                    self.writer.add_scalar('train/predicted_mel_length', outputs['predicted_mel'].shape[2], self.global_step)
                    self.writer.add_scalar('train/target_mel_length_mean', mel_lengths.float().mean().item(), self.global_step)
                    self.writer.add_scalar('train/predicted_mel_max', outputs['predicted_mel'].max().item(), self.global_step)
                    self.writer.add_scalar('train/predicted_mel_min', outputs['predicted_mel'].min().item(), self.global_step)

                    self.writer.add_histogram(
                        "duration_predictions",
                        outputs['duration'].detach().cpu(),
                        self.global_step
                    )

                    # Print warning summary every 500 steps
                    if self.global_step % 500 == 0:
                        print("Printing warning summary...")
                        self.print_warning_summary()
                        reset_warning_summary(self.config)
                        
                # Checkpoint
                if self.global_step % self.config.get('checkpoint_interval', 1000) == 0:
                    self.save_checkpoint()
                
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                continue

        # End-of-epoch skip summary
        total_batches = valid_batches + skipped_batches
        if skipped_batches > 0:
            skip_ratio = skipped_batches / max(1, total_batches)
            print(f"\n[skip summary] epoch {self.epoch + 1}: {skipped_batches}/{total_batches} batches skipped ({skip_ratio:.1%})")
            for reason, count in epoch_skip_reasons.items():
                print(f"  {reason}: {count}")
            self.writer.add_scalar('train/epoch_skip_ratio', skip_ratio, self.epoch)
            max_skip_ratio = float(self.config.get('max_skipped_ratio', 0.5))
            if skip_ratio > max_skip_ratio:
                raise RuntimeError(
                    f"Training aborted: {skip_ratio:.1%} of batches were skipped in epoch {self.epoch + 1} "
                    f"(threshold: {max_skip_ratio:.1%}). "
                    "Check for NaN/Inf in model outputs or extreme duration values."
                )

        return total_loss / max(1, valid_batches)
        
        
    
    @torch.no_grad()
    def validate(self) -> float:
        self.model.eval()
        total_loss = 0.0
        valid_batches = 0

        pbar = tqdm(self.val_loader, desc="Validation")

        for batch in pbar:
            phoneme_ids = batch["phoneme_ids"].to(self.device)
            mel_spec = batch["mel_spec"].to(self.device)
            phoneme_lengths = batch["phoneme_lengths"].to(self.device)
            mel_lengths = batch["mel_lengths"].to(self.device)
            duration_targets = batch.get("duration_targets")
            if duration_targets is not None:
                duration_targets = duration_targets.to(self.device)

            outputs = self.model(
                phoneme_ids,
                mel_spec=mel_spec,
                lengths=phoneme_lengths,
                mel_lengths=mel_lengths,
            )

            loss_dict = self.compute_loss(
                outputs, mel_spec, phoneme_lengths, mel_lengths,
                self.config, duration_targets=duration_targets,
            )
            loss = loss_dict["total_loss"]

            if len(loss_dict) > 0 and valid_batches == 0:
                self.log_duration_debug("val", batch, outputs, loss_dict)
            
            # New v0.4.3: NaN protection for validation - skip batch if loss is NaN or Inf 
            if torch.isnan(loss) or torch.isinf(loss):
                print("Invalid loss detected — skipping batch")
                continue
            
            # New v0.4.3: Additional NaN protection for validation - skip batch if any output tensor contains NaN or Inf values
            bad_tensor = False

            for name, value in outputs.items():
                if torch.is_tensor(value):
                    if torch.isnan(value).any() or torch.isinf(value).any():
                        print(f"Invalid output tensor detected in {name} — skipping batch")
                        bad_tensor = True
                        break

            if bad_tensor:
                continue
            
            # Validation diagnostics
            self.writer.add_scalar("val/pred_duration_sum_mean", loss_dict['pred_duration_sum_mean'], self.global_step)
            self.writer.add_scalar("val/target_duration_sum_mean", loss_dict['target_duration_sum_mean'], self.global_step)
            self.writer.add_scalar("val/token_duration_mae", loss_dict['token_duration_mae'], self.global_step)
            self.writer.add_scalar("val/sum_error_mean", loss_dict['sum_error_mean'], self.global_step)
            self.writer.add_scalar("val/pred_speech_rate_proxy", loss_dict['speech_rate_proxy_mean'], self.global_step)

            self.writer.add_scalar(
                "val/duration_max",
                outputs['duration'].max().item(),
                self.global_step
            )

            self.writer.add_scalar(
                "val/duration_min",
                outputs['duration'].min().item(),
                self.global_step
            )

            self.writer.add_scalar(
                "val/predicted_mel_max",
                outputs['predicted_mel'].max().item(),
                self.global_step
            )

            self.writer.add_scalar(
                "val/predicted_mel_min",
                outputs['predicted_mel'].min().item(),
                self.global_step
            )

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            total_loss += loss.item()
            valid_batches += 1

        avg_loss = total_loss / max(1, valid_batches)
        self.writer.add_scalar("val/loss", avg_loss, self.global_step)
        return avg_loss
    
    @torch.no_grad()
    def log_audio_samples(self, epoch: int):
        """
        Log predicted mel-spectrograms to TensorBoard as images.
        
        This method generates mel-spectrograms from sample texts and visualizes them
        as heatmaps in TensorBoard for qualitative evaluation during training.
        No gradients are computed (inference-only).
        """
        # Switch to evaluation mode (disables dropout, batch norm, etc.)
        self.model.eval()

        # Process each sample text
        for i, text in enumerate(SAMPLE_TEXTS):
            try:
                seq = _text_to_phoneme_ids(text)
                if not seq:
                    print(f"Skipping TensorBoard sample {i}: g2p unavailable or empty output")
                    continue

                x = torch.LongTensor(seq).unsqueeze(0).to(self.device)
                # Phoneme sequence length for masking during inference
                x_lengths = torch.LongTensor([x.size(1)]).to(self.device)

                # Forward pass: inference returns mel-spectrogram (B, n_mel_channels, T)
                mel = self.model.inference(x, lengths=x_lengths)

                # Remove batch dimension if present (some model versions return (1, 80, T))
                if mel.dim() == 3:
                    mel = mel.squeeze(0)   # Now shape (80, T)

                # Move to CPU and detach from computation graph (no grad tracking needed)
                mel = mel.detach().cpu()

                # Normalize mel-spectrogram using a fixed/global range so TensorBoard
                # images are comparable across different samples and epochs.
                # Prefer dataset-wide statistics from config when available.
                mel_min = float(self.config.get('tensorboard_mel_min', -11.5))
                mel_max = float(self.config.get('tensorboard_mel_max', 2.5))
                if mel_max <= mel_min:
                    raise ValueError("tensorboard_mel_max must be greater than tensorboard_mel_min")
                mel = mel.clamp(min=mel_min, max=mel_max)
                mel = (mel - mel_min) / (mel_max - mel_min)

                # Expand to 3D for TensorBoard: (C, H, W) = (1, 80, T)
                # TensorBoard expects format: (channels, height, width)
                mel = mel.unsqueeze(0)

                # Log as image heatmap in TensorBoard
                self.writer.add_image(
                    tag=f"sample_mel_{i}",           # Unique tag for each sample
                    img_tensor=mel,                   # Normalized mel-spectrogram
                    global_step=epoch,                # X-axis: training step/epoch
                    dataformats="CHW",                # Channel-Height-Width format
                )

            except Exception as e:
                # Fail gracefully: log error but don't crash training
                print(f"TensorBoard mel logging failed: {e}")

        # Switch back to training mode (re-enables dropout, batch norm updates, etc.)
        self.model.train()
    
    def save_checkpoint(self):
        """Save model checkpoint"""
        checkpoint_path = os.path.join(
            self.config['checkpoint_dir'],
            f"checkpoint_step_{self.global_step:06d}.pt"
        )

        save_shared_checkpoint(
            path=checkpoint_path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            epoch=self.epoch,
            global_step=self.global_step,
            config=self.config,
        )
        
        print(f"Saved checkpoint: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load from checkpoint"""
        checkpoint = load_shared_checkpoint(checkpoint_path, device=self.device)

        validate_checkpoint_compatibility(checkpoint, self.config)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.global_step = int(checkpoint.get('global_step', 0))
        self.epoch = int(checkpoint.get('epoch', 0))
        
        # Restore GradScaler state if using AMP
        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        print(f"Loaded checkpoint from {checkpoint_path}")
        print(f"Checkpoint epoch value: {self.epoch}")
        print(f"Checkpoint global step: {self.global_step}")
    
    def train(self):
        """Train for multiple epochs"""
        num_epochs = self.config['num_epochs']
        
        print(f"\nStarting training for {num_epochs} epochs...")
        # Print resumption info if continuing from checkpoint
        if self.epoch > 0 or self.global_step > 0:
            print(f"Resuming from epoch {self.epoch}, global_step {self.global_step}")
        
        print(f"Device: {self.device}")
        print(f"Total steps per epoch: {len(self.train_loader)}")
        
        # Print initial warning summary before training starts
        print("Initial warning summary before training:")
        self.print_warning_summary()
        reset_warning_summary(self.config)
        
        # Main training loop - continues from current epoch (handles mid-epoch resumption)
        for epoch in range(self.epoch, num_epochs):
            self.epoch = epoch
            
            # Train
            train_loss = self.train_epoch()
            print(f"\nEpoch {epoch + 1}/{num_epochs} - Training loss: {train_loss:.4f}")
            
            # Validate
            val_loss = self.validate()
            print(f"Epoch {epoch + 1}/{num_epochs} - Validation loss: {val_loss:.4f}")
            
            # Generate audio samples every 5 epochs
            if (epoch + 1) % 5 == 0:
                if _text_to_phoneme_ids(SAMPLE_TEXTS[0]) is None:
                    print("Skipping sample generation: g2p_en unavailable or returned empty output")
                else:
                    try:
                        generate_samples(
                            model=self.model,
                            texts=SAMPLE_TEXTS,
                            text_to_sequence_fn=_text_to_phoneme_ids,
                            output_dir="samples",
                            epoch=epoch + 1,
                            device=self.device.type,
                            sample_rate=self.config.get("sample_rate", 22050),
                            config=self.config,
                        )
                        self.log_audio_samples(epoch + 1)
                        print(f"Saved audio samples for epoch {epoch + 1}")
                    except Exception as e:
                        print(f"Sample generation failed: {e}")
            
            # Learning rate schedule (epoch-based; held back until warmup finishes)
            if self.scheduler and self.global_step >= self.warmup_total_steps:
                self.scheduler.step()
            
            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                # Save best model
                best_path = os.path.join(self.config['checkpoint_dir'], 'best_model.pt')
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'config': self.config,
                }, best_path)
                
                print(f"New best validation loss: {val_loss:.4f} (saved to {best_path})")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config.get('early_stopping_patience', 10):
                    print(f"Early stopping after {self.patience_counter} epochs without improvement")
                    break
        
        print("\nTraining complete!")
        self.writer.close()
        
        
    def print_warning_summary(self):
        """Print summarized dataset/data-loader warnings."""
        summary = get_warning_summary(self.config)

        has_any_warning = any(len(items) > 0 for items in summary.values())
        if not has_any_warning:
            return

        print("\nWarning summary:")
        print("-" * 50)

        if summary["unknown_phoneme"]:
            total_unknown = sum(summary["unknown_phoneme"].values())
            print(f"Unknown phoneme warnings: {total_unknown}")
            for phoneme, count in sorted(summary["unknown_phoneme"].items(), key=lambda x: x[1], reverse=True)[:10]:
                print(f"  {phoneme}: {count}")

        if summary["audio_load_error"]:
            total_audio_errors = sum(summary["audio_load_error"].values())
            print(f"Audio load errors: {total_audio_errors}")
            for filename, count in sorted(summary["audio_load_error"].items(), key=lambda x: x[1], reverse=True)[:10]:
                print(f"  {filename}: {count}")

        print("-" * 50)


def run_training(config: dict, device: torch.device, checkpoint: str | None = None) -> None:
    """Run training using the package-owned trainer implementation."""
    trainer = VITSTrainer(config, device)
    if checkpoint:
        trainer.load_checkpoint(checkpoint)
    trainer.train()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/base.yaml',
                       help='Path to config file')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')
    args = parser.parse_args()
    
    # Load config
    print(f"Loading config from {args.config}")
    config = load_config(args.config)
    
    # Set device
    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU (training will be slow)")
    
    # Create trainer
    trainer = VITSTrainer(config, device)
    
    # Resume from checkpoint if provided
    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)
    
    # Train
    trainer.train()


if __name__ == "__main__":
    main()
