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
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard.writer import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import argparse

from vits_model import VITS,VOCAB_SIZE
# from vits_data import create_dataloaders, get_warning_summary, reset_warning_summary
# Patched to use cached dataloader with warning tracking
from vits_data import create_dataloaders, get_warning_summary, reset_warning_summary

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
        config['vocab_size'] = VOCAB_SIZE
        print(f"Using vocabulary size: {config['vocab_size']}")
        
        self.model = VITS(config).to(device)
        
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Total parameters: {total_params / 1e6:.2f}M")
        
        # Create dataloaders
        print("\nCreating dataloaders...")
        self.train_loader, self.val_loader = create_dataloaders(
            config,
            batch_size=config['batch_size'],
            num_workers=config.get('num_workers', 0)
        )
        
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
        
        # AMP scaler (for mixed precision training)
        self.scaler = GradScaler() if config.get('use_amp', False) else None
        
        # TensorBoard
        self.writer = SummaryWriter(log_dir=config['log_dir'])
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
    
    def compute_loss(self, outputs: dict, mel_spec: torch.Tensor, mel_lengths: torch.Tensor, config: dict) -> dict:
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
        duration = outputs['duration'].squeeze(-1)
        mu = outputs['mu']
        logvar = outputs['logvar']
        
        # Match predicted mel time length to target mel time length
        if predicted_mel.size(2) != mel_spec.size(2):
            predicted_mel = nn.functional.interpolate(
                predicted_mel,
                size=mel_spec.size(2),
                mode="linear",
                align_corners=False,
            )
        # Debug: print duration stats to verify they are reasonable    
        # print("Duration mean:", duration.mean().item())
        # print("Duration std:", duration.std().item())
        # print("Target duration sum:", mel_lengths.float().mean().item())
        
        # 1. Reconstruction loss (L1 or MSE)
        recon_loss = nn.functional.l1_loss(predicted_mel, mel_spec, reduction='mean')
        
        # 2. KL divergence loss (for VAE component)
        if mu is not None and logvar is not None:
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.numel()
        else:
            kl_loss = torch.tensor(0.0, device=predicted_mel.device)
        
        # 3. Duration loss (not computed here as we don't have ground truth durations)
        # In a real implementation, you'd extract durations from alignments
        # duration_loss = torch.tensor(0.0, device=predicted_mel.device) # Old
        
        # NEW: duration supervision by matching total predicted length to mel length
        pred_duration_sum = duration.sum(dim=1)
        target_duration_sum = mel_lengths.float()
        duration_loss = nn.functional.l1_loss(
            pred_duration_sum, target_duration_sum, reduction="mean"
        )
        
        
        
        # Weighted sum
        total_loss = (
            config['loss_weight_reconstruction'] * recon_loss +
            config['loss_weight_kl'] * kl_loss +
            config['loss_weight_duration'] * duration_loss
        )
        
        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss.item(),
            'kl_loss': kl_loss.item(),
            'duration_loss': duration_loss.item(),
        }
    
    def train_epoch(self) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        valid_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch + 1}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            phoneme_ids = batch['phoneme_ids'].to(self.device)
            mel_spec = batch['mel_spec'].to(self.device)
            phoneme_lengths = batch['phoneme_lengths'].to(self.device)
            mel_lengths = batch['mel_lengths'].to(self.device)
            
            # Forward pass
            try:
                with autocast() if self.scaler else torch.enable_grad():
                    outputs = self.model(
                        phoneme_ids,
                        mel_spec=mel_spec,
                        lengths=phoneme_lengths
                    )
                    # Compute loss with optional NaN protection
                    # loss_dict = self.compute_loss(outputs, mel_spec, self.config)
                    loss_dict = self.compute_loss(outputs, mel_spec, mel_lengths, self.config)
                    loss = loss_dict['total_loss']
                    # Check for NaN or Inf loss
                    if torch.isnan(loss) or torch.isinf(loss):
                        print("NaN detected — skipping batch")
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
                        "dur": f'{loss_dict["duration_loss"]:.4f}',
                    })
                    
                    # TensorBoard
                    self.writer.add_scalar('train/loss', avg_loss, self.global_step)
                    self.writer.add_scalar('train/recon_loss', loss_dict['recon_loss'], self.global_step)
                    self.writer.add_scalar('train/kl_loss', loss_dict['kl_loss'], self.global_step)
                    self.writer.add_scalar('train/duration_loss', loss_dict['duration_loss'], self.global_step)
                    self.writer.add_scalar('lr', self.optimizer.param_groups[0]['lr'], self.global_step)

                    # Print warning summary every 500 steps
                    if self.global_step % 500 == 0:
                        print("Printing warning summary...")
                        self.print_warning_summary()
                        reset_warning_summary()
                        
                # Checkpoint
                if self.global_step % self.config.get('checkpoint_interval', 1000) == 0:
                    self.save_checkpoint()
                
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                continue
        # old 
        # avg_epoch_loss = total_loss / len(self.train_loader)
        # return avg_epoch_loss
        # patched to handle potential skipped batches due to NaN loss
        return total_loss / max(1, valid_batches) # Avoid division by zero
        
        
    
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

            outputs = self.model(
                phoneme_ids,
                mel_spec=mel_spec,
                lengths=phoneme_lengths,
            )

            loss_dict = self.compute_loss(outputs, mel_spec, mel_lengths, self.config)
            loss = loss_dict["total_loss"]

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            total_loss += loss.item()
            valid_batches += 1

        avg_loss = total_loss / max(1, valid_batches)
        self.writer.add_scalar("val/loss", avg_loss, self.global_step)
        return avg_loss
    
    def save_checkpoint(self):
        """Save model checkpoint"""
        checkpoint_path = os.path.join(
            self.config['checkpoint_dir'],
            f"checkpoint_step_{self.global_step:06d}.pt"
        )
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'global_step': self.global_step,
            'epoch': self.epoch,
            'config': self.config,
        }, checkpoint_path)
        
        print(f"Saved checkpoint: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load from checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.global_step = checkpoint['global_step']
        self.epoch = checkpoint['epoch']
        
        print(f"Loaded checkpoint from {checkpoint_path}")
    
    def train(self):
        """Train for multiple epochs"""
        num_epochs = self.config['num_epochs']
        
        print(f"\nStarting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Total steps per epoch: {len(self.train_loader)}")
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            
            # Train
            train_loss = self.train_epoch()
            print(f"\nEpoch {epoch + 1}/{num_epochs} - Training loss: {train_loss:.4f}")
            
            # Validate
            val_loss = self.validate()
            print(f"Epoch {epoch + 1}/{num_epochs} - Validation loss: {val_loss:.4f}")
            
            # Learning rate schedule
            if self.scheduler:
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
        summary = get_warning_summary()

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='vits_config.yaml',
                       help='Path to config file')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')
    args = parser.parse_args()
    
    # Load config
    print(f"Loading config from {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
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
