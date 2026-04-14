"""
VITS Model Implementation
Complete neural network architecture for Text-to-Speech
Patched VITS-like model.

    Main change:
    - expanded text features are projected into latent space and used as a prior
    - posterior latent is combined with the text prior during training
    - inference uses text prior + small noise instead of pure random latent
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

# Standard Arpabet phoneme set size
# VOCAB_SIZE = 149
from hextts.data.raw_dataset import VOCAB_SIZE

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer attention"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model)
        Returns:
            x + positional encoding
        """
        return x + self.pe[:, :x.size(1), :]


class TransformerBlock(nn.Module):
    """Single transformer block with attention and FFN"""
    
    def __init__(self, hidden_size: int, num_heads: int, kernel_size: int, dropout: float):
        super().__init__()
        
        # Self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, hidden_size)
            mask: optional attention mask
        Returns:
            (batch_size, seq_len, hidden_size)
        """
        # Self-attention with residual
        attn_out, _ = self.attention(x, x, x, key_padding_mask=mask)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)
        
        # FFN with residual
        ffn_out = self.ffn(x)
        x = x + self.dropout(ffn_out)
        x = self.norm2(x)
        
        return x


class TextEncoder(nn.Module):
    """Encodes phoneme sequence to embeddings"""
    
    def __init__(self, vocab_size: int, hidden_size: int, num_layers: int,
                 num_heads: int, kernel_size: int, dropout: float):
        super().__init__()
        
        # Embedding layer (phoneme → vector)
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.pos_encoding = PositionalEncoding(hidden_size)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, kernel_size, dropout)
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: phoneme indices (batch_size, seq_len)
            lengths: sequence lengths for masking
        Returns:
            embeddings (batch_size, seq_len, hidden_size)
        """
        # Embed and add positional encoding
        x = self.embedding(x) * math.sqrt(self.embedding.embedding_dim)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # Create mask if lengths provided
        mask = None
        if lengths is not None:
            mask = self._get_mask(x.size(1), lengths, x.device)
        
        # Pass through transformer blocks
        for block in self.transformer_blocks:
            x = block(x, mask=mask)
        
        return x
    
    @staticmethod
    def _get_mask(max_len: int, lengths: torch.Tensor, device: torch.device) -> torch.Tensor:
        """Create attention mask from lengths"""
        batch_size = lengths.size(0)
        mask = torch.arange(max_len, device=device).unsqueeze(0) >= lengths.unsqueeze(1)
        return mask


class DurationPredictor(nn.Module):
    """Predicts duration for each phoneme"""
    
    def __init__(self, hidden_size: int, filters: int, kernel_sizes: list, dropout: float):
        super().__init__()
        
        self.conv_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        in_channels = hidden_size
        for kernel_size in kernel_sizes:
            self.conv_layers.append(
                nn.Conv1d(in_channels, filters, kernel_size, padding=kernel_size // 2)
            )
            self.norms.append(nn.LayerNorm(filters))
            in_channels = filters
        
        self.linear = nn.Linear(filters, 1)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, hidden_size)
        Returns:
            durations (batch_size, seq_len, 1)
        """
        # Transpose for conv1d
        x = x.transpose(1, 2)  # (batch_size, hidden_size, seq_len)
        
        # Pass through conv layers
        for conv, norm in zip(self.conv_layers, self.norms):
            x = conv(x)
            x = x.transpose(1, 2)  # (batch_size, seq_len, filters)
            x = norm(x)
            x = F.relu(x)
            x = self.dropout(x)
            x = x.transpose(1, 2)  # (batch_size, filters, seq_len)
        
        # Predict duration
        x = x.transpose(1, 2)  # (batch_size, seq_len, filters)
        duration = self.linear(x)  # (batch_size, seq_len, 1)
        # duration = torch.clamp(F.softplus(duration), min=1.0)  # Ensure positive
        # clamp duration to prevent extreme values that can cause memory issues during length regulation
        duration = torch.clamp(F.softplus(duration), min=1.0, max=20.0) 
        return duration


class PosteriorEncoder(nn.Module):
    """Encodes mel-spectrogram to latent distribution (training only)"""
    
    def __init__(self, mel_channels: int, hidden_size: int, latent_dim: int):
        super().__init__()
        
        self.conv1 = nn.Conv1d(mel_channels, hidden_size, 1)
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, 1)
        
        self.mu_linear = nn.Linear(hidden_size, latent_dim)
        self.logvar_linear = nn.Linear(hidden_size, latent_dim)
    
    def forward(self, mel_spec: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            mel_spec: (batch_size, mel_channels, time_steps)
        Returns:
            (z, mu, logvar) where z is sampled latent code
        """
        x = F.relu(self.conv1(mel_spec))
        x = F.relu(self.conv2(x))
        x = x.transpose(1, 2)  # (batch_size, time_steps, hidden_size)
        
        mu = self.mu_linear(x)
        logvar = self.logvar_linear(x)
        logvar = torch.clamp(logvar, min=-10.0, max=10.0) # Prevent extreme values
        
        # Reparameterization trick: sample z
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        
        return z, mu, logvar


class Decoder(nn.Module):
    """Decodes latent codes to mel-spectrogram"""
    
    def __init__(self, latent_dim: int, hidden_size: int, mel_channels: int, num_layers: int):
        super().__init__()
        
        # Initial projection
        self.projection = nn.Linear(latent_dim, hidden_size)
        
        # Transformer decoder blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads=2, kernel_size=3, dropout=0.1)
            for _ in range(num_layers)
        ])
        
        # Output layer
        self.output = nn.Linear(hidden_size, mel_channels)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: latent codes (batch_size, time_steps, latent_dim)
        Returns:
            mel-spectrogram (batch_size, mel_channels, time_steps)
        """
        x = self.projection(z)
        residual = x

        # Pass through transformer blocks
        for block in self.transformer_blocks:
            x = block(x)

        # Residual skip helps preserve coarse latent structure
        x = x + residual

        # Output mel-spectrogram
        mel = self.output(x)
        mel = torch.tanh(mel)
        mel = mel.transpose(1, 2)

        return mel

# New v0.4.3: PostNet for mel-spectrogram refinement
class PostNet(nn.Module):
    """
    Refines predicted mel-spectrogram to reduce rough / buzzy artifacts.
    
    PostNet is a convolutional residual network that learns to refine the decoder's 
    mel-spectrogram output by predicting mel-scale residuals. During training/inference,
    the PostNet output is added to the decoder output: refined_mel = decoder_mel + postnet(decoder_mel)
    
    Architecture:
        - Input: Raw mel-spectrogram from decoder
        - Hidden layers: Conv1d with BatchNorm and Tanh activation for non-linearity
        - Output: Residual mel-spectrogram of same shape as input
    """

    def __init__(self, mel_channels: int, hidden_channels: int = 256, kernel_size: int = 5, num_layers: int = 5):
        super().__init__()

        layers = []

        # First layer: Project mel-spectrogram from mel_channels to hidden_channels
        # Conv1d with kernel_size=5 captures local temporal dependencies in mel-spectrogram
        # Batch normalization stabilizes training and accelerates convergence
        # Tanh activation provides bounded non-linearity suitable for spectrogram refinement
        layers.append(
            nn.Sequential(
                nn.Conv1d(mel_channels, hidden_channels, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(hidden_channels),
                nn.Tanh(),
                nn.Dropout(0.1),  # Prevent co-adaptation of features
            )
        )

        # Middle layers: Maintain hidden_channels dimension while refining features
        # Multiple layers allow the network to learn hierarchical mel-scale artifacts
        # Each layer refines the representation through local convolutions and normalization
        for _ in range(num_layers - 2):
            layers.append(
                nn.Sequential(
                    nn.Conv1d(hidden_channels, hidden_channels, kernel_size, padding=kernel_size // 2),
                    nn.BatchNorm1d(hidden_channels),
                    nn.Tanh(),
                    nn.Dropout(0.1),  # Regularization
                )
            )

        # Final layer: Project from hidden_channels back to mel_channels
        # Outputs residuals that will be added to the decoder's mel-spectrogram
        # No activation here (linear output) to allow both positive and negative corrections
        layers.append(
            nn.Sequential(
                nn.Conv1d(hidden_channels, mel_channels, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(mel_channels),
                nn.Dropout(0.1),
            )
        )

        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute mel-spectrogram refinement residuals.
        
        Args:
            x: Mel-spectrogram from decoder (batch_size, mel_channels, time_steps)
        
        Returns:
            Residual mel-spectrogram of same shape: (batch_size, mel_channels, time_steps)
            
        Note:
            The output should be added to the original mel-spectrogram:
            refined_mel = original_mel + postnet(original_mel)
        """
        # Pass through each convolutional layer sequentially
        # Each layer refines the spectrogram representation
        for layer in self.layers:
            x = layer(x)
        
        return x

class VITS(nn.Module):
    """
    Complete VITS Text-to-Speech Model
    Patched VITS-like model.

    Main change:
    - expanded text features are projected into latent space and used as a prior
    - posterior latent is combined with the text prior during training
    - inference uses text prior + small noise instead of pure random latent
    
    Pipeline:
        Phoneme sequence → Text Encoder → Duration Predictor
                                       ↓
                        Duration Expansion (Length Regulator)
                                       ↓
                    Posterior Encoder (training only)
                                       ↓
                        Variational Sampler
                                       ↓
                           Decoder → Mel-spectrogram
    """
    
    def __init__(self, config: dict):
        super().__init__()
        
        self.config = config
        
        # Vocabulary size (number of unique phonemes)
        # Common values: 149 for Arpabet
        self.vocab_size = config.get('vocab_size', 149)
        
        # Text Encoder
        self.encoder = TextEncoder(
            vocab_size=self.vocab_size,
            hidden_size=config['encoder_hidden_size'],
            num_layers=config['encoder_num_layers'],
            num_heads=config['encoder_num_heads'],
            kernel_size=config['encoder_kernel_size'],
            dropout=config['encoder_dropout']
        )
        
        # Duration Predictor
        self.duration_predictor = DurationPredictor(
            hidden_size=config['encoder_hidden_size'],
            filters=config['duration_predictor_filters'],
            kernel_sizes=config['duration_predictor_kernel_sizes'],
            dropout=config['duration_predictor_dropout']
        )
        
        # Posterior Encoder (training only)
        self.posterior_encoder = PosteriorEncoder(
            mel_channels=config['n_mel_channels'],
            hidden_size=config['decoder_hidden_size'],
            latent_dim=config['latent_dim']
        )
        
        # Decoder
        self.decoder = Decoder(
            latent_dim=config['latent_dim'],
            hidden_size=config['decoder_hidden_size'],
            mel_channels=config['n_mel_channels'],
            num_layers=config['decoder_num_layers']
        )
        
        # New v0.4.3: PostNet for mel-spectrogram refinement
        self.postnet = PostNet(mel_channels=config['n_mel_channels'])
        
        # NEW: project expanded text features into latent space
        self.prior_proj = nn.Linear(config["encoder_hidden_size"], config["latent_dim"])
    
    def forward(self, phonemes: torch.Tensor, mel_spec: Optional[torch.Tensor] = None,
                lengths: Optional[torch.Tensor] = None) -> dict:
        """
        Forward pass for training
        
        Args:
            phonemes: (batch_size, seq_len) phoneme indices
            mel_spec: (batch_size, mel_channels, time_steps) ground truth mel-spec (training)
            lengths: (batch_size,) sequence lengths
        
        Returns:
            dict with:
                - predicted_mel: predicted mel-spectrogram
                - duration: predicted duration
                - z, mu, logvar: latent variables (training only)
        """
        
        # Text Encoder: Phonemes → Embeddings
        encoder_out = self.encoder(phonemes, lengths=lengths)  # (batch_size, seq_len, hidden)
        
        # Duration Predictor
        duration = self.duration_predictor(encoder_out)  # (batch_size, seq_len, 1)
        
        # Length Regulator (Expand by duration)
        expanded = self._length_regulate(encoder_out, duration)  # (batch_size, total_len, hidden)
        
        # NEW: text-conditioned latent prior
        prior_latent = self.prior_proj(expanded)  # (batch_size, total_len, latent_dim)
        
        # Get latent codes from posterior encoder during training, sample from prior during inference
        if mel_spec is not None and self.training:
            # Training: use posterior encoder
            z_post, mu, logvar = self.posterior_encoder(mel_spec)

            # match text prior length to mel length
            prior_latent = self._match_time_length(prior_latent, z_post.size(1))

            # combine posterior info with text-conditioned prior
            z = z_post + prior_latent
        else:
            # inference: use text prior + small noise instead of pure random latent
            noise = torch.randn_like(prior_latent) * 0.3
            z = prior_latent + noise
            mu, logvar = None, None
        
        # Decoder: Latent → Mel-spectrogram
        predicted_mel = self.decoder(z)  # (batch_size, mel_channels, time_steps)
        # New v0.4.3: Apply PostNet for mel-spectrogram refinement
        predicted_mel = predicted_mel + self.postnet(predicted_mel)

        return {
            'predicted_mel': predicted_mel,
            'duration': duration,
            'z': z,
            'mu': mu,
            'logvar': logvar
        }
    
    # Inference method for generating mel-spectrogram from phonemes
    def inference(
        self,
        phonemes: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        duration_scale: float = 1.0,
        noise_scale: float = 0.3,
    ) -> torch.Tensor:
        """
        Inference mode: phonemes → predicted mel-spectrogram
        
        Args:
            phonemes: Phoneme indices tensor (batch_size, seq_len)
            lengths: Optional sequence lengths for masking (batch_size,)
            duration_scale: Multiplicative scale for predicted durations (controls speech speed)
            noise_scale: Standard deviation of Gaussian noise added to latent code
        
        Returns:
            predicted_mel: Mel-spectrogram tensor (batch_size, mel_channels, time_steps)
        """
        # Step 1: Text Encoder - Convert phoneme indices to contextual embeddings
        encoder_out = self.encoder(phonemes, lengths=lengths)

        # Step 2: Duration Predictor - Predict frame-level duration for each phoneme
        # Clamp raw predicted durations to a valid range before applying duration_scale
        # so speech-rate control is not aggressively truncated by the ceiling.
        duration = self.duration_predictor(encoder_out)
        duration = torch.clamp(duration, min=1.0, max=20.0)
        # Scale duration by duration_scale parameter (>1.0 = slower, <1.0 = faster)
        duration = duration * duration_scale

        # Step 3: Length Regulator - Expand encoder output by repeating frames according to duration
        # This aligns the text representation with the expected mel-spectrogram length
        expanded = self._length_regulate(encoder_out, duration)

        # Step 4: Project expanded text features into latent space as a prior
        # This text-conditioned prior guides the decoder towards phonetically meaningful representations
        prior_latent = self.prior_proj(expanded)

        # Step 5: Add stochastic noise to the latent code
        # This provides variability in the generated speech while maintaining phonetic content
        noise = torch.randn_like(prior_latent) * noise_scale
        z = prior_latent + noise

        # Step 6: Decode latent code to mel-spectrogram
        # z contains both text information (prior) and stochastic variation (noise)
        predicted_mel = self.decoder(z)
        # New v0.4.3: Apply PostNet for mel-spectrogram refinement
        predicted_mel = predicted_mel + self.postnet(predicted_mel)

        return predicted_mel

    
    @staticmethod
    def _match_time_length(x: torch.Tensor, target_len: int) -> torch.Tensor:
        """
        Resize time dimension to target length.
        x: (B, T, D) -> (B, target_len, D)
        """
        if x.size(1) == target_len:
            return x

        x = x.transpose(1, 2)
        x = F.interpolate(x, size=target_len, mode="linear", align_corners=False)
        x = x.transpose(1, 2)
        return x
    
    @staticmethod
    def _length_regulate(x: torch.Tensor, duration: torch.Tensor) -> torch.Tensor:
        """
        Expand sequence by repeating frames according to predicted duration
        
        Args:
            x: (batch_size, seq_len, hidden_size)
            duration: (batch_size, seq_len, 1) predicted durations
        
        Returns:
            expanded: (batch_size, total_len, hidden_size)
        """
        batch_size, seq_len, hidden_size = x.size()
        
        # Round duration to nearest integer
        duration = torch.round(duration.squeeze(-1)).long()  # (batch_size, seq_len)
        
        outputs = []
        for i in range(batch_size):
            output = []
            for j in range(seq_len):
                # Repeat each frame according to predicted duration
                repeat_count = min(20, max(1, int(duration[i, j].item())))
                output.append(x[i, j].unsqueeze(0).repeat(repeat_count, 1))
            outputs.append(torch.cat(output, dim=0))
        
        # Pad to same length in batch
        max_len = max([o.size(0) for o in outputs])
        padded = torch.zeros(batch_size, max_len, hidden_size, device=x.device)
        
        for i, output in enumerate(outputs):
            padded[i, :output.size(0), :] = output
        
        return padded


if __name__ == "__main__":
    # Test the model
    config = {
        'vocab_size': VOCAB_SIZE,
        'encoder_hidden_size': 384,
        'encoder_num_layers': 4,
        'encoder_num_heads': 2,
        'encoder_kernel_size': 3,
        'encoder_dropout': 0.1,
        'duration_predictor_filters': 256,
        'duration_predictor_kernel_sizes': [3, 3],
        'duration_predictor_dropout': 0.5,
        'n_mel_channels': 80,
        'decoder_hidden_size': 512,
        'decoder_num_layers': 4,
        'latent_dim': 192,
    }
    
    model = VITS(config)
    print("VITS model created successfully!")
    print(f"VOCAB_SIZE: {VOCAB_SIZE}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
