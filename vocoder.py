"""
vocoder.py
Wrapper for pretrained HiFi-GAN vocoder.
Drop-in replacement for Griffin-Lim in the inference pipeline.
"""

import torch
import torch.nn as nn
import json
import numpy as np
from pathlib import Path


# ── Minimal HiFi-GAN Generator (matches the official checkpoint) ──────────

class ResBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilations):
        super().__init__()
        self.convs = nn.ModuleList()
        for d in dilations:
            self.convs.append(nn.Sequential(
                nn.LeakyReLU(0.1),
                nn.utils.weight_norm(nn.Conv1d(
                    channels, channels, kernel_size,
                    dilation=d, padding=(kernel_size * d - d) // 2
                ))
            ))

    def forward(self, x):
        for c in self.convs:
            x = x + c(x)
        return x


class HiFiGANGenerator(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.num_kernels   = len(cfg["resblock_kernel_sizes"])
        self.num_upsamples = len(cfg["upsample_rates"])

        self.conv_pre = nn.utils.weight_norm(
            nn.Conv1d(cfg["num_mels"], cfg["upsample_initial_channel"], 7, padding=3)
        )

        self.ups = nn.ModuleList()
        ch = cfg["upsample_initial_channel"]
        for i, (u, k) in enumerate(zip(cfg["upsample_rates"], cfg["upsample_kernel_sizes"])):
            self.ups.append(nn.utils.weight_norm(
                nn.ConvTranspose1d(ch, ch // 2, k, u, padding=(k - u) // 2)
            ))
            ch //= 2

        self.resblocks = nn.ModuleList()
        ch = cfg["upsample_initial_channel"]
        for i in range(self.num_upsamples):
            ch //= 2
            for k, d in zip(cfg["resblock_kernel_sizes"], cfg["resblock_dilation_sizes"]):
                self.resblocks.append(ResBlock(ch, k, d))

        self.conv_post = nn.utils.weight_norm(nn.Conv1d(ch, 1, 7, padding=3))

    def forward(self, x):
        x = self.conv_pre(x)
        for i, up in enumerate(self.ups):
            x = torch.nn.functional.leaky_relu(x, 0.1)
            x = up(x)
            xs = 0
            for j in range(self.num_kernels):
                xs = xs + self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = torch.nn.functional.leaky_relu(x, 0.1)
        x = self.conv_post(x)
        return torch.tanh(x)

    def remove_weight_norm(self):
        for l in self.ups + self.resblocks:
            for m in l.modules():
                if hasattr(m, 'weight_g'):
                    nn.utils.remove_weight_norm(m)
        nn.utils.remove_weight_norm(self.conv_pre)
        nn.utils.remove_weight_norm(self.conv_post)


# ── Public wrapper ─────────────────────────────────────────────────────────

class HiFiGANVocoder:
    """
    Loads a pretrained HiFi-GAN and converts mel-spectrograms to waveforms.
    Replaces Griffin-Lim with no changes required in training.
    """

    def __init__(self, checkpoint_path: str, config_path: str, device: torch.device):
        with open(config_path) as f:
            cfg = json.load(f)

        self.device      = device
        self.sample_rate = cfg.get("sampling_rate", 22050)

        self.model = HiFiGANGenerator(cfg).to(device)

        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        # Official checkpoint stores generator under 'generator' key
        state = ckpt.get("generator", ckpt)
        self.model.load_state_dict(state)
        self.model.eval()
        self.model.remove_weight_norm()

        print(f"HiFi-GAN loaded from {checkpoint_path}")

    @torch.no_grad()
    def __call__(self, mel: np.ndarray) -> np.ndarray:
        """
        Args:
            mel: (n_mel_channels, time_steps) numpy array, NOT in dB
                 Should be in the same log-scale range your training used.
        Returns:
            audio: (samples,) numpy float32 waveform, normalised to [-1, 1]
        """
        x = torch.FloatTensor(mel).unsqueeze(0).to(self.device)  # (1, 80, T)
        y = self.model(x)                                         # (1, 1, samples)
        audio = y.squeeze().cpu().numpy()
        audio = audio / (np.abs(audio).max() + 1e-7)
        return audio