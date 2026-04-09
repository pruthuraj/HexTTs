"""
vocoder.py
Wrapper for pretrained HiFi-GAN vocoder.
Drop-in replacement for Griffin-Lim in the inference pipeline.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import numpy as np
from pathlib import Path

import yaml


# ── Minimal HiFi-GAN Generator (matches the official checkpoint) ──────────

def _get_padding(kernel_size: int, dilation: int = 1) -> int:
    return (kernel_size * dilation - dilation) // 2


class ResBlock1(nn.Module):
    """Official HiFi-GAN ResBlock1 with convs1/convs2 key layout."""

    def __init__(self, channels, kernel_size, dilations):
        super().__init__()
        self.convs1 = nn.ModuleList()
        self.convs2 = nn.ModuleList()

        for d in dilations:
            self.convs1.append(
                nn.utils.weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        dilation=d,
                        padding=_get_padding(kernel_size, d),
                    )
                )
            )
            self.convs2.append(
                nn.utils.weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        dilation=1,
                        padding=_get_padding(kernel_size, 1),
                    )
                )
            )

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, 0.1)
            xt = c1(xt)
            xt = F.leaky_relu(xt, 0.1)
            xt = c2(xt)
            x = xt + x

        return x

    def remove_weight_norm(self):
        for conv in self.convs1:
            nn.utils.remove_weight_norm(conv)
        for conv in self.convs2:
            nn.utils.remove_weight_norm(conv)


class ResBlock2(nn.Module):
    """Official HiFi-GAN ResBlock2 variant."""

    def __init__(self, channels, kernel_size, dilations):
        super().__init__()
        self.convs = nn.ModuleList()
        for d in dilations:
            self.convs.append(
                nn.utils.weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        dilation=d,
                        padding=_get_padding(kernel_size, d),
                    )
                )
            )

    def forward(self, x):
        for c in self.convs:
            xt = F.leaky_relu(x, 0.1)
            xt = c(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for conv in self.convs:
            nn.utils.remove_weight_norm(conv)


class HiFiGANGenerator(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.num_kernels   = len(cfg["resblock_kernel_sizes"])
        self.num_upsamples = len(cfg["upsample_rates"])
        self.resblock_type = str(cfg.get("resblock", "1"))

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
                if self.resblock_type == "1":
                    self.resblocks.append(ResBlock1(ch, k, d))
                else:
                    self.resblocks.append(ResBlock2(ch, k, d))

        self.conv_post = nn.utils.weight_norm(nn.Conv1d(ch, 1, 7, padding=3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_pre(x)
        for i, up in enumerate(self.ups):
            x = F.leaky_relu(x, 0.1)
            x = up(x)
            xs = torch.zeros_like(x)
            for j in range(self.num_kernels):
                xs = xs + self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x, 0.1)
        x = self.conv_post(x)
        return torch.tanh(x)

    def remove_weight_norm(self):
        for up in self.ups:
            nn.utils.remove_weight_norm(up)
        for resblock in self.resblocks:
            resblock.remove_weight_norm()
        nn.utils.remove_weight_norm(self.conv_pre)
        nn.utils.remove_weight_norm(self.conv_post)


# ── Public wrapper ─────────────────────────────────────────────────────────

class HiFiGANVocoder:
    """
    Loads a pretrained HiFi-GAN and converts mel-spectrograms to waveforms.
    Replaces Griffin-Lim with no changes required in training.
    """

    def __init__(self, checkpoint_path: str, config_path: str, device: torch.device):
        config_file = Path(config_path)
        with open(config_file, "r", encoding="utf-8") as f:
            if config_file.suffix.lower() == ".json":
                cfg = json.load(f)
            else:
                cfg = yaml.safe_load(f)
                if cfg is None:
                    raise ValueError(f"Empty vocoder config file: {config_path}")

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
    