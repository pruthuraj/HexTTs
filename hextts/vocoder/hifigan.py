"""HiFi-GAN vocoder wrapper used by the inference pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml


def _get_padding(kernel_size: int, dilation: int = 1) -> int:
    """Return symmetric Conv1d padding that preserves temporal length."""
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
        for conv in self.convs:
            xt = F.leaky_relu(x, 0.1)
            xt = conv(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for conv in self.convs:
            nn.utils.remove_weight_norm(conv)


class HiFiGANGenerator(nn.Module):
    """HiFi-GAN generator: mel-spectrogram → waveform via stacked transposed convs + ResBlocks."""

    def __init__(self, cfg):
        """Build HiFi-GAN generator from config dictionary."""
        super().__init__()
        self.num_kernels = len(cfg["resblock_kernel_sizes"])
        self.num_upsamples = len(cfg["upsample_rates"])
        self.resblock_type = str(cfg.get("resblock", "1"))

        # Project mel features to the first upsampling channel width.
        self.conv_pre = nn.utils.weight_norm(
            nn.Conv1d(cfg["num_mels"], cfg["upsample_initial_channel"], 7, padding=3)
        )

        # Transposed-conv stack expands time dimension toward waveform resolution.
        self.ups = nn.ModuleList()
        channels = cfg["upsample_initial_channel"]
        for upsample_rate, kernel_size in zip(cfg["upsample_rates"], cfg["upsample_kernel_sizes"]):
            self.ups.append(
                nn.utils.weight_norm(
                    nn.ConvTranspose1d(
                        channels,
                        channels // 2,
                        kernel_size,
                        upsample_rate,
                        padding=(kernel_size - upsample_rate) // 2,
                    )
                )
            )
            channels //= 2

        # Residual blocks refine each upsample stage and suppress checkerboard artifacts.
        self.resblocks = nn.ModuleList()
        channels = cfg["upsample_initial_channel"]
        for _ in range(self.num_upsamples):
            channels //= 2
            for kernel_size, dilations in zip(cfg["resblock_kernel_sizes"], cfg["resblock_dilation_sizes"]):
                if self.resblock_type == "1":
                    self.resblocks.append(ResBlock1(channels, kernel_size, dilations))
                else:
                    self.resblocks.append(ResBlock2(channels, kernel_size, dilations))

        self.conv_post = nn.utils.weight_norm(nn.Conv1d(channels, 1, 7, padding=3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run mel-to-waveform forward pass."""
        x = self.conv_pre(x)
        for i, upsample in enumerate(self.ups):
            x = F.leaky_relu(x, 0.1)
            x = upsample(x)
            xs = torch.zeros_like(x)
            for j in range(self.num_kernels):
                xs = xs + self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x, 0.1)
        x = self.conv_post(x)
        return torch.tanh(x)

    def remove_weight_norm(self):
        """Strip weight normalization for faster/cleaner inference."""
        for upsample in self.ups:
            nn.utils.remove_weight_norm(upsample)
        for resblock in self.resblocks:
            resblock.remove_weight_norm()
        nn.utils.remove_weight_norm(self.conv_pre)
        nn.utils.remove_weight_norm(self.conv_post)


def _load_vocoder_config(config_path: str) -> Dict[str, Any]:
    """Load vocoder config from JSON or YAML file."""
    config_file = Path(config_path)
    with open(config_file, "r", encoding="utf-8") as handle:
        if config_file.suffix.lower() == ".json":
            return json.load(handle)

        cfg = yaml.safe_load(handle)
        if cfg is None:
            raise ValueError(f"Empty vocoder config file: {config_path}")
        return cfg


class HiFiGANVocoder:
    """Loads a pretrained HiFi-GAN and converts mel-spectrograms to waveforms."""

    def __init__(self, checkpoint_path: str, config_path: str, device: torch.device):
        """Load pretrained HiFi-GAN weights and prepare inference graph."""
        cfg = _load_vocoder_config(config_path)

        self.device = device
        self.sample_rate = cfg.get("sampling_rate", 22050)

        self.model = HiFiGANGenerator(cfg).to(device)

        # Checkpoint may store weights either under "generator" or at root level.
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state = checkpoint.get("generator", checkpoint)
        self.model.load_state_dict(state)
        self.model.eval()
        self.model.remove_weight_norm()

    @torch.no_grad()
    def __call__(self, mel: np.ndarray) -> np.ndarray:
        """Synthesize waveform from a single mel spectrogram array."""
        x = torch.FloatTensor(mel).unsqueeze(0).to(self.device)
        y = self.model(x)
        audio = y.squeeze().cpu().numpy()
        audio = audio / (np.abs(audio).max() + 1e-7)
        return audio