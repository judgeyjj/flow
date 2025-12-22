"""
HiFiGAN-style discriminators for speech super-resolution.

Adapted from the HiFiSR open-source code (MPD/MSD + multi-band STFT discriminator).
These are used ONLY when `gan_enabled=True`.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import AvgPool1d, Conv1d, Conv2d
from torch.nn.utils import spectral_norm, weight_norm
from torchaudio.transforms import Spectrogram

LRELU_SLOPE = 0.1


def get_padding(kernel_size: int, dilation: int = 1) -> int:
    return int((kernel_size * dilation - dilation) / 2)


class DiscriminatorP(nn.Module):
    def __init__(self, period: int, kernel_size: int = 5, stride: int = 3, use_spectral_norm: bool = False):
        super().__init__()
        self.period = int(period)
        norm_f = weight_norm if not use_spectral_norm else spectral_norm
        self.convs = nn.ModuleList(
            [
                norm_f(Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
                norm_f(Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
                norm_f(Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
                norm_f(Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
                norm_f(Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0))),
            ]
        )
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        fmap: List[torch.Tensor] = []
        b, c, t = x.shape
        if t % self.period != 0:
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), mode="reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        return x, fmap


class MultiPeriodDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList(
            [DiscriminatorP(2), DiscriminatorP(3), DiscriminatorP(5), DiscriminatorP(7), DiscriminatorP(11)]
        )

    def forward(
        self, y: torch.Tensor, y_hat: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[List[torch.Tensor]], List[List[torch.Tensor]]]:
        y_d_rs: List[torch.Tensor] = []
        y_d_gs: List[torch.Tensor] = []
        fmap_rs: List[List[torch.Tensor]] = []
        fmap_gs: List[List[torch.Tensor]] = []
        for d in self.discriminators:
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DiscriminatorS(nn.Module):
    def __init__(self, use_spectral_norm: bool = False):
        super().__init__()
        norm_f = weight_norm if not use_spectral_norm else spectral_norm
        self.convs = nn.ModuleList(
            [
                norm_f(Conv1d(1, 128, 15, 1, padding=7)),
                norm_f(Conv1d(128, 128, 41, 2, groups=4, padding=20)),
                norm_f(Conv1d(128, 256, 41, 2, groups=16, padding=20)),
                norm_f(Conv1d(256, 512, 41, 4, groups=16, padding=20)),
                norm_f(Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
                norm_f(Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
                norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
            ]
        )
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        fmap: List[torch.Tensor] = []
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        return x, fmap


class MultiScaleDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList([DiscriminatorS(use_spectral_norm=True), DiscriminatorS(), DiscriminatorS()])
        self.meanpools = nn.ModuleList([AvgPool1d(4, 2, padding=2), AvgPool1d(4, 2, padding=2)])

    def forward(
        self, y: torch.Tensor, y_hat: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[List[torch.Tensor]], List[List[torch.Tensor]]]:
        y_d_rs: List[torch.Tensor] = []
        y_d_gs: List[torch.Tensor] = []
        fmap_rs: List[List[torch.Tensor]] = []
        fmap_gs: List[List[torch.Tensor]] = []
        for i, d in enumerate(self.discriminators):
            if i != 0:
                y = self.meanpools[i - 1](y)
                y_hat = self.meanpools[i - 1](y_hat)
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DiscriminatorB(nn.Module):
    """Multi-band, multi-scale STFT discriminator (Descript Audio Codec / Vocos style)."""

    def __init__(
        self,
        window_length: int,
        channels: int = 32,
        hop_factor: float = 0.25,
        bands: Tuple[Tuple[float, float], ...] = (
            (0.0, 0.1),
            (0.1, 0.25),
            (0.25, 0.5),
            (0.5, 0.75),
            (0.75, 1.0),
        ),
    ):
        super().__init__()
        self.window_length = int(window_length)
        self.hop_factor = float(hop_factor)
        self.spec_fn = Spectrogram(
            n_fft=self.window_length,
            hop_length=int(self.window_length * self.hop_factor),
            win_length=self.window_length,
            power=None,
        )
        n_fft = self.window_length // 2 + 1
        self.bands = [(int(b[0] * n_fft), int(b[1] * n_fft)) for b in bands]

        def _stack():
            return nn.ModuleList(
                [
                    weight_norm(nn.Conv2d(2, channels, (3, 9), (1, 1), padding=(1, 4))),
                    weight_norm(nn.Conv2d(channels, channels, (3, 9), (1, 2), padding=(1, 4))),
                    weight_norm(nn.Conv2d(channels, channels, (3, 9), (1, 2), padding=(1, 4))),
                    weight_norm(nn.Conv2d(channels, channels, (3, 9), (1, 2), padding=(1, 4))),
                    weight_norm(nn.Conv2d(channels, channels, (3, 3), (1, 1), padding=(1, 1))),
                ]
            )

        self.band_convs = nn.ModuleList([_stack() for _ in range(len(self.bands))])
        self.conv_post = weight_norm(nn.Conv2d(channels, 1, (3, 3), (1, 1), padding=(1, 1)))

    def spectrogram(self, x: torch.Tensor) -> List[torch.Tensor]:
        # Remove DC offset
        x = x - x.mean(dim=-1, keepdims=True)
        # Peak normalize volume
        x = 0.8 * x / (x.abs().max(dim=-1, keepdim=True)[0] + 1e-9)
        x = self.spec_fn(x)  # (B, F, T) complex
        x = torch.view_as_real(x)  # (B, F, T, 2)
        x = x.permute(0, 3, 2, 1)  # (B, 2, T, F)
        return [x[..., b0:b1] for (b0, b1) in self.bands]

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        x_bands = self.spectrogram(x.squeeze(1))
        fmap: List[torch.Tensor] = []
        outs: List[torch.Tensor] = []
        for band, stack in zip(x_bands, self.band_convs):
            for i, layer in enumerate(stack):
                band = layer(band)
                band = F.leaky_relu(band, 0.1)
                if i > 0:
                    fmap.append(band)
            outs.append(band)
        x_cat = torch.cat(outs, dim=-1)
        x_cat = self.conv_post(x_cat)
        fmap.append(x_cat)
        return x_cat, fmap


class MultiBandDiscriminator(nn.Module):
    def __init__(self, fft_sizes: List[int] | Tuple[int, ...] = (2048, 1024, 512)):
        super().__init__()
        self.fft_sizes = [int(x) for x in fft_sizes]
        self.discriminators = nn.ModuleList([DiscriminatorB(window_length=w) for w in self.fft_sizes])

    def forward(
        self, y: torch.Tensor, y_hat: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[List[torch.Tensor]], List[List[torch.Tensor]]]:
        y_d_rs: List[torch.Tensor] = []
        y_d_gs: List[torch.Tensor] = []
        fmap_rs: List[List[torch.Tensor]] = []
        fmap_gs: List[List[torch.Tensor]] = []
        for d in self.discriminators:
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


