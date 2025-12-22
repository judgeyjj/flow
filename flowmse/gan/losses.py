"""GAN loss helpers (HiFiGAN-style)."""

from __future__ import annotations

from typing import List, Tuple

import torch


def feature_loss(fmap_r: List[List[torch.Tensor]], fmap_g: List[List[torch.Tensor]]) -> torch.Tensor:
    loss = 0.0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss = loss + torch.mean(torch.abs(rl - gl))
    return loss * 2.0


def discriminator_loss(
    disc_real_outputs: List[torch.Tensor], disc_generated_outputs: List[torch.Tensor]
) -> Tuple[torch.Tensor, List[float], List[float]]:
    loss = 0.0
    r_losses: List[float] = []
    g_losses: List[float] = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1.0 - dr) ** 2)
        g_loss = torch.mean(dg**2)
        loss = loss + (r_loss + g_loss)
        r_losses.append(float(r_loss.detach().cpu()))
        g_losses.append(float(g_loss.detach().cpu()))
    return loss, r_losses, g_losses


def generator_loss(disc_outputs: List[torch.Tensor]) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    loss = 0.0
    gen_losses: List[torch.Tensor] = []
    for dg in disc_outputs:
        l = torch.mean((1.0 - dg) ** 2)
        gen_losses.append(l)
        loss = loss + l
    return loss, gen_losses


