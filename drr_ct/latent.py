from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class LatentBottleneck(nn.Module):
    """Variational bottleneck for the latent grid."""

    def __init__(self, in_channels: int = 1, latent_channels: int = 128):
        super().__init__()
        self.mu = nn.Conv3d(in_channels, latent_channels, kernel_size=1)
        self.logvar = nn.Conv3d(in_channels, latent_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu = self.mu(x)
        logvar = self.logvar(x)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return mu, logvar, z

    @staticmethod
    def kl_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """KL divergence to unit Gaussian."""
        return 0.5 * torch.sum(torch.exp(logvar) + mu**2 - 1.0 - logvar) / mu.shape[0]
