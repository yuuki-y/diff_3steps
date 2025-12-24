from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock3D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.norm1 = nn.BatchNorm3d(channels)
        self.norm2 = nn.BatchNorm3d(channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = x
        x = self.act(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return self.act(x + skip)


class CTDecoder(nn.Module):
    """
    Lightweight 3D UNet-style decoder conditioned on latent grid.
    Output channels default to 1 HU volume.
    """

    def __init__(self, in_channels: int = 128, base_channels: int = 64, out_channels: int = 1):
        super().__init__()
        self.up1 = nn.ConvTranspose3d(in_channels, base_channels * 2, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose3d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose3d(base_channels, base_channels // 2, kernel_size=2, stride=2)

        self.res1 = ResidualBlock3D(base_channels * 2)
        self.res2 = ResidualBlock3D(base_channels)
        self.res3 = ResidualBlock3D(base_channels // 2)
        self.out = nn.Conv3d(base_channels // 2, out_channels, kernel_size=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: latent grid (B, C, G, G, G)
        """
        x = self.res1(self.up1(z))
        x = self.res2(self.up2(x))
        x = self.res3(self.up3(x))
        # Final CT volume at 128^3 resolution if grid_size=16 (upsampled 2^3 times)
        return self.out(x)
