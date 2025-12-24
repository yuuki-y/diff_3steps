from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .decoder import CTDecoder
from .encoder import DualViewEncoder
from .fusion import EpipolarFusion
from .geometry import EpipolarMapping, ProjectionGeometry
from .latent import LatentBottleneck


class DrrToCtModel(nn.Module):
    """
    End-to-end model: two DRRs -> latent grid -> CT volume.
    """

    def __init__(self, grid_size: int = 16, grid_channels: int = 16, latent_channels: int = 192):
        super().__init__()
        self.encoder = DualViewEncoder()
        fusion_dim = 192
        self.fusion_dim = fusion_dim
        self.cross_scale_reduce = nn.Conv2d(384 + 192, fusion_dim, kernel_size=1)
        self.fusion = EpipolarFusion(dim=fusion_dim, grid_size=grid_size, grid_channels=grid_channels, cross_depth=2)
        self.bottleneck = LatentBottleneck(in_channels=grid_channels, latent_channels=latent_channels)
        self.decoder = CTDecoder(in_channels=latent_channels, base_channels=96, out_channels=1)

    def forward(
        self,
        drr_a: torch.Tensor,
        drr_b: torch.Tensor,
        geom_a: ProjectionGeometry,
        geom_b: ProjectionGeometry,
        mapping_a2b: Optional[EpipolarMapping] = None,
    ):
        """
        Args:
            drr_a, drr_b: (B, 1, 256, 256) DRRs.
            mapping_a2b: epipolar indices aligned to the fused 16x16 token grid (post pooling).
        """
        feats_a = self.encoder(drr_a, geom_a)
        feats_b = self.encoder(drr_b, geom_b)
        # fuse 1/8 and 1/16 to retain more detail while keeping memory bounded
        def _reduce(feats):
            f16 = feats["1/16"].permute(0, 3, 1, 2)
            f8 = feats["1/8"].permute(0, 3, 1, 2)
            f8_down = F.adaptive_avg_pool2d(f8, f16.shape[-2:])
            fused = torch.cat([f16, f8_down], dim=1)
            return self.cross_scale_reduce(fused).permute(0, 2, 3, 1)

        feat_a = _reduce(feats_a)
        feat_b = _reduce(feats_b)
        grid = self.fusion(feat_a, feat_b, geom_a, geom_b, mapping_a2b)
        mu, logvar, z = self.bottleneck(grid)
        ct = self.decoder(z)
        return {"ct": ct, "mu": mu, "logvar": logvar, "latent": z}

    @staticmethod
    def reconstruction_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.abs(pred - target))

    def loss_fn(self, outputs, target_ct: torch.Tensor, kl_weight: float = 1e-4) -> torch.Tensor:
        l1 = self.reconstruction_loss(outputs["ct"], target_ct)
        kl = self.bottleneck.kl_loss(outputs["mu"], outputs["logvar"])
        return l1 + kl_weight * kl
