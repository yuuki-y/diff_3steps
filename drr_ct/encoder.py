from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .geometry import ProjectionGeometry


def _pair(value: int) -> Tuple[int, int]:
    return (value, value)


class PatchEmbed(nn.Module):
    """Convolutional patch embedding with optional positional encoding."""

    def __init__(self, in_chans: int, embed_dim: int, patch_size: int = 4):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        x = self.proj(x)  # (B, C, H/ps, W/ps)
        b, c, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, HW, C)
        x = self.norm(x)
        return x.view(b, h, w, c)


class DepthwiseConvFFN(nn.Module):
    """Depthwise separable FFN for vision tokens."""

    def __init__(self, dim: int, hidden_dim: int, kernel_size: int = 3):
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim)
        self.pw1 = nn.Conv2d(dim, hidden_dim, kernel_size=1)
        self.act = nn.GELU()
        self.pw2 = nn.Conv2d(hidden_dim, dim, kernel_size=1)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, H, W, C)
        b, h, w, c = x.shape
        y = x.permute(0, 3, 1, 2)  # (B, C, H, W)
        y = self.dw(y)
        y = self.pw1(y)
        y = self.act(y)
        y = self.pw2(y)
        y = y.permute(0, 2, 3, 1)
        y = self.norm(y + x)
        return y


class AttentionBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.ffn = DepthwiseConvFFN(dim, hidden_dim=dim * 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, H, W, C)
        b, h, w, c = x.shape
        tokens = x.view(b, h * w, c)
        tokens = self.norm(tokens)
        attn_out, _ = self.attn(tokens, tokens, tokens)
        tokens = tokens + attn_out
        tokens = tokens.view(b, h, w, c)
        tokens = self.ffn(tokens)
        return tokens


class GeometryToken(nn.Module):
    """Encodes projection geometry into per-token embeddings."""

    def __init__(self, dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(16, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(self, geom: ProjectionGeometry, spatial_shape: Tuple[int, int]) -> torch.Tensor:
        # Flatten intrinsics/extrinsics to 16 scalars (coarse token)
        b = geom.intrinsics.shape[0]
        geo_vec = torch.cat(
            [
                geom.intrinsics.view(b, -1),
                geom.extrinsics.view(b, -1),
                geom.source_to_detector.view(b, -1),
            ],
            dim=1,
        )
        geo_vec = geo_vec[:, :16]  # clamp to 16 values for simplicity
        token = self.mlp(geo_vec)  # (B, C)
        h, w = spatial_shape
        return token[:, None, None, :].expand(-1, h, w, -1)


class EncoderStage(nn.Module):
    def __init__(self, dim: int, num_heads: int, depth: int):
        super().__init__()
        self.blocks = nn.ModuleList([AttentionBlock(dim, num_heads) for _ in range(depth)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        return x


class DualViewEncoder(nn.Module):
    """
    Shared-weight encoder for two DRR views.

    Returns multi-scale features keyed by scale name.
    """

    def __init__(self, in_chans: int = 1, base_dim: int = 96, patch_size: int = 4):
        super().__init__()
        dims = [base_dim, base_dim * 2, base_dim * 4]
        heads = [3, 6, 12]
        depths = [2, 2, 2]

        self.patch_embed = PatchEmbed(in_chans, dims[0], patch_size=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, 64, 64, dims[0]))
        self.geom_token = GeometryToken(dims[0])

        self.stages = nn.ModuleList(
            [EncoderStage(dim=d, num_heads=h, depth=dep) for d, h, dep in zip(dims, heads, depths)]
        )
        self.downsamples = nn.ModuleList(
            [
                nn.Conv2d(dims[0], dims[1], kernel_size=2, stride=2),
                nn.Conv2d(dims[1], dims[2], kernel_size=2, stride=2),
            ]
        )

    def _add_pos_and_geom(self, x: torch.Tensor, geom: ProjectionGeometry) -> torch.Tensor:
        b, h, w, c = x.shape
        pos = F.interpolate(self.pos_embed.permute(0, 3, 1, 2), size=(h, w), mode="bilinear", align_corners=False)
        pos = pos.permute(0, 2, 3, 1)
        geo = self.geom_token(geom, (h, w))
        return x + pos + geo

    def forward(self, x: torch.Tensor, geom: ProjectionGeometry) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: (B, C, 256, 256) DRR image.
            geom: projection geometry.
        Returns:
            Dict with multi-scale features at 1/4, 1/8, 1/16 resolutions.
        """
        feats: List[torch.Tensor] = []
        x = self.patch_embed(x)  # (B, 64, 64, C)
        x = self._add_pos_and_geom(x, geom)
        x = self.stages[0](x)
        feats.append(x)

        for idx, down in enumerate(self.downsamples):
            x_conv = down(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            x_conv = self.stages[idx + 1](x_conv)
            feats.append(x_conv)
            x = x_conv

        return {"1/4": feats[0], "1/8": feats[1], "1/16": feats[2]}
