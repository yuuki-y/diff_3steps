from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .geometry import EpipolarMapping, ProjectionGeometry


class EpipolarAttention(nn.Module):
    """
    Sparse attention along precomputed epipolar correspondences.
    If no mapping is provided, falls back to standard multi-head attention.
    Includes optional token pruning to limit memory use.
    """

    def __init__(self, dim: int, num_heads: int, k: int = 32, prune_tokens: int = 1024):
        super().__init__()
        self.num_heads = num_heads
        self.k = k
        self.prune_tokens = prune_tokens
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
        self.scale = (dim // num_heads) ** -0.5
        self.fallback_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)

    def _maybe_prune(self, src: torch.Tensor, mapping: Optional[EpipolarMapping]) -> Tuple[torch.Tensor, Optional[EpipolarMapping]]:
        if mapping is None or src.shape[1] <= self.prune_tokens:
            return src, mapping
        # simple L2-norm token scoring
        scores = torch.norm(src, dim=-1)
        topk = min(self.prune_tokens, src.shape[1])
        _, idx = torch.topk(scores, k=topk, dim=1)
        pruned_src = torch.gather(src, 1, idx.unsqueeze(-1).expand(-1, -1, src.shape[-1]))

        if mapping is None:
            return pruned_src, None
        # remap epipolar indices to pruned token order (approximate by clipping)
        new_indices = torch.clamp(mapping.indices, max=topk - 1)
        new_weights = mapping.weights
        return pruned_src, EpipolarMapping(indices=new_indices, weights=new_weights, shape_hw=mapping.shape_hw)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        mapping: Optional[EpipolarMapping] = None,
    ) -> torch.Tensor:
        b, hw, c = src.shape
        if mapping is None or mapping.indices.numel() == 0:
            out, _ = self.fallback_attn(src, tgt, tgt)
            return out

        src, mapping = self._maybe_prune(src, mapping)

        dim_head = c // self.num_heads
        q = self.query(src).view(b, hw, self.num_heads, dim_head).transpose(1, 2)  # (B, heads, HW, Dh)
        k = self.key(tgt).view(b, tgt.shape[1], self.num_heads, dim_head).transpose(1, 2)  # (B, heads, HWb, Dh)
        v = self.value(tgt).view(b, tgt.shape[1], self.num_heads, dim_head).transpose(1, 2)

        idx = mapping.indices.unsqueeze(1).unsqueeze(-1).expand(-1, self.num_heads, -1, -1, dim_head)
        k_gather = torch.gather(k.unsqueeze(2).expand(-1, -1, idx.shape[2], -1, -1), 3, idx)
        v_gather = torch.gather(v.unsqueeze(2).expand(-1, -1, idx.shape[2], -1, -1), 3, idx)

        q = q.unsqueeze(3)  # (B, heads, HW, 1, Dh)
        attn_logits = (q * k_gather).sum(-1) * self.scale  # (B, heads, HW, K)
        if mapping.weights is not None:
            attn_logits = attn_logits + mapping.weights.unsqueeze(1)
        attn = attn_logits.softmax(dim=-1)
        out = (attn.unsqueeze(-1) * v_gather).sum(dim=3)  # (B, heads, HW, Dh)
        out = out.transpose(1, 2).contiguous().view(b, hw, c)
        return self.out(out)


class CrossViewBlock(nn.Module):
    """Combines self- and cross-attention with gating."""

    def __init__(self, dim: int, num_heads: int, prune_tokens: int = 1024):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.cross_attn = EpipolarAttention(dim, num_heads, prune_tokens=prune_tokens)
        self.gate = nn.Parameter(torch.tensor(0.5))
        self.norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        mapping: Optional[EpipolarMapping],
    ) -> torch.Tensor:
        src_norm = self.norm(src)
        self_out, _ = self.self_attn(src_norm, src_norm, src_norm)
        cross_out = self.cross_attn(src_norm, tgt, mapping)
        fused = src + (1 - torch.sigmoid(self.gate)) * self_out + torch.sigmoid(self.gate) * cross_out
        fused = fused + self.ffn(self.norm(fused))
        return fused


class EpipolarFusion(nn.Module):
    """
    Geometry-aware fusion that produces a latent 3D grid.
    """

    def __init__(self, dim: int = 192, grid_size: int = 16, grid_channels: int = 8, cross_depth: int = 2):
        super().__init__()
        self.cross_blocks = nn.ModuleList(
            [CrossViewBlock(dim, num_heads=6, prune_tokens=1024) for _ in range(cross_depth)]
        )
        self.grid_size = grid_size
        self.grid_channels = grid_channels
        self.grid_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, grid_size * grid_size * grid_size * grid_channels),
        )
        self.grid_norm = nn.LayerNorm(dim)
        self.token_pool = nn.Linear(dim, dim)

    def forward(
        self,
        feat_a: torch.Tensor,
        feat_b: torch.Tensor,
        geom_a: ProjectionGeometry,
        geom_b: ProjectionGeometry,
        mapping_a2b: Optional[EpipolarMapping] = None,
    ) -> torch.Tensor:
        """
        Args:
            feat_a: (B, H, W, C) features from view A.
            feat_b: (B, H, W, C) features from view B.
            mapping_a2b: optional epipolar mapping from A to B tokens.
        Returns:
            Latent grid (B, grid_channels, G, G, G).
        """
        b, h, w, c = feat_a.shape
        tokens_a = feat_a.view(b, h * w, c)
        tokens_b = feat_b.view(b, h * w, c)
        fused = tokens_a
        for blk in self.cross_blocks:
            fused = blk(fused, tokens_b, mapping_a2b)
        fused = fused.view(b, h, w, c)

        # Tri-planar volumization (compact approximation with channel capacity)
        fused = self.grid_norm(fused.view(b, h * w, c))
        fused = self.token_pool(fused)
        logits = self.grid_proj(fused)  # (B, HW, grid_channels * G^3)
        grid = logits.mean(dim=1)  # pooled over tokens
        grid = grid.view(b, self.grid_channels, self.grid_size, self.grid_size, self.grid_size)
        return grid
