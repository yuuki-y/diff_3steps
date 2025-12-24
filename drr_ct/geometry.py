from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch


@dataclass
class ProjectionGeometry:
    """
    Minimal camera geometry needed for epipolar-aware fusion.

    Attributes:
        intrinsics: (B, 3, 3) camera intrinsics per view.
        extrinsics: (B, 4, 4) camera-to-world matrices per view.
        source_to_detector: (B,) scalar distance (mm).
        view_id: string identifier for bookkeeping/logging.
    """

    intrinsics: torch.Tensor
    extrinsics: torch.Tensor
    source_to_detector: torch.Tensor
    view_id: str = "unknown"

    def to(self, device: torch.device) -> "ProjectionGeometry":
        return ProjectionGeometry(
            intrinsics=self.intrinsics.to(device),
            extrinsics=self.extrinsics.to(device),
            source_to_detector=self.source_to_detector.to(device),
            view_id=self.view_id,
        )


@dataclass
class EpipolarMapping:
    """
    Precomputed epipolar correspondences from view A to view B.

    Attributes:
        indices: (B, L, K) long tensor of target token indices in view B for each token in view A.
        weights: (B, L, K) attention bias weights for correspondences; optional.
        shape_hw: (H, W) token map size for sanity checks.
    """

    indices: torch.Tensor
    weights: Optional[torch.Tensor]
    shape_hw: Tuple[int, int]

    def to(self, device: torch.device) -> "EpipolarMapping":
        return EpipolarMapping(
            indices=self.indices.to(device),
            weights=self.weights.to(device) if self.weights is not None else None,
            shape_hw=self.shape_hw,
        )
