"""Prompt 1 — Tri-Perspective View representation in pure PyTorch.

Three orthogonal learnable feature planes (HW / ZH / WZ).
Points are projected onto each plane with bilinear interpolation via
F.grid_sample, then aggregated by sum or mean. All ops are batched.

Coordinate convention (mirroring tpvformer notation):
  x ↔ W-axis,  y ↔ H-axis,  z ↔ Z-axis
  plane_hw  stores  (B, C, H, W)  — XY / bird's-eye
  plane_zh  stores  (B, C, Z, H)  — side / range view
  plane_wz  stores  (B, C, W, Z)  — front view
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TriPerspectiveView(nn.Module):
    def __init__(
        self,
        channels: int,
        tpv_h: int,
        tpv_w: int,
        tpv_z: int,
        aggregation: str = "sum",
    ) -> None:
        """
        Args:
            channels:    Feature depth C for every plane.
            tpv_h/w/z:  Spatial resolution of the three planes.
            aggregation: 'sum' or 'mean' across the three plane features.
        """
        super().__init__()
        assert aggregation in ("sum", "mean")
        self.tpv_h = tpv_h
        self.tpv_w = tpv_w
        self.tpv_z = tpv_z
        self.aggregation = aggregation

        # Learnable plane parameters — batch dim=1 so expand handles batches.
        self.plane_hw = nn.Parameter(torch.empty(1, channels, tpv_h, tpv_w))
        self.plane_zh = nn.Parameter(torch.empty(1, channels, tpv_z, tpv_h))
        self.plane_wz = nn.Parameter(torch.empty(1, channels, tpv_w, tpv_z))

        nn.init.trunc_normal_(self.plane_hw, std=0.02)
        nn.init.trunc_normal_(self.plane_zh, std=0.02)
        nn.init.trunc_normal_(self.plane_wz, std=0.02)

    # ---------------------------------------------------------------------- #
    def project(self, points: torch.Tensor) -> torch.Tensor:
        """Bilinear-interpolate features for a batch of 3-D points.

        Args:
            points: (B, N, 3) float32, coordinates in [-1, 1].
                    points[..., 0] = x (W-axis)
                    points[..., 1] = y (H-axis)
                    points[..., 2] = z (Z-axis)

        Returns:
            (B, C, N) aggregated feature vectors.
        """
        B = points.shape[0]
        # grid_sample expects grid shape (B, 1, N, 2), output (B, C, 1, N)
        # last dim is (x, y) in normalised image coords, matching (W, H).

        # HW plane: sample at (x, y)
        grid_hw = points[..., [0, 1]].unsqueeze(1)           # B, 1, N, 2
        feat_hw = F.grid_sample(
            self.plane_hw.expand(B, -1, -1, -1),
            grid_hw,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        ).squeeze(2)                                           # B, C, N

        # ZH plane (stored as Z, H): sample at (y, z) → grid (h, z)
        grid_zh = points[..., [1, 2]].unsqueeze(1)           # B, 1, N, 2
        feat_zh = F.grid_sample(
            self.plane_zh.expand(B, -1, -1, -1),
            grid_zh,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        ).squeeze(2)                                           # B, C, N

        # WZ plane (stored as W, Z): sample at (z, x) → grid (z, w)
        grid_wz = points[..., [2, 0]].unsqueeze(1)           # B, 1, N, 2
        feat_wz = F.grid_sample(
            self.plane_wz.expand(B, -1, -1, -1),
            grid_wz,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        ).squeeze(2)                                           # B, C, N

        fused = feat_hw + feat_zh + feat_wz
        return fused / 3.0 if self.aggregation == "mean" else fused

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """Alias for project — allows use as a standard nn.Module."""
        return self.project(points)

    # ---------------------------------------------------------------------- #
    def plane_shapes(self) -> dict[str, tuple]:
        return {
            "hw": tuple(self.plane_hw.shape),
            "zh": tuple(self.plane_zh.shape),
            "wz": tuple(self.plane_wz.shape),
        }
