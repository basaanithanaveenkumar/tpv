"""Prompt 3 — TPV4: minimal single-file TPV in pure PyTorch.

Design rules:
  * No custom CUDA extensions — F.grid_sample only.
  * integrate_points() updates the planes from new 3-D point observations
    using bilinear splatting (the transpose of grid_sample), implemented
    via bilinear weight decomposition + index_put_ (MPS-compatible).
  * Everything is in this one file.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Bilinear splatting (reverse of grid_sample) — MPS-safe
# --------------------------------------------------------------------------- #
def _bilinear_splat(
    plane: torch.Tensor,
    grid: torch.Tensor,
    values: torch.Tensor,
    alpha: float = 0.1,
) -> torch.Tensor:
    """Scatter `values` onto `plane` at fractional `grid` positions.

    Uses explicit bilinear weight decomposition + index_put_(accumulate=True).
    This avoids scatter_nd which has limited MPS support.

    Args:
        plane:  (B, C, Ph, Pw) plane to update (not in-place on the Parameter;
                caller passes a cloned buffer).
        grid:   (B, N, 2) sampling grid in [-1, 1], format (x, y) = (w, h).
        values: (B, C, N) feature values to splat.
        alpha:  blend coefficient for the update.

    Returns:
        Updated plane (B, C, Ph, Pw).
    """
    B, C, Ph, Pw = plane.shape
    _, N, _ = grid.shape

    # Map grid from [-1, 1] to pixel coords
    px = (grid[..., 0] * 0.5 + 0.5) * (Pw - 1)  # B, N
    py = (grid[..., 1] * 0.5 + 0.5) * (Ph - 1)  # B, N

    x0 = px.floor().long().clamp(0, Pw - 2)
    y0 = py.floor().long().clamp(0, Ph - 2)
    x1 = x0 + 1
    y1 = y0 + 1

    wx1 = (px - x0.float()).clamp(0, 1)          # B, N
    wy1 = (py - y0.float()).clamp(0, 1)
    wx0 = 1.0 - wx1
    wy0 = 1.0 - wy1

    out = plane.clone()
    for b in range(B):
        for (xidx, yidx, wx, wy) in [
            (x0[b], y0[b], wx0[b] * wy0[b], None),
            (x1[b], y0[b], wx1[b] * wy0[b], None),
            (x0[b], y1[b], wx0[b] * wy1[b], None),
            (x1[b], y1[b], wx1[b] * wy1[b], None),
        ]:
            w = wx.unsqueeze(0)                   # 1, N
            v = values[b] * w                     # C, N
            for c in range(C):
                flat_idx = yidx * Pw + xidx       # N
                out[b, c].view(-1).index_put_(
                    (flat_idx,), v[c], accumulate=True
                )

    # Blend: new = old + alpha * splat_delta
    return plane + alpha * (out - plane)


# --------------------------------------------------------------------------- #
# TPV4
# --------------------------------------------------------------------------- #
class TPV4(nn.Module):
    """Minimal Tri-Perspective View module.

    Three parametrised feature planes (HW / ZH / WZ) plus:
      * project()          — differentiable F.grid_sample query
      * integrate_points() — update planes from new 3-D point observations

    Args:
        channels:    Feature depth C.
        tpv_h/w/z:  Plane spatial sizes.
        aggregation: 'sum' or 'mean'.
    """

    def __init__(
        self,
        channels: int,
        tpv_h: int,
        tpv_w: int,
        tpv_z: int,
        aggregation: str = "sum",
    ) -> None:
        super().__init__()
        assert aggregation in ("sum", "mean")
        self.C = channels
        self.tpv_h = tpv_h
        self.tpv_w = tpv_w
        self.tpv_z = tpv_z
        self.aggregation = aggregation

        self.plane_hw = nn.Parameter(torch.empty(1, channels, tpv_h, tpv_w))
        self.plane_zh = nn.Parameter(torch.empty(1, channels, tpv_z, tpv_h))
        self.plane_wz = nn.Parameter(torch.empty(1, channels, tpv_w, tpv_z))

        for p in [self.plane_hw, self.plane_zh, self.plane_wz]:
            nn.init.trunc_normal_(p, std=0.02)

    # ---------------------------------------------------------------------- #
    def project(self, points: torch.Tensor) -> torch.Tensor:
        """Query features at 3-D point positions via bilinear interpolation.

        Args:
            points: (B, N, 3) in [-1, 1]. (x=W, y=H, z=Z)

        Returns:
            (B, C, N) aggregated features.
        """
        B = points.shape[0]

        def _sample(plane, grid):
            return F.grid_sample(
                plane.expand(B, -1, -1, -1),
                grid.unsqueeze(1),
                mode="bilinear",
                padding_mode="border",
                align_corners=False,
            ).squeeze(2)

        feat = (
            _sample(self.plane_hw, points[..., [0, 1]])  # (x, y)
            + _sample(self.plane_zh, points[..., [1, 2]])  # (y, z)
            + _sample(self.plane_wz, points[..., [2, 0]])  # (z, x)
        )
        return feat / 3.0 if self.aggregation == "mean" else feat

    # ---------------------------------------------------------------------- #
    def integrate_points(
        self,
        points: torch.Tensor,
        features: torch.Tensor,
        alpha: float = 0.1,
    ) -> None:
        """Update all three planes with observations from new 3-D points.

        This is an *in-place* operation on the plane data (via no_grad).
        The update uses bilinear splatting so nearby plane cells are also
        touched proportionally.

        Args:
            points:   (B, N, 3) in [-1, 1], same convention as project().
            features: (B, C, N) observed feature values.
            alpha:    Blend coefficient (0 = no update, 1 = full overwrite).
        """
        B = points.shape[0]
        with torch.no_grad():
            self.plane_hw.data = _bilinear_splat(
                self.plane_hw.data.expand(B, -1, -1, -1).clone(),
                points[..., [0, 1]], features, alpha,
            ).mean(0, keepdim=True)

            self.plane_zh.data = _bilinear_splat(
                self.plane_zh.data.expand(B, -1, -1, -1).clone(),
                points[..., [1, 2]], features, alpha,
            ).mean(0, keepdim=True)

            self.plane_wz.data = _bilinear_splat(
                self.plane_wz.data.expand(B, -1, -1, -1).clone(),
                points[..., [2, 0]], features, alpha,
            ).mean(0, keepdim=True)

    # ---------------------------------------------------------------------- #
    def forward(self, points: torch.Tensor) -> torch.Tensor:
        return self.project(points)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def flops_per_query(self, n_points: int) -> int:
        # grid_sample bilinear: 4 multiplies + 3 adds per channel per point,
        # three planes → ~21 * C * N
        return 21 * self.C * n_points
