"""Prompt 2 — BEV → Tri-Perspective View converter.

Transforms a Bird's-Eye-View feature map (B, C, H, W) into three TPV planes:
  plane_hw  (B, C_out, tpv_h, tpv_w)  — from BEV directly
  plane_zh  (B, C_out, tpv_z, tpv_h)  — from column pooling along W
  plane_wz  (B, C_out, tpv_w, tpv_z)  — from column pooling along H

Each projection is a small conv stack so the mapping is learned, not just pooled.
Works on CPU, CUDA, and MPS (no custom CUDA ops used).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv_bn_relu(in_c: int, out_c: int, k: int = 3, p: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, k, padding=p, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    )


class BEVToTPV(nn.Module):
    """Convert a BEV feature map into three TPV planes.

    Args:
        in_channels:  Channels of the input BEV tensor.
        out_channels: Channels of every output plane.
        tpv_h, tpv_w, tpv_z: Target spatial sizes of the three planes.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        tpv_h: int,
        tpv_w: int,
        tpv_z: int,
    ) -> None:
        super().__init__()
        self.tpv_h = tpv_h
        self.tpv_w = tpv_w
        self.tpv_z = tpv_z

        # HW plane: conv then spatial resize to (tpv_h, tpv_w).
        self.proj_hw = nn.Sequential(
            _conv_bn_relu(in_channels, out_channels),
            _conv_bn_relu(out_channels, out_channels),
        )

        # ZH plane: pool along W → (B, C, H, 1) → expand Z with 1-D conv.
        self.compress_w = nn.AdaptiveAvgPool2d((None, 1))   # (B,C,H,1)
        self.proj_zh = nn.Sequential(
            nn.Conv1d(in_channels, out_channels * tpv_z, 1, bias=False),
            nn.BatchNorm1d(out_channels * tpv_z),
            nn.ReLU(inplace=True),
        )

        # WZ plane: pool along H → (B, C, 1, W) → expand Z with 1-D conv.
        self.compress_h = nn.AdaptiveAvgPool2d((1, None))   # (B,C,1,W)
        self.proj_wz = nn.Sequential(
            nn.Conv1d(in_channels, out_channels * tpv_z, 1, bias=False),
            nn.BatchNorm1d(out_channels * tpv_z),
            nn.ReLU(inplace=True),
        )

        self.out_channels = out_channels

    # ---------------------------------------------------------------------- #
    def forward(self, bev: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            bev: (B, in_channels, H, W)

        Returns:
            plane_hw: (B, out_channels, tpv_h, tpv_w)
            plane_zh: (B, out_channels, tpv_z, tpv_h)
            plane_wz: (B, out_channels, tpv_w, tpv_z)
        """
        B = bev.shape[0]
        C_out = self.out_channels

        # ── HW plane ──────────────────────────────────────────────────────
        hw = self.proj_hw(bev)
        hw = F.interpolate(hw, size=(self.tpv_h, self.tpv_w), mode="bilinear", align_corners=False)

        # ── ZH plane ──────────────────────────────────────────────────────
        # Compress W via avg pool → (B, C_in, H, 1) → squeeze → (B, C_in, H)
        zh_1d = self.compress_w(bev).squeeze(-1)                     # B, C_in, H_bev
        # Resize H_bev → tpv_h via 1-D linear interpolation
        zh_1d = F.interpolate(zh_1d, size=self.tpv_h, mode="linear", align_corners=False)
        # Conv1d produces C_out * tpv_z channels: (B, C_out*Z, tpv_h)
        zh = self.proj_zh(zh_1d)
        zh = zh.reshape(B, C_out, self.tpv_z, self.tpv_h)

        # ── WZ plane ──────────────────────────────────────────────────────
        # Compress H via avg pool → (B, C_in, 1, W) → squeeze → (B, C_in, W)
        wz_1d = self.compress_h(bev).squeeze(2)                      # B, C_in, W_bev
        wz_1d = F.interpolate(wz_1d, size=self.tpv_w, mode="linear", align_corners=False)
        # Conv1d produces C_out * tpv_z channels: (B, C_out*Z, tpv_w)
        wz = self.proj_wz(wz_1d)
        wz = wz.reshape(B, C_out, self.tpv_w, self.tpv_z)

        return hw, zh, wz


# --------------------------------------------------------------------------- #
# Drop-in demo: swap a BEV detection head for this TPV converter
# --------------------------------------------------------------------------- #
class BEVDetectionHead(nn.Module):
    """Minimal BEV head that this module replaces."""
    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, num_classes),
        )

    def forward(self, bev: torch.Tensor) -> torch.Tensor:
        return self.head(bev)


class TPVDetectionHead(nn.Module):
    """Same interface as BEVDetectionHead but queries the TPV representation."""
    def __init__(self, tpv_channels: int, num_classes: int, tpv_h: int, tpv_w: int, tpv_z: int) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(tpv_channels * 3, num_classes)
        self.tpv_h = tpv_h
        self.tpv_w = tpv_w
        self.tpv_z = tpv_z

    def forward(
        self,
        plane_hw: torch.Tensor,
        plane_zh: torch.Tensor,
        plane_wz: torch.Tensor,
    ) -> torch.Tensor:
        B = plane_hw.shape[0]
        f_hw = plane_hw.flatten(2).mean(-1)   # B, C
        f_zh = plane_zh.flatten(2).mean(-1)
        f_wz = plane_wz.flatten(2).mean(-1)
        return self.fc(torch.cat([f_hw, f_zh, f_wz], dim=-1))
