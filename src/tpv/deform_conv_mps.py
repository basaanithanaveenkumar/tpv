"""Prompt 6 — Custom deformable convolution for macOS (CPU + MPS).

torchvision.ops.deform_conv2d does NOT support MPS.
This module reimplements the same behaviour using F.grid_sample +
manual offset generation, which is fully MPS-compatible.

Algorithm:
  1. `offset_conv` predicts 2*k*k offsets and `mask_conv` predicts k*k
     modulation weights from the input.
  2. For each of the k*k kernel positions we build a 2-D sampling grid
     (output-space coords → input-space fractional coords) and call
     F.grid_sample once.
  3. The k*k sampled maps are modulated and concatenated into
     (B, C_in * k*k, H_out, W_out), then projected with a linear weight.

Memory note: we iterate over k*k positions (9 for a 3×3 kernel) rather
than building one giant tensor, keeping peak memory lower on Apple Silicon.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class DeformableConvMPS(nn.Module):
    """Deformable Conv2d — works on CPU, CUDA, and MPS.

    Args:
        in_channels:  Input feature channels.
        out_channels: Output feature channels.
        kernel_size:  Square kernel size (default 3).
        stride:       Convolution stride (default 1).
        padding:      Zero-padding added to input (default 1).
        bias:         Add a learnable bias to the output.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k = kernel_size
        self.stride = stride
        self.padding = padding

        k = kernel_size
        # Predicts (dy, dx) offset for every kernel position
        self.offset_conv = nn.Conv2d(
            in_channels, 2 * k * k, k, stride=stride, padding=padding, bias=True
        )
        # Predicts per-position modulation weight in (0, 1)
        self.mask_conv = nn.Conv2d(
            in_channels, k * k, k, stride=stride, padding=padding, bias=True
        )
        # Output projection weight: (C_out, C_in * k*k)
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels * k * k))
        self.bias_param = nn.Parameter(torch.zeros(out_channels)) if bias else None

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.zeros_(self.offset_conv.weight)
        nn.init.zeros_(self.offset_conv.bias)
        nn.init.constant_(self.mask_conv.bias, 0.5)
        nn.init.zeros_(self.mask_conv.weight)

    # ---------------------------------------------------------------------- #
    def _make_base_grid(
        self,
        H: int, W: int,
        H_out: int, W_out: int,
        ki: int, kj: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Normalised [-1, 1] base grid for one kernel position (ki, kj).

        Returns (H_out, W_out, 2) where [..., 0] = x (col) and [..., 1] = y (row).
        """
        s, p = self.stride, self.padding
        # Output pixel (oh, ow) samples from input at
        #   (oh*s - p + ki,  ow*s - p + kj)
        oy = (
            torch.arange(H_out, device=device, dtype=dtype) * s - p + ki
        )  # H_out
        ox = (
            torch.arange(W_out, device=device, dtype=dtype) * s - p + kj
        )  # W_out

        # Normalise to [-1, 1] using input size
        ny = oy / (H - 1) * 2 - 1
        nx = ox / (W - 1) * 2 - 1

        grid_y = ny.view(-1, 1).expand(H_out, W_out)
        grid_x = nx.view(1, -1).expand(H_out, W_out)
        return torch.stack([grid_x, grid_y], dim=-1)   # H_out, W_out, 2

    # ---------------------------------------------------------------------- #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, in_channels, H, W)

        Returns:
            (B, out_channels, H_out, W_out)
        """
        B, C, H, W = x.shape
        k = self.k

        offsets = self.offset_conv(x)                   # B, 2*k*k, H_out, W_out
        masks   = torch.sigmoid(self.mask_conv(x))      # B, k*k,   H_out, W_out
        H_out, W_out = offsets.shape[2], offsets.shape[3]

        sampled: list[torch.Tensor] = []
        for idx, (ki, kj) in enumerate(
            (i, j) for i in range(k) for j in range(k)
        ):
            base = self._make_base_grid(
                H, W, H_out, W_out, ki, kj, x.device, x.dtype
            )  # H_out, W_out, 2
            base = base.unsqueeze(0).expand(B, -1, -1, -1)  # B, H_out, W_out, 2

            # Offset in pixel units → normalise to [-1, 1] scale
            dy = offsets[:, idx,        :, :] / (H - 1) * 2  # B, H_out, W_out
            dx = offsets[:, k * k + idx, :, :] / (W - 1) * 2

            grid = base + torch.stack([dx, dy], dim=-1)      # B, H_out, W_out, 2

            feat = F.grid_sample(
                x, grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            )  # B, C, H_out, W_out

            mod = masks[:, idx : idx + 1, :, :]              # B, 1, H_out, W_out
            sampled.append(feat * mod)                        # B, C, H_out, W_out

        # Stack → B, C*k*k, H_out, W_out
        stacked = torch.cat(sampled, dim=1)

        # Flatten spatial, apply weight, restore spatial
        flat   = stacked.permute(0, 2, 3, 1).reshape(B * H_out * W_out, C * k * k)
        out    = flat @ self.weight.t()                       # B*H*W, C_out
        if self.bias_param is not None:
            out = out + self.bias_param
        return out.reshape(B, H_out, W_out, self.out_channels).permute(0, 3, 1, 2)

    # ---------------------------------------------------------------------- #
    def extra_repr(self) -> str:
        return (
            f"in={self.in_channels}, out={self.out_channels}, "
            f"k={self.k}, stride={self.stride}, padding={self.padding}"
        )
