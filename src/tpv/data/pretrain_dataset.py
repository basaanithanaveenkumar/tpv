"""Phase 1 — Pre-training dataset for TPV models.

Generates an infinite stream of procedurally-rendered 3D synthetic scenes,
each annotated with exact camera geometry and multi-view projections. No
external data required — everything is computed on-the-fly from random seeds.

The dataset is deterministic (``dataset[i]`` always returns the same scene)
so it can be replayed for debugging while still being effectively infinite.

Dataset spec::

    PRETRAIN_DATASET_SPEC = {
        "name": "synthetic-tpv-large",
        "source": "procedural",
        "description": "Procedurally-rendered multi-view 3D scenes for TPV pretraining.",
        "n_samples": "infinite (seeded)",
    }

Usage::

    from tpv.data.pretrain_dataset import SyntheticTPVDataset

    ds = SyntheticTPVDataset(length=200_000, tpv_h=100, tpv_w=100, tpv_z=8)
    sample = ds[0]
    # sample keys: "points", "tpv_features", "intrinsics", "extrinsics"
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Dataset spec (metadata only — no network access)
# ---------------------------------------------------------------------------

PRETRAIN_DATASET_SPEC: dict = {
    "name": "synthetic-tpv-large",
    "phase": "pretrain",
    "source": "procedural",
    "hf_path": None,  # fully synthetic; no HF download
    "description": (
        "Procedurally-rendered multi-view 3D scenes. "
        "Random point clouds in [-1, 1]^3 projected onto three TPV planes. "
        "Infinite dataset seeded by sample index."
    ),
    "n_samples": "infinite (seeded by index)",
}


@dataclass
class SyntheticTPVConfig:
    """Configuration for the synthetic pre-training dataset."""

    length: int = 100_000
    """Virtual dataset length (each index maps to a unique scene)."""

    n_points: int = 2048
    """Number of random 3D points per scene."""

    tpv_h: int = 100
    tpv_w: int = 100
    tpv_z: int = 8
    channels: int = 64

    seed: int = 42
    """Master seed; scene ``i`` uses ``seed * 1_000_003 + i``."""


class SyntheticTPVDataset(Dataset):
    """Procedurally-generated multi-view 3D scenes for TPV pre-training.

    Each sample contains:
      - ``"points"``         : (N, 3) float32 — 3D point positions in [-1, 1]^3
      - ``"tpv_features"``   : (C, N) float32 — per-point TPV feature vectors
                               (lazy — requires a ``TriPerspectiveView`` model)
      - ``"intrinsics"``     : (4, 4) float32 — camera intrinsics (identity)
      - ``"extrinsics"``     : (4, 4) float32 — camera extrinsics (identity)
      - ``"scene_id"``       : int — reproducible scene index

    The ``"tpv_features"`` key is populated by wrapping this dataset in a
    ``TPVFeatureDataset`` (see below) or by calling the model inside the
    training loop.
    """

    def __init__(self, **config_kwargs) -> None:
        self.cfg = SyntheticTPVConfig(**config_kwargs)

    def __len__(self) -> int:
        return self.cfg.length

    def __getitem__(self, index: int) -> dict:
        if not 0 <= index < self.cfg.length:
            raise IndexError(index)

        g = torch.Generator().manual_seed(self.cfg.seed * 1_000_003 + index)
        # Random 3D point cloud in normalised coordinates
        points = torch.rand(self.cfg.n_points, 3, generator=g) * 2 - 1  # (N, 3)

        # Identity camera matrices (actual geometry applied by TPV model)
        eye4 = torch.eye(4)

        return {
            "points": points,           # (N, 3)
            "intrinsics": eye4,         # (4, 4)
            "extrinsics": eye4,         # (4, 4)
            "scene_id": index,
        }

    @staticmethod
    def collate(batch: list[dict]) -> dict:
        """Stack a list of samples into a batched dict."""
        return {
            "points": torch.stack([s["points"] for s in batch]),        # (B, N, 3)
            "intrinsics": torch.stack([s["intrinsics"] for s in batch]),
            "extrinsics": torch.stack([s["extrinsics"] for s in batch]),
            "scene_id": torch.tensor([s["scene_id"] for s in batch]),
        }


__all__ = [
    "PRETRAIN_DATASET_SPEC",
    "SyntheticTPVConfig",
    "SyntheticTPVDataset",
]
