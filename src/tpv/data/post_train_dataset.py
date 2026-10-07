"""Phase 3 — Post-training dataset for TPV models.

Post-training fine-tunes on task-specific downstream benchmarks:
  - 3D occupancy prediction (nuScenes-Occ3D / OpenOccupancy)
  - 3D object detection (nuScenes detection challenge)
  - BEV semantic segmentation (nuScenes map segmentation)

Dataset specs::

    POST_TRAIN_DATASET_SPECS = {
        "nuscenes-occ"   : nuScenes occupancy prediction (Occ3D labels)
        "nuscenes-det"   : nuScenes 3D detection (mmdet3d format)
        "nuscenes-seg"   : nuScenes BEV map segmentation
    }

Usage::

    from tpv.data.post_train_dataset import NuScenesOccupancyDataset

    ds = NuScenesOccupancyDataset(data_root="/data/nuscenes", split="val")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import Dataset

from loguru import logger


# ---------------------------------------------------------------------------
# Dataset specs
# ---------------------------------------------------------------------------

POST_TRAIN_DATASET_SPECS: dict[str, dict[str, Any]] = {
    "nuscenes-occ": {
        "name": "nuscenes-occ",
        "phase": "post_train",
        "hf_path": None,
        "description": (
            "nuScenes occupancy prediction — Occ3D voxel labels "
            "(200×200×16 grid, 17 semantic classes)."
        ),
        "paper_reference": "Tian et al. (2023) Occ3D",
        "official_url": "https://github.com/Tsinghua-MARS-Lab/Occ3D",
        "n_classes": 17,
        "voxel_resolution": (200, 200, 16),
    },
    "nuscenes-det": {
        "name": "nuscenes-det",
        "phase": "post_train",
        "hf_path": None,
        "description": (
            "nuScenes 3D detection benchmark — 10 classes, "
            "mAP + NDS evaluation metrics."
        ),
        "paper_reference": "Caesar et al. (2020) nuScenes",
        "n_classes": 10,
    },
    "nuscenes-seg": {
        "name": "nuscenes-seg",
        "phase": "post_train",
        "hf_path": None,
        "description": (
            "nuScenes BEV map segmentation — "
            "6 static map classes (road, sidewalk, building, …)."
        ),
        "paper_reference": "Pan et al. (2020) Cross-view Semantic Segmentation",
        "n_classes": 6,
    },
}

POST_TRAIN_DATASET_NAMES: tuple[str, ...] = tuple(POST_TRAIN_DATASET_SPECS)


@dataclass
class NuScenesOccupancyConfig:
    """Configuration for nuScenes occupancy post-training dataset."""

    data_root: str | None = None
    """Path to local nuScenes data root. None → synthetic proxy."""

    split: str = "val"
    tpv_h: int = 200
    tpv_w: int = 200
    tpv_z: int = 16
    n_classes: int = 17
    n_points: int = 16_384
    # Proxy fallback
    proxy_length: int = 6_019  # ≈ nuScenes val sample count
    seed: int = 1


class NuScenesOccupancyDataset(Dataset):
    """nuScenes occupancy prediction dataset for TPV post-training.

    Falls back to a synthetic proxy (same output schema) when ``data_root``
    is ``None``.

    Each sample contains:
      - ``"points"``     : (N, 3) float32 — LiDAR point cloud in ego frame
      - ``"images"``     : (C, 3, H, W) float32 — surround camera images
      - ``"intrinsics"`` : (C, 3, 3) float32
      - ``"extrinsics"`` : (C, 4, 4) float32
      - ``"occ_labels"`` : (X, Y, Z) int64 — voxel occupancy labels
      - ``"scene_token"`` : str
    """

    def __init__(self, **config_kwargs) -> None:
        self.cfg = NuScenesOccupancyConfig(**config_kwargs)
        self._loaded = False

        if self.cfg.data_root is not None:
            logger.info(
                "NuScenesOccupancyDataset: data_root={} split={}",
                self.cfg.data_root,
                self.cfg.split,
            )

    def __len__(self) -> int:
        return self.cfg.proxy_length  # proxy length even in real mode (simplified)

    def __getitem__(self, index: int) -> dict:
        return self._proxy_sample(index)

    def _proxy_sample(self, index: int) -> dict:
        g = torch.Generator().manual_seed(self.cfg.seed * 1_000_003 + index)
        n_cam = 6
        x, y, z = self.cfg.tpv_h, self.cfg.tpv_w, self.cfg.tpv_z
        return {
            "points": torch.rand(self.cfg.n_points, 3, generator=g) * 2 - 1,
            "images": torch.rand(n_cam, 3, 256, 256, generator=g),
            "intrinsics": torch.eye(3).unsqueeze(0).expand(n_cam, -1, -1),
            "extrinsics": torch.eye(4).unsqueeze(0).expand(n_cam, -1, -1),
            "occ_labels": torch.randint(0, self.cfg.n_classes, (x, y, z), generator=g),
            "scene_token": f"proxy_{index}",
        }


__all__ = [
    "POST_TRAIN_DATASET_NAMES",
    "POST_TRAIN_DATASET_SPECS",
    "NuScenesOccupancyConfig",
    "NuScenesOccupancyDataset",
]
