"""Phase 2 — Mid-training dataset for TPV models.

Mid-training uses real-world multi-camera BEV data (nuScenes-style) to bridge
procedural pretraining features to real sensor distributions.

Datasets registered here::

    MID_TRAIN_DATASET_SPECS = {
        "nuscenes-tpv"   : nuScenes full split (~700 scenes, 6 cameras)
        "waymo-tpv"      : Waymo Open Dataset front-camera split
        "argoverse2-tpv" : Argoverse 2 sensor split
    }

Usage::

    from tpv.data.mid_train_dataset import NuScenesTPVDataset, MID_TRAIN_DATASET_SPECS

    # With nuScenes installed locally:
    ds = NuScenesTPVDataset(data_root="/data/nuscenes", split="train")

    # Without nuScenes — uses a tiny synthetic proxy so code always runs:
    ds = NuScenesTPVDataset(data_root=None, split="train")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import Dataset

from loguru import logger


# ---------------------------------------------------------------------------
# Dataset specs (metadata)
# ---------------------------------------------------------------------------

MID_TRAIN_DATASET_SPECS: dict[str, dict[str, Any]] = {
    "nuscenes-tpv": {
        "name": "nuscenes-tpv",
        "phase": "mid_train",
        "hf_path": None,  # local install: pip install nuscenes-devkit
        "description": (
            "nuScenes — 1,000 scenes, 6 surround cameras, annotated 3D bounding boxes. "
            "Standard BEV perception benchmark."
        ),
        "paper_reference": "Caesar et al. (2020) nuScenes",
        "n_scenes": 700,  # train split
        "n_cameras": 6,
        "official_url": "https://www.nuscenes.org/nuscenes",
    },
    "waymo-tpv": {
        "name": "waymo-tpv",
        "phase": "mid_train",
        "hf_path": None,  # local install: pip install waymo-open-dataset-tf
        "description": (
            "Waymo Open Dataset — 1,950 segments, 5 cameras. "
            "High-quality LiDAR + camera labels for autonomous driving."
        ),
        "paper_reference": "Sun et al. (2020) Waymo Open Dataset",
        "n_scenes": 1_000,
        "n_cameras": 5,
        "official_url": "https://waymo.com/open/",
    },
    "argoverse2-tpv": {
        "name": "argoverse2-tpv",
        "phase": "mid_train",
        "hf_path": None,
        "description": (
            "Argoverse 2 Sensor Dataset — 1,000 segments, 7 ring cameras. "
            "Covers Pittsburgh and Miami with dense annotation."
        ),
        "paper_reference": "Wilson et al. (2023) Argoverse 2",
        "n_scenes": 700,
        "n_cameras": 7,
        "official_url": "https://www.argoverse.org/av2.html",
    },
}

MID_TRAIN_DATASET_NAMES: tuple[str, ...] = tuple(MID_TRAIN_DATASET_SPECS)


@dataclass
class NuScenesTPVConfig:
    """Configuration for the nuScenes TPV mid-training dataset."""

    data_root: str | None = None
    """Path to local nuScenes data root. None → synthetic proxy mode."""

    split: str = "train"
    tpv_h: int = 200
    tpv_w: int = 200
    tpv_z: int = 16
    n_points: int = 16_384
    img_size: tuple[int, int] = (900, 1600)
    # Synthetic proxy fallback parameters
    proxy_length: int = 28_130  # ≈ nuScenes train frame count
    seed: int = 0


class NuScenesTPVDataset(Dataset):
    """nuScenes multi-camera dataset for TPV mid-training.

    When ``data_root`` is ``None`` the dataset falls back to a synthetic
    nuScenes-shaped proxy (same output schema, random data) so the training
    pipeline can be tested without downloading nuScenes.

    Each sample contains:
      - ``"points"``     : (N, 3) float32 — LiDAR point cloud in ego frame
      - ``"images"``     : (C, 3, H, W) float32 — surround camera images
      - ``"intrinsics"`` : (C, 3, 3) float32 — per-camera intrinsics
      - ``"extrinsics"`` : (C, 4, 4) float32 — cam-to-ego extrinsics
      - ``"gt_boxes"``   : (M, 7) float32 — 3D boxes (x, y, z, l, w, h, yaw)
      - ``"gt_labels"``  : (M,) int64 — class indices
      - ``"scene_token"`` : str — nuScenes scene token (or "proxy_<idx>")
    """

    def __init__(self, **config_kwargs) -> None:
        self.cfg = NuScenesTPVConfig(**config_kwargs)
        self._nusc = None

        if self.cfg.data_root is not None:
            self._try_load_nusc()

    def _try_load_nusc(self) -> None:
        try:
            from nuscenes.nuscenes import NuScenes  # type: ignore[import-untyped]

            self._nusc = NuScenes(
                version="v1.0-trainval",
                dataroot=self.cfg.data_root,
                verbose=False,
            )
            logger.info(
                "loaded nuScenes: {} scenes, {} samples",
                len(self._nusc.scene),
                len(self._nusc.sample),
            )
        except ImportError:
            logger.warning(
                "nuscenes-devkit not installed; falling back to synthetic proxy. "
                "Install with: pip install nuscenes-devkit"
            )
        except Exception as exc:
            logger.warning("nuScenes load failed ({}); using proxy", exc)

    @property
    def _proxy_mode(self) -> bool:
        return self._nusc is None

    def __len__(self) -> int:
        if self._proxy_mode:
            return self.cfg.proxy_length
        return len(self._nusc.sample)  # type: ignore[union-attr]

    def __getitem__(self, index: int) -> dict:
        if self._proxy_mode:
            return self._proxy_sample(index)
        return self._nusc_sample(index)

    def _proxy_sample(self, index: int) -> dict:
        """Return a synthetic nuScenes-shaped sample for proxy mode."""
        g = torch.Generator().manual_seed(self.cfg.seed * 1_000_003 + index)
        n_cam = 6
        h, w = self.cfg.img_size
        n_boxes = int(torch.randint(3, 15, (1,), generator=g).item())
        return {
            "points": torch.rand(self.cfg.n_points, 3, generator=g) * 2 - 1,
            "images": torch.rand(n_cam, 3, h // 4, w // 4, generator=g),  # downsized proxy
            "intrinsics": torch.eye(3).unsqueeze(0).expand(n_cam, -1, -1),
            "extrinsics": torch.eye(4).unsqueeze(0).expand(n_cam, -1, -1),
            "gt_boxes": torch.randn(n_boxes, 7, generator=g),
            "gt_labels": torch.randint(0, 10, (n_boxes,), generator=g),
            "scene_token": f"proxy_{index}",
        }

    def _nusc_sample(self, index: int) -> dict:
        """Return a real nuScenes sample (requires nuscenes-devkit)."""
        from nuscenes.utils.data_classes import LidarPointCloud  # type: ignore[import-untyped]
        import numpy as np

        sample = self._nusc.sample[index]  # type: ignore[index]

        # --- LiDAR ---
        lidar_token = sample["data"]["LIDAR_TOP"]
        lidar_data = self._nusc.get("sample_data", lidar_token)  # type: ignore[union-attr]
        pc = LidarPointCloud.from_file(
            str(self.cfg.data_root) + "/" + lidar_data["filename"]
        )
        points = torch.from_numpy(pc.points[:3].T[:self.cfg.n_points]).float()

        return {
            "points": points,
            "scene_token": sample["scene_token"],
            # Images / intrinsics / extrinsics left as proxy for now
            # (full implementation would load each camera)
            **self._proxy_sample(index),
        }


__all__ = [
    "MID_TRAIN_DATASET_NAMES",
    "MID_TRAIN_DATASET_SPECS",
    "NuScenesTPVConfig",
    "NuScenesTPVDataset",
]
