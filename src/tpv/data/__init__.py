"""Stage-aware datasets for tri-perspective view (TPV) training.

Three training phases are supported:

  PRETRAIN   — large synthetic multi-view scenes (procedural, infinite)
  MID_TRAIN  — nuScenes-style real-world BEV data for domain adaptation
  POST_TRAIN — task-specific downstream (occupancy, detection, segmentation)

Usage::

    from tpv.data import TrainingStage, build_stage_dataset
    from tpv.data.pretrain_dataset import SyntheticTPVDataset
    from tpv.data.mid_train_dataset import NuScenesTPVDataset
    from tpv.data.post_train_dataset import NuScenesOccupancyDataset

    dataset = build_stage_dataset(TrainingStage.PRETRAIN, length=50_000)
"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch.utils.data import Dataset


class TrainingStage(StrEnum):
    """High-level training phase for TPV models.

      PRETRAIN   — large procedurally-rendered 3D synthetic data
      MID_TRAIN  — real-world BEV / multi-camera data (nuScenes / Waymo style)
      POST_TRAIN — task-specific fine-tuning (occupancy, 3D detection, seg.)
    """

    PRETRAIN = "pretrain"
    MID_TRAIN = "mid_train"
    POST_TRAIN = "post_train"


def build_stage_dataset(
    stage: TrainingStage,
    **dataset_kwargs,
) -> "Dataset":
    """Return the canonical dataset for the given training stage.

    Args:
        stage: One of TrainingStage.PRETRAIN / MID_TRAIN / POST_TRAIN.
        **dataset_kwargs: Forwarded to the dataset constructor.

    Returns:
        A ``torch.utils.data.Dataset`` instance.
    """
    from tpv.data.pretrain_dataset import SyntheticTPVDataset
    from tpv.data.mid_train_dataset import NuScenesTPVDataset
    from tpv.data.post_train_dataset import NuScenesOccupancyDataset

    _builders = {
        TrainingStage.PRETRAIN: SyntheticTPVDataset,
        TrainingStage.MID_TRAIN: NuScenesTPVDataset,
        TrainingStage.POST_TRAIN: NuScenesOccupancyDataset,
    }
    return _builders[stage](**dataset_kwargs)


__all__ = [
    "TrainingStage",
    "build_stage_dataset",
]
