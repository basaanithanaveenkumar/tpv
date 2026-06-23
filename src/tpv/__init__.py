"""Tri-Perspective View — pure PyTorch, CPU / CUDA / MPS."""
import torch

from .tpv_core import TriPerspectiveView
from .bev_to_tpv import BEVToTPV
from .tpv4 import TPV4
from .tpv10 import TPV10Encoder
from .deform_conv_mps import DeformableConvMPS

__all__ = ["TriPerspectiveView", "BEVToTPV", "TPV4", "TPV10Encoder", "DeformableConvMPS", "get_device"]


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        import os
        # grid_sampler_2d_backward is not yet in MPS kernels; this flag lets
        # PyTorch fall back to CPU silently for unsupported ops while keeping
        # everything else on MPS.
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
