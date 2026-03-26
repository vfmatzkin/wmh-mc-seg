"""Backward-compatibility shim.

Imports from this module are forwarded to their new locations:
- WMHModel, compute_metrics -> src/models/, src/utils/
- UNet3D -> src/models/
"""

from models.unet3d import UNet3D
from models.wmh_module import WMHModel
from utils.metrics import compute_metrics

__all__ = ["UNet3D", "WMHModel", "compute_metrics"]
