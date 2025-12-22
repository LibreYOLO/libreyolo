"""
CAM (Class Activation Mapping) methods for YOLO model interpretability.

This module provides various explainability techniques for understanding
what YOLO models focus on when making predictions.
"""

from .base import BaseCAM
from .activations import ActivationsAndGradients
from .eigen_cam import EigenCAM
from .gradcam import GradCAM
from .hirescam import HiResCAM
from .xgradcam import XGradCAM
from .layercam import LayerCAM
from .eigengradcam import EigenGradCAM
from .gradcampp import GradCAMPlusPlus

# Registry of available CAM methods
CAM_METHODS = {
    "eigencam": EigenCAM,
    "gradcam": GradCAM,
    "hirescam": HiResCAM,
    "xgradcam": XGradCAM,
    "layercam": LayerCAM,
    "eigengradcam": EigenGradCAM,
    "gradcam++": GradCAMPlusPlus,
    "gradcampp": GradCAMPlusPlus,
}

__all__ = [
    "BaseCAM",
    "ActivationsAndGradients",
    "EigenCAM",
    "GradCAM",
    "HiResCAM",
    "XGradCAM",
    "LayerCAM",
    "EigenGradCAM",
    "GradCAMPlusPlus",
    "CAM_METHODS",
]

