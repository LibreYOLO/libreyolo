"""
Per-architecture distillation configuration registry.

Each supported architecture declares its feature tap points (module paths
where forward hooks are registered) and the corresponding channel dimensions.

This module provides fallback configs for cases where the model wrapper
doesn't implement ``get_distill_config()`` directly.

To add a new architecture:
    1. Identify the FPN/neck output modules and their channel dimensions
    2. Add an entry to ``DISTILL_CONFIGS`` below
    3. (Preferred) Also add ``get_distill_config()`` to the model wrapper class
"""

from __future__ import annotations

from typing import Dict, List

# =============================================================================
# YOLOv9 configs
# =============================================================================

# Channel dimensions from YOLO9_CONFIGS in libreyolo/models/yolo9/nn.py
_YOLO9_CHANNELS = {
    "t": (64, 96, 128),
    "s": (128, 192, 256),
    "m": (240, 360, 480),
    "c": (256, 512, 512),
}

_YOLO9_TAP_POINTS = ["neck.elan_up2", "neck.elan_down1", "neck.elan_down2"]

# =============================================================================
# YOLOX configs
# =============================================================================

# Base channels are [256, 512, 1024], scaled by width multiplier
_YOLOX_WIDTH = {
    "n": 0.25,
    "t": 0.375,
    "s": 0.50,
    "m": 0.75,
    "l": 1.00,
    "x": 1.25,
}

_YOLOX_TAP_POINTS = ["backbone.C3_p3", "backbone.C3_n3", "backbone.C3_n4"]


def _yolox_channels(size: str) -> tuple:
    w = _YOLOX_WIDTH[size]
    return (int(256 * w), int(512 * w), int(1024 * w))


# =============================================================================
# Public API
# =============================================================================


def get_distill_config(family: str, size: str) -> Dict:
    """Get the distillation config for a given model family and size.

    Args:
        family: Model family string (e.g., "yolo9", "yolox").
        size: Model size string (e.g., "t", "s", "m", "c", "l", "x").

    Returns:
        Dict with keys:
            - tap_points: List[str] — module paths for forward hooks
            - channels: List[int] — channel dimensions per tap point
            - strides: List[int] — spatial strides per tap point

    Raises:
        ValueError: If the family/size combination is not supported.
    """
    if family == "yolo9":
        if size not in _YOLO9_CHANNELS:
            raise ValueError(
                f"Unknown YOLOv9 size '{size}'. "
                f"Available: {list(_YOLO9_CHANNELS.keys())}"
            )
        return {
            "tap_points": list(_YOLO9_TAP_POINTS),
            "channels": list(_YOLO9_CHANNELS[size]),
            "strides": [8, 16, 32],
        }

    elif family == "yolox":
        if size not in _YOLOX_WIDTH:
            raise ValueError(
                f"Unknown YOLOX size '{size}'. "
                f"Available: {list(_YOLOX_WIDTH.keys())}"
            )
        return {
            "tap_points": list(_YOLOX_TAP_POINTS),
            "channels": list(_yolox_channels(size)),
            "strides": [8, 16, 32],
        }

    else:
        raise ValueError(
            f"Distillation not yet configured for family '{family}'. "
            f"Supported: yolo9, yolox. "
            f"To add support, implement get_distill_config() on your model class "
            f"or add an entry to libreyolo/distillation/configs.py."
        )


def list_supported() -> List[str]:
    """Return list of model families with distillation support."""
    return ["yolo9", "yolox"]
