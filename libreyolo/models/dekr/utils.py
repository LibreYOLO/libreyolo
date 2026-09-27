"""DEKR checkpoint helpers plus preprocess/postprocess re-exports."""

from __future__ import annotations

from typing import Mapping

from ...postprocess.dekr import (  # noqa: F401  (re-exported for family locality)
    DEKR_KEYPOINT_THRESHOLD,
    DEKR_MAX_NUM_PEOPLE,
    DEKR_NMS_NUM_THRESHOLD,
    DEKR_NMS_THRESHOLD,
    DEKR_OUTPUT_STRIDE,
    decode_poses,
    derive_boxes_from_keypoints,
    postprocess_dekr,
)
from ...preprocess.dekr import (  # noqa: F401  (re-exported for family locality)
    DEKR_IMAGENET_MEAN,
    DEKR_IMAGENET_STD,
    DEKR_PAD_VALUE,
    preprocess_image,
    preprocess_numpy,
)

__all__ = [
    "DEKR_IMAGENET_MEAN",
    "DEKR_IMAGENET_STD",
    "DEKR_KEYPOINT_THRESHOLD",
    "DEKR_MAX_NUM_PEOPLE",
    "DEKR_NMS_NUM_THRESHOLD",
    "DEKR_NMS_THRESHOLD",
    "DEKR_OUTPUT_STRIDE",
    "DEKR_PAD_VALUE",
    "decode_poses",
    "derive_boxes_from_keypoints",
    "postprocess_dekr",
    "preprocess_image",
    "preprocess_numpy",
    "strip_module_prefix",
    "unwrap_dekr_checkpoint",
]

_DDP_PREFIX = "module."


def unwrap_dekr_checkpoint(checkpoint):
    """Extract the weight mapping from a released DEKR checkpoint.

    The upstream artifact is a dict with ``net``, ``acc``, ``epoch``,
    ``optimizer_state_dict`` and ``scaler_state_dict``. Only ``net`` is a model
    state; the optimizer and scaler blobs are deliberately not carried into any
    LibreYOLO checkpoint. A bare state dict passes through unchanged.
    """
    if not isinstance(checkpoint, Mapping):
        raise TypeError(
            "DEKR checkpoint must be a mapping, got "
            f"{type(checkpoint).__name__}. Arbitrary pickled module objects are "
            "rejected."
        )
    for key in ("net", "model", "state_dict"):
        value = checkpoint.get(key)
        if isinstance(value, Mapping):
            return value
    return checkpoint


def strip_module_prefix(state_dict: Mapping) -> dict:
    """Strip exactly one leading ``module.`` from every key."""
    return {
        (key[len(_DDP_PREFIX) :] if key.startswith(_DDP_PREFIX) else key): value
        for key, value in state_dict.items()
    }
