"""Upstream RiO-DETRv2 checkpoint key remapping.

Shared by the runtime auto-converter and ``weights/convert_riodetrv2_weights.py``.
Upstream checkpoints (``RicePasteM/RiO-DETRv2``) name the encoder input
projection ``.conv`` / ``.norm``, which LibreYOLO's RT-DETR encoder stores as
Sequential ``.0`` / ``.1``.  The training-only DINOv3 distillation projector
(``encoder.feature_projector``) is dropped.  Every other tensor, including the
saved ``decoder.anchors`` / ``decoder.valid_mask`` buffers, is kept unchanged.
"""

from __future__ import annotations

import torch

from ..rtdetr.convert import _remap_input_proj

# Keys that exist only in RiO-DETRv2: the separate angle-distribution head
# and the angle-aware quality head.
RIO_MARKERS = ("decoder.dec_angle_head.", "decoder.decoder.angle_lqe.")
DROP_PREFIXES = ("encoder.feature_projector.",)


def is_riodetrv2_state_dict(state_dict: dict) -> bool:
    """True for RiO-DETRv2 tensors in either upstream or LibreYOLO naming."""
    return all(any(k.startswith(marker) for k in state_dict) for marker in RIO_MARKERS)


def is_upstream_state_dict(state_dict: dict) -> bool:
    """True only for the upstream naming (named encoder input projection)."""
    if not is_riodetrv2_state_dict(state_dict):
        return False
    return any(
        k.startswith("encoder.input_proj.") and k.split(".")[3] in ("conv", "norm")
        for k in state_dict
    )


def convert_upstream(state_dict: dict) -> dict[str, torch.Tensor]:
    """Remap upstream keys to the native port's naming (syntactic only)."""
    return {
        _remap_input_proj(key): value
        for key, value in state_dict.items()
        if not key.startswith(DROP_PREFIXES)
    }
