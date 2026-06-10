"""Upstream RT-DETRv4 checkpoint cleanup.

Upstream RT-DETRv4 training checkpoints carry ``encoder.feature_projector``
tensors used only for distillation during training; LibreYOLO's inference
module does not have them. Their presence is also what distinguishes a raw
v4 checkpoint from its D-FINE/DEIM siblings, which share every other
architecture key.

Shared by the runtime auto-converter and ``weights/convert_rtdetrv4_weights.py``.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch

TRAINING_ONLY_PREFIXES = ("encoder.feature_projector.",)


def has_training_only_keys(state_dict: dict) -> bool:
    return any(
        k.startswith(prefix) for k in state_dict for prefix in TRAINING_ONLY_PREFIXES
    )


def drop_training_only_keys(
    state_dict: dict,
) -> Tuple[Dict[str, torch.Tensor], List[str]]:
    """Return ``(cleaned, dropped_keys)`` without the training-only tensors."""
    dropped = [
        k
        for k in state_dict
        if any(k.startswith(prefix) for prefix in TRAINING_ONLY_PREFIXES)
    ]
    cleaned = {k: v for k, v in state_dict.items() if k not in dropped}
    return cleaned, dropped
