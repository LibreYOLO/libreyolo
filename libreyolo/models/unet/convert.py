"""Recognize and unwrap an official mmseg UNet-S5-D16 checkpoint.

Shared by the runtime auto-converter and ``weights/convert_unet_weights.py``.
"""

from __future__ import annotations

import torch

_UNIQUE = (
    "backbone.encoder.4.1.convs.1.conv.weight",
    "backbone.decoder.3.upsample.interp_upsample.1.conv.weight",
    "decode_head.convs.0.conv.weight",
    "decode_head.conv_seg.weight",
    "auxiliary_head.convs.0.conv.weight",
)


def _strip_module_prefix(state_dict: dict) -> dict:
    if any(str(key).startswith("module.") for key in state_dict):
        return {
            (str(key).removeprefix("module.")): value
            for key, value in state_dict.items()
        }
    return state_dict


def is_upstream_state_dict(state_dict: dict) -> bool:
    """True only for the mmseg UNet-S5-D16 + FCN layout."""
    keys = set(_strip_module_prefix(state_dict))
    return all(token in keys for token in _UNIQUE)


def convert_upstream(state_dict: dict) -> dict[str, torch.Tensor]:
    """Native keys already match; drop non-tensor mmseg bookkeeping if present."""
    cleaned = _strip_module_prefix(state_dict)
    return {str(key): value for key, value in cleaned.items() if torch.is_tensor(value)}


def convert_upstream_unet_state_dict(state_dict: dict) -> dict | None:
    if not is_upstream_state_dict(state_dict):
        return None
    return convert_upstream(state_dict)


__all__ = [
    "convert_upstream",
    "convert_upstream_unet_state_dict",
    "is_upstream_state_dict",
]
