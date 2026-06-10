"""Upstream RT-DETR / RT-DETRv2 checkpoint key remapping.

Upstream PyTorch releases (Apache-2.0, lyuwenyu/RT-DETR) name the encoder
input-projection submodules (``.conv`` / ``.norm``) while LibreYOLO's port
uses Sequential numeric keys (``.0`` / ``.1``). Two conversion targets exist:

- :func:`convert_to_v1` — RT-DETR (v1) family. Also remaps
  ``decoder.enc_output`` and drops v2-only buffers the v1 module does not
  have. Used for v1 checkpoints and for v2 HGNetv2-L/X checkpoints, which
  LibreYOLO ships under the v1 family.
- :func:`convert_to_v2` — RT-DETRv2 (ResNet) family. Only the encoder
  input projection needs remapping; the ported v2 decoder keeps upstream's
  named submodules and its precomputed buffers.

Shared by the runtime auto-converter and the offline ``weights/`` scripts.
"""

from __future__ import annotations

import re
from typing import Dict

import torch

# v2-only tensor key fragments the v1 module does not have.
V2_ONLY_FRAGMENTS = (
    "decoder.anchors",
    "decoder.valid_mask",
    "cross_attn.num_points_scale",
)

# Unique to v2's discrete-sampling decoder; v1 checkpoints never carry it.
V2_SAMPLING_FRAGMENT = "cross_attn.num_points_scale"

_UPSTREAM_INPUT_PROJ_RE = re.compile(r"^encoder\.input_proj\.\d+\.(conv|norm)\.")


def has_upstream_input_proj_keys(state_dict: dict) -> bool:
    """True when the encoder input projection uses upstream named submodules."""
    return any(_UPSTREAM_INPUT_PROJ_RE.match(k) for k in state_dict)


def _remap_input_proj(key: str) -> str:
    """``encoder.input_proj.{i}.conv/.norm`` -> ``.{i}.0/.1``."""
    if key.startswith("encoder.input_proj."):
        parts = key.split(".")
        if len(parts) >= 4:
            sub = parts[3]
            if sub == "conv":
                parts[3] = "0"
                return ".".join(parts)
            if sub == "norm":
                parts[3] = "1"
                return ".".join(parts)
    return key


def _remap_enc_output(key: str) -> str:
    """``decoder.enc_output.proj/.norm`` -> ``.0/.1`` (v1 target only)."""
    if key.startswith("decoder.enc_output."):
        parts = key.split(".")
        if len(parts) >= 3:
            sub = parts[2]
            if sub == "proj":
                parts[2] = "0"
                return ".".join(parts)
            if sub == "norm":
                parts[2] = "1"
                return ".".join(parts)
    return key


def convert_to_v1(state_dict: dict) -> Dict[str, torch.Tensor]:
    """Remap an upstream checkpoint to the LibreRTDETR (v1) key layout."""
    out: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if any(fragment in key for fragment in V2_ONLY_FRAGMENTS):
            continue
        out[_remap_enc_output(_remap_input_proj(key))] = value
    return out


def convert_to_v2(state_dict: dict) -> Dict[str, torch.Tensor]:
    """Remap an upstream checkpoint to the LibreRTDETRv2 key layout.

    Buffers (``decoder.anchors`` / ``decoder.valid_mask`` /
    ``num_points_scale``) are kept so the strict load overrides init-time
    values with the upstream-saved tensors.
    """
    return {_remap_input_proj(key): value for key, value in state_dict.items()}
