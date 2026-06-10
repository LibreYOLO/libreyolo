"""Upstream PicoDet checkpoint key remapping.

The supported upstream is the Bo396543018/Picodet_Pytorch re-port of
PaddleDetection's PicoDet (the original Paddle ``.pdparams`` files are not
PyTorch and are out of scope). Bo's checkpoints carry mmdet-style key naming
because his ``ESNet`` / ``CSPPAN`` / ``PICODETHead`` are wrapped in mmcv's
``ConvModule`` / ``DepthwiseSeparableConvModule`` / ``SELayer``. LibreYOLO's
port keeps the same numerics but flattens those wrappers, so the key remap is
purely syntactic:

  bbox_head.*                           -> head.*
  backbone.<stage>_<i>.*                -> backbone.blocks.<flat_idx>.*
  neck.trans.trans.<i>.*                -> neck.trans.<i>.*
  *.se.conv{1,2}.conv.{w,b}             -> *.se.conv{1,2}.{w,b}

Shared by the runtime auto-converter and ``weights/convert_picodet_weights.py``.
"""

from __future__ import annotations

import re
from typing import Dict, Optional

import torch

# ESNet stage repeats: stage_id (2,3,4) -> repeats. Used to flatten Bo's
# ``<stage>_<i>`` (1-indexed) names into ``blocks.<flat_idx>`` (0-indexed).
ESNET_STAGE_REPEATS = (3, 7, 3)


def _build_block_index_map() -> Dict[str, int]:
    """Map ``<stage>_<i>`` -> flat block index. Bo numbers stages 2,3,4."""
    out: Dict[str, int] = {}
    flat = 0
    for stage_idx, repeats in enumerate(ESNET_STAGE_REPEATS):
        stage_id = stage_idx + 2
        for i in range(repeats):
            out[f"{stage_id}_{i + 1}"] = flat
            flat += 1
    return out


_BLOCK_MAP = _build_block_index_map()
# Pattern matches Bo's per-block prefix: e.g. ``backbone.2_1.`` or ``backbone.4_3.``
_BACKBONE_BLOCK_RE = re.compile(r"^backbone\.(\d+_\d+)\.")
# SE wraps with ConvModule, adding an extra ``.conv.`` we need to drop.
_SE_CONV_RE = re.compile(r"\.se\.conv([12])\.conv\.")


def is_upstream_state_dict(state_dict: dict) -> bool:
    """True for Bo-style mmdet key naming (vs LibreYOLO's flattened port)."""
    has_mmdet_head = any(k.startswith("bbox_head.gfl_cls") for k in state_dict)
    has_staged_backbone = any(_BACKBONE_BLOCK_RE.match(k) for k in state_dict)
    return has_mmdet_head and has_staged_backbone


def remap_key(key: str) -> Optional[str]:
    """Translate a single Bo-style key to LibreYOLO naming.

    Returns ``None`` if the key should be dropped (e.g. a buffer LibreYOLO
    doesn't carry). Currently no keys are dropped — all numerics survive.
    """
    new = key

    # Top-level rename: bbox_head -> head
    if new.startswith("bbox_head."):
        new = "head." + new[len("bbox_head.") :]

    # Backbone block flattening: backbone.<stage>_<i>. -> backbone.blocks.<flat>.
    m = _BACKBONE_BLOCK_RE.match(new)
    if m is not None:
        token = m.group(1)
        flat = _BLOCK_MAP.get(token)
        if flat is None:
            raise ValueError(
                f"Unexpected backbone block token {token!r} in key {key!r}; "
                "expected one of " + ", ".join(sorted(_BLOCK_MAP))
            )
        new = f"backbone.blocks.{flat}." + new[m.end() :]

    # Neck transformation: neck.trans.trans.X.* -> neck.trans.X.*
    if new.startswith("neck.trans.trans."):
        new = "neck.trans." + new[len("neck.trans.trans.") :]

    # SE ConvModule unwrap: *.se.conv1.conv.X -> *.se.conv1.X (and conv2)
    new = _SE_CONV_RE.sub(lambda mm: f".se.conv{mm.group(1)}.", new)

    return new


def remap_state_dict(state_dict: dict) -> Dict[str, torch.Tensor]:
    """Remap an entire Bo-format state dict. Detects collisions early."""
    out: Dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        new = remap_key(k)
        if new is None:
            continue
        if new in out:
            raise ValueError(f"Key collision after remap: {k!r} and another -> {new!r}")
        out[new] = v
    return out


def convert_upstream(state_dict: dict) -> Dict[str, torch.Tensor]:
    """Full Bo-checkpoint conversion: filter training-only keys, then remap.

    Keeps the regular (non-EMA) weights — that is what Bo's mmdet
    ``init_detector`` actually loads, and what his published mAP corresponds
    to. The ``integral.project`` linspace buffer is dropped because LibreYOLO
    computes DFL inline in PicoHead.
    """
    filtered = {
        k: v
        for k, v in state_dict.items()
        if not k.startswith("ema_") and not k.endswith("integral.project")
    }
    return remap_state_dict(filtered)
