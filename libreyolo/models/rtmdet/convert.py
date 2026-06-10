"""Upstream RTMDet checkpoint key remapping.

Upstream mm-series RTMDet checkpoints name the detection head ``bbox_head``
where LibreRTMDet uses ``head``, and carry normalization constants under
``data_preprocessor`` that LibreYOLO applies in its own preprocessing. The
remap is purely syntactic; with ``share_conv=True`` upstream stores the same
conv weight at all three pyramid levels, and keeping the redundant entries is
fine — LibreRTMDet's aliased modules overwrite each other with equal values.

Shared by the runtime auto-converter and ``weights/convert_rtmdet_weights.py``.
"""

from __future__ import annotations

from typing import Dict

import torch

DROP_PREFIXES = ("data_preprocessor.",)


def is_upstream_state_dict(state_dict: dict) -> bool:
    """True for mm-series ``bbox_head`` naming (vs LibreRTMDet's ``head``)."""
    return any(
        k.startswith("bbox_head.") and ("rtm_cls" in k or "rtm_reg" in k)
        for k in state_dict
    )


def convert_upstream(state_dict: dict) -> Dict[str, torch.Tensor]:
    """Apply the ``bbox_head`` -> ``head`` rename and drop ``data_preprocessor``."""
    out: Dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        if any(k.startswith(prefix) for prefix in DROP_PREFIXES):
            continue
        new_key = k
        if k.startswith("bbox_head."):
            new_key = "head." + k[len("bbox_head.") :]
        out[new_key] = v
    return out
