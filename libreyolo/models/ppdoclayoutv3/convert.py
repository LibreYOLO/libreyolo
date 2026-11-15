"""Upstream PP-DocLayoutV3 checkpoint recognition and key remapping.

The supported upstream is the official safetensors release
(``PaddlePaddle/PP-DocLayoutV3_safetensors``). Its keys already match the
native port under a ``model.`` prefix, so the remap strips that prefix; the
learned parameters are unchanged. The Paddle-format release
(``PaddlePaddle/PP-DocLayoutV3``) is not a PyTorch file and is out of scope.

Shared by the runtime auto-converter and
``weights/convert_ppdoclayoutv3_weights.py``.
"""

from __future__ import annotations

from typing import Dict

import torch

UPSTREAM_PREFIX = "model."

# Keys that only the PP-DocLayoutV3 head carries. All three are required: the
# reading-order pointer, the per-layer order projections and the mask
# prototype branch. Plain RT-DETR-style checkpoints have none of them.
SIGNATURE_KEYS = (
    "decoder_global_pointer.dense.weight",
    "decoder_order_head.0.weight",
    "encoder.encoder_mask_output.conv.weight",
)


def is_native_state_dict(state_dict: dict) -> bool:
    """True for the native (prefix-free) PP-DocLayoutV3 key layout."""
    return all(key in state_dict for key in SIGNATURE_KEYS)


def is_upstream_state_dict(state_dict: dict) -> bool:
    """True only for the official safetensors layout (``model.``-prefixed)."""
    return all(UPSTREAM_PREFIX + key in state_dict for key in SIGNATURE_KEYS)


def convert_upstream(state_dict: dict) -> Dict[str, torch.Tensor]:
    """Strip the ``model.`` prefix; every key of the release carries it."""
    return {
        key[len(UPSTREAM_PREFIX) :] if key.startswith(UPSTREAM_PREFIX) else key: value
        for key, value in state_dict.items()
    }
