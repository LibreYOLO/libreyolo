# SPDX-License-Identifier: Apache-2.0
# Ported from https://github.com/Tencent/WeVisOne-VL
# (wevisone_vl/feature_bridge.py) at commit
# 9a6bcd85404505df0d05cfb5d97512ccacf35dee.
# Copyright (C) 2026 Tencent. All rights reserved.
# Modified for LibreYOLO: single-image only, and the hook helpers resolve the
# Qwen3-VL submodules of the native transformers implementation.
"""Read the Qwen3-VL features the WeVisOne-VL pixel decoder consumes.

The decoder needs four ViT levels taken *before* Qwen's 2x2 spatial merge, and
the final language-model hidden state at every ``<OBJ>`` token. Both are
captured with forward hooks during one teacher-forced pass.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Dict, Tuple

import torch
import torch.nn as nn


def premerge_grid_hw(image_grid_thw: torch.Tensor) -> Tuple[int, int]:
    """Patch grid ``(h, w)`` of the single image before the spatial merge."""
    t, h, w = (int(v) for v in image_grid_thw[0].tolist())
    if t != 1:
        raise ValueError(f"the pixel decoder supports still images only, got t={t}")
    return h, w


def premerge_tokens_to_2d(
    tokens: torch.Tensor, grid_hw: Tuple[int, int], merge_size: int
) -> torch.Tensor:
    """Undo the processor's merge-local patch order and return ``(1, C, h, w)``.

    Patches arrive ordered as ``[h/m, w/m, m, m]`` so each merge group is
    contiguous; a plain ``reshape(h, w)`` would scramble every ``m x m`` block.
    """
    h, w = grid_hw
    m = int(merge_size)
    if h % m or w % m:
        raise ValueError(f"patch grid {(h, w)} is not divisible by merge size {m}")
    if tokens.dim() != 2 or int(tokens.shape[0]) != h * w:
        raise ValueError(f"ViT tokens {tuple(tokens.shape)} do not match grid {h}x{w}")
    c = int(tokens.shape[-1])
    x = tokens.reshape(h // m, w // m, m, m, c).permute(0, 2, 1, 3, 4).reshape(h, w, c)
    return x.permute(2, 0, 1).unsqueeze(0)


def _visual_module(model: nn.Module) -> nn.Module:
    base = getattr(model, "model", model)
    visual = getattr(base, "visual", None) or getattr(model, "visual", None)
    if visual is None:
        raise RuntimeError("cannot locate the Qwen3-VL vision tower")
    return visual


def _last_decoder_layer(model: nn.Module) -> nn.Module:
    base = getattr(model, "model", model)
    for owner in (getattr(base, "language_model", None), base):
        layers = getattr(owner, "layers", None)
        if layers is not None and len(layers) > 0:
            return layers[-1]
    raise RuntimeError("cannot locate the last Qwen3-VL decoder layer")


class VisualFeatureCapture:
    """Capture ViT block outputs at the inputs of their spatial mergers.

    Qwen3-VL feeds the outputs of its DeepStack blocks to per-block mergers
    and the last block to the final merger, so a forward pre-hook on each
    merger sees exactly the pre-merge features of that block.
    """

    def __init__(self, model: nn.Module, feature_layers: Sequence[int]):
        self.model = model
        self.feature_layers = tuple(int(v) for v in feature_layers)
        self.features: Dict[int, torch.Tensor] = {}
        self._handles = []

    def _hook(self, layer_idx: int):
        def hook(_module, inputs):
            self.features[layer_idx] = inputs[0]

        return hook

    def __enter__(self) -> "VisualFeatureCapture":
        visual = _visual_module(self.model)
        deep_indexes = [int(v) for v in visual.deepstack_visual_indexes]
        final_idx = len(visual.blocks) - 1
        missing = [
            v for v in self.feature_layers if v not in set(deep_indexes) | {final_idx}
        ]
        if missing:
            raise RuntimeError(
                f"cannot capture pre-merge ViT layers {missing}; merger inputs "
                f"exist at {sorted(set(deep_indexes) | {final_idx})}"
            )
        self.features.clear()
        for idx, merger in zip(deep_indexes, visual.deepstack_merger_list):
            if idx in self.feature_layers:
                self._handles.append(merger.register_forward_pre_hook(self._hook(idx)))
        if final_idx in self.feature_layers:
            self._handles.append(
                visual.merger.register_forward_pre_hook(self._hook(final_idx))
            )
        return self

    def __exit__(self, *exc) -> bool:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        return False

    def feature_maps(self, grid_hw: Tuple[int, int], merge_size: int) -> list:
        maps = []
        for idx in self.feature_layers:
            if idx not in self.features:
                raise RuntimeError(
                    f"ViT layer {idx} was not captured before its merger"
                )
            maps.append(premerge_tokens_to_2d(self.features[idx], grid_hw, merge_size))
        return maps


class LastHiddenCapture:
    """Capture the last decoder layer's output (before the final norm)."""

    def __init__(self, model: nn.Module):
        self.model = model
        self.hidden = None
        self._handle = None

    def _hook(self, _module, _inputs, output):
        self.hidden = output[0] if isinstance(output, (tuple, list)) else output

    def __enter__(self) -> "LastHiddenCapture":
        self._handle = _last_decoder_layer(self.model).register_forward_hook(self._hook)
        return self

    def __exit__(self, *exc) -> bool:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        return False
