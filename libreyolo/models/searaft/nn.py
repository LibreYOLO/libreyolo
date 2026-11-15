"""LibreYOLO flow network built on the vendored SEA-RAFT model."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Dict, Tuple

import torch

from ._vendor.raft import RAFT

# Per-size configuration, matching upstream ``config/eval/{sintel,kitti}-{S,M,L}.json``.
# S and M are distinct checkpoints (ResNet18 / ResNet34 encoders). L is the M
# checkpoint run with 12 refinement iterations instead of 4, which is how
# upstream defines SEA-RAFT(L): there is no separate L checkpoint.
SEARAFT_CONFIGS: Dict[str, Dict] = {
    "s": {"pretrain": "resnet18", "iters": 4},
    "m": {"pretrain": "resnet34", "iters": 4},
    "l": {"pretrain": "resnet34", "iters": 12},
}

_COMMON = {
    "use_var": True,
    "var_min": 0,
    "var_max": 10,
    "initial_dim": 64,
    "block_dims": [64, 128, 256],
    "radius": 4,
    "dim": 128,
    "num_blocks": 2,
}


def searaft_args(size: str) -> SimpleNamespace:
    if size not in SEARAFT_CONFIGS:
        raise ValueError(
            f"Unknown SEA-RAFT size {size!r}; expected one of {sorted(SEARAFT_CONFIGS)}"
        )
    cfg = dict(_COMMON)
    cfg["block_dims"] = list(_COMMON["block_dims"])
    cfg.update(SEARAFT_CONFIGS[size])
    return SimpleNamespace(**cfg)


class LibreSEARAFTNet(RAFT):
    """SEA-RAFT adapted to LibreYOLO's flow contract.

    ``forward`` takes two ``[0, 1]`` RGB frames ``(B, 3, H, W)`` whose height
    and width are multiples of 8 and maps them to upstream's ``[-1, 1]`` range
    (``2 * x - 1``, the same arithmetic as upstream's ``2 * (x / 255) - 1`` once
    the preprocess has divided by 255). The refinement iteration count is fixed
    per size. It returns ``(flow, info)``: the ``(B, 2, H, W)`` flow in pixels
    and the ``(B, 4, H, W)`` Laplace mixture parameters of the last iteration.

    Parameter names are upstream's, so an upstream state dict loads strictly.
    """

    def __init__(self, size: str = "m"):
        self.size = size
        super().__init__(searaft_args(size))

    def forward(
        self, image1: torch.Tensor, image2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        image1 = 2 * image1 - 1.0
        image2 = 2 * image2 - 1.0
        return super().forward(image1, image2, iters=self.args.iters)


def info_to_confidence(info: torch.Tensor, var_min: float = 0.0, var_max: float = 10.0) -> torch.Tensor:
    """Map the Laplace mixture parameters to a ``[0, 1]`` confidence map.

    Follows upstream ``custom.py::get_heatmap``: the expected log scale of the
    mixture, ``sum_k softmax(w)_k * clamp(log_b_k)``, lies in ``[var_min, var_max]``
    (0 for a sharp, certain match; ``var_max`` for the wide component).
    Confidence is ``1 - heatmap / var_max``, so 1 means most certain.
    ``info`` is ``(B, 4, H, W)``; returns ``(B, H, W)``.
    """
    weight = info[:, :2].softmax(dim=1)
    raw_b = info[:, 2:]
    log_b = torch.stack(
        [
            raw_b[:, 0].clamp(min=0, max=var_max),
            raw_b[:, 1].clamp(min=var_min, max=0),
        ],
        dim=1,
    )
    heatmap = (log_b * weight).sum(dim=1)
    return (1.0 - heatmap / var_max).clamp(0.0, 1.0)
