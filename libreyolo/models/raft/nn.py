"""LibreRAFT network: the vendored upstream RAFT with the iteration count baked in.

``LibreRAFTNet`` subclasses the vendored ``RAFT`` module, so the parameter names
are exactly upstream's (minus the DataParallel ``module.`` prefix) and the
weight conversion is a metadata wrap.
"""

from __future__ import annotations

import argparse

import torch

from ._vendor.raft import RAFT

# Upstream demo.py runs both raft-things and raft-small at iters=20 on
# arbitrary frame pairs; that is the inference default baked in here.
# (evaluate.py uses 32 on Sintel and 24 on KITTI/Chairs for benchmark numbers.)
DEFAULT_ITERS = 20


class LibreRAFTNet(RAFT):
    """RAFT with a fixed refinement count; forward(image1, image2) -> flow."""

    def __init__(self, small: bool, iters: int = DEFAULT_ITERS) -> None:
        args = argparse.Namespace(
            small=bool(small),
            mixed_precision=False,
            alternate_corr=False,
            dropout=0.0,
        )
        super().__init__(args)
        self.iters = int(iters)

    def forward(self, image1: torch.Tensor, image2: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Return the full-resolution flow ``(B, 2, H, W)`` from image1 to image2.

        Inputs are RGB float tensors in [0, 255] with H and W multiples of 8.
        """
        _, flow_up = super().forward(image1, image2, iters=self.iters, test_mode=True)
        return flow_up


__all__ = ["DEFAULT_ITERS", "LibreRAFTNet"]
