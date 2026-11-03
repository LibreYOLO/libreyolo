"""LibreYOLO flow network built on the vendored NeuFlow v2 model."""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from ._vendor.neuflow import NeuFlow

# Encoder stride product: the coarsest feature map is 1/16 of the input.
NEUFLOWV2_STRIDE = 16
# Upstream inference defaults (infer.py / evaluate.py): one refinement step at
# 1/16 scale, eight at 1/8 scale.
DEFAULT_ITERS_S16 = 1
DEFAULT_ITERS_S8 = 8


class LibreNeuFlowv2Net(NeuFlow):
    """NeuFlow v2 adapted to LibreYOLO's flow-task contract.

    Upstream keeps per-resolution grids and zero states as attributes that the
    caller must build with ``init_bhwd(batch, height, width, device, amp)``
    before every change of geometry. This wrapper rebuilds them lazily when the
    batch, spatial size, device or dtype changes, so callers only see
    ``forward(image1, image2)``.

    Inputs are ``(B, 3, H, W)`` RGB in ``[0, 1]`` (LibreYOLO's dense-task
    convention), ``H`` and ``W`` multiples of 16. Upstream consumes ``[0, 255]``
    and divides by 255 internally; the wrapper multiplies back, so the network
    sees the same values. The output is the final full-resolution flow
    ``(B, 2, H, W)`` in input pixels, channel 0 is dx and channel 1 is dy.

    The grids are created from Python ints, so a traced or exported graph
    bakes them in as constants for the fixed export canvas; nothing in the
    graph depends on the flow values except upstream's own correlation
    lookup (``grid_sample``).
    """

    def __init__(
        self,
        iters_s16: int = DEFAULT_ITERS_S16,
        iters_s8: int = DEFAULT_ITERS_S8,
    ):
        super().__init__()
        self.iters_s16 = int(iters_s16)
        self.iters_s8 = int(iters_s8)
        self._bhwd_key: Optional[Tuple] = None

    def _ensure_bhwd(self, image: torch.Tensor) -> None:
        b, _, h, w = image.shape
        if h % NEUFLOWV2_STRIDE or w % NEUFLOWV2_STRIDE:
            raise ValueError(
                f"NeuFlow v2 input height and width must be multiples of "
                f"{NEUFLOWV2_STRIDE}, got {h}x{w}."
            )
        key = (int(b), int(h), int(w), image.device, image.dtype)
        if key == self._bhwd_key:
            return
        self.init_bhwd(
            int(b), int(h), int(w), image.device, amp=image.dtype == torch.float16
        )
        self._bhwd_key = key

    def _apply(self, fn, *args, **kwargs):
        # .to() / .half() / .cuda() move parameters; the cached grids are plain
        # attributes, so drop them and rebuild on the next forward.
        self._bhwd_key = None
        return super()._apply(fn, *args, **kwargs)

    def forward(self, image1: torch.Tensor, image2: torch.Tensor) -> torch.Tensor:
        if image1.shape != image2.shape:
            raise ValueError(
                f"NeuFlow v2 needs two frames of the same shape, got "
                f"{tuple(image1.shape)} and {tuple(image2.shape)}."
            )
        self._ensure_bhwd(image1)
        flows = super().forward(
            image1 * 255.0,
            image2 * 255.0,
            iters_s16=self.iters_s16,
            iters_s8=self.iters_s8,
        )
        return flows[-1]
