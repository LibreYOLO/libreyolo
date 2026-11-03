"""SEA-RAFT preprocessing helpers (pad to a multiple of 8, upstream geometry)."""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

STRIDE = 8
# The correlation pyramid halves the 1/8 feature map four times, so the padded
# canvas needs at least 16 cells per side (upstream fails the same way).
MIN_SIDE = 128


def compute_padding(h: int, w: int) -> List[int]:
    """Upstream ``InputPadder(mode="sintel")``: centred pad to a multiple of 8.

    Returns ``[left, right, top, bottom]``.
    """
    pad_ht = (((h // STRIDE) + 1) * STRIDE - h) % STRIDE
    pad_wd = (((w // STRIDE) + 1) * STRIDE - w) % STRIDE
    return [pad_wd // 2, pad_wd - pad_wd // 2, pad_ht // 2, pad_ht - pad_ht // 2]


def resized_hw(h: int, w: int, input_size: Optional[int]) -> Tuple[int, int]:
    """Working canvas before padding.

    ``input_size`` caps the long side: frames whose long side exceeds it are
    downscaled (aspect kept) and smaller frames run at native resolution, which
    is how upstream evaluates (``scale`` 0 on Sintel and KITTI, a 0.5 downscale
    for 1080p Spring).
    """
    if input_size is None or int(input_size) <= 0 or max(h, w) <= int(input_size):
        return h, w
    scale = float(input_size) / float(max(h, w))
    return max(1, int(round(h * scale))), max(1, int(round(w * scale)))


def preprocess_numpy(
    image: np.ndarray, input_size: Optional[int] = None
) -> Tuple[torch.Tensor, dict]:
    """RGB uint8 ``(H, W, 3)`` to a padded ``[0, 1]`` tensor ``(1, 3, Hp, Wp)``.

    The returned ``meta`` carries what postprocess needs to go back to the
    original canvas: ``orig_hw``, ``work_hw`` (canvas before padding) and
    ``pad`` (``[left, right, top, bottom]``).
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected an (H, W, 3) RGB image, got {image.shape}")
    h, w = int(image.shape[0]), int(image.shape[1])
    x = torch.from_numpy(np.array(image, dtype=np.uint8, copy=True)).permute(2, 0, 1)[None].float()
    x = x / 255.0
    work_h, work_w = resized_hw(h, w, input_size)
    if (work_h, work_w) != (h, w):
        x = F.interpolate(x, size=(work_h, work_w), mode="bilinear", align_corners=False)
    pad = compute_padding(work_h, work_w)
    if work_h + pad[2] + pad[3] < MIN_SIDE or work_w + pad[0] + pad[1] < MIN_SIDE:
        raise ValueError(
            f"SEA-RAFT needs frames of at least {MIN_SIDE}x{MIN_SIDE} pixels, got "
            f"{work_w}x{work_h}."
        )
    x = F.pad(x, pad, mode="replicate")
    return x, {"orig_hw": (h, w), "work_hw": (work_h, work_w), "pad": pad}


def unpad(x: torch.Tensor, pad: List[int]) -> torch.Tensor:
    ht, wd = x.shape[-2:]
    return x[..., pad[2] : ht - pad[3], pad[0] : wd - pad[1]]


def flow_to_original(
    flow: torch.Tensor, pad: List[int], orig_hw: Tuple[int, int]
) -> torch.Tensor:
    """Un-pad ``(B, 2, Hp, Wp)`` flow and bring it to ``orig_hw``, rescaling vectors."""
    flow = unpad(flow, pad)
    h, w = flow.shape[-2:]
    oh, ow = orig_hw
    if (h, w) != (oh, ow):
        flow = F.interpolate(flow, size=(oh, ow), mode="bilinear", align_corners=False)
        scale = flow.new_tensor([ow / w, oh / h]).view(1, 2, 1, 1)
        flow = flow * scale
    return flow


def map_to_original(
    x: torch.Tensor, pad: List[int], orig_hw: Tuple[int, int]
) -> torch.Tensor:
    """Un-pad a ``(B, C, Hp, Wp)`` scalar map and resize it to ``orig_hw``."""
    x = unpad(x, pad)
    if tuple(x.shape[-2:]) != tuple(orig_hw):
        x = F.interpolate(x, size=tuple(orig_hw), mode="bilinear", align_corners=False)
    return x
