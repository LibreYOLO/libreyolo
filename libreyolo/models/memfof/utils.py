"""Pre- and postprocessing for the MEMFOF optical flow family."""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

# MEMFOF's correlation pyramid starts at 1/16 and is halved four times while
# it is built (the last halving is computed even though unused), so the padded
# side must be at least 16 * 16 = 256 pixels: any side above 240 pads to it.
MIN_SIDE = 241


def input_hw(orig_h: int, orig_w: int, max_side: int) -> Tuple[int, int]:
    """Return the network canvas for an image: native unless the long side exceeds ``max_side``."""
    long_side = max(orig_h, orig_w)
    if long_side <= max_side:
        return orig_h, orig_w
    scale = max_side / float(long_side)
    return max(1, round(orig_h * scale)), max(1, round(orig_w * scale))


def preprocess_numpy(
    img_rgb_hwc: np.ndarray, max_side: int = 1920
) -> Tuple[np.ndarray, Tuple[float, float]]:
    """Prepare one RGB frame for MEMFOF.

    Upstream feeds frames at their native resolution as float ``[0, 255]`` and
    applies ``2 * x / 255 - 1`` plus replicate padding to a multiple of 16
    inside the network, so this returns a ``(3, H, W)`` float32 array still in
    ``[0, 255]``. Frames whose long side exceeds ``max_side`` are downscaled
    with area interpolation (upstream's ``resize_data`` filter). The second
    value is ``(sx, sy)``, the factors that map network-canvas vectors back to
    the original canvas.
    """
    img = np.asarray(img_rgb_hwc)
    orig_h, orig_w = img.shape[:2]
    in_h, in_w = input_hw(orig_h, orig_w, max_side)
    if (in_h, in_w) != (orig_h, orig_w):
        img = cv2.resize(img, (in_w, in_h), interpolation=cv2.INTER_AREA)
    chw = np.ascontiguousarray(img.transpose(2, 0, 1), dtype=np.float32)
    return chw, (orig_w / float(in_w), orig_h / float(in_h))


def flow_to_original(flow: torch.Tensor, orig_h: int, orig_w: int) -> torch.Tensor:
    """Map a ``(2, h, w)`` network-canvas flow to ``(orig_h, orig_w, 2)`` original-canvas pixels."""
    flow = flow.float()
    h, w = flow.shape[-2:]
    if (h, w) != (orig_h, orig_w):
        flow = F.interpolate(
            flow[None], size=(orig_h, orig_w), mode="bilinear", align_corners=False
        )[0]
        scale = flow.new_tensor([orig_w / float(w), orig_h / float(h)]).view(2, 1, 1)
        flow = flow * scale
    return flow.permute(1, 2, 0).contiguous()
