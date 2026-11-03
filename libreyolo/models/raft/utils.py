"""Pre and postprocessing helpers for LibreRAFT.

Padding mirrors upstream ``core/utils/utils.py::InputPadder`` in its default
("sintel") mode: replicate padding to a multiple of 8, split evenly between
the two sides of each axis.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from ...utils.image_loader import ImageInput, ImageLoader

PAD_MULTIPLE = 8


def pad_amounts(height: int, width: int, multiple: int = PAD_MULTIPLE) -> Tuple[int, int, int, int]:
    """Return ``(left, right, top, bottom)`` padding exactly as InputPadder."""
    pad_ht = (((height // multiple) + 1) * multiple - height) % multiple
    pad_wd = (((width // multiple) + 1) * multiple - width) % multiple
    return pad_wd // 2, pad_wd - pad_wd // 2, pad_ht // 2, pad_ht - pad_ht // 2


def pad_image(tensor: torch.Tensor, multiple: int = PAD_MULTIPLE) -> torch.Tensor:
    """Replicate-pad a ``(B, C, H, W)`` tensor to a multiple of ``multiple``."""
    _, _, h, w = tensor.shape
    pads = pad_amounts(h, w, multiple)
    if not any(pads):
        return tensor
    return F.pad(tensor, list(pads), mode="replicate")


def preprocess_image(
    image: ImageInput,
    *,
    color_format: str = "auto",
) -> Tuple[torch.Tensor, Image.Image, Tuple[int, int], float]:
    """Load one RGB frame as a padded float ``(1, 3, H, W)`` tensor in [0, 255].

    RAFT normalises to [-1, 1] inside its forward (``2 * (x / 255) - 1``), so
    the tensor stays in the 0 to 255 range here, as in upstream ``demo.py``.
    No resize: RAFT runs at native resolution, so the ratio is 1.
    """
    img = ImageLoader.load(image, color_format=color_format)
    original_size = img.size
    arr = np.asarray(img, dtype=np.float32)
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).contiguous()
    return pad_image(tensor), img, original_size, 1.0


def preprocess_numpy(img_rgb_hwc: np.ndarray, input_size):
    """Exporter calibration helper: RGB HWC to CHW float in [0, 255]."""
    del input_size
    arr = np.asarray(img_rgb_hwc, dtype=np.float32)
    if arr.max() <= 1.0:
        arr = arr * 255.0
    return arr.transpose(2, 0, 1).astype(np.float32), 1.0


def postprocess_flow(flow: torch.Tensor, original_size: Tuple[int, int]) -> torch.Tensor:
    """Map a padded ``(B, 2, Hp, Wp)`` flow back to the original canvas.

    Removes the InputPadder padding, then, if the remaining canvas still differs
    from the original image (a resized input), resizes it and scales the vectors
    by ``W_orig / W_in`` and ``H_orig / H_in``. Returns ``(H, W, 2)`` float32.
    """
    orig_w, orig_h = original_size
    flow = flow.float()
    _, _, hp, wp = flow.shape
    # Padding is a function of the unpadded size; recover it from the input.
    if hp - orig_h < PAD_MULTIPLE and wp - orig_w < PAD_MULTIPLE and hp >= orig_h and wp >= orig_w:
        left, right, top, bottom = pad_amounts(orig_h, orig_w)
        flow = flow[..., top : hp - bottom, left : wp - right]
    _, _, h_in, w_in = flow.shape
    if (h_in, w_in) != (orig_h, orig_w):
        flow = F.interpolate(flow, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
        scale = flow.new_tensor([orig_w / w_in, orig_h / h_in]).view(1, 2, 1, 1)
        flow = flow * scale
    return flow[0].permute(1, 2, 0).contiguous().cpu()


__all__ = [
    "PAD_MULTIPLE",
    "pad_amounts",
    "pad_image",
    "postprocess_flow",
    "preprocess_image",
    "preprocess_numpy",
]
