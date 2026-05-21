"""Preprocessing and bin-expectation decoding for L2CS gaze inference."""

from __future__ import annotations

import math
from typing import Iterable, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torchvision import transforms


_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


# Upstream L2CS preprocessing: ``ToPILImage → Resize(448) → ToTensor →
# Normalize(ImageNet)``. We keep ``Resize(448)`` as-is because the released
# checkpoints were trained against that exact transform's bilinear behavior;
# changing the interpolation backend silently shifts predicted angles by
# a fraction of a degree.
_TRANSFORM = transforms.Compose(
    [
        transforms.Resize(448),
        transforms.ToTensor(),
        transforms.Normalize(mean=list(_IMAGENET_MEAN), std=list(_IMAGENET_STD)),
    ]
)


def clamp_box(
    box: Sequence[float],
    width: int,
    height: int,
) -> Tuple[int, int, int, int]:
    """Clamp an xyxy box to image bounds and convert to ints."""
    x1, y1, x2, y2 = box
    x1 = max(0, int(round(float(x1))))
    y1 = max(0, int(round(float(y1))))
    x2 = min(int(width), int(round(float(x2))))
    y2 = min(int(height), int(round(float(y2))))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Degenerate face box after clamping: {(x1, y1, x2, y2)}")
    return x1, y1, x2, y2


def crop_face(image_rgb: np.ndarray, box: Sequence[float]) -> Image.Image:
    """Crop a face from an RGB ``HxWxC`` numpy image and return a PIL image."""
    h, w = image_rgb.shape[:2]
    x1, y1, x2, y2 = clamp_box(box, w, h)
    crop = image_rgb[y1:y2, x1:x2]
    return Image.fromarray(crop)


def preprocess_face_crops(
    crops: Iterable[Image.Image],
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Apply L2CS preprocessing to a batch of PIL face crops.

    Returns a ``(N, 3, 448, 448)`` float tensor on ``device``,
    normalized with ImageNet statistics.
    """
    tensors = [_TRANSFORM(crop) for crop in crops]
    if not tensors:
        return torch.empty((0, 3, 448, 448), device=device)
    return torch.stack(tensors).to(device)


def bin_logits_to_angles(
    yaw_logits: torch.Tensor,
    pitch_logits: torch.Tensor,
    num_bins: int = 90,
    bin_width_deg: float = 4.0,
    offset_deg: float = -180.0,
) -> torch.Tensor:
    """Convert L2CS bin logits to per-face (pitch_rad, yaw_rad).

    Upstream maps bin index i to ``offset_deg + (i + 0.5) * bin_width_deg``
    approximated as ``i * bin_width_deg + offset_deg`` (the +0.5 center
    correction is not applied in L2CS-Net's reference code, so we match
    that for parity).

    The math is performed in fp32 even if the input is fp16, since the
    90-bin softmax expectation can underflow at lower precision.

    Returns
    -------
    angles : torch.Tensor
        Shape ``(N, 2)``, columns ``[pitch, yaw]`` in radians.
    """
    yaw_logits = yaw_logits.float()
    pitch_logits = pitch_logits.float()

    yaw_probs = torch.softmax(yaw_logits, dim=1)
    pitch_probs = torch.softmax(pitch_logits, dim=1)

    idx = torch.arange(num_bins, dtype=torch.float32, device=yaw_logits.device)
    yaw_deg = (yaw_probs * idx).sum(dim=1) * bin_width_deg + offset_deg
    pitch_deg = (pitch_probs * idx).sum(dim=1) * bin_width_deg + offset_deg

    deg_to_rad = math.pi / 180.0
    return torch.stack([pitch_deg * deg_to_rad, yaw_deg * deg_to_rad], dim=1)
