# Adapted from TRELLIS (https://github.com/microsoft/TRELLIS, commit
# 193e0502b5cb3eb5997ec457d5340e95d09b158e), MIT, Copyright (c) Microsoft
# Corporation: ``trellis/pipelines``.
"""TRELLIS around the networks: preprocessing, image tokens, the two flow samplers.

``run_stages`` is upstream's ``TrellisImageTo3DPipeline.run`` for one image
and one sample, with each stage exposed for the parity gate. Noise is drawn
from a CPU generator seeded like upstream seeds the global one, so a seed
gives the same starting noise, and so the same object, on every device.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .sparse import SparseTensor

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGE_SIZE = 518


def preprocess_image(
    image: Image.Image, birefnet, crop_margin: float = 1.2
) -> Image.Image:
    """Return the 518 x 518 RGB image the encoder sees, the object premultiplied on black.

    Upstream's steps: an image with a real alpha channel is its own cutout;
    otherwise it is downscaled to at most 1024 px and its background removed
    (upstream with rembg's U2Net, here with LibreYOLO's BiRefNet); the square
    crop is ``crop_margin`` x the bounding box of alpha > 0.8; resized to 518
    with Lanczos; RGB multiplied by alpha.
    """
    from ..image3d import has_real_alpha, remove_background

    if image.mode in ("LA", "PA") or (
        image.mode == "P" and "transparency" in image.info
    ):
        image = image.convert("RGBA")
    if has_real_alpha(image):
        cutout = image
    else:
        rgb = image.convert("RGB")
        scale = min(1, 1024 / max(rgb.size))
        if scale < 1:
            rgb = rgb.resize(
                (int(rgb.width * scale), int(rgb.height * scale)),
                Image.Resampling.LANCZOS,
            )
        cutout = remove_background(birefnet, rgb)
    alpha = np.array(cutout)[:, :, 3]
    found = np.argwhere(alpha > 0.8 * 255)
    if found.shape[0] == 0:
        raise ValueError(
            "No object found in the image: its alpha channel is empty after background removal."
        )
    left, top = found[:, 1].min(), found[:, 0].min()
    right, bottom = found[:, 1].max(), found[:, 0].max()
    centre = (left + right) / 2, (top + bottom) / 2
    size = int(max(right - left, bottom - top) * crop_margin)
    box = (
        centre[0] - size // 2,
        centre[1] - size // 2,
        centre[0] + size // 2,
        centre[1] + size // 2,
    )
    cutout = cutout.crop(box).resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)
    pixels = np.array(cutout).astype(np.float32) / 255
    pixels = pixels[:, :, :3] * pixels[:, :, 3:4]
    return Image.fromarray((pixels * 255).astype(np.uint8))


@torch.no_grad()
def encode_image(dinov2, image: Image.Image, device) -> torch.Tensor:
    """DINOv2 ViT-L/14 with registers: the pre-norm tokens of the last block, layer-normed (no affine)."""
    pixels = np.array(image.convert("RGB")).astype(np.float32) / 255
    tensor = torch.from_numpy(pixels).permute(2, 0, 1).float()[None].to(device)
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    tensor = (tensor - mean) / std
    tokens = dinov2(pixel_values=tensor, output_hidden_states=True).hidden_states[-1]
    return F.layer_norm(tokens, tokens.shape[-1:])


def time_grid(steps: int, shift: float) -> np.ndarray:
    grid = np.linspace(1, 0, steps + 1)
    return shift * grid / (1 + (shift - 1) * grid)


@torch.no_grad()
def sample_flow(
    model,
    noise,
    cond: torch.Tensor,
    steps: int,
    guidance: float,
    interval: Tuple[float, float],
    shift: float,
):
    """Upstream's ``FlowEulerGuidanceIntervalSampler``: Euler steps from t = 1 to 0.

    Inside ``interval`` the velocity is ``(1 + guidance) * v(cond) - guidance *
    v(zeroed cond)``; outside it, ``v(cond)``. ``noise`` is a dense tensor or
    a ``SparseTensor``."""
    neg_cond = torch.zeros_like(cond)
    t_seq = time_grid(steps, shift)
    sample = noise
    for i in range(steps):
        t, t_prev = float(t_seq[i]), float(t_seq[i + 1])
        batch = sample.shape[0]
        times = torch.tensor(
            [1000 * t] * batch, device=cond.device, dtype=torch.float32
        )
        v = model(sample, times, cond)
        if guidance > 0 and interval[0] <= t <= interval[1]:
            v_neg = model(sample, times, neg_cond)
            v = (1 + guidance) * v - guidance * v_neg
        sample = sample - (t - t_prev) * v
    return sample


@torch.no_grad()
def run_stages(
    modules,
    image: Image.Image,
    seed: int,
    *,
    structure: Dict[str, float],
    latent: Dict[str, float],
    stop_after: Optional[str] = None,
) -> Dict[str, object]:
    """Image (already preprocessed) -> Gaussians, returning every stage for inspection.

    ``structure`` and ``latent`` hold ``steps``, ``guidance``, ``interval`` and
    ``shift`` of the two samplers.
    """
    device = modules.device
    out: Dict[str, object] = {}
    cond = encode_image(modules.dinov2, image, device)
    out["cond"] = cond
    generator = torch.Generator().manual_seed(seed)
    flow = modules.structure_flow
    side = flow.resolution
    noise = torch.randn(1, flow.in_channels, side, side, side, generator=generator).to(
        device
    )
    z_s = sample_flow(flow, noise, cond, **structure)
    out["structure_latent"] = z_s
    occupancy = modules.structure_decoder(z_s)
    coords = torch.argwhere(occupancy > 0)[:, [0, 2, 3, 4]].int()
    if coords.shape[0] == 0:
        raise RuntimeError(
            "The structure stage found no occupied voxel for this image and seed."
        )
    out["coords"] = coords
    if stop_after == "structure":
        return out
    slat_flow = modules.latent_flow
    feats = torch.randn(coords.shape[0], slat_flow.in_channels, generator=generator).to(
        device
    )
    slat = sample_flow(slat_flow, SparseTensor(feats, coords), cond, **latent)
    mean = torch.tensor(modules.normalization["mean"], device=device)[None]
    std = torch.tensor(modules.normalization["std"], device=device)[None]
    slat = slat * std + mean
    out["latent"] = slat
    if stop_after == "latent":
        return out
    decoded = modules.gaussian_decoder(slat)
    out["gaussians"] = modules.gaussian_decoder.to_gaussians(decoded)
    return out
