# Adapted from TRELLIS.2 (https://github.com/microsoft/TRELLIS.2, commit
# 75fbf0183001ed9876c8dbb35de6b68552ee08bd), MIT, Copyright (c) Microsoft
# Corporation: ``trellis2/pipelines`` and ``trellis2/modules/image_feature_extractor.py``.
"""TRELLIS.2 around the networks: preprocessing, image tokens, sampler, stages.

``run_stages`` is upstream's ``Trellis2ImageTo3DPipeline.run`` for one image
and one sample, geometry only, with every stage returned for the parity
gate. Noise is drawn from a CPU generator seeded the way upstream seeds the
global one (upstream draws on the CPU too), so a seed gives the same
starting noise, and so the same object, on every device.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from ..trellis.sparse import SparseTensor
from .mesh import flexible_dual_grid_to_mesh

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SIGMA_MIN = 1e-5
# Upstream's pipeline types, by output resolution: the sparse-structure grid,
# and the shape stages run (512 alone, or 512 then a finer one).
PIPELINES = {512: "512", 1024: "1024_cascade", 1536: "1536_cascade"}


def preprocess_image(
    image: Image.Image, birefnet, crop_margin: float = 1.0
) -> Image.Image:
    """The object premultiplied on black, cropped square around it, at most 1024 px.

    Upstream's steps: the image is downscaled to at most 1024 px; an image
    with a real alpha channel is its own cutout, otherwise its background is
    removed (upstream with BRIA's RMBG-2.0, here with LibreYOLO's BiRefNet);
    the square crop is ``crop_margin`` x the bounding box of alpha > 0.8
    (upstream: 1); RGB multiplied by alpha. The encoder resizes it.
    """
    from ..image3d import has_real_alpha, remove_background

    if image.mode in ("LA", "PA") or (
        image.mode == "P" and "transparency" in image.info
    ):
        image = image.convert("RGBA")
    scale = min(1, 1024 / max(image.size))
    if scale < 1:
        image = image.resize(
            (int(image.width * scale), int(image.height * scale)),
            Image.Resampling.LANCZOS,
        )
    if has_real_alpha(image):
        cutout = image
    else:
        cutout = remove_background(birefnet, image.convert("RGB"))
    alpha = np.array(cutout)[:, :, 3]
    found = np.argwhere(alpha > 0.8 * 255)
    if found.shape[0] == 0:
        raise ValueError(
            "No object found in the image: its alpha channel is empty after background removal."
        )
    left, top = np.min(found[:, 1]), np.min(found[:, 0])
    right, bottom = np.max(found[:, 1]), np.max(found[:, 0])
    centre = (left + right) / 2, (top + bottom) / 2
    size = int(max(right - left, bottom - top) * crop_margin)
    box = (
        centre[0] - size // 2,
        centre[1] - size // 2,
        centre[0] + size // 2,
        centre[1] + size // 2,
    )
    pixels = np.array(cutout.crop(box)).astype(np.float32) / 255
    pixels = pixels[:, :, :3] * pixels[:, :, 3:4]
    return Image.fromarray((pixels * 255).astype(np.uint8))


@torch.no_grad()
def encode_image(dinov3, image: Image.Image, size: int, device) -> torch.Tensor:
    """DINOv3 ViT-L/16 tokens of ``image`` at ``size`` px: the last block's output, layer-normed (no affine).

    Upstream calls the embeddings, the rotary table and the blocks itself and
    skips the model's final norm; so does this.
    """
    resized = image.resize((size, size), Image.Resampling.LANCZOS)
    pixels = np.array(resized.convert("RGB")).astype(np.float32) / 255
    tensor = torch.from_numpy(pixels).permute(2, 0, 1).float()[None].to(device)
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    tensor = (tensor - mean) / std
    tensor = tensor.to(dinov3.embeddings.patch_embeddings.weight.dtype)
    hidden = dinov3.embeddings(tensor, bool_masked_pos=None)
    position = dinov3.rope_embeddings(tensor)
    # transformers 4.x keeps the blocks on the model, 5.x on its encoder.
    layers = dinov3.layer if hasattr(dinov3, "layer") else dinov3.model.layer
    for layer in layers:
        hidden = layer(hidden, position_embeddings=position)
    return F.layer_norm(hidden, hidden.shape[-1:])


# -- Sampler -------------------------------------------------------------------


def time_grid(steps: int, rescale_t: float) -> list:
    grid = np.linspace(1, 0, steps + 1)
    return (rescale_t * grid / (1 + (rescale_t - 1) * grid)).tolist()


def _std(x) -> torch.Tensor:
    """Upstream's per-object std: torch's (unbiased) for dense tensors; sqrt(E[x^2] - E[x]^2) for sparse ones."""
    if isinstance(x, SparseTensor):
        lengths = torch.tensor(
            [s.stop - s.start for s in x.layout], dtype=torch.long, device=x.device
        )

        def mean(feats):
            per_row = feats.mean(dim=(1,), keepdim=True)
            if per_row.device.type == "mps":
                # segment_reduce has no MPS kernel.
                return torch.stack([per_row[s].mean(dim=0) for s in x.layout])
            return torch.segment_reduce(per_row, reduce="mean", lengths=lengths)

        return (mean(x.feats**2) - mean(x.feats) ** 2).sqrt()
    return x.std(dim=list(range(1, x.ndim)), keepdim=True)


@torch.no_grad()
def sample_flow(
    model,
    noise,
    cond: torch.Tensor,
    *,
    steps: int,
    guidance_strength: float,
    guidance_rescale: float,
    guidance_interval: Sequence[float],
    rescale_t: float,
    sigma_min: float = SIGMA_MIN,
    **model_kwargs: Any,
):
    """Upstream's ``FlowEulerGuidanceIntervalSampler``: Euler steps from t = 1 to 0.

    Inside ``guidance_interval`` the velocity is ``g * v(cond) + (1 - g) *
    v(zeroed cond)``, its x0 prediction rescaled toward the conditional one's
    std by ``guidance_rescale``; outside it, ``v(cond)``.
    """
    neg_cond = torch.zeros_like(cond)
    t_seq = time_grid(steps, rescale_t)
    sample = noise
    for i in range(steps):
        t, t_prev = t_seq[i], t_seq[i + 1]
        times = torch.tensor(
            [1000 * t] * sample.shape[0], device=cond.device, dtype=torch.float32
        )
        g = (
            guidance_strength
            if guidance_interval[0] <= t <= guidance_interval[1]
            else 1
        )
        if g == 1:
            pred = model(sample, times, cond, **model_kwargs)
        elif g == 0:
            pred = model(sample, times, neg_cond, **model_kwargs)
        else:
            pred_pos = model(sample, times, cond, **model_kwargs)
            pred_neg = model(sample, times, neg_cond, **model_kwargs)
            pred = g * pred_pos + (1 - g) * pred_neg
            if guidance_rescale > 0:
                scale = sigma_min + (1 - sigma_min) * t
                x_0_pos = (1 - sigma_min) * sample - scale * pred_pos
                x_0_cfg = (1 - sigma_min) * sample - scale * pred
                ratio = _std(x_0_pos) / _std(x_0_cfg)
                x_0 = guidance_rescale * (x_0_cfg * ratio) + (
                    1 - guidance_rescale
                ) * x_0_cfg
                pred = ((1 - sigma_min) * sample - x_0) / scale
        sample = sample - (t - t_prev) * pred
    return sample


def sampler_settings(params: Dict[str, Any], steps: int, guidance: float) -> Dict[str, Any]:
    """One stage's sampler arguments from upstream's ``pipeline.json`` params and the two knobs."""
    return {
        "steps": steps,
        "guidance_strength": guidance,
        "guidance_rescale": float(params["guidance_rescale"]),
        "guidance_interval": tuple(params["guidance_interval"]),
        "rescale_t": float(params["rescale_t"]),
    }


# -- Stages --------------------------------------------------------------------


def _denormalize(slat: SparseTensor, normalization: Dict[str, Sequence[float]]):
    std = torch.tensor(normalization["std"])[None].to(slat.device)
    mean = torch.tensor(normalization["mean"])[None].to(slat.device)
    return slat * std + mean


def _cascade_coords(
    hr_coords: torch.Tensor,
    lr_resolution: int,
    resolution: int,
    max_tokens: int,
) -> Tuple[torch.Tensor, int]:
    """Upstream's choice of the fine stage's voxels: shrink the resolution by 128 until they fit ``max_tokens``."""
    while True:
        quant = torch.cat(
            [
                hr_coords[:, :1],
                ((hr_coords[:, 1:] + 0.5) / lr_resolution * (resolution // 16)).int(),
            ],
            dim=1,
        )
        coords = quant.unique(dim=0)
        if coords.shape[0] < max_tokens or resolution == 1024:
            return coords, resolution
        resolution -= 128


@torch.no_grad()
def run_stages(
    modules,
    image: Image.Image,
    seed: int,
    *,
    resolution: int = 1024,
    structure: Dict[str, Any],
    shape: Dict[str, Any],
    max_tokens: int = 49152,
    conds: Optional[Dict[int, torch.Tensor]] = None,
    stop_after: Optional[str] = None,
) -> Dict[str, Any]:
    """Preprocessed image -> mesh, returning every stage for inspection.

    ``structure`` and ``shape`` are the two samplers' arguments
    (``sampler_settings``). ``conds`` reuses DINOv3 tokens already computed
    for ``image``, keyed by input size (512, 1024).
    """
    if resolution not in PIPELINES:
        raise ValueError(f"resolution must be one of {sorted(PIPELINES)}, got {resolution!r}.")
    device = modules.device
    out: Dict[str, Any] = {}
    conds = dict(conds or {})
    sizes = (512,) if resolution == 512 else (512, 1024)
    for size in sizes:
        if size not in conds:
            conds[size] = encode_image(modules.dinov3, image, size, device)
    out["conds"] = conds
    generator = torch.Generator().manual_seed(seed)

    # Sparse structure: 16^3 latent -> 64^3 occupancy, pooled to 32^3.
    flow = modules.structure_flow
    side = flow.resolution
    noise = torch.randn(
        1, flow.in_channels, side, side, side, generator=generator
    ).to(device)
    z_s = sample_flow(flow, noise, conds[512], **structure)
    out["structure_latent"] = z_s
    decoded = modules.structure_decoder(z_s) > 0
    ratio = decoded.shape[2] // 32
    if ratio != 1:
        decoded = F.max_pool3d(decoded.float(), ratio, ratio, 0) > 0.5
    coords = torch.argwhere(decoded)[:, [0, 2, 3, 4]].int()
    if coords.shape[0] == 0:
        raise RuntimeError(
            "The structure stage found no occupied voxel for this image and seed."
        )
    out["coords"] = coords
    if stop_after == "structure":
        return out

    # Shape latent at 512 (32^3 voxels).
    lr_flow = modules.shape_flow_512
    feats = torch.randn(coords.shape[0], lr_flow.in_channels, generator=generator).to(device)
    slat = sample_flow(lr_flow, SparseTensor(feats, coords), conds[512], **shape)
    slat = _denormalize(slat, modules.shape_normalization)
    out["shape_latent_512"] = slat
    decoder = modules.shape_decoder
    if resolution != 512:
        # Cascade: the 512 latent's predicted voxels seed the fine stage.
        hr_coords = decoder.upsample(slat, upsample_times=4)
        coords, resolution = _cascade_coords(hr_coords, 512, resolution, max_tokens)
        out["cascade_coords"] = coords
        hr_flow = modules.shape_flow_1024
        feats = torch.randn(
            coords.shape[0], hr_flow.in_channels, generator=generator
        ).to(device)
        slat = sample_flow(hr_flow, SparseTensor(feats, coords), conds[1024], **shape)
        slat = _denormalize(slat, modules.shape_normalization)
        out["shape_latent"] = slat
    else:
        out["shape_latent"] = slat
    out["resolution"] = resolution
    if stop_after == "latent":
        return out

    decoder.set_resolution(resolution)
    fields, subs = decoder.decode_fields(slat)
    out["fields"] = fields
    out["subdivisions"] = subs
    s = fields["layout"][0]
    vertices, faces = flexible_dual_grid_to_mesh(
        fields["coords"][s, 1:],
        fields["dual_vertices"][s],
        fields["intersected"][s],
        fields["split_weight"][s],
        resolution,
    )
    out["vertices"] = vertices
    out["faces"] = faces
    return out
