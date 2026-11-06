# Adapted from CRM (https://github.com/thu-ml/CRM, commit
# 4964e36a593070a3045eb0f300935a771ff5c172, MIT, Copyright (c) 2024 TSAIL
# group): run.py (preprocessing), pipelines.py and libs/sample.py (the two
# diffusion stages) and inference.py (reconstruction and export).
"""CRM end to end: preprocessing, the two diffusion stages, reconstruction."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import torch
from PIL import Image

from ..image3d import has_real_alpha, remove_background
from .diffusion import run_stage
from .reconstruction import (
    NOISE_STEP,
    add_noise,
    bake_texture,
    clean_mesh,
    extract_mesh,
    stage_noise,
    to_object_frame,
    triplane_input,
    vertex_colors,
)

_BACKGROUND = (127, 127, 127)
_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def preprocess_image(image: Image.Image, birefnet) -> Image.Image:
    """Upstream's run.py: cut out, pad to a square, composite on grey (127).

    A real alpha channel is the cutout; otherwise LibreYOLO's BiRefNet removes
    the background (upstream: rembg).
    """
    if image.mode in ("LA", "PA") or (image.mode == "P" and "transparency" in image.info):
        image = image.convert("RGBA")
    if not has_real_alpha(image):
        image = remove_background(birefnet, image.convert("RGB"))
    image = image.convert("RGBA")
    side = max(image.size)
    if image.size != (side, side):
        square = Image.new("RGBA", (side, side), (0, 0, 0, 0))
        square.paste(image, ((side - image.width) // 2, (side - image.height) // 2))
        image = square
    background = Image.new("RGBA", image.size, _BACKGROUND + (255,))
    return Image.alpha_composite(background, image).convert("RGB")


def _pixels(image: Image.Image, device) -> torch.Tensor:
    """256 x 256 bilinear, in [-1, 1], [1, 3, 256, 256] (torchvision's Resize on a PIL image)."""
    array = np.asarray(image.convert("RGB").resize((256, 256), Image.BILINEAR), dtype=np.float32)
    return (torch.from_numpy(array / 255.0).permute(2, 0, 1)[None] * 2 - 1).to(device)


def _clip_pixels(image: Image.Image, device) -> torch.Tensor:
    """OpenCLIP's eval transform: bicubic resize of the short side to 224, centre crop, normalize."""
    width, height = image.size
    scale = 224 / min(width, height)
    size = (max(224, int(round(width * scale))), max(224, int(round(height * scale))))
    image = image.convert("RGB").resize(size, Image.BICUBIC)
    left, top = int(round((size[0] - 224) / 2.0)), int(round((size[1] - 224) / 2.0))
    image = image.crop((left, top, left + 224, top + 224))
    array = np.asarray(image, dtype=np.float32) / 255.0
    mean, std = np.array(_CLIP_MEAN, np.float32), np.array(_CLIP_STD, np.float32)
    return torch.from_numpy((array - mean) / std).permute(2, 0, 1)[None].to(device)


@torch.no_grad()
def multiview(modules, prepared: Image.Image, noise: Dict[str, torch.Tensor], steps: int, guidance: float):
    """Both diffusion stages: six views and their CCMs, uint8 [6, 256, 256, 3] each."""
    device = modules.device
    pixels = _pixels(prepared, device)
    clip_pixels = _clip_pixels(prepared, device)
    views = run_stage(
        modules.pixel,
        pixels=pixels,
        clip_pixels=clip_pixels,
        noise={"posterior": noise["stage1_posterior"], "latents": noise["stage1_latents"]},
        steps=steps,
        guidance=guidance,
        half=modules.half,
    )
    views = views[:6]  # drop the reference view
    view_pixels = torch.stack(
        [_pixels(Image.fromarray(v), device)[0] for v in views]
    )
    ccms = run_stage(
        modules.ccm,
        pixels=pixels,
        clip_pixels=clip_pixels,
        noise={"posterior": noise["stage2_posterior"], "latents": noise["stage2_latents"]},
        steps=steps,
        guidance=guidance,
        stage2_views=view_pixels,
        half=modules.half,
    )
    return views, ccms


@torch.no_grad()
def reconstruct(modules, views: np.ndarray, ccms: np.ndarray, triplane_noise: torch.Tensor):
    """Views and CCMs -> 32-channel triplane features [1, 32, 256, 768]."""
    device = modules.device
    triplane = triplane_input(views, ccms).to(device)
    noisy = add_noise(triplane, (triplane_noise * 0.5 + 0.5).to(device), NOISE_STEP)
    t = torch.full((1,), NOISE_STEP, dtype=torch.long, device=device)
    return modules.net.unet2(noisy, t)


@torch.no_grad()
def generate_mesh(modules, prepared: Image.Image, seed: int, *, steps: int, guidance: float, texture: bool) -> Dict[str, Any]:
    noise = stage_noise(seed)
    views, ccms = multiview(modules, prepared, noise, steps, guidance)
    features = reconstruct(modules, views, ccms, noise["triplane"])
    vertices, faces = extract_mesh(modules.net, features, modules.grid)
    vertices, faces = clean_mesh(vertices.cpu().numpy(), faces.cpu().numpy())
    colors = vertex_colors(modules.net, features, torch.from_numpy(vertices).to(modules.device))
    out: Dict[str, Any] = {
        "views": views,
        "ccms": ccms,
        "vertex_colors": (colors.float().cpu().numpy() * 0.5 + 0.5).clip(0, 1).astype(np.float32),
    }
    if texture:
        out.update(bake_texture(modules.net, features, vertices, faces))
    out["vertices"], out["faces"] = to_object_frame(vertices, faces)
    if texture:
        out["uv_faces"] = out["uv_faces"][:, [2, 1, 0]].copy()
    return out
