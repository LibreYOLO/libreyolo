# Ported from Speridlabs Idol (https://github.com/speridlabs/idol-sdk, commit
# 4702a7f00850e34b6e83a2170cc123c3e9570744), Apache-2.0, Copyright Speridlabs.
# Idol builds on TripoSplat (https://github.com/VAST-AI-Research/TripoSplat,
# commit d8db9e018b413dd9c4a9fe22463781bf98e8e68d, MIT, Copyright (c) 2026
# VAST), whose preprocessing and image encoding these functions also
# reproduce.
"""What the TripoSplat-lineage image-to-3D families share.

Speridlabs Idol and TripoSplat see their input the same way: the object is
cut out (BiRefNet, unless the image has a real alpha channel), cropped to a
square with some margin, resized to 1024 px and composited on black; DINOv3
ViT-H+/16 tokens and FLUX.2 VAE tokens of that image condition the
denoiser; and the denoiser's camera token decodes to the same shot. Each
step keeps upstream's arithmetic so a seed reproduces upstream's object.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Meta's facebook/dinov3-vith16plus-pretrain-lvd1689m@c807c9e, cast to bf16
# (the precision both families run it in) and mirrored ungated under the
# DINOv3 License.
DINOV3_REPO = "LibreYOLO/LibreDINOv3vith16plus"
DINOV3_REVISION = "e849a3492905c6dabc89758042bf1402770fb4e6"
# The FLUX.2 VAE, from the Apache-2.0, ungated klein repo (tensor-identical in
# bf16 to the copy in the gated FLUX.2-dev repo the upstream pipelines name).
FLUX2_VAE_REPO = "black-forest-labs/FLUX.2-klein-4B"
FLUX2_VAE_REVISION = "e7b7dc27f91deacad38e78976d1f2b499d76a294"


# -- Background removal and crop ---------------------------------------------


def has_real_alpha(image: Image.Image) -> bool:
    return (
        image.mode == "RGBA"
        and np.array(image.getchannel(3), dtype=np.int32).min() < 255
    )


@torch.no_grad()
def remove_background(birefnet, image: Image.Image) -> Image.Image:
    """RGB image -> RGBA with BiRefNet's alpha (1024 x 1024 inference, resized back)."""
    parameter = next(birefnet.parameters())
    width, height = image.size
    pixels = np.array(image, dtype=np.float32) / 255.0
    batch = torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0)
    batch = F.interpolate(batch, size=(1024, 1024), mode="bilinear", align_corners=True)
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    batch = ((batch - mean) / std).to(device=parameter.device, dtype=parameter.dtype)
    alpha = torch.sigmoid(birefnet(batch))
    alpha = F.interpolate(
        alpha.float(), size=(height, width), mode="bilinear", align_corners=True
    )[0, 0]
    alpha = (alpha.clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
    rgba = image.copy()
    rgba.putalpha(Image.fromarray(alpha, mode="L"))
    return rgba


def preprocess_image(
    image: Image.Image, birefnet, image_size: int = 1024, crop_margin: float = 1.2
) -> Image.Image:
    """Return the RGB image the encoders see.

    Resize so the short side is ``image_size``; if the image has no real alpha, BiRefNet predicts one; erode the
    alpha by 1 px so no background-coloured fringe survives; crop a square of side ``crop_margin`` x the object's
    bounding box; resize to ``image_size``; composite on black.

    Unlike upstream, a cutout stored as grayscale with alpha or as a palette
    with transparency counts as a cutout too; RGB and RGBA inputs are
    unchanged."""
    if image.mode in ("LA", "PA") or (
        image.mode == "P" and "transparency" in image.info
    ):
        image = image.convert("RGBA")
    width, height = image.size
    scale = image_size / min(width, height)
    image = image.resize(
        (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
        Image.LANCZOS,
    )
    if not has_real_alpha(image):
        image = remove_background(birefnet, image.convert("RGB"))
    image.putalpha(image.getchannel(3).filter(ImageFilter.MinFilter(3)))
    alpha = np.array(image.getchannel(3))
    ys, xs = np.nonzero(alpha)
    if len(xs) == 0:
        raise ValueError(
            "No object found in the image: its alpha channel is empty after background removal."
        )
    left, top, right, bottom = xs.min(), ys.min(), xs.max(), ys.max()
    center_x, center_y = (left + right) / 2, (top + bottom) / 2
    half = max(right - left, bottom - top) / 2 * crop_margin
    image = image.crop(
        [
            int(center_x - half),
            int(center_y - half),
            int(center_x + half),
            int(center_y + half),
        ]
    )
    image = image.resize((image_size, image_size), Image.LANCZOS)
    composite = Image.new("RGB", (image_size, image_size), (0, 0, 0))
    composite.paste(image, mask=image.split()[3])
    return composite


# -- Image encoders ----------------------------------------------------------


class ImageEncoder:
    """DINOv3 ViT-H+/16 tokens and FLUX.2 VAE tokens of a 1024 x 1024 image."""

    def __init__(self, dinov3, flux2_vae):
        self.dinov3 = dinov3
        self.flux2_vae = flux2_vae

    def _dinov3_tokens(self, images: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor(IMAGENET_MEAN, device=images.device).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, device=images.device).view(1, 3, 1, 1)
        images = (images.float() - mean) / std
        dtype = next(self.dinov3.parameters()).dtype
        tokens = self.dinov3(pixel_values=images.to(dtype)).last_hidden_state.float()
        return F.layer_norm(tokens, tokens.shape[-1:])

    def _flux2_tokens(
        self, images: torch.Tensor, generator: Optional[torch.Generator]
    ) -> torch.Tensor:
        vae = self.flux2_vae
        latents = vae.encode(images.to(torch.bfloat16) * 2 - 1).latent_dist.sample(
            generator=generator
        )
        batch, channels, height, width = latents.shape
        latents = latents.view(batch, channels, height // 2, 2, width // 2, 2).permute(
            0, 1, 3, 5, 2, 4
        )
        latents = latents.reshape(batch, channels * 4, height // 2, width // 2)
        mean = vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
        std = torch.sqrt(
            vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps
        ).to(latents.device, latents.dtype)
        tokens = (
            ((latents - mean) / std).float().flatten(2).transpose(1, 2).contiguous()
        )
        padding = torch.zeros(
            batch, 5, tokens.shape[2], dtype=tokens.dtype, device=tokens.device
        )
        return torch.cat([padding, tokens], dim=1)

    @torch.no_grad()
    def __call__(
        self, images: torch.Tensor, generator: Optional[torch.Generator] = None
    ) -> Dict[str, torch.Tensor]:
        """images [B, 3, 1024, 1024] in [0, 1] -> {'feature1': DINOv3 tokens, 'feature2': FLUX.2 VAE tokens}."""
        return {
            "feature1": self._dinov3_tokens(images),
            "feature2": self._flux2_tokens(images, generator),
        }

    def to(self, device) -> "ImageEncoder":
        self.dinov3.to(device)
        self.flux2_vae.to(device)
        return self


def load_image_encoder(device) -> ImageEncoder:
    """The two encoders at their pinned revisions, in bf16, on ``device``."""
    import diffusers
    from diffusers.models import AutoencoderKLFlux2
    from transformers import AutoModel

    # ``dtype`` replaced ``torch_dtype`` in diffusers 0.40.
    release = tuple(int(part) for part in diffusers.__version__.split(".")[:2])
    vae_dtype_argument = "dtype" if release >= (0, 40) else "torch_dtype"
    dinov3 = AutoModel.from_pretrained(
        DINOV3_REPO, revision=DINOV3_REVISION, dtype=torch.bfloat16
    )
    flux2_vae = AutoencoderKLFlux2.from_pretrained(
        FLUX2_VAE_REPO,
        subfolder="vae",
        revision=FLUX2_VAE_REVISION,
        **{vae_dtype_argument: torch.bfloat16},
    )
    return ImageEncoder(
        dinov3.eval().requires_grad_(False).to(device),
        flux2_vae.eval().requires_grad_(False).to(device),
    )


def pixels_of(image: Image.Image, device) -> torch.Tensor:
    """RGB PIL image -> [1, 3, H, W] float in [0, 1] on ``device``."""
    return (
        torch.from_numpy(np.array(image))
        .permute(2, 0, 1)
        .float()
        .div(255.0)[None]
        .to(device)
    )


# -- Camera ------------------------------------------------------------------


def decode_camera(camera_latent: torch.Tensor) -> Dict[str, float]:
    """camera_latent [5] -> yaw and pitch (radians), radius and vertical field of view (degrees)."""
    direction = camera_latent[:3].float()
    inverse_distance, half_width = float(camera_latent[3]), float(camera_latent[4])
    norm = float(torch.norm(direction))
    dx, dy, dz = (direction / norm).tolist() if norm > 1e-6 else (0.0, 0.0, 1.0)
    if inverse_distance < 1e-3:  # orthographic
        radius = 1000.0
        fov = math.degrees(2 * math.atan(half_width / radius))
    else:
        radius = 1.0 / inverse_distance
        fov = math.degrees(2 * math.atan(half_width * inverse_distance))
    return {
        "yaw": math.atan2(dx, dy),
        "pitch": math.asin(float(np.clip(dz, -1.0, 1.0))),
        "radius": radius,
        "fov": fov,
    }
