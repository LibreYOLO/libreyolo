# Adapted from LGM (https://github.com/3DTopia/LGM, commit
# fe8d12cff8c827df7bb77a3c8e8b37408cb6fe4c), MIT, Copyright (c) 2024 3D Topia:
# ``app.py`` / ``infer.py`` (preprocessing and the two stages),
# ``core/models.py`` (``prepare_default_rays``), ``core/utils.py``
# (``get_rays``), ``mvdream/pipeline_mvdream.py`` (the ImageDream sampler) and
# ``mvdream/mv_unet.py`` (``get_camera``). ``orbit_camera``, ``look_at`` and
# ``recenter`` follow kiuikit (https://github.com/ashawkey/kiuikit, v0.2.7,
# Apache-2.0, Copyright (c) Jiaxiang Tang), which upstream imports.
"""LGM around the networks: preprocessing, the multi-view sampler, rays, frame.

``run_stages`` is upstream's image-to-3D path for one image and one seed,
with each stage exposed for the parity gate: the object is cut out and
recentred on white; ImageDream samples four views around it with DDIM and
classifier-free guidance; LGM turns the four views and their camera rays into
Gaussians. Noise comes from a generator on the model's device seeded with the
seed, drawn in upstream's order, so on CUDA a seed reproduces upstream's
object after ``kiui.seed_everything(seed)``.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
VIEW_SIZE = 256  # ImageDream's output views and LGM's input size
FOVY = 49.1  # LGM's camera field of view, degrees
CAMERA_RADIUS = 1.5  # LGM's camera distance, in its [-1, 1] object units
BORDER_RATIO = 0.2  # recenter: the object spans 1 - 0.2 of the square
MIN_OPACITY = 0.005  # upstream's save_ply drops fainter Gaussians


# -- Cameras ------------------------------------------------------------------


def _normalize(x: np.ndarray) -> np.ndarray:
    return x / np.sqrt(np.maximum(np.sum(x * x, axis=-1, keepdims=True), 1e-20))


def orbit_camera(elevation: float, azimuth: float, radius: float = 1.0) -> np.ndarray:
    """OpenGL camera-to-world pose looking at the origin, y up (kiuikit's convention, degrees).

    Elevation -90 looks down from +y; azimuth 0 sits on +z, 90 on +x.
    """
    elevation, azimuth = np.deg2rad(elevation), np.deg2rad(azimuth)
    campos = np.array(
        [
            radius * np.cos(elevation) * np.sin(azimuth),
            -radius * np.sin(elevation),
            radius * np.cos(elevation) * np.cos(azimuth),
        ]
    ) + np.zeros([3], dtype=np.float32)
    forward = _normalize(campos)
    right = _normalize(np.cross(np.array([0, 1, 0], dtype=np.float32), forward))
    up = _normalize(np.cross(forward, right))
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = np.stack([right, up, forward], axis=1)
    pose[:3, 3] = campos
    return pose


def imagedream_cameras(elevation: float, views: int = 4) -> torch.Tensor:
    """``[views + 1, 16]`` flattened poses in Blender axes for ImageDream; the extra row (zeros) is the input view."""
    cameras = []
    for azimuth in np.arange(0, 360, 360 / views):
        pose = orbit_camera(elevation, azimuth, radius=1)
        pose[2] *= -1
        pose[[1, 2]] = pose[[2, 1]]
        cameras.append(pose.flatten())
    cameras.append(np.zeros_like(cameras[0]))
    return torch.from_numpy(np.stack(cameras, axis=0)).float()


def ray_embeddings(elevation: float, size: int = VIEW_SIZE) -> torch.Tensor:
    """``[4, 6, size, size]`` Plücker rays (o x d, d) of LGM's four views at azimuth 0, 90, 180, 270."""
    focal = size * 0.5 / np.tan(0.5 * np.deg2rad(FOVY))
    x, y = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="xy")
    x, y = x.flatten(), y.flatten()
    directions = F.pad(
        torch.stack(
            [(x - size * 0.5 + 0.5) / focal, (y - size * 0.5 + 0.5) / focal * -1.0],
            dim=-1,
        ),
        (0, 1),
        value=-1.0,
    )
    rays = []
    for azimuth in (0, 90, 180, 270):
        pose = torch.from_numpy(orbit_camera(elevation, azimuth, radius=CAMERA_RADIUS))
        d = directions @ pose[:3, :3].transpose(0, 1)
        o = pose[:3, 3].unsqueeze(0).expand_as(d).view(size, size, 3)
        d = (
            d / torch.sqrt(torch.clamp(torch.sum(d * d, -1, keepdim=True), min=1e-20))
        ).view(size, size, 3)
        rays.append(torch.cat([torch.cross(o, d, dim=-1), d], dim=-1))
    return torch.stack(rays, dim=0).permute(0, 3, 1, 2).contiguous()


# -- Input --------------------------------------------------------------------


def recenter(
    image: np.ndarray, mask: np.ndarray, border_ratio: float = BORDER_RATIO
) -> np.ndarray:
    """Square uint8 RGBA canvas (zeros) with the masked object scaled to ``1 - border_ratio`` of its side, centred."""
    import cv2

    pixels = image.astype(np.float32) / 255
    height, width, channels = pixels.shape
    size = max(height, width)
    result = np.zeros((size, size, channels), dtype=np.float32)
    rows, cols = np.nonzero(mask)
    top, bottom, left, right = rows.min(), rows.max(), cols.min(), cols.max()
    # Upstream's exclusive extents, kept so normal crops stay bit-identical;
    # a one-pixel-thin object keeps a one-pixel span instead of an empty crop.
    h, w = max(bottom - top, 1), max(right - left, 1)
    scale = int(size * (1 - border_ratio)) / max(h, w)
    h2, w2 = max(int(h * scale), 1), max(int(w * scale), 1)
    y0, x0 = (size - h2) // 2, (size - w2) // 2
    result[y0 : y0 + h2, x0 : x0 + w2] = cv2.resize(
        pixels[top : top + h, left : left + w], (w2, h2), interpolation=cv2.INTER_AREA
    ).reshape(h2, w2, channels)
    return (result * 255).astype(np.uint8)


def preprocess_image(image: Image.Image, birefnet) -> np.ndarray:
    """``[S, S, 3]`` float32 in [0, 1]: the object recentred to 80 % of a square, on white.

    Upstream's steps, with one difference: an image with a real alpha channel
    is its own cutout (upstream always runs rembg's U2Net), otherwise
    LibreYOLO's BiRefNet removes the background.
    """
    from ..image3d import has_real_alpha, remove_background

    if image.mode in ("LA", "PA") or (
        image.mode == "P" and "transparency" in image.info
    ):
        image = image.convert("RGBA")
    cutout = (
        image
        if has_real_alpha(image)
        else remove_background(birefnet, image.convert("RGB"))
    )
    carved = np.array(cutout.convert("RGBA"))
    mask = carved[..., 3] > 0
    if not mask.any():
        raise ValueError(
            "No object found in the image: its alpha channel is empty after background removal."
        )
    pixels = recenter(carved, mask).astype(np.float32) / 255
    return pixels[..., :3] * pixels[..., 3:4] + (1 - pixels[..., 3:4])


# -- Multi-view stage ---------------------------------------------------------


@torch.no_grad()
def encode_text(tokenizer, text_encoder, device) -> torch.Tensor:
    """CLIP text tokens of the empty prompt, ``[1, 77, 1024]``: image-to-3D runs ImageDream with no text."""
    ids = tokenizer(
        "",
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).input_ids
    return text_encoder(ids.to(device))[0]


@torch.no_grad()
def encode_image(modules, image: np.ndarray) -> torch.Tensor:
    """Penultimate-layer CLIP ViT-H/14 tokens of the input, ``[1, 257, 1280]``."""
    dtype = next(modules.image_encoder.parameters()).dtype
    pixels = modules.feature_extractor(
        (image * 255).astype(np.uint8), return_tensors="pt"
    ).pixel_values
    pixels = pixels.to(device=modules.device, dtype=dtype)
    return modules.image_encoder(pixels, output_hidden_states=True).hidden_states[-2]


@torch.no_grad()
def sample_views(
    modules,
    image: np.ndarray,
    generator: torch.Generator,
    *,
    steps: int,
    guidance: float,
    elevation: float,
    image_tokens: torch.Tensor | None = None,
    out: dict[str, object] | None = None,
) -> np.ndarray:
    """ImageDream: the recentred image -> ``[4, 256, 256, 3]`` float32 views at azimuth 0, 90, 180, 270.

    Upstream's ``MVDreamPipeline.__call__`` with an image and the empty
    prompt: DDIM (eta 0), guidance on when ``guidance > 1``. The input's VAE
    latent sample and the starting noise are drawn from ``generator``, in
    that order.
    """
    device, unet, vae, scheduler = (
        modules.device,
        modules.unet,
        modules.vae,
        modules.scheduler,
    )
    # set_timesteps mutates the scheduler; each call samples with its own copy.
    scheduler = copy.deepcopy(scheduler)
    dtype = next(unet.parameters()).dtype
    frames = 5  # four views plus the input image's slot
    if image_tokens is None:
        image_tokens = encode_image(modules, image)
    pixels = torch.from_numpy(image).unsqueeze(0).permute(0, 3, 1, 2).to(device)
    pixels = F.interpolate(
        2 * pixels - 1, (VIEW_SIZE, VIEW_SIZE), mode="bilinear", align_corners=False
    )
    posterior = vae.encode(pixels.to(dtype)).latent_dist
    noise = torch.randn(
        posterior.mean.shape,
        generator=generator,
        device=device,
        dtype=posterior.mean.dtype,
    )
    image_latents = (posterior.mean + posterior.std * noise) * vae.config.scaling_factor
    latent_size = VIEW_SIZE // 8
    latents = torch.randn(
        (frames, unet.in_channels, latent_size, latent_size),
        generator=generator,
        device=device,
        dtype=dtype,
    )
    latents = latents * scheduler.init_noise_sigma
    if out is not None:
        out["image_tokens"], out["image_latents"], out["noise"] = (
            image_tokens,
            image_latents,
            latents,
        )
    camera = imagedream_cameras(elevation).to(dtype=dtype, device=device)
    text = modules.text_tokens.to(dtype)
    guided = guidance > 1.0
    if guided:
        context = torch.cat([text] * frames * 2)
        ip = torch.cat(
            [torch.zeros_like(image_tokens)] * frames + [image_tokens] * frames
        )
        ip_img = torch.cat([torch.zeros_like(image_latents), image_latents])
        camera = torch.cat([camera] * 2)
    else:
        context, ip, ip_img = (
            torch.cat([text] * frames),
            torch.cat([image_tokens] * frames),
            image_latents,
        )
    scheduler.set_timesteps(steps, device=device)
    for t in scheduler.timesteps:
        model_input = torch.cat([latents] * (2 if guided else 1))
        model_input = scheduler.scale_model_input(model_input, t)
        timesteps = torch.tensor(
            [t] * model_input.shape[0], dtype=model_input.dtype, device=device
        )
        noise_pred = unet(
            model_input,
            timesteps,
            context,
            camera=camera,
            num_frames=frames,
            ip=ip,
            ip_img=ip_img,
        )
        if guided:
            uncond, cond = noise_pred.chunk(2)
            noise_pred = uncond + guidance * (cond - uncond)
        latents = scheduler.step(noise_pred, t, latents, eta=0.0, return_dict=False)[0]
    if out is not None:
        out["latents"] = latents
    decoded = vae.decode(latents / vae.config.scaling_factor).sample
    views = (decoded / 2 + 0.5).clamp(0, 1).cpu().permute(0, 2, 3, 1).float().numpy()
    # ImageDream's view k sits at azimuth 90 k in its own frame; LGM's view 0 is ImageDream's view 1.
    return np.stack([views[1], views[2], views[3], views[0]], axis=0)


# -- Gaussian stage -----------------------------------------------------------


@torch.no_grad()
def views_to_gaussians(modules, views: np.ndarray, elevation: float) -> torch.Tensor:
    """LGM: ``[4, 256, 256, 3]`` views -> ``[1, 65536, 14]`` Gaussians in LGM's frame (y up, ``[-1, 1]^3``)."""
    device = modules.device
    images = torch.from_numpy(views).permute(0, 3, 1, 2).float().to(device)
    images = F.interpolate(
        images, size=(VIEW_SIZE, VIEW_SIZE), mode="bilinear", align_corners=False
    )
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    images = (images - mean) / std
    rays = ray_embeddings(elevation).to(device)
    images = torch.cat([images, rays], dim=1).unsqueeze(0)
    if modules.autocast:
        with torch.autocast(device_type=device.type, dtype=torch.float16):
            return modules.lgm(images)
    return modules.lgm(images)


# LGM's frame is y up with the views on the xz plane; LibreYOLO's object frame
# is z up in [-0.5, 0.5]^3: (x, y, z) -> (x, -z, y) / 2, +90 degrees about x.
_LGM_TO_OBJECT = (np.sqrt(0.5), np.sqrt(0.5), 0.0, 0.0)


def to_object_frame(gaussians: torch.Tensor) -> dict[str, torch.Tensor]:
    """``[1, N, 14]`` LGM Gaussians -> payload fields in the object frame, without the faintest (opacity < 0.005)."""
    g = gaussians[0].float()
    g = g[g[:, 3] >= MIN_OPACITY]
    x, y, z = g[:, 0], g[:, 1], g[:, 2]
    xyz = torch.stack([x, -z, y], dim=1) * 0.5
    w, a, b, c = _LGM_TO_OBJECT
    qw, qx, qy, qz = g[:, 7], g[:, 8], g[:, 9], g[:, 10]
    rotations = torch.stack(
        [
            w * qw - a * qx - b * qy - c * qz,
            w * qx + a * qw + b * qz - c * qy,
            w * qy - a * qz + b * qw + c * qx,
            w * qz + a * qy - b * qx + c * qw,
        ],
        dim=1,
    )
    return {
        "xyz": xyz,
        "scales": g[:, 4:7] * 0.5,
        "rotations": rotations,
        "opacities": g[:, 3],
        "sh": ((g[:, 11:14] - 0.5) / 0.28209479177387814).unsqueeze(1),
    }


def input_camera(elevation: float) -> dict[str, float]:
    """LGM's view 0 in the object frame: on -y, ``elevation`` degrees down from level, 0.75 away."""
    return {
        "yaw": float(np.pi),
        "pitch": float(-np.deg2rad(elevation)),
        "radius": CAMERA_RADIUS * 0.5,
        "fov": FOVY,
        "margin": 1 / (1 - BORDER_RATIO),
    }


@torch.no_grad()
def run_stages(
    modules,
    image: np.ndarray,
    seed: int,
    *,
    steps: int = 30,
    guidance: float = 5.0,
    elevation: float = 0.0,
    image_tokens: torch.Tensor | None = None,
) -> dict[str, object]:
    """Recentred image (``preprocess_image``) -> Gaussians, returning every stage for inspection."""
    out: dict[str, object] = {}
    generator = torch.Generator(device=modules.device).manual_seed(seed)
    views = sample_views(
        modules,
        image,
        generator,
        steps=steps,
        guidance=guidance,
        elevation=elevation,
        image_tokens=image_tokens,
        out=out,
    )
    out["views"] = views
    out["lgm"] = views_to_gaussians(modules, views, elevation)
    out["gaussians"] = to_object_frame(out["lgm"])
    return out
