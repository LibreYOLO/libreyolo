# Adapted from Direct3D-S2 (https://github.com/DreamTechAI/Direct3D-S2, commit
# a1cf235b2881cff04a91900060a9546b40e7ee5d), MIT, Copyright (c) 2025
# DreamTechAI: ``direct3d_s2/pipeline.py``, ``direct3d_s2/utils`` and
# ``third_party/voxelize``.
"""Direct3D-S2 around the networks: preprocessing, the three flow stages, meshing.

``run_stages`` is upstream's ``Direct3DS2Pipeline.__call__`` for one image
and one sample, with each stage exposed for the parity gate:

1. dense: DINOv2 tokens condition the dense DiT over a 16^3 latent; the
   dense decoder's 64^3 occupancy gives the active latent voxels.
2. sparse 512: the sparse DiT denoises a 16-channel latent on those voxels;
   the SDF decoder turns it into a sparse SDF at 512^3, the refiner fills the
   dense 512^3 SDF, and marching cubes gives a mesh.
3. sparse 1024 (``resolution=1024``): that mesh is normalised and voxelised
   at 1024^3 within 4 voxels of its surface (upstream's CUDA UDF kernel, here
   in PyTorch), the second sparse DiT and decoder give a sparse SDF at
   1024^3, the sign refiner fills the inside/outside sign, and marching cubes
   gives the final mesh.

Noise is drawn from a ``torch.Generator`` on the model's device seeded with
the seed; upstream draws from the global generator of its CUDA device, so on
CUDA a seed reproduces upstream's starting noise.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

from .sparse import SparseTensor, sort_block

IMAGE_SIZE = 518
# Upstream's per-call constants (Direct3DS2Pipeline.__call__ and inference).
_DENSE_OCCUPANCY = 0.1
_SDF_LEVEL = 0.2
_UDF_VOXELS = 4.0
_UDF_FACTOR = 8


# -- Image ---------------------------------------------------------------------


def preprocess_image(image: Image.Image, birefnet=None) -> Image.Image:
    """Return the 518 x 518 RGB image the encoder sees: the object on white, padded square.

    Upstream's steps: an RGBA image is its own cutout, any other image has its
    background removed (upstream with BiRefNet through transformers, here
    LibreYOLO's BiRefNet network); crop to the alpha > 0 bounding box;
    composite on white; pad to a square with white; resize to 518; pad 16 px
    of white on every side and resize back to 518 (bilinear).
    """
    import torchvision.transforms.functional as TF

    from ..image3d import remove_background

    if image.mode in ("LA", "PA") or (
        image.mode == "P" and "transparency" in image.info
    ):
        image = image.convert("RGBA")
    if image.mode != "RGBA":
        if birefnet is None:
            raise ValueError("An image without alpha needs the background remover.")
        image = remove_background(birefnet, image.convert("RGB"))
    pixels = np.array(image)
    alpha = pixels[..., 3]
    mask = alpha > 0
    height, width = pixels.shape[:2]
    if mask.sum() < height * width * 0.001:
        top, bottom, left, right = 0, height - 1, 0, width - 1
    else:
        rows, cols = mask.nonzero()
        top, bottom, left, right = rows.min(), rows.max(), cols.min(), cols.max()
    pixels = pixels[top : bottom + 1, left : right + 1] / 255.0
    rgb = pixels[..., :3] * pixels[..., 3:] + (1 - pixels[..., 3:])
    rgb = Image.fromarray((rgb * 255).astype(np.uint8))
    width, height = rgb.size
    side = max(width, height)
    top = (side - height) // 2
    left = (side - width) // 2
    rgb = TF.pad(rgb, (left, top, side - width - left, side - height - top), fill=255)
    size = (IMAGE_SIZE, IMAGE_SIZE)
    rgb = TF.resize(rgb, size, interpolation=TF.InterpolationMode.BILINEAR)
    rgb = TF.pad(rgb, 16, fill=255)
    return TF.resize(rgb, size, interpolation=TF.InterpolationMode.BILINEAR)


# -- Sampling ------------------------------------------------------------------


def _scheduler(shift: float):
    from diffusers import FlowMatchEulerDiscreteScheduler

    return FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=shift)


@torch.no_grad()
def sample(
    model,
    noise: torch.Tensor,
    cond: torch.Tensor,
    *,
    steps: int,
    guidance: float,
    shift: float,
    coords: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Upstream's flow-matching Euler sampler with classifier-free guidance against zeroed tokens.

    As upstream (``do_classifier_free_guidance = guidance_scale > 0``),
    ``guidance=0`` runs the conditional pass only.

    ``coords`` makes the latent sparse: ``noise`` is then ``[T, C]`` features
    on those voxels.
    """
    scheduler = _scheduler(shift)
    scheduler.set_timesteps(steps, device=noise.device)
    uncond = torch.zeros_like(cond) if guidance > 0 else None
    latents = noise
    for t in scheduler.timesteps:
        time = torch.tensor([t], dtype=latents.dtype, device=latents.device)
        x = latents if coords is None else SparseTensor(latents, coords)
        velocity = model(x, time, cond)
        if coords is not None:
            velocity = velocity.feats
        if uncond is not None:
            negative = model(x, time, uncond)
            if coords is not None:
                negative = negative.feats
            velocity = negative + guidance * (velocity - negative)
        latents = scheduler.step(velocity, t, latents).prev_sample
    return latents


# -- Meshing -------------------------------------------------------------------


def marching_cubes(volume: np.ndarray, level: float) -> Tuple[np.ndarray, np.ndarray]:
    """Lewiner marching cubes (scikit-image); vertices scaled from voxel indices to [-1, 1)."""
    try:
        from skimage import measure
    except ImportError as exc:
        raise ImportError(
            "Direct3D-S2 meshes its SDF with scikit-image. Install with:\n"
            "    pip install 'libreyolo[gen]'"
        ) from exc
    vertices, faces, _, _ = measure.marching_cubes(volume, level, method="lewiner")
    return vertices / volume.shape[0] * 2 - 1, faces


def normalize_vertices(vertices: np.ndarray, scale: float = 0.95) -> np.ndarray:
    """Upstream's ``normalize_mesh``: centre the bounding box, longest side to ``2 * scale``."""
    low, high = vertices.min(axis=0), vertices.max(axis=0)
    return (vertices - (low + high) / 2) * (2.0 * scale / max(high - low))


def _point_segment_distance(p, a, b):
    d = b - a
    t = ((p - a) * d).sum(-1) / (d * d).sum(-1)
    t = t.unsqueeze(-1)
    closest = torch.where(t < 0, a, torch.where(t > 1, b, a + t * d))
    return (p - closest).norm(dim=-1)


def _point_triangle_distance(p, v0, v1, v2):
    """Upstream's ``pointToTriangleDistance`` in fp32, vectorised (``p`` ``[F, K, 3]``, vertices ``[F, 1, 3]``)."""
    same01 = (v0 == v1).all(-1)
    same02 = (v0 == v2).all(-1)
    same12 = (v1 == v2).all(-1)
    normal = torch.linalg.cross(v1 - v0, v2 - v0)
    flat = (normal * normal).sum(-1) == 0
    normal = normal / (normal * normal).sum(-1, keepdim=True).sqrt().clamp_min(1e-30)
    offset = p - v0
    plane = (normal * offset).sum(-1)
    projection = p - plane.unsqueeze(-1) * normal
    c0 = torch.linalg.cross(v0 - v1, v0 - projection)
    c1 = torch.linalg.cross(v1 - v2, v1 - projection)
    c2 = torch.linalg.cross(v2 - v0, v2 - projection)
    inside = ((c0 * c1).sum(-1) > 0) & ((c1 * c2).sum(-1) > 0) & ((c0 * c2).sum(-1) > 0)
    edges = torch.minimum(
        torch.minimum(
            _point_segment_distance(p, v0, v1), _point_segment_distance(p, v0, v2)
        ),
        _point_segment_distance(p, v1, v2),
    )
    distance = torch.where(inside, plane.abs(), edges)
    distance = torch.where(flat, offset.norm(dim=-1), distance)
    distance = torch.where(same12, _point_segment_distance(p, v0, v1), distance)
    distance = torch.where(same02, _point_segment_distance(p, v0, v1), distance)
    distance = torch.where(same01, _point_segment_distance(p, v0, v2), distance)
    return torch.where(same01 & same02, (p - v0).norm(dim=-1), distance)


@torch.no_grad()
def mesh_to_voxels(
    vertices: np.ndarray,
    faces: np.ndarray,
    device,
    size: int = 1024,
    threshold: float = _UDF_VOXELS,
    factor: int = _UDF_FACTOR,
) -> torch.Tensor:
    """Upstream's ``mesh2index``: latent voxels ``(0, x, y, z)`` whose fine grid points lie near the mesh.

    The mesh (normalised to [-0.95, 0.95]) is halved into [-0.5, 0.5]; a point
    ``i / (size - 1) - 0.5`` of the ``size^3`` grid counts when its distance to
    some triangle is under ``threshold / size``; each counted point marks the
    latent voxel ``index // factor``. Upstream evaluates the distance in a CUDA
    kernel over every triangle's bounding box widened by ``threshold + 1``
    points; this does the same in chunks of triangles.
    """
    verts = torch.as_tensor(np.ascontiguousarray(vertices), dtype=torch.float32, device=device) * 0.5
    tris = torch.as_tensor(np.ascontiguousarray(faces), dtype=torch.long, device=device)
    corners = verts[tris]  # [F, 3, 3]
    scale = size - 1
    reach = int(threshold + 1)
    low = torch.floor((corners.min(dim=1).values + 0.5) * scale).long().clamp(0, size - 1)
    high = torch.floor((corners.max(dim=1).values + 0.5) * scale).long().clamp(0, size - 1)
    low = (low - reach).clamp(min=0)
    high = (high + reach).clamp(max=size - 1)
    extent = high - low + 1  # [F, 3]
    limit = threshold / size
    side = size // factor
    occupied = torch.zeros(side**3, dtype=torch.bool, device=device)
    # Group triangles by box size so each chunk enumerates a fixed cube.
    order = torch.argsort(extent.max(dim=1).values)
    budget = 1 << 22
    widths = extent.max(dim=1).values[order].tolist()
    position = 0
    count = tris.shape[0]
    while position < count:
        # Boxes are sorted by size; the chunk enumerates its widest box's cube.
        stop = min(count, position + max(1, budget // widths[position] ** 3))
        width = widths[stop - 1]
        stop = min(stop, position + max(1, budget // width**3))
        chosen = order[position:stop]
        grid = torch.stack(
            torch.meshgrid(*[torch.arange(width, device=device)] * 3, indexing="ij"), -1
        ).view(-1, 3)
        points = low[chosen].unsqueeze(1) + grid.unsqueeze(0)  # [f, K, 3]
        inside = (points <= high[chosen].unsqueeze(1)).all(-1)
        coords = points.float() / scale - 0.5
        tri = corners[chosen]
        distance = _point_triangle_distance(
            coords, tri[:, 0:1], tri[:, 1:2], tri[:, 2:3]
        )
        hit = inside & (distance < limit)
        cells = points[hit] // factor
        occupied[(cells[:, 0] * side + cells[:, 1]) * side + cells[:, 2]] = True
        position = stop
    index = occupied.nonzero().squeeze(1)
    x = index // (side * side)
    y = index // side % side
    z = index % side
    return torch.stack([torch.zeros_like(x), x, y, z], dim=1)


# -- Stages --------------------------------------------------------------------


@torch.no_grad()
def run_stages(
    modules,
    image: Image.Image,
    generator: torch.Generator,
    *,
    resolution: int = 1024,
    steps: Sequence[int] = (50, 30, 15),
    guidance: float = 7.0,
    cond: Optional[torch.Tensor] = None,
    stop_after: Optional[str] = None,
    noise: Optional[Dict[str, torch.Tensor]] = None,
) -> Dict[str, object]:
    """Preprocessed image -> mesh, returning every stage for inspection.

    ``noise`` replaces the drawn starting noise of a stage (``"dense"``,
    ``"sparse512"``, ``"sparse1024"``), for the parity gate.
    """
    from ..trellis.pipeline import encode_image

    device, dtype = modules.device, modules.dtype
    noise = noise or {}
    out: Dict[str, object] = {}
    if cond is None:
        cond = encode_image(modules.dinov2, image, device)
    out["cond"] = cond

    def draw(name, shape):
        if name in noise:
            return noise[name].to(device=device, dtype=dtype)
        return torch.randn(shape, dtype=dtype, device=device, generator=generator)

    dense_dit, dense_vae = modules.dense_dit, modules.dense_vae
    latent = sample(
        dense_dit,
        draw("dense", (1, *dense_dit.latent_shape)),
        cond,
        steps=steps[0],
        guidance=guidance,
        shift=modules.shifts[0],
    )
    latent = 1.0 / dense_vae.latents_scale * latent + dense_vae.latents_shift
    out["dense_latent"] = latent
    voxels = dense_vae.occupied_voxels(latent, _DENSE_OCCUPANCY)[0]
    if voxels.shape[0] == 0:
        raise RuntimeError("The dense stage found no occupied voxel for this image and seed.")
    voxels = sort_block(voxels, modules.sparse_dit_512.selection_block_size)
    out["voxels512"] = voxels
    if stop_after == "dense":
        return out

    vertices, faces, stage = _sparse_stage(
        modules, cond, draw, voxels, "sparse512", steps[1], guidance, out
    )
    out["mesh512"] = (vertices, faces)
    if stop_after == "sparse512" or resolution == 512:
        out["mesh"] = (vertices, faces)
        return out

    # Upstream voxelises at 1024^3: the 1024 decoder's latent grid times its 8x upsampling.
    size = modules.sparse_vae_1024.resolution * _UDF_FACTOR
    voxels = mesh_to_voxels(normalize_vertices(vertices), faces, device, size=size)
    voxels = sort_block(voxels, modules.sparse_dit_1024.selection_block_size)
    out["voxels1024"] = voxels
    vertices, faces, _ = _sparse_stage(
        modules, cond, draw, voxels, "sparse1024", steps[2], guidance, out
    )
    out["mesh"] = (vertices, faces)
    return out


def _sparse_stage(modules, cond, draw, voxels, name, steps, guidance, out):
    high = name == "sparse1024"
    dit = modules.sparse_dit_1024 if high else modules.sparse_dit_512
    vae = modules.sparse_vae_1024 if high else modules.sparse_vae_512
    shift = modules.shifts[2 if high else 1]
    coords = voxels.int()
    latent = sample(
        dit,
        draw(name, (coords.shape[0], dit.out_channels)),
        cond,
        steps=steps,
        guidance=guidance,
        shift=shift,
        coords=coords,
    )
    latent = 1.0 / vae.latents_scale * latent + vae.latents_shift
    out[f"{name}_latent"] = latent
    sparse = SparseTensor(latent, coords)
    if high:
        sdf = vae.decoder(sparse, factor=1.0)
        out[f"{name}_sdf"] = sdf
        volume = modules.refiner_1024.refine(sdf)
        level = _SDF_LEVEL
    else:
        sdf, feat = vae.decoder(sparse, factor=1.0, return_feat=True)
        out[f"{name}_sdf"] = sdf
        volume = modules.refiner.refine(sdf, feat)
        level = _SDF_LEVEL * 2.0
    grid = volume[0, 0].cpu().numpy()
    del volume
    vertices, faces = marching_cubes(grid, level)
    return vertices, faces, sdf
