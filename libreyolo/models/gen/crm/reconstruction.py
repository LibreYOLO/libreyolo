# Adapted from CRM (https://github.com/thu-ml/CRM, commit
# 4964e36a593070a3045eb0f300935a771ff5c172, MIT, Copyright (c) 2024 TSAIL
# group): model/crm/model.py, model/archs/{unet,mlp_head}.py,
# model/archs/decoders/shape_texture_net.py, util/utils.py (get_tri),
# util/renderer.py and inference.py. See NOTICE for what is written for
# LibreYOLO instead (FlexiCubes grid wrapper, mesh cleanup, UV rasterizer).
"""CRM's convolutional reconstruction model: six views and CCMs in, a textured mesh out.

The six generated views and their canonical coordinate maps are rearranged
into a triplane-shaped image (the views are orthographic, so each pair of
opposite views lines up with one plane), a diffusers ``UNet2DModel`` maps it
to a 32-channel triplane, small MLPs read the signed distance, a vertex
offset and FlexiCubes weights at the corners of an 80^3 grid, and FlexiCubes
extracts the surface. Colour comes from a third MLP at the surface points.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .flexicubes import FlexiCubes

GRID_RES = 80
PLANE_RES = 256
SPHERE_RADIUS = 0.5
# Upstream's renderer scales the offsets by scale / 0.95 with scale 0.95.
_INPUT_SCALE = 0.95
# The triplane is re-noised to this DDPM step before the UNet, with SD 2.1's
# scheduler (scaled-linear betas, 0.00085 to 0.012).
NOISE_STEP = 20


class _UNetPP(nn.Module):
    def __init__(self, in_channels: int = 32, block_out_channels=(64, 128, 128, 256, 256, 512, 512)):
        super().__init__()
        from diffusers import UNet2DModel

        self.in_channels = in_channels
        self.unet = UNet2DModel(
            sample_size=[256, 256 * 3],
            in_channels=in_channels,
            out_channels=32,
            layers_per_block=2,
            block_out_channels=tuple(block_out_channels),
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",
                "AttnDownBlock2D",
                "AttnDownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",
                "AttnUpBlock2D",
                "AttnUpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
        )
        self.learned_plane = nn.Parameter(torch.zeros([1, in_channels - 12, 256, 256 * 3]))

    def forward(self, x, t):
        plane = self.learned_plane.expand(x.shape[0], -1, -1, -1).to(x.device)
        return self.unet(torch.cat([x, plane], dim=1), t).sample


class _MLP(nn.Module):
    def __init__(self, input_dim: int, hidden: int, out: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, out)

    def forward(self, x):
        return self.fc3(F.relu(self.fc2(F.relu(self.fc1(x)))))


def sample_triplane(features: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """Sum of the xy, yz and zx plane features at ``points`` [B, N, 3] -> [B, N, C]."""
    total = 0
    for i, axes in enumerate(([0, 1], [1, 2], [2, 0])):
        plane = features[..., i * PLANE_RES : (i + 1) * PLANE_RES]
        grid = points[:, :, axes][:, :, None].float()
        total = total + F.grid_sample(
            plane, grid, padding_mode="border", align_corners=True, mode="bilinear"
        ).squeeze(-1)
    return total.permute(0, 2, 1)


class CRMNet(nn.Module):
    """``CRM.pth``: the triplane UNet and the SDF, colour and FlexiCubes-weight MLPs."""

    def __init__(self, **unet):
        super().__init__()
        self.unet2 = _UNetPP(32, **unet)
        self.weightMlp = nn.Sequential(nn.Linear(32 * 8, 512), nn.SiLU(), nn.Linear(512, 21))
        self.sdfMlp = _MLP(32, 512, 4)
        self.rgbMlp = _MLP(32, 512, 3)


# -- Views to triplane ----------------------------------------------------------


def _views(strip: torch.Tensor) -> torch.Tensor:
    """[C, H, 6W] strip of views 1..5, 0 -> [6, C, H, W] in upstream's order (view 0 first)."""
    width = strip.shape[-1] // 6
    order = [5, 0, 1, 2, 3, 4]
    return torch.stack([strip[..., width * i : width * (i + 1)] for i in order])


def _to_triplane(v: torch.Tensor, fix: bool) -> torch.Tensor:
    """Upstream's get_tri(dim=0, blender=True, scale=1): six views -> [2C, H, 3W]."""
    t0 = torch.rot90(v[2], k=2, dims=[1, 2])
    t1 = torch.rot90(v[4], k=0, dims=[1, 2]).flip(2).flip(1)
    t2 = torch.rot90(torch.rot90(v[0], k=3, dims=[1, 2]).flip(2), k=2, dims=[1, 2])
    t3 = torch.rot90(torch.rot90(v[5], k=2, dims=[1, 2]).flip(2), k=2, dims=[1, 2])
    t4 = torch.rot90(v[1], k=2, dims=[1, 2]).flip(2)
    t5 = torch.rot90(v[3], k=1, dims=[1, 2]).flip(1).flip(2)
    if fix:
        # Each CCM keeps only the coordinate along its view axis.
        t0, t3 = t0 * torch.tensor([1, 0, 0]).view(3, 1, 1), t3 * torch.tensor([1, 0, 0]).view(3, 1, 1)
        t1, t4 = t1 * torch.tensor([0, 0, 1]).view(3, 1, 1), t4 * torch.tensor([0, 0, 1]).view(3, 1, 1)
        t2, t5 = t2 * torch.tensor([0, 1, 0]).view(3, 1, 1), t5 * torch.tensor([0, 1, 0]).view(3, 1, 1)
    return torch.cat((torch.cat((t0, t1, t2), dim=2), torch.cat((t3, t4, t5), dim=2)), dim=0)


def triplane_input(views: np.ndarray, ccms: np.ndarray) -> torch.Tensor:
    """uint8 [6, 256, 256, 3] views and CCMs -> the UNet's [1, 12, 256, 768] input."""
    color = torch.from_numpy(np.concatenate(list(views), 1)) / 255
    # Upstream reads the CCM strip with its channels reversed.
    xyz = torch.from_numpy(np.concatenate(list(ccms), 1)[:, :, (2, 1, 0)]) / 255
    color = _to_triplane(_views(color.permute(2, 0, 1)), fix=False)
    xyz = _to_triplane(_views(xyz.permute(2, 0, 1)), fix=True)
    return torch.cat([color, xyz], dim=0).unsqueeze(0)


def add_noise(x: torch.Tensor, noise: torch.Tensor, step: int) -> torch.Tensor:
    """DDPM forward noising at ``step`` with SD 2.1's scaled-linear schedule, in fp32."""
    betas = torch.linspace(0.00085**0.5, 0.012**0.5, 1000, dtype=torch.float32) ** 2
    alpha = torch.cumprod(1.0 - betas, dim=0)[step]
    return alpha**0.5 * x + (1 - alpha) ** 0.5 * noise


# -- Geometry -------------------------------------------------------------------


class Grid:
    """The FlexiCubes voxel grid CRM decodes on: 80^3 cubes over [-0.5, 0.5]^3."""

    def __init__(self, device, res: int = GRID_RES):
        self.res = res
        self.fc = FlexiCubes(device=device)
        self.verts, self.indices = self.fc.construct_voxel_grid(res)


@torch.no_grad()
def extract_mesh(net: CRMNet, features: torch.Tensor, grid: Grid) -> Tuple[torch.Tensor, torch.Tensor]:
    """32-channel triplane [1, 32, 256, 768] -> vertices [V, 3], faces [F, 3] in CRM's frame."""
    verts = grid.verts.unsqueeze(0)
    point_features = sample_triplane(features, verts)
    out = net.sdfMlp(point_features)
    corner = torch.index_select(point_features, 1, grid.indices.reshape(-1))
    corner = corner.reshape(1, grid.indices.shape[0], -1)
    weight = net.weightMlp(corner) * 0.1
    sdf, deformation = out[..., 0], out[..., 1:]
    # A sphere prior: positive inside a ball of radius 0.5.
    sdf = sdf + SPHERE_RADIUS - torch.sqrt((verts**2).sum(-1))
    deformation = torch.tanh(deformation) / grid.res * _INPUT_SCALE / 0.95 * 0.5
    moved = (verts + deformation)[0]
    w = weight[0]
    vertices, faces = grid.fc(
        moved, sdf[0], grid.indices, grid.res, w[:, :12], w[:, 12:20], w[:, 20]
    )
    return vertices, faces


@torch.no_grad()
def vertex_colors(net: CRMNet, features: torch.Tensor, points: torch.Tensor, chunk: int = 262144) -> torch.Tensor:
    """RGB in [-1, 1] (unclipped) at ``points`` [N, 3]."""
    out = []
    for start in range(0, points.shape[0], chunk):
        part = points[start : start + chunk][None]
        out.append(net.rgbMlp(sample_triplane(features, part))[0])
    return torch.cat(out) if out else points.new_zeros((0, 3))


def clean_mesh(vertices: np.ndarray, faces: np.ndarray, merge_fraction: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
    """Upstream's cleanup, written for LibreYOLO in NumPy.

    Upstream calls a mesh-processing library to drop unreferenced vertices,
    merge vertices closer than 1% of the bounding-box diagonal, and drop
    duplicate and zero-area faces. This does the same operations: a greedy
    merge in vertex order (each vertex absorbs the not-yet-merged vertices
    within the threshold and keeps its position), then the face cleanup.
    """
    from scipy.spatial import cKDTree

    faces = faces.astype(np.int64)
    used = np.unique(faces)
    remap = np.full(len(vertices), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    vertices, faces = vertices[used], remap[faces]
    if len(vertices) == 0:
        return vertices.astype(np.float32), faces.reshape(0, 3)
    diagonal = float(np.linalg.norm(vertices.max(0) - vertices.min(0)))
    threshold = diagonal * merge_fraction
    if threshold > 0:
        tree = cKDTree(vertices)
        neighbours = tree.query_ball_point(vertices, threshold)
        target = np.full(len(vertices), -1, dtype=np.int64)
        for i, near in enumerate(neighbours):
            if target[i] != -1:
                continue
            near = np.asarray(near, dtype=np.int64)
            near = near[target[near] == -1]
            target[near] = i
            target[i] = i
        faces = target[faces]
    faces = faces[(faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2])]
    _, first = np.unique(np.sort(faces, axis=1), axis=0, return_index=True)
    faces = faces[np.sort(first)]
    a, b, c = (vertices[faces[:, i]] for i in range(3))
    faces = faces[np.linalg.norm(np.cross(b - a, c - a), axis=1) > 0]
    used = np.unique(faces)
    remap = np.full(len(vertices), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    return vertices[used].astype(np.float32), remap[faces]


# -- Texture ------------------------------------------------------------------------


def rasterize_uv(uvs: torch.Tensor, faces: torch.Tensor, resolution: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Rasterize UV triangles into a ``resolution``^2 texture, written for LibreYOLO.

    Returns ``(face_id, bary)``: [R, R] long (-1 where no triangle covers the
    texel centre) and [R, R, 3] barycentric weights. Texel (row, col) samples
    uv ((col + 0.5) / R, (row + 0.5) / R).
    """
    device = uvs.device
    res = resolution
    tri = uvs[faces] * res - 0.5  # texel-centre coordinates
    face_id = torch.full((res * res,), -1, dtype=torch.long, device=device)
    bary = torch.zeros((res * res, 3), device=device)
    lo = torch.clamp(torch.floor(tri.min(1).values), 0, res - 1).long()
    hi = torch.clamp(torch.ceil(tri.max(1).values), 0, res - 1).long()
    size = hi - lo + 1
    span = torch.maximum(size[:, 0], size[:, 1])
    order = torch.argsort(span)
    span_sorted = span[order]
    start = 0
    total = faces.shape[0]
    while start < total:
        s = int(span_sorted[start])
        end = int(torch.searchsorted(span_sorted, s, right=True))
        end = max(end, start + 1)
        # Cap the work per chunk.
        end = min(end, start + max(1, (1 << 22) // (s * s)))
        idx = order[start:end]
        offsets = torch.stack(
            torch.meshgrid(torch.arange(s, device=device), torch.arange(s, device=device), indexing="xy"),
            -1,
        ).reshape(-1, 2)
        px = lo[idx][:, None, :] + offsets[None]  # [n, s*s, 2] (x, y)
        p = px.float()
        a, b, c = tri[idx, 0][:, None], tri[idx, 1][:, None], tri[idx, 2][:, None]
        area = (b[..., 0] - a[..., 0]) * (c[..., 1] - a[..., 1]) - (b[..., 1] - a[..., 1]) * (c[..., 0] - a[..., 0])
        w0 = (b[..., 0] - p[..., 0]) * (c[..., 1] - p[..., 1]) - (b[..., 1] - p[..., 1]) * (c[..., 0] - p[..., 0])
        w1 = (c[..., 0] - p[..., 0]) * (a[..., 1] - p[..., 1]) - (c[..., 1] - p[..., 1]) * (a[..., 0] - p[..., 0])
        w2 = area - w0 - w1
        safe = torch.where(area.abs() > 1e-12, area, torch.ones_like(area))
        weights = torch.stack([w0, w1, w2], -1) / safe[..., None]
        inside = (
            (weights >= 0).all(-1)
            & (area.abs() > 1e-12)
            & (px[..., 0] < res)
            & (px[..., 1] < res)
        )
        flat = (px[..., 1] * res + px[..., 0])[inside]
        face_id[flat] = idx[:, None].expand_as(inside)[inside]
        bary[flat] = weights[inside]
        start = end
    return face_id.view(res, res), bary.view(res, res, 3)


@torch.no_grad()
def bake_texture(
    net: CRMNet,
    features: torch.Tensor,
    vertices: np.ndarray,
    faces: np.ndarray,
    resolution: int = 1024,
) -> Dict[str, np.ndarray]:
    """UV-unwrap with xatlas and bake the colour field: upstream's export_mesh_wt_uv.

    Returns ``uv`` [T, 2], ``uv_faces`` [F, 3] (indices into ``uv``) and
    ``texture`` uint8 [R, R, 3], row 0 at v = 0.
    """
    try:
        import xatlas
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "texture=True needs the MIT-licensed xatlas package: pip install xatlas"
        ) from exc
    import cv2

    device = features.device
    _vmapping, indices, uvs = xatlas.parametrize(vertices, faces)
    indices = indices.astype(np.int64)
    uv_t = torch.tensor(uvs, dtype=torch.float32, device=device)
    face_id, bary = rasterize_uv(uv_t, torch.tensor(indices, device=device), resolution)
    mask = face_id >= 0
    v_t = torch.tensor(vertices, dtype=torch.float32, device=device)
    f_t = torch.tensor(faces, dtype=torch.long, device=device)
    corners = v_t[f_t[face_id[mask]]]  # [M, 3, 3]
    points = (corners * bary[mask][..., None]).sum(1)
    colors = vertex_colors(net, features, points)
    colors = ((colors + 1) * (255 / 2)).clip(0, 255)
    tex = torch.ones((resolution, resolution, 3), device=device)
    tex[mask] = colors
    img = tex.cpu().numpy().astype(np.float32)
    background = (np.sum(img.astype(float), axis=-1, keepdims=True) <= 3.0).astype(float)
    dilated = cv2.dilate(img, np.ones((3, 3), "uint8"), iterations=1)
    img = img * (1 - background) + dilated * background
    return {
        "uv": uvs.astype(np.float32),
        "uv_faces": indices,
        "texture": img.clip(0, 255).astype(np.uint8),
    }


def to_object_frame(vertices: np.ndarray, faces: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """CRM's frame -> LibreGen's: z up, object in [-0.5, 0.5]^3, faces wound outward.

    CRM's SDF is positive inside, so FlexiCubes winds faces inward; upstream
    reverses them on export, and so does this.
    """
    return vertices.astype(np.float32), faces[:, [2, 1, 0]].copy()


def stage_noise(seed: int, device: Optional[torch.device] = None) -> Dict[str, torch.Tensor]:
    """Every random draw of one object, from a CPU generator: same numbers on every device."""
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    draws = {
        "stage1_posterior": torch.randn((1, 4, 32, 32), generator=generator),
        "stage1_latents": torch.randn((7, 4, 32, 32), generator=generator),
        "stage2_posterior": torch.randn((6, 4, 32, 32), generator=generator),
        "stage2_latents": torch.randn((6, 4, 32, 32), generator=generator),
        "triplane": torch.randn((1, 12, 256, 768), generator=generator),
    }
    return draws
