"""Preprocessing and camera recovery for the MoGe-3 family.

The focal and shift recovery and the depth-to-point unprojection follow
Microsoft MoGe at commit 74fbce054ebed49800de42d0ad0e83495065719a (MIT,
``moge/utils/geometry_torch.py``, ``moge/utils/geometry_numpy.py``, and the
``infer`` method of ``moge/model/v3.py``) and the utils3d helpers MoGe pins
(EasternJournalist/utils3d-moge at 62f09d58509485564e24d5d9f6aac9ee9ebc0c37,
MIT): ``uv_map``, ``intrinsics_from_focal_center``, ``unproject_cv``.
"""

from __future__ import annotations

from functools import partial

import numpy as np
import torch
import torch.nn.functional as F

from ..moge2.nn import PATCH_SIZE, normalized_view_plane_uv


def num_tokens_for(input_size: int) -> int:
    """Token budget for an ``imgsz``: a square ``imgsz`` canvas of 14 px patches."""
    if input_size < PATCH_SIZE:
        raise ValueError(
            f"MoGe-3 imgsz must be at least {PATCH_SIZE}, got {input_size}."
        )
    return (int(input_size) // PATCH_SIZE) ** 2


def preprocess_numpy(img_rgb_hwc: np.ndarray, input_size: int = 840):
    """RGB HWC uint8 to a CHW float32 tensor in ``[0, 1]`` at native resolution.

    MoGe resizes inside the network: the encoder samples the image onto the
    token grid that ``input_size`` sets, and every output is interpolated back
    to the input resolution. ``input_size`` therefore selects the token budget
    (see :func:`num_tokens_for`) rather than a resize target.
    """
    del input_size
    image = np.asarray(img_rgb_hwc)
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    chw = image.astype(np.float32).transpose(2, 0, 1) / 255.0
    return np.ascontiguousarray(chw), 1.0


def _solve_optimal_focal_shift(uv: np.ndarray, xyz: np.ndarray):
    """Solve ``min |focal * xy / (z + shift) - uv|`` for shift and focal."""
    from scipy.optimize import least_squares

    uv, xy, z = uv.reshape(-1, 2), xyz[..., :2].reshape(-1, 2), xyz[..., 2].reshape(-1)

    def residual(uv, xy, z, shift):
        xy_proj = xy / (z + shift)[:, None]
        focal = (xy_proj * uv).sum() / np.square(xy_proj).sum()
        return (focal * xy_proj - uv).ravel()

    solution = least_squares(partial(residual, uv, xy, z), x0=0, ftol=1e-3, method="lm")
    shift = solution["x"].squeeze().astype(np.float32)
    xy_proj = xy / (z + shift)[:, None]
    focal = (xy_proj * uv).sum() / np.square(xy_proj).sum()
    return shift, focal


def recover_focal_shift(
    points: torch.Tensor,
    mask: torch.Tensor | None = None,
    downsample_size: tuple[int, int] = (64, 64),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Focal (relative to the half diagonal) and z shift of an affine point map."""
    shape = points.shape
    height, width = points.shape[-3], points.shape[-2]
    points = points.reshape(-1, *shape[-3:])
    mask = None if mask is None else mask.reshape(-1, *shape[-3:-1])
    uv = normalized_view_plane_uv(
        width, height, dtype=points.dtype, device=points.device
    )

    points_lr = F.interpolate(
        points.permute(0, 3, 1, 2), downsample_size, mode="nearest"
    ).permute(0, 2, 3, 1)
    uv_lr = (
        F.interpolate(
            uv.unsqueeze(0).permute(0, 3, 1, 2), downsample_size, mode="nearest"
        )
        .squeeze(0)
        .permute(1, 2, 0)
    )
    mask_lr = (
        None
        if mask is None
        else F.interpolate(
            mask.to(torch.float32).unsqueeze(1), downsample_size, mode="nearest"
        ).squeeze(1)
        > 0
    )

    uv_lr_np = uv_lr.cpu().numpy()
    points_lr_np = points_lr.detach().cpu().numpy()
    mask_lr_np = None if mask is None else mask_lr.cpu().numpy()
    shifts, focals = [], []
    for index in range(points.shape[0]):
        points_i = (
            points_lr_np[index]
            if mask is None
            else points_lr_np[index][mask_lr_np[index]]
        )
        uv_i = uv_lr_np if mask is None else uv_lr_np[mask_lr_np[index]]
        if uv_i.shape[0] < 2:
            focals.append(1)
            shifts.append(0)
            continue
        shift_i, focal_i = _solve_optimal_focal_shift(uv_i, points_i)
        focals.append(float(focal_i))
        shifts.append(float(shift_i))
    shift = torch.tensor(shifts, device=points.device, dtype=points.dtype).reshape(
        shape[:-3]
    )
    focal = torch.tensor(focals, device=points.device, dtype=points.dtype).reshape(
        shape[:-3]
    )
    return focal, shift


def _uv_map(height: int, width: int, *, dtype, device) -> torch.Tensor:
    """Pixel-center image coordinates in ``[0, 1]``, ``(H, W, 2)``."""
    u = torch.linspace(0.5 / width, 1 - 0.5 / width, width, dtype=dtype, device=device)
    v = torch.linspace(
        0.5 / height, 1 - 0.5 / height, height, dtype=dtype, device=device
    )
    return torch.stack(torch.meshgrid(u, v, indexing="xy"), dim=-1)


def _inverse(matrix: torch.Tensor) -> torch.Tensor:
    # torch.linalg.inv_ex is not implemented on every accelerator backend; the
    # 4x4 camera matrices are tiny, so invert on the CPU for those devices.
    if matrix.device.type == "mps":
        return torch.linalg.inv_ex(matrix.cpu())[0].to(matrix.device)
    return torch.linalg.inv_ex(matrix)[0]


def depth_to_points(depth: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
    """Unproject ``(B, H, W)`` depth with normalized ``(B, 3, 3)`` intrinsics."""
    height, width = depth.shape[-2:]
    uv = _uv_map(height, width, dtype=depth.dtype, device=depth.device)
    intrinsics = intrinsics[..., None, :, :]
    transform = torch.cat(
        (
            torch.cat(
                (
                    intrinsics,
                    torch.zeros(
                        (*intrinsics.shape[:-2], 3, 1),
                        dtype=intrinsics.dtype,
                        device=intrinsics.device,
                    ),
                ),
                dim=-1,
            ),
            torch.tensor(
                [[0, 0, 0, 1]], dtype=intrinsics.dtype, device=intrinsics.device
            ).expand(*intrinsics.shape[:-2], 1, 4),
        ),
        dim=-2,
    )
    points = (
        torch.cat(
            (uv, torch.ones((*uv.shape[:-1], 1), dtype=uv.dtype, device=uv.device)),
            dim=-1,
        )
        * depth[..., None]
    )
    points = torch.cat(
        (points, torch.ones((*points.shape[:-1], 1), dtype=uv.dtype, device=uv.device)),
        dim=-1,
    )
    return (points @ _inverse(transform).mT)[..., :3]


def recover_geometry(
    affine_points: torch.Tensor,
    mask: torch.Tensor,
    metric_scale: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Turn the network's affine point map into metric camera geometry.

    Mirrors upstream ``infer`` with ``force_projection=True`` and
    ``apply_mask=True``: recover focal and shift on the masked point map,
    recompute points from depth and intrinsics, apply the metric scale, and set
    invalid pixels to ``inf``. Intrinsics are normalized: ``fx``/``cx`` are in
    units of image width, ``fy``/``cy`` in units of image height.
    """
    affine_points = affine_points.float()
    mask = mask.float()
    metric_scale = metric_scale.float()
    height, width = affine_points.shape[-3:-1]
    aspect_ratio = width / height
    valid = mask > 0.5

    focal, shift = recover_focal_shift(affine_points, valid)
    diagonal = (1 + aspect_ratio**2) ** 0.5
    fx = focal / 2 * diagonal / aspect_ratio
    fy = focal / 2 * diagonal
    zeros, ones = torch.zeros_like(fx), torch.ones_like(fx)
    center = torch.full_like(fx, 0.5)
    intrinsics = torch.stack(
        (fx, zeros, center, zeros, fy, center, zeros, zeros, ones), dim=-1
    ).unflatten(-1, (3, 3))

    points = affine_points.clone()
    points[..., 2] += shift[..., None, None]
    depth = points[..., 2].clone()
    points = depth_to_points(depth, intrinsics)
    points *= metric_scale[:, None, None, None]
    depth *= metric_scale[:, None, None]

    valid = valid & (depth > 0)
    points = torch.where(valid[..., None], points, torch.inf)
    depth = torch.where(valid, depth, torch.inf)
    return {"points": points, "depth": depth, "intrinsics": intrinsics, "mask": valid}


def pixel_intrinsics(intrinsics: torch.Tensor, width: int, height: int) -> torch.Tensor:
    """Normalized intrinsics to OpenCV pixel units on a ``width x height`` canvas.

    Pixel centers sit at integer coordinates, so the normalized center 0.5
    maps to ``(width - 1) / 2``.
    """
    scale = intrinsics.new_tensor([[width, 0, 0], [0, height, 0], [0, 0, 1]])
    shift = intrinsics.new_tensor([[1, 0, -0.5], [0, 1, -0.5], [0, 0, 1]])
    return shift @ scale @ intrinsics


__all__ = [
    "depth_to_points",
    "num_tokens_for",
    "pixel_intrinsics",
    "preprocess_numpy",
    "recover_focal_shift",
    "recover_geometry",
]
