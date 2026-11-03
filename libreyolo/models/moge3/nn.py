"""Native MoGe-3 network.

Ported from Microsoft MoGe at commit 74fbce054ebed49800de42d0ad0e83495065719a
(MIT): ``moge/model/v3.py`` (forward, voxelization, iterative log-depth
refinement) on top of the MoGe-2 encoder, neck, and heads. The DINOv2 encoder
wrapper and convolutional decoder are shared with ``libreyolo.models.moge2``;
the sparse refiner lives in :mod:`.sparse`.
"""

from __future__ import annotations

import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..moge2.nn import PATCH_SIZE, ConvStack, DINOv2Encoder, normalized_view_plane_uv
from .sparse import Sparse3DUNet, SparseLevel

_RESAMPLERS = ["conv_transpose", "conv_transpose", "conv_transpose", "bilinear"]

SIZE_CONFIGS = {
    "l": {
        "backbone": "dinov2_vitl14",
        "intermediate_layers": [5, 11, 17, 23],
        "dim": 1024,
        "decoder_dims": [1024, 256, 128, 64, 32],
    },
    "g": {
        "backbone": "dinov2_vitg14",
        "intermediate_layers": [9, 19, 29, 39],
        "dim": 1536,
        "decoder_dims": [1024, 512, 256, 128, 64],
    },
}

REFINER_CONFIG = {
    "in_channels": 3,
    "out_channels": 1,
    "model_channels": [32, 64, 128, 256, 512],
    "encoder_blocks_per_level": 1,
    "decoder_blocks_per_level": 1,
    "bottleneck_blocks": 1,
    "downsample_factors": [2, 2, 2, 2],
    "encoder_downsample": 16,
}
REFINER_DEPTH_RESOLUTION = 256
DEFAULT_REFINE_STEPS = 3

# Top-level submodules each task graph builds.
TASK_MODULES = {
    "depth": ("encoder", "neck", "points_head", "mask_head", "scale_head", "refiner"),
    "normal": ("encoder", "neck", "normal_head"),
}
# The normal graph has no refiner, and its tensors match MoGe-2 ViT-L by name
# and shape, so it carries this persistent buffer to stay recognizable as
# MoGe-3 after save().
FAMILY_TAG_KEY = "moge3_tag"


class MLP(nn.Sequential):
    """Linear and ReLU stack; the last layer has no activation."""

    def __init__(self, dims: list[int]):
        super().__init__(
            *itertools.chain(
                *[
                    (nn.Linear(dim_in, dim_out), nn.ReLU(inplace=True))
                    for dim_in, dim_out in zip(dims[:-2], dims[1:-1])
                ]
            ),
            nn.Linear(dims[-2], dims[-1]),
        )


def token_grid(num_tokens: int, aspect_ratio: float) -> tuple[int, int]:
    """Upstream's base token grid for a token budget and width/height ratio."""
    return round((num_tokens / aspect_ratio) ** 0.5), round(
        (num_tokens * aspect_ratio) ** 0.5
    )


def _resize(x: torch.Tensor, size: tuple[int, int], channel_last: bool = False):
    return F.interpolate(
        x.movedim(-1, -3) if channel_last else x,
        size,
        mode="bilinear",
        align_corners=False,
        antialias=False,
    ).movedim(-3, -1 if channel_last else -3)


class MoGe3Net(nn.Module):
    """MoGe-3 ViT-L/ViT-g graph for one task.

    ``task="depth"`` builds the point, mask, and metric-scale heads plus the
    sparse refiner and returns affine point maps; ``task="normal"`` builds the
    normal head only and returns ``(B, 3, H, W)`` unit normals.
    """

    def __init__(self, size: str = "l", task: str = "depth"):
        super().__init__()
        if size not in SIZE_CONFIGS:
            raise ValueError(
                f"MoGe-3 size must be one of {sorted(SIZE_CONFIGS)}, got {size!r}."
            )
        if task not in TASK_MODULES:
            raise ValueError(
                f"MoGe-3 task must be one of {sorted(TASK_MODULES)}, got {task!r}."
            )
        config = SIZE_CONFIGS[size]
        dim = config["dim"]
        decoder_dims = config["decoder_dims"]
        self.task = task
        self.encoder = DINOv2Encoder(
            backbone=config["backbone"],
            intermediate_layers=config["intermediate_layers"],
            dim_out=dim,
        )
        self.neck = ConvStack(
            dim_in=[dim + 2, 2, 2, 2, 2],
            dim_out=None,
            dim_res_blocks=decoder_dims,
            num_res_blocks=[0, 2, 2, 2, 0],
            res_block_in_norm="none",
            res_block_hidden_norm="none",
            resamplers=_RESAMPLERS,
        )

        def head(channels: int) -> ConvStack:
            return ConvStack(
                dim_in=decoder_dims,
                dim_out=[None, None, None, None, channels],
                dim_res_blocks=decoder_dims,
                num_res_blocks=[0, 1, 1, 1, 0],
                res_block_in_norm="none",
                res_block_hidden_norm="none",
                resamplers=_RESAMPLERS,
            )

        if task == "normal":
            self.normal_head = head(3)
            self.register_buffer(FAMILY_TAG_KEY, torch.tensor(3))
        else:
            self.points_head = head(3)
            self.mask_head = head(1)
            self.scale_head = MLP([dim, dim, dim, 1])
            self.refiner = Sparse3DUNet(encoder_channels=dim + 2, **REFINER_CONFIG)
        self.refiner_depth_resolution = REFINER_DEPTH_RESOLUTION
        self.onnx_compatible_mode = False

    def set_onnx_compatible_mode(self, enabled: bool) -> None:
        self.onnx_compatible_mode = bool(enabled)
        self.encoder.onnx_compatible_mode = bool(enabled)

    def _voxelize(self, point_coord: torch.Tensor):
        """Dense ``(B, H, W, 3)`` coords at ``(x/z, y/z, log z)`` to sparse sites.

        Binning runs in FP32: the 1/256 log-depth quantization is finer than
        FP16 resolution over the usual range.
        """
        point_coord = point_coord.float()
        batch, height, width, _ = point_coord.shape
        device = point_coord.device
        logz = point_coord[..., 2]
        quantized = torch.round(logz * self.refiner_depth_resolution).long()
        offset = quantized.amin(dim=(1, 2), keepdim=True)
        z_index = quantized - offset
        z_extent = int(z_index.amax().item()) + 1
        rows = torch.arange(height, device=device).view(1, height, 1)
        cols = torch.arange(width, device=device).view(1, 1, width)
        items = torch.arange(batch, device=device).view(batch, 1, 1)
        coords = torch.stack(
            (
                items.expand(batch, height, width),
                rows.expand(batch, height, width),
                cols.expand(batch, height, width),
                z_index,
            ),
            dim=-1,
        ).reshape(-1, 4)
        feats = point_coord.reshape(-1, 3)
        level = SparseLevel(coords, (batch, height, width, z_extent))
        return feats, level, logz

    def _refine_logz(self, point_coord: torch.Tensor, encoder_feature: torch.Tensor):
        batch, height, width, _ = point_coord.shape
        feats, level, logz = self._voxelize(point_coord)
        out = self.refiner(feats, level, encoder_feature)
        return logz + out.float().squeeze(-1).reshape(batch, height, width)

    def forward(
        self,
        image: torch.Tensor,
        num_tokens: int | None = None,
        refine_steps: int = DEFAULT_REFINE_STEPS,
    ):
        # Plain ints keep the token grid arithmetic traceable for export.
        batch_size, _, img_h, img_w = (int(dim) for dim in image.shape)
        device, dtype = image.device, image.dtype
        if num_tokens is None:
            num_tokens = (img_h // PATCH_SIZE) * (img_w // PATCH_SIZE)
        aspect_ratio = img_w / img_h
        base_h, base_w = token_grid(num_tokens, aspect_ratio)

        feature, cls_token = self.encoder(image, base_h, base_w)
        features: list[torch.Tensor | None] = [feature, None, None, None, None]
        for level in range(5):
            uv = normalized_view_plane_uv(
                width=base_w * 2**level,
                height=base_h * 2**level,
                aspect_ratio=aspect_ratio,
                dtype=dtype,
                device=device,
            )
            uv = uv.permute(2, 0, 1).unsqueeze(0).expand(batch_size, -1, -1, -1)
            features[level] = (
                uv
                if features[level] is None
                else torch.cat((features[level], uv), dim=1)
            )
        neck_features = self.neck(features)

        if self.task == "normal":
            normal = _resize(self.normal_head(neck_features)[-1], (img_h, img_w))
            normal = F.normalize(normal.permute(0, 2, 3, 1), dim=-1)
            return normal.permute(0, 3, 1, 2)

        raw_coord = self.points_head(neck_features)[-1]
        mask = self.mask_head(neck_features)[-1]
        metric_scale = self.scale_head(cls_token)

        coord = raw_coord.permute(0, 2, 3, 1).float()
        for _ in range(refine_steps):
            refined_logz = self._refine_logz(coord.detach(), features[0].detach())
            coord = torch.cat((coord[..., :2], refined_logz.unsqueeze(-1)), dim=-1)

        coord = _resize(coord, (img_h, img_w), channel_last=True)
        xy, z = coord.split([2, 1], dim=-1)
        z = torch.exp(z)
        points = torch.cat((xy * z, z), dim=-1)
        return {
            "points": points,
            "mask": _resize(mask, (img_h, img_w)).squeeze(1).sigmoid(),
            "metric_scale": metric_scale.squeeze(1).exp(),
        }


__all__ = ["FAMILY_TAG_KEY", "MoGe3Net", "SIZE_CONFIGS", "TASK_MODULES", "token_grid"]
