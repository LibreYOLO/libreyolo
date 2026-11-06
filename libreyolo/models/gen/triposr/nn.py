# Adapted from TripoSR (https://github.com/VAST-AI-Research/TripoSR, commit
# 107cefdc244c39106fa830359024f6a2f1c78871), MIT, Copyright (c) 2024 Tripo AI &
# Stability AI. Its transformer files (tsr/models/transformer/) carry
# HuggingFace diffusers code, Apache-2.0, Copyright 2023 The HuggingFace Team.
# The DINO image encoder follows the ViT of HuggingFace transformers 4.35.0
# (Apache-2.0, Copyright The HuggingFace Team), the version TripoSR pins.
"""TripoSR's networks in plain PyTorch, with upstream's parameter names.

An image goes through a DINO ViT-B/16; a 16-layer transformer turns learned
triplane tokens into a 3 x 40 x 64 x 64 triplane while cross-attending to
the image tokens; a small MLP reads the triplane at any 3D point and returns
a density and a colour. The mesh is the density's isosurface.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch import nn

# Upstream's config.yaml (stabilityai/TripoSR), the only published variant.
CONFIG = {
    "cond_image_size": 512,
    "plane_size": 32,
    "num_channels": 1024,
    "num_attention_heads": 16,
    "attention_head_dim": 64,
    "num_layers": 16,
    "cross_attention_dim": 768,
    "triplane_channels": 40,
    "mlp_neurons": 64,
    "mlp_hidden_layers": 9,
    "radius": 0.87,
    "density_bias": -1.0,
}
# facebook/dino-vitb16's config.json, which upstream builds its encoder from.
DINO_CONFIG = {
    "hidden_size": 768,
    "num_hidden_layers": 12,
    "num_attention_heads": 12,
    "intermediate_size": 3072,
    "patch_size": 16,
    "image_size": 224,
    "layer_norm_eps": 1e-12,
}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# -- DINO ViT-B/16 (upstream: transformers ViTModel) ---------------------------


class _PatchEmbeddings(nn.Module):
    def __init__(self, hidden: int, patch: int):
        super().__init__()
        self.projection = nn.Conv2d(3, hidden, kernel_size=patch, stride=patch)

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        return self.projection(pixels).flatten(2).transpose(1, 2)


class _ViTEmbeddings(nn.Module):
    def __init__(self, cfg: Dict[str, int]):
        super().__init__()
        hidden, patch = cfg["hidden_size"], cfg["patch_size"]
        positions = (cfg["image_size"] // patch) ** 2
        self.patch_size = patch
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden))
        self.position_embeddings = nn.Parameter(torch.zeros(1, positions + 1, hidden))
        self.patch_embeddings = _PatchEmbeddings(hidden, patch)

    def _positions(self, num_patches: int, height: int, width: int) -> torch.Tensor:
        """Interpolate the 14 x 14 grid as transformers 4.35 does (DINO's +0.1 scale factor)."""
        num_positions = self.position_embeddings.shape[1] - 1
        if num_patches == num_positions and height == width:
            return self.position_embeddings
        cls_pos = self.position_embeddings[:, 0]
        patch_pos = self.position_embeddings[:, 1:]
        dim = patch_pos.shape[-1]
        side = int(math.sqrt(num_positions))
        h0 = height // self.patch_size + 0.1
        w0 = width // self.patch_size + 0.1
        patch_pos = patch_pos.reshape(1, side, side, dim).permute(0, 3, 1, 2)
        patch_pos = F.interpolate(
            patch_pos,
            scale_factor=(h0 / math.sqrt(num_positions), w0 / math.sqrt(num_positions)),
            mode="bicubic",
            align_corners=False,
        )
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, -1, dim)
        return torch.cat((cls_pos.unsqueeze(0), patch_pos), dim=1)

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        height, width = pixels.shape[-2:]
        tokens = self.patch_embeddings(pixels)
        cls = self.cls_token.expand(tokens.shape[0], -1, -1)
        tokens = torch.cat((cls, tokens), dim=1)
        return tokens + self._positions(tokens.shape[1] - 1, height, width)


class _ViTSelfAttention(nn.Module):
    def __init__(self, hidden: int, heads: int):
        super().__init__()
        self.heads = heads
        self.query = nn.Linear(hidden, hidden)
        self.key = nn.Linear(hidden, hidden)
        self.value = nn.Linear(hidden, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, hidden = x.shape
        shape = (batch, length, self.heads, hidden // self.heads)
        q, k, v = (
            proj(x).view(shape).transpose(1, 2)
            for proj in (self.query, self.key, self.value)
        )
        out = F.scaled_dot_product_attention(q, k, v)
        return out.transpose(1, 2).reshape(batch, length, hidden)


class _ViTDense(nn.Module):
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.dense = nn.Linear(dim_in, dim_out)


class _ViTAttention(nn.Module):
    def __init__(self, hidden: int, heads: int):
        super().__init__()
        self.attention = _ViTSelfAttention(hidden, heads)
        self.output = _ViTDense(hidden, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output.dense(self.attention(x))


class _ViTLayer(nn.Module):
    def __init__(self, cfg: Dict[str, int]):
        super().__init__()
        hidden, eps = cfg["hidden_size"], cfg["layer_norm_eps"]
        self.attention = _ViTAttention(hidden, cfg["num_attention_heads"])
        self.intermediate = _ViTDense(hidden, cfg["intermediate_size"])
        self.output = _ViTDense(cfg["intermediate_size"], hidden)
        self.layernorm_before = nn.LayerNorm(hidden, eps=eps)
        self.layernorm_after = nn.LayerNorm(hidden, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.layernorm_before(x))
        hidden = F.gelu(self.intermediate.dense(self.layernorm_after(x)))
        return x + self.output.dense(hidden)


class _ViTEncoder(nn.Module):
    def __init__(self, cfg: Dict[str, int]):
        super().__init__()
        self.layer = nn.ModuleList(
            _ViTLayer(cfg) for _ in range(cfg["num_hidden_layers"])
        )


class _ViTPooler(nn.Module):
    """Loaded for strict key matching; upstream discards its output."""

    def __init__(self, hidden: int):
        super().__init__()
        self.dense = nn.Linear(hidden, hidden)


class DINOViT(nn.Module):
    """ViT-B/16 returning the last hidden state (upstream's ``ViTModel(...).last_hidden_state``)."""

    def __init__(self, cfg: Dict[str, int] = DINO_CONFIG):
        super().__init__()
        self.embeddings = _ViTEmbeddings(cfg)
        self.encoder = _ViTEncoder(cfg)
        self.layernorm = nn.LayerNorm(cfg["hidden_size"], eps=cfg["layer_norm_eps"])
        self.pooler = _ViTPooler(cfg["hidden_size"])

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        x = self.embeddings(pixels)
        for layer in self.encoder.layer:
            x = layer(x)
        return self.layernorm(x)


class ImageTokenizer(nn.Module):
    """[B, 3, H, W] RGB in [0, 1] -> DINO tokens [B, 1 + (H/16)(W/16), 768]."""

    def __init__(self, cfg: Dict[str, int] = DINO_CONFIG):
        super().__init__()
        self.model = DINOViT(cfg)
        self.register_buffer(
            "image_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "image_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.model((images - self.image_mean) / self.image_std)


# -- Triplane transformer (upstream: Transformer1D, diffusers-derived) ---------


class _Attention(nn.Module):
    def __init__(
        self, dim: int, heads: int, head_dim: int, context_dim: int | None = None
    ):
        super().__init__()
        inner = heads * head_dim
        context_dim = dim if context_dim is None else context_dim
        self.heads = heads
        self.to_q = nn.Linear(dim, inner, bias=False)
        self.to_k = nn.Linear(context_dim, inner, bias=False)
        self.to_v = nn.Linear(context_dim, inner, bias=False)
        self.to_out = nn.ModuleList([nn.Linear(inner, dim), nn.Dropout(0.0)])

    def forward(
        self, x: torch.Tensor, context: torch.Tensor | None = None
    ) -> torch.Tensor:
        context = x if context is None else context
        batch = x.shape[0]
        q, k, v = self.to_q(x), self.to_k(context), self.to_v(context)
        head_dim = k.shape[-1] // self.heads
        q, k, v = (
            t.view(batch, -1, self.heads, head_dim).transpose(1, 2) for t in (q, k, v)
        )
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(batch, -1, self.heads * head_dim)
        return self.to_out[0](out)


class _GEGLU(nn.Module):
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden, gate = self.proj(x).chunk(2, dim=-1)
        return hidden * F.gelu(gate)


class _FeedForward(nn.Module):
    def __init__(self, dim: int, mult: int = 4):
        super().__init__()
        self.net = nn.ModuleList(
            [_GEGLU(dim, dim * mult), nn.Dropout(0.0), nn.Linear(dim * mult, dim)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for module in self.net:
            x = module(x)
        return x


class _TransformerBlock(nn.Module):
    """Self-attention, cross-attention to the image tokens, GEGLU feed-forward; pre-norm."""

    def __init__(self, dim: int, heads: int, head_dim: int, context_dim: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn1 = _Attention(dim, heads, head_dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn2 = _Attention(dim, heads, head_dim, context_dim)
        self.norm3 = nn.LayerNorm(dim)
        self.ff = _FeedForward(dim)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        x = x + self.attn1(self.norm1(x))
        x = x + self.attn2(self.norm2(x), context)
        return x + self.ff(self.norm3(x))


class Transformer1D(nn.Module):
    """[B, C, N] tokens cross-attending to [B, M, D] context -> [B, C, N]."""

    def __init__(
        self,
        in_channels: int,
        num_attention_heads: int,
        attention_head_dim: int,
        num_layers: int,
        cross_attention_dim: int,
        norm_num_groups: int = 32,
    ):
        super().__init__()
        inner = num_attention_heads * attention_head_dim
        self.norm = nn.GroupNorm(norm_num_groups, in_channels, eps=1e-6, affine=True)
        self.proj_in = nn.Linear(in_channels, inner)
        self.transformer_blocks = nn.ModuleList(
            _TransformerBlock(
                inner, num_attention_heads, attention_head_dim, cross_attention_dim
            )
            for _ in range(num_layers)
        )
        self.proj_out = nn.Linear(inner, in_channels)

    def forward(self, tokens: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        batch, channels, length = tokens.shape
        x = self.norm(tokens).permute(0, 2, 1).reshape(batch, length, channels)
        x = self.proj_in(x)
        for block in self.transformer_blocks:
            x = block(x, context)
        x = self.proj_out(x).reshape(batch, length, channels).permute(0, 2, 1)
        return x + tokens


class TriplaneTokenizer(nn.Module):
    """The learned triplane queries: 3 planes of ``plane_size``^2 tokens."""

    def __init__(self, plane_size: int, num_channels: int):
        super().__init__()
        self.plane_size = plane_size
        self.embeddings = nn.Parameter(
            torch.randn(3, num_channels, plane_size, plane_size)
            / math.sqrt(num_channels)
        )

    def forward(self, batch: int) -> torch.Tensor:
        """-> [B, C, 3 * P * P]."""
        planes = self.embeddings.unsqueeze(0).expand(batch, -1, -1, -1, -1)
        return planes.permute(0, 2, 1, 3, 4).reshape(batch, planes.shape[2], -1)

    def detokenize(self, tokens: torch.Tensor) -> torch.Tensor:
        """[B, C, 3 * P * P] -> [B, 3, C, P, P]."""
        batch, channels, _ = tokens.shape
        side = self.plane_size
        return tokens.view(batch, channels, 3, side, side).permute(0, 2, 1, 3, 4)


class TriplaneUpsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=2, stride=2
        )

    def forward(self, planes: torch.Tensor) -> torch.Tensor:
        batch = planes.shape[0]
        out = self.upsample(planes.flatten(0, 1))
        return out.view(batch, 3, *out.shape[1:])


class NeRFMLP(nn.Module):
    """Triplane features [N, 120] -> density [N, 1] and colour features [N, 3]."""

    def __init__(self, in_channels: int, neurons: int, hidden_layers: int):
        super().__init__()
        layers = [nn.Linear(in_channels, neurons), nn.SiLU(inplace=True)]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(neurons, neurons), nn.SiLU(inplace=True)]
        layers.append(nn.Linear(neurons, 4))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.layers(x)
        return out[..., :1], out[..., 1:4]


# -- The whole model ------------------------------------------------------------


def _scale(x: torch.Tensor, source: Tuple[float, float], target: Tuple[float, float]):
    """Map ``source`` range to ``target`` range with upstream's arithmetic (same rounding)."""
    x = (x - source[0]) / (source[1] - source[0])
    return x * (target[1] - target[0]) + target[0]


class TripoSR(nn.Module):
    """Upstream's ``TSR`` system; state-dict keys equal ``model.ckpt``'s."""

    def __init__(
        self, cfg: Dict[str, float] = CONFIG, dino_cfg: Dict[str, int] = DINO_CONFIG
    ):
        super().__init__()
        self.cfg = dict(cfg)
        channels = cfg["num_channels"]
        self.image_tokenizer = ImageTokenizer(dino_cfg)
        self.tokenizer = TriplaneTokenizer(cfg["plane_size"], channels)
        self.backbone = Transformer1D(
            channels,
            cfg["num_attention_heads"],
            cfg["attention_head_dim"],
            cfg["num_layers"],
            cfg["cross_attention_dim"],
        )
        self.post_processor = TriplaneUpsample(channels, cfg["triplane_channels"])
        self.decoder = NeRFMLP(
            3 * cfg["triplane_channels"], cfg["mlp_neurons"], cfg["mlp_hidden_layers"]
        )

    @property
    def radius(self) -> float:
        return float(self.cfg["radius"])

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """[B, 3, S, S] RGB in [0, 1] -> triplanes [B, 3, 40, 2P, 2P] (upstream's scene codes)."""
        context = self.image_tokenizer(images)
        tokens = self.backbone(self.tokenizer(images.shape[0]), context)
        return self.post_processor(self.tokenizer.detokenize(tokens))

    def query(
        self, triplane: torch.Tensor, positions: torch.Tensor, chunk: int = 262144
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """One triplane [3, C, H, W], points [N, 3] in [-radius, radius] -> density [N], colour [N, 3]."""
        radius = self.radius
        densities, colours = [], []
        for start in range(0, positions.shape[0], chunk):
            x = _scale(positions[start : start + chunk], (-radius, radius), (-1.0, 1.0))
            grid = torch.stack((x[:, [0, 1]], x[:, [0, 2]], x[:, [1, 2]]), dim=0)
            feats = F.grid_sample(
                triplane, grid.unsqueeze(1), align_corners=False, mode="bilinear"
            )  # [3, C, 1, N], one plane per batch row
            feats = feats.squeeze(2).permute(2, 0, 1).reshape(x.shape[0], -1)
            density, features = self.decoder(feats)
            densities.append(torch.exp(density[:, 0] + self.cfg["density_bias"]))
            colours.append(torch.sigmoid(features))
        return torch.cat(densities), torch.cat(colours)

    @torch.no_grad()
    def density_grid(self, triplane: torch.Tensor, resolution: int) -> torch.Tensor:
        """Density on upstream's marching-cubes lattice, [R, R, R] indexed (x, y, z)."""
        axis = torch.linspace(0.0, 1.0, resolution)
        x, y, z = torch.meshgrid(axis, axis, axis, indexing="ij")
        points = torch.stack((x, y, z), dim=-1).reshape(-1, 3)
        points = _scale(points, (0.0, 1.0), (-self.radius, self.radius))
        points = points.to(triplane.device)
        density, _ = self.query(triplane, points)
        return density.view(resolution, resolution, resolution)


@torch.no_grad()
def extract_mesh(
    model: TripoSR, triplane: torch.Tensor, resolution: int, threshold: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Upstream's ``extract_mesh`` with vertex colours.

    Returns vertices [V, 3] float32 in upstream's frame (within
    ``[-radius, radius]^3``), faces [F, 3] int64 wound counter-clockwise seen
    from outside, and vertex colours [V, 3] in [0, 1]. Marching cubes runs on
    the CPU through scikit-image (BSD-3-Clause) instead of upstream's
    torchmcubes: the same isosurface of the same density grid, triangulated
    with scikit-image's case tables.
    """
    from skimage.measure import marching_cubes

    level = (model.density_grid(triplane, resolution) - threshold).float().cpu().numpy()
    if not (level.min() < 0.0 < level.max()):
        raise ValueError(
            f"No surface at threshold={threshold}: the density field spans "
            f"[{level.min() + threshold:.3g}, {level.max() + threshold:.3g}]."
        )
    verts, faces, _normals, _values = marching_cubes(
        level, 0.0, method="lorensen", gradient_direction="ascent"
    )
    radius = model.radius
    vertices = torch.from_numpy(verts.astype("float32")) / (resolution - 1.0)
    vertices = _scale(vertices, (0.0, 1.0), (-radius, radius))
    faces = torch.from_numpy(faces.astype("int64"))
    _density, colours = model.query(triplane, vertices.to(triplane.device))
    return vertices, faces, colours.float().cpu()
