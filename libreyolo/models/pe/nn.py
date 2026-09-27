"""Native PyTorch Perception Encoder (PE) Core towers.

PE Core is a dual-tower vision-language encoder from Meta
(https://arxiv.org/abs/2504.13181). This module is a **native LibreYOLO
re-implementation** -- neither ``timm`` nor ``open_clip`` is imported at
runtime.

Code provenance
---------------
The architecture is adapted from two permissively licensed sources, pinned to
the exact revisions used to validate exact parity:

* Vision tower (``PEVisionTransformer``, ``PEBlock``, ``PEAttentionRope``,
  ``PEAttentionPoolLatent``, ``RotaryEmbeddingCat`` and the fourier/rope
  helpers) is adapted from **huggingface/pytorch-image-models v1.0.28**
  (commit ``8ef73809f622e0031bd7f4940265734aef8b9978``), Apache-2.0, files
  ``timm/models/eva.py``, ``timm/layers/pos_embed_sincos.py`` and
  ``timm/layers/attention_pool.py``.
* Text tower (``PETextTransformer``, ``ResidualAttentionBlock``) is adapted
  from **mlfoundations/open_clip v3.2.0** (commit
  ``6f939057c792a2f3d4d58df748de60ca47c4aed4``), MIT, file
  ``src/open_clip/transformer.py``.

Module and parameter names deliberately mirror the upstream OpenCLIP-converted
checkpoint layout (``visual.trunk.*`` / ``text.*`` / ``logit_scale``) so that
conversion is a strict metadata-wrap with no key remapping.

See ``libreyolo/models/pe/NOTICE.md`` for the full notice text.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "PE_CONFIGS",
    "PEConfig",
    "LibrePEModel",
    "build_pe_model",
]


# =============================================================================
# Closed configuration table
# =============================================================================


@dataclass(frozen=True)
class PEConfig:
    """Complete, closed configuration for one PE Core size.

    Values are transcribed from the pinned timm model definitions
    (``vit_pe_core_*`` in ``timm/models/eva.py`` v1.0.28) and the pinned
    OpenCLIP model configs (``PE-Core-*`` in open_clip v3.2.0). The converter
    asserts every field against the source config; an unknown or changed
    upstream config must fail rather than silently produce a wrong model.
    """

    # vision trunk
    image_size: int
    patch_size: int
    embed_dim: int
    depth: int
    num_heads: int
    mlp_ratio: float
    ref_feat_shape: Tuple[int, int]
    rope_grid_offset: float
    class_token: bool
    attn_pool_num_heads: int
    attn_pool_mlp_ratio: float
    # shared projection dim (the joint image/text embedding space)
    projection_dim: int
    # text tower
    text_width: int
    text_heads: int
    text_layers: int
    context_length: int
    vocab_size: int
    # upstream identity
    timm_model_name: str
    open_clip_model_name: str

    @property
    def grid_size(self) -> Tuple[int, int]:
        side = self.image_size // self.patch_size
        return (side, side)

    @property
    def num_patches(self) -> int:
        return self.grid_size[0] * self.grid_size[1]

    @property
    def num_prefix_tokens(self) -> int:
        return 1 if self.class_token else 0

    @property
    def head_dim(self) -> int:
        return self.embed_dim // self.num_heads


PE_CONFIGS: Dict[str, PEConfig] = {
    "t16": PEConfig(
        image_size=384,
        patch_size=16,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4.0,
        ref_feat_shape=(24, 24),
        rope_grid_offset=1.0,
        class_token=True,
        attn_pool_num_heads=8,
        attn_pool_mlp_ratio=4.0,
        projection_dim=512,
        text_width=512,
        text_heads=8,
        text_layers=12,
        context_length=32,
        vocab_size=49408,
        timm_model_name="vit_pe_core_tiny_patch16_384",
        open_clip_model_name="PE-Core-T-16-384",
    ),
    "s16": PEConfig(
        image_size=384,
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.0,
        ref_feat_shape=(24, 24),
        rope_grid_offset=1.0,
        class_token=True,
        attn_pool_num_heads=8,
        attn_pool_mlp_ratio=4.0,
        projection_dim=512,
        text_width=512,
        text_heads=8,
        text_layers=12,
        context_length=32,
        vocab_size=49408,
        timm_model_name="vit_pe_core_small_patch16_384",
        open_clip_model_name="PE-Core-S-16-384",
    ),
    "b16": PEConfig(
        image_size=224,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        ref_feat_shape=(14, 14),
        rope_grid_offset=1.0,
        class_token=True,
        attn_pool_num_heads=8,
        attn_pool_mlp_ratio=4.0,
        projection_dim=1024,
        text_width=1024,
        text_heads=16,
        text_layers=24,
        context_length=32,
        vocab_size=49408,
        timm_model_name="vit_pe_core_base_patch16_224",
        open_clip_model_name="PE-Core-B-16",
    ),
    "l14": PEConfig(
        image_size=336,
        patch_size=14,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        ref_feat_shape=(24, 24),
        rope_grid_offset=1.0,
        class_token=True,
        attn_pool_num_heads=8,
        attn_pool_mlp_ratio=4.0,
        projection_dim=1024,
        text_width=1024,
        text_heads=16,
        text_layers=24,
        context_length=32,
        vocab_size=49408,
        timm_model_name="vit_pe_core_large_patch14_336",
        open_clip_model_name="PE-Core-L-14-336",
    ),
    "g14": PEConfig(
        image_size=448,
        patch_size=14,
        embed_dim=1536,
        depth=50,
        num_heads=16,
        mlp_ratio=8960 / 1536,
        ref_feat_shape=(32, 32),
        # The gigantic variant is the one size that does NOT set
        # rope_grid_offset upstream, and drops the class token.
        rope_grid_offset=0.0,
        class_token=False,
        attn_pool_num_heads=8,
        attn_pool_mlp_ratio=4.0,
        projection_dim=1280,
        text_width=1280,
        text_heads=20,
        text_layers=24,
        context_length=72,
        vocab_size=49408,
        timm_model_name="vit_pe_core_gigantic_patch14_448",
        open_clip_model_name="PE-Core-bigG-14-448",
    ),
}

# PE preprocessing is symmetric [-1, 1], NOT ImageNet or CLIP statistics.
PE_MEAN: Tuple[float, float, float] = (0.5, 0.5, 0.5)
PE_STD: Tuple[float, float, float] = (0.5, 0.5, 0.5)


# =============================================================================
# Rotary position embedding (adapted from timm/layers/pos_embed_sincos.py)
# =============================================================================


def _freq_bands(num_bands: int, temperature: float = 10000.0) -> torch.Tensor:
    exp = (
        torch.arange(0, num_bands, 1, dtype=torch.int64).to(torch.float32) / num_bands
    )
    return 1.0 / (temperature**exp)


def _swap_shape_xy(shape: Tuple[int, ...]) -> Tuple[int, ...]:
    return tuple(reversed(shape))


def build_rotary_pos_embed(
    feat_shape: Tuple[int, int],
    dim: int,
    temperature: float = 10000.0,
    ref_feat_shape: Optional[Tuple[int, int]] = None,
    grid_offset: float = 0.0,
    grid_indexing: str = "xy",
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Concatenated sin/cos rotary embedding in EVA's "language" (non-pixel) mode.

    Mirrors ``timm.layers.pos_embed_sincos.build_rotary_pos_embed`` with
    ``in_pixels=False``, which is what ``attn_type='rope'`` EVA/PE models use.
    """
    num_bands = dim // 4
    bands = _freq_bands(num_bands, temperature=temperature).to(device=device)

    if grid_indexing == "xy":
        feat_shape = _swap_shape_xy(feat_shape)
        if ref_feat_shape is not None:
            ref_feat_shape = _swap_shape_xy(ref_feat_shape)

    t = [
        torch.arange(s, device=device, dtype=torch.int64).to(torch.float32)
        + grid_offset
        for s in feat_shape
    ]
    if ref_feat_shape is not None:
        # EVA's scheme for resizing rope embeddings (ref shape = pretrain shape).
        t = [x / f * r for x, f, r in zip(t, feat_shape, ref_feat_shape)]

    grid = torch.stack(torch.meshgrid(t, indexing=grid_indexing), dim=-1)
    grid = grid.unsqueeze(-1)
    pos = grid * bands

    num_spatial_dim = 1
    for x in feat_shape:
        num_spatial_dim *= x

    sin_emb = pos.sin().to(dtype=dtype).reshape(num_spatial_dim, -1)
    cos_emb = pos.cos().to(dtype=dtype).reshape(num_spatial_dim, -1)
    return sin_emb.repeat_interleave(2, -1), cos_emb.repeat_interleave(2, -1)


def _rot(x: torch.Tensor) -> torch.Tensor:
    """``[x0, x1, x2, x3] -> [-x1, x0, -x3, x2]`` (interleaved rotation)."""
    return torch.stack([-x[..., 1::2], x[..., ::2]], -1).reshape(x.shape)


def apply_rot_embed_cat(x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
    sin_emb, cos_emb = emb.chunk(2, -1)
    return x * cos_emb + _rot(x) * sin_emb


class RotaryEmbeddingCat(nn.Module):
    """Rotary position embedding with concatenated sin/cos, cached for a fixed grid.

    Adapted from ``timm.layers.pos_embed_sincos.RotaryEmbeddingCat``. PE always
    uses a static feature shape (``dynamic_img_size`` is disabled upstream), so
    only the cached-embedding path is implemented here.
    """

    def __init__(
        self,
        dim: int,
        feat_shape: Tuple[int, int],
        temperature: float = 10000.0,
        ref_feat_shape: Optional[Tuple[int, int]] = None,
        grid_offset: float = 0.0,
        grid_indexing: str = "xy",
    ) -> None:
        super().__init__()
        self.dim = dim
        self.feat_shape = feat_shape
        self.temperature = temperature
        self.ref_feat_shape = ref_feat_shape
        self.grid_offset = grid_offset
        self.grid_indexing = grid_indexing

        sin_emb, cos_emb = build_rotary_pos_embed(
            feat_shape=feat_shape,
            dim=dim,
            temperature=temperature,
            ref_feat_shape=ref_feat_shape,
            grid_offset=grid_offset,
            grid_indexing=grid_indexing,
        )
        # Non-persistent: the embedding is fully determined by the config, so it
        # must not appear in (or be required by) the checkpoint state dict.
        self.register_buffer(
            "pos_embed_cat",
            torch.cat([sin_emb, cos_emb], dim=-1),
            persistent=False,
        )

    def get_embed(self) -> torch.Tensor:
        return self.pos_embed_cat


# =============================================================================
# Vision tower (adapted from timm/models/eva.py)
# =============================================================================


class Mlp(nn.Module):
    """Standard transformer MLP (timm ``Mlp`` with default GELU / no norm)."""

    def __init__(self, in_features: int, hidden_features: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class PatchEmbed(nn.Module):
    """Image to patch embedding. PE uses a bias-free patch projection."""

    def __init__(self, patch_size: int, in_chans: int, embed_dim: int) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)  # (B, N, C)


class PEAttentionRope(nn.Module):
    """Fused-QKV self-attention with rotary embeddings applied to non-prefix tokens.

    Adapted from ``timm.models.eva.EvaAttention`` restricted to the
    configuration PE uses: fused qkv with bias, no q/k norm, no attention
    gating, no inner scale-norm, interleaved (non-half) rotation.
    """

    def __init__(self, dim: int, num_heads: int, num_prefix_tokens: int) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim {dim} not divisible by num_heads {num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.num_prefix_tokens = num_prefix_tokens

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, rope: Optional[torch.Tensor]) -> torch.Tensor:
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)  # (B, heads, N, head_dim)

        if rope is not None:
            npt = self.num_prefix_tokens
            q = torch.cat(
                [q[:, :, :npt, :], apply_rot_embed_cat(q[:, :, npt:, :], rope)], dim=2
            ).type_as(v)
            k = torch.cat(
                [k[:, :, :npt, :], apply_rot_embed_cat(k[:, :, npt:, :], rope)], dim=2
            ).type_as(v)

        x = F.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class PEBlock(nn.Module):
    """Pre-norm transformer block (timm ``EvaBlock`` in its PE configuration)."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float,
        num_prefix_tokens: int,
        norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=norm_eps)
        self.attn = PEAttentionRope(dim, num_heads, num_prefix_tokens)
        self.norm2 = nn.LayerNorm(dim, eps=norm_eps)
        self.mlp = Mlp(dim, int(round(dim * mlp_ratio)))

    def forward(self, x: torch.Tensor, rope: Optional[torch.Tensor]) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), rope=rope)
        x = x + self.mlp(self.norm2(x))
        return x


class PEAttentionPoolLatent(nn.Module):
    """Single-latent attention pooling head (timm ``AttentionPoolLatent``).

    PE uses 8 pooling heads regardless of trunk width, one latent query, an
    MLP residual, and token pooling of the single latent.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float,
        norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim {dim} not divisible by attn_pool heads {num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.latent_len = 1

        self.latent = nn.Parameter(torch.zeros(1, self.latent_len, dim))
        self.q = nn.Linear(dim, dim, bias=True)
        self.kv = nn.Linear(dim, dim * 2, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim, eps=norm_eps)
        self.mlp = Mlp(dim, int(round(dim * mlp_ratio)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        q_latent = self.latent.expand(B, -1, -1)
        q = (
            self.q(q_latent)
            .reshape(B, self.latent_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        kv = (
            self.kv(x)
            .reshape(B, N, 2, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        k, v = kv.unbind(0)

        x = F.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).reshape(B, self.latent_len, C)
        x = self.proj(x)
        x = x + self.mlp(self.norm(x))
        return x[:, 0]


class PEVisionTransformer(nn.Module):
    """PE Core vision trunk: RoPE ViT with attention pooling and a linear head.

    Mirrors the timm ``Eva`` submodule names so that OpenCLIP-converted PE
    checkpoints (``visual.trunk.*``) load strictly with no key remapping.
    """

    def __init__(self, cfg: PEConfig, norm_eps: float = 1e-5) -> None:
        super().__init__()
        self.cfg = cfg
        self.num_prefix_tokens = cfg.num_prefix_tokens

        self.patch_embed = PatchEmbed(cfg.patch_size, 3, cfg.embed_dim)

        self.cls_token = (
            nn.Parameter(torch.zeros(1, 1, cfg.embed_dim)) if cfg.class_token else None
        )
        num_pos_tokens = cfg.num_patches + self.num_prefix_tokens
        self.pos_embed = nn.Parameter(torch.zeros(1, num_pos_tokens, cfg.embed_dim))

        self.rope = RotaryEmbeddingCat(
            dim=cfg.head_dim,
            feat_shape=cfg.grid_size,
            ref_feat_shape=cfg.ref_feat_shape,
            grid_offset=cfg.rope_grid_offset,
            grid_indexing="xy",
        )

        self.norm_pre = nn.LayerNorm(cfg.embed_dim, eps=norm_eps)
        self.blocks = nn.ModuleList(
            [
                PEBlock(
                    dim=cfg.embed_dim,
                    num_heads=cfg.num_heads,
                    mlp_ratio=cfg.mlp_ratio,
                    num_prefix_tokens=self.num_prefix_tokens,
                    norm_eps=norm_eps,
                )
                for _ in range(cfg.depth)
            ]
        )
        self.norm = nn.LayerNorm(cfg.embed_dim, eps=norm_eps)

        self.attn_pool = PEAttentionPoolLatent(
            dim=cfg.embed_dim,
            num_heads=cfg.attn_pool_num_heads,
            mlp_ratio=cfg.attn_pool_mlp_ratio,
            norm_eps=norm_eps,
        )
        self.head = nn.Linear(cfg.embed_dim, cfg.projection_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        if self.cls_token is not None:
            x = torch.cat([self.cls_token.expand(x.shape[0], -1, -1), x], dim=1)
        x = x + self.pos_embed

        rope = self.rope.get_embed()
        x = self.norm_pre(x)
        for blk in self.blocks:
            x = blk(x, rope=rope)
        x = self.norm(x)

        x = self.attn_pool(x)
        return self.head(x)


class PEVisualWrapper(nn.Module):
    """Thin ``visual.trunk`` container matching the OpenCLIP ``TimmModel`` layout."""

    def __init__(self, cfg: PEConfig) -> None:
        super().__init__()
        self.trunk = PEVisionTransformer(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.trunk(x)


# =============================================================================
# Text tower (adapted from open_clip/transformer.py)
# =============================================================================


class ResidualAttentionBlock(nn.Module):
    """OpenCLIP residual attention block.

    Uses ``nn.MultiheadAttention`` exactly as upstream does, so the fused
    attention kernel -- and therefore the numerics -- are identical.
    """

    def __init__(self, d_model: int, n_head: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        # batch_first=True matches open_clip v3.2.0, which keeps the sequence in
        # NLD throughout. Running seq-first instead changes the attention kernel
        # and costs exact parity (~6e-6).
        self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=True)
        self.ln_2 = nn.LayerNorm(d_model)
        mlp_width = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential()
        self.mlp.add_module("c_fc", nn.Linear(d_model, mlp_width))
        self.mlp.add_module("gelu", nn.GELU())
        self.mlp.add_module("c_proj", nn.Linear(mlp_width, d_model))

    def attention(
        self, q_x: torch.Tensor, attn_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        attn_mask = attn_mask.to(q_x.dtype) if attn_mask is not None else None
        return self.attn(q_x, q_x, q_x, need_weights=False, attn_mask=attn_mask)[0]

    def forward(
        self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = x + self.attention(self.ln_1(x), attn_mask=attn_mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int) -> None:
        super().__init__()
        self.resblocks = nn.ModuleList(
            [ResidualAttentionBlock(width, heads) for _ in range(layers)]
        )

    def forward(
        self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        for blk in self.resblocks:
            x = blk(x, attn_mask=attn_mask)
        return x


class PETextTransformer(nn.Module):
    """OpenCLIP ``TextTransformer`` with argmax (EOT) pooling and a linear projection."""

    def __init__(self, cfg: PEConfig) -> None:
        super().__init__()
        self.context_length = cfg.context_length
        self.vocab_size = cfg.vocab_size

        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.text_width)
        self.positional_embedding = nn.Parameter(
            torch.empty(cfg.context_length, cfg.text_width)
        )
        self.transformer = Transformer(cfg.text_width, cfg.text_layers, cfg.text_heads)
        self.ln_final = nn.LayerNorm(cfg.text_width)
        self.text_projection = nn.Parameter(
            torch.empty(cfg.text_width, cfg.projection_dim)
        )

        self.register_buffer(
            "attn_mask", self._build_causal_mask(cfg.context_length), persistent=False
        )

    @staticmethod
    def _build_causal_mask(context_length: int) -> torch.Tensor:
        mask = torch.empty(context_length, context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)
        return mask

    def forward(self, text: torch.Tensor) -> torch.Tensor:
        seq_len = text.shape[1]
        x = self.token_embedding(text)
        x = x + self.positional_embedding[:seq_len]
        # Sequence stays in NLD (batch-first) end to end, as upstream does.
        x = self.transformer(x, attn_mask=self.attn_mask[:seq_len, :seq_len])
        x = self.ln_final(x)
        # EOT pooling: the highest token id in each row is the EOT marker.
        pooled = x[torch.arange(x.shape[0], device=x.device), text.argmax(dim=-1)]
        return pooled @ self.text_projection


# =============================================================================
# Full dual-tower model
# =============================================================================


class LibrePEModel(nn.Module):
    """PE Core dual-tower model: image tower, text tower, learned logit scale."""

    def __init__(self, cfg: PEConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.visual = PEVisualWrapper(cfg)
        self.text = PETextTransformer(cfg)
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))

    # -- properties mirroring the sibling zero-shot families -----------------

    @property
    def context_length(self) -> int:
        return self.cfg.context_length

    @property
    def embedding_dim(self) -> int:
        return self.cfg.projection_dim

    @property
    def image_size(self) -> int:
        return self.cfg.image_size

    # -- encoders ------------------------------------------------------------

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """Un-normalized image embeddings ``(B, D)``."""
        return self.visual(images)

    def encode_text(self, tokens: torch.Tensor) -> torch.Tensor:
        """Un-normalized text embeddings ``(B, D)``."""
        return self.text(tokens)

    def encode_video(self, clips: torch.Tensor) -> torch.Tensor:
        """Whole-clip embeddings ``(B, D)`` from a ``(B, F, C, H, W)`` tensor.

        PE defines a video embedding as the arithmetic mean of independently
        encoded frame embeddings, L2-normalized exactly once at the end.
        """
        if clips.ndim != 5:
            raise ValueError(
                "encode_video expects a 5D (B, F, C, H, W) tensor; "
                f"got shape {tuple(clips.shape)}."
            )
        b, f = clips.shape[:2]
        frames = clips.reshape(b * f, *clips.shape[2:])
        feats = self.encode_image(frames).reshape(b, f, -1)
        return F.normalize(feats.mean(dim=1), dim=-1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.encode_image(images)


def build_pe_model(size: str) -> LibrePEModel:
    """Instantiate a PE Core model from the closed configuration table."""
    if size not in PE_CONFIGS:
        raise ValueError(
            f"Unknown PE size {size!r}; expected one of {tuple(PE_CONFIGS)}."
        )
    return LibrePEModel(PE_CONFIGS[size])
