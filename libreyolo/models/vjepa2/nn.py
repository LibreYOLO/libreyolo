"""Native V-JEPA 2.0 encoder, attentive pooler and classifier.

Provenance
----------
This file is adapted from the Hugging Face Transformers implementation of
V-JEPA 2, pinned at tag ``v5.1.0`` (commit
``3fa4da70f5d1da3ab1a8304bd2339eab7004b157``):

    src/transformers/models/vjepa2/modeling_vjepa2.py

Copyright 2025 The HuggingFace Inc. team. All rights reserved.
Licensed under the Apache License, Version 2.0. A copy of the Apache-2.0
license text accompanies this directory in ``NOTICE``.

The semantic upstream (architecture behaviour, attentive-probe recipe) is
``facebookresearch/vjepa2`` at commit
``204698b45b3712590f06245fbfba32d3be539812``, MIT licensed.

Deliberate deviations from the Transformers source, all of which preserve
numerics exactly:

* Only the encoder, attentive pooler and classifier are ported. The
  self-supervised predictor (``VJEPA2Predictor`` and friends) is out of scope
  and is not present here.
* ``PreTrainedModel``/``ModelOutput`` plumbing is replaced with plain
  ``nn.Module`` and tensor returns, so LibreYOLO does not depend on
  ``transformers`` at inference or export time.
* Parameter names are kept byte-identical to upstream so conversion is a
  key-filter rather than a remap, and so the parity diff stays readable.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "VJEPA2Config",
    "LibreVJEPA2Encoder",
    "LibreVJEPA2Classifier",
    "VJEPA2_CONFIGS",
]


# --- Architecture table -----------------------------------------------------
#
# Values are transcribed from the pinned Hugging Face ``config.json`` of each
# released snapshot. Depth / heads / mlp_ratio are NOT inferable from the size
# label: g256 and g384 share hidden_size 1408 but differ in crop_size, and the
# g variants use 22 heads and a non-integer mlp_ratio.
VJEPA2_CONFIGS: dict[str, dict] = {
    "l256": {
        "hidden_size": 1024,
        "num_attention_heads": 16,
        "num_hidden_layers": 24,
        "mlp_ratio": 4.0,
        "crop_size": 256,
    },
    "h256": {
        "hidden_size": 1280,
        "num_attention_heads": 16,
        "num_hidden_layers": 32,
        "mlp_ratio": 4.0,
        "crop_size": 256,
    },
    "g256": {
        "hidden_size": 1408,
        "num_attention_heads": 22,
        "num_hidden_layers": 40,
        "mlp_ratio": 4.363636363636363,
        "crop_size": 256,
    },
    "g384": {
        "hidden_size": 1408,
        "num_attention_heads": 22,
        "num_hidden_layers": 40,
        "mlp_ratio": 4.363636363636363,
        "crop_size": 384,
    },
}


class VJEPA2Config:
    """Family-local config mirroring the pinned upstream ``VJEPA2Config``.

    Only the fields the encoder actually reads are kept. Holding this locally
    keeps ``transformers`` off the inference and export import path.
    """

    def __init__(
        self,
        hidden_size: int = 1024,
        num_attention_heads: int = 16,
        num_hidden_layers: int = 24,
        mlp_ratio: float = 4.0,
        crop_size: int = 256,
        patch_size: int = 16,
        tubelet_size: int = 2,
        frames_per_clip: int = 64,
        in_chans: int = 3,
        layer_norm_eps: float = 1e-6,
        qkv_bias: bool = True,
        drop_path_rate: float = 0.0,
        attention_probs_dropout_prob: float = 0.0,
        hidden_act: str = "gelu",
        attn_implementation: str = "sdpa",
    ):
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.mlp_ratio = mlp_ratio
        self.crop_size = crop_size
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size
        self.frames_per_clip = frames_per_clip
        self.in_chans = in_chans
        self.layer_norm_eps = layer_norm_eps
        self.qkv_bias = qkv_bias
        self.drop_path_rate = drop_path_rate
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.hidden_act = hidden_act
        self.attn_implementation = attn_implementation

    @classmethod
    def for_size(cls, size: str, **overrides) -> "VJEPA2Config":
        if size not in VJEPA2_CONFIGS:
            raise ValueError(
                f"unknown V-JEPA 2 size {size!r}; expected one of "
                f"{sorted(VJEPA2_CONFIGS)}"
            )
        params = dict(VJEPA2_CONFIGS[size])
        params.update(overrides)
        return cls(**params)

    @property
    def grid_size(self) -> int:
        return self.crop_size // self.patch_size

    @property
    def grid_depth(self) -> int:
        return self.frames_per_clip // self.tubelet_size


def _act(name: str):
    if name == "gelu":
        # Upstream ACT2FN["gelu"] is the exact (erf) GELU, not the tanh
        # approximation. Using the wrong one silently breaks exact parity.
        return nn.GELU()
    if name == "silu":
        return nn.SiLU()
    raise ValueError(f"unsupported activation {name!r}")


class VJEPA2PatchEmbeddings3D(nn.Module):
    """Tubelet (3D) patch embedding: Conv3d with stride == kernel."""

    def __init__(self, config: VJEPA2Config, hidden_size: int):
        super().__init__()
        self.patch_size = config.patch_size
        self.tubelet_size = config.tubelet_size
        self.hidden_size = hidden_size
        self.proj = nn.Conv3d(
            in_channels=config.in_chans,
            out_channels=hidden_size,
            kernel_size=(config.tubelet_size, config.patch_size, config.patch_size),
            stride=(config.tubelet_size, config.patch_size, config.patch_size),
        )

    def forward(self, pixel_values_videos: torch.Tensor) -> torch.Tensor:
        return self.proj(pixel_values_videos).flatten(2).transpose(1, 2)


class VJEPA2Embeddings(nn.Module):
    """Accepts public ``(B, F, C, H, W)`` and permutes to Conv3d layout."""

    def __init__(self, config: VJEPA2Config, hidden_size: int):
        super().__init__()
        self.config = config
        self.hidden_size = hidden_size
        self.patch_embeddings = VJEPA2PatchEmbeddings3D(config, hidden_size=hidden_size)
        self.patch_size = config.patch_size

    def forward(self, pixel_values_videos: torch.Tensor) -> torch.Tensor:
        num_frames = pixel_values_videos.shape[1]
        # (B, F, C, H, W) -> (B, C, F, H, W). This is the single adapter
        # between the public layout and the native Conv3d layout.
        pixel_values_videos = pixel_values_videos.permute(0, 2, 1, 3, 4)
        if num_frames < self.config.tubelet_size:
            pixel_values_videos = pixel_values_videos.repeat(
                1, 1, self.config.tubelet_size, 1, 1
            )
        target_dtype = self.patch_embeddings.proj.weight.dtype
        pixel_values_videos = pixel_values_videos.to(dtype=target_dtype)
        return self.patch_embeddings(pixel_values_videos)


def rotate_queries_or_keys(x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    """Factorized rotary embedding, transcribed from the pinned upstream.

    Note the deliberate asymmetry, which is upstream behaviour and must be
    preserved bit-for-bit: the sin/cos tables are built by ``repeat`` (so they
    are the half-width table *concatenated* with itself), while the rotated
    companion ``y`` is built by pairing *adjacent* channels. Rewriting this to
    the conventional interleaved or half-split rotary changes the numerics and
    breaks parity against every released checkpoint.
    """
    _, _, _, D = x.size()

    omega = torch.arange(D // 2, dtype=x.dtype, device=x.device)
    omega = omega / (D / 2.0)
    omega = 1.0 / 10000**omega  # (D/2,)
    freq = pos.unsqueeze(-1) * omega  # (..., N, D/2), outer product

    emb_sin = freq.sin()
    emb_cos = freq.cos()

    emb_sin = emb_sin.squeeze(-1).repeat(1, 1, 1, 2)
    emb_cos = emb_cos.squeeze(-1).repeat(1, 1, 1, 2)

    y = x.unflatten(-1, (-1, 2))
    y1, y2 = y.unbind(dim=-1)
    y = torch.stack((-y2, y1), dim=-1)
    y = y.flatten(-2)
    return (x * emb_cos) + (y * emb_sin)


def eager_attention_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scaling: float,
    dropout: float = 0.0,
    training: bool = False,
):
    attn_weights = torch.matmul(query, key.transpose(-1, -2)) * scaling
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = F.dropout(attn_weights, p=dropout, training=training)
    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


def sdpa_attention_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scaling: float,
    dropout: float = 0.0,
    training: bool = False,
):
    """Mirrors ``transformers.integrations.sdpa_attention.sdpa_attention_forward``.

    Kept separate from the eager path so parity can be asserted against the
    oracle under *both* attention implementations rather than only the default.
    """
    attn_output = F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=None,
        dropout_p=dropout if training else 0.0,
        scale=scaling,
        is_causal=False,
    )
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, None


class VJEPA2RopeAttention(nn.Module):
    def __init__(
        self,
        config: VJEPA2Config,
        hidden_size: int,
        num_attention_heads: int,
    ):
        super().__init__()
        self.config = config
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"hidden size {hidden_size} is not a multiple of the number of "
                f"attention heads {num_attention_heads}"
            )

        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.key = nn.Linear(hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.value = nn.Linear(hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.proj = nn.Linear(hidden_size, hidden_size)

        self.dropout_prob = config.attention_probs_dropout_prob
        self.dropout = nn.Dropout(self.dropout_prob)

        self.grid_size = config.crop_size // config.patch_size
        self.grid_depth = config.frames_per_clip // config.tubelet_size

        # Each of depth/height/width gets an even slice of the head dim; any
        # remainder is left unrotated. Integer division order matters.
        self.d_dim = int(2 * ((self.attention_head_size // 3) // 2))
        self.h_dim = int(2 * ((self.attention_head_size // 3) // 2))
        self.w_dim = int(2 * ((self.attention_head_size // 3) // 2))

        self.scaling = self.attention_head_size**-0.5
        self.is_causal = False

    def _get_frame_pos(self, ids: torch.Tensor) -> torch.Tensor:
        tokens_per_frame = int(self.grid_size * self.grid_size)
        return ids // tokens_per_frame

    def _get_height_pos(self, ids: torch.Tensor) -> torch.Tensor:
        tokens_per_frame = int(self.grid_size * self.grid_size)
        frame_ids = self._get_frame_pos(ids)
        ids = ids - tokens_per_frame * frame_ids
        tokens_per_row = self.grid_size
        return ids // tokens_per_row

    def get_position_ids(self, x: torch.Tensor):
        device = x.device
        token_size = x.size(1)
        ids = torch.arange(token_size, device=device)
        tokens_per_frame = int(self.grid_size * self.grid_size)
        frame_ids = self._get_frame_pos(ids)
        tokens_per_row = self.grid_size
        height_ids = self._get_height_pos(ids)
        width_ids = (ids - tokens_per_frame * frame_ids) - tokens_per_row * height_ids
        return frame_ids, height_ids, width_ids

    def apply_rotary_embeddings(self, qk: torch.Tensor, pos_ids) -> torch.Tensor:
        d_mask, h_mask, w_mask = pos_ids
        s = 0
        qkd = rotate_queries_or_keys(qk[..., s : s + self.d_dim], pos=d_mask)
        s += self.d_dim
        qkh = rotate_queries_or_keys(qk[..., s : s + self.h_dim], pos=h_mask)
        s += self.h_dim
        qkw = rotate_queries_or_keys(qk[..., s : s + self.w_dim], pos=w_mask)
        s += self.w_dim
        if s < self.attention_head_size:
            qkr = qk[..., s:]
            qk = torch.cat([qkd, qkh, qkw, qkr], dim=-1)
        else:
            qk = torch.cat([qkd, qkh, qkw], dim=-1)
        return qk

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, _, _ = hidden_states.shape
        shape = (batch_size, -1, self.num_attention_heads, self.attention_head_size)
        query_layer = self.query(hidden_states).view(*shape).transpose(1, 2)
        key_layer = self.key(hidden_states).view(*shape).transpose(1, 2)
        value_layer = self.value(hidden_states).view(*shape).transpose(1, 2)

        pos_ids = self.get_position_ids(hidden_states)
        key_layer = self.apply_rotary_embeddings(key_layer, pos_ids)
        query_layer = self.apply_rotary_embeddings(query_layer, pos_ids)

        impl = (
            sdpa_attention_forward
            if self.config.attn_implementation == "sdpa"
            else eager_attention_forward
        )
        context_layer, _ = impl(
            query_layer,
            key_layer,
            value_layer,
            scaling=self.scaling,
            dropout=0.0 if not self.training else self.dropout_prob,
            training=self.training,
        )

        new_shape = context_layer.size()[:-2] + (self.all_head_size,)
        return self.proj(context_layer.reshape(new_shape))


def drop_path(
    x: torch.Tensor, drop_prob: float = 0.0, training: bool = False
) -> torch.Tensor:
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class VJEPA2DropPath(nn.Module):
    def __init__(self, drop_prob: Optional[float] = None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return drop_path(hidden_states, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return f"p={self.drop_prob}"


class VJEPA2MLP(nn.Module):
    def __init__(self, config: VJEPA2Config, hidden_size: int, mlp_ratio: float):
        super().__init__()
        in_features = out_features = hidden_size
        # int() truncation, not round(): g's mlp_ratio of 4.363636... yields
        # 6144 for hidden_size 1408. Rounding would give 6144 too, but the
        # truncating form is what upstream uses and what the checkpoints match.
        hidden_features = int(hidden_size * mlp_ratio)
        self.fc1 = nn.Linear(in_features, hidden_features, bias=True)
        self.activation = _act(config.hidden_act)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=True)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.activation(self.fc1(hidden_state)))


class VJEPA2Layer(nn.Module):
    def __init__(
        self,
        config: VJEPA2Config,
        drop_path_rate: float,
        hidden_size: int,
        num_attention_heads: int,
        mlp_ratio: float,
    ):
        super().__init__()
        self.config = config
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.mlp_ratio = mlp_ratio

        self.norm1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_eps)
        self.attention = VJEPA2RopeAttention(config, hidden_size, num_attention_heads)
        self.drop_path = (
            VJEPA2DropPath(drop_path_rate) if config.drop_path_rate > 0.0 else nn.Identity()
        )
        self.norm2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_eps)
        self.mlp = VJEPA2MLP(config, hidden_size=hidden_size, mlp_ratio=mlp_ratio)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.attention(hidden_states)
        hidden_states = self.drop_path(hidden_states) + residual

        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.drop_path(hidden_states) + residual
        return hidden_states


class LibreVJEPA2Encoder(nn.Module):
    """V-JEPA 2.0 encoder producing spatiotemporal tokens.

    Input is the public 5D layout ``(B, F, C, H, W)``. Output is the final
    normalized token sequence ``(B, N, D)`` in upstream flattening order
    (time-major, then height, then width).
    """

    def __init__(self, config: VJEPA2Config):
        super().__init__()
        self.config = config
        self.embeddings = VJEPA2Embeddings(config, hidden_size=config.hidden_size)
        drop_path_rates = [
            (
                config.drop_path_rate * i / (config.num_hidden_layers - 1)
                if config.num_hidden_layers > 1
                else 0.0
            )
            for i in range(config.num_hidden_layers)
        ]
        self.layer = nn.ModuleList(
            [
                VJEPA2Layer(
                    config,
                    drop_path_rate=drop_path_rates[i],
                    hidden_size=config.hidden_size,
                    num_attention_heads=config.num_attention_heads,
                    mlp_ratio=config.mlp_ratio,
                )
                for i in range(config.num_hidden_layers)
            ]
        )
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, pixel_values_videos: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embeddings(pixel_values_videos)
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states)
        return self.layernorm(hidden_states)


# --- Attentive probe (milestone 2) ------------------------------------------


# The pooler's MLP is built upstream as ``VJEPA2MLP(config, hidden_size=...)``
# with the mlp_ratio argument left at its default. It is therefore ALWAYS 4.0,
# even for the g sizes whose encoder MLP ratio is 4.363636... Passing the
# encoder ratio here would build fc1 as 6144 instead of 5632 and fail the
# strict load of every released probe checkpoint.
_POOLER_MLP_RATIO = 4.0

# ``num_pooler_layers`` defaults to 3 upstream: three self-attention layers
# followed by one cross-attention layer. This is the "three-layer attentive
# pooler" the handoff refers to.
_POOLER_DEPTH = 3


class VJEPA2PoolerMLP(nn.Module):
    def __init__(self, hidden_size: int, mlp_ratio: float = _POOLER_MLP_RATIO):
        super().__init__()
        hidden_features = int(hidden_size * mlp_ratio)
        self.fc1 = nn.Linear(hidden_size, hidden_features, bias=True)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, hidden_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.activation(self.fc1(x)))


class VJEPA2PoolerSelfAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.embed_dim = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim**-0.5
        self.dropout = 0.0
        self.is_causal = False

        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_length, embed_dim = hidden_states.shape
        shape = (batch_size, seq_length, self.num_heads, self.head_dim)
        queries = self.q_proj(hidden_states).view(*shape).transpose(1, 2)
        keys = self.k_proj(hidden_states).view(*shape).transpose(1, 2)
        values = self.v_proj(hidden_states).view(*shape).transpose(1, 2)

        attn_output = F.scaled_dot_product_attention(
            queries, keys, values, attn_mask=None, dropout_p=0.0,
            scale=self.scale, is_causal=False,
        )
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_length, embed_dim)
        return self.out_proj(attn_output)


class VJEPA2PoolerCrossAttention(nn.Module):
    """Cross-attention where the query is the learned probe token.

    Unlike the pooler's self-attention, this block deliberately has **no**
    output projection -- upstream omits ``out_proj`` here. Adding one would
    introduce two tensors that no released probe checkpoint provides.
    """

    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.embed_dim = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim**-0.5
        self.dropout = 0.0
        self.is_causal = False

        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.q_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, queries: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, kv_length, embed_dim = hidden_states.shape
        q_length = queries.shape[1]

        queries = (
            self.q_proj(queries)
            .view(batch_size, q_length, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        kv_shape = (batch_size, kv_length, self.num_heads, self.head_dim)
        keys = self.k_proj(hidden_states).view(*kv_shape).transpose(1, 2)
        values = self.v_proj(hidden_states).view(*kv_shape).transpose(1, 2)

        attn_output = F.scaled_dot_product_attention(
            queries, keys, values, attn_mask=None, dropout_p=0.0,
            scale=self.scale, is_causal=False,
        )
        attn_output = attn_output.transpose(1, 2).contiguous()
        return attn_output.reshape(batch_size, q_length, embed_dim).contiguous()


class VJEPA2PoolerSelfAttentionLayer(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float, eps: float):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(hidden_size, eps=eps)
        self.self_attn = VJEPA2PoolerSelfAttention(hidden_size, num_heads)
        self.layer_norm2 = nn.LayerNorm(hidden_size, eps=eps)
        self.mlp = VJEPA2PoolerMLP(hidden_size, mlp_ratio)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class VJEPA2PoolerCrossAttentionLayer(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float, eps: float):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(hidden_size, eps=eps)
        self.cross_attn = VJEPA2PoolerCrossAttention(hidden_size, num_heads)
        self.layer_norm2 = nn.LayerNorm(hidden_size, eps=eps)
        self.mlp = VJEPA2PoolerMLP(hidden_size, mlp_ratio)

    def forward(self, queries: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = queries
        hidden_states = self.layer_norm1(hidden_states)
        queries = self.cross_attn(queries, hidden_states)
        queries = residual + queries

        residual = queries
        queries = self.layer_norm2(queries)
        queries = self.mlp(queries)
        return residual + queries


class VJEPA2AttentivePooler(nn.Module):
    """Three-layer attentive pooler: N self-attention blocks then cross-attention."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = _POOLER_MLP_RATIO,
        depth: int = _POOLER_DEPTH,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.query_tokens = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.cross_attention_layer = VJEPA2PoolerCrossAttentionLayer(
            hidden_size, num_heads, mlp_ratio, eps
        )
        self.self_attention_layers = nn.ModuleList(
            [
                VJEPA2PoolerSelfAttentionLayer(hidden_size, num_heads, mlp_ratio, eps)
                for _ in range(depth)
            ]
        )

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        for layer in self.self_attention_layers:
            hidden_state = layer(hidden_state)
        queries = self.query_tokens.repeat(hidden_state.shape[0], 1, 1)
        hidden_state = self.cross_attention_layer(queries, hidden_state)
        return hidden_state.squeeze(1)


class LibreVJEPA2Classifier(nn.Module):
    """Frozen-encoder attentive probe: encoder -> attentive pooler -> linear."""

    def __init__(self, config: VJEPA2Config, nc: int, probe_depth: int = _POOLER_DEPTH):
        super().__init__()
        self.config = config
        self.nc = nc
        self.probe_depth = probe_depth
        self.encoder = LibreVJEPA2Encoder(config)
        self.pooler = VJEPA2AttentivePooler(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            mlp_ratio=_POOLER_MLP_RATIO,
            depth=probe_depth,
            eps=config.layer_norm_eps,
        )
        self.classifier = nn.Linear(config.hidden_size, nc, bias=True)

    def forward(self, pixel_values_videos: torch.Tensor) -> torch.Tensor:
        tokens = self.encoder(pixel_values_videos)
        pooled = self.pooler(tokens)          # (B, D)
        return self.classifier(pooled)        # (B, K)
