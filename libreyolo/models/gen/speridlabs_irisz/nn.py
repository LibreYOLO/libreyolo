"""Iris-Z transformer: FLUX.2 with a modality embedding and per-grid noise levels.

Ported from the Iris-Z SDK (``iris_z/model/model.py``), which takes the
transformer from FLUX.2 by Black Forest Labs and adds two things: a learned
embedding per token modality (RGB, depth, given RGB, given depth) and separate
timestep modulation for the RGB and depth grids of one sequence. See
``NOTICE``. Parameter names mirror upstream so the released safetensors load
with ``strict=True``.

Only the forward pass Iris-Z samples with is ported: upstream's reference-image
KV-cache paths are unused by Iris-Z and left out. With no reference tokens,
upstream's attention reduces to one ``scaled_dot_product_attention`` over the
text and image tokens, which is what this module runs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass
class Flux2Params:
    """FLUX.2 [dev], the base of Iris-Z-big."""

    in_channels: int = 128
    context_in_dim: int = 15360
    hidden_size: int = 6144
    num_heads: int = 48
    depth: int = 8
    depth_single_blocks: int = 48
    axes_dim: List[int] = field(default_factory=lambda: [32, 32, 32, 32])
    theta: int = 2000
    mlp_ratio: float = 3.0
    use_guidance_embed: bool = True


@dataclass
class Klein9BParams:
    """FLUX.2 [klein] Base 9B, the base of Iris-Z-small."""

    in_channels: int = 128
    context_in_dim: int = 12288
    hidden_size: int = 4096
    num_heads: int = 32
    depth: int = 8
    depth_single_blocks: int = 24
    axes_dim: List[int] = field(default_factory=lambda: [32, 32, 32, 32])
    theta: int = 2000
    mlp_ratio: float = 3.0
    use_guidance_embed: bool = False


Mod = Tuple[Tensor, Tensor, Tensor]


class IrisZTransformer(nn.Module):
    """The FLUX.2 transformer as Iris-Z runs it (upstream class ``Flux2``)."""

    def __init__(self, params, num_modalities: int = 0):
        super().__init__()
        self.in_channels = params.in_channels
        self.out_channels = params.in_channels
        if params.hidden_size % params.num_heads != 0:
            raise ValueError(
                f"Hidden size {params.hidden_size} must be divisible by num_heads {params.num_heads}"
            )
        pe_dim = params.hidden_size // params.num_heads
        if sum(params.axes_dim) != pe_dim:
            raise ValueError(f"Got {params.axes_dim} but expected positional dim {pe_dim}")
        self.hidden_size = params.hidden_size
        self.num_heads = params.num_heads
        self.pe_embedder = EmbedND(dim=pe_dim, theta=params.theta, axes_dim=params.axes_dim)
        self.img_in = nn.Linear(self.in_channels, self.hidden_size, bias=False)
        self.time_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size, disable_bias=True)
        self.txt_in = nn.Linear(params.context_in_dim, self.hidden_size, bias=False)

        self.use_guidance_embed = params.use_guidance_embed
        if self.use_guidance_embed:
            self.guidance_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size, disable_bias=True)

        self.double_blocks = nn.ModuleList(
            [
                DoubleStreamBlock(self.hidden_size, self.num_heads, mlp_ratio=params.mlp_ratio)
                for _ in range(params.depth)
            ]
        )
        self.single_blocks = nn.ModuleList(
            [
                SingleStreamBlock(self.hidden_size, self.num_heads, mlp_ratio=params.mlp_ratio)
                for _ in range(params.depth_single_blocks)
            ]
        )
        self.double_stream_modulation_img = Modulation(self.hidden_size, double=True, disable_bias=True)
        self.double_stream_modulation_txt = Modulation(self.hidden_size, double=True, disable_bias=True)
        self.single_stream_modulation = Modulation(self.hidden_size, double=False, disable_bias=True)
        self.final_layer = LastLayer(self.hidden_size, self.out_channels)
        self.modality_in = nn.Embedding(num_modalities, self.hidden_size) if num_modalities else None

    def forward(
        self,
        x: Tensor,
        x_ids: Tensor,
        timesteps: Tensor,
        ctx: Tensor,
        ctx_ids: Tensor,
        guidance: Optional[Tensor],
        modality_ids: Optional[Tensor] = None,
        rgb_timesteps: Optional[Tensor] = None,
        depth_timesteps: Optional[Tensor] = None,
        depth_start: Optional[int] = None,
    ) -> Tensor:
        num_txt_tokens = ctx.shape[1]

        vec = self.time_in(timestep_embedding(timesteps, 256))
        if self.use_guidance_embed:
            vec = vec + self.guidance_in(timestep_embedding(guidance, 256))

        double_block_mod_txt = self.double_stream_modulation_txt(vec)
        per_grid = rgb_timesteps is not None
        if per_grid:
            seq_len = x.shape[1]
            rgb_vec = self.time_in(timestep_embedding(rgb_timesteps, 256))
            depth_vec = self.time_in(timestep_embedding(depth_timesteps, 256))
            double_block_mod_img = _blend_double_mods(
                self.double_stream_modulation_img(depth_vec),
                self.double_stream_modulation_img(rgb_vec),
                depth_start,
                seq_len,
            )
            text_and_rgb_mod = _blend_mod_triple(
                self.single_stream_modulation(rgb_vec)[0],
                self.single_stream_modulation(vec)[0],
                num_txt_tokens,
                num_txt_tokens + depth_start,
            )
            single_block_mod = _blend_mod_triple(
                self.single_stream_modulation(depth_vec)[0],
                text_and_rgb_mod,
                num_txt_tokens + depth_start,
                num_txt_tokens + seq_len,
            )
        else:
            double_block_mod_img = self.double_stream_modulation_img(vec)
            single_block_mod, _ = self.single_stream_modulation(vec)

        img = self.img_in(x)
        if modality_ids is not None:
            img = img + self.modality_in(modality_ids).to(img.dtype)
        txt = self.txt_in(ctx)

        pe_x = self.pe_embedder(x_ids)
        pe_ctx = self.pe_embedder(ctx_ids)

        for block in self.double_blocks:
            img, txt = block(img, txt, pe_x, pe_ctx, double_block_mod_img, double_block_mod_txt)

        img = torch.cat((txt, img), dim=1)
        pe = torch.cat((pe_ctx, pe_x), dim=2)
        for block in self.single_blocks:
            img = block(img, pe, single_block_mod)
        img = img[:, num_txt_tokens:, ...]

        if per_grid:
            return torch.cat(
                [
                    self.final_layer(img[:, :depth_start], rgb_vec),
                    self.final_layer(img[:, depth_start:], depth_vec),
                ],
                dim=1,
            )
        return self.final_layer(img, vec)


def _blend_mod_triple(img_m: tuple, ref_m: tuple, num_ref: int, seq_len: int) -> tuple:
    """Blend a (shift, scale, gate) triple: the first ``num_ref`` positions take ``ref_m``."""
    blended = []
    for im, rm in zip(img_m, ref_m):
        if im.ndim == 2:
            im = im[:, None, :]
            rm = rm[:, None, :]
        batch = im.shape[0]
        blended.append(
            torch.cat(
                [rm.expand(batch, num_ref, -1), im.expand(batch, seq_len, -1)[:, num_ref:, :]],
                dim=1,
            )
        )
    return tuple(blended)


def _blend_double_mods(img_mod, ref_mod, num_ref: int, seq_len: int):
    img_mod1, img_mod2 = img_mod
    ref_mod1, ref_mod2 = ref_mod
    return (
        _blend_mod_triple(img_mod1, ref_mod1, num_ref, seq_len),
        _blend_mod_triple(img_mod2, ref_mod2, num_ref, seq_len),
    )


def _split_heads(qkv: Tensor, num_heads: int) -> Tuple[Tensor, Tensor, Tensor]:
    """``B L (3 H D) -> 3 x (B H L D)``."""
    batch, length, _ = qkv.shape
    qkv = qkv.view(batch, length, 3, num_heads, -1).permute(2, 0, 3, 1, 4)
    return qkv[0], qkv[1], qkv[2]


def _attention(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
    out = F.scaled_dot_product_attention(q, k, v)
    batch, heads, length, dim = out.shape
    return out.transpose(1, 2).reshape(batch, length, heads * dim)


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.norm = QKNorm(head_dim)
        self.proj = nn.Linear(dim, dim, bias=False)


class SiLUActivation(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_fn = nn.SiLU()

    def forward(self, x: Tensor) -> Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return self.gate_fn(x1) * x2


class Modulation(nn.Module):
    def __init__(self, dim: int, double: bool, disable_bias: bool = False):
        super().__init__()
        self.is_double = double
        self.multiplier = 6 if double else 3
        self.lin = nn.Linear(dim, self.multiplier * dim, bias=not disable_bias)

    def forward(self, vec: Tensor):
        out = self.lin(nn.functional.silu(vec))
        if out.ndim == 2:
            out = out[:, None, :]
        out = out.chunk(self.multiplier, dim=-1)
        return out[:3], out[3:] if self.is_double else None


class LastLayer(nn.Module):
    def __init__(self, hidden_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=False)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=False)
        )

    def forward(self, x: Tensor, vec: Tensor) -> Tensor:
        shift, scale = self.adaLN_modulation(vec).chunk(2, dim=-1)
        if shift.ndim == 2:
            shift = shift[:, None, :]
            scale = scale[:, None, :]
        x = (1 + scale) * self.norm_final(x) + shift
        return self.linear(x)


class SingleStreamBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.hidden_dim = hidden_size
        self.num_heads = num_heads
        head_dim = hidden_size // num_heads
        self.scale = head_dim**-0.5
        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp_mult_factor = 2
        self.linear1 = nn.Linear(
            hidden_size,
            hidden_size * 3 + self.mlp_hidden_dim * self.mlp_mult_factor,
            bias=False,
        )
        self.linear2 = nn.Linear(hidden_size + self.mlp_hidden_dim, hidden_size, bias=False)
        self.norm = QKNorm(head_dim)
        self.hidden_size = hidden_size
        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp_act = SiLUActivation()

    def forward(self, x: Tensor, pe: Tensor, mod: Mod) -> Tensor:
        mod_shift, mod_scale, mod_gate = mod
        x_mod = (1 + mod_scale) * self.pre_norm(x) + mod_shift
        qkv, mlp = torch.split(
            self.linear1(x_mod),
            [3 * self.hidden_size, self.mlp_hidden_dim * self.mlp_mult_factor],
            dim=-1,
        )
        q, k, v = _split_heads(qkv, self.num_heads)
        q, k = self.norm(q, k, v)
        q, k = apply_rope(q, k, pe)
        # Upstream concatenates the (empty) reference tokens back in, which
        # leaves v contiguous; SDPA can pick another kernel for a strided v.
        attn = _attention(q, k, v.contiguous())
        output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), 2))
        return x + mod_gate * output


class DoubleStreamBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float):
        super().__init__()
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.num_heads = num_heads
        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size={hidden_size} must be divisible by num_heads={num_heads}")
        self.hidden_size = hidden_size
        self.mlp_mult_factor = 2
        self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_attn = SelfAttention(dim=hidden_size, num_heads=num_heads)
        self.img_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim * self.mlp_mult_factor, bias=False),
            SiLUActivation(),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=False),
        )
        self.txt_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_attn = SelfAttention(dim=hidden_size, num_heads=num_heads)
        self.txt_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim * self.mlp_mult_factor, bias=False),
            SiLUActivation(),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=False),
        )

    def forward(
        self,
        img: Tensor,
        txt: Tensor,
        pe: Tensor,
        pe_ctx: Tensor,
        mod_img: Tuple[Mod, Mod],
        mod_txt: Tuple[Mod, Mod],
    ) -> Tuple[Tensor, Tensor]:
        (img_mod1_shift, img_mod1_scale, img_mod1_gate), (
            img_mod2_shift,
            img_mod2_scale,
            img_mod2_gate,
        ) = mod_img
        (txt_mod1_shift, txt_mod1_scale, txt_mod1_gate), (
            txt_mod2_shift,
            txt_mod2_scale,
            txt_mod2_gate,
        ) = mod_txt

        img_modulated = (1 + img_mod1_scale) * self.img_norm1(img) + img_mod1_shift
        img_q, img_k, img_v = _split_heads(self.img_attn.qkv(img_modulated), self.num_heads)
        img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)

        txt_modulated = (1 + txt_mod1_scale) * self.txt_norm1(txt) + txt_mod1_shift
        txt_q, txt_k, txt_v = _split_heads(self.txt_attn.qkv(txt_modulated), self.num_heads)
        txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k, txt_v)

        q = torch.cat((txt_q, img_q), dim=2)
        k = torch.cat((txt_k, img_k), dim=2)
        v = torch.cat((txt_v, img_v), dim=2)
        num_txt_tokens = txt_q.shape[2]
        q, k = apply_rope(q, k, torch.cat((pe_ctx, pe), dim=2))
        attn = _attention(q, k, v)
        txt_attn, img_attn = attn[:, :num_txt_tokens], attn[:, num_txt_tokens:]

        img = img + img_mod1_gate * self.img_attn.proj(img_attn)
        img = img + img_mod2_gate * self.img_mlp(
            (1 + img_mod2_scale) * self.img_norm2(img) + img_mod2_shift
        )
        txt = txt + txt_mod1_gate * self.txt_attn.proj(txt_attn)
        txt = txt + txt_mod2_gate * self.txt_mlp(
            (1 + txt_mod2_scale) * self.txt_norm2(txt) + txt_mod2_shift
        )
        return img, txt


class MLPEmbedder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, disable_bias: bool = False):
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden_dim, bias=not disable_bias)
        self.silu = nn.SiLU()
        self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=not disable_bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.out_layer(self.silu(self.in_layer(x)))


class EmbedND(nn.Module):
    def __init__(self, dim: int, theta: int, axes_dim: List[int]):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: Tensor) -> Tensor:
        emb = torch.cat(
            [rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(len(self.axes_dim))],
            dim=-3,
        )
        return emb.unsqueeze(1)


def timestep_embedding(t: Tensor, dim: int, max_period: int = 10000, time_factor: float = 1000.0) -> Tensor:
    """Sinusoidal embedding of (possibly fractional) timesteps, one row per element."""
    t = time_factor * t
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, device=t.device, dtype=torch.float32) / half
    )
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    if torch.is_floating_point(t):
        embedding = embedding.to(t)
    return embedding


class RMSNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        x_dtype = x.dtype
        x = x.float()
        rrms = torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + 1e-6)
        return (x * rrms).to(dtype=x_dtype) * self.scale


class QKNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.query_norm = RMSNorm(dim)
        self.key_norm = RMSNorm(dim)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tuple[Tensor, Tensor]:
        return self.query_norm(q).to(v), self.key_norm(k).to(v)


def rope(pos: Tensor, dim: int, theta: int) -> Tensor:
    if dim % 2:
        raise ValueError(f"RoPE axis dim must be even, got {dim}")
    scale = torch.arange(0, dim, 2, dtype=pos.dtype, device=pos.device) / dim
    omega = 1.0 / (theta**scale)
    out = torch.einsum("...n,d->...nd", pos, omega)
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    return out.reshape(*out.shape[:-1], 2, 2).float()


def apply_rope(xq: Tensor, xk: Tensor, freqs_cis: Tensor) -> Tuple[Tensor, Tensor]:
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)
