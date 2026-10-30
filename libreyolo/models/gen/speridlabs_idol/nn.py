# Ported from Speridlabs Idol (https://github.com/speridlabs/idol-sdk, commit
# 4702a7f00850e34b6e83a2170cc123c3e9570744), Apache-2.0, Copyright Speridlabs.
# Idol builds on TripoSplat (https://github.com/VAST-AI-Research/TripoSplat, MIT).
#
# ``LatentSeqMMFlowModel`` is adapted by Idol from diffusers'
# ``transformer_wan.py`` (https://github.com/huggingface/diffusers):
#
# Copyright 2025 The Wan Team and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Speridlabs Idol networks: the denoiser and the two decoders.

Module and parameter names mirror upstream so the released safetensors load
with ``strict=True``, and every forward keeps upstream's operation order and
precision casts so outputs match it bit for bit. Upstream's TensorDict inputs
are plain tensors here, and attention always runs through PyTorch SDPA
(upstream's default when flash-attn is not installed).
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# "sdpa" lets PyTorch pick the kernel; "sdpa_cudnn" pins the cuDNN kernel,
# which compiled sampling selects for the denoiser (see ``cudnn_attention``).
_ATTENTION_BACKEND = "sdpa"


@contextmanager
def cudnn_attention():
    """Run attention on the cuDNN SDPA kernel inside the block."""
    global _ATTENTION_BACKEND
    previous = _ATTENTION_BACKEND
    _ATTENTION_BACKEND = "sdpa_cudnn"
    try:
        yield
    finally:
        _ATTENTION_BACKEND = previous


def scaled_dot_product_attention(*args) -> torch.Tensor:
    """(qkv [N, L, 3, H, C]) or (q [N, L, H, C], k [N, S, H, C], v [N, S, H, C]) -> [N, L, H, C]."""
    q, k, v = args[0].unbind(dim=2) if len(args) == 1 else args
    q, k, v = q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3)
    if _ATTENTION_BACKEND == "sdpa_cudnn":
        from torch.nn.attention import SDPBackend, sdpa_kernel

        with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
            out = F.scaled_dot_product_attention(q, k, v)
    else:
        out = F.scaled_dot_product_attention(q, k, v)
    return out.permute(0, 2, 1, 3)


def convert_module_to_f16(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        for parameter in module.parameters():
            parameter.data = parameter.data.half()


class LayerNorm32(nn.LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.float()).type(x.dtype)


class MultiHeadRMSNorm(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(heads, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (F.normalize(x.float(), dim=-1) * self.gamma * self.scale).to(x.dtype)


class FeedForwardNet(nn.Module):
    def __init__(self, channels: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(channels, int(channels * mlp_ratio)),
            nn.GELU(approximate="tanh"),
            nn.Linear(int(channels * mlp_ratio), channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class MLP(nn.Module):
    def __init__(
        self,
        channels: int,
        inner_channels: int,
        channels_out: int,
        mlp_layer_num: int = 2,
    ):
        super().__init__()
        layers = []
        for i in range(mlp_layer_num - 1):
            layers.append(
                nn.Linear(channels if i == 0 else inner_channels, inner_channels)
            )
            layers.append(nn.GELU(approximate="tanh"))
        layers.append(nn.Linear(inner_channels, channels_out))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


# -- Positional and scalar embeddings ----------------------------------------


class PcdAbsolutePositionEmbedder(nn.Module):
    """Sinusoidal embedding of 3D positions in [0, 1] (the denoiser's latent token positions)."""

    def __init__(self, channels: int, in_channels: int = 3, max_res: int = 16):
        super().__init__()
        self.channels = channels
        self.in_channels = in_channels
        self.freq_dim = channels // in_channels // 2
        freqs_2exp = torch.arange(max_res, dtype=torch.float32)
        res_dim = max(0, self.freq_dim - max_res)
        freqs_res = torch.arange(res_dim, dtype=torch.float32) / res_dim * max_res
        self.freqs = 2 ** torch.cat([freqs_2exp, freqs_res], dim=0)[: self.freq_dim]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        *dims, _ = x.shape
        self.freqs = self.freqs.to(x.device)
        out = torch.outer(x.reshape(-1), self.freqs) * 2 * torch.pi
        embed = torch.cat([torch.sin(out), torch.cos(out)], dim=-1).reshape(*dims, -1)
        if embed.shape[-1] < self.channels:
            embed = torch.cat(
                [
                    embed,
                    torch.zeros(
                        *dims, self.channels - embed.shape[-1], device=embed.device
                    ),
                ],
                dim=-1,
            )
        return embed


class PcdAbsolutePositionEmbedderv2(nn.Module):
    """Sinusoidal embedding of 3D points in [0, 1] (the decoders' query points)."""

    def __init__(self, channels: int, in_channels: int = 3, max_res: int = 10):
        super().__init__()
        self.channels = channels
        self.freq_dim = channels // in_channels // 2
        logs = torch.linspace(
            0.0, float(max_res), steps=self.freq_dim, dtype=torch.float32
        )
        self.register_buffer("freqs", torch.pow(2.0, logs), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N = x.shape[0]
        freqs = self.freqs.to(x.device, dtype=x.dtype)
        ang = x.unsqueeze(-1) * freqs * torch.pi
        embed = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1).reshape(N, -1)
        if embed.shape[1] < self.channels:
            embed = torch.cat(
                [
                    embed,
                    torch.zeros(
                        N,
                        self.channels - embed.shape[1],
                        device=embed.device,
                        dtype=embed.dtype,
                    ),
                ],
                dim=-1,
            )
        return embed


class SinusoidalEmbedder(nn.Module):
    """Embeds a scalar per batch element (diffusion time, octree level) into a vector."""

    def __init__(
        self, hidden_size, frequency_embedding_size=256, max_period=10000, two_pi=False
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
        self.max_period = max_period
        self.two_pi = two_pi

    def forward(self, t):
        half = self.frequency_embedding_size // 2
        freqs = torch.exp(
            -np.log(self.max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        if self.two_pi:
            args = args * 2 * torch.pi
        return self.mlp(torch.cat([torch.cos(args), torch.sin(args)], dim=-1))


def apply_rotary_emb(hidden_states: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """hidden_states [B, L, H, D] rotated by the complex frequencies freqs [B, L, H or 1, D/2], or by the same
    rotation as real (cos, sin) pairs [B, L, H, D/2, 2]."""
    if not freqs.is_complex():
        # the complex product written out, so torch.compile can fuse it
        x0, x1 = (
            hidden_states.float().reshape(*hidden_states.shape[:-1], -1, 2).unbind(-1)
        )
        cos, sin = freqs.unbind(-1)
        return (
            torch.stack([x0 * cos - x1 * sin, x0 * sin + x1 * cos], dim=-1)
            .flatten(-2)
            .type_as(hidden_states)
        )
    x_rotated = torch.view_as_complex(
        hidden_states.float().reshape(*hidden_states.shape[:-1], -1, 2)
    )
    x_rotated = x_rotated * freqs
    x_out = torch.view_as_real(x_rotated).reshape(*x_rotated.shape[:-1], -1)
    return x_out.type_as(hidden_states)


def clamp_mul(x, f):
    f_t = f.tanh()
    return x * f_t + x.detach() * (f - f_t)


class RePo3DRotaryEmbedding(nn.Module):
    """Learned rotary position: each token's features predict a 3D offset per head, turned into RoPE angles."""

    # True: return (cos, sin) pairs instead of a complex tensor, which
    # torch.compile cannot fuse (compiled sampling sets it).
    real_valued = False

    def __init__(
        self,
        model_channels: int,
        num_heads: int,
        head_dim: int,
        repo_hidden_ratio: float = 0.125,
        max_freq: float = 16.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        repo_hidden_size = int(model_channels * repo_hidden_ratio)
        self.norm = LayerNorm32(model_channels)
        self.gate_map = nn.Linear(model_channels, repo_hidden_size, bias=False)
        self.content_map = nn.Linear(model_channels, repo_hidden_size, bias=False)
        self.act = nn.SiLU()
        self.final_map = nn.Linear(repo_hidden_size, 3 * num_heads, bias=False)
        # head_dim split into 3 parts for x, y, z
        dim_0 = dim_1 = 2 * (head_dim // 6)
        dims = [dim_0, dim_1, head_dim - dim_0 - dim_1]
        freqs = [
            torch.linspace(1.0, float(max_freq), steps=d // 2, dtype=torch.float32)
            for d in dims
        ]
        self.freqs_0 = nn.Parameter(freqs[0])
        self.freqs_1 = nn.Parameter(freqs[1])
        self.freqs_2 = nn.Parameter(freqs[2])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        h = self.norm(hidden_states.float())
        out = self.final_map(self.act(self.gate_map(h)) * self.content_map(h))
        B, L, _ = out.shape
        delta_pos = out.reshape(B, L, self.num_heads, 3)
        ang_0 = clamp_mul(delta_pos[..., 0].unsqueeze(-1), self.freqs_0) * torch.pi
        ang_1 = clamp_mul(delta_pos[..., 1].unsqueeze(-1), self.freqs_1) * torch.pi
        ang_2 = clamp_mul(delta_pos[..., 2].unsqueeze(-1), self.freqs_2) * torch.pi
        ang = torch.cat([ang_0, ang_1, ang_2], dim=-1)
        if self.real_valued:
            return torch.stack([ang.cos(), ang.sin()], dim=-1)
        return torch.polar(torch.ones_like(ang), ang).type(torch.complex64)


# -- Attention ---------------------------------------------------------------


class MultiHeadAttention(nn.Module):
    """Self or cross attention with RMS-normalised queries and keys (the decoders)."""

    def __init__(
        self,
        channels: int,
        num_heads: int,
        ctx_channels: Optional[int] = None,
        cross: bool = False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.cross = cross
        head_dim = channels // num_heads
        if cross:
            self.to_q = nn.Linear(channels, channels)
            self.to_kv = nn.Linear(ctx_channels, channels * 2)
        else:
            self.to_qkv = nn.Linear(channels, channels * 3)
        self.q_rms_norm = MultiHeadRMSNorm(head_dim, num_heads)
        self.k_rms_norm = MultiHeadRMSNorm(head_dim, num_heads)
        self.to_out = nn.Linear(channels, channels)

    def forward(
        self, x: torch.Tensor, context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, L, C = x.shape
        if self.cross:
            q = self.q_rms_norm(self.to_q(x).reshape(B, L, self.num_heads, -1))
            k, v = (
                self.to_kv(context)
                .reshape(B, context.shape[1], 2, self.num_heads, -1)
                .unbind(dim=2)
            )
            h = scaled_dot_product_attention(q, self.k_rms_norm(k), v)
        else:
            q, k, v = self.to_qkv(x).reshape(B, L, 3, self.num_heads, -1).unbind(dim=2)
            h = scaled_dot_product_attention(
                torch.stack([self.q_rms_norm(q), self.k_rms_norm(k), v], dim=2)
            )
        return self.to_out(h.reshape(B, L, -1))


class RopeMultiHeadAttention(nn.Module):
    """Self attention with rotary embeddings and RMS-normalised queries and keys (the denoiser)."""

    def __init__(self, channels: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.qkv = nn.Linear(channels, channels * 3)
        self.q_norm = MultiHeadRMSNorm(self.head_dim, num_heads)
        self.k_norm = MultiHeadRMSNorm(self.head_dim, num_heads)
        self.out = nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor, rope_emb: torch.Tensor) -> torch.Tensor:
        B, L, C = x.shape
        q, k, v = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim).unbind(2)
        q = self.q_norm(apply_rotary_emb(q, rope_emb))
        k = self.k_norm(apply_rotary_emb(k, rope_emb))
        return self.out(scaled_dot_product_attention(q, k, v).reshape(B, L, C))


# -- Transformer blocks ------------------------------------------------------


class UnifiedTransformerBlock(nn.Module):
    """Denoiser block: RoPE self attention + FFN, modulated by the diffusion time through a shared adaLN plus
    this block's own shift table (modulation=False for the condition refiner, which has plain LayerNorms)."""

    def __init__(
        self,
        channels: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        modulation: bool = True,
    ):
        super().__init__()
        self.modulation = modulation
        self.norm1 = LayerNorm32(channels, elementwise_affine=not modulation, eps=1e-6)
        self.norm2 = LayerNorm32(channels, elementwise_affine=not modulation, eps=1e-6)
        self.attn = RopeMultiHeadAttention(channels, num_heads=num_heads)
        self.mlp = FeedForwardNet(channels, mlp_ratio=mlp_ratio)
        if modulation:
            self.shift_table = nn.Parameter(torch.zeros(1, 6 * channels))

    def forward(
        self, x: torch.Tensor, mod: Optional[torch.Tensor], rotary_emb: torch.Tensor
    ) -> torch.Tensor:
        if not self.modulation:
            x = x + self.attn(self.norm1(x), rope_emb=rotary_emb)
            return x + self.mlp(self.norm2(x))
        mod = mod + self.shift_table.type(mod.dtype)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(
            6, dim=1
        )
        h = self.norm1(x)
        h = h * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        h = self.attn(h, rope_emb=rotary_emb)
        x = x + h * gate_msa.unsqueeze(1)
        h = self.norm2(x)
        h = h * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        h = self.mlp(h)
        return x + h * gate_mlp.unsqueeze(1)


class TransformerCrossBlock(nn.Module):
    """Gaussian decoder block: self attention + cross attention to the latent + FFN."""

    def __init__(
        self, channels: int, ctx_channels: int, num_heads: int, mlp_ratio: float = 4.0
    ):
        super().__init__()
        self.norm1 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.norm2 = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        self.norm3 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.self_attn = MultiHeadAttention(channels, num_heads=num_heads)
        self.cross_attn = MultiHeadAttention(
            channels, num_heads=num_heads, ctx_channels=ctx_channels, cross=True
        )
        self.mlp = FeedForwardNet(channels, mlp_ratio=mlp_ratio)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.norm1(x))
        x = x + self.cross_attn(self.norm2(x), context)
        return x + self.mlp(self.norm3(x))


class ModulatedTransformerCrossOnlyBlock(nn.Module):
    """Octree decoder block: cross attention to the latent + FFN, modulated by the octree level (shared adaLN)."""

    def __init__(
        self, channels: int, ctx_channels: int, num_heads: int, mlp_ratio: float = 4.0
    ):
        super().__init__()
        self.norm1 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.norm2 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.cross_attn = MultiHeadAttention(
            channels, num_heads=num_heads, ctx_channels=ctx_channels, cross=True
        )
        self.mlp = FeedForwardNet(channels, mlp_ratio=mlp_ratio)

    def forward(
        self, x: torch.Tensor, mod: torch.Tensor, context: torch.Tensor
    ) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(
            6, dim=1
        )
        h = self.norm1(x)
        h = h * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        h = self.cross_attn(h, context)
        x = x + h * gate_msa.unsqueeze(1)
        h = self.norm2(x)
        h = h * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        h = self.mlp(h)
        return x + h * gate_mlp.unsqueeze(1)


# -- Denoiser ----------------------------------------------------------------


class LatentSeqMMFlowModel(nn.Module):
    """The denoiser: a rectified-flow transformer over the object latent (8192 tokens) and the input camera
    (1 token), conditioned on the image tokens. Joint attention over [latent, image, camera] with learned
    rotary positions."""

    def __init__(
        self,
        q_token_length: int,
        in_channels: int,
        cam_channels: int,
        model_channels: int,
        cond_channels: int,
        cond2_channels: int,
        num_blocks: int,
        num_refiner_blocks: int = 2,
        num_heads: int = 16,
        num_head_channels: int = 64,
        mlp_ratio: float = 4,
    ):
        super().__init__()
        self.q_token_length = q_token_length
        self.in_channels = in_channels
        self.cam_channels = cam_channels
        self.dtype = torch.float16

        def blocks(count, modulation=True):
            return nn.ModuleList(
                [
                    UnifiedTransformerBlock(
                        model_channels,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        modulation=modulation,
                    )
                    for _ in range(count)
                ]
            )

        def rope_layers(count):
            return nn.ModuleList(
                [
                    RePo3DRotaryEmbedding(
                        model_channels, num_heads=num_heads, head_dim=num_head_channels
                    )
                    for _ in range(count)
                ]
            )

        self.t_embedder = SinusoidalEmbedder(model_channels)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(model_channels, 6 * model_channels, bias=True)
        )
        self.input_layer = nn.Linear(in_channels, model_channels)
        self.cond_embedder = nn.Linear(cond_channels, model_channels)
        self.cond_embedder2 = nn.Linear(cond2_channels, model_channels)

        # positions of the latent tokens (a fixed scrambled Sobol sequence, stored with the weights)
        self.register_buffer("pos_pe", torch.zeros(1, q_token_length, 3))
        self.pos_embedder = PcdAbsolutePositionEmbedder(model_channels)
        self.noise_repo_layers = rope_layers(num_refiner_blocks)
        self.context_repo_layers = rope_layers(num_refiner_blocks)
        self.repo_layers = rope_layers(num_blocks)

        # each stream is refined on its own, then all tokens go through the joint blocks
        self.noise_refiner = blocks(num_refiner_blocks)
        self.context_refiner = blocks(num_refiner_blocks, modulation=False)
        self.cam_refiner = MLP(
            cam_channels,
            model_channels,
            model_channels,
            mlp_layer_num=num_refiner_blocks,
        )
        self.blocks = blocks(num_blocks)

        self.shift_table = nn.Parameter(torch.zeros(1, 2, model_channels))
        self.out_layer = nn.Linear(model_channels, in_channels)
        self.cam_out_layer = nn.Linear(model_channels, cam_channels)

        self.blocks.apply(convert_module_to_f16)
        self.noise_refiner.apply(convert_module_to_f16)
        self.context_refiner.apply(convert_module_to_f16)

    def forward(
        self,
        latent: torch.Tensor,
        camera: torch.Tensor,
        t: torch.Tensor,
        cond: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Velocity of the object latent [B, 8192, 16] and camera [B, 1, 5] at time t [B] in [0, 1000], given the
        image tokens cond {'feature1': [B, L_c, 1280] DINOv3, 'feature2': [B, L_c, 128] FLUX.2 VAE}."""
        z, cam = latent, camera
        h_x = self.input_layer(z).type(self.dtype)
        h_cond = self.cond_embedder(cond["feature1"]) + self.cond_embedder2(
            cond["feature2"]
        )
        h_cond = h_cond.type(self.dtype)

        t_emb = self.t_embedder(t)
        t_mod = self.adaLN_modulation(t_emb).type(self.dtype)

        h_x = h_x + self.pos_embedder(self.pos_pe).type(self.dtype)
        for repo, block in zip(self.noise_repo_layers, self.noise_refiner):
            h_x = block(h_x, mod=t_mod, rotary_emb=repo(h_x))
        for repo, block in zip(self.context_repo_layers, self.context_refiner):
            h_cond = block(h_cond, mod=None, rotary_emb=repo(h_cond))
        h_cam = self.cam_refiner(cam).type(self.dtype)

        h = torch.cat([h_x, h_cond, h_cam], dim=1)
        for repo, block in zip(self.repo_layers, self.blocks):
            h = block(h, mod=t_mod, rotary_emb=repo(h))

        h_x = h[:, : z.shape[1]].type(z.dtype)
        h_x = F.layer_norm(h_x, h_x.shape[-1:])
        h_cam = h[:, -cam.shape[1] :].type(cam.dtype)
        h_cam = F.layer_norm(h_cam, h_cam.shape[-1:])
        shift, scale = (self.shift_table + t_emb.unsqueeze(1)).chunk(2, dim=1)
        h_x = h_x * (1 + scale) + shift
        h_cam = h_cam * (1 + scale) + shift
        return self.out_layer(h_x), self.cam_out_layer(h_cam)


def compile_for_sampling(denoiser: LatentSeqMMFlowModel) -> None:
    """torch.compile the denoiser's transformer blocks and RePo layers, once per class and bound to this model's
    instances only (the decoders stay eager), and switch the RePo layers to real-valued RoPE."""
    import types

    compiled = {}
    for module in denoiser.modules():
        if isinstance(module, RePo3DRotaryEmbedding):
            module.real_valued = True
        if isinstance(module, (UnifiedTransformerBlock, RePo3DRotaryEmbedding)):
            if type(module) not in compiled:
                compiled[type(module)] = torch.compile(
                    type(module).forward, dynamic=False
                )
            module.forward = types.MethodType(compiled[type(module)], module)


# -- Decoders ----------------------------------------------------------------


class OctreeProbabilityFixedlenDecoder(nn.Module):
    def __init__(
        self,
        model_channels: int,
        cond_channels: int,
        num_blocks: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.dtype = torch.float16
        self.in_proj = nn.Linear(3, model_channels)
        self.pos_embedder = PcdAbsolutePositionEmbedderv2(
            channels=model_channels, in_channels=3
        )
        self.input_layer = nn.Linear(model_channels, model_channels)
        self.l_embedder = SinusoidalEmbedder(
            model_channels, max_period=1024, two_pi=True
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(model_channels, 6 * model_channels, bias=True)
        )
        self.blocks = nn.ModuleList(
            [
                ModulatedTransformerCrossOnlyBlock(
                    model_channels,
                    ctx_channels=cond_channels,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                )
                for _ in range(num_blocks)
            ]
        )
        self.out_proj = nn.Linear(model_channels, 2**3)
        self.blocks.apply(convert_module_to_f16)

    def forward(
        self, x: torch.Tensor, level: torch.Tensor, cond: torch.Tensor
    ) -> torch.Tensor:
        """x: cell centres [B, L, 3] in [0, 1]; level: the cells' octree resolution [B]; cond: latent [B, Lc, C].
        Returns the logits [B, L, 8] of how each cell's points split among its 8 children."""
        B, L, _ = x.shape
        h = self.in_proj(x) + self.pos_embedder(x.view(-1, 3)).view(B, L, -1)
        h = self.input_layer(h)
        level_emb = self.adaLN_modulation(self.l_embedder(level.type(h.dtype))).type(
            self.dtype
        )
        out = h.type(self.dtype)
        context = cond.type(self.dtype)
        for block in self.blocks:
            out = block(out, level_emb, context)
        out = out.type(h.dtype).type(cond.dtype)
        return self.out_proj(F.layer_norm(out, out.shape[-1:]))

    @torch.no_grad()
    def sample(
        self,
        cond: torch.Tensor,
        num_points: int,
        level: int,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Place num_points anchor points [B, num_points, 3] in [0, 1]^3, descending the octree to ``level``."""
        B = cond.shape[0]
        device = cond.device
        child_offset = torch.tensor(
            [[i, j, k] for k in [0, 1] for j in [0, 1] for i in [0, 1]],
            dtype=torch.long,
            device=device,
        )
        prev_coords_int = torch.zeros(B, 1, 3, dtype=torch.long, device=device)
        prev_counts = torch.full((B, 1), num_points, dtype=torch.long, device=device)
        batch_indices_range = torch.arange(B, device=device).unsqueeze(1)

        for l in range(1, level + 1):
            res_p = 1 << (l - 1)
            parent_coords_norm = (prev_coords_int.to(torch.float32) + 0.5) / res_p
            res_tensor = torch.full((B,), 1 << l, dtype=torch.long, device=device)
            pred_probs = torch.softmax(
                self(parent_coords_norm, res_tensor, cond), dim=-1
            )  # [B, Np, 8]
            sampled = systematic_sample(pred_probs, prev_counts, generator).flatten(
                1, 2
            )  # [B, Np*8]
            child_coords_int = (
                prev_coords_int[:, :, None, :] * 2 + child_offset[None, None, :, :]
            ).flatten(1, 2)

            # keep the children that received points, packed to the front
            mask = sampled > 0
            max_valid = mask.sum(dim=1).max().item()
            valid_scatter_indices = (mask.cumsum(dim=1) - 1)[mask]
            valid_batch_indices = batch_indices_range.expand_as(mask)[mask]
            prev_coords_int = torch.zeros(
                B, max_valid, 3, dtype=child_coords_int.dtype, device=device
            )
            prev_coords_int[valid_batch_indices, valid_scatter_indices] = (
                child_coords_int[mask]
            )
            prev_counts = torch.zeros(B, max_valid, dtype=sampled.dtype, device=device)
            prev_counts[valid_batch_indices, valid_scatter_indices] = sampled[mask]

        # each point lands at a uniformly random position inside its leaf cell
        coords_int = torch.repeat_interleave(
            prev_coords_int.flatten(0, 1), prev_counts.flatten(0, 1), dim=0
        ).reshape(B, num_points, -1)
        jitter = torch.rand(
            coords_int.shape, generator=generator, dtype=torch.float32, device=device
        )
        return (coords_int.to(torch.float32) + jitter) / (1 << level)


@torch.no_grad()
def systematic_sample(
    probs: torch.Tensor,
    counts: torch.Tensor,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Split counts[...] items among the P bins of probs[..., P] by systematic resampling -> [..., P] counts."""
    batch_shape = counts.shape
    B = counts.numel()
    P = probs.size(-1)
    device = probs.device
    probs = probs.view(B, P)
    counts = counts.view(B)

    # normalise, with a uniform fallback for zero-sum rows
    probs = probs.to(torch.float32).clamp_min_(0)
    row_sums = probs.sum(1, keepdim=True)
    zero_mask = row_sums.eq(0)
    probs = probs / row_sums.clamp_min_(1)
    if zero_mask.any():
        probs = probs.clone()
        probs[zero_mask.expand_as(probs)] = 1.0 / P

    counts = counts.to(device=device, dtype=torch.long)
    out = torch.zeros(B, P, dtype=torch.long, device=device)
    cdf = probs.cumsum(dim=1).clamp(max=1.0 - 1e-12)
    unique_n, inv = counts.unique(sorted=False, return_inverse=True)
    for index, n in enumerate(unique_n.tolist()):
        if n == 0:
            continue
        rows = (inv == index).nonzero(as_tuple=False).squeeze(1)
        r = rows.numel()
        u0 = torch.rand(r, 1, generator=generator, device=device) / float(
            n
        )  # random offset in [0, 1/n)
        grid = torch.arange(n, device=device, dtype=torch.float32)[None, :] / float(
            n
        )  # stratified points
        us = (u0 + grid).clamp(max=1.0 - 1e-12)
        idx = torch.searchsorted(cdf.index_select(0, rows), us).clamp_max(P - 1)
        buf = torch.zeros(r, P, dtype=torch.float32, device=device)
        buf.scatter_add_(1, idx, torch.ones_like(idx, dtype=buf.dtype))
        out.index_copy_(0, rows, buf.to(torch.long))
    return out.view(*batch_shape, P)


def inverse_sigmoid(x: torch.Tensor) -> torch.Tensor:
    return torch.log(x / (1 - x))


def inverse_softplus(x: torch.Tensor) -> torch.Tensor:
    return x + torch.log(-torch.expm1(-x))


class GaussianFixedlenDecoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        model_channels: int,
        cond_channels: int,
        num_blocks: int,
        num_heads: int,
        representation_config: dict,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.dtype = torch.float16
        self.rep_config = representation_config
        per_anchor = representation_config["num_gaussians"]
        # output channels per anchor: offset (3), colour (3), scale (3), rotation (4), opacity (1), offset scale (1)
        self.layout = {
            "_xyz": (3,),
            "_features_dc": (1, 3),
            "_scaling": (3,),
            "_rotation": (4,),
            "_opacity": (1,),
            "_offset_scale": (1,),
        }
        start = 0
        self.ranges = {}
        for name, shape in self.layout.items():
            size = per_anchor * int(torch.tensor(shape).prod())
            self.ranges[name] = (start, start + size)
            start += size

        self.in_proj = nn.Linear(in_channels, model_channels)
        self.pos_embedder = PcdAbsolutePositionEmbedderv2(
            channels=model_channels, in_channels=3
        )
        self.input_layer = nn.Linear(model_channels, model_channels)
        self.blocks = nn.ModuleList(
            [
                TransformerCrossBlock(
                    model_channels,
                    ctx_channels=cond_channels,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                )
                for _ in range(num_blocks)
            ]
        )
        self.out_proj = nn.Linear(model_channels, start)
        # fixed per-Gaussian offset pattern and the base offset scale, stored with the weights
        self.register_buffer("points_offset_perturbation", torch.zeros(per_anchor, 3))
        self.register_buffer("base_offset_scale", torch.zeros(()))
        self.blocks.apply(convert_module_to_f16)

    def forward(self, points: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """points: anchors [B, L, 3]; cond: latent [B, Lc, C] -> per-anchor outputs [B, L, 32 x 15]"""
        B, L, C = points.shape
        h = self.in_proj(points) + self.pos_embedder(points.view(-1, C)).view(B, L, -1)
        h = self.input_layer(h)
        out = h.type(self.dtype)
        context = cond.type(self.dtype)
        for block in self.blocks:
            out = block(out, context)
        out = out.type(h.dtype).type(cond.dtype)
        return self.out_proj(F.layer_norm(out, out.shape[-1:]))

    def _slice(self, features, name):
        start, end = self.ranges[name]
        return features[:, :, start:end].reshape(
            features.shape[0], -1, self.rep_config["num_gaussians"], *self.layout[name]
        )

    def to_gaussians(
        self, points: torch.Tensor, features: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Activated Gaussians of the first object in the batch: xyz [N, 3] (z-up, inside [-0.5, 0.5]^3),
        features_dc [N, 1, 3], scaling [N, 3], rotation [N, 4] unit (w, x, y, z), opacity [N, 1]."""
        config = self.rep_config
        offset_scale = F.softplus(
            self._slice(features, "_offset_scale") + self.base_offset_scale
        )
        offset = (
            self._slice(features, "_xyz") * config["lr"]["_xyz"]
            + self.points_offset_perturbation
        )
        offset = torch.tanh(offset) * 0.5 * config["perturbe_size"] * offset_scale

        raw = {
            name: (
                self._slice(features[0:1], name)[0].flatten(0, 1) * config["lr"][name]
            )
            for name in ("_features_dc", "_scaling", "_rotation", "_opacity")
        }
        raw_xyz = (offset[0] + points[0, :, None, :]).flatten(0, 1)
        device = raw_xyz.device
        aabb = torch.tensor(
            [-0.5, -0.5, -0.5, 1.0, 1.0, 1.0], dtype=torch.float32, device=device
        )
        scaling_offset = inverse_softplus(torch.tensor(config["scaling_bias"])).to(
            device
        )
        opacity_offset = inverse_sigmoid(torch.tensor(config["opacity_bias"])).to(
            device
        )
        rotation_offset = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)

        scaling = F.softplus(raw["_scaling"] + scaling_offset)
        kernel = config["3d_filter_kernel_size"]
        return {
            "xyz": raw_xyz * aabb[None, 3:] + aabb[None, :3],
            "features_dc": raw["_features_dc"],
            "scaling": torch.sqrt(torch.square(scaling) + kernel**2),
            "rotation": F.normalize(raw["_rotation"] + rotation_offset[None, :]),
            "opacity": torch.sigmoid(raw["_opacity"] + opacity_offset),
        }
