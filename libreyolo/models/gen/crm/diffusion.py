# Adapted from CRM (https://github.com/thu-ml/CRM, commit
# 4964e36a593070a3045eb0f300935a771ff5c172, MIT, Copyright (c) 2024 TSAIL
# group), whose imagedream/ package derives from ImageDream
# (https://github.com/bytedance/ImageDream, Apache-2.0) and MVDream
# (https://github.com/bytedance/MVDream, MIT), which build on the latent
# diffusion code of Stable Diffusion (MIT). The image-prompt resampler is
# modified from OpenFlamingo (MIT); the text and image towers follow OpenCLIP
# (https://github.com/mlfoundations/open_clip, MIT). See NOTICE.
"""CRM's two multi-view diffusion models, in plain PyTorch.

Both are ImageDream-style UNets: Stable Diffusion 2.1's latent UNet whose
self-attention runs across all views of an object at once, a camera
embedding per view, and a 16-token image prompt (an OpenCLIP ViT-H/14 image
embedding through a perceiver resampler) that every cross-attention reads
next to the text. The first model generates six views of the object plus the
input view; the second turns those six views into canonical coordinate maps
(CCMs), with the six view latents as four extra input channels.

Module and parameter names follow upstream, so the released ``.pth`` files
load unchanged. Attention runs through
``torch.nn.functional.scaled_dot_product_attention`` instead of xformers.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

# -- Shared layers --------------------------------------------------------------


class GroupNorm32(nn.GroupNorm):
    """GroupNorm computed in fp32 and cast back, as the UNet's ``normalization``."""

    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


def _attention(q, k, v, heads: int, causal: bool = False):
    """[B, L, heads*d] -> scaled dot-product attention -> [B, L, heads*d]."""
    b, lq, inner = q.shape
    d = inner // heads
    q, k, v = (t.reshape(b, t.shape[1], heads, d).transpose(1, 2) for t in (q, k, v))
    out = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
    return out.transpose(1, 2).reshape(b, lq, inner)


def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000):
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(0, half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


# -- Multi-view UNet ------------------------------------------------------------


class TimestepEmbedSequential(nn.Sequential):
    def forward(self, x, emb, context=None, num_frames=1):
        for layer in self:
            if isinstance(layer, ResBlock):
                x = layer(x, emb)
            elif isinstance(layer, SpatialTransformer3D):
                x = layer(x, context, num_frames=num_frames)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    def __init__(self, channels: int, out_channels: int | None = None):
        super().__init__()
        self.conv = nn.Conv2d(channels, out_channels or channels, 3, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))


class Downsample(nn.Module):
    def __init__(self, channels: int, out_channels: int | None = None):
        super().__init__()
        self.op = nn.Conv2d(channels, out_channels or channels, 3, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)


class ResBlock(nn.Module):
    def __init__(self, channels: int, emb_channels: int, out_channels: int | None = None):
        super().__init__()
        out_channels = out_channels or channels
        self.in_layers = nn.Sequential(
            GroupNorm32(32, channels), nn.SiLU(), nn.Conv2d(channels, out_channels, 3, padding=1)
        )
        self.emb_layers = nn.Sequential(nn.SiLU(), nn.Linear(emb_channels, out_channels))
        self.out_layers = nn.Sequential(
            GroupNorm32(32, out_channels),
            nn.SiLU(),
            nn.Dropout(0.0),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )
        self.skip_connection = (
            nn.Identity() if out_channels == channels else nn.Conv2d(channels, out_channels, 1)
        )

    def forward(self, x, emb):
        h = self.in_layers(x)
        h = h + self.emb_layers(emb).type(h.dtype)[..., None, None]
        return self.skip_connection(x) + self.out_layers(h)


class CrossAttention(nn.Module):
    """Attention with an optional second key/value set for the image-prompt tokens."""

    def __init__(
        self,
        query_dim: int,
        context_dim: int | None,
        heads: int,
        dim_head: int,
        with_ip: bool = False,
        ip_dim: int = 16,
    ):
        super().__init__()
        inner = heads * dim_head
        context_dim = context_dim or query_dim
        self.heads = heads
        self.with_ip = with_ip
        self.ip_dim = ip_dim
        if with_ip:
            self.to_k_ip = nn.Linear(context_dim, inner, bias=False)
            self.to_v_ip = nn.Linear(context_dim, inner, bias=False)
        self.to_q = nn.Linear(query_dim, inner, bias=False)
        self.to_k = nn.Linear(context_dim, inner, bias=False)
        self.to_v = nn.Linear(context_dim, inner, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner, query_dim), nn.Dropout(0.0))

    def forward(self, x, context=None):
        q = self.to_q(x)
        has_ip = self.with_ip and context is not None
        if has_ip:
            context_ip = context[:, -self.ip_dim :]
            k_ip, v_ip = self.to_k_ip(context_ip), self.to_v_ip(context_ip)
            context = context[:, : context.shape[1] - self.ip_dim]
        context = x if context is None else context
        out = _attention(q, self.to_k(context), self.to_v(context), self.heads)
        if has_ip:
            out = out + _attention(q, k_ip, v_ip, self.heads)
        return self.to_out(out)


class GEGLU(nn.Module):
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim: int, mult: int = 4):
        super().__init__()
        self.net = nn.Sequential(GEGLU(dim, dim * mult), nn.Dropout(0.0), nn.Linear(dim * mult, dim))

    def forward(self, x):
        return self.net(x)


class BasicTransformerBlock3D(nn.Module):
    """Self-attention over the tokens of all views together, then per-view cross-attention."""

    def __init__(self, dim: int, heads: int, dim_head: int, context_dim: int, ip_dim: int):
        super().__init__()
        self.attn1 = CrossAttention(dim, None, heads, dim_head)
        self.ff = FeedForward(dim)
        self.attn2 = CrossAttention(dim, context_dim, heads, dim_head, with_ip=True, ip_dim=ip_dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

    def forward(self, x, context, num_frames: int):
        bf, length, channels = x.shape
        x = x.reshape(bf // num_frames, num_frames * length, channels)
        x = self.attn1(self.norm1(x)) + x
        x = x.reshape(bf, length, channels)
        x = self.attn2(self.norm2(x), context=context) + x
        return self.ff(self.norm3(x)) + x


class SpatialTransformer3D(nn.Module):
    def __init__(self, channels: int, heads: int, dim_head: int, context_dim: int, ip_dim: int):
        super().__init__()
        inner = heads * dim_head
        self.norm = nn.GroupNorm(32, channels, eps=1e-6)
        self.proj_in = nn.Linear(channels, inner)
        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock3D(inner, heads, dim_head, context_dim, ip_dim)]
        )
        self.proj_out = nn.Linear(channels, inner)

    def forward(self, x, context, num_frames: int):
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x).flatten(2).transpose(1, 2)
        x = self.proj_in(x)
        for block in self.transformer_blocks:
            x = block(x, context, num_frames)
        x = self.proj_out(x)
        return x.transpose(1, 2).reshape(b, c, h, w) + x_in


class PerceiverAttention(nn.Module):
    def __init__(self, dim: int, dim_head: int, heads: int):
        super().__init__()
        inner = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, inner, bias=False)
        self.to_kv = nn.Linear(dim, inner * 2, bias=False)
        self.to_out = nn.Linear(inner, dim, bias=False)

    def forward(self, x, latents):
        x = self.norm1(x)
        latents = self.norm2(latents)
        b, length, _ = latents.shape
        q = self.to_q(latents)
        k, v = self.to_kv(torch.cat((x, latents), dim=-2)).chunk(2, dim=-1)
        q, k, v = (t.view(b, t.shape[1], self.heads, -1).transpose(1, 2) for t in (q, k, v))
        scale = 1 / math.sqrt(math.sqrt(self.dim_head))
        weight = (q * scale) @ (k * scale).transpose(-2, -1)
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        out = (weight @ v).permute(0, 2, 1, 3).reshape(b, length, -1)
        return self.to_out(out)


class Resampler(nn.Module):
    """Perceiver resampler: 257 CLIP image tokens -> ``num_queries`` context tokens."""

    def __init__(
        self,
        dim: int = 1024,
        depth: int = 4,
        dim_head: int = 64,
        heads: int = 12,
        num_queries: int = 16,
        embedding_dim: int = 1280,
        output_dim: int = 1024,
        ff_mult: int = 4,
    ):
        super().__init__()
        self.latents = nn.Parameter(torch.zeros(1, num_queries, dim))
        self.proj_in = nn.Linear(embedding_dim, dim)
        self.proj_out = nn.Linear(dim, output_dim)
        self.norm_out = nn.LayerNorm(output_dim)
        self.layers = nn.ModuleList(
            nn.ModuleList(
                [
                    PerceiverAttention(dim, dim_head, heads),
                    nn.Sequential(
                        nn.LayerNorm(dim),
                        nn.Linear(dim, dim * ff_mult, bias=False),
                        nn.GELU(),
                        nn.Linear(dim * ff_mult, dim, bias=False),
                    ),
                ]
            )
            for _ in range(depth)
        )

    def forward(self, x):
        latents = self.latents.repeat(x.size(0), 1, 1)
        x = self.proj_in(x)
        for attn, ff in self.layers:
            latents = attn(x, latents) + latents
            latents = ff(latents) + latents
        return self.norm_out(self.proj_out(latents))


class MultiViewUNet(nn.Module):
    """ImageDream's MultiViewUNetModel with CRM's configuration (SD 2.1 base sizes).

    ``stage2=True`` is the CCM model: the six view latents are concatenated to
    the noisy latents (8 input channels) and no view is replaced by the input
    image.
    """

    def __init__(
        self,
        in_channels: int = 4,
        model_channels: int = 320,
        out_channels: int = 4,
        channel_mult: Sequence[int] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        attention_resolutions: Sequence[int] = (4, 2, 1),
        num_head_channels: int = 64,
        context_dim: int = 1024,
        camera_dim: int = 16,
        ip_dim: int = 16,
        image_dim: int = 1280,
        resampler_heads: int = 12,
        stage2: bool = False,
    ):
        super().__init__()
        self.model_channels = model_channels
        self.stage2 = stage2
        emb_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim)
        )
        self.camera_embed = nn.Sequential(
            nn.Linear(camera_dim, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim)
        )
        self.image_embed = Resampler(
            dim=context_dim,
            depth=4,
            dim_head=64,
            heads=resampler_heads,
            num_queries=ip_dim,
            embedding_dim=image_dim,
            output_dim=context_dim,
        )

        def transformer(ch):
            return SpatialTransformer3D(
                ch, ch // num_head_channels, num_head_channels, context_dim, ip_dim
            )

        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(nn.Conv2d(in_channels, model_channels, 3, padding=1))]
        )
        chans = [model_channels]
        ch, ds = model_channels, 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers: list[nn.Module] = [ResBlock(ch, emb_dim, mult * model_channels)]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    layers.append(transformer(ch))
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                chans.append(ch)
            if level != len(channel_mult) - 1:
                self.input_blocks.append(TimestepEmbedSequential(Downsample(ch)))
                chans.append(ch)
                ds *= 2
        self.middle_block = TimestepEmbedSequential(
            ResBlock(ch, emb_dim), transformer(ch), ResBlock(ch, emb_dim)
        )
        self.output_blocks = nn.ModuleList()
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                layers = [ResBlock(ch + chans.pop(), emb_dim, model_channels * mult)]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    layers.append(transformer(ch))
                if level and i == num_res_blocks:
                    layers.append(Upsample(ch))
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
        self.out = nn.Sequential(
            GroupNorm32(32, ch), nn.SiLU(), nn.Conv2d(model_channels, out_channels, 3, padding=1)
        )

    def forward(
        self,
        x,
        timesteps,
        context,
        camera,
        num_frames: int,
        ip,
        ip_img=None,
        pixel_images=None,
    ):
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        emb = emb + self.camera_embed(camera)
        if ip_img is not None:
            # The last view of every object is the input image's latent.
            x = x.clone()
            x[num_frames - 1 :: num_frames] = ip_img
        if self.stage2:
            x = torch.cat((x, pixel_images), dim=1)
        context = torch.cat((context, self.image_embed(ip)), 1)
        h = x.float()
        skips = []
        for module in self.input_blocks:
            h = module(h, emb, context, num_frames=num_frames)
            skips.append(h)
        h = self.middle_block(h, emb, context, num_frames=num_frames)
        for module in self.output_blocks:
            h = module(torch.cat([h, skips.pop()], dim=1), emb, context, num_frames=num_frames)
        return self.out(h.type(x.dtype))


# -- Autoencoder (Stable Diffusion's KL VAE) -----------------------------------


def _vae_norm(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(32, channels, eps=1e-6)


class VAEResnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.norm1 = _vae_norm(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = _vae_norm(out_channels)
        self.dropout = nn.Dropout(0.0)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        if in_channels != out_channels:
            self.nin_shortcut = nn.Conv2d(in_channels, out_channels, 1)
        self.same = in_channels == out_channels

    def forward(self, x):
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return (x if self.same else self.nin_shortcut(x)) + h


class VAEAttnBlock(nn.Module):
    """Single-head self-attention over the latent's pixels."""

    def __init__(self, channels: int):
        super().__init__()
        self.norm = _vae_norm(channels)
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.proj_out = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        y = self.norm(x)
        q, k, v = (m(y).flatten(2).transpose(1, 2)[:, None] for m in (self.q, self.k, self.v))
        out = F.scaled_dot_product_attention(q, k, v)[:, 0]
        return x + self.proj_out(out.transpose(1, 2).reshape(b, c, h, w))


class VAEDownsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=0)

    def forward(self, x):
        return self.conv(F.pad(x, (0, 1, 0, 1)))


class VAEUpsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2.0, mode="nearest"))


def _vae_mid(channels: int) -> nn.Module:
    mid = nn.Module()
    mid.block_1 = VAEResnetBlock(channels, channels)
    mid.attn_1 = VAEAttnBlock(channels)
    mid.block_2 = VAEResnetBlock(channels, channels)
    return mid


class VAEEncoder(nn.Module):
    def __init__(self, ch=128, ch_mult=(1, 2, 4, 4), num_res_blocks=2, in_channels=3, z_channels=4):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, ch, 3, padding=1)
        in_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()
        for level, mult in enumerate(ch_mult):
            down = nn.Module()
            block_in = ch * in_mult[level]
            down.block = nn.ModuleList()
            for _ in range(num_res_blocks):
                down.block.append(VAEResnetBlock(block_in, ch * mult))
                block_in = ch * mult
            down.attn = nn.ModuleList()
            if level != len(ch_mult) - 1:
                down.downsample = VAEDownsample(block_in)
            self.down.append(down)
        self.mid = _vae_mid(block_in)
        self.norm_out = _vae_norm(block_in)
        self.conv_out = nn.Conv2d(block_in, 2 * z_channels, 3, padding=1)

    def forward(self, x):
        h = self.conv_in(x)
        for level, down in enumerate(self.down):
            for block in down.block:
                h = block(h)
            if level != len(self.down) - 1:
                h = down.downsample(h)
        h = self.mid.block_2(self.mid.attn_1(self.mid.block_1(h)))
        return self.conv_out(F.silu(self.norm_out(h)))


class VAEDecoder(nn.Module):
    def __init__(self, ch=128, ch_mult=(1, 2, 4, 4), num_res_blocks=2, out_ch=3, z_channels=4):
        super().__init__()
        block_in = ch * ch_mult[-1]
        self.conv_in = nn.Conv2d(z_channels, block_in, 3, padding=1)
        self.mid = _vae_mid(block_in)
        ups = []
        for level in reversed(range(len(ch_mult))):
            up = nn.Module()
            up.block = nn.ModuleList()
            for _ in range(num_res_blocks + 1):
                up.block.append(VAEResnetBlock(block_in, ch * ch_mult[level]))
                block_in = ch * ch_mult[level]
            up.attn = nn.ModuleList()
            if level != 0:
                up.upsample = VAEUpsample(block_in)
            ups.insert(0, up)
        self.up = nn.ModuleList(ups)
        self.norm_out = _vae_norm(block_in)
        self.conv_out = nn.Conv2d(block_in, out_ch, 3, padding=1)

    def forward(self, z):
        h = self.conv_in(z)
        h = self.mid.block_2(self.mid.attn_1(self.mid.block_1(h)))
        for level in reversed(range(len(self.up))):
            for block in self.up[level].block:
                h = block(h)
            if level != 0:
                h = self.up[level].upsample(h)
        return self.conv_out(F.silu(self.norm_out(h)))


class AutoencoderKL(nn.Module):
    def __init__(self, embed_dim: int = 4, **ddconfig):
        super().__init__()
        self.encoder = VAEEncoder(**ddconfig)
        self.decoder = VAEDecoder(**ddconfig)
        self.quant_conv = nn.Conv2d(2 * embed_dim, 2 * embed_dim, 1)
        self.post_quant_conv = nn.Conv2d(embed_dim, embed_dim, 1)

    def encode_moments(self, x) -> tuple[torch.Tensor, torch.Tensor]:
        """Posterior mean and standard deviation."""
        mean, logvar = torch.chunk(self.quant_conv(self.encoder(x)), 2, dim=1)
        return mean, torch.exp(0.5 * torch.clamp(logvar, -30.0, 20.0))

    def decode(self, z):
        return self.decoder(self.post_quant_conv(z))


# -- OpenCLIP ViT-H/14 towers ---------------------------------------------------


class ResidualAttentionBlock(nn.Module):
    """OpenCLIP's transformer block; ``attn`` keeps nn.MultiheadAttention's parameter names."""

    def __init__(self, width: int, heads: int):
        super().__init__()
        self.heads = heads
        self.ln_1 = nn.LayerNorm(width)
        self.attn = nn.Module()
        self.attn.in_proj_weight = nn.Parameter(torch.zeros(3 * width, width))
        self.attn.in_proj_bias = nn.Parameter(torch.zeros(3 * width))
        self.attn.out_proj = nn.Linear(width, width)
        self.ln_2 = nn.LayerNorm(width)
        self.mlp = nn.Sequential()
        self.mlp.add_module("c_fc", nn.Linear(width, width * 4))
        self.mlp.add_module("gelu", nn.GELU())
        self.mlp.add_module("c_proj", nn.Linear(width * 4, width))

    def forward(self, x, causal: bool = False):
        y = self.ln_1(x)
        q, k, v = F.linear(y, self.attn.in_proj_weight, self.attn.in_proj_bias).chunk(3, dim=-1)
        x = x + self.attn.out_proj(_attention(q, k, v, self.heads, causal=causal))
        return x + self.mlp(self.ln_2(x))


class _Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int):
        super().__init__()
        self.resblocks = nn.ModuleList(ResidualAttentionBlock(width, heads) for _ in range(layers))


class VisionTower(nn.Module):
    def __init__(self, width=1280, layers=32, heads=16, patch=14, image_size=224, output_dim=1024):
        super().__init__()
        grid = image_size // patch
        self.conv1 = nn.Conv2d(3, width, patch, stride=patch, bias=False)
        self.class_embedding = nn.Parameter(torch.zeros(width))
        self.positional_embedding = nn.Parameter(torch.zeros(grid * grid + 1, width))
        self.ln_pre = nn.LayerNorm(width)
        self.transformer = _Transformer(width, layers, heads)
        # Unused by CRM (it reads the penultimate block's tokens); kept so the
        # checkpoint loads strictly.
        self.ln_post = nn.LayerNorm(width)
        self.proj = nn.Parameter(torch.zeros(width, output_dim))

    def penultimate_tokens(self, x):
        x = self.conv1(x).flatten(2).transpose(1, 2)
        cls = self.class_embedding.to(x.dtype) + torch.zeros(
            x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
        )
        x = torch.cat([cls, x], dim=1) + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)
        for block in self.transformer.resblocks[:-1]:
            x = block(x)
        return x


class OpenCLIP(nn.Module):
    """OpenCLIP ViT-H/14: the text tower (penultimate layer, then ln_final) and the vision tower."""

    def __init__(
        self, width=1024, layers=24, heads=16, context_length=77, vocab_size=49408, vision=None
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, width)
        self.positional_embedding = nn.Parameter(torch.zeros(context_length, width))
        self.transformer = _Transformer(width, layers, heads)
        self.ln_final = nn.LayerNorm(width)
        self.text_projection = nn.Parameter(torch.zeros(width, 1024))
        self.logit_scale = nn.Parameter(torch.zeros(()))
        self.visual = VisionTower(**(vision or {}))

    def encode_text(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(tokens) + self.positional_embedding
        for block in self.transformer.resblocks[:-1]:
            x = block(x, causal=True)
        return self.ln_final(x)


class _CLIPHolder(nn.Module):
    def __init__(self, **config):
        super().__init__()
        self.model = OpenCLIP(**config)


class _UNetHolder(nn.Module):
    def __init__(self, stage2: bool, **config):
        super().__init__()
        self.diffusion_model = MultiViewUNet(
            in_channels=8 if stage2 else 4, stage2=stage2, **config
        )


class MultiViewDiffusion(nn.Module):
    """One of CRM's two diffusion checkpoints: UNet, OpenCLIP and VAE under upstream's names.

    The defaults are the released sizes; the config arguments exist for
    small random test networks.
    """

    def __init__(self, stage2: bool, unet=None, clip=None, vae_ch: int = 128):
        super().__init__()
        self.model = _UNetHolder(stage2, **(unet or {}))
        self.clip_model = _CLIPHolder(**(clip or {}))
        self.vae_model = AutoencoderKL(
            embed_dim=4, ch=vae_ch, ch_mult=(1, 2, 4, 4), num_res_blocks=2, z_channels=4
        )


# -- Text prompts ---------------------------------------------------------------

# CRM conditions every object on the same two texts; their OpenCLIP BPE tokens
# (start, words, end, zero padding to 77) are fixed, so no tokenizer is needed.
PROMPT = "3D assets"
NEGATIVE_PROMPT = (
    "uniform low no texture ugly, boring, bad anatomy, blurry, pixelated,  "
    "obscure, unnatural colors, poor lighting, dull, and unclear."
)
_PROMPT_TOKENS = (49406, 274, 323, 13170, 49407)
_NEGATIVE_TOKENS = (
    49406, 11075, 1042, 871, 16505, 8159, 267, 12519, 267, 2103, 15376, 267,
    21977, 267, 14384, 943, 267, 33337, 267, 569, 3288, 5389, 267, 3665, 5799,
    267, 19433, 267, 537, 32955, 269, 49407,
)


def prompt_tokens(device) -> tuple[torch.Tensor, torch.Tensor]:
    """(prompt, negative prompt) token ids, each [1, 77]."""

    def pad(ids):
        out = torch.zeros(1, 77, dtype=torch.long)
        out[0, : len(ids)] = torch.tensor(ids)
        return out.to(device)

    return pad(_PROMPT_TOKENS), pad(_NEGATIVE_TOKENS)


# -- Cameras and schedule -------------------------------------------------------


def _camera_to_world(elevation: float, azimuth: float) -> np.ndarray:
    elevation, azimuth = np.radians(elevation), np.radians(azimuth)
    position = np.array(
        [np.cos(elevation) * np.sin(azimuth), np.sin(elevation), np.cos(elevation) * np.cos(azimuth)]
    )
    forward = -position / np.linalg.norm(position)
    right = np.cross(forward, np.array([0, 1, 0]))
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    up /= np.linalg.norm(up)
    matrix = np.eye(4)
    matrix[:3, :3] = np.array([right, up, -forward]).T
    matrix[:3, 3] = position
    # OpenGL to Blender axes.
    flip_yz = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    return flip_yz @ matrix


# Upstream's view indices: front, left, bottom, back, right, top.
_VIEW_ANGLES = ((0, 0), (0, -90), (-90, 0), (0, 180), (0, 90), (90, 0))


def cameras(views: Sequence[int], reference: int | None = None) -> torch.Tensor:
    """[len(views), 16] flattened camera-to-world matrices; the reference view's camera is zero."""
    rows = [
        torch.tensor(_camera_to_world(*_VIEW_ANGLES[i]).flatten()).float() for i in views
    ]
    if reference is not None:
        rows[reference] = torch.zeros(16)
    return torch.stack(rows)


def alphas_cumprod(dtype: torch.dtype) -> torch.Tensor:
    """SD's linear-in-sqrt schedule rescaled to zero terminal SNR, as the model holds it.

    Upstream casts the whole model, schedule buffers included, to the model
    dtype; the sampler reads the schedule back from there.
    """
    betas = torch.linspace(0.00085**0.5, 0.0120**0.5, 1000, dtype=torch.float64) ** 2
    root = torch.cumprod(1 - betas, 0).sqrt()
    first, last = root[0].clone(), root[-1].clone()
    root = (root - last) * (first / (first - last))
    bar = root**2
    alphas = torch.cat([bar[0:1], bar[1:] / bar[:-1]])
    cumprod = np.cumprod((1.0 - (1 - alphas)).numpy(), axis=0)
    return torch.tensor(cumprod, dtype=torch.float32).to(dtype).float()


def ddim_sample(
    denoise,
    x: torch.Tensor,
    steps: int,
    guidance: float,
    schedule: torch.Tensor,
    callback=None,
) -> torch.Tensor:
    """Deterministic DDIM (eta 0) over upstream's uniform timesteps, with classifier-free guidance.

    ``denoise(x_in, t_in)`` runs the unconditional and conditional halves in
    one batch and returns their noise predictions concatenated.
    """
    stride = 1000 // steps
    timesteps = np.asarray(list(range(0, 1000, stride))) + 1
    a = schedule[timesteps].double().numpy()
    a_prev = np.asarray([float(schedule[0])] + schedule[timesteps[:-1]].tolist())
    sqrt_one_minus = np.sqrt(1.0 - a)
    b = x.shape[0]
    for i, step in enumerate(np.flip(timesteps)):
        index = len(timesteps) - i - 1
        t = torch.full((b,), int(step), device=x.device, dtype=torch.long)
        uncond, cond = denoise(torch.cat([x] * 2), torch.cat([t] * 2)).chunk(2)
        e_t = uncond + guidance * (cond - uncond)
        a_t = torch.full((b, 1, 1, 1), float(a[index]), device=x.device)
        a_p = torch.full((b, 1, 1, 1), float(a_prev[index]), device=x.device)
        s_t = torch.full((b, 1, 1, 1), float(sqrt_one_minus[index]), device=x.device)
        pred_x0 = (x - s_t * e_t) / a_t.sqrt()
        x = a_p.sqrt() * pred_x0 + (1.0 - a_p).sqrt() * e_t
        if callback is not None:
            callback(i, x)
    return x


def autocast(device: torch.device, enabled: bool):
    """fp16 autocast on CUDA and MPS, as upstream runs its diffusion models; off on CPU."""
    if not enabled or device.type not in ("cuda", "mps"):
        return torch.autocast(device_type="cpu", enabled=False)
    return torch.autocast(device_type=device.type, dtype=torch.float16)


SCALE_FACTOR = 0.18215
NUM_FRAMES_STAGE1 = 7
NUM_FRAMES_STAGE2 = 6
STAGE1_VIEWS = (1, 2, 3, 4, 5, 0, 0)
STAGE1_REFERENCE = 6
STAGE2_VIEWS = (1, 2, 3, 4, 5, 0)


@torch.no_grad()
def run_stage(
    model: MultiViewDiffusion,
    *,
    pixels: torch.Tensor,
    clip_pixels: torch.Tensor,
    noise: dict[str, torch.Tensor],
    steps: int,
    guidance: float,
    stage2_views: torch.Tensor | None = None,
    half: bool,
) -> torch.Tensor:
    """One diffusion stage; returns the decoded views as uint8 [V, 256, 256, 3].

    ``pixels`` is the input image at 256 x 256 in [-1, 1], ``clip_pixels`` the
    CLIP-normalized 224 x 224 input; ``noise`` holds the posterior draws
    (``posterior``) and the initial latents (``latents``); ``stage2_views`` the
    six first-stage views in [-1, 1] for the CCM model.
    """
    device = pixels.device
    stage2 = stage2_views is not None
    frames = NUM_FRAMES_STAGE2 if stage2 else NUM_FRAMES_STAGE1
    unet = model.model.diffusion_model
    clip = model.clip_model.model
    vae = model.vae_model
    prompt, negative = prompt_tokens(device)
    # Upstream encodes the negative prompt outside autocast.
    uc_text = clip.encode_text(negative)
    camera = (
        cameras(STAGE2_VIEWS) if stage2 else cameras(STAGE1_VIEWS, STAGE1_REFERENCE)
    ).to(device)
    with autocast(device, half):
        c_text = clip.encode_text(prompt)
        ip = clip.visual.penultimate_tokens(clip_pixels.to(next(clip.parameters()).dtype))
        ip = ip.repeat(frames, 1, 1)
        ip_in = torch.cat([torch.zeros_like(ip), ip])
        context = torch.cat([uc_text.repeat(frames, 1, 1), c_text.repeat(frames, 1, 1)])
        camera_in = torch.cat([camera, camera])
        encoded = pixels if not stage2 else stage2_views
        mean, std = vae.encode_moments(encoded.to(next(vae.parameters()).dtype))
        latent = (mean + std * noise["posterior"].to(device)) * SCALE_FACTOR
        extra = {}
        if stage2:
            extra["pixel_images"] = torch.cat([torch.zeros_like(latent), latent])
        else:
            extra["ip_img"] = torch.cat([torch.zeros_like(latent), latent])

        def denoise(x_in, t_in):
            return unet(x_in, t_in, context, camera_in, frames, ip_in, **extra)

        schedule = alphas_cumprod(torch.float16 if half else torch.float32)
        x = ddim_sample(denoise, noise["latents"].to(device), steps, guidance, schedule)
        images = vae.decode(x / SCALE_FACTOR)
        images = torch.clamp((images + 1.0) / 2.0, 0.0, 1.0)
        images = (255.0 * images.permute(0, 2, 3, 1).float().cpu().numpy()).astype(np.uint8)
    return images
