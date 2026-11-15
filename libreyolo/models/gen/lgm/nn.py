# Adapted from LGM (https://github.com/3DTopia/LGM, commit
# fe8d12cff8c827df7bb77a3c8e8b37408cb6fe4c), MIT, Copyright (c) 2024 3D Topia:
# ``core/unet.py``, ``core/models.py``, ``core/attention.py`` (Apache-2.0,
# Copyright (c) Meta Platforms, Inc. and affiliates, from DINOv2) and
# ``mvdream/mv_unet.py`` (the ImageDream multi-view UNet, after MVDream, MIT,
# and ImageDream, Apache-2.0, both by ByteDance).
"""The two networks of LGM, in plain PyTorch.

``MultiViewUNet`` is ImageDream's multi-view diffusion UNet: a Stable
Diffusion 2.1 UNet whose self-attention runs across the views, with a camera
embedding added to the timestep embedding and an image-prompt branch (a
perceiver resampler over CLIP image tokens, read by extra key and value
projections in every cross-attention). ``LGMNetwork`` is LGM's asymmetric
UNet: four views with their ray embeddings in, one Gaussian per output pixel
of each view out.

Upstream calls xformers' memory-efficient attention, a CUDA-only kernel;
here every attention is ``torch.nn.functional.scaled_dot_product_attention``,
the same computation on any device. Module and parameter names are
upstream's, so the released safetensors load with ``strict=True``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn

# -- ImageDream multi-view UNet ------------------------------------------------


def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000):
    """Sinusoidal embedding, cosines first, computed in float32."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(0, half, dtype=torch.float32) / half
    ).to(timesteps.device)
    args = timesteps[:, None] * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def _heads(x: torch.Tensor, heads: int) -> torch.Tensor:
    batch, length, _ = x.shape
    return x.reshape(batch, length, heads, -1).transpose(1, 2)


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
        inner = int(dim * mult)
        self.net = nn.Sequential(
            GEGLU(dim, inner), nn.Dropout(0.0), nn.Linear(inner, dim)
        )

    def forward(self, x):
        return self.net(x)


class CrossAttention(nn.Module):
    """Attention over ``context`` (itself when None); ``ip_dim`` trailing context tokens go through their own projections."""

    def __init__(
        self,
        query_dim: int,
        context_dim: int | None = None,
        heads: int = 8,
        dim_head: int = 64,
        ip_dim: int = 0,
        ip_weight: float = 1.0,
    ):
        super().__init__()
        inner = dim_head * heads
        context_dim = context_dim if context_dim is not None else query_dim
        self.heads = heads
        self.ip_dim = ip_dim
        self.ip_weight = ip_weight
        if ip_dim > 0:
            self.to_k_ip = nn.Linear(context_dim, inner, bias=False)
            self.to_v_ip = nn.Linear(context_dim, inner, bias=False)
        self.to_q = nn.Linear(query_dim, inner, bias=False)
        self.to_k = nn.Linear(context_dim, inner, bias=False)
        self.to_v = nn.Linear(context_dim, inner, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner, query_dim), nn.Dropout(0.0))

    def forward(self, x, context=None):
        context = x if context is None else context
        q = _heads(self.to_q(x), self.heads)
        if self.ip_dim > 0:
            image_tokens = context[:, -self.ip_dim :]
            context = context[:, : context.shape[1] - self.ip_dim]
        k = _heads(self.to_k(context), self.heads)
        v = _heads(self.to_v(context), self.heads)
        out = F.scaled_dot_product_attention(q, k, v)
        if self.ip_dim > 0:
            k_ip = _heads(self.to_k_ip(image_tokens), self.heads)
            v_ip = _heads(self.to_v_ip(image_tokens), self.heads)
            out = out + self.ip_weight * F.scaled_dot_product_attention(q, k_ip, v_ip)
        batch, _, length, _ = out.shape
        return self.to_out(out.transpose(1, 2).reshape(batch, length, -1))


class BasicTransformerBlock3D(nn.Module):
    """Self-attention across all views of an object, then cross-attention and a GEGLU MLP per view."""

    def __init__(self, dim, n_heads, d_head, context_dim, ip_dim=0, ip_weight=1.0):
        super().__init__()
        self.attn1 = CrossAttention(dim, None, n_heads, d_head)
        self.ff = FeedForward(dim)
        self.attn2 = CrossAttention(
            dim, context_dim, n_heads, d_head, ip_dim, ip_weight
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

    def forward(self, x, context=None, num_frames: int = 1):
        batch_frames, length, channels = x.shape
        x = x.reshape(batch_frames // num_frames, num_frames * length, channels)
        x = self.attn1(self.norm1(x)) + x
        x = x.reshape(batch_frames, length, channels)
        x = self.attn2(self.norm2(x), context=context) + x
        return self.ff(self.norm3(x)) + x


class SpatialTransformer3D(nn.Module):
    def __init__(
        self,
        in_channels,
        n_heads,
        d_head,
        context_dim,
        depth=1,
        ip_dim=0,
        ip_weight=1.0,
    ):
        super().__init__()
        inner = n_heads * d_head
        self.norm = nn.GroupNorm(32, in_channels, eps=1e-6, affine=True)
        self.proj_in = nn.Linear(in_channels, inner)
        self.transformer_blocks = nn.ModuleList(
            BasicTransformerBlock3D(
                inner, n_heads, d_head, context_dim, ip_dim, ip_weight
            )
            for _ in range(depth)
        )
        self.proj_out = nn.Linear(in_channels, inner)

    def forward(self, x, context=None, num_frames: int = 1):
        batch, _, height, width = x.shape
        x_in = x
        x = self.norm(x).flatten(2).transpose(1, 2).contiguous()
        x = self.proj_in(x)
        for block in self.transformer_blocks:
            x = block(x, context=context, num_frames=num_frames)
        x = self.proj_out(x)
        x = x.transpose(1, 2).reshape(batch, -1, height, width).contiguous()
        return x + x_in


class PerceiverAttention(nn.Module):
    def __init__(self, *, dim, dim_head=64, heads=8):
        super().__init__()
        self.dim_head = dim_head
        self.heads = heads
        inner = dim_head * heads
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, inner, bias=False)
        self.to_kv = nn.Linear(dim, inner * 2, bias=False)
        self.to_out = nn.Linear(inner, dim, bias=False)

    def forward(self, x, latents):
        x = self.norm1(x)
        latents = self.norm2(latents)
        batch, length, _ = latents.shape
        q = _heads(self.to_q(latents), self.heads)
        k, v = self.to_kv(torch.cat((x, latents), dim=-2)).chunk(2, dim=-1)
        k, v = _heads(k, self.heads), _heads(v, self.heads)
        # Upstream scales q and k by d^-1/4 each and takes the softmax in float32.
        scale = 1 / math.sqrt(math.sqrt(self.dim_head))
        weight = (q * scale) @ (k * scale).transpose(-2, -1)
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        out = (weight @ v).permute(0, 2, 1, 3).reshape(batch, length, -1)
        return self.to_out(out)


class Resampler(nn.Module):
    """Perceiver resampler: CLIP image tokens -> ``num_queries`` context tokens."""

    def __init__(
        self,
        dim=1024,
        depth=8,
        dim_head=64,
        heads=16,
        num_queries=8,
        embedding_dim=768,
        output_dim=1024,
        ff_mult=4,
    ):
        super().__init__()
        self.latents = nn.Parameter(torch.zeros(1, num_queries, dim))
        self.proj_in = nn.Linear(embedding_dim, dim)
        self.proj_out = nn.Linear(dim, output_dim)
        self.norm_out = nn.LayerNorm(output_dim)
        self.layers = nn.ModuleList(
            nn.ModuleList(
                [
                    PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
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
    def __init__(
        self, channels: int, emb_channels: int, out_channels: int | None = None
    ):
        super().__init__()
        out_channels = out_channels or channels
        self.in_layers = nn.Sequential(
            nn.GroupNorm(32, channels),
            nn.SiLU(),
            nn.Conv2d(channels, out_channels, 3, padding=1),
        )
        self.emb_layers = nn.Sequential(
            nn.SiLU(), nn.Linear(emb_channels, out_channels)
        )
        self.out_layers = nn.Sequential(
            nn.GroupNorm(32, out_channels),
            nn.SiLU(),
            nn.Dropout(0.0),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )
        self.skip_connection = (
            nn.Identity()
            if out_channels == channels
            else nn.Conv2d(channels, out_channels, 1)
        )

    def forward(self, x, emb):
        h = self.in_layers(x)
        h = h + self.emb_layers(emb).type(h.dtype)[..., None, None]
        return self.skip_connection(x) + self.out_layers(h)


class CondSequential(nn.Sequential):
    def forward(self, x, emb, context=None, num_frames: int = 1):
        for layer in self:
            if isinstance(layer, ResBlock):
                x = layer(x, emb)
            elif isinstance(layer, SpatialTransformer3D):
                x = layer(x, context, num_frames=num_frames)
            else:
                x = layer(x)
        return x


class MultiViewUNet(nn.Module):
    """ImageDream's multi-view UNet (upstream ``MultiViewUNetModel``), built from its ``unet/config.json``.

    Only the configuration the released checkpoints use is supported: 2D,
    no class conditioning, convolutional resampling, no scale-shift norm.
    """

    def __init__(
        self,
        image_size: int,
        in_channels: int,
        model_channels: int,
        out_channels: int,
        num_res_blocks: int,
        attention_resolutions: Sequence[int],
        channel_mult: Sequence[int] = (1, 2, 4, 8),
        num_head_channels: int = -1,
        transformer_depth: int = 1,
        context_dim: int = 1024,
        camera_dim: int | None = None,
        ip_dim: int = 0,
        ip_weight: float = 1.0,
        **_ignored,
    ):
        super().__init__()
        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.ip_dim = ip_dim
        if ip_dim > 0:
            self.image_embed = Resampler(
                dim=context_dim,
                depth=4,
                dim_head=64,
                heads=12,
                num_queries=ip_dim,
                embedding_dim=1280,
                output_dim=context_dim,
                ff_mult=4,
            )
        time_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.camera_embed = None
        if camera_dim is not None:
            self.camera_embed = nn.Sequential(
                nn.Linear(camera_dim, time_dim),
                nn.SiLU(),
                nn.Linear(time_dim, time_dim),
            )

        def transformer(ch):
            return SpatialTransformer3D(
                ch,
                ch // num_head_channels,
                num_head_channels,
                context_dim,
                transformer_depth,
                ip_dim,
                ip_weight,
            )

        self.input_blocks = nn.ModuleList(
            [CondSequential(nn.Conv2d(in_channels, model_channels, 3, padding=1))]
        )
        skip_channels = [model_channels]
        ch, ds = model_channels, 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers: list[nn.Module] = [
                    ResBlock(ch, time_dim, mult * model_channels)
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    layers.append(transformer(ch))
                self.input_blocks.append(CondSequential(*layers))
                skip_channels.append(ch)
            if level != len(channel_mult) - 1:
                self.input_blocks.append(CondSequential(Downsample(ch, ch)))
                skip_channels.append(ch)
                ds *= 2
        self.middle_block = CondSequential(
            ResBlock(ch, time_dim), transformer(ch), ResBlock(ch, time_dim)
        )
        self.output_blocks = nn.ModuleList()
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                layers = [
                    ResBlock(ch + skip_channels.pop(), time_dim, model_channels * mult)
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    layers.append(transformer(ch))
                if level and i == num_res_blocks:
                    layers.append(Upsample(ch, ch))
                    ds //= 2
                self.output_blocks.append(CondSequential(*layers))
        self.out = nn.Sequential(
            nn.GroupNorm(32, ch),
            nn.SiLU(),
            nn.Conv2d(model_channels, out_channels, 3, padding=1),
        )

    def forward(
        self,
        x,
        timesteps,
        context,
        camera=None,
        num_frames: int = 1,
        ip=None,
        ip_img=None,
    ):
        """x ``[(B x F), C, H, W]`` latents of F views per object; the last view of each object is the input image's slot."""
        emb = self.time_embed(
            timestep_embedding(timesteps, self.model_channels).to(x.dtype)
        )
        if camera is not None:
            emb = emb + self.camera_embed(camera)
        if self.ip_dim > 0:
            x = x.clone()
            x[num_frames - 1 :: num_frames] = ip_img
            context = torch.cat((context, self.image_embed(ip)), 1)
        hs = []
        h = x
        for module in self.input_blocks:
            h = module(h, emb, context, num_frames=num_frames)
            hs.append(h)
        h = self.middle_block(h, emb, context, num_frames=num_frames)
        for module in self.output_blocks:
            h = module(
                torch.cat([h, hs.pop()], dim=1), emb, context, num_frames=num_frames
            )
        return self.out(h.type(x.dtype))


# -- LGM: views + rays -> Gaussians -------------------------------------------


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(0.0)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(0.0)

    def forward(self, x):
        batch, length, channels = x.shape
        qkv = self.qkv(x).reshape(
            batch, length, 3, self.num_heads, channels // self.num_heads
        )
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        x = F.scaled_dot_product_attention(q, k, v)
        return self.proj(x.transpose(1, 2).reshape(batch, length, channels))


class MVAttention(nn.Module):
    """Self-attention over the pixels of all views of an object."""

    def __init__(
        self, dim: int, num_heads: int = 8, skip_scale: float = 1.0, num_frames: int = 4
    ):
        super().__init__()
        self.skip_scale = skip_scale
        self.num_frames = num_frames
        self.norm = nn.GroupNorm(32, dim, eps=1e-5, affine=True)
        self.attn = Attention(dim, num_heads)

    def forward(self, x):
        batch_views, channels, height, width = x.shape
        batch = batch_views // self.num_frames
        res = x
        x = self.norm(x)
        x = x.reshape(batch, self.num_frames, channels, height, width).permute(
            0, 1, 3, 4, 2
        )
        x = self.attn(x.reshape(batch, -1, channels))
        x = x.reshape(batch, self.num_frames, height, width, channels).permute(
            0, 1, 4, 2, 3
        )
        return (x.reshape(batch_views, channels, height, width) + res) * self.skip_scale


class ResnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, skip_scale: float = 1.0):
        super().__init__()
        self.skip_scale = skip_scale
        self.norm1 = nn.GroupNorm(32, in_channels, eps=1e-5, affine=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(32, out_channels, eps=1e-5, affine=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.shortcut = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, 1)
        )

    def forward(self, x):
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return (h + self.shortcut(x)) * self.skip_scale


class DownBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_layers,
        downsample,
        attention,
        skip_scale,
        heads=16,
    ):
        super().__init__()
        self.nets = nn.ModuleList(
            ResnetBlock(
                in_channels if i == 0 else out_channels, out_channels, skip_scale
            )
            for i in range(num_layers)
        )
        self.attns = nn.ModuleList(
            MVAttention(out_channels, heads, skip_scale) if attention else None
            for _ in range(num_layers)
        )
        self.downsample = (
            nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)
            if downsample
            else None
        )

    def forward(self, x):
        xs = []
        for attn, net in zip(self.attns, self.nets):
            x = net(x)
            if attn is not None:
                x = attn(x)
            xs.append(x)
        if self.downsample is not None:
            x = self.downsample(x)
            xs.append(x)
        return x, xs


class MidBlock(nn.Module):
    def __init__(self, channels, attention, skip_scale, heads=16):
        super().__init__()
        self.nets = nn.ModuleList(
            ResnetBlock(channels, channels, skip_scale) for _ in range(2)
        )
        self.attns = nn.ModuleList(
            [MVAttention(channels, heads, skip_scale) if attention else None]
        )

    def forward(self, x):
        x = self.nets[0](x)
        for attn, net in zip(self.attns, self.nets[1:]):
            if attn is not None:
                x = attn(x)
            x = net(x)
        return x


class UpBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        num_layers,
        upsample,
        attention,
        skip_scale,
        heads=16,
    ):
        super().__init__()
        self.nets = nn.ModuleList(
            ResnetBlock(
                (in_channels if i == 0 else out_channels)
                + (skip_channels if i == num_layers - 1 else out_channels),
                out_channels,
                skip_scale,
            )
            for i in range(num_layers)
        )
        self.attns = nn.ModuleList(
            MVAttention(out_channels, heads, skip_scale) if attention else None
            for _ in range(num_layers)
        )
        self.upsample = (
            nn.Conv2d(out_channels, out_channels, 3, padding=1) if upsample else None
        )

    def forward(self, x, xs):
        for attn, net in zip(self.attns, self.nets):
            x = net(torch.cat([x, xs.pop()], dim=1))
            if attn is not None:
                x = attn(x)
        if self.upsample is not None:
            x = self.upsample(F.interpolate(x, scale_factor=2.0, mode="nearest"))
        return x


class UNet(nn.Module):
    """LGM's asymmetric UNet: the decoder may stop before the input resolution."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        down_channels: Sequence[int],
        down_attention: Sequence[bool],
        mid_attention: bool,
        up_channels: Sequence[int],
        up_attention: Sequence[bool],
        layers_per_block: int = 2,
        skip_scale: float = math.sqrt(0.5),
    ):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, down_channels[0], 3, padding=1)
        blocks, cout = [], down_channels[0]
        for i, channels in enumerate(down_channels):
            cin, cout = cout, channels
            blocks.append(
                DownBlock(
                    cin,
                    cout,
                    layers_per_block,
                    i != len(down_channels) - 1,
                    down_attention[i],
                    skip_scale,
                )
            )
        self.down_blocks = nn.ModuleList(blocks)
        self.mid_block = MidBlock(down_channels[-1], mid_attention, skip_scale)
        blocks, cout = [], up_channels[0]
        for i, channels in enumerate(up_channels):
            cin, cout = cout, channels
            cskip = down_channels[max(-2 - i, -len(down_channels))]
            blocks.append(
                UpBlock(
                    cin,
                    cskip,
                    cout,
                    layers_per_block + 1,
                    i != len(up_channels) - 1,
                    up_attention[i],
                    skip_scale,
                )
            )
        self.up_blocks = nn.ModuleList(blocks)
        self.norm_out = nn.GroupNorm(32, up_channels[-1], eps=1e-5)
        self.conv_out = nn.Conv2d(up_channels[-1], out_channels, 3, padding=1)

    def forward(self, x):
        x = self.conv_in(x)
        skips = [x]
        for block in self.down_blocks:
            x, xs = block(x)
            skips.extend(xs)
        x = self.mid_block(x)
        for block in self.up_blocks:
            count = len(block.nets)
            x = block(x, skips[-count:])
            skips = skips[:-count]
        return self.conv_out(F.silu(self.norm_out(x)))


# LGM's released "big" configuration (core/options.py).
LGM_BIG = {
    "down_channels": (64, 128, 256, 512, 1024, 1024),
    "down_attention": (False, False, False, True, True, True),
    "mid_attention": True,
    "up_channels": (1024, 1024, 512, 256, 128),
    "up_attention": (True, True, True, False, False),
}


class LGMNetwork(nn.Module):
    """LGM: ``[B, 4, 9, H, W]`` views (normalised RGB + Plücker rays) -> ``[B, N, 14]`` activated Gaussians.

    Each row is position (3, in ``[-1, 1]``), opacity, scale (3), rotation
    (4, ``w, x, y, z``) and RGB (3), as upstream's ``forward_gaussians``.
    """

    def __init__(self, splat_size: int = 128, **unet):
        super().__init__()
        self.splat_size = splat_size
        self.unet = UNet(9, 14, **(unet or LGM_BIG))
        self.conv = nn.Conv2d(14, 14, kernel_size=1)

    def forward(self, images):
        batch, views, channels, height, width = images.shape
        x = self.unet(images.view(batch * views, channels, height, width))
        x = self.conv(x)
        x = x.reshape(batch, views, 14, self.splat_size, self.splat_size)
        x = x.permute(0, 1, 3, 4, 2).reshape(batch, -1, 14)
        position = x[..., 0:3].clamp(-1, 1)
        opacity = torch.sigmoid(x[..., 3:4])
        scale = 0.1 * F.softplus(x[..., 4:7])
        rotation = F.normalize(x[..., 7:11], dim=-1)
        rgb = 0.5 * torch.tanh(x[..., 11:]) + 0.5
        return torch.cat([position, opacity, scale, rotation, rgb], dim=-1)
