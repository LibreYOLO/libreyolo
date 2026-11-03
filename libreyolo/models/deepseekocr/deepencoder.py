"""DeepEncoder: the DeepSeek-OCR vision tower (SAM ViT-B into CLIP-L/14).

Ported from ``deepencoder.py`` of the DeepSeek-OCR Hugging Face release
(deepseek-ai/DeepSeek-OCR @ 9f30c71f441d010e5429c532364a86705536c53a, MIT).
The SAM image encoder there is itself lightly adapted from the detectron2
ViTDet backbone (Apache-2.0), ``LayerNorm2d`` from ConvNeXt (MIT) and the
decomposed relative positions from mvit (Apache-2.0). See this package's
``NOTICE``.

Module and parameter names match the released checkpoint, so the weights load
strictly. Changes from the release: only the ``linear`` projector is kept
(the one the checkpoint uses), the TorchScript ``quick_gelu`` is written as the
float32 computation TorchScript performs under autocast, and the training-only
or tensor-parallel scaffolding is dropped. The math is otherwise unchanged,
which is what token-level parity with the release relies on.
"""

from __future__ import annotations

import math
from functools import partial
from typing import Optional, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F


class MlpProjector(nn.Module):
    """Linear projection of the concatenated CLIP and SAM features (2048 to 1280)."""

    def __init__(self, input_dim: int = 2048, n_embed: int = 1280):
        super().__init__()
        self.layers = nn.Linear(input_dim, n_embed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


# =============================================================================
# CLIP-L/14 (consumes the SAM grid as its patch embedding)
# =============================================================================


def get_abs_pos(abs_pos: torch.Tensor, tgt_size: int) -> torch.Tensor:
    """Resize CLIP's (1 + 16x16) position table to ``tgt_size`` tokens."""
    dim = abs_pos.size(-1)
    abs_pos_new = abs_pos.squeeze(0)
    cls_token, old_pos_embed = abs_pos_new[:1], abs_pos_new[1:]
    src_size = int(math.sqrt(abs_pos_new.shape[0] - 1))
    tgt_size = int(math.sqrt(tgt_size))
    dtype = abs_pos.dtype
    if src_size == tgt_size:
        return abs_pos
    old_pos_embed = (
        old_pos_embed.view(1, src_size, src_size, dim).permute(0, 3, 1, 2).contiguous()
    )
    old_pos_embed = old_pos_embed.to(torch.float32)
    new_pos_embed = F.interpolate(
        old_pos_embed,
        size=(tgt_size, tgt_size),
        mode="bicubic",
        antialias=True,
        align_corners=False,
    ).to(dtype)
    new_pos_embed = new_pos_embed.permute(0, 2, 3, 1)
    new_pos_embed = new_pos_embed.view(tgt_size * tgt_size, dim)
    vision_pos_embed = torch.cat([cls_token, new_pos_embed], dim=0)
    return vision_pos_embed.view(1, tgt_size * tgt_size + 1, dim)


def quick_gelu(x: torch.Tensor) -> torch.Tensor:
    # The release scripts this function with TorchScript; under CUDA autocast
    # the scripted graph evaluates it in float32 and returns float32. Doing the
    # same here keeps the CLIP tower bitwise identical to the release.
    x = x.float()
    return x * torch.sigmoid(1.702 * x)


class CLIPVisionEmbeddings(nn.Module):
    def __init__(self, hidden_size=1024, image_size=224, patch_size=14, num_channels=3):
        super().__init__()
        self.embed_dim = hidden_size
        self.image_size = image_size
        self.patch_size = patch_size
        self.class_embedding = nn.Parameter(torch.randn(self.embed_dim))
        # Unused at inference (SAM features replace it) but part of the checkpoint.
        self.patch_embedding = nn.Conv2d(
            in_channels=num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
        )
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_positions).expand((1, -1)),
            persistent=False,
        )

    def forward(self, pixel_values, patch_embeds):
        batch_size = pixel_values.shape[0]
        if patch_embeds is None:
            patch_embeds = self.patch_embedding(pixel_values)
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)
        class_embeds = self.class_embedding.expand(batch_size, 1, -1)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        embeddings = embeddings + get_abs_pos(
            self.position_embedding(self.position_ids), embeddings.size(1)
        )
        return embeddings


class NoTPFeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim, bias=True)
        self.fc2 = nn.Linear(hidden_dim, dim, bias=True)

    def forward(self, x):
        return self.fc2(quick_gelu(self.fc1(x)))


class NoTPAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.qkv_proj = nn.Linear(hidden_size, hidden_size * 3, bias=True)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, _ = x.shape
        xqkv = self.qkv_proj(x)
        xqkv = xqkv.view(bsz, seqlen, 3, self.num_heads, self.head_dim)
        xq, xk, xv = torch.split(xqkv, 1, dim=2)
        xq = xq.squeeze(2).permute(0, 2, 1, 3)
        xk = xk.squeeze(2).permute(0, 2, 1, 3)
        xv = xv.squeeze(2).permute(0, 2, 1, 3)
        output = F.scaled_dot_product_attention(xq, xk, xv, attn_mask=None)
        output = output.permute(0, 2, 1, 3).reshape(bsz, seqlen, -1)
        return self.out_proj(output)


class NoTPTransformerBlock(nn.Module):
    def __init__(
        self, hidden_size: int, num_heads: int, ffn_hidden_size: int, eps: float
    ):
        super().__init__()
        self.self_attn = NoTPAttention(hidden_size, num_heads)
        self.mlp = NoTPFeedForward(hidden_size, ffn_hidden_size)
        self.layer_norm1 = nn.LayerNorm(hidden_size, eps=eps)
        self.layer_norm2 = nn.LayerNorm(hidden_size, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x + self.self_attn(self.layer_norm1(x))
        return h + self.mlp(self.layer_norm2(h))


class NoTPTransformer(nn.Module):
    def __init__(self, num_layers, hidden_size, num_heads, ffn_hidden_size, eps):
        super().__init__()
        self.layers = nn.ModuleList(
            NoTPTransformerBlock(hidden_size, num_heads, ffn_hidden_size, eps)
            for _ in range(num_layers)
        )

    def forward(self, hidden_states):
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


class VitModel(nn.Module):
    """CLIP-L/14 transformer (24 layers, width 1024)."""

    def __init__(
        self,
        num_layers: int = 24,
        hidden_size: int = 1024,
        num_heads: int = 16,
        ffn_hidden_size: int = 4096,
        image_size: int = 224,
        patch_size: int = 14,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.embeddings = CLIPVisionEmbeddings(
            hidden_size=hidden_size, image_size=image_size, patch_size=patch_size
        )
        self.transformer = NoTPTransformer(
            num_layers, hidden_size, num_heads, ffn_hidden_size, eps
        )
        self.pre_layrnorm = nn.LayerNorm(hidden_size, eps=eps)

    def forward(self, x, patch_embeds):
        x = self.embeddings(x, patch_embeds)
        return self.transformer(self.pre_layrnorm(x))


# =============================================================================
# SAM ViT-B image encoder (windowed attention, extra stride-2 convs)
# =============================================================================


def get_abs_pos_sam(abs_pos: torch.Tensor, tgt_size: int) -> torch.Tensor:
    dtype = abs_pos.dtype
    src_size = abs_pos.size(1)
    if src_size == tgt_size:
        return abs_pos
    old_pos_embed = abs_pos.permute(0, 3, 1, 2).to(torch.float32)
    new_pos_embed = F.interpolate(
        old_pos_embed,
        size=(tgt_size, tgt_size),
        mode="bicubic",
        antialias=True,
        align_corners=False,
    ).to(dtype)
    return new_pos_embed.permute(0, 2, 3, 1)


class MLPBlock(nn.Module):
    def __init__(
        self, embedding_dim: int, mlp_dim: int, act: Type[nn.Module] = nn.GELU
    ):
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


def window_partition(x: torch.Tensor, window_size: int):
    B, H, W, C = x.shape
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w
    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = (
        x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    )
    return windows, (Hp, Wp)


def window_unpartition(windows, window_size: int, pad_hw, hw):
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.view(
        B, Hp // window_size, Wp // window_size, window_size, window_size, -1
    )
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)
    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()
    return x


def get_rel_pos(q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    if rel_pos.shape[0] != max_rel_dist:
        dtype = rel_pos.dtype
        rel_pos = rel_pos.to(torch.float32)
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        ).to(dtype)
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos
    q_coords = torch.arange(q_size, device=rel_pos.device)[:, None] * max(
        k_size / q_size, 1.0
    )
    k_coords = torch.arange(k_size, device=rel_pos.device)[None, :] * max(
        q_size / k_size, 1.0
    )
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)
    return rel_pos_resized[relative_coords.long()]


def add_decomposed_rel_pos(q, rel_pos_h, rel_pos_w, q_size, k_size):
    q_h, q_w = q_size
    k_h, k_w = k_size
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)
    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, dim)
    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)
    rel_h = rel_h.unsqueeze(-1).reshape(B, q_h * q_w, k_h, 1)
    rel_w = rel_w.unsqueeze(-2).reshape(B, q_h * q_w, 1, k_w)
    return rel_h, rel_w


class Attention(nn.Module):
    """Multi-head attention with decomposed relative position bias."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        use_rel_pos: bool = False,
        input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            assert input_size is not None
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = x.shape
        qkv = (
            self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)
        rel_h, rel_w = None, None
        if self.use_rel_pos:
            rel_h, rel_w = add_decomposed_rel_pos(
                q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W)
            )
        q = q.view(B, self.num_heads, H * W, -1)
        k = k.view(B, self.num_heads, H * W, -1)
        v = v.view(B, self.num_heads, H * W, -1)
        if self.use_rel_pos:
            rel_h = rel_h.view(
                B, self.num_heads, rel_h.size(1), rel_h.size(2), rel_h.size(3)
            )
            rel_w = rel_w.view(
                B, self.num_heads, rel_w.size(1), rel_w.size(2), rel_w.size(3)
            )
            attn_bias = (rel_h + rel_w).view(
                B, self.num_heads, rel_h.size(2), rel_h.size(3) * rel_w.size(4)
            )
            x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias)
        else:
            x = F.scaled_dot_product_attention(q, k, v)
        x = (
            x.view(B, self.num_heads, H, W, -1)
            .permute(0, 2, 3, 1, 4)
            .reshape(B, H, W, -1)
        )
        return self.proj(x)


class Block(nn.Module):
    """Transformer block with optional window attention."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_rel_pos: bool = False,
        window_size: int = 0,
        input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            input_size=input_size if window_size == 0 else (window_size, window_size),
        )
        self.norm2 = norm_layer(dim)
        self.mlp = MLPBlock(dim, int(dim * mlp_ratio), act=act_layer)
        self.window_size = window_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.norm1(x)
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)
        x = self.attn(x)
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))
        x = shortcut + x
        return x + self.mlp(self.norm2(x))


class PatchEmbed(nn.Module):
    def __init__(
        self, kernel_size=(16, 16), stride=(16, 16), in_chans=3, embed_dim=768
    ):
        super().__init__()
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=kernel_size, stride=stride
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).permute(0, 2, 3, 1)


class ImageEncoderViT(nn.Module):
    """SAM ViT-B with two extra stride-2 convs (256 to 1024 channels)."""

    def __init__(
        self,
        img_size: int = 1024,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        out_chans: int = 256,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_rel_pos: bool = True,
        window_size: int = 14,
        global_attn_indexes: Tuple[int, ...] = (2, 5, 8, 11),
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        self.pos_embed = nn.Parameter(
            torch.zeros(1, img_size // patch_size, img_size // patch_size, embed_dim)
        )
        self.blocks = nn.ModuleList(
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos,
                window_size=window_size if i not in global_attn_indexes else 0,
                input_size=(img_size // patch_size, img_size // patch_size),
            )
            for i in range(depth)
        )
        self.neck = nn.Sequential(
            nn.Conv2d(embed_dim, out_chans, kernel_size=1, bias=False),
            LayerNorm2d(out_chans),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(out_chans),
        )
        self.net_2 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False)
        self.net_3 = nn.Conv2d(
            512, 1024, kernel_size=3, stride=2, padding=1, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = x + get_abs_pos_sam(self.pos_embed, x.size(1))
        for blk in self.blocks:
            x = blk(x)
        x = self.neck(x.permute(0, 3, 1, 2))
        x2 = self.net_2(x)
        return self.net_3(x2.clone())


def build_sam_vit_b() -> ImageEncoderViT:
    return ImageEncoderViT(
        depth=12,
        embed_dim=768,
        img_size=1024,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        num_heads=12,
        patch_size=16,
        qkv_bias=True,
        use_rel_pos=True,
        global_attn_indexes=(2, 5, 8, 11),
        window_size=14,
        out_chans=256,
    )


def build_clip_l() -> VitModel:
    return VitModel()
