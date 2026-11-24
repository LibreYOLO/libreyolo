#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from functools import partial
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


def _require_timm():
    """Import ``timm`` lazily with an actionable extra hint."""
    try:
        import timm
    except ImportError as exc:
        raise ModuleNotFoundError(
            "Depth Pro 2 requires timm. Install with: pip install 'libreyolo[depthpro2]'"
        ) from exc
    return timm


def eva_qkv_rope(
    block: Any, x: torch.Tensor, rope: torch.Tensor | None, num_prefix: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split a timm ``EvaBlock``: input ``(B, L, C)`` to ``q, k, v``, roped past ``num_prefix``.

    Raises:
        NotImplementedError: On gated attention, which ``eva_finish`` drops; no DINOv3 preset gates.
    """
    attn: Any = block.attn
    if getattr(attn, "gate", None) is not None:
        raise NotImplementedError(
            "Gated EVA attention is not supported in unbuffered mode."
        )
    y = block.norm1(x)
    batch, length, _ = y.shape

    if attn.qkv is not None:
        if attn.q_bias is None:
            qkv = attn.qkv(y)
        else:
            bias = torch.cat((attn.q_bias, attn.k_bias, attn.v_bias))
            if attn.qkv_bias_separate:
                qkv = attn.qkv(y) + bias
            else:
                qkv = F.linear(y, weight=attn.qkv.weight, bias=bias)
        q, k, v = (
            qkv.reshape(batch, length, 3, attn.num_heads, -1)
            .permute(2, 0, 3, 1, 4)
            .unbind(0)
        )
    else:
        q = attn.q_proj(y).reshape(batch, length, attn.num_heads, -1).transpose(1, 2)
        k = attn.k_proj(y).reshape(batch, length, attn.num_heads, -1).transpose(1, 2)
        v = attn.v_proj(y).reshape(batch, length, attn.num_heads, -1).transpose(1, 2)
    q, k = attn.q_norm(q), attn.k_norm(k)

    if rope is not None:
        _require_timm()
        from timm.layers.pos_embed_sincos import apply_rot_embed_cat

        half = getattr(attn, "rotate_half", False)
        rotate = partial(apply_rot_embed_cat, emb=rope, half=half)
        q = torch.cat([q[:, :, :num_prefix], rotate(q[:, :, num_prefix:])], dim=2).type_as(v)
        k = torch.cat([k[:, :, :num_prefix], rotate(k[:, :, num_prefix:])], dim=2).type_as(v)
    return q, k, v


def eva_finish(
    block: Any, x: torch.Tensor, attended: torch.Tensor
) -> torch.Tensor:
    """Finish a timm ``EvaBlock`` from its input ``(B, L, C)`` and attention ``(B, h, L, d)``.

    The second half of :func:`eva_qkv_rope`; eval mode only, skipping dropout and drop-path.
    """
    attn: Any = block.attn
    merged = attended.transpose(1, 2).reshape(x.shape[0], x.shape[1], -1)
    merged = attn.proj_drop(attn.proj(attn.norm(merged)))
    if block.gamma_1 is None:
        x = x + merged
        return x + block.mlp(block.norm2(x))
    x = x + block.gamma_1 * merged
    return x + block.gamma_2 * block.mlp(block.norm2(x))


class KVCache:
    """Unbuffered mode's FIFO cache of the last ``window`` frames, the current one included.

    Each cross-frame block keeps a ring of the frames' class and register tokens, and the keys and
    values of their mean, the pooled prefix, ahead of a ring of the frames' patch keys and values.
    Per frame and block, call ``update_prefix``, then ``update``.
    """

    def __init__(self, window: int):
        """Open an empty cache of ``window`` frames, allocated per block on its first frame."""
        self.window = window
        self.seen: dict[int, int] = {}
        self.prefix: dict[int, torch.Tensor] = {}
        self.kv: dict[int, torch.Tensor] = {}

    def update_prefix(self, idx: int, prefix: torch.Tensor) -> torch.Tensor:
        """Store a frame's class and register tokens ``(1, P, C)`` in block ``idx``'s ring.

        Returns:
            The pooled prefix: their mean ``(1, P, C)`` over the frames visible to this one.
        """
        if idx not in self.prefix:
            self.seen[idx] = 0
            self.prefix[idx] = prefix.new_empty((self.window, *prefix.shape))
        seen = self.seen[idx]
        self.prefix[idx][seen % self.window] = prefix
        return self.prefix[idx][: min(seen + 1, self.window)].mean(dim=0)

    def update(
        self,
        idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
        pooled_key: torch.Tensor,
        pooled_value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Store a frame's patch keys and values ``(1, h, N, d)``, and the pooled prefix's.

        Returns:
            The keys and values to attend, ``(1, h, P + n * N, d)`` over the ``n`` visible frames.
        """
        num_prefix, num_patches = pooled_key.shape[2], key.shape[2]
        length = num_prefix + self.window * num_patches
        if idx not in self.kv:
            self.kv[idx] = key.new_empty((2, 1, key.shape[1], length, key.shape[3]))
        kv, seen = self.kv[idx], self.seen[idx]
        # Any slot order is correct (no temporal position), but this one is the bit-exact reference.
        slot = seen % self.window
        rows = slice(
            num_prefix + slot * num_patches, num_prefix + (slot + 1) * num_patches
        )
        kv[0, :, :, rows], kv[1, :, :, rows] = key, value
        kv[0, :, :, :num_prefix], kv[1, :, :, :num_prefix] = pooled_key, pooled_value
        self.seen[idx] = seen + 1
        live = kv[:, :, :, : num_prefix + min(seen + 1, self.window) * num_patches]
        return live[0], live[1]


class MultiViewEncoder(nn.Module):
    """Multi-view transformer producing a 4-level feature pyramid per frame."""

    def __init__(
        self,
        backbone_preset: str,
        reassembled_blocks: Sequence[int],
        cross_frame_blocks: Sequence[int],
        dims_reassemble: Sequence[int],
    ):
        """Initialize the encoder.

        Args:
            backbone_preset: Name of the timm DINOv3 architecture to build the backbone from.
            reassembled_blocks: Zero-based indices of the 4 blocks whose outputs feed the pyramid.
            cross_frame_blocks: Zero-based indices of the blocks that attend across frames.
            dims_reassemble: Channel counts of the 4 pyramid levels.
        """
        super().__init__()
        timm = _require_timm()
        vit = timm.create_model(
            backbone_preset, pretrained=False, num_classes=0, dynamic_img_size=True
        )
        vit.rope.normalize_coords = "max"

        self.patch_embed = vit.patch_embed
        self.cls_token = vit.cls_token
        self.reg_token = vit.reg_token
        self.rope = vit.rope
        self.blocks = vit.blocks

        self.embed_dim: int = vit.embed_dim
        self.patch_size: int = vit.patch_embed.patch_size[0]
        self.num_prefix_tokens: int = vit.num_prefix_tokens
        self.reassembled_blocks = tuple(reassembled_blocks)
        self.cross_frame_blocks = frozenset(cross_frame_blocks)

        dim = self.embed_dim
        self.reassemble = nn.ModuleList(
            [
                # 1/16 -> 1/4.
                nn.Sequential(
                    nn.Conv2d(dim, dims_reassemble[0], kernel_size=1),
                    nn.ConvTranspose2d(
                        dims_reassemble[0], dims_reassemble[0], kernel_size=4, stride=4
                    ),
                ),
                # 1/16 -> 1/8.
                nn.Sequential(
                    nn.Conv2d(dim, dims_reassemble[1], kernel_size=1),
                    nn.ConvTranspose2d(
                        dims_reassemble[1], dims_reassemble[1], kernel_size=2, stride=2
                    ),
                ),
                # 1/16.
                nn.Sequential(nn.Conv2d(dim, dims_reassemble[2], kernel_size=1)),
                # 1/16 -> 1/32.
                nn.Sequential(
                    nn.Conv2d(dim, dims_reassemble[3], kernel_size=1),
                    nn.Conv2d(
                        dims_reassemble[3],
                        dims_reassemble[3],
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    ),
                ),
            ]
        )

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Patch-embed frames ``(T, 3, H, W)`` and prepend the class and register tokens."""
        tokens = self.patch_embed(x).flatten(1, 2)
        num_frames = tokens.shape[0]
        prefix = [
            self.cls_token.expand(num_frames, -1, -1),
            self.reg_token.expand(num_frames, -1, -1),
        ]
        return torch.cat(prefix + [tokens], dim=1)

    def cross_frame_attention(
        self, block: nn.Module, tokens: torch.Tensor, rope: torch.Tensor
    ) -> torch.Tensor:
        """Run one block jointly over the patch tokens of all frames."""
        num_frames, _, dim = tokens.shape
        npt = self.num_prefix_tokens
        prefix, patches = tokens[:, :npt], tokens[:, npt:]
        x = torch.cat(
            [prefix.mean(dim=0, keepdim=True), patches.reshape(1, -1, dim)], dim=1
        )
        x = block(x, rope=rope.repeat(num_frames, *(1,) * (rope.ndim - 1)))
        return torch.cat([prefix, x[:, npt:].reshape(num_frames, -1, dim)], dim=1)

    def cached_cross_frame_attention(
        self, idx: int, tokens: torch.Tensor, rope: torch.Tensor, cache: KVCache
    ) -> torch.Tensor:
        """Run block ``idx`` on one frame, its patches attending the cached frames' as well."""
        block = self.blocks[idx]
        npt = self.num_prefix_tokens
        prefix, patches = tokens[:, :npt], tokens[:, npt:]
        query, key, value = eva_qkv_rope(block, patches, rope, 0)
        pooled = cache.update_prefix(idx, prefix)
        _, pooled_key, pooled_value = eva_qkv_rope(block, pooled, None, npt)
        keys, values = cache.update(idx, key, value, pooled_key, pooled_value)
        attended = F.scaled_dot_product_attention(query, keys, values)
        return torch.cat([prefix, eva_finish(block, patches, attended)], dim=1)

    def encode(
        self, x: torch.Tensor, cache: KVCache | None = None
    ) -> tuple[list[torch.Tensor], tuple[int, int]]:
        """Run the blocks over frames ``(T, 3, H, W)``, returning tapped tokens and the grid.

        With a ``cache``, ``x`` is one frame ``(1, 3, H, W)``, attending the cached frames too.
        """
        grid_size = (x.shape[-2] // self.patch_size, x.shape[-1] // self.patch_size)
        rope = self.rope.get_embed(shape=grid_size)

        tokens = self.embed(x)
        features = []
        for idx, block in enumerate(self.blocks):
            if idx not in self.cross_frame_blocks:
                tokens = block(tokens, rope=rope)
            elif cache is None:
                tokens = self.cross_frame_attention(block, tokens, rope)
            else:
                tokens = self.cached_cross_frame_attention(idx, tokens, rope, cache)
            if idx in self.reassembled_blocks:
                features.append(tokens[:, self.num_prefix_tokens :])
        return features, grid_size

    def reassemble_tokens(
        self, features: Sequence[torch.Tensor], grid_size: tuple[int, int]
    ) -> list[torch.Tensor]:
        """Reassemble tapped tokens into the 4-level pyramid. Independent per frame."""
        num_frames = features[0].shape[0]
        return [
            # Without .contiguous() the view is channels-last and results differ in the last bits.
            layer(
                feature.transpose(1, 2).reshape(num_frames, -1, *grid_size).contiguous()
            )
            for layer, feature in zip(self.reassemble, features)
        ]


def autocast_for(device: torch.device, precision: torch.dtype) -> partial:
    """Autocast on ``device`` at ``precision``, disabled when that precision is float32."""
    return partial(
        torch.autocast,
        device_type=device.type,
        dtype=precision,
        enabled=precision != torch.float32,
    )
