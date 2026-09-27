"""Native BEN2 Base inference graph.

This is a dependency-free port of the MIT-licensed BEN2 Base network from
PramaLLC/BEN2. The public checkpoint contains a Swin-v1 encoder followed by
MVANet-style multi-field cross-attention and refinement modules. LibreYOLO
reuses its existing audited Swin-v1 implementation because its parameter
layout and fp32 outputs match BEN2 exactly, while the BEN2-specific decoder is
implemented here with native tensor reshapes instead of ``einops``.

Only the public Base mask network is included. Upstream video utilities and
the separately attributed optional foreground-colour refinement helper are not
part of this module.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..birefnet.nn import SwinTransformer


def _image_to_patches(x: torch.Tensor) -> torch.Tensor:
    """Split a ``(B,C,2H,2W)`` image into four spatial patches per image."""
    b, c, height, width = x.shape
    if height % 2 or width % 2:
        raise ValueError("BEN2 patch splitting requires even spatial dimensions.")
    h, w = height // 2, width // 2
    return (
        x.reshape(b, c, 2, h, 2, w)
        .permute(2, 4, 0, 1, 3, 5)
        .contiguous()
        .reshape(4 * b, c, h, w)
    )


def _patches_to_image(x: torch.Tensor) -> torch.Tensor:
    """Reassemble four spatial patches per image into ``(B,C,2H,2W)``."""
    patch_batch, c, h, w = x.shape
    if patch_batch % 4:
        raise ValueError("BEN2 patch reassembly requires a multiple of four patches.")
    b = patch_batch // 4
    return (
        x.reshape(2, 2, b, c, h, w)
        .permute(2, 3, 0, 4, 1, 5)
        .contiguous()
        .reshape(b, c, 2 * h, 2 * w)
    )


def _to_sequence(x: torch.Tensor) -> torch.Tensor:
    """Convert ``(B,C,H,W)`` to attention layout ``(H*W,B,C)``."""
    b, c, h, w = x.shape
    return x.permute(2, 3, 0, 1).contiguous().reshape(h * w, b, c)


def _from_sequence(x: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """Convert attention layout ``(H*W,B,C)`` to ``(B,C,H,W)``."""
    _, b, c = x.shape
    return x.reshape(h, w, b, c).permute(2, 3, 0, 1).contiguous()


def _make_conv_block(in_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1),
        nn.InstanceNorm2d(out_dim),
        nn.GELU(),
    )


def _rescale(
    x: torch.Tensor, scale_factor: float = 2, mode: str = "nearest"
) -> torch.Tensor:
    return F.interpolate(x, scale_factor=scale_factor, mode=mode)


def _resize_as(
    x: torch.Tensor, reference: torch.Tensor, mode: str = "bilinear"
) -> torch.Tensor:
    return F.interpolate(x, size=reference.shape[-2:], mode=mode)


class BEN2Backbone(SwinTransformer):
    """BEN2's five-output Swin-v1 encoder.

    BEN2 consumes the patch-embedding map as well as all four normalized stage
    outputs. The inherited implementation supplies the exact checkpoint key
    layout and attention arithmetic; this override only retains that extra
    first feature map.
    """

    def __init__(self) -> None:
        super().__init__(
            embed_dim=128,
            depths=(2, 2, 18, 2),
            num_heads=(4, 8, 16, 32),
            window_size=12,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        x = self.patch_embed(x)
        outs = [x.contiguous()]
        height, width = x.shape[2:]
        x = self.pos_drop(x.flatten(2).transpose(1, 2))

        for index, layer in enumerate(self.layers):
            x_out, h, w, x, height, width = layer(x, height, width)
            if index in self.out_indices:
                x_out = getattr(self, f"norm{index}")(x_out)
                out = (
                    x_out.view(-1, h, w, self.num_features[index])
                    .permute(0, 3, 1, 2)
                    .contiguous()
                )
                outs.append(out)
        return tuple(outs)


class PositionEmbeddingSine:
    """Two-dimensional sine/cosine positions used by BEN2 cross-attention."""

    def __init__(
        self,
        num_pos_feats: int = 64,
        temperature: int = 10000,
        normalize: bool = False,
        scale: float | None = None,
    ) -> None:
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and not normalize:
            raise ValueError("normalize must be True when scale is provided")
        self.scale = 2 * math.pi if scale is None else scale
        self.dim_t = torch.arange(0, num_pos_feats, dtype=torch.float32)

    def __call__(self, batch: int, height: int, width: int) -> torch.Tensor:
        device = self.dim_t.device
        not_mask = torch.ones((batch, height, width), dtype=torch.bool, device=device)
        y_embed = not_mask.cumsum(dim=1, dtype=torch.float32)
        x_embed = not_mask.cumsum(dim=2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = (y_embed - 0.5) / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = (x_embed - 0.5) / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = self.temperature ** (
            2 * (self.dim_t.to(device) // 2) / self.num_pos_feats
        )
        pos_x = x_embed[..., None] / dim_t
        pos_y = y_embed[..., None] / dim_t
        pos_x = torch.stack(
            (pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=4
        ).flatten(3)
        return torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)


class MultiFieldCrossAttention(nn.Module):
    """Fuse four local fields with the half-resolution global field."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        pool_ratios: tuple[int, ...] = (1, 4, 8),
    ) -> None:
        super().__init__()
        self.attention = nn.ModuleList(
            [nn.MultiheadAttention(d_model, num_heads, dropout=0.1) for _ in range(5)]
        )
        self.linear1 = nn.Linear(d_model, d_model * 2)
        self.linear2 = nn.Linear(d_model * 2, d_model)
        self.linear3 = nn.Linear(d_model, d_model * 2)
        self.linear4 = nn.Linear(d_model * 2, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.pool_ratios = pool_ratios
        self.positional_encoding = PositionEmbeddingSine(
            num_pos_feats=d_model // 2, normalize=True
        )
        # BEN2 export is intentionally fixed at 1024px. Precomputing the four
        # bottleneck position maps on CPU preserves upstream's exact fp32
        # arithmetic, while registered buffers follow the model for fp16 CUDA
        # tracing and artifact reload. They are derived constants, not weights.
        for size in (16, 4, 2):
            self.register_buffer(
                f"_export_pos_{size}",
                self.positional_encoding(1, size, size),
                persistent=False,
            )

    def forward(self, local: torch.Tensor, global_: torch.Tensor) -> torch.Tensor:
        _, _, h, w = local.shape
        concatenated_local = _patches_to_image(local)

        pools = []
        pool_positions = []
        export_positions = (
            self._export_pos_16,
            self._export_pos_4,
            self._export_pos_2,
        )
        for index, ratio in enumerate(self.pool_ratios):
            if torch.onnx.is_in_onnx_export():
                # concatenated_local is 2H x 2W, so this is exactly the
                # adaptive H/ratio x W/ratio partition with static kernels.
                factor = 2 * ratio
                pool = F.avg_pool2d(
                    concatenated_local, kernel_size=factor, stride=factor
                )
            else:
                pool = F.adaptive_avg_pool2d(
                    concatenated_local, (h // ratio, w // ratio)
                )
            pools.append(_to_sequence(pool))
            if torch.jit.is_tracing() or torch.onnx.is_in_onnx_export():
                pos = export_positions[index].expand(pool.shape[0], -1, -1, -1)
            else:
                pos = self.positional_encoding(pool.shape[0], *pool.shape[-2:])
            pool_positions.append(_to_sequence(pos))
        pooled = torch.cat(pools, dim=0)
        pooled_pos = torch.cat(pool_positions, dim=0).to(
            device=pooled.device, dtype=pooled.dtype
        )

        if torch.jit.is_tracing() or torch.onnx.is_in_onnx_export():
            global_pos = self._export_pos_16.expand(global_.shape[0], -1, -1, -1)
        else:
            global_pos = self.positional_encoding(
                global_.shape[0], global_.shape[2], global_.shape[3]
            )
        global_pos = _to_sequence(global_pos).to(
            device=global_.device, dtype=global_.dtype
        )
        global_seq = _to_sequence(global_)
        global_seq = global_seq + self.dropout1(
            self.attention[0](
                global_seq + global_pos,
                pooled + pooled_pos,
                pooled,
            )[0]
        )
        global_seq = self.norm1(global_seq)
        global_seq = global_seq + self.dropout2(
            self.linear2(self.dropout(F.gelu(self.linear1(global_seq)).clone()))
        )
        global_seq = self.norm2(global_seq)

        local_seq = _to_sequence(local)
        global_fields = _to_sequence(
            _image_to_patches(_from_sequence(global_seq, h, w))
        )
        refreshed = []
        for index, (local_field, global_field) in enumerate(
            zip(local_seq.chunk(4, dim=1), global_fields.chunk(4, dim=1))
        ):
            refreshed.append(
                self.attention[index + 1](local_field, global_field, global_field)[0]
            )
        local_seq = local_seq + self.dropout1(torch.cat(refreshed, dim=1))
        local_seq = self.norm1(local_seq)
        local_seq = local_seq + self.dropout2(
            self.linear4(self.dropout(F.gelu(self.linear3(local_seq)).clone()))
        )
        local_seq = self.norm2(local_seq)
        return _from_sequence(torch.cat((local_seq, global_seq), dim=1), h, w)


class MultiFieldRefinement(nn.Module):
    """Refine local fields using pooled global context."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        pool_ratios: tuple[int, ...] = (4, 8, 16),
    ) -> None:
        super().__init__()
        self.attention = nn.ModuleList(
            [nn.MultiheadAttention(d_model, num_heads, dropout=0.1) for _ in range(4)]
        )
        self.linear3 = nn.Linear(d_model, d_model * 2)
        self.linear4 = nn.Linear(d_model * 2, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.sigmoid = nn.Sigmoid()
        self.sal_conv = nn.Conv2d(d_model, 1, 1)
        self.pool_ratios = pool_ratios

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, c, h, w = x.shape
        local, global_ = x.split((4, 1), dim=0)
        patched_global = _image_to_patches(global_)

        token_attention = self.sigmoid(self.sal_conv(global_))
        token_attention = F.interpolate(
            token_attention,
            size=_patches_to_image(local).shape[-2:],
            mode="nearest",
        )
        local = local * _image_to_patches(token_attention)

        pools = []
        for ratio in self.pool_ratios:
            if torch.onnx.is_in_onnx_export():
                # patched_global is H/2 x W/2. The upstream adaptive target is
                # H/ratio x W/ratio, equivalent to this static average pool.
                factor = ratio // 2
                pool = F.avg_pool2d(patched_global, kernel_size=factor, stride=factor)
            else:
                pool = F.adaptive_avg_pool2d(patched_global, (h // ratio, w // ratio))
            pools.append(pool.flatten(2))
        pooled = torch.cat(pools, dim=2).permute(0, 2, 1).unsqueeze(2)
        local_queries = local.flatten(2).permute(0, 2, 1).unsqueeze(2)

        outputs = []
        for index, query in enumerate(local_queries.unbind(dim=0)):
            context = pooled[index]
            outputs.append(self.attention[index](query, context, context)[0])

        output = torch.cat(outputs, dim=1)
        source = local.view(4, c, -1).permute(2, 0, 1) + self.dropout1(output)
        source = self.norm1(source)
        source = source + self.dropout2(
            self.linear4(self.dropout(F.gelu(self.linear3(source)).clone()))
        )
        source = self.norm2(source)
        source = source.permute(1, 2, 0).reshape(4, c, h, w)
        global_ = global_ + F.interpolate(
            _patches_to_image(source), size=global_.shape[-2:], mode="nearest"
        )
        return torch.cat((source, global_), dim=0), token_attention


class LibreBEN2Model(nn.Module):
    """BEN2 Base: RGB image to a single-channel alpha logit map."""

    def __init__(self) -> None:
        super().__init__()
        emb_dim = 128
        self.backbone = BEN2Backbone()

        # Training side outputs are retained so the published checkpoint loads
        # strictly even though they are not used by the inference forward.
        self.sideout5 = nn.Sequential(nn.Conv2d(emb_dim, 1, 3, padding=1))
        self.sideout4 = nn.Sequential(nn.Conv2d(emb_dim, 1, 3, padding=1))
        self.sideout3 = nn.Sequential(nn.Conv2d(emb_dim, 1, 3, padding=1))
        self.sideout2 = nn.Sequential(nn.Conv2d(emb_dim, 1, 3, padding=1))
        self.sideout1 = nn.Sequential(nn.Conv2d(emb_dim, 1, 3, padding=1))

        self.output5 = _make_conv_block(1024, emb_dim)
        self.output4 = _make_conv_block(512, emb_dim)
        self.output3 = _make_conv_block(256, emb_dim)
        self.output2 = _make_conv_block(128, emb_dim)
        self.output1 = _make_conv_block(128, emb_dim)

        # Attribute names mirror the released state dict.
        self.multifieldcrossatt = MultiFieldCrossAttention(emb_dim, 1, (1, 4, 8))
        self.conv1 = _make_conv_block(emb_dim, emb_dim)
        self.conv2 = _make_conv_block(emb_dim, emb_dim)
        self.conv3 = _make_conv_block(emb_dim, emb_dim)
        self.conv4 = _make_conv_block(emb_dim, emb_dim)
        self.dec_blk1 = MultiFieldRefinement(emb_dim, 1, (2, 4, 8))
        self.dec_blk2 = MultiFieldRefinement(emb_dim, 1, (2, 4, 8))
        self.dec_blk3 = MultiFieldRefinement(emb_dim, 1, (2, 4, 8))
        self.dec_blk4 = MultiFieldRefinement(emb_dim, 1, (2, 4, 8))

        self.insmask_head = nn.Sequential(
            nn.Conv2d(emb_dim, 384, 3, padding=1),
            nn.InstanceNorm2d(384),
            nn.GELU(),
            nn.Conv2d(384, 384, 3, padding=1),
            nn.InstanceNorm2d(384),
            nn.GELU(),
            nn.Conv2d(384, emb_dim, 3, padding=1),
        )
        self.shallow = nn.Sequential(nn.Conv2d(3, emb_dim, 3, padding=1))
        self.upsample1 = _make_conv_block(emb_dim, emb_dim)
        self.upsample2 = _make_conv_block(emb_dim, emb_dim)
        self.output = nn.Sequential(nn.Conv2d(emb_dim, 1, 3, padding=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        real_batch = x.shape[0]
        shallow_batch = self.shallow(x)
        global_batch = _rescale(x, scale_factor=0.5, mode="bilinear")
        encoder_groups = [
            torch.cat((_image_to_patches(x[i : i + 1]), global_batch[i : i + 1]), dim=0)
            for i in range(real_batch)
        ]
        features = self.backbone(torch.cat(encoder_groups, dim=0))

        outputs = []
        for index in range(real_batch):
            start, end = index * 5, (index + 1) * 5
            e5 = self.output5(features[4][start:end])
            e4 = self.output4(features[3][start:end])
            e3 = self.output3(features[2][start:end])
            e2 = self.output2(features[1][start:end])
            e1 = self.output1(features[0][start:end])

            local_e5, global_e5 = e5.split((4, 1), dim=0)
            e5 = self.multifieldcrossatt(local_e5, global_e5)
            e4, _ = self.dec_blk4(e4 + _resize_as(e5, e4))
            e4 = self.conv4(e4)
            e3, _ = self.dec_blk3(e3 + _resize_as(e4, e3))
            e3 = self.conv3(e3)
            e2, _ = self.dec_blk2(e2 + _resize_as(e3, e2))
            e2 = self.conv2(e2)
            e1, _ = self.dec_blk1(e1 + _resize_as(e2, e1))
            e1 = self.conv1(e1)

            local_e1, global_e1 = e1.split((4, 1), dim=0)
            merged = _patches_to_image(local_e1)
            merged = merged + _resize_as(global_e1, merged)
            merged = self.insmask_head(merged)
            shallow = shallow_batch[index : index + 1]
            merged = merged + _resize_as(shallow, merged)
            merged = self.upsample1(_rescale(merged))
            merged = _rescale(merged + _resize_as(shallow, merged))
            merged = self.upsample2(merged)
            outputs.append(self.output(merged))

        # Return logits. The shared matte contract applies sigmoid exactly once
        # in postprocessing and in exported-runtime result construction.
        return torch.cat(outputs, dim=0)


__all__ = ["BEN2Backbone", "LibreBEN2Model"]
