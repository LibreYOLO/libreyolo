"""Transformer decoder primitives used by DDColor.

Adapted from ``piddnad/DDColor`` commit
``2adb63f2656ac41cbdf7b894cddd94121a3faf13`` (Apache-2.0). The attention
layers follow ``facebookresearch/Mask2Former`` commit
``9b0651c6c1d5b3af2e6da0589b719c514ec0d69a`` (MIT), and the positional
encoding follows ``facebookresearch/detr`` commit
``29901c51d7fe8712168b8d0d64351170bc0f83e0`` (Apache-2.0). See ``NOTICE``.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class SelfAttentionLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dropout: float = 0.0,
        activation: str = "relu",
        normalize_before: bool = False,
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)

    @staticmethod
    def with_pos_embed(tensor: Tensor, pos: Optional[Tensor]) -> Tensor:
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        target: Tensor,
        target_mask: Optional[Tensor] = None,
        target_key_padding_mask: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ) -> Tensor:
        query = key = self.with_pos_embed(target, query_pos)
        update = self.self_attn(
            query,
            key,
            value=target,
            attn_mask=target_mask,
            key_padding_mask=target_key_padding_mask,
        )[0]
        target = target + self.dropout(update)
        return self.norm(target)

    def forward_pre(
        self,
        target: Tensor,
        target_mask: Optional[Tensor] = None,
        target_key_padding_mask: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ) -> Tensor:
        normalized = self.norm(target)
        query = key = self.with_pos_embed(normalized, query_pos)
        update = self.self_attn(
            query,
            key,
            value=normalized,
            attn_mask=target_mask,
            key_padding_mask=target_key_padding_mask,
        )[0]
        return target + self.dropout(update)

    def forward(
        self,
        target: Tensor,
        target_mask: Optional[Tensor] = None,
        target_key_padding_mask: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ) -> Tensor:
        if self.normalize_before:
            return self.forward_pre(
                target,
                target_mask,
                target_key_padding_mask,
                query_pos,
            )
        return self.forward_post(
            target,
            target_mask,
            target_key_padding_mask,
            query_pos,
        )


class CrossAttentionLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dropout: float = 0.0,
        activation: str = "relu",
        normalize_before: bool = False,
    ) -> None:
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)

    @staticmethod
    def with_pos_embed(tensor: Tensor, pos: Optional[Tensor]) -> Tensor:
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        target: Tensor,
        memory: Tensor,
        memory_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ) -> Tensor:
        update = self.multihead_attn(
            query=self.with_pos_embed(target, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        target = target + self.dropout(update)
        return self.norm(target)

    def forward_pre(
        self,
        target: Tensor,
        memory: Tensor,
        memory_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ) -> Tensor:
        normalized = self.norm(target)
        update = self.multihead_attn(
            query=self.with_pos_embed(normalized, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        return target + self.dropout(update)

    def forward(
        self,
        target: Tensor,
        memory: Tensor,
        memory_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ) -> Tensor:
        if self.normalize_before:
            return self.forward_pre(
                target,
                memory,
                memory_mask,
                memory_key_padding_mask,
                pos,
                query_pos,
            )
        return self.forward_post(
            target,
            memory,
            memory_mask,
            memory_key_padding_mask,
            pos,
            query_pos,
        )


class FFNLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.0,
        activation: str = "relu",
        normalize_before: bool = False,
    ) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)

    def forward_post(self, target: Tensor) -> Tensor:
        update = self.linear2(self.dropout(self.activation(self.linear1(target))))
        return self.norm(target + self.dropout(update))

    def forward_pre(self, target: Tensor) -> Tensor:
        normalized = self.norm(target)
        update = self.linear2(self.dropout(self.activation(self.linear1(normalized))))
        return target + self.dropout(update)

    def forward(self, target: Tensor) -> Tensor:
        if self.normalize_before:
            return self.forward_pre(target)
        return self.forward_post(target)


def _get_activation_fn(activation: str):
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu/glu, not {activation}.")


class MLP(nn.Module):
    """Simple multi-layer perceptron used for color embeddings."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        hidden = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(in_features, out_features)
            for in_features, out_features in zip(
                [input_dim] + hidden,
                hidden + [output_dim],
            )
        )

    def forward(self, x: Tensor) -> Tensor:
        for index, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if index < self.num_layers - 1 else layer(x)
        return x


class PositionEmbeddingSine(nn.Module):
    """Two-dimensional sine/cosine positional encoding from DETR."""

    def __init__(
        self,
        num_pos_feats: int = 64,
        temperature: int = 10000,
        normalize: bool = False,
        scale: float | None = None,
    ) -> None:
        super().__init__()
        if scale is not None and not normalize:
            raise ValueError("normalize must be True when scale is provided.")
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = 2 * math.pi if scale is None else scale

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        if mask is None:
            mask = torch.zeros(
                (x.size(0), x.size(2), x.size(3)),
                device=x.device,
                dtype=torch.bool,
            )
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            epsilon = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + epsilon) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + epsilon) * self.scale

        dim_t = torch.arange(
            self.num_pos_feats,
            dtype=torch.float32,
            device=x.device,
        )
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()),
            dim=4,
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()),
            dim=4,
        ).flatten(3)
        return torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)


__all__ = [
    "CrossAttentionLayer",
    "FFNLayer",
    "MLP",
    "PositionEmbeddingSine",
    "SelfAttentionLayer",
]
