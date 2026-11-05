# Adapted from TRELLIS (https://github.com/microsoft/TRELLIS, commit
# 193e0502b5cb3eb5997ec457d5340e95d09b158e), MIT, Copyright (c) Microsoft
# Corporation: ``trellis/modules/sparse``.
"""Sparse voxel tensors and the operations TRELLIS runs on them, in plain PyTorch.

Upstream builds these on spconv (sparse convolution) and xformers or
flash-attn (variable-length attention), CUDA-only extensions. Here every
operation is PyTorch, so the same code runs on CUDA, Apple MPS and CPU:

- ``SparseTensor`` keeps upstream's contract: ``feats`` ``[T, ...]``,
  ``coords`` ``[T, 4]`` int32 rows ``(batch, x, y, z)``, the rows of one
  batch element contiguous, and elementwise arithmetic that broadcasts a
  ``[B, C]`` tensor per batch element.
- ``SubMConv3d`` is spconv's submanifold convolution (kernel 3, stride 1):
  each active voxel sums its active neighbours only, so the output keeps the
  input's coordinates. It runs as one gathered GEMM over the 27 neighbours.
- ``sparse_downsample`` / ``sparse_upsample`` are upstream's mean pooling and
  its nearest-neighbour inverse.
- Attention runs through ``F.scaled_dot_product_attention``: per batch element
  for full attention, and over padded windows with a key mask for the
  shifted-window attention of the Gaussian decoder.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Rows of the gathered GEMM per chunk; bounds the [rows, 27 * C_in] buffer.
_CONV_CHUNK_ROWS = 4096


def _layout(coords: torch.Tensor, batch_size: int) -> List[slice]:
    counts = torch.bincount(coords[:, 0].long(), minlength=batch_size).tolist()
    layout, start = [], 0
    for count in counts:
        layout.append(slice(start, start + count))
        start += count
    return layout


class SparseTensor:
    """Features on a set of active voxels: ``feats`` ``[T, ...]`` and ``coords`` ``[T, 4]`` (batch, x, y, z)."""

    def __init__(
        self,
        feats: torch.Tensor,
        coords: torch.Tensor,
        *,
        batch_size: Optional[int] = None,
        layout: Optional[List[slice]] = None,
        cache: Optional[Dict] = None,
        pools: Tuple = (),
    ):
        self.feats = feats
        self.coords = coords
        if batch_size is None:
            batch_size = int(coords[:, 0].max().item()) + 1 if coords.shape[0] else 1
        self.batch_size = batch_size
        self.layout = layout if layout is not None else _layout(coords, batch_size)
        # Shared by every tensor derived through replace(): window partitions.
        self.cache = cache if cache is not None else {}
        # Stack of the downsamplings this tensor went through, for upsampling.
        self.pools = pools

    @property
    def shape(self) -> torch.Size:
        return torch.Size([self.batch_size, *self.feats.shape[1:]])

    @property
    def dtype(self) -> torch.dtype:
        return self.feats.dtype

    @property
    def device(self) -> torch.device:
        return self.feats.device

    def replace(
        self, feats: torch.Tensor, coords: Optional[torch.Tensor] = None
    ) -> "SparseTensor":
        return SparseTensor(
            feats,
            self.coords if coords is None else coords,
            batch_size=self.batch_size,
            layout=self.layout,
            cache=self.cache,
            pools=self.pools,
        )

    def type(self, dtype: torch.dtype) -> "SparseTensor":
        return self.replace(self.feats.type(dtype))

    def float(self) -> "SparseTensor":
        return self.replace(self.feats.float())

    def to(self, dtype: torch.dtype) -> "SparseTensor":
        return self.replace(self.feats.to(dtype))

    def reshape(self, *shape) -> "SparseTensor":
        return self.replace(self.feats.reshape(self.feats.shape[0], *shape))

    def unbind(self, dim: int) -> List["SparseTensor"]:
        return [self.replace(part) for part in self.feats.unbind(dim)]

    def _batch_rows(self) -> torch.Tensor:
        rows = self.cache.get("batch_rows")
        if rows is None or rows.shape[0] != self.feats.shape[0]:
            rows = self.coords[:, 0].long()
            self.cache["batch_rows"] = rows
        return rows

    def _operand(self, other):
        """Upstream's broadcast rule: a tensor shaped like ``self.shape`` applies per batch element."""
        if isinstance(other, SparseTensor):
            return other.feats
        if isinstance(other, torch.Tensor):
            try:
                expanded = torch.broadcast_to(other, self.shape)
            except RuntimeError:
                return other
            return expanded.to(self.feats.dtype)[self._batch_rows()]
        return other

    def __add__(self, other):
        return self.replace(torch.add(self.feats, self._operand(other)))

    __radd__ = __add__

    def __sub__(self, other):
        return self.replace(torch.sub(self.feats, self._operand(other)))

    def __rsub__(self, other):
        return self.replace(torch.sub(self._operand(other), self.feats))

    def __mul__(self, other):
        return self.replace(torch.mul(self.feats, self._operand(other)))

    __rmul__ = __mul__

    def __truediv__(self, other):
        return self.replace(torch.div(self.feats, self._operand(other)))

    def __neg__(self):
        return self.replace(-self.feats)


# -- Convolution -------------------------------------------------------------


def _neighbour_rows(coords: torch.Tensor, batch_size: int) -> torch.Tensor:
    """For each of the 27 offsets of a 3x3x3 kernel, the row of each voxel's neighbour, or -1.

    Offset ``i`` is ``(dx, dy, dz) = (i // 9 - 1, i // 3 % 3 - 1, i % 3 - 1)``,
    the order of spconv's ``[out, kx, ky, kz, in]`` weight layout.
    """
    device = coords.device
    xyz = coords[:, 1:].long()
    size = (xyz.max(dim=0).values + 3).tolist() if coords.shape[0] else [3, 3, 3]
    grid = torch.full((batch_size, *size), -1, dtype=torch.long, device=device)
    b = coords[:, 0].long()
    rows = torch.arange(coords.shape[0], device=device)
    grid[b, xyz[:, 0] + 1, xyz[:, 1] + 1, xyz[:, 2] + 1] = rows
    out = torch.empty(27, coords.shape[0], dtype=torch.long, device=device)
    for index in range(27):
        dx, dy, dz = index // 9, index // 3 % 3, index % 3
        out[index] = grid[b, xyz[:, 0] + dx, xyz[:, 1] + dy, xyz[:, 2] + dz]
    return out


class SubMConv3d(nn.Module):
    """spconv's ``SubMConv3d(in, out, 3)``: a 3x3x3 convolution evaluated on the input's active voxels.

    ``weight`` has spconv's layout ``[out, 3, 3, 3, in]`` so checkpoints load
    unchanged. The 27 neighbour features are gathered side by side (a missing
    neighbour contributes zeros) and multiplied by the flattened kernel in one
    GEMM per chunk of rows.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        bias: bool = True,
    ):
        super().__init__()
        if kernel_size != 3:
            raise ValueError("TRELLIS uses 3x3x3 submanifold convolutions only.")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.empty(out_channels, 3, 3, 3, in_channels))
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        nn.init.kaiming_uniform_(self.weight.view(out_channels, -1), a=math.sqrt(5))

    def forward(self, x: SparseTensor) -> SparseTensor:
        key = ("subm_neighbours", x.coords.data_ptr(), x.coords.shape[0])
        neighbours = x.cache.get(key)
        if neighbours is None:
            neighbours = _neighbour_rows(x.coords, x.batch_size)
            x.cache[key] = neighbours
        feats = x.feats
        kernel = self.weight.reshape(self.out_channels, 27 * self.in_channels).t()
        padded = torch.cat([feats, feats.new_zeros(1, feats.shape[1])], dim=0)
        missing = feats.shape[0]
        out = feats.new_empty(feats.shape[0], self.out_channels)
        for start in range(0, feats.shape[0], _CONV_CHUNK_ROWS):
            stop = min(start + _CONV_CHUNK_ROWS, feats.shape[0])
            index = neighbours[:, start:stop].clone()
            index[index < 0] = missing
            gathered = padded[index.t().reshape(-1)].reshape(
                stop - start, 27 * self.in_channels
            )
            if self.bias is not None:
                out[start:stop] = torch.addmm(self.bias, gathered, kernel)
            else:
                out[start:stop] = gathered @ kernel
        return x.replace(out)


def sparse_downsample(x: SparseTensor, factor: int = 2) -> SparseTensor:
    """Average the features of the voxels that fall in the same coarse voxel (upstream's ``SparseDownsample``)."""
    coord = list(x.coords.unbind(dim=-1))
    for i in range(1, 4):
        coord[i] = coord[i] // factor
    extent = [int(coord[i].max().item()) + 1 for i in range(1, 4)]
    offset = torch.cumprod(torch.tensor(extent[::-1]), 0).tolist()[::-1] + [1]
    code = sum(c.long() * o for c, o in zip(coord, offset))
    code, index = code.unique(return_inverse=True)
    feats = torch.scatter_reduce(
        torch.zeros(code.shape[0], x.feats.shape[1], device=x.device, dtype=x.dtype),
        dim=0,
        index=index.unsqueeze(1).expand(-1, x.feats.shape[1]),
        src=x.feats,
        reduce="mean",
    )
    coords = torch.stack(
        [code // offset[0]] + [(code // offset[i + 1]) % extent[i] for i in range(3)],
        dim=-1,
    ).to(x.coords.dtype)
    return SparseTensor(
        feats,
        coords,
        batch_size=x.batch_size,
        cache={},
        pools=x.pools + ((x.coords, x.layout, index, x.cache),),
    )


def sparse_upsample(x: SparseTensor) -> SparseTensor:
    """Copy each coarse voxel's features back to the fine voxels it pooled (upstream's ``SparseUpsample``)."""
    if not x.pools:
        raise ValueError(
            "sparse_upsample needs a tensor that went through sparse_downsample."
        )
    coords, layout, index, cache = x.pools[-1]
    return SparseTensor(
        x.feats[index],
        coords,
        batch_size=x.batch_size,
        layout=layout,
        cache=cache,
        pools=x.pools[:-1],
    )


# -- Attention ---------------------------------------------------------------


def _sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """[N, L, H, C] in and out, PyTorch's fused attention in between."""
    out = F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=mask
    )
    return out.transpose(1, 2)


def sparse_self_attention(qkv: SparseTensor) -> SparseTensor:
    """Full self attention within each batch element: qkv feats ``[T, 3, H, C]`` -> ``[T, H, C]``."""
    q, k, v = qkv.feats.unbind(dim=1)
    if qkv.batch_size == 1:
        out = _sdpa(q[None], k[None], v[None])[0]
    else:
        out = torch.cat(
            [_sdpa(q[s][None], k[s][None], v[s][None])[0] for s in qkv.layout], dim=0
        )
    return qkv.replace(out)


def sparse_cross_attention(q: SparseTensor, kv: torch.Tensor) -> SparseTensor:
    """Sparse queries ``[T, H, C]`` attend to dense keys and values ``[B, L, 2, H, C]`` of their batch element."""
    k, v = kv.unbind(dim=2)
    parts = [
        _sdpa(q.feats[s][None], k[i : i + 1], v[i : i + 1])[0]
        for i, s in enumerate(q.layout)
    ]
    return q.replace(parts[0] if len(parts) == 1 else torch.cat(parts, dim=0))


def window_partition(
    x: SparseTensor, window_size: int, shift: int
) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    """Upstream's ``calc_window_partition``: order the voxels by window; return (fwd, bwd, window lengths)."""
    shifted = x.coords.clone()
    shifted[:, 1:] += shift
    max_coords = shifted[:, 1:].max(dim=0).values.tolist()
    num_windows = [math.ceil((m + 1) / window_size) for m in max_coords]
    offset = torch.cumprod(torch.tensor([1] + num_windows[::-1]), dim=0).tolist()[::-1]
    shifted[:, 1:] //= window_size
    keys = (shifted.long() * torch.tensor(offset, device=x.device).unsqueeze(0)).sum(
        dim=1
    )
    fwd = torch.argsort(keys)
    bwd = torch.empty_like(fwd)
    bwd[fwd] = torch.arange(fwd.shape[0], device=x.device)
    lengths = torch.bincount(keys)
    lengths = lengths[lengths != 0].tolist()
    return fwd, bwd, lengths


def sparse_windowed_self_attention(
    qkv: SparseTensor, window_size: int, shift: int
) -> SparseTensor:
    """Self attention within cubic windows of ``window_size`` voxels, the grid shifted by ``shift``.

    Windows hold different numbers of voxels, so they are padded to the
    longest one and the padding is masked out of the keys.
    """
    key = ("window_partition", window_size, shift)
    partition = qkv.cache.get(key)
    if partition is None:
        fwd, bwd, lengths = window_partition(qkv, window_size, shift)
        length_tensor = torch.tensor(lengths, device=qkv.device)
        longest = max(lengths)
        starts = torch.cumsum(length_tensor, 0) - length_tensor
        slot = torch.arange(fwd.shape[0], device=qkv.device) - torch.repeat_interleave(
            starts, length_tensor
        )
        window = torch.repeat_interleave(
            torch.arange(len(lengths), device=qkv.device), length_tensor
        )
        flat = window * longest + slot
        valid = torch.zeros(len(lengths) * longest, dtype=torch.bool, device=qkv.device)
        valid[flat] = True
        mask = valid.view(len(lengths), 1, 1, longest)
        partition = (fwd, bwd, flat, mask, len(lengths), longest)
        qkv.cache[key] = partition
    fwd, bwd, flat, mask, windows, longest = partition
    feats = qkv.feats[fwd]
    _, three, heads, channels = feats.shape
    padded = feats.new_zeros(windows * longest, three, heads, channels)
    padded[flat] = feats
    padded = padded.view(windows, longest, three, heads, channels)
    q, k, v = padded.unbind(dim=2)
    out = _sdpa(q, k, v, mask).reshape(windows * longest, heads, channels)[flat]
    return qkv.replace(out[bwd])
