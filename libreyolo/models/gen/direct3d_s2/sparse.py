# Adapted from Direct3D-S2 (https://github.com/DreamTechAI/Direct3D-S2, commit
# a1cf235b2881cff04a91900060a9546b40e7ee5d), MIT, Copyright (c) 2025
# DreamTechAI: ``direct3d_s2/modules/sparse``. The spatial sparse attention
# follows ``direct3d_s2/modules/sparse/attention/spatial_sparse_attention``,
# whose compression and selection kernels are adapted there from
# native-sparse-attention-triton (https://github.com/XunhaoLai/native-sparse-attention-triton),
# Apache-2.0, Copyright 2025 Xunhao Lai and Jianqiao Lu.
"""Direct3D-S2's sparse voxel operations, in plain PyTorch.

Upstream runs these on torchsparse (sparse convolution), flash-attn
(variable-length attention) and Triton kernels (spatial sparse attention),
all CUDA-only. Here each is PyTorch, so the same code runs on CUDA, Apple MPS
and CPU. The voxel tensor itself is the ``SparseTensor`` of the TRELLIS port,
whose contract Direct3D-S2 shares.

- ``SparseConv3d`` is torchsparse's ``Conv3d`` at stride 1: a submanifold
  convolution (the output keeps the input's voxels), with torchsparse's weight
  layout ``kernel`` ``[27, in, out]`` (``[in, out]`` for a 1x1x1 kernel) and its
  offset order (x fastest), so checkpoints load unchanged. It runs as one
  gathered GEMM over the 27 neighbours.
- ``sparse_mean_pool`` is upstream's ``SparseDownsample``: the mean of the
  voxels in each coarse voxel, accumulated in float64 as upstream does (float32
  on MPS, which has no float64).
- ``sparse_subdivide`` is upstream's ``SparseSubdivide``: each voxel becomes its
  eight children.
- ``spatial_sparse_attention`` is upstream's SSA (compression, selection and
  window branches with grouped-query heads), computed in chunks of queries
  with ``torch.matmul`` and ``F.scaled_dot_product_attention``.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..trellis.sparse import SparseTensor, window_partition

# Rows of the gathered GEMM per chunk; bounds the [rows, 27 * C_in] buffer.
_CONV_CHUNK_ROWS = 8192
# Elements of the per-chunk attention buffers (logits, gathered keys).
_ATTENTION_CHUNK_ELEMENTS = 1 << 27

# torchsparse's kernel order for a 3x3x3 kernel: offset j is
# (dx, dy, dz) = (j % 3 - 1, j // 3 % 3 - 1, j // 9 - 1).
_OFFSETS = torch.tensor(
    [(j % 3 - 1, j // 3 % 3 - 1, j // 9 - 1) for j in range(27)], dtype=torch.long
)


def _voxel_codes(coords: torch.Tensor, extent: List[int]) -> torch.Tensor:
    """One int64 per voxel, ordered like ``(batch, x, y, z)`` lexicographically."""
    c = coords.long()
    return ((c[:, 0] * extent[0] + c[:, 1]) * extent[1] + c[:, 2]) * extent[2] + c[:, 3]


def neighbour_rows(coords: torch.Tensor) -> torch.Tensor:
    """For each of torchsparse's 27 offsets, the row of each voxel's neighbour, or -1.

    Rows are found by binary search over sorted voxel codes, so memory grows
    with the number of voxels, not with the grid.
    """
    device = coords.device
    count = coords.shape[0]
    if count == 0:
        return torch.empty(27, 0, dtype=torch.long, device=device)
    shifted = coords.long().clone()
    shifted[:, 1:] += 1
    extent = (shifted[:, 1:].max(dim=0).values + 2).tolist()
    codes = _voxel_codes(shifted, extent)
    sorted_codes, order = torch.sort(codes)
    out = torch.empty(27, count, dtype=torch.long, device=device)
    for index, (dx, dy, dz) in enumerate(_OFFSETS.tolist()):
        neighbour = shifted.clone()
        neighbour[:, 1] += dx
        neighbour[:, 2] += dy
        neighbour[:, 3] += dz
        query = _voxel_codes(neighbour, extent)
        position = torch.searchsorted(sorted_codes, query).clamp_(max=count - 1)
        found = sorted_codes[position] == query
        out[index] = torch.where(found, order[position], torch.full_like(query, -1))
    return out


class _TorchsparseConv(nn.Module):
    """Parameters named and shaped like torchsparse's ``Conv3d`` (``kernel``, ``bias``)."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, bias: bool):
        super().__init__()
        if kernel_size not in (1, 3):
            raise ValueError("Direct3D-S2 uses 1x1x1 and 3x3x3 sparse convolutions only.")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        shape = (
            (in_channels, out_channels)
            if kernel_size == 1
            else (27, in_channels, out_channels)
        )
        self.kernel = nn.Parameter(torch.zeros(*shape))
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        bound = 1 / math.sqrt(in_channels * (1 if kernel_size == 1 else 27))
        nn.init.uniform_(self.kernel, -bound, bound)

    def forward(self, x: SparseTensor) -> SparseTensor:
        feats = x.feats
        if self.kernel_size == 1:
            out = feats @ self.kernel
            return x.replace(out + self.bias if self.bias is not None else out)
        key = ("torchsparse_neighbours", x.coords.data_ptr(), x.coords.shape[0])
        neighbours = x.cache.get(key)
        if neighbours is None:
            neighbours = neighbour_rows(x.coords)
            x.cache[key] = neighbours
        kernel = self.kernel.reshape(27 * self.in_channels, self.out_channels)
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


class SparseConv3d(nn.Module):
    """Upstream's ``SparseConv3d`` wrapper (keeps the ``.conv.`` parameter prefix), stride 1."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int = 0,
        bias: bool = True,
    ):
        super().__init__()
        del padding  # torchsparse pads odd kernels at stride 1 itself
        self.conv = _TorchsparseConv(in_channels, out_channels, kernel_size, bias)

    def forward(self, x: SparseTensor) -> SparseTensor:
        return self.conv(x)


def _float64_capable(device: torch.device) -> bool:
    return device.type != "mps"


def sparse_mean_pool(x: SparseTensor, factor: int) -> SparseTensor:
    """Upstream's ``SparseDownsample``: average the voxels that share a coarse voxel.

    Coarse voxels come out in ``(batch, x, y, z)`` order. As upstream's
    ``scatter_reduce`` into zeros, the zero it starts from counts in the mean
    (a coarse voxel of ``n`` voxels divides their sum by ``n + 1``). The mean
    is accumulated in float64 as upstream does, float32 on MPS.
    """
    coord = list(x.coords.unbind(dim=-1))
    for i in range(1, 4):
        coord[i] = coord[i] // factor
    extent = [int(coord[i].max().item()) + 1 for i in range(1, 4)]
    offset = torch.cumprod(torch.tensor(extent[::-1]), 0).tolist()[::-1] + [1]
    code = sum(c.long() * o for c, o in zip(coord, offset))
    code, index = code.unique(return_inverse=True)
    accumulate = torch.float64 if _float64_capable(x.device) else torch.float32
    feats = torch.scatter_reduce(
        torch.zeros(code.shape[0], x.feats.shape[1], device=x.device, dtype=accumulate),
        dim=0,
        index=index.unsqueeze(1).expand(-1, x.feats.shape[1]),
        src=x.feats.to(accumulate),
        reduce="mean",
    ).to(x.dtype)
    coords = torch.stack(
        [code // offset[0]] + [(code // offset[i + 1]) % extent[i] for i in range(3)],
        dim=-1,
    ).to(x.coords.dtype)
    return SparseTensor(feats, coords, batch_size=x.batch_size)


def sparse_subdivide(x: SparseTensor) -> SparseTensor:
    """Upstream's ``SparseSubdivide``: each voxel becomes its eight children, ``z`` fastest."""
    children = torch.nonzero(torch.ones(2, 2, 2, device=x.device, dtype=torch.int))
    children = torch.cat([torch.zeros_like(children[:, :1]), children], dim=-1)
    coords = x.coords.clone()
    coords[:, 1:] *= 2
    coords = coords.unsqueeze(1) + children.unsqueeze(0).to(coords.dtype)
    feats = x.feats.unsqueeze(1).expand(x.feats.shape[0], 8, *x.feats.shape[1:])
    return SparseTensor(
        feats.flatten(0, 1), coords.flatten(0, 1), batch_size=x.batch_size
    )


def sort_block(latent_index: torch.Tensor, block_size: int) -> torch.Tensor:
    """Upstream's ``sort_block``: order voxels by selection block, then by position inside it."""
    device = latent_index.device
    coords = latent_index.cpu().numpy()
    block = coords[..., 1:] // block_size
    inner = coords[..., 1:] % block_size
    order = np.lexsort(
        (inner[:, 2], inner[:, 1], inner[:, 0], block[:, 2], block[:, 1], block[:, 0])
    )
    return latent_index[torch.from_numpy(order).to(device)]


# -- Spatial sparse attention ------------------------------------------------


def _chunk_rows(per_row: int) -> int:
    return max(1, _ATTENTION_CHUNK_ELEMENTS // max(per_row, 1))


def compressed_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, topk: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Queries attend to the compressed blocks; each kv group then picks its ``topk`` blocks.

    ``q`` ``[N, Hq, C]``, ``k`` and ``v`` ``[M, Hkv, C]`` for one batch element.
    Returns the attention output ``[N, Hq, C]`` and the block indices
    ``[Hkv, N, topk]``, ascending, ``-1`` where fewer than ``topk`` blocks
    exist. A block's score is the attention probability summed over the query
    heads that share its kv head, as upstream's ``get_block_score``.
    """
    n, heads, channels = q.shape
    m, kv_heads = k.shape[:2]
    group = heads // kv_heads
    scale = 1.0 / math.sqrt(channels)
    keys = k.float().permute(1, 2, 0)  # [Hkv, C, M]
    values = v.permute(1, 0, 2)  # [Hkv, M, C]
    out = q.new_empty(n, heads, channels)
    real_topk = min(topk, m)
    selected = torch.full((kv_heads, n, topk), -1, dtype=torch.long, device=q.device)
    rows = _chunk_rows(heads * m)
    for start in range(0, n, rows):
        stop = min(start + rows, n)
        queries = q[start:stop].float().view(stop - start, kv_heads, group, channels)
        queries = queries.permute(1, 0, 2, 3).reshape(kv_heads, -1, channels)
        logits = torch.bmm(queries, keys).mul_(scale)  # [Hkv, rows*group, M]
        probs = torch.softmax(logits, dim=-1)
        attended = torch.bmm(probs.to(values.dtype), values)  # [Hkv, rows*group, C]
        out[start:stop] = (
            attended.view(kv_heads, stop - start, group, channels)
            .permute(1, 0, 2, 3)
            .reshape(stop - start, heads, channels)
        )
        score = probs.view(kv_heads, stop - start, group, m).sum(dim=2)
        top = score.topk(real_topk, dim=-1).indices.sort(dim=-1).values
        selected[:, start:stop, :real_topk] = top
    return out, selected


def selection_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    selected: torch.Tensor,
    block_starts: torch.Tensor,
    block_sizes: torch.Tensor,
) -> torch.Tensor:
    """Each query attends to the tokens of the blocks its kv group selected.

    ``selected`` ``[Hkv, N, topk]`` indexes blocks whose tokens are the
    contiguous rows ``block_starts[b] : block_starts[b] + block_sizes[b]``.
    The selected blocks' tokens are laid side by side per query, padded only
    to the longest such list in a chunk of queries.
    """
    n, heads, channels = q.shape
    kv_heads = k.shape[1]
    group = heads // kv_heads
    topk = selected.shape[-1]
    scale = 1.0 / math.sqrt(channels)
    out = q.new_empty(n, heads, channels)
    padded_k = torch.cat([k, k.new_zeros(1, kv_heads, channels)], dim=0)
    padded_v = torch.cat([v, v.new_zeros(1, kv_heads, channels)], dim=0)
    missing = k.shape[0]
    heads_index = torch.arange(kv_heads, device=q.device).view(1, kv_heads, 1)
    # Tokens each query attends to, per kv group: the sizes of its selected blocks.
    valid_block = selected >= 0
    sizes = torch.where(
        valid_block, block_sizes[selected.clamp(min=0)], torch.zeros_like(selected)
    ).permute(1, 0, 2)  # [N, Hkv, topk]
    ends = torch.cumsum(sizes, dim=-1)
    totals = ends[..., -1].amax(dim=1)  # [N]
    longest = int(totals.max().item()) if n else 0
    rows = _chunk_rows(kv_heads * max(longest, 1) * max(channels, group) * 4)
    for start in range(0, n, rows):
        stop = min(start + rows, n)
        width = max(int(totals[start:stop].max().item()), 1)
        slots = torch.arange(width, device=q.device)
        chunk_ends = ends[start:stop]  # [rows, Hkv, topk]
        which = torch.searchsorted(
            chunk_ends.contiguous(),
            slots.expand(stop - start, kv_heads, width).contiguous(),
            right=True,
        ).clamp_(max=topk - 1)
        offset = slots - (torch.gather(chunk_ends, 2, which) - torch.gather(sizes[start:stop], 2, which))
        blocks = torch.gather(selected[:, start:stop].permute(1, 0, 2), 2, which).clamp(min=0)
        valid = slots < chunk_ends[..., -1:]
        token = torch.where(valid, block_starts[blocks] + offset, torch.full_like(offset, missing))
        keys = padded_k[token, heads_index]  # [rows, Hkv, width, C]
        values = padded_v[token, heads_index]
        queries = q[start:stop].view(stop - start, kv_heads, group, channels)
        logits = torch.matmul(queries.float(), keys.float().transpose(-1, -2)).mul_(scale)
        logits = logits.masked_fill_(~valid.unsqueeze(2), float("-inf"))
        probs = torch.softmax(logits, dim=-1)
        out[start:stop] = torch.matmul(probs.to(values.dtype), values).reshape(
            stop - start, heads, channels
        )
    return out


def window_attention(
    x: SparseTensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    window_size: int,
    shift: int,
) -> torch.Tensor:
    """Self attention within cubic windows of ``window_size`` voxels, kv heads shared by query groups.

    Windows are padded to the longest one in a chunk and the padding is
    masked out of the keys.
    """
    key = ("ssa_window_partition", window_size, shift)
    partition = x.cache.get(key)
    if partition is None:
        fwd, bwd, lengths = window_partition(x, window_size, shift)
        partition = (fwd, bwd, lengths)
        x.cache[key] = partition
    fwd, bwd, lengths = partition
    heads, channels = q.shape[1:]
    group = heads // k.shape[1]
    q, k, v = q[fwd], k[fwd], v[fwd]
    out = torch.empty_like(q)
    starts = np.concatenate([[0], np.cumsum(lengths)]).tolist()
    windows = len(lengths)
    first = 0
    while first < windows:
        longest = lengths[first]
        last = first
        # Grow the chunk while the padded buffer stays within budget.
        while last + 1 < windows:
            candidate = max(longest, lengths[last + 1])
            if (last + 2 - first) * candidate * candidate * heads > _ATTENTION_CHUNK_ELEMENTS:
                break
            longest = candidate
            last += 1
        count = last + 1 - first
        lo, hi = starts[first], starts[last + 1]
        sizes = torch.tensor(lengths[first : last + 1], device=q.device)
        slot = torch.arange(hi - lo, device=q.device) - torch.repeat_interleave(
            torch.tensor(starts[first : last + 1], device=q.device) - lo, sizes
        )
        window = torch.repeat_interleave(torch.arange(count, device=q.device), sizes)
        flat = window * longest + slot
        valid = torch.zeros(count * longest, dtype=torch.bool, device=q.device)
        valid[flat] = True
        mask = valid.view(count, 1, 1, longest)

        def pad(t: torch.Tensor) -> torch.Tensor:
            buffer = t.new_zeros(count * longest, *t.shape[1:])
            buffer[flat] = t[lo:hi]
            return buffer.view(count, longest, *t.shape[1:]).transpose(1, 2)

        kq = pad(q)
        kk = pad(k).repeat_interleave(group, dim=1)
        kv = pad(v).repeat_interleave(group, dim=1)
        attended = F.scaled_dot_product_attention(kq, kk, kv, attn_mask=mask)
        out[lo:hi] = attended.transpose(1, 2).reshape(count * longest, heads, channels)[flat]
        first = last + 1
    return out[bwd]


def selection_blocks(
    coords: torch.Tensor, block_size: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Start row and token count of each non-empty selection block of one batch element.

    The voxels must be sorted by block (``sort_block``), so each block's
    tokens are contiguous and the blocks come in ascending order.
    """
    block = coords[:, 1:].long() // block_size
    extent = int(block.max().item()) + 1 if block.numel() else 1
    flat = (block[:, 0] * extent + block[:, 1]) * extent + block[:, 2]
    if flat.numel() > 1 and bool((flat[1:] < flat[:-1]).any()):
        raise ValueError("Spatial sparse attention needs voxels sorted by block (sort_block).")
    _, counts = torch.unique_consecutive(flat, return_counts=True)
    starts = torch.cumsum(counts, 0) - counts
    return starts, counts


def spatial_sparse_attention(
    x: SparseTensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    compressed_k: SparseTensor,
    compressed_v: SparseTensor,
    *,
    topk: int,
    selection_block_size: int,
    compression_block_size: int,
    window_size: int,
    shift: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The three SSA branches for every batch element: (compressed, selected, windowed) outputs ``[T, Hq, C]``."""
    if selection_block_size != compression_block_size:
        raise ValueError(
            "The released Direct3D-S2 models select blocks of the compression size; "
            "other selection sizes are not supported."
        )
    compressed_out = torch.empty_like(q)
    selected_out = torch.empty_like(q)
    for index, rows in enumerate(x.layout):
        blocks = compressed_k.layout[index]
        out, selected = compressed_attention(
            q[rows], compressed_k.feats[blocks], compressed_v.feats[blocks], topk
        )
        compressed_out[rows] = out
        key = ("ssa_blocks", index, selection_block_size)
        cached = x.cache.get(key)
        if cached is None:
            cached = selection_blocks(x.coords[rows], selection_block_size)
            x.cache[key] = cached
        starts, counts = cached
        selected_out[rows] = selection_attention(
            q[rows], k[rows], v[rows], selected, starts, counts
        )
    windowed = window_attention(x, q, k, v, window_size, shift)
    return compressed_out, selected_out, windowed


__all__ = [
    "SparseTensor",
    "SparseConv3d",
    "neighbour_rows",
    "sort_block",
    "sparse_mean_pool",
    "sparse_subdivide",
    "spatial_sparse_attention",
    "compressed_attention",
    "selection_attention",
    "window_attention",
]
