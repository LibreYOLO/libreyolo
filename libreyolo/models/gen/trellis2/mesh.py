# Adapted from TRELLIS.2 (https://github.com/microsoft/TRELLIS.2, commit
# 75fbf0183001ed9876c8dbb35de6b68552ee08bd), MIT, Copyright (c) Microsoft
# Corporation: ``o-voxel/o_voxel/convert/flexible_dual_grid.py``.
"""Triangles from TRELLIS.2's flexible dual grid, in plain PyTorch.

Each occupied voxel holds one mesh vertex (its dual vertex, an offset inside
the voxel) and three flags: whether the surface crosses the voxel's edge
along x, y and z. A crossed edge is shared by four voxels; when all four are
occupied their dual vertices form a quad, split into two triangles along the
diagonal whose end vertices carry the larger product of split weights.

Upstream finds the four voxels through a CUDA hash map (o-voxel); here they
are found by binary search in the sorted voxel keys, which returns the same
rows, so the same triangles in the same order.
"""

from __future__ import annotations

from typing import Tuple

import torch

from ..trellis.sparse import lookup_rows, voxel_keys

# The four voxels around a voxel's x, y and z edge, as offsets from it.
_EDGE_NEIGHBOURS = (
    ((0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0)),
    ((0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)),
    ((0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0)),
)
_SPLIT_02 = (0, 1, 2, 0, 2, 3)
_SPLIT_13 = (0, 1, 3, 3, 1, 2)


def flexible_dual_grid_to_mesh(
    coords: torch.Tensor,
    dual_vertices: torch.Tensor,
    intersected: torch.Tensor,
    split_weight: torch.Tensor,
    resolution: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Mesh of one object in the cube ``[-0.5, 0.5]^3``.

    Args:
        coords: ``[N, 3]`` integer voxel coordinates in ``[0, resolution)``.
        dual_vertices: ``[N, 3]`` vertex offsets inside each voxel, in voxels.
        intersected: ``[N, 3]`` bool, the surface crosses the x/y/z edge.
        split_weight: ``[N, 1]`` positive quad split weights.
        resolution: Voxels per side of the cube.

    Returns:
        ``vertices`` ``[N, 3]`` float32 (one per voxel, as upstream) and
        ``faces`` ``[F, 3]`` int64.
    """
    device = coords.device
    n = coords.shape[0]
    coords = coords.int()
    size = [resolution + 2] * 3
    padded = torch.cat([torch.zeros_like(coords[:, :1]), coords], dim=1)
    keys = voxel_keys(padded, size)
    sorted_keys, order = torch.sort(keys)

    offsets = torch.tensor(_EDGE_NEIGHBOURS, dtype=torch.int32, device=device)
    neighbours = coords.reshape(n, 1, 1, 3) + offsets.unsqueeze(0)  # [N, 3, 4, 3]
    around = neighbours[intersected]  # [M, 4, 3]
    m = around.shape[0]
    query = torch.cat(
        [torch.zeros((m * 4, 1), dtype=torch.int32, device=device), around.reshape(-1, 3)],
        dim=1,
    )
    rows = lookup_rows(sorted_keys, order, voxel_keys(query, size)).reshape(m, 4)
    quads = rows[(rows >= 0).all(dim=1)]

    aabb_min = torch.full((3,), -0.5, dtype=torch.float32, device=device)
    voxel_size = (
        torch.ones(3, dtype=torch.float32, device=device)
        / torch.full((3,), resolution, dtype=torch.int32, device=device)
    )
    vertices = (coords.float() + dual_vertices) * voxel_size + aabb_min.reshape(1, 3)

    weights = split_weight[quads]
    w02 = weights[:, 0] * weights[:, 2]
    w13 = weights[:, 1] * weights[:, 3]
    split_02 = torch.tensor(_SPLIT_02, dtype=torch.long, device=device)
    split_13 = torch.tensor(_SPLIT_13, dtype=torch.long, device=device)
    faces = torch.where(w02 > w13, quads[:, split_02], quads[:, split_13]).reshape(-1, 3)
    return vertices, faces
