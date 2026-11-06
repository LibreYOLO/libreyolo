# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
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
#
# Adapted for LibreYOLO from FlexiCubes (https://github.com/nv-tlabs/FlexiCubes,
# commit eac497031e5138aa544065f357d093f9f3e101e9, flexicubes.py), which is
# Apache-2.0 since that commit. Kept: the voxel grid and the triangle-mesh
# extraction at inference (training=False, no grad_func, no tetrahedral
# output). Removed: the training-time quad split, the QEF path, the
# tetrahedralization and the L_dev regularizer. Arithmetic unchanged.
"""FlexiCubes isosurface extraction (Shen et al., SIGGRAPH 2023), inference path."""

from __future__ import annotations

import torch

from ._flexicubes_tables import check_table, dmc_table, num_vd_table


class FlexiCubes:
    def __init__(self, device="cpu", weight_scale: float = 0.99):
        self.device = device
        self.dmc_table = torch.tensor(dmc_table, dtype=torch.long, device=device)
        self.num_vd_table = torch.tensor(num_vd_table, dtype=torch.long, device=device)
        self.check_table = torch.tensor(check_table, dtype=torch.long, device=device)
        self.quad_split_1 = torch.tensor([0, 1, 2, 0, 2, 3], dtype=torch.long, device=device)
        self.quad_split_2 = torch.tensor([0, 1, 3, 3, 1, 2], dtype=torch.long, device=device)
        self.cube_corners = torch.tensor(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]],
            dtype=torch.float,
            device=device,
        )
        self.cube_corners_idx = torch.pow(2, torch.arange(8))
        self.cube_edges = torch.tensor(
            [0, 1, 1, 5, 4, 5, 0, 4, 2, 3, 3, 7, 6, 7, 2, 6, 2, 0, 3, 1, 7, 5, 6, 4],
            dtype=torch.long,
            device=device,
        )
        self.weight_scale = weight_scale

    def construct_voxel_grid(self, res: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Grid vertices centred at the origin with side 1, and each cube's 8 corner indices."""
        base_cube_f = torch.arange(8).to(self.device)
        res3 = (res, res, res)
        voxel_grid_template = torch.ones(res3, device=self.device)
        res_t = torch.tensor([res3], dtype=torch.float, device=self.device)
        coords = torch.nonzero(voxel_grid_template).float() / res_t
        verts = (self.cube_corners.unsqueeze(0) / res_t + coords.unsqueeze(1)).reshape(-1, 3)
        cubes = (
            base_cube_f.unsqueeze(0)
            + torch.arange(coords.shape[0], device=self.device).unsqueeze(1) * 8
        ).reshape(-1)
        verts_rounded = torch.round(verts * 10**5) / (10**5)
        verts_unique, inverse_indices = torch.unique(verts_rounded, dim=0, return_inverse=True)
        cubes = inverse_indices[cubes.reshape(-1)].reshape(-1, 8)
        return verts_unique - 0.5, cubes

    def __call__(self, x_nx3, s_n, cube_fx8, res: int, beta_fx12, alpha_fx8, gamma_f):
        """Triangle mesh (vertices [V, 3], faces [F, 3]) of the zero level set of ``s_n``."""
        surf_cubes, occ_fx8 = self._identify_surf_cubes(s_n, cube_fx8)
        if surf_cubes.sum() == 0:
            return (
                torch.zeros((0, 3), device=self.device),
                torch.zeros((0, 3), dtype=torch.long, device=self.device),
            )
        beta_fx12 = (torch.tanh(beta_fx12) * self.weight_scale + 1)[surf_cubes]
        alpha_fx8 = (torch.tanh(alpha_fx8) * self.weight_scale + 1)[surf_cubes]
        gamma_f = (torch.sigmoid(gamma_f) * self.weight_scale + (1 - self.weight_scale) / 2)[
            surf_cubes
        ]
        case_ids = self._get_case_id(occ_fx8, surf_cubes, res)
        surf_edges, idx_map, edge_counts, surf_edges_mask = self._identify_surf_edges(
            s_n, cube_fx8, surf_cubes
        )
        vd, vd_gamma, vd_idx_map = self._compute_vd(
            x_nx3, surf_edges, s_n, case_ids, beta_fx12, alpha_fx8, gamma_f, idx_map
        )
        return self._triangulate(
            s_n, surf_edges, vd, vd_gamma, edge_counts, idx_map, vd_idx_map, surf_edges_mask
        )

    @torch.no_grad()
    def _get_case_id(self, occ_fx8, surf_cubes, res: int):
        case_ids = (occ_fx8[surf_cubes] * self.cube_corners_idx.to(self.device).unsqueeze(0)).sum(-1)
        problem_config = self.check_table[case_ids]
        to_check = problem_config[..., 0] == 1
        problem_config = problem_config[to_check]
        res3 = [res, res, res]
        problem_config_full = torch.zeros(res3 + [5], device=self.device, dtype=torch.long)
        vol_idx = torch.nonzero(problem_config_full[..., 0] == 0)
        vol_idx_problem = vol_idx[surf_cubes][to_check]
        problem_config_full[
            vol_idx_problem[..., 0], vol_idx_problem[..., 1], vol_idx_problem[..., 2]
        ] = problem_config
        vol_idx_problem_adj = vol_idx_problem + problem_config[..., 1:4]
        within_range = (
            (vol_idx_problem_adj[..., 0] >= 0)
            & (vol_idx_problem_adj[..., 0] < res3[0])
            & (vol_idx_problem_adj[..., 1] >= 0)
            & (vol_idx_problem_adj[..., 1] < res3[1])
            & (vol_idx_problem_adj[..., 2] >= 0)
            & (vol_idx_problem_adj[..., 2] < res3[2])
        )
        vol_idx_problem_adj = vol_idx_problem_adj[within_range]
        problem_config = problem_config[within_range]
        problem_config_adj = problem_config_full[
            vol_idx_problem_adj[..., 0], vol_idx_problem_adj[..., 1], vol_idx_problem_adj[..., 2]
        ]
        to_invert = problem_config_adj[..., 0] == 1
        idx = torch.arange(case_ids.shape[0], device=self.device)[to_check][within_range][to_invert]
        case_ids.index_put_((idx,), problem_config[to_invert][..., -1])
        return case_ids

    @torch.no_grad()
    def _identify_surf_edges(self, s_n, cube_fx8, surf_cubes):
        occ_n = s_n < 0
        all_edges = cube_fx8[surf_cubes][:, self.cube_edges].reshape(-1, 2)
        unique_edges, _idx_map, counts = torch.unique(
            all_edges, dim=0, return_inverse=True, return_counts=True
        )
        unique_edges = unique_edges.long()
        mask_edges = occ_n[unique_edges.reshape(-1)].reshape(-1, 2).sum(-1) == 1
        surf_edges_mask = mask_edges[_idx_map]
        counts = counts[_idx_map]
        mapping = torch.ones((unique_edges.shape[0]), dtype=torch.long, device=cube_fx8.device) * -1
        mapping[mask_edges] = torch.arange(mask_edges.sum(), device=cube_fx8.device)
        idx_map = mapping[_idx_map]
        surf_edges = unique_edges[mask_edges]
        return surf_edges, idx_map, counts, surf_edges_mask

    @torch.no_grad()
    def _identify_surf_cubes(self, s_n, cube_fx8):
        occ_n = s_n < 0
        occ_fx8 = occ_n[cube_fx8.reshape(-1)].reshape(-1, 8)
        _occ_sum = torch.sum(occ_fx8, -1)
        surf_cubes = (_occ_sum > 0) & (_occ_sum < 8)
        return surf_cubes, occ_fx8

    def _linear_interp(self, edges_weight, edges_x):
        edge_dim = edges_weight.dim() - 2
        edges_weight = torch.cat(
            [
                torch.index_select(edges_weight, edge_dim, torch.tensor(1, device=self.device)),
                -torch.index_select(edges_weight, edge_dim, torch.tensor(0, device=self.device)),
            ],
            edge_dim,
        )
        denominator = edges_weight.sum(edge_dim)
        return (edges_x * edges_weight).sum(edge_dim) / denominator

    def _compute_vd(self, x_nx3, surf_edges, s_n, case_ids, beta_fx12, alpha_fx8, gamma_f, idx_map):
        alpha_nx12x2 = torch.index_select(alpha_fx8, 1, self.cube_edges).reshape(-1, 12, 2)
        surf_edges_x = torch.index_select(x_nx3, 0, surf_edges.reshape(-1)).reshape(-1, 2, 3)
        surf_edges_s = torch.index_select(s_n, 0, surf_edges.reshape(-1)).reshape(-1, 2, 1)
        idx_map = idx_map.reshape(-1, 12)
        num_vd = torch.index_select(self.num_vd_table, 0, case_ids)
        edge_group, edge_group_to_vd, edge_group_to_cube, vd_gamma = [], [], [], []
        total_num_vd = 0
        vd_idx_map = torch.zeros((case_ids.shape[0], 12), dtype=torch.long, device=self.device)
        for num in torch.unique(num_vd):
            cur_cubes = num_vd == num
            curr_num_vd = cur_cubes.sum() * num
            curr_edge_group = self.dmc_table[case_ids[cur_cubes], :num].reshape(-1, num * 7)
            curr_edge_group_to_vd = (
                torch.arange(curr_num_vd, device=self.device).unsqueeze(-1).repeat(1, 7) + total_num_vd
            )
            total_num_vd += curr_num_vd
            curr_edge_group_to_cube = (
                torch.arange(idx_map.shape[0], device=self.device)[cur_cubes]
                .unsqueeze(-1)
                .repeat(1, num * 7)
                .reshape_as(curr_edge_group)
            )
            curr_mask = curr_edge_group != -1
            edge_group.append(torch.masked_select(curr_edge_group, curr_mask))
            edge_group_to_vd.append(
                torch.masked_select(curr_edge_group_to_vd.reshape_as(curr_edge_group), curr_mask)
            )
            edge_group_to_cube.append(torch.masked_select(curr_edge_group_to_cube, curr_mask))
            vd_gamma.append(
                torch.masked_select(gamma_f, cur_cubes).unsqueeze(-1).repeat(1, num).reshape(-1)
            )
        edge_group = torch.cat(edge_group)
        edge_group_to_vd = torch.cat(edge_group_to_vd)
        edge_group_to_cube = torch.cat(edge_group_to_cube)
        vd_gamma = torch.cat(vd_gamma)

        vd = torch.zeros((total_num_vd, 3), device=self.device)
        beta_sum = torch.zeros((total_num_vd, 1), device=self.device)
        idx_group = torch.gather(idx_map.reshape(-1), 0, edge_group_to_cube * 12 + edge_group)
        x_group = torch.index_select(surf_edges_x, 0, idx_group.reshape(-1)).reshape(-1, 2, 3)
        s_group = torch.index_select(surf_edges_s, 0, idx_group.reshape(-1)).reshape(-1, 2, 1)
        alpha_group = torch.index_select(
            alpha_nx12x2.reshape(-1, 2), 0, edge_group_to_cube * 12 + edge_group
        ).reshape(-1, 2, 1)
        ue_group = self._linear_interp(s_group * alpha_group, x_group)
        beta_group = torch.gather(
            beta_fx12.reshape(-1), 0, edge_group_to_cube * 12 + edge_group
        ).reshape(-1, 1)
        beta_sum = beta_sum.index_add_(0, index=edge_group_to_vd, source=beta_group)
        vd = vd.index_add_(0, index=edge_group_to_vd, source=ue_group * beta_group) / beta_sum
        v_idx = torch.arange(vd.shape[0], device=self.device)
        vd_idx_map = vd_idx_map.reshape(-1).scatter(
            dim=0, index=edge_group_to_cube * 12 + edge_group, src=v_idx[edge_group_to_vd]
        )
        return vd, vd_gamma, vd_idx_map

    def _triangulate(self, s_n, surf_edges, vd, vd_gamma, edge_counts, idx_map, vd_idx_map, surf_edges_mask):
        with torch.no_grad():
            group_mask = (edge_counts == 4) & surf_edges_mask
            group = idx_map.reshape(-1)[group_mask]
            vd_idx = vd_idx_map[group_mask]
            edge_indices, indices = torch.sort(group, stable=True)
            quad_vd_idx = vd_idx[indices].reshape(-1, 4)
            s_edges = s_n[surf_edges[edge_indices.reshape(-1, 4)[:, 0]].reshape(-1)].reshape(-1, 2)
            flip_mask = s_edges[:, 0] > 0
            quad_vd_idx[flip_mask] = quad_vd_idx[flip_mask][:, [0, 1, 3, 2]]
            quad_vd_idx[~flip_mask] = quad_vd_idx[~flip_mask][:, [2, 3, 1, 0]]
        quad_gamma = torch.index_select(vd_gamma, 0, quad_vd_idx.reshape(-1)).reshape(-1, 4)
        gamma_02 = quad_gamma[:, 0:1] * quad_gamma[:, 2:3]
        gamma_13 = quad_gamma[:, 1:2] * quad_gamma[:, 3:4]
        mask = (gamma_02 > gamma_13).squeeze(1)
        faces = torch.zeros((quad_gamma.shape[0], 6), dtype=torch.long, device=quad_vd_idx.device)
        faces[mask] = quad_vd_idx[mask][:, self.quad_split_1]
        faces[~mask] = quad_vd_idx[~mask][:, self.quad_split_2]
        return vd, faces.reshape(-1, 3)
