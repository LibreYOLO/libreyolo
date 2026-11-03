# Vendored from https://github.com/neufieldrobotics/NeuFlow_v2 (NeuFlow/utils.py) at
# commit 204b5e3744461d90303b9ff82caa7a1bb56a2ca2, Apache-2.0 (see LICENSE).
# LibreYOLO changes: package-relative imports; see _vendor/__init__.py.

import torch
import torch.nn.functional as F

# def normalize(x):
#     x_min = x.min()
#     return (x - x_min) / (x.max() - x_min)

def coords_grid(b, h, w, device, amp):
    ys, xs = torch.meshgrid(torch.arange(h, dtype=torch.half if amp else torch.float, device=device), torch.arange(w, dtype=torch.half if amp else torch.float, device=device), indexing='ij')  # [H, W]

    grid = torch.stack([xs, ys], dim=0)  # [2, H, W] or [3, H, W]

    grid = grid[None].repeat(b, 1, 1, 1)  # [B, 2, H, W] or [B, 3, H, W]

    return grid

def bilinear_sample(img, sample_coords):

    b, _, h, w = sample_coords.shape

    # Normalize to [-1, 1]
    x_grid = 2 * sample_coords[:, 0] / (w - 1) - 1
    y_grid = 2 * sample_coords[:, 1] / (h - 1) - 1

    grid = torch.stack([x_grid, y_grid], dim=-1)  # [B, H, W, 2]

    img = F.grid_sample(img, grid, mode='bilinear', padding_mode='zeros', align_corners=True)

    return img
