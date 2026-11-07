"""Decode InstantHMR head outputs into MHR parameters.

Written clean-room from a functional spec for the LibreYOLO InstantHMR port.
The math follows Zhou et al. 2019 ("On the Continuity of Rotation
Representations in Neural Networks") for the 6D rotation and the standard
extrinsic XYZ Euler extraction. All index tables and linear bases are inputs
that come from the checkpoint, not from code.

Everything is pure PyTorch with no data-dependent control flow, so it traces
to ONNX opset 17 with ``torch.onnx.export(..., dynamo=False)``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

# Layout of the 447-dim head output, in order.
ROOT_DIM = 6
BODY_N3, BODY_N1, BODY_RAW = 23, 58, 6
BODY_DIM = 6 * BODY_N3 + 2 * BODY_N1 + BODY_RAW  # 260
SHAPE_DIM = 45
COEFF_DIM = 28
HAND_N3, HAND_N1 = 5, 12
HAND_DIM = 6 * HAND_N3 + 2 * HAND_N1  # 54
PRED_DIM = ROOT_DIM + BODY_DIM + SHAPE_DIM + COEFF_DIM + 2 * HAND_DIM  # 447

BODY_KEEP = 130  # body angles kept after the scatter
FLEX_START, FLEX_END = 124, 130  # body slots squashed into [flex_lo, flex_hi]


def _columns(c0: torch.Tensor, c1: torch.Tensor, c2: torch.Tensor) -> torch.Tensor:
    return torch.stack([c0, c1, c2], dim=-1)


def rot6d_to_matrix(x: torch.Tensor) -> torch.Tensor:
    """(..., 6) -> (..., 3, 3) via Gram-Schmidt; columns are c0, c1, c2."""
    a, b = x[..., :3], x[..., 3:]
    c0 = F.normalize(a, dim=-1)
    c1 = F.normalize(b - (c0 * b).sum(-1, keepdim=True) * c0, dim=-1)
    c2 = torch.cross(c0, c1, dim=-1)
    return _columns(c0, c1, c2)


def cross6d_to_matrix(x: torch.Tensor) -> torch.Tensor:
    """(..., 6) -> (..., 3, 3) via cross products; columns are c0, c1, c2."""
    a, b = x[..., :3], x[..., 3:]
    c0 = F.normalize(a, dim=-1)
    c2 = F.normalize(torch.cross(c0, b, dim=-1), dim=-1)
    c1 = torch.cross(c2, c0, dim=-1)
    return _columns(c0, c1, c2)


def matrix_to_euler_xyz(R: torch.Tensor) -> torch.Tensor:
    """(..., 3, 3) -> (..., 3) angles (a, b, c) with R = Rz(c) @ Ry(b) @ Rx(a)."""
    r00, r10, r20 = R[..., 0, 0], R[..., 1, 0], R[..., 2, 0]
    s = torch.sqrt(r00 * r00 + r10 * r10)
    singular = (s < 1e-6).to(R.dtype)  # blend weight, not a branch
    a_reg = torch.atan2(R[..., 2, 1], R[..., 2, 2])
    a_sing = torch.atan2(-R[..., 1, 2], R[..., 1, 1])
    a = a_reg * (1 - singular) + a_sing * singular
    b = torch.atan2(-r20, s)
    c = torch.atan2(r10, r00) * (1 - singular)
    return torch.stack([a, b, c], dim=-1)


def euler_xyz_to_matrix(e: torch.Tensor) -> torch.Tensor:
    """(..., 3) -> (..., 3, 3), Rz(c) @ Ry(b) @ Rx(a)."""
    ca, cb, cc = torch.cos(e).unbind(-1)
    sa, sb, sc = torch.sin(e).unbind(-1)
    rows = [
        [cc * cb, cc * sb * sa - sc * ca, cc * sb * ca + sc * sa],
        [sc * cb, sc * sb * sa + cc * ca, sc * sb * ca - cc * sa],
        [-sb, cb * sa, cb * ca],
    ]
    return torch.stack([torch.stack(r, dim=-1) for r in rows], dim=-2)


def angles_from_continuous(x: torch.Tensor, n3: int, n1: int) -> torch.Tensor:
    """(..., 6*n3 + 2*n1 + n_raw) -> (..., 3*n3 + n1 + n_raw).

    3-DoF joints come as cross-6D groups, 1-DoF joints as (sin, cos) pairs,
    and the remaining dims pass through.
    """
    lead = x.shape[:-1]
    d3, d1 = 6 * n3, 2 * n1
    groups = x[..., :d3].reshape(*lead, n3, 6)
    euler = matrix_to_euler_xyz(cross6d_to_matrix(groups)).reshape(*lead, 3 * n3)
    pairs = x[..., d3 : d3 + d1].reshape(*lead, n1, 2)
    hinge = torch.atan2(pairs[..., 0], pairs[..., 1])
    return torch.cat([euler, hinge, x[..., d3 + d1 :]], dim=-1)


def to_range(x: torch.Tensor, lo, hi) -> torch.Tensor:
    """Squash x into (lo, hi) with tanh."""
    return lo + (hi - lo) * 0.5 * (torch.tanh(x) + 1)


def decode_mhr(
    pred: torch.Tensor,
    *,
    body_scatter: torch.Tensor,
    hand_cont_gather: torch.Tensor,
    hand_param_scatter: torch.Tensor,
    full_gather: torch.Tensor,
    hand_mean: torch.Tensor,
    hand_comps: torch.Tensor,
    scale_mean: torch.Tensor,
    scale_comps: torch.Tensor,
    coeff_lo,
    coeff_hi,
    flex_lo=None,
    flex_hi=None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(B, 447) head output -> (mhr_params (B, 204), shape (B, 45), root_R (B, 3, 3))."""
    root6d, body_c, shape, coeff, hand_l, hand_r = pred.split(
        [ROOT_DIM, BODY_DIM, SHAPE_DIM, COEFF_DIM, HAND_DIM, HAND_DIM], dim=-1
    )

    root_R = rot6d_to_matrix(root6d)
    root_euler = matrix_to_euler_xyz(root_R)

    body = angles_from_continuous(body_c, BODY_N3, BODY_N1)
    body = body.index_select(-1, body_scatter)[..., :BODY_KEEP]
    if flex_lo is not None:
        flex = to_range(body[..., FLEX_START:FLEX_END], flex_lo, flex_hi)
        body = torch.cat([body[..., :FLEX_START], flex, body[..., FLEX_END:]], dim=-1)

    def hand(h: torch.Tensor) -> torch.Tensor:
        v = (hand_mean + h @ hand_comps).index_select(-1, hand_cont_gather)
        angles = angles_from_continuous(v, HAND_N3, HAND_N1)
        return angles.index_select(-1, hand_param_scatter)

    zeros = torch.zeros_like(root_euler)
    src = torch.cat([zeros, root_euler, body, hand(hand_l), hand(hand_r)], dim=-1)
    full = src.index_select(-1, full_gather)

    scales = scale_mean + to_range(coeff, coeff_lo, coeff_hi) @ scale_comps
    return torch.cat([full, scales], dim=-1), shape, root_R
