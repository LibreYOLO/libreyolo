"""InstantHMR network: person crop + box conditioning -> MHR parameters.

Ported from InstantHMR ``training/train.py`` (``InstantHMRStudent`` and the
module layout of ``ContMHRHead``), https://github.com/Naturalpad-studio/InstantHMR,
Apache-2.0. Parameter and buffer names match upstream, so an upstream
``train.py`` checkpoint loads after the decode index tables are added.

Architecture: an HGNetV2 backbone (reused from the D-FINE port, same weights
layout as timm's ``hgnetv2_b*``), a linear projection of the last feature map
to a 9x9 memory with fixed sin-cos positions, and a 4-layer pre-norm
transformer decoder over 71 queries (1 global + 70 keypoints) that are shifted
by an MLP embedding of the box conditioning. The global token feeds the
continuous MHR head and the camera head; the keypoint tokens feed a 1D
soft-argmax head per axis.

Differences from upstream:
- The 447 -> 204 decode is LibreYOLO's clean-room ``decode.decode_mhr``; its
  index tables are buffers carried by the checkpoint.
- Training-only branches that the released models do not use (the previous-
  pose token, the contact head, the non-continuous global head and its
  separate scale bound) are not ported.
- The MHR body (``body.MHRBody``) is a submodule, so one checkpoint holds the
  network and the body.
"""

from __future__ import annotations

import torch
from torch import nn

from ..dfine.backbone import HGNetv2
from .body import MHRBody
from .decode import (
    BODY_N1,
    BODY_N3,
    BODY_RAW,
    COEFF_DIM,
    HAND_DIM,
    PRED_DIM,
    decode_mhr,
)

# size -> (HGNetV2 variant, learnable affine blocks)
BACKBONES = {
    "n": ("B0", True),
    "t": ("B1", True),
    "s": ("B2", True),
    "m": ("B4", False),
}

IMAGE_SIZE = 288
D_MODEL = 512
N_HEADS = 8
N_LAYERS = 4
NUM_JOINTS = 70
CLIFF_DIM = 3
CAM_DIM = 3
KP2D_BINS = 96
KP2D_RANGE = 1.5
N_FLEX = 6
N_BODY_PARAMS = 3 * BODY_N3 + BODY_N1 + BODY_RAW  # 133
N_HAND_PARAMS = 27
N_FULL = 136
N_SCALES = 68


def sincos_pos_embed(embed_dim: int, grid_size: int) -> torch.Tensor:
    """``(1, grid*grid, embed_dim)`` fixed 2D sin-cos positions, y block then x block."""
    gy, gx = torch.meshgrid(
        torch.arange(grid_size), torch.arange(grid_size), indexing="ij"
    )
    omega = 1.0 / (10000 ** (torch.arange(embed_dim // 4).float() / (embed_dim // 4)))
    out_y = gy.flatten()[:, None] * omega[None]
    out_x = gx.flatten()[:, None] * omega[None]
    pe = torch.cat([out_y.sin(), out_y.cos(), out_x.sin(), out_x.cos()], dim=1)
    return pe.unsqueeze(0)


class ContMHRHead(nn.Module):
    """Global token -> 447 continuous values -> 204 MHR params, 45 identity, root R.

    The bases, bounds and index tables are buffers. Their values come from the
    released weights (see ``weights/convert_instanthmr_weights.py``), so a
    freshly built head is only meaningful after a checkpoint is loaded.
    """

    def __init__(self, d_model: int = D_MODEL):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, PRED_DIM)
        )
        self.init_estimate = nn.Parameter(torch.zeros(1, PRED_DIM))
        self.register_buffer("scale_mean", torch.zeros(N_SCALES))
        self.register_buffer("scale_comps", torch.zeros(COEFF_DIM, N_SCALES))
        self.register_buffer("coeff_lo", torch.zeros(COEFF_DIM))
        self.register_buffer("coeff_hi", torch.zeros(COEFF_DIM))
        self.register_buffer("hand_mean", torch.zeros(HAND_DIM))
        self.register_buffer("hand_comps", torch.zeros(HAND_DIM, HAND_DIM))
        self.register_buffer("flex_lo", torch.zeros(N_FLEX))
        self.register_buffer("flex_hi", torch.zeros(N_FLEX))
        self.register_buffer("full_gather", torch.arange(N_FULL))
        self.register_buffer("body_scatter", torch.arange(N_BODY_PARAMS))
        self.register_buffer("hand_cont_gather", torch.arange(HAND_DIM))
        self.register_buffer("hand_param_scatter", torch.arange(N_HAND_PARAMS))

    def forward(self, feat: torch.Tensor):
        return self.decode(self.mlp(self.norm(feat)) + self.init_estimate)

    def decode(self, pred: torch.Tensor):
        return decode_mhr(
            pred,
            body_scatter=self.body_scatter,
            hand_cont_gather=self.hand_cont_gather,
            hand_param_scatter=self.hand_param_scatter,
            full_gather=self.full_gather,
            hand_mean=self.hand_mean,
            hand_comps=self.hand_comps,
            scale_mean=self.scale_mean,
            scale_comps=self.scale_comps,
            coeff_lo=self.coeff_lo,
            coeff_hi=self.coeff_hi,
            flex_lo=self.flex_lo,
            flex_hi=self.flex_hi,
        )


class InstantHMRNet(nn.Module):
    """Full InstantHMR model: network plus MHR body."""

    def __init__(self, size: str = "m", dropout: float = 0.1):
        super().__init__()
        if size not in BACKBONES:
            raise ValueError(
                f"Unknown InstantHMR size {size!r}; expected one of {sorted(BACKBONES)}"
            )
        name, use_lab = BACKBONES[size]
        self.size = size
        self.image_size = IMAGE_SIZE
        self.backbone = HGNetv2(
            name, use_lab=use_lab, return_idx=[3], freeze_at=-1, freeze_norm=False
        )
        embed_dim = self.backbone._out_channels[-1]
        self.feat_proj = nn.Linear(embed_dim, D_MODEL)
        self.register_buffer(
            "mem_pos_embed", sincos_pos_embed(D_MODEL, IMAGE_SIZE // 32)
        )

        self.num_2d = NUM_JOINTS
        self.query_embed = nn.Parameter(torch.randn(1, 1 + NUM_JOINTS, D_MODEL) * 0.02)
        self.cond_proj = nn.Sequential(
            nn.Linear(CLIFF_DIM, 128), nn.GELU(), nn.Linear(128, D_MODEL)
        )
        layer = nn.TransformerDecoderLayer(
            d_model=D_MODEL,
            nhead=N_HEADS,
            dim_feedforward=D_MODEL * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerDecoder(layer, num_layers=N_LAYERS)

        self.cont_head = ContMHRHead(D_MODEL)
        self.head_cam = nn.Linear(D_MODEL, CAM_DIM)
        self.head_2d_feat = nn.Sequential(nn.Linear(D_MODEL, 256), nn.GELU())
        self.head_2d_logits = nn.Linear(256, 2 * KP2D_BINS)
        self.register_buffer(
            "kp2d_bin_centers", torch.linspace(-KP2D_RANGE, KP2D_RANGE, KP2D_BINS)
        )

        self.body = MHRBody()

    def forward(self, images: torch.Tensor, cliff_cond: torch.Tensor) -> dict:
        """``images`` (B, 3, 288, 288) ImageNet-normalised RGB crops; ``cliff_cond`` (B, 3).

        Returns ``mhr_params`` (B, 204), ``shape_params`` (B, 45), ``cam_trans``
        (B, 3) metres, ``joints_2d`` (B, 70, 2) in [-1, 1] over the crop, plus
        ``joints_2d_logits`` and ``root_rotmat`` for training losses.
        """
        b = images.shape[0]
        feats = self.backbone(images)[-1].flatten(2).transpose(1, 2)
        memory = self.feat_proj(feats) + self.mem_pos_embed
        queries = self.query_embed.expand(b, -1, -1) + self.cond_proj(
            cliff_cond
        ).unsqueeze(1)
        tokens = self.transformer(tgt=queries, memory=memory)

        feat_global = tokens[:, 0]
        mhr_params, shape_params, root_rotmat = self.cont_head(feat_global)
        cam_trans = self.head_cam(feat_global)

        logits = self.head_2d_logits(self.head_2d_feat(tokens[:, 1 : 1 + self.num_2d]))
        logits = logits.unflatten(-1, (2, KP2D_BINS))
        prob = torch.softmax(logits.float(), dim=-1)
        joints_2d = (prob * self.kp2d_bin_centers.float()).sum(dim=-1)
        return {
            "mhr_params": mhr_params,
            "shape_params": shape_params,
            "cam_trans": cam_trans,
            "joints_2d": joints_2d,
            "joints_2d_logits": logits,
            "root_rotmat": root_rotmat,
        }


class InstantHMRExport(nn.Module):
    """Deploy graph: crop + conditioning -> parameters, 2D/3D keypoints and mesh."""

    OUTPUTS = (
        "mhr_params",
        "shape_params",
        "cam_trans",
        "joints_2d",
        "joints_3d",
        "vertices",
    )

    def __init__(self, net: InstantHMRNet):
        super().__init__()
        self.net = net

    def forward(self, image: torch.Tensor, cliff_cond: torch.Tensor):
        out = self.net(image, cliff_cond)
        joints_3d, vertices = self.net.body(out["mhr_params"], out["shape_params"])
        return (
            out["mhr_params"],
            out["shape_params"],
            out["cam_trans"],
            out["joints_2d"],
            joints_3d,
            vertices,
        )
