"""Random-weight forward-pass shape tests for backbone + encoder + decoder.

This exercises the full LibreYOLO-ported pipeline without loading any
checkpoint. Purpose: catch shape/wiring mistakes before we attempt parity
against a real D-FINE checkpoint.
"""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.unit

from libreyolo.models.dfine.backbone import HGNetv2
from libreyolo.models.dfine.decoder import DFINETransformer
from libreyolo.models.dfine.encoder import HybridEncoder


# Size → (backbone, return_idx, encoder in_channels, encoder feat_strides,
#         hidden_dim, dim_ff, expansion, depth_mult, use_encoder_idx,
#         decoder feat_channels, num_levels, num_layers, num_points,
#         eval_idx, reg_scale)
SIZE_CONFIGS = {
    "n": {
        "backbone": "B0",
        "use_lab": True,
        "return_idx": (2, 3),
        "enc_in_channels": (512, 1024),
        "enc_feat_strides": (16, 32),
        "hidden_dim": 128,
        "dim_ff": 512,
        "expansion": 0.34,
        "depth_mult": 0.5,
        "use_encoder_idx": (1,),
        "dec_feat_channels": (128, 128),
        "dec_feat_strides": (16, 32),
        "num_levels": 2,
        "num_layers": 3,
        "num_points": (6, 6),
        "eval_idx": -1,
        "reg_scale": 4.0,
    },
    "s": {
        "backbone": "B0",
        "use_lab": True,
        "return_idx": (1, 2, 3),
        "enc_in_channels": (256, 512, 1024),
        "enc_feat_strides": (8, 16, 32),
        "hidden_dim": 256,
        "dim_ff": 1024,
        "expansion": 1.0,
        "depth_mult": 1.0,
        "use_encoder_idx": (2,),
        "dec_feat_channels": (256, 256, 256),
        "dec_feat_strides": (8, 16, 32),
        "num_levels": 3,
        "num_layers": 3,
        "num_points": 4,
        "eval_idx": -1,
        "reg_scale": 4.0,
    },
    "x": {
        "backbone": "B5",
        "use_lab": False,
        "return_idx": (1, 2, 3),
        "enc_in_channels": (512, 1024, 2048),
        "enc_feat_strides": (8, 16, 32),
        "hidden_dim": 384,
        "dim_ff": 2048,
        "expansion": 1.0,
        "depth_mult": 1.0,
        "use_encoder_idx": (2,),
        "dec_feat_channels": (384, 384, 384),
        "dec_feat_strides": (8, 16, 32),
        "num_levels": 3,
        "num_layers": 6,
        "num_points": 4,
        "eval_idx": -1,
        "reg_scale": 8.0,
    },
}


@pytest.mark.parametrize("size", ["n", "s", "x"])
def test_full_pipeline_random_weights(size):
    """Build backbone -> encoder -> decoder, run one batch, check output shapes."""
    cfg = SIZE_CONFIGS[size]

    backbone = HGNetv2(
        name=cfg["backbone"],
        use_lab=cfg["use_lab"],
        return_idx=cfg["return_idx"],
        freeze_at=-1,
        freeze_norm=False,
    )
    encoder = HybridEncoder(
        in_channels=cfg["enc_in_channels"],
        feat_strides=cfg["enc_feat_strides"],
        hidden_dim=cfg["hidden_dim"],
        dim_feedforward=cfg["dim_ff"],
        expansion=cfg["expansion"],
        depth_mult=cfg["depth_mult"],
        use_encoder_idx=cfg["use_encoder_idx"],
        eval_spatial_size=(640, 640),
    )
    decoder = DFINETransformer(
        num_classes=80,
        hidden_dim=cfg["hidden_dim"],
        num_queries=300,
        feat_channels=cfg["dec_feat_channels"],
        feat_strides=cfg["dec_feat_strides"],
        num_levels=cfg["num_levels"],
        num_points=cfg["num_points"],
        num_layers=cfg["num_layers"],
        dim_feedforward=cfg["dim_ff"],
        eval_spatial_size=(640, 640),
        eval_idx=cfg["eval_idx"],
        reg_scale=cfg["reg_scale"],
    )

    backbone.eval()
    encoder.eval()
    decoder.eval()

    x = torch.randn(1, 3, 640, 640)
    with torch.no_grad():
        feats = backbone(x)
        enc_feats = encoder(feats)
        out = decoder(enc_feats)

    assert set(out.keys()) == {"pred_logits", "pred_boxes"}
    assert out["pred_logits"].shape == (1, 300, 80)
    assert out["pred_boxes"].shape == (1, 300, 4)
    # Boxes should be in [0, 1] after sigmoid.
    assert (out["pred_boxes"] >= 0).all() and (out["pred_boxes"] <= 1).all()
