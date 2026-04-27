"""Random-weight forward-pass shape tests for backbone + encoder + decoder.

This exercises the full LibreYOLO-ported pipeline without loading any
checkpoint. Purpose: catch shape/wiring mistakes before we attempt parity
against a real D-FINE checkpoint.
"""

from __future__ import annotations

import pytest
import torch

from libreyolo.models.dfine.backbone import HGNetv2
from libreyolo.models.dfine.decoder import DFINETransformer
from libreyolo.models.dfine.encoder import HybridEncoder
from libreyolo.models.dfine.nn import SIZE_CONFIGS

pytestmark = pytest.mark.unit


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
        hidden_dim=cfg["enc_hidden_dim"],
        dim_feedforward=cfg["enc_dim_feedforward"],
        expansion=cfg["enc_expansion"],
        depth_mult=cfg["enc_depth_mult"],
        use_encoder_idx=cfg["enc_use_encoder_idx"],
        eval_spatial_size=(640, 640),
    )
    decoder = DFINETransformer(
        num_classes=80,
        hidden_dim=cfg["dec_hidden_dim"],
        num_queries=300,
        feat_channels=cfg["dec_feat_channels"],
        feat_strides=cfg["dec_feat_strides"],
        num_levels=cfg["dec_num_levels"],
        num_points=cfg["dec_num_points"],
        num_layers=cfg["dec_num_layers"],
        dim_feedforward=cfg["dec_dim_feedforward"],
        eval_spatial_size=(640, 640),
        eval_idx=cfg["dec_eval_idx"],
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
