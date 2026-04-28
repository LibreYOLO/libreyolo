"""Random-weight forward-pass shape tests for ECDet (backbone + encoder + decoder)."""

from __future__ import annotations

import pytest
import torch

from libreyolo.models.ecdet.backbone import ViTAdapter
from libreyolo.models.ecdet.decoder import ECTransformer
from libreyolo.models.ecdet.encoder import HybridEncoder
from libreyolo.models.ecdet.nn import SIZE_CONFIGS

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("size", ["s", "m", "l", "x"])
def test_full_pipeline_random_weights(size):
    cfg = SIZE_CONFIGS[size]

    backbone = ViTAdapter(
        embed_dim=cfg["embed_dim"],
        num_heads=cfg["num_heads"],
        proj_dim=cfg["proj_dim"],
        ffn_ratio=cfg.get("ffn_ratio", 4.0),
        interaction_indexes=(10, 11),
    )
    encoder = HybridEncoder(
        in_channels=cfg["enc_in_channels"],
        hidden_dim=cfg["enc_hidden_dim"],
        dim_feedforward=cfg["enc_dim_feedforward"],
        expansion=cfg["enc_expansion"],
        depth_mult=cfg["enc_depth_mult"],
        eval_spatial_size=[640, 640],
    )
    decoder = ECTransformer(
        num_classes=80,
        hidden_dim=cfg["dec_hidden_dim"],
        feat_channels=cfg["dec_feat_channels"],
        dim_feedforward=cfg["dec_dim_feedforward"],
        num_layers=4,
        num_points=(3, 6, 3),
        eval_idx=-1,
        reg_max=32,
        reg_scale=4.0,
        eval_spatial_size=[640, 640],
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
