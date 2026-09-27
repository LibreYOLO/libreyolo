"""Cross-check the LibreTinyFormer port against the official TinyFormer engine.

Loads each released COCO-PBM checkpoint into both the upstream engine
(mmpmmpmmpjosh/TinyFormer, must be cloned locally) and the LibreYOLO port,
runs identical inputs through both in eval mode, and asserts
``max_abs_diff == 0`` on ``pred_logits`` / ``pred_boxes``.

Usage:
    set TINYFORMER_OFFICIAL_REPO=path/to/TinyFormer/clone
    set TINYFORMER_OFFICIAL_CKPT_DIR=path/to/dir/with/TinyFormer-*-pbm.pth
    python weights/parity_tinyformer.py [--sizes s m l x xl]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REPO = os.environ.get("TINYFORMER_OFFICIAL_REPO")
CKPT_DIR = os.environ.get("TINYFORMER_OFFICIAL_CKPT_DIR")
if not REPO or not CKPT_DIR:
    raise SystemExit(
        "Set TINYFORMER_OFFICIAL_REPO (upstream clone) and "
        "TINYFORMER_OFFICIAL_CKPT_DIR (dir with TinyFormer-*-pbm.pth)"
    )

CKPT_FILES = {
    "s": "TinyFormer-S-pbm.pth",
    "m": "TinyFormer-M-pbm.pth",
    "l": "TinyFormer-L-pbm.pth",
    "x": "TinyFormer-X-pbm.pth",
    "xl": "TinyFormer-XL-pbm.pth",
}

# Constructor kwargs flattened from the upstream PBM YAMLs — kept in sync with
# libreyolo/models/tinyformer/nn.py SIZE_CONFIGS.
UPSTREAM_BACKBONE = {
    "s": dict(name="vit_tiny", embed_dim=192, interaction_indexes=[3, 7, 11], num_heads=3),
    "m": dict(name="vit_tinyplus", embed_dim=256, interaction_indexes=[3, 7, 11], num_heads=4),
    "l": dict(name="dinov3_vits16", interaction_indexes=[5, 8, 11], conv_inplane=32, hidden_dim=224),
    "x": dict(name="dinov3_vits16plus", interaction_indexes=[5, 8, 11], conv_inplane=64, hidden_dim=256),
    "xl": dict(name="dinov3_vitb16", interaction_indexes=[5, 8, 11], conv_inplane=128, hidden_dim=384),
}
UPSTREAM_ENCODER = {
    "s": dict(in_channels=[192] * 4, hidden_dim=192, dim_feedforward=512, expansion=0.34, depth_mult=0.67),
    "m": dict(in_channels=[256] * 4, hidden_dim=256, dim_feedforward=512, expansion=0.67, depth_mult=1.0),
    "l": dict(in_channels=[224] * 4, hidden_dim=224, dim_feedforward=896),
    "x": dict(in_channels=[256] * 4, hidden_dim=256, dim_feedforward=1024, expansion=1.25, depth_mult=1.37),
    "xl": dict(in_channels=[384] * 4, hidden_dim=384, dim_feedforward=1024, expansion=1.25, depth_mult=1.37),
}
UPSTREAM_DECODER = {
    "s": dict(feat_channels=[192] * 3, hidden_dim=192, dim_feedforward=512, num_layers=4),
    "m": dict(feat_channels=[256] * 3, hidden_dim=256, dim_feedforward=512, num_layers=4),
    "l": dict(feat_channels=[224] * 3, hidden_dim=224, dim_feedforward=1792, num_layers=4),
    "x": dict(feat_channels=[256] * 3, hidden_dim=256, dim_feedforward=2048, num_layers=6),
    "xl": dict(feat_channels=[384] * 3, hidden_dim=256, dim_feedforward=2048, num_layers=6),
}
DECODER_BASE = dict(
    num_classes=80,
    feat_strides=[8, 16, 32],
    num_levels=3,
    num_points=[3, 6, 3],
    eval_idx=-1,
    num_queries=300,
    reg_max=32,
    reg_scale=4,
    activation="silu",
    mlp_act="silu",
    eval_spatial_size=[640, 640],
)


def build_upstream(size: str) -> torch.nn.Module:
    sys.path.insert(0, REPO)
    from engine.backbone.dinov3_adapter import DINOv3SSAs_4Scale
    from engine.deim.hybrid_encoder import HybridEncoder_4Scale
    from engine.deim.deim_decoder import DEIMTransformer

    class Upstream(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = DINOv3SSAs_4Scale(**UPSTREAM_BACKBONE[size])
            self.encoder = HybridEncoder_4Scale(
                feat_strides=[4, 8, 16, 32],
                out_indices=[1, 2, 3],
                use_encoder_idx=[3],
                **UPSTREAM_ENCODER[size],
            )
            self.decoder = DEIMTransformer(**{**DECODER_BASE, **UPSTREAM_DECODER[size]})

        def forward(self, x):
            return self.decoder(self.encoder(self.backbone(x)))

    return Upstream()


def main(sizes: list[str]) -> None:
    from libreyolo.models.tinyformer.nn import LibreTinyFormerModel

    for size in sizes:
        ckpt = torch.load(
            Path(CKPT_DIR) / CKPT_FILES[size], map_location="cpu", weights_only=False
        )
        sd = ckpt["ema"]["module"] if "ema" in ckpt else ckpt["model"]

        upstream = build_upstream(size)
        missing, unexpected = upstream.load_state_dict(sd, strict=False)
        assert not unexpected, f"upstream unexpected: {sorted(unexpected)[:5]}"
        upstream.eval()

        ours = LibreTinyFormerModel(config=size, nb_classes=80)
        missing, unexpected = ours.load_state_dict(sd, strict=False)
        assert not unexpected, f"ours unexpected: {sorted(unexpected)[:5]}"
        ours.eval()

        torch.manual_seed(0)
        x = torch.randn(1, 3, 640, 640)
        with torch.no_grad():
            up = upstream(x)
            our = ours(x)

        for key in ("pred_logits", "pred_boxes"):
            diff = (up[key] - our[key]).abs().max().item()
            status = "OK" if diff == 0.0 else "FAIL"
            print(f"{size} {key}: max_abs_diff={diff} {status}")
            assert diff == 0.0, f"size={size} key={key} max_abs_diff={diff}"
        print(f"size={size}: parity OK")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--sizes", nargs="+", default=list(CKPT_FILES))
    main(p.parse_args().sizes)
