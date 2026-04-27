"""Layer-by-layer parity test for ECDet vs upstream EdgeCrafter.

Builds the same model with the same weights in both implementations, feeds an
identical input, and asserts intermediate tensors agree at 1e-5. Skipped if
the upstream EdgeCrafter sources or dependencies are not present.

Setup the environment to run this test by adding upstream to PYTHONPATH:

    cd downloads/EdgeCrafter/ecdetseg
    pip install tensorboard pyyaml  # missing transitive deps
    cd ../../..
    UPSTREAM_PATH=downloads/EdgeCrafter/ecdetseg pytest tests/unit/test_ecdet_parity.py -v
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.unit

UPSTREAM_PATH = Path(os.environ.get("UPSTREAM_PATH", "downloads/EdgeCrafter/ecdetseg"))


def _try_import_upstream():
    if not UPSTREAM_PATH.exists():
        pytest.skip(f"Upstream EdgeCrafter not found at {UPSTREAM_PATH}")
    sys.path.insert(0, str(UPSTREAM_PATH))
    try:
        from engine.edgecrafter.ecvit import ViTAdapter as UpstreamBackbone
        from engine.edgecrafter.hybrid_encoder import HybridEncoder as UpstreamEncoder
        from engine.edgecrafter.decoder import ECTransformer as UpstreamDecoder
    except ImportError as e:
        pytest.skip(f"Upstream deps missing: {e}")
    return UpstreamBackbone, UpstreamEncoder, UpstreamDecoder


CKPT_PATH = Path("downloads/ec_weights/ecdet_s.pth")


@pytest.mark.skipif(not CKPT_PATH.exists(), reason=f"{CKPT_PATH} not found")
def test_backbone_parity_ecdet_s():
    UpBB, _, _ = _try_import_upstream()
    from libreyolo.models.ecdet.backbone import ViTAdapter as LibreBB

    upstream = UpBB(
        name="ecvitt",
        embed_dim=192,
        num_heads=3,
        interaction_indexes=[10, 11],
        skip_load_backbone=True,
    )
    libre = LibreBB(embed_dim=192, num_heads=3, interaction_indexes=(10, 11))

    ck = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)["model"]
    bb_sd = {k.removeprefix("backbone."): v for k, v in ck.items() if k.startswith("backbone.")}
    upstream.load_state_dict(bb_sd, strict=True)
    libre.load_state_dict(bb_sd, strict=True)

    upstream.eval()
    libre.eval()

    torch.manual_seed(0)
    x = torch.randn(1, 3, 640, 640)
    with torch.no_grad():
        up_out = upstream(x)
        lb_out = libre(x)

    assert len(up_out) == len(lb_out)
    for i, (u, l) in enumerate(zip(up_out, lb_out)):
        assert u.shape == l.shape, f"level {i}: shape {u.shape} vs {l.shape}"
        max_err = (u - l).abs().max().item()
        assert max_err < 1e-5, f"level {i}: max abs err {max_err:.2e}"


@pytest.mark.skipif(not CKPT_PATH.exists(), reason=f"{CKPT_PATH} not found")
def test_full_pipeline_parity_ecdet_s():
    UpBB, UpEnc, UpDec = _try_import_upstream()
    from libreyolo.models.ecdet.backbone import ViTAdapter as LBB
    from libreyolo.models.ecdet.encoder import HybridEncoder as LEnc
    from libreyolo.models.ecdet.decoder import ECTransformer as LDec

    enc_kwargs = dict(
        in_channels=[192, 192, 192],
        hidden_dim=192,
        dim_feedforward=512,
        depth_mult=0.67,
        expansion=0.34,
        eval_spatial_size=[640, 640],
        csp_type="csp2",
        fuse_op="sum",
    )
    dec_kwargs = dict(
        num_classes=80,
        hidden_dim=192,
        feat_channels=[192, 192, 192],
        dim_feedforward=512,
        num_layers=4,
        num_points=[3, 6, 3],
        eval_idx=-1,
        eval_spatial_size=[640, 640],
    )

    up_bb = UpBB(name="ecvitt", embed_dim=192, num_heads=3,
                 interaction_indexes=[10, 11], skip_load_backbone=True)
    lb_bb = LBB(embed_dim=192, num_heads=3, interaction_indexes=(10, 11))
    up_enc = UpEnc(**enc_kwargs)
    lb_enc = LEnc(**enc_kwargs)
    up_dec = UpDec(**dec_kwargs)
    lb_dec = LDec(**dec_kwargs)

    ck = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)["model"]
    bb_sd = {k.removeprefix("backbone."): v for k, v in ck.items() if k.startswith("backbone.")}
    enc_sd = {k.removeprefix("encoder."): v for k, v in ck.items() if k.startswith("encoder.")}
    dec_sd = {k.removeprefix("decoder."): v for k, v in ck.items() if k.startswith("decoder.")}

    for u, l, sd in [(up_bb, lb_bb, bb_sd), (up_enc, lb_enc, enc_sd)]:
        u.load_state_dict(sd, strict=True)
        l.load_state_dict(sd, strict=True)
    # Decoder: upstream has segmentation_head args; we don't. Use strict=False
    # on upstream-only since our decoder is det-only.
    up_dec.load_state_dict(dec_sd, strict=False)
    lb_dec.load_state_dict(dec_sd, strict=False)

    for m in (up_bb, lb_bb, up_enc, lb_enc, up_dec, lb_dec):
        m.eval()

    torch.manual_seed(42)
    x = torch.randn(1, 3, 640, 640)
    with torch.no_grad():
        up_feats = up_enc(up_bb(x))
        lb_feats = lb_enc(lb_bb(x))
        for i, (u, l) in enumerate(zip(up_feats, lb_feats)):
            err = (u - l).abs().max().item()
            assert err < 1e-5, f"encoder level {i}: max err {err:.2e}"

        up_out = up_dec(up_feats)
        lb_out = lb_dec(lb_feats)

    for k in ("pred_logits", "pred_boxes"):
        u, l = up_out[k], lb_out[k]
        assert u.shape == l.shape
        err = (u - l).abs().max().item()
        assert err < 1e-5, f"{k}: max err {err:.2e}"
