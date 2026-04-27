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


# (size, ckpt_basename, upstream_name, embed, heads, proj_dim, ffn_ratio,
#  enc_hidden, enc_ff, enc_exp, enc_dep, dec_hidden, dec_ff)
SIZE_PARAMS = [
    ("s", "ecdet_s.pth", "ecvitt",     192, 3, None, 4.0, 192, 512,  0.34, 0.67, 192, 512),
    ("m", "ecdet_m.pth", "ecvittplus", 256, 4, None, 4.0, 256, 512,  0.75, 0.67, 256, 1024),
    ("l", "ecdet_l.pth", "ecvits",     384, 6, 256,  4.0, 256, 1024, 0.75, 1.0,  256, 1024),
    ("x", "ecdet_x.pth", "ecvitsplus", 384, 6, 256,  6.0, 256, 2048, 1.5,  1.0,  256, 2048),
]
WEIGHTS_DIR = Path("downloads/ec_weights")


@pytest.mark.parametrize("params", SIZE_PARAMS, ids=[p[0] for p in SIZE_PARAMS])
def test_backbone_parity(params):
    size, ckpt_name, upstream_name, embed, heads, proj_dim, ffn_ratio, *_ = params
    ckpt = WEIGHTS_DIR / ckpt_name
    if not ckpt.exists():
        pytest.skip(f"{ckpt} not found")
    UpBB, _, _ = _try_import_upstream()
    from libreyolo.models.ecdet.backbone import ViTAdapter as LibreBB

    up_kw = dict(name=upstream_name, embed_dim=embed, num_heads=heads,
                 interaction_indexes=[10, 11], ffn_ratio=ffn_ratio,
                 skip_load_backbone=True)
    lb_kw = dict(embed_dim=embed, num_heads=heads,
                 interaction_indexes=(10, 11), ffn_ratio=ffn_ratio)
    if proj_dim is not None:
        up_kw["proj_dim"] = proj_dim
        lb_kw["proj_dim"] = proj_dim

    upstream = UpBB(**up_kw)
    libre = LibreBB(**lb_kw)

    ck = torch.load(ckpt, map_location="cpu", weights_only=False)["model"]
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
        assert u.shape == l.shape, f"{size} level {i}: {u.shape} vs {l.shape}"
        err = (u - l).abs().max().item()
        assert err < 1e-5, f"{size} level {i}: max err {err:.2e}"


@pytest.mark.parametrize("params", SIZE_PARAMS, ids=[p[0] for p in SIZE_PARAMS])
def test_full_pipeline_parity(params):
    (size, ckpt_name, upstream_name, embed, heads, proj_dim, ffn_ratio,
     enc_hidden, enc_ff, enc_exp, enc_dep, dec_hidden, dec_ff) = params
    ckpt = WEIGHTS_DIR / ckpt_name
    if not ckpt.exists():
        pytest.skip(f"{ckpt} not found")
    UpBB, UpEnc, UpDec = _try_import_upstream()
    from libreyolo.models.ecdet.backbone import ViTAdapter as LBB
    from libreyolo.models.ecdet.encoder import HybridEncoder as LEnc
    from libreyolo.models.ecdet.decoder import ECTransformer as LDec

    enc_kwargs = dict(
        in_channels=[enc_hidden] * 3,
        hidden_dim=enc_hidden,
        dim_feedforward=enc_ff,
        depth_mult=enc_dep,
        expansion=enc_exp,
        eval_spatial_size=[640, 640],
        csp_type="csp2",
        fuse_op="sum",
    )
    dec_kwargs = dict(
        num_classes=80,
        hidden_dim=dec_hidden,
        feat_channels=[dec_hidden] * 3,
        dim_feedforward=dec_ff,
        num_layers=4,
        num_points=[3, 6, 3],
        eval_idx=-1,
        eval_spatial_size=[640, 640],
    )

    up_bb_kw = dict(name=upstream_name, embed_dim=embed, num_heads=heads,
                    interaction_indexes=[10, 11], ffn_ratio=ffn_ratio,
                    skip_load_backbone=True)
    lb_bb_kw = dict(embed_dim=embed, num_heads=heads,
                    interaction_indexes=(10, 11), ffn_ratio=ffn_ratio)
    if proj_dim is not None:
        up_bb_kw["proj_dim"] = proj_dim
        lb_bb_kw["proj_dim"] = proj_dim

    up_bb, lb_bb = UpBB(**up_bb_kw), LBB(**lb_bb_kw)
    up_enc, lb_enc = UpEnc(**enc_kwargs), LEnc(**enc_kwargs)
    up_dec, lb_dec = UpDec(**dec_kwargs), LDec(**dec_kwargs)

    ck = torch.load(ckpt, map_location="cpu", weights_only=False)["model"]
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
