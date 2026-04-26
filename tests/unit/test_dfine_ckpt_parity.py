"""Checkpoint-level parity between LibreDFINE and upstream D-FINE.

Loads the same ``dfine_n_coco.pth`` into both our port and the upstream
reference model (by importing only the modules we need, sidestepping the
reference's heavyweight package imports), runs both forward on the same
deterministic input, and asserts element-wise closeness of ``pred_logits``
and ``pred_boxes``.

Skipped if either the reference repo or the checkpoint is missing.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.unit

from libreyolo import LibreDFINE

_DFINE_REF_PATH_ENV = os.environ.get("LIBREYOLO_DFINE_REF_PATH")
_DFINE_REF_PATH = Path(_DFINE_REF_PATH_ENV) if _DFINE_REF_PATH_ENV else None
_CKPT_PATH = Path("weights/dfine_n_coco.pth")


def _load_ref_dfine_module():
    """Import the upstream D-FINE decoder/encoder/backbone as standalone modules."""
    if _DFINE_REF_PATH is None:
        pytest.skip("LIBREYOLO_DFINE_REF_PATH unset; point it at a D-FINE clone to run this parity test")
    dfine_src = _DFINE_REF_PATH / "src"
    if not dfine_src.is_dir():
        pytest.skip(f"D-FINE reference not at {_DFINE_REF_PATH}")

    pkg_name = "_dfine_ref_full"
    pkg = sys.modules.get(pkg_name)
    if pkg is None:
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [str(dfine_src)]
        sys.modules[pkg_name] = pkg

    # Satisfy relative imports inside the reference modules by preloading dependencies.
    def _load(modpath: str, relpath: str):
        fq = f"{pkg_name}.{modpath}"
        if fq in sys.modules:
            return sys.modules[fq]
        # Create parent packages as empty namespaces.
        parts = modpath.split(".")
        parent = pkg_name
        for part in parts[:-1]:
            qual = f"{parent}.{part}"
            if qual not in sys.modules:
                sub = types.ModuleType(qual)
                sub.__path__ = [str(dfine_src / parent.split(".", 1)[-1].replace(".", "/")) if "." in parent else str(dfine_src / "")]
                # Actually simpler: compute path from relpath's leading dirs
                sub.__path__ = [str((dfine_src / relpath).parent.parent)]
                sys.modules[qual] = sub
                setattr(sys.modules[parent], part, sub)
            parent = qual
        spec = importlib.util.spec_from_file_location(fq, dfine_src / relpath)
        module = importlib.util.module_from_spec(spec)
        sys.modules[fq] = module
        if "." in modpath:
            setattr(sys.modules[parent], modpath.rsplit(".", 1)[-1], module)
        spec.loader.exec_module(module)
        return module

    # Stub the ``core`` registry since the decoder imports ``register`` from it.
    core_pkg_name = f"{pkg_name}.core"
    if core_pkg_name not in sys.modules:
        core_mod = types.ModuleType(core_pkg_name)
        core_mod.__path__ = [str(dfine_src / "core")]

        def register(*args, **kwargs):
            def deco(cls):
                return cls

            return deco

        core_mod.register = register
        sys.modules[core_pkg_name] = core_mod
        setattr(pkg, "core", core_mod)

    # Order matters — dependencies first.
    common = _load("nn.backbone.common", "nn/backbone/common.py")
    hgnetv2 = _load("nn.backbone.hgnetv2", "nn/backbone/hgnetv2.py")
    box_ops = _load("zoo.dfine.box_ops", "zoo/dfine/box_ops.py")
    utils = _load("zoo.dfine.utils", "zoo/dfine/utils.py")
    dfine_utils = _load("zoo.dfine.dfine_utils", "zoo/dfine/dfine_utils.py")
    denoising = _load("zoo.dfine.denoising", "zoo/dfine/denoising.py")
    hybrid_encoder = _load("zoo.dfine.hybrid_encoder", "zoo/dfine/hybrid_encoder.py")
    dfine_decoder = _load("zoo.dfine.dfine_decoder", "zoo/dfine/dfine_decoder.py")

    return {
        "HGNetv2": hgnetv2.HGNetv2,
        "HybridEncoder": hybrid_encoder.HybridEncoder,
        "DFINETransformer": dfine_decoder.DFINETransformer,
    }


import torch.nn as nn


class RefDFINE(nn.Module):
    def __init__(self, backbone, encoder, decoder):
        super().__init__()
        self.backbone = backbone
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x, targets=None):
        feats = self.backbone(x)
        feats = self.encoder(feats)
        return self.decoder(feats, targets=targets)


# Per-size reference model configuration, read directly from D-FINE's shipped
# YAMLs. Must match libreyolo/models/dfine/nn.py::SIZE_CONFIGS.
REF_SIZE_KWARGS = {
    "n": dict(
        bb_name="B0", use_lab=True, return_idx=[2, 3], freeze_norm=False,
        enc_in_channels=[512, 1024], enc_feat_strides=[16, 32],
        enc_hidden_dim=128, enc_dim_ff=512, enc_expansion=0.34, enc_depth_mult=0.5,
        enc_use_encoder_idx=[1],
        dec_feat_channels=[128, 128], dec_feat_strides=[16, 32],
        dec_hidden_dim=128, dec_dim_ff=512, dec_num_levels=2, dec_num_layers=3,
        dec_num_points=[6, 6], reg_scale=4.0,
    ),
    "s": dict(
        bb_name="B0", use_lab=True, return_idx=[1, 2, 3], freeze_norm=False,
        enc_in_channels=[256, 512, 1024], enc_feat_strides=[8, 16, 32],
        enc_hidden_dim=256, enc_dim_ff=1024, enc_expansion=0.5, enc_depth_mult=0.34,
        enc_use_encoder_idx=[2],
        dec_feat_channels=[256, 256, 256], dec_feat_strides=[8, 16, 32],
        dec_hidden_dim=256, dec_dim_ff=1024, dec_num_levels=3, dec_num_layers=3,
        dec_num_points=[3, 6, 3], reg_scale=4.0,
    ),
    "m": dict(
        bb_name="B2", use_lab=True, return_idx=[1, 2, 3], freeze_norm=False,
        enc_in_channels=[384, 768, 1536], enc_feat_strides=[8, 16, 32],
        enc_hidden_dim=256, enc_dim_ff=1024, enc_expansion=1.0, enc_depth_mult=0.67,
        enc_use_encoder_idx=[2],
        dec_feat_channels=[256, 256, 256], dec_feat_strides=[8, 16, 32],
        dec_hidden_dim=256, dec_dim_ff=1024, dec_num_levels=3, dec_num_layers=4,
        dec_num_points=[3, 6, 3], reg_scale=4.0,
    ),
    "l": dict(
        bb_name="B4", use_lab=False, return_idx=[1, 2, 3], freeze_norm=True,
        enc_in_channels=[512, 1024, 2048], enc_feat_strides=[8, 16, 32],
        enc_hidden_dim=256, enc_dim_ff=1024, enc_expansion=1.0, enc_depth_mult=1.0,
        enc_use_encoder_idx=[2],
        dec_feat_channels=[256, 256, 256], dec_feat_strides=[8, 16, 32],
        dec_hidden_dim=256, dec_dim_ff=1024, dec_num_levels=3, dec_num_layers=6,
        dec_num_points=[3, 6, 3], reg_scale=4.0,
    ),
    "x": dict(
        bb_name="B5", use_lab=False, return_idx=[1, 2, 3], freeze_norm=True,
        enc_in_channels=[512, 1024, 2048], enc_feat_strides=[8, 16, 32],
        enc_hidden_dim=384, enc_dim_ff=2048, enc_expansion=1.0, enc_depth_mult=1.0,
        enc_use_encoder_idx=[2],
        dec_feat_channels=[384, 384, 384], dec_feat_strides=[8, 16, 32],
        dec_hidden_dim=256, dec_dim_ff=1024, dec_num_levels=3, dec_num_layers=6,
        dec_num_points=[3, 6, 3], reg_scale=8.0,
    ),
}


def _build_reference(size: str):
    ref = _load_ref_dfine_module()
    kw = REF_SIZE_KWARGS[size]
    backbone = ref["HGNetv2"](
        name=kw["bb_name"],
        use_lab=kw["use_lab"],
        return_idx=kw["return_idx"],
        freeze_stem_only=True,
        freeze_at=0,
        freeze_norm=kw["freeze_norm"],
        pretrained=False,
    )
    encoder = ref["HybridEncoder"](
        in_channels=kw["enc_in_channels"],
        feat_strides=kw["enc_feat_strides"],
        hidden_dim=kw["enc_hidden_dim"],
        dim_feedforward=kw["enc_dim_ff"],
        expansion=kw["enc_expansion"],
        depth_mult=kw["enc_depth_mult"],
        use_encoder_idx=kw["enc_use_encoder_idx"],
        eval_spatial_size=(640, 640),
    )
    decoder = ref["DFINETransformer"](
        num_classes=80,
        hidden_dim=kw["dec_hidden_dim"],
        feat_channels=kw["dec_feat_channels"],
        feat_strides=kw["dec_feat_strides"],
        num_levels=kw["dec_num_levels"],
        num_points=kw["dec_num_points"],
        num_layers=kw["dec_num_layers"],
        dim_feedforward=kw["dec_dim_ff"],
        eval_spatial_size=(640, 640),
        eval_idx=-1,
        reg_scale=kw["reg_scale"],
    )
    m = RefDFINE(backbone, encoder, decoder)
    ckpt = torch.load(f"weights/dfine_{size}_coco.pth", map_location="cpu", weights_only=False)
    state = ckpt.get("ema", {}).get("module") if isinstance(ckpt.get("ema"), dict) else None
    state = state or ckpt["model"]
    m.load_state_dict(state, strict=True)
    m.eval()
    return m


@pytest.mark.parametrize("size", ["n", "s", "m", "l", "x"])
def test_parity_all_sizes(size):
    ckpt_path = Path(f"weights/dfine_{size}_coco.pth")
    if not ckpt_path.exists():
        pytest.skip(f"{ckpt_path} not present")

    ref = _build_reference(size)
    ours = LibreDFINE(str(ckpt_path), size=size, device="cpu")
    ours.model.eval()

    torch.manual_seed(0)
    x = torch.randn(1, 3, 640, 640)
    with torch.no_grad():
        theirs = ref(x)
        ourout = ours.model(x)

    logit_max_abs_diff = (theirs["pred_logits"] - ourout["pred_logits"]).abs().max().item()
    box_max_abs_diff = (theirs["pred_boxes"] - ourout["pred_boxes"]).abs().max().item()
    assert logit_max_abs_diff < 1e-5, f"{size}: pred_logits max abs diff = {logit_max_abs_diff}"
    assert box_max_abs_diff < 1e-5, f"{size}: pred_boxes max abs diff = {box_max_abs_diff}"
