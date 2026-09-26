"""Pinned upstream GTR pose graph alignment with the four real checkpoints.

Same portable-operator substitution as ``test_gtr_parity.py``: architecture and
weight mapping, not FLA kernel parity. The shared ECPose ``Integral`` sums
elementwise products instead of calling ``F.linear`` (an MPS backward
workaround), so outputs agree to float32 rounding rather than bit for bit. Set GTR_UPSTREAM and
GTR_POSE_CHECKPOINTS (a directory with ``gtrpose_{s,m,l,x}_coco.pth``). No
downloads.
"""

import os
import subprocess

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.external_data]


def _install_reference(monkeypatch, root):
    import sys
    import types

    from libreyolo.models.gtr.attention import RMSNormGated, recurrent_gla

    def package(name, path=None):
        module = types.ModuleType(name)
        module.__path__ = [str(path)] if path else []
        monkeypatch.setitem(sys.modules, name, module)
        return module

    package("gtr_reference", root)
    package("gtr_reference.gtr", root / "gtr")
    package("gtr_reference.gtr.backbone", root / "gtr/backbone")
    package("gtr_reference.gtr.pose", root / "gtr/pose")
    core = package("gtr_reference.core")
    core.register = lambda: lambda cls: cls
    for name in [
        "fla",
        "fla.layers",
        "fla.layers.utils",
        "fla.modules",
        "fla.modules.activations",
        "fla.ops",
        "fla.ops.gla",
        "fla.models",
        "fla.models.utils",
        "transformers",
        "transformers.processing_utils",
    ]:
        package(name)

    class Norm(RMSNormGated):
        def __init__(self, hidden_size, elementwise_affine=True, eps=1e-5, **kw):
            super().__init__(hidden_size, eps)

    sys.modules["fla.modules"].FusedRMSNormGated = Norm
    sys.modules["fla.modules"].RMSNorm = Norm
    sys.modules["fla.modules"].ShortConvolution = None
    sys.modules["fla.modules.activations"].ACT2FN = {}
    sys.modules["fla.models.utils"].Cache = None
    sys.modules["transformers.processing_utils"].Unpack = None
    for name in ["get_unpad_data", "index_first_axis", "pad_input"]:
        setattr(sys.modules["fla.layers.utils"], name, None)

    def op(q, k, v, g=None, gk=None, **kw):
        return recurrent_gla(q, k, v, g if g is not None else gk), None

    for name in ["chunk_gla", "fused_chunk_gla", "fused_recurrent_gla"]:
        setattr(sys.modules["fla.ops.gla"], name, op)


def _config(path):
    import yaml

    def merge(left, right):
        for key, value in right.items():
            if isinstance(value, dict) and isinstance(left.get(key), dict):
                merge(left[key], value)
            else:
                left[key] = value
        return left

    content = yaml.safe_load(path.read_text())
    result = {}
    for include in content.pop("__include__", []):
        merge(result, _config(path.parent / include))
    return merge(result, content)


def test_upstream_pose_graph_alignment_with_portable_operators(monkeypatch):
    if not os.environ.get("GTR_UPSTREAM") or not os.environ.get("GTR_POSE_CHECKPOINTS"):
        pytest.skip("Set GTR_UPSTREAM and GTR_POSE_CHECKPOINTS")
    revision = subprocess.check_output(
        ["git", "-C", os.environ["GTR_UPSTREAM"], "rev-parse", "HEAD"], text=True
    ).strip()
    assert revision == "782e737efe2e6437ac537fbdcee089673d3376c1"
    import pathlib

    import torch

    from libreyolo.models.gtr.pose import LibreGTRPoseModel

    root = pathlib.Path(os.environ["GTR_UPSTREAM"]) / "engine"
    _install_reference(monkeypatch, root)
    from gtr_reference.gtr.backbone.vit_adapter_spatial_swiglu import (
        ViTAdapterSpatialSwiGLU,
    )
    from gtr_reference.gtr.hybrid_encoder import GTREncoder
    from gtr_reference.gtr.pose.decoder import GTRPoseTransformer
    from gtr_reference.gtr.pose.gtrpose import GTRPose

    checkpoints = pathlib.Path(os.environ["GTR_POSE_CHECKPOINTS"])
    old_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        for size in ("s", "m", "l", "x"):
            torch.manual_seed(45)
            ours = LibreGTRPoseModel(size).eval()
            sd = torch.load(
                checkpoints / f"gtrpose_{size}_coco.pth",
                weights_only=True,
                map_location="cpu",
            )["ema"]["module"]
            ours.load_state_dict(sd, strict=True)
            cfg = _config(
                root.parent / f"configs/pose/coco_pose_finetune/gtrpose_{size}.yml"
            )
            spatial = cfg["eval_spatial_size"]
            backbone = dict(cfg["ViTAdapterSpatialSwiGLU"])
            backbone.update(
                weights_path=None, skip_weights_warning=True, eval_spatial_size=spatial
            )
            encoder = dict(cfg["GTREncoder"], eval_spatial_size=spatial)
            decoder = dict(
                cfg["GTRPoseTransformer"],
                num_classes=cfg["num_classes"],
                eval_spatial_size=spatial,
            )
            upstream = GTRPose(
                ViTAdapterSpatialSwiGLU(**backbone),
                GTREncoder(**encoder),
                GTRPoseTransformer(**decoder),
            ).eval()
            upstream.load_state_dict(sd, strict=True)
            x = torch.randn(1, 3, 640, 640)
            with torch.no_grad():
                a = ours(x)
                b = upstream(x)
            for key in ("pred_logits", "pred_keypoints"):
                diff = (a[key] - b[key]).abs().max().item()
                print(size, key, diff, flush=True)
                assert diff < 1e-4
    finally:
        torch.set_num_threads(old_threads)
