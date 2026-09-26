"""Pinned upstream GTR depth graph alignment with the four real checkpoints.

Same portable-operator substitution as ``test_gtr_parity.py``. Set GTR_UPSTREAM
to the pinned checkout and GTR_CHECKPOINTS to a directory holding
``gtrdepth_{s,m,l,x}.pth`` (directly or under ``depth/``). No downloads.
"""

import os
import subprocess

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.external_data]


def _checkpoint(size):
    import pathlib

    root = pathlib.Path(os.environ["GTR_CHECKPOINTS"])
    for path in (
        root / f"gtrdepth_{size}.pth",
        root / "depth" / f"gtrdepth_{size}.pth",
    ):
        if path.exists():
            return path
    pytest.skip(f"gtrdepth_{size}.pth not found under GTR_CHECKPOINTS")


def _stub_upstream(monkeypatch, root):
    import sys
    import types

    from libreyolo.models.gtr.attention import RMSNormGated, recurrent_gla

    def package(n, p=None):
        m = types.ModuleType(n)
        m.__path__ = [str(p)] if p else []
        monkeypatch.setitem(sys.modules, n, m)
        return m

    package("gtr_reference", root)
    package("gtr_reference.gtr", root / "gtr")
    package("gtr_reference.gtr.backbone", root / "gtr/backbone")
    package("gtr_reference.gtr.depth", root / "gtr/depth")
    core = package("gtr_reference.core")
    core.register = lambda: lambda cls: cls
    for n in [
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
        package(n)

    class Norm(RMSNormGated):
        def __init__(self, hidden_size, elementwise_affine=True, eps=1e-5, **kw):
            super().__init__(hidden_size, eps)

    sys.modules["fla.modules"].FusedRMSNormGated = Norm
    sys.modules["fla.modules"].RMSNorm = Norm
    sys.modules["fla.modules"].ShortConvolution = None
    sys.modules["fla.modules.activations"].ACT2FN = {}
    sys.modules["fla.models.utils"].Cache = None
    sys.modules["transformers.processing_utils"].Unpack = None
    for n in ["get_unpad_data", "index_first_axis", "pad_input"]:
        setattr(sys.modules["fla.layers.utils"], n, None)

    def op(q, k, v, g=None, gk=None, **kw):
        return recurrent_gla(q, k, v, g if g is not None else gk), None

    for n in ["chunk_gla", "fused_chunk_gla", "fused_recurrent_gla"]:
        setattr(sys.modules["fla.ops.gla"], n, op)


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


def test_upstream_depth_graph_alignment_with_portable_operators(monkeypatch):
    if not os.environ.get("GTR_UPSTREAM") or not os.environ.get("GTR_CHECKPOINTS"):
        pytest.skip("Set GTR_UPSTREAM and GTR_CHECKPOINTS")
    revision = subprocess.check_output(
        ["git", "-C", os.environ["GTR_UPSTREAM"], "rev-parse", "HEAD"], text=True
    ).strip()
    assert revision == "782e737efe2e6437ac537fbdcee089673d3376c1"
    import pathlib

    import torch

    from libreyolo.models.gtr.depth import LibreGTRDepthModel

    root = pathlib.Path(os.environ["GTR_UPSTREAM"]) / "engine"
    _stub_upstream(monkeypatch, root)
    from gtr_reference.gtr.backbone.vit_adapter_spatial_swiglu import (
        ViTAdapterSpatialSwiGLU,
    )
    from gtr_reference.gtr.depth.gtrdepth import DPTDepthHead, GTRDepth
    from gtr_reference.gtr.hybrid_encoder import GTREncoder

    old_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        for size in ("s", "m", "l", "x"):
            state = torch.load(
                _checkpoint(size), weights_only=False, map_location="cpu"
            )["ema"]["module"]
            ours = LibreGTRDepthModel(size).eval()
            ours.load_state_dict(state, strict=True)
            cfg = _config(root.parent / f"configs/depth/pretrain/gtrdepth_{size}.yml")
            backbone = dict(cfg["ViTAdapterSpatialSwiGLU"])
            backbone.update(
                weights_path=None,
                skip_weights_warning=True,
                eval_spatial_size=cfg["eval_spatial_size"],
            )
            encoder = dict(cfg["GTREncoder"])
            encoder["eval_spatial_size"] = cfg["eval_spatial_size"]
            head = dict(cfg["DPTDepthHead"])
            head["max_depth"] = cfg["max_depth"]
            upstream = GTRDepth(
                ViTAdapterSpatialSwiGLU(**backbone),
                GTREncoder(**encoder),
                DPTDepthHead(**head),
            ).eval()
            upstream.load_state_dict(state, strict=True)
            torch.manual_seed(0)
            x = torch.rand(1, 3, 640, 640)
            normalized = (x - ours.pixel_mean) / ours.pixel_std
            with torch.no_grad():
                a = ours.forward_metric(x)
                b = upstream(normalized)["pred_depth"]
                inverse = ours(x)
            diff = (a - b).abs().max().item()
            print(size, "pred_depth max abs diff", diff, flush=True)
            assert diff == 0
            torch.testing.assert_close(inverse[:, 0], b.reciprocal())
    finally:
        torch.set_num_threads(old_threads)
