"""Pinned upstream graph alignment with all four real checkpoints.

CPU substitutes the portable attention and normalization operators in the
reference. This tests architecture and weight mapping, not FLA kernel parity.
Set GTR_UPSTREAM and GTR_CHECKPOINTS to local checkouts/artifacts. No downloads.
"""

import os
import subprocess

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.external_data]


def test_upstream_graph_alignment_with_portable_operators(monkeypatch):
    if not os.environ.get("GTR_UPSTREAM") or not os.environ.get("GTR_CHECKPOINTS"):
        pytest.skip("Set GTR_UPSTREAM and GTR_CHECKPOINTS")
    revision = subprocess.check_output(
        ["git", "-C", os.environ["GTR_UPSTREAM"], "rev-parse", "HEAD"], text=True
    ).strip()
    assert revision == "782e737efe2e6437ac537fbdcee089673d3376c1"
    import pathlib
    import sys
    import types

    import torch

    from libreyolo.models.gtr.attention import RMSNormGated, recurrent_gla
    from libreyolo.models.gtr.nn import LibreGTRModel

    root = pathlib.Path(os.environ["GTR_UPSTREAM"]) / "engine"

    def package(n, p=None):
        m = types.ModuleType(n)
        m.__path__ = [str(p)] if p else []
        monkeypatch.setitem(sys.modules, n, m)
        return m

    package("gtr_reference", root)
    package("gtr_reference.gtr", root / "gtr")
    package("gtr_reference.gtr.backbone", root / "gtr/backbone")
    package("gtr_reference.gtr.det", root / "gtr/det")
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
    import yaml
    from gtr_reference.gtr.backbone.vit_adapter_spatial_swiglu import (
        ViTAdapterSpatialSwiGLU,
    )
    from gtr_reference.gtr.det.decoder import GTRTransformer
    from gtr_reference.gtr.gtr import GTR
    from gtr_reference.gtr.hybrid_encoder import GTREncoder

    def merge(left, right):
        for key, value in right.items():
            if isinstance(value, dict) and isinstance(left.get(key), dict):
                merge(left[key], value)
            else:
                left[key] = value
        return left

    def config(path):
        content = yaml.safe_load(path.read_text())
        result = {}
        for include in content.pop("__include__", []):
            merge(result, config(path.parent / include))
        return merge(result, content)

    old_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    for size in ("s", "m", "l", "x"):
        torch.manual_seed(45)
        ours = LibreGTRModel(size, eval_spatial_size=(640, 640)).eval()
        sd = torch.load(
            pathlib.Path(os.environ["GTR_CHECKPOINTS"]) / f"gtr_{size}_coco.pth",
            weights_only=True,
            map_location="cpu",
        )["ema"]["module"]
        ours.load_state_dict(sd, strict=True)
        cfg = config(root.parent / f"configs/det/coco_finetune/gtr_{size}.yml")
        backbone = dict(cfg["ViTAdapterSpatialSwiGLU"])
        backbone.update(
            weights_path=None,
            skip_weights_warning=True,
            eval_spatial_size=cfg["eval_spatial_size"],
        )
        encoder = dict(cfg["GTREncoder"])
        encoder["eval_spatial_size"] = cfg["eval_spatial_size"]
        decoder = dict(cfg["GTRTransformer"])
        decoder.update(
            num_classes=cfg["num_classes"], eval_spatial_size=cfg["eval_spatial_size"]
        )
        upstream = GTR(
            ViTAdapterSpatialSwiGLU(**backbone),
            GTREncoder(**encoder),
            GTRTransformer(**decoder),
        ).eval()
        upstream.load_state_dict(ours.state_dict(), strict=True)
        x = torch.randn(1, 3, 640, 640)
        with torch.no_grad():
            a = ours(x)
            b = upstream(x)
        for k in a:
            diff = (a[k] - b[k]).abs().max().item()
            print(size, k, diff, flush=True)
            assert diff == 0
    torch.set_num_threads(old_threads)
