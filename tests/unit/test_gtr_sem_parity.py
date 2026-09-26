"""Pinned upstream GTR semantic graph alignment with the real checkpoints.

CPU substitutes the portable attention and normalization operators in the
reference, as in test_gtr_parity.py. Covers the single 1024px window and the
upstream sliding-window evaluation on a Cityscapes-shaped 1024x2048 input.
Set GTR_UPSTREAM and GTR_SEM_CHECKPOINTS (gtrsemseg_{s,m,l,x}_cityscapes.pth).
Optionally set GTR_SEM_SIZES (default "smlx"). No downloads.
"""

import ast
import os
import pathlib
import subprocess
import sys
import types

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.external_data]


def _install_reference(monkeypatch, root):
    from libreyolo.models.gtr.attention import RMSNormGated, recurrent_gla

    def package(name, path=None):
        module = types.ModuleType(name)
        module.__path__ = [str(path)] if path else []
        monkeypatch.setitem(sys.modules, name, module)
        return module

    package("gtr_reference", root)
    package("gtr_reference.gtr", root / "gtr")
    package("gtr_reference.gtr.backbone", root / "gtr/backbone")
    package("gtr_reference.gtr.semseg", root / "gtr/semseg")
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


def _upstream_slide_inference(root):
    """Load ``slide_inference`` alone; its module imports the full solver."""
    import torch

    source = (root / "solver/semseg_solver.py").read_text()
    tree = ast.parse(source)
    node = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "slide_inference"
    )
    namespace = {"torch": torch}
    exec(compile(ast.Module([node], []), "semseg_solver.py", "exec"), namespace)
    return namespace["slide_inference"]


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


def test_upstream_semantic_graph_alignment(monkeypatch):
    if not os.environ.get("GTR_UPSTREAM") or not os.environ.get("GTR_SEM_CHECKPOINTS"):
        pytest.skip("Set GTR_UPSTREAM and GTR_SEM_CHECKPOINTS")
    upstream_root = pathlib.Path(os.environ["GTR_UPSTREAM"])
    revision = subprocess.check_output(
        ["git", "-C", str(upstream_root), "rev-parse", "HEAD"], text=True
    ).strip()
    assert revision == "782e737efe2e6437ac537fbdcee089673d3376c1"
    import torch

    from libreyolo.models.gtr.sem import LibreGTRSemModel

    root = upstream_root / "engine"
    _install_reference(monkeypatch, root)
    from gtr_reference.gtr.backbone.vit_adapter_spatial_swiglu import (
        ViTAdapterSpatialSwiGLU,
    )
    from gtr_reference.gtr.hybrid_encoder import GTREncoder
    from gtr_reference.gtr.semseg.gtrsemseg import GTRSemSeg, SemSegHead

    slide_inference = _upstream_slide_inference(root)
    old_threads = torch.get_num_threads()
    torch.set_num_threads(4)
    try:
        for size in os.environ.get("GTR_SEM_SIZES", "smlx"):
            ours = LibreGTRSemModel(size).eval()
            sd = torch.load(
                pathlib.Path(os.environ["GTR_SEM_CHECKPOINTS"])
                / f"gtrsemseg_{size}_cityscapes.pth",
                weights_only=True,
                map_location="cpu",
            )["ema"]["module"]
            ours.load_state_dict(sd, strict=True)
            cfg = _config(
                upstream_root
                / f"configs/semseg/cityscapes_finetune/gtrsemseg_{size}.yml"
            )
            window = tuple(cfg["eval_spatial_size"])
            assert window == (ours.window, ours.window)
            assert tuple(cfg["slide_stride"]) == (768, 768)
            backbone = dict(cfg["ViTAdapterSpatialSwiGLU"])
            backbone.update(
                weights_path=None, skip_weights_warning=True, eval_spatial_size=window
            )
            encoder = dict(cfg["GTREncoder"])
            encoder["eval_spatial_size"] = window
            upstream = GTRSemSeg(
                ViTAdapterSpatialSwiGLU(**backbone),
                GTREncoder(**encoder),
                SemSegHead(**cfg["SemSegHead"]),
            ).eval()
            upstream.load_state_dict(ours.state_dict(), strict=True)

            torch.manual_seed(7)
            image = torch.rand(1, 3, 1024, 2048)
            normalized = (image - ours.pixel_mean) / ours.pixel_std
            with torch.no_grad():
                a = ours.forward_normalized(normalized[..., :1024])
                b = upstream(normalized[..., :1024])["pred_sem_seg"]
                window_diff = (a - b).abs().max().item()
                a = ours(image)
                b = slide_inference(upstream, normalized, window, (768, 768))
                slide_diff = (a - b["pred_sem_seg"]).abs().max().item()
            print(size, "window", window_diff, "slide", slide_diff, flush=True)
            assert window_diff == 0
            assert slide_diff < 1e-5
    finally:
        torch.set_num_threads(old_threads)
