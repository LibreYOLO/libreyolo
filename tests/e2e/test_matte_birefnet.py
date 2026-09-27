"""Gated end-to-end tests for LibreYOLO matte families.

These need real weights. ``LibreBiRefNetl-matte.pt`` auto-downloads from the
LibreYOLO Hugging Face org (``LibreYOLO/LibreBiRefNetl-matte``); the lite ``t``
tests use a locally converted ``weights/LibreBiRefNett-matte.pt`` when present.
``LibreBEN2b-matte.pt`` likewise auto-downloads from its public LibreYOLO repo.
Each test skips cleanly when its weights cannot be obtained (offline CI).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

pytestmark = pytest.mark.e2e

REPO = Path(__file__).resolve().parents[2]
FIXT = REPO / "tests" / "fixtures" / "matte8"
_SAMPLE = REPO / "libreyolo" / "assets" / "parkour.jpg"
_LOCAL_T = REPO / "weights" / "LibreBiRefNett-matte.pt"


def _load(name: str, device: str = "cpu"):
    from libreyolo import LibreYOLO

    try:
        return LibreYOLO(name, device=device)
    except Exception as exc:  # network/weights unavailable
        pytest.skip(f"Could not obtain {name}: {exc}")


def test_matte_predict_cutout_save_autodownload(tmp_path):
    """l (general) auto-downloads from HF and the full matte UX works."""
    model = _load("LibreBiRefNetl-matte.pt")
    assert model.FAMILY == "birefnet" and model.size == "l" and model.task == "matte"

    res = model.predict(str(_SAMPLE), verbose=False)[0]
    assert res.matte is not None
    w, h = Image.open(_SAMPLE).size
    matte = res.matte.array
    assert matte.shape == (h, w)  # original canvas
    assert 0.0 <= float(matte.min()) and float(matte.max()) <= 1.0

    cut = res.cutout()
    assert cut.shape == (h, w, 4) and cut.dtype == np.uint8

    out = tmp_path / "cutout.png"
    res.save(out)
    saved = Image.open(out)
    assert saved.mode == "RGBA" and saved.size == (w, h)


def test_matte_ben2_predict_cutout_save_autodownload(tmp_path):
    """BEN2 auto-downloads from HF and preserves the original image canvas."""
    model = _load("LibreBEN2b-matte.pt", device="auto")
    assert model.FAMILY == "ben2" and model.size == "b" and model.task == "matte"

    res = model.predict(str(_SAMPLE), verbose=False)[0]
    assert res.matte is not None
    w, h = Image.open(_SAMPLE).size
    matte = res.matte.array
    assert matte.shape == (h, w)
    assert 0.0 <= float(matte.min()) and float(matte.max()) <= 1.0

    out = tmp_path / "ben2-cutout.png"
    res.save(out)
    saved = Image.open(out)
    assert saved.mode == "RGBA" and saved.size == (w, h)


@pytest.mark.skipif(not _LOCAL_T.exists(), reason="lite weights not staged locally")
def test_matte_golden_stable_cpu():
    """OUR CPU t-model output reproduces the pinned goldens (pipeline is stable)."""
    model = _load(str(_LOCAL_T))
    for name in ("product_circle", "fine_hair_star", "portrait_silhouette"):
        img = FIXT / "images" / f"{name}.png"
        golden = np.asarray(Image.open(FIXT / "goldens_t" / f"{name}.png").convert("L"), np.int16)
        pred = model.predict(str(img), verbose=False)[0].matte.array
        pred_u8 = np.rint(np.clip(pred, 0, 1) * 255).astype(np.int16)
        max_abs = int(np.abs(pred_u8 - golden).max())
        assert max_abs <= 2, f"{name}: golden drift max_abs={max_abs} (>2 LSB)"


@pytest.mark.skipif(not _LOCAL_T.exists(), reason="lite weights not staged locally")
def test_matte_matches_official_reference():
    """OUR matte agrees with the official reference mattes on matte8 (MAE small)."""
    from libreyolo.validation.matte_validator import matte_mae

    model = _load(str(_LOCAL_T))
    for name in ("product_circle", "portrait_silhouette", "two_shapes"):
        img = FIXT / "images" / f"{name}.png"
        ref = np.asarray(Image.open(FIXT / "mattes" / f"{name}.png").convert("L"), np.float32) / 255.0
        pred = model.predict(str(img), verbose=False)[0].matte.array
        assert matte_mae(pred, ref) <= 0.02


_LOCAL_FEYNOBG = REPO / "weights" / "LibreFeyNobgl-matte.pt"
_LOCAL_FEYNOBG_FP16 = REPO / "weights" / "LibreFeyNobgl-matte-fp16.pt"
_LOCAL_FEYNOBG_FP8 = REPO / "weights" / "LibreFeyNobgl-matte-fp8.pt"
_LOCAL_FEYNOBG_NVFP4 = REPO / "weights" / "LibreFeyNobgl-matte-nvfp4.pt"


def test_matte_feynobg_autodownload_or_local(tmp_path):
    """LibreFeyNobg loads (HF auto-download, or local staging) and predicts."""
    name = str(_LOCAL_FEYNOBG) if _LOCAL_FEYNOBG.exists() else "LibreFeyNobgl-matte.pt"
    model = _load(name)
    assert model.FAMILY == "feynobg" and model.size == "l" and model.task == "matte"

    res = model.predict(str(_SAMPLE), verbose=False)[0]
    w, h = Image.open(_SAMPLE).size
    assert res.matte is not None and res.matte.array.shape == (h, w)


@pytest.mark.parametrize(
    "path, recipe",
    [
        (_LOCAL_FEYNOBG_FP16, "fp16"),
        (_LOCAL_FEYNOBG_FP8, "fp8"),
        (_LOCAL_FEYNOBG_NVFP4, "nvfp4"),
    ],
    ids=["fp16", "fp8", "nvfp4"],
)
def test_matte_feynobg_quantized_checkpoint_loads_and_predicts(path, recipe):
    """Quantized LibreFeyNobg checkpoints load via plain weights= and predict.

    The quantized variants never auto-download; users fetch them from HF and
    pass the .pt path directly, so that exact flow is what we exercise.
    fp16 is a cast recipe (no packed finalized state) and is GPU-oriented, so
    its predict leg runs only when CUDA is available.
    """
    if not path.exists():
        pytest.skip(f"{path.name} not staged locally")
    import torch

    if recipe == "fp16" and not torch.cuda.is_available():
        pytest.skip("fp16 cast checkpoint is impractically slow on CPU")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _load(str(path), device=device)
    assert model.FAMILY == "feynobg" and model.size == "l" and model.task == "matte"
    info = model.quant_info()
    assert info is not None and info["recipe"] == recipe
    if recipe != "fp16":
        assert info.get("state") == "finalized"

    res = model.predict(str(_SAMPLE), verbose=False)[0]
    w, h = Image.open(_SAMPLE).size
    assert res.matte is not None and res.matte.array.shape == (h, w)


def test_matte_feynobg_cuda_graph_parity():
    """Graphed predict (encoder capture, eager deformable decoder) is
    bit-identical to eager for every staged precision variant."""
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA required for graph capture")
    staged = [p for p in (_LOCAL_FEYNOBG, _LOCAL_FEYNOBG_FP16, _LOCAL_FEYNOBG_FP8) if p.exists()]
    if not staged:
        pytest.skip("no feynobg weights staged locally")
    for path in staged:
        model = _load(str(path), device="cuda")
        eager = model.predict(str(_SAMPLE), verbose=False)[0].matte.array
        model.predict(str(_SAMPLE), verbose=False, cuda_graph=True)
        graphed = model.predict(str(_SAMPLE), verbose=False, cuda_graph=True)[0].matte.array
        assert float(np.abs(graphed - eager).max()) == 0.0, path.name
        del model
        torch.cuda.empty_cache()
