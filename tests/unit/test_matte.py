"""Offline unit tests for the matte task and the LibreBiRefNet family.

These require no weights: they cover the task contract (Results payload, cutout,
transparent-PNG save, plotting), postprocess, the validator metric functions,
paired-folder resolution, checkpoint discrimination, and the train() guard.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

pytestmark = pytest.mark.unit

FIXT = Path(__file__).resolve().parents[1] / "fixtures" / "matte8"


@pytest.mark.slow
@pytest.mark.skipif(
    os.environ.get("LIBREYOLO_RUN_REAL_EXPORT_PARITY") != "1",
    reason="set LIBREYOLO_RUN_REAL_EXPORT_PARITY=1 for BiRefNet export parity",
)
@pytest.mark.parametrize(
    "format",
    [
        pytest.param(
            "onnx",
            marks=pytest.mark.xfail(
                strict=True,
                reason="ONNX Runtime CPU has no DeformConv(19) implementation",
            ),
        ),
        "torchscript",
    ],
)
def test_birefnet_real_export_raw_parity(tmp_path, format):
    if format == "onnx":
        pytest.importorskip("onnx")
        pytest.importorskip("onnxruntime")

    from libreyolo import LibreBiRefNet, LibreYOLO
    from libreyolo.export.exporter import OnnxExporter

    torch.manual_seed(0)
    model = LibreBiRefNet(None, size="t", device="cpu")
    model.model.eval()
    tensor = torch.rand(1, 3, 1024, 1024)
    exporter = OnnxExporter(model)
    with exporter._model_context("cpu", False, False, 1, (1024, 1024)) as (
        wrapped,
        _,
    ):
        with torch.no_grad():
            expected = wrapped(tensor)
    if isinstance(expected, torch.Tensor):
        expected = (expected,)

    artifact = model.export(
        format=format,
        imgsz=1024,
        dynamic=False,
        simplify=False,
        output_path=str(tmp_path / f"birefnet-matte.{format}"),
    )
    actual = LibreYOLO(artifact, device="cpu")._run_inference(tensor.numpy())

    assert len(actual) == len(expected)
    rtol, atol = (2e-3, 2e-2) if format == "onnx" else (1e-3, 1e-3)
    for actual_output, expected_output in zip(actual, expected):
        np.testing.assert_allclose(
            actual_output,
            expected_output.detach().cpu().numpy(),
            rtol=rtol,
            atol=atol,
        )


# --------------------------------------------------------------------------
# Results payload: Matte
# --------------------------------------------------------------------------


def test_matte_payload_shape_and_clamp():
    from libreyolo.utils.results import Matte

    m = Matte(torch.rand(30, 40) * 2 - 0.5, (30, 40))  # deliberately out of [0,1]
    arr = m.array
    assert arr.shape == (30, 40)
    assert arr.dtype == np.float32
    assert arr.min() >= 0.0 and arr.max() <= 1.0


def test_matte_payload_rejects_non_2d():
    from libreyolo.utils.results import Matte

    with pytest.raises(ValueError):
        Matte(torch.rand(3, 30, 40))


def test_matte_payload_kept_intact_under_slicing():
    from libreyolo.utils.results import Matte

    m = Matte(torch.rand(30, 40))
    # Dense whole-image payload must survive Results slicing paths untouched.
    assert m[0].data.shape == (30, 40)
    assert len(m) == 1


def test_results_matte_cutout_and_save(tmp_path):
    from libreyolo.utils.results import Matte, Results

    h, w = 24, 32
    matte = np.zeros((h, w), np.float32)
    matte[6:18, 8:24] = 1.0
    rgb = (np.random.default_rng(0).integers(0, 255, (h, w, 3))).astype(np.uint8)
    r = Results(
        boxes=None, orig_shape=(h, w), path=None, names={0: "matte"}, matte=Matte(matte)
    )

    cut = r.cutout(image=rgb)
    assert cut.shape == (h, w, 4) and cut.dtype == np.uint8
    # alpha channel equals the (quantized) matte
    assert np.array_equal(cut[..., 3] > 127, matte > 0.5)

    out = tmp_path / "cut.png"
    r.save(out, image=rgb)
    saved = Image.open(out)
    assert saved.mode == "RGBA" and saved.size == (w, h)


def test_results_matte_summary_and_len():
    from libreyolo.utils.results import Matte, Results

    matte = np.zeros((10, 10), np.float32)
    matte[:5] = 1.0  # half foreground
    r = Results(boxes=None, orig_shape=(10, 10), path=None, matte=Matte(matte))
    assert len(r) == 1
    summary = r.summary()
    assert summary[0]["name"] == "matte"
    assert summary[0]["coverage"] == pytest.approx(0.5, abs=1e-6)


def test_save_of_a_non_matte_result_needs_its_source_image():
    # Non-matte results save their annotated image, so without a source
    # image there is nothing to draw on.
    from libreyolo.utils.results import Results

    r = Results(boxes=None, orig_shape=(4, 4), path=None)
    with pytest.raises(ValueError, match="source image"):
        r.save("x.png")


# --------------------------------------------------------------------------
# Drawing
# --------------------------------------------------------------------------


def test_draw_matte_checkerboard_preview():
    from libreyolo.utils.drawing import draw_matte

    img = Image.new("RGB", (40, 30), (200, 50, 50))
    matte = np.zeros((30, 40), np.float32)
    matte[10:20, 10:30] = 1.0
    preview = draw_matte(img, matte)
    assert preview.size == (40, 30) and preview.mode == "RGB"
    arr = np.asarray(preview)
    # foreground region keeps the source colour; background shows the checkerboard
    assert tuple(arr[15, 20]) == (200, 50, 50)
    assert tuple(arr[2, 2]) != (200, 50, 50)


# --------------------------------------------------------------------------
# Postprocess
# --------------------------------------------------------------------------


def test_birefnet_postprocess_resizes_and_clamps():
    from libreyolo.postprocess.birefnet import postprocess

    logits = torch.randn(1, 1, 64, 64) * 5
    out = postprocess(logits, original_size=(100, 80))  # (orig_w, orig_h)
    matte = out["matte"]
    assert matte.shape == (80, 100)
    assert float(matte.min()) >= 0.0 and float(matte.max()) <= 1.0
    # accepts a list output (upstream returns a list of scaled preds)
    out2 = postprocess([logits], original_size=(100, 80))
    assert out2["matte"].shape == (80, 100)


# --------------------------------------------------------------------------
# Validator metrics
# --------------------------------------------------------------------------


def test_matte_mae_and_smeasure_perfect_and_perturbed():
    from libreyolo.validation.matte_validator import matte_mae, s_measure

    gt = np.zeros((64, 64), np.float32)
    gt[16:48, 16:48] = 1.0
    assert matte_mae(gt, gt) == 0.0
    assert s_measure(gt, gt) == pytest.approx(1.0, abs=1e-6)

    worse = gt.copy()
    worse[16:48, 16:48] = 0.3  # foreground under-predicted
    assert matte_mae(worse, gt) > 0.0
    assert s_measure(worse, gt) < s_measure(gt, gt)


def test_smeasure_all_background_and_all_foreground():
    from libreyolo.validation.matte_validator import s_measure

    gt0 = np.zeros((16, 16), np.float32)
    assert s_measure(np.zeros_like(gt0), gt0) == pytest.approx(1.0)
    gt1 = np.ones((16, 16), np.float32)
    assert s_measure(np.ones_like(gt1), gt1) == pytest.approx(1.0)


# --------------------------------------------------------------------------
# Paired-folder dataset resolution
# --------------------------------------------------------------------------


def test_resolve_matte_pairs_on_matte8_fixture():
    from libreyolo.data.matte_dataset import resolve_matte_pairs

    pairs = resolve_matte_pairs(FIXT)
    assert len(pairs) == 8
    for img, matte in pairs:
        assert img.stem == matte.stem
        assert img.exists() and matte.exists()


# --------------------------------------------------------------------------
# Checkpoint discrimination + family metadata
# --------------------------------------------------------------------------


def _synthetic_birefnet_sd(embed_dim: int, stage3_depth: int = 18) -> dict:
    sd = {
        "bb.patch_embed.proj.weight": torch.zeros(embed_dim, 3, 4, 4),
        "squeeze_module.0.conv_in.weight": torch.zeros(1),
        "decoder.ipt_blk5.conv1.weight": torch.zeros(1),
        "decoder.gdt_convs_attn_4.0.weight": torch.zeros(1),
    }
    if stage3_depth >= 24:
        sd["bb.layers.2.blocks.23.norm1.weight"] = torch.zeros(1)
    return sd


def test_birefnet_can_load_and_size_detection():
    from libreyolo.models.birefnet import LibreBiRefNet

    assert LibreBiRefNet.can_load(_synthetic_birefnet_sd(96)) is True
    assert LibreBiRefNet.detect_size(_synthetic_birefnet_sd(96)) == "t"
    assert LibreBiRefNet.detect_size(_synthetic_birefnet_sd(192)) == "l"
    assert LibreBiRefNet.detect_checkpoint_task(_synthetic_birefnet_sd(192)) == "matte"


def test_birefnet_and_feynobg_claims_are_disjoint():
    from libreyolo.models.birefnet import LibreBiRefNet
    from libreyolo.models.feynobg import LibreFeyNobg

    # FeyNobg shares the Swin-L width; the 24-block stage-3 marker key routes
    # the checkpoint to the feynobg family and away from birefnet.
    feynobg_sd = _synthetic_birefnet_sd(192, stage3_depth=24)
    assert LibreBiRefNet.can_load(feynobg_sd) is False
    assert LibreBiRefNet.detect_size(feynobg_sd) is None
    assert LibreFeyNobg.can_load(feynobg_sd) is True
    assert LibreFeyNobg.detect_size(feynobg_sd) == "l"
    assert LibreFeyNobg.detect_checkpoint_task(feynobg_sd) == "matte"

    # birefnet's own checkpoints are never claimed by feynobg.
    assert LibreFeyNobg.can_load(_synthetic_birefnet_sd(192)) is False
    assert LibreFeyNobg.can_load(_synthetic_birefnet_sd(96)) is False


def test_birefnet_rejects_foreign_state_dicts():
    from libreyolo.models.birefnet import LibreBiRefNet

    # NAFNet-like and a generic detector dict must not be claimed.
    assert LibreBiRefNet.can_load({"intro.weight": torch.zeros(32, 3, 3, 3)}) is False
    assert LibreBiRefNet.can_load({"model.0.conv.weight": torch.zeros(1)}) is False


def test_siblings_reject_birefnet_state_dict():
    from libreyolo.models.birefnet import LibreBiRefNet  # noqa: F401
    from libreyolo.models.nafnet import LibreNAFNet

    sd = _synthetic_birefnet_sd(192)
    assert LibreNAFNet.can_load(sd) is False


def test_birefnet_filename_detection():
    from libreyolo.models.birefnet import LibreBiRefNet

    assert LibreBiRefNet.detect_size_from_filename("LibreBiRefNetl-matte.pt") == "l"
    assert LibreBiRefNet.detect_size_from_filename("LibreBiRefNett-matte.pt") == "t"


def test_feynobg_filename_detection():
    from libreyolo.models.feynobg import LibreFeyNobg

    assert LibreFeyNobg.detect_size_from_filename("LibreFeyNobgl-matte.pt") == "l"


def test_birefnet_train_raises():
    from libreyolo.models.birefnet.model import LibreBiRefNet

    # Build without weights (random init) just to reach train()'s guard.
    model = LibreBiRefNet(model_path=None, size="t", device="cpu")
    with pytest.raises(NotImplementedError, match="fine-tun"):
        model.train(data="whatever.yaml")
