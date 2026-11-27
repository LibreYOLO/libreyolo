"""PP-OCR text detector on the Axelera Metis: canvas preprocessing, host decode and pipeline wiring (no device)."""

from pathlib import Path

import numpy as np
import pytest
import torch

from libreyolo.models.ppocr.axelera_detector import (
    OUTPUT_WINDOW,
    PPOCR_DB_LAYOUT,
    _DetLogits,
    canvas_input,
    fuse_rep_layers,
    metis_forms,
    probability_map,
)
from libreyolo.models.ppocr.backbones import Act, LearnableAffineBlock, LearnableRepLayer
from libreyolo.models.ppocr.det import SEModule
from libreyolo.models.ppocr.det import PPOCRDetModel
from libreyolo.models.ppocr.preprocess import det_normalize, det_resize

pytestmark = pytest.mark.unit


def _image(h, w, seed=0):
    return np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8)


def test_canvas_input_equals_native_preprocess_when_it_fits():
    img = _image(720, 1280)
    blob, h, w = canvas_input(img, (960, 960))
    native = det_normalize(det_resize(img, limit_side_len=960)[0]).numpy()
    assert (h, w) == native.shape[2:] == (544, 960)
    np.testing.assert_array_equal(blob[:, :, :h, :w], native)
    assert not blob[:, :, h:, :].any()


def test_canvas_input_shrinks_to_fit_a_narrow_canvas():
    blob, h, w = canvas_input(_image(1280, 720), (544, 960))
    assert blob.shape == (1, 3, 544, 960)
    assert h <= 544 and w <= 960 and h % 32 == 0 and w % 32 == 0
    assert not blob[:, :, :, w:].any()


def test_probability_map_crops_and_applies_sigmoid():
    logits = np.zeros((1, 1, 64, 96), np.float32)
    logits[0, 0, :32, :64] = 10.0
    prob = probability_map(logits, 32, 64)
    assert prob.shape == (32, 64)
    assert prob.min() > 0.99


def test_logit_graph_matches_the_fp32_detector_after_sigmoid():
    torch.manual_seed(0)
    det = PPOCRDetModel("t").eval()
    net = _DetLogits.build(det)
    x = torch.randn(1, 3, 128, 160)
    with torch.no_grad():
        ref = det(x)
        got = torch.sigmoid(net(x))
    torch.testing.assert_close(got, ref, atol=1e-6, rtol=0)


def _randomize_bn(module, seed=0):
    g = torch.Generator().manual_seed(seed)
    for m in module.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.running_mean.copy_(torch.randn(m.num_features, generator=g) * 0.1)
            m.running_var.copy_(torch.rand(m.num_features, generator=g) + 0.5)
            m.weight.data.copy_(torch.rand(m.num_features, generator=g) + 0.5)
            m.bias.data.copy_(torch.randn(m.num_features, generator=g) * 0.1)
        if isinstance(m, LearnableAffineBlock):
            m.scale.data.fill_(1.3)
            m.bias.data.fill_(-0.2)


@pytest.mark.parametrize("rewrite", [fuse_rep_layers, metis_forms])
def test_metis_rewrites_are_exact(rewrite):
    torch.manual_seed(0)
    det = PPOCRDetModel("t").eval()
    _randomize_bn(det)
    x = torch.randn(1, 3, 128, 160)
    with torch.no_grad():
        ref = _DetLogits.build(det)(x)
    import copy

    fused = rewrite(copy.deepcopy(det)).eval()
    assert not any(isinstance(m, LearnableRepLayer) for m in fused.modules())
    if rewrite is metis_forms:
        assert not any(isinstance(m, (SEModule, Act, LearnableAffineBlock)) for m in fused.modules())
    with torch.no_grad():
        got = _DetLogits.build(fused)(x)
    torch.testing.assert_close(got, ref, atol=1e-4 * float(ref.abs().max()), rtol=0)


def test_window_round_trips_through_the_host_decode():
    torch.manual_seed(0)
    det = PPOCRDetModel("t").eval()
    x = torch.randn(1, 3, 64, 96)
    with torch.no_grad():
        logits = _DetLogits.build(det)(x).numpy()
        windowed = _DetLogits.build(det, OUTPUT_WINDOW)(x).numpy()
    assert windowed.min() >= 0.0 and windowed.max() <= 6.0
    expected = 1.0 / (1.0 + np.exp(-np.clip(logits[0, 0], *OUTPUT_WINDOW)))
    np.testing.assert_allclose(probability_map(windowed, 64, 96, OUTPUT_WINDOW), expected, atol=1e-5)


def test_server_head_is_not_wired_yet():
    with pytest.raises(NotImplementedError):
        _DetLogits.build(PPOCRDetModel("l").eval())


def test_export_rejects_unknown_arguments():
    from libreyolo.models.ppocr.model import LibrePPOCR

    model = LibrePPOCR(size="t")
    with pytest.raises(TypeError, match="batch"):
        model.export(format="axelera", batch=4)


def test_pipeline_takes_the_probability_map_from_the_text_detector(monkeypatch):
    from PIL import Image

    from libreyolo.models.ppocr.inference import OCRInferenceRunner
    from libreyolo.models.ppocr.model import LibrePPOCR

    model = LibrePPOCR(size="t")
    model.charset = ["blank", "a", " "]
    calls = []

    class _Detector:
        def prob_map(self, image_bgr):
            calls.append(image_bgr.shape)
            return np.zeros(image_bgr.shape[:2], np.float32)

    model.text_detector = _Detector()
    monkeypatch.setattr(model, "forward_det", lambda *a, **k: pytest.fail("FP32 detector ran"))
    result = OCRInferenceRunner(model).run_pipeline(
        Image.fromarray(_image(64, 96)), conf=0.0, imgsz=None, rec_batch=1, image_path=None
    )
    assert calls == [(64, 96, 3)]
    assert len(result.ocr) == 0


def test_layout_name_is_versioned():
    assert PPOCR_DB_LAYOUT == "ppocr_db_logits_v1"


def _ocr_root(base, split="train", images="images"):
    from PIL import Image

    folder = base / images / split
    folder.mkdir(parents=True)
    for name in ("b.jpg", "a.png"):
        Image.new("RGB", (32, 32)).save(folder / name)
    (folder / "notes.txt").write_text("not an image")
    return [folder / "a.png", folder / "b.jpg"]


@pytest.mark.parametrize(
    "layout",
    ["minimal_yaml", "yaml_without_path", "relative_yaml_without_path", "named_dirs_yaml", "directory", "val_only"],
)
def test_calibration_images_accepts_every_ocr_layout(tmp_path, monkeypatch, layout):
    """docs/dataset_schema.md: a root directory, or a YAML with ``path`` plus optional
    ``images``/``labels``; calibration needs images only, no JSONL labels."""
    from libreyolo.models.ppocr.axelera_detector import _calibration_images

    names = "nc: 1\nnames: {0: text}\n"
    if layout == "minimal_yaml":
        expected = _ocr_root(tmp_path / "ds")
        (tmp_path / "data.yaml").write_text(f"path: ds\n{names}")
        data = tmp_path / "data.yaml"
    elif layout == "yaml_without_path":
        expected = _ocr_root(tmp_path / "ds")
        (tmp_path / "ds" / "data.yaml").write_text(names)
        data = tmp_path / "ds" / "data.yaml"
    elif layout == "relative_yaml_without_path":
        expected = _ocr_root(tmp_path / "ds")
        (tmp_path / "ds" / "data.yaml").write_text(names)
        monkeypatch.chdir(tmp_path)
        data = "ds/data.yaml"  # the YAML's folder, not ds/ds
        expected = [p.resolve() for p in expected]
    elif layout == "named_dirs_yaml":
        expected = _ocr_root(tmp_path / "ds", images="imgs")
        (tmp_path / "data.yaml").write_text(f"path: {tmp_path / 'ds'}\nimages: imgs\nlabels: lbls\n{names}")
        data = tmp_path / "data.yaml"
    elif layout == "directory":
        expected = _ocr_root(tmp_path / "ds")
        data = tmp_path / "ds"
    else:  # val_only
        expected = _ocr_root(tmp_path / "ds", split="val")
        data = tmp_path / "ds"
    got = _calibration_images(data)
    assert [Path(p).resolve() for p in got] == [Path(p).resolve() for p in expected]


def test_calibration_images_without_an_image_split_says_where_it_looked(tmp_path):
    from libreyolo.models.ppocr.axelera_detector import _calibration_images

    (tmp_path / "data.yaml").write_text("nc: 1\nnames: {0: text}\n")
    with pytest.raises(ValueError, match="images/<train|val>"):
        _calibration_images(tmp_path / "data.yaml")


def test_resolve_ocr_root_reads_path_relative_to_the_yaml(tmp_path, monkeypatch):
    from libreyolo.data.ocr_dataset import resolve_ocr_root

    (tmp_path / "ds").mkdir()
    (tmp_path / "ds" / "data.yaml").write_text("nc: 1\n")
    monkeypatch.chdir(tmp_path)
    assert resolve_ocr_root("ds/data.yaml") == ((tmp_path / "ds").resolve(), "images", "labels")
    (tmp_path / "ds" / "data.yaml").write_text("path: root\nimages: i\nlabels: l\n")
    assert resolve_ocr_root("ds/data.yaml") == ((tmp_path / "ds" / "root").resolve(), "i", "l")
