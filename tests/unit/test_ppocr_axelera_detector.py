"""PP-OCR text detector on the Axelera Metis: canvas preprocessing, host decode and pipeline wiring (no device)."""

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
