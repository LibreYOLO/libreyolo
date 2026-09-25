"""Offline unit tests for the ocr task and the LibrePPOCR family.

These require no weights: they cover the task contract (Results.ocr payload,
JSON summary, slicing), the DB quad postprocess on synthetic probability maps,
CTC greedy decode on synthetic logits, the crop warp, reading-order sort,
checkpoint discrimination, the dataset resolver on the ocr20 fixture, the
validator metric functions on hand-computed micro-fixtures, and the train()
and export guards.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.unit

FIXT = Path(__file__).resolve().parents[1] / "fixtures" / "ocr20"


# --------------------------------------------------------------------------
# Task registration
# --------------------------------------------------------------------------


def test_ocr_task_aliases():
    from libreyolo.tasks import normalize_task, suffix_to_task, task_to_suffix

    assert normalize_task("ocr") == "ocr"
    assert normalize_task("text") == "ocr"
    assert normalize_task("text-recognition") == "ocr"
    assert normalize_task("text_recognition") == "ocr"
    assert task_to_suffix("ocr") == "ocr"
    assert suffix_to_task("-ocr") == "ocr"


# --------------------------------------------------------------------------
# Results payload: OCRRegions
# --------------------------------------------------------------------------


def _regions(n=2):
    from libreyolo.utils.results import OCRRegions

    polys = torch.tensor(
        [[[0.0, 0.0], [10.0, 0.0], [10.0, 4.0], [0.0, 4.0]]] * n
    )
    return OCRRegions(polys, [f"t{i}" for i in range(n)], [0.9] * n, [0.8] * n, (20, 30))


def test_ocr_payload_shapes_and_fields():
    r = _regions(3)
    assert len(r) == 3
    assert r.texts == ["t0", "t1", "t2"]
    assert r.xyxy.shape == (3, 4)
    assert float(r.conf[0]) == pytest.approx(0.9)
    assert float(r.det_conf[0]) == pytest.approx(0.8)


def test_ocr_payload_rejects_bad_shapes():
    from libreyolo.utils.results import OCRRegions

    with pytest.raises(ValueError):
        OCRRegions(torch.zeros(2, 3, 2))
    with pytest.raises(ValueError):
        OCRRegions(torch.zeros(2, 4, 2), ["only-one"])
    with pytest.raises(ValueError):
        OCRRegions(torch.zeros(2, 4, 2), ["a", "b"], [0.5])


def test_ocr_payload_empty():
    from libreyolo.utils.results import OCRRegions

    r = OCRRegions(torch.zeros(0, 4, 2), [])
    assert len(r) == 0
    assert r.xyxy.shape == (0, 4)


def test_ocr_payload_slice_and_moves_keep_texts():
    r = _regions(3)
    sliced = r[1]
    assert len(sliced) == 1 and sliced.texts == ["t1"]
    assert r.numpy().texts == r.cpu().texts == ["t0", "t1", "t2"]


def test_results_ocr_summary_and_len():
    from libreyolo.utils.results import Results

    result = Results(boxes=None, orig_shape=(20, 30), ocr=_regions(2))
    assert len(result) == 2
    rows = json.loads(result.to_json())
    assert rows[0]["text"] == "t0"
    assert rows[0]["polygon"]["x"] == [0.0, 10.0, 10.0, 0.0]
    normalized = result.summary(normalize=True)
    assert normalized[0]["polygon"]["x"][1] == pytest.approx(10.0 / 30.0, abs=1e-5)


# --------------------------------------------------------------------------
# DB postprocess geometry
# --------------------------------------------------------------------------


def test_db_postprocess_synthetic_rectangle():
    from libreyolo.postprocess.ppocr import db_postprocess

    prob = np.zeros((160, 320), dtype=np.float32)
    prob[40:80, 60:200] = 0.9
    boxes, scores = db_postprocess(prob, 320, 160)
    assert boxes.shape == (1, 4, 2)
    assert scores[0] == pytest.approx(0.9, abs=1e-3)
    x1, y1 = boxes[0].min(axis=0)
    x2, y2 = boxes[0].max(axis=0)
    # The unclip expansion must strictly contain the source blob.
    assert x1 < 60 and y1 < 40 and x2 > 199 and y2 > 79
    # DB unclip distance for this blob: area*1.5/perimeter = 5600*1.5/360 ~ 23.3 px.
    assert x1 == pytest.approx(60 - 23.3, abs=2.5)


def test_db_postprocess_empty_and_low_score():
    from libreyolo.postprocess.ppocr import db_postprocess

    empty = np.zeros((64, 64), dtype=np.float32)
    boxes, scores = db_postprocess(empty, 64, 64)
    assert boxes.shape == (0, 4, 2) and scores == []

    weak = np.zeros((64, 64), dtype=np.float32)
    weak[10:30, 10:50] = 0.4  # above thresh=0.3 but below box_thresh=0.6
    boxes, _ = db_postprocess(weak, 64, 64)
    assert boxes.shape == (0, 4, 2)


def test_db_postprocess_rotated_blob_quad():
    import cv2

    from libreyolo.postprocess.ppocr import db_postprocess

    prob = np.zeros((200, 200), dtype=np.float32)
    quad = np.array([[[60, 90], [140, 60], [150, 90], [70, 120]]], dtype=np.int32)
    cv2.fillPoly(prob, quad, 0.95)
    boxes, scores = db_postprocess(prob, 200, 200)
    assert boxes.shape[0] == 1
    # Rotated source must produce a genuinely rotated quad (not axis-aligned).
    ys = boxes[0][:, 1]
    assert len(np.unique(np.round(ys))) > 2


def test_sort_quads_reading_order():
    from libreyolo.postprocess.ppocr import sort_quads_reading_order

    def quad(x, y):
        return [[x, y], [x + 20, y], [x + 20, y + 10], [x, y + 10]]

    quads = np.array(
        [quad(200, 52), quad(10, 50), quad(10, 5)], dtype=np.float32
    )
    order = sort_quads_reading_order(quads)
    assert order.tolist() == [2, 1, 0]


# --------------------------------------------------------------------------
# CTC decode
# --------------------------------------------------------------------------


def test_ctc_decode_collapses_repeats_and_blanks():
    from libreyolo.postprocess.ppocr import ctc_decode

    charset = ["blank", "a", "b", "c"]
    # Timesteps: a a blank b b c -> "abc"
    probs = np.zeros((1, 6, 4), dtype=np.float32)
    for t, idx in enumerate([1, 1, 0, 2, 2, 3]):
        probs[0, t, idx] = 0.9
    [(text, conf)] = ctc_decode(probs, charset)
    assert text == "abc"
    assert conf == pytest.approx(0.9, abs=1e-6)


def test_ctc_decode_all_blank_is_empty_with_zero_conf():
    from libreyolo.postprocess.ppocr import ctc_decode

    probs = np.zeros((1, 4, 3), dtype=np.float32)
    probs[:, :, 0] = 1.0
    [(text, conf)] = ctc_decode(probs, ["blank", "a", "b"])
    assert text == "" and conf == 0.0


def test_ctc_decode_keeps_repeat_after_blank():
    from libreyolo.postprocess.ppocr import ctc_decode

    charset = ["blank", "o"]
    probs = np.zeros((1, 3, 2), dtype=np.float32)
    for t, idx in enumerate([1, 0, 1]):  # o blank o -> "oo"
        probs[0, t, idx] = 1.0
    [(text, _)] = ctc_decode(probs, charset)
    assert text == "oo"


# --------------------------------------------------------------------------
# Crop warp
# --------------------------------------------------------------------------


def test_rotate_crop_recovers_axis_aligned_patch():
    from libreyolo.models.ppocr.preprocess import get_rotate_crop_image

    img = np.zeros((100, 200, 3), dtype=np.uint8)
    img[20:40, 50:150] = (10, 200, 30)
    quad = np.array([[50, 20], [150, 20], [150, 40], [50, 40]], dtype=np.float32)
    crop = get_rotate_crop_image(img, quad)
    assert crop.shape[:2] == (20, 100)
    assert (crop[10, 50] == (10, 200, 30)).all()


def test_rotate_crop_rotates_tall_quads():
    from libreyolo.models.ppocr.preprocess import get_rotate_crop_image

    img = np.random.default_rng(0).integers(0, 255, (200, 100, 3), dtype=np.uint8)
    quad = np.array([[30, 20], [60, 20], [60, 170], [30, 170]], dtype=np.float32)
    crop = get_rotate_crop_image(img, quad)
    # h/w = 150/30 = 5 >= 1.5, so the crop comes back rotated to landscape.
    assert crop.shape[0] < crop.shape[1]


def test_rec_batches_bucket_by_aspect():
    from libreyolo.models.ppocr.preprocess import rec_batches

    crops = [
        np.zeros((48, 100, 3), dtype=np.uint8),
        np.zeros((48, 800, 3), dtype=np.uint8),
        np.zeros((48, 90, 3), dtype=np.uint8),
    ]
    batches = rec_batches(crops, batch_size=2)
    assert [sorted(idx) for idx, _ in batches] == [[0, 2], [1]]
    # First bucket keeps the default 320 width; the wide crop drives its own.
    assert batches[0][1].shape == (2, 3, 48, 320)
    assert batches[1][1].shape[3] == int(48 * (800 / 48))


# --------------------------------------------------------------------------
# Checkpoint discrimination
# --------------------------------------------------------------------------


def _composite_keys(size: str):
    if size == "t":
        stem = "det.backbone.conv1.conv.weight"
    else:
        stem = "det.backbone.stem.stem1.conv.weight"
    return {
        stem: torch.zeros(16, 3, 3, 3),
        "det.head.binarize.conv1.weight": torch.zeros(24, 96, 3, 3),
        "rec.head.ctc_head.fc.weight": torch.zeros(18385, 120),
    }


def test_can_load_and_size_detection():
    from libreyolo.models.ppocr.model import LibrePPOCR

    assert LibrePPOCR.can_load(_composite_keys("t"))
    assert LibrePPOCR.detect_size(_composite_keys("t")) == "t"
    assert LibrePPOCR.detect_size(_composite_keys("l")) == "l"
    assert LibrePPOCR.detect_checkpoint_task(_composite_keys("l")) == "ocr"
    assert LibrePPOCR.detect_nb_classes(_composite_keys("t")) == 1


def test_can_load_rejects_foreign_checkpoints():
    from libreyolo.models.ppocr.model import LibrePPOCR

    assert not LibrePPOCR.can_load({"model.0.conv.weight": torch.zeros(1)})
    # Half a composite (det without rec) is rejected too.
    partial = {"det.head.binarize.conv1.weight": torch.zeros(1)}
    assert not LibrePPOCR.can_load(partial)


def test_filename_contract():
    from libreyolo.models.ppocr.model import LibrePPOCR

    assert LibrePPOCR.detect_size_from_filename("LibrePPOCRt-ocr.pt") == "t"
    assert LibrePPOCR.detect_task_from_filename("LibrePPOCRl-ocr.pt") == "ocr"
    # REQUIRE_TASK_SUFFIX: a bare filename is not a canonical ocr checkpoint.
    assert LibrePPOCR.detect_size_from_filename("LibrePPOCRl.pt") is None
    url = LibrePPOCR.get_download_url("LibrePPOCRl-ocr.pt")
    assert url.endswith("LibrePPOCRl-ocr/resolve/main/LibrePPOCRl-ocr.pt")


def test_train_and_export_guards():
    from libreyolo.models.ppocr.model import LibrePPOCR

    model = LibrePPOCR(model_path=None, size="t")
    with pytest.raises(NotImplementedError, match="convert_ppocr_weights"):
        model.train()
    with pytest.raises(NotImplementedError, match="two networks"):
        model.export(format="onnx")


def test_predict_without_charset_raises():
    from libreyolo.models.ppocr.model import LibrePPOCR

    model = LibrePPOCR(model_path=None, size="t")
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    with pytest.raises(RuntimeError, match="charset"):
        model.predict(img)


def test_predict_in_memory_source_can_plot():
    """plot() works on an OCR result from an in-memory image (no path)."""
    from libreyolo.models.ppocr.model import PPOCR_V5_NUM_CLASSES, LibrePPOCR

    model = LibrePPOCR(model_path=None, size="t")
    model.charset = ["x"] * PPOCR_V5_NUM_CLASSES
    img = np.full((48, 64, 3), 255, dtype=np.uint8)
    result = model.predict(img)

    assert result.path is None
    assert result.orig_img.shape == (48, 64, 3)
    plotted = result.plot()
    assert isinstance(plotted, np.ndarray)
    assert plotted.shape == (48, 64, 3)


# --------------------------------------------------------------------------
# Dataset resolver on the committed fixture
# --------------------------------------------------------------------------


def test_resolve_ocr_samples_fixture():
    from libreyolo.data.ocr_dataset import resolve_ocr_samples

    samples = resolve_ocr_samples(FIXT, split="val")
    assert len(samples) == 20
    image_path, regions = samples[0]
    assert image_path.name == "ocr_00.png"
    assert regions[0]["polygon"].shape == (4, 2)
    assert regions[0]["text"] == "LibreYOLO"
    dont_care = [r for _, regs in samples for r in regs if r["text"] == "###"]
    assert len(dont_care) == 2


def test_resolve_ocr_samples_missing_dir():
    from libreyolo.data.ocr_dataset import resolve_ocr_samples

    with pytest.raises(ValueError, match="not found"):
        resolve_ocr_samples(FIXT, split="train")


# --------------------------------------------------------------------------
# Validator metrics on hand-computed micro-fixtures
# --------------------------------------------------------------------------


def _square(x, y, s=10.0):
    return np.array([[x, y], [x + s, y], [x + s, y + s], [x, y + s]], dtype=np.float64)


def test_polygon_iou_identity_disjoint_half():
    from libreyolo.validation.ocr_validator import polygon_iou

    a = _square(0, 0)
    assert polygon_iou(a, a) == pytest.approx(1.0)
    assert polygon_iou(a, _square(100, 100)) == 0.0
    # Half overlap: inter 50, union 150 -> 1/3.
    assert polygon_iou(a, _square(5, 0)) == pytest.approx(1.0 / 3.0, abs=1e-6)


def test_text_normalization_and_ned():
    from libreyolo.validation.ocr_validator import (
        edit_distance,
        normalize_text,
        one_minus_ned,
    )

    assert normalize_text(" Hello  World ") == "HelloWorld"
    assert normalize_text("ﬁt") == "fit"  # NFKC folds the ligature
    assert normalize_text("Case") != normalize_text("case")  # case-sensitive
    assert edit_distance("kitten", "sitting") == 3
    assert one_minus_ned("abc", "abc") == 1.0
    assert one_minus_ned("", "") == 1.0
    assert one_minus_ned("abcd", "abce") == pytest.approx(0.75)


def test_match_image_counts():
    from libreyolo.validation.ocr_validator import match_image

    gt = [
        {"polygon": _square(0, 0), "text": "one"},
        {"polygon": _square(50, 0), "text": "two"},
        {"polygon": _square(100, 0), "text": "###"},
    ]
    preds = [
        _square(0.5, 0.5),  # matches gt "one"
        _square(50, 0),  # matches gt "two" but transcript is wrong
        _square(100, 0),  # sits on the don't-care region -> not penalized
        _square(200, 0),  # false positive
    ]
    texts = ["one", "TWO", "junk", "ghost"]
    counts = match_image(preds, texts, gt)
    assert counts["num_gt"] == 2
    assert counts["num_pred"] == 3  # don't-care hit removed
    assert counts["det_matches"] == 2
    assert counts["e2e_matches"] == 1
    assert len(counts["ned_scores"]) == 2


def test_match_image_one_to_one():
    from libreyolo.validation.ocr_validator import match_image

    gt = [{"polygon": _square(0, 0), "text": "x"}]
    preds = [_square(0, 0), _square(1, 1)]  # both overlap the single GT
    counts = match_image(preds, ["x", "x"], gt)
    assert counts["det_matches"] == 1
    assert counts["num_pred"] == 2  # the second stays a false positive


def test_match_image_optimal_assignment_in_crowded_layouts():
    """A greedy best-IoU-first pass would strand pred B here.

    Pred A overlaps both GTs (best IoU 0.82 with GT1, 0.67 with GT2); pred B
    only clears the threshold with GT1 (0.67). Greedy takes A-GT1 first and
    ends with 1 match; the optimal one-to-one assignment pairs A-GT2 and
    B-GT1 for 2 matches.
    """
    from libreyolo.validation.ocr_validator import match_image

    def box(x1, x2):
        return np.array([[x1, 0], [x2, 0], [x2, 10], [x1, 10]], dtype=np.float64)

    gt = [
        {"polygon": box(0, 20), "text": "one"},
        {"polygon": box(6, 26), "text": "two"},
    ]
    pred_a = box(2, 22)
    pred_b = box(0, 16)
    counts = match_image([pred_a, pred_b], ["two", "one"], gt)
    assert counts["det_matches"] == 2
    assert counts["e2e_matches"] == 2
