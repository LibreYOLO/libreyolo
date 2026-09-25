"""Tests for rectangular (height, width) imgsz support (PR #649).

Covers the CLI parser, TrainConfig validation, per-family preprocessing,
postprocess coordinate handling, and the trainer capability gates.
"""

import numpy as np
import pytest
import torch

from libreyolo.cli.command_utils import parse_imgsz_str
from libreyolo.postprocess.common import _input_size_hw, postprocess_detections
from libreyolo.training.config import TrainConfig
from libreyolo.training.trainer import RECTANGULAR_TRAINING_FAMILIES

pytestmark = pytest.mark.unit

RECT_HW = (64, 320)  # deliberately wide, both divisible by 64


# ---------------------------------------------------------------------------
# CLI parsing
# ---------------------------------------------------------------------------


class TestParseImgszStr:
    def test_square_int_string(self):
        assert parse_imgsz_str("640") == 640

    def test_rect_lowercase(self):
        assert parse_imgsz_str("480x640") == (480, 640)

    def test_rect_uppercase(self):
        assert parse_imgsz_str("480X640") == (480, 640)

    def test_rect_legacy_comma(self):
        assert parse_imgsz_str("480,640") == (480, 640)

    def test_square_tuple_collapses_to_int(self):
        assert parse_imgsz_str("640x640") == 640

    def test_square_legacy_comma_collapses_to_int(self):
        assert parse_imgsz_str("640,640") == 640

    def test_none_passthrough(self):
        assert parse_imgsz_str(None) is None

    def test_int_passthrough(self):
        assert parse_imgsz_str(640) == 640

    @pytest.mark.parametrize("bad", ["-1x0", "0x640", "640x-32", "0", "-640"])
    def test_nonpositive_rejected(self, bad):
        with pytest.raises(ValueError):
            parse_imgsz_str(bad)

    @pytest.mark.parametrize(
        "bad",
        ["abc", "12ax640", "x640", "640x", "480,640,800"],
    )
    def test_garbage_rejected(self, bad):
        with pytest.raises(ValueError):
            parse_imgsz_str(bad)


# ---------------------------------------------------------------------------
# TrainConfig validation
# ---------------------------------------------------------------------------


class TestTrainConfigImgsz:
    def test_scalar_kept(self):
        assert TrainConfig(imgsz=640).imgsz == 640

    def test_rect_tuple_kept(self):
        assert TrainConfig(imgsz=(480, 640)).imgsz == (480, 640)

    def test_rect_list_normalized_to_tuple(self):
        assert TrainConfig(imgsz=[480, 640]).imgsz == (480, 640)

    def test_square_tuple_normalized_to_scalar(self):
        assert TrainConfig(imgsz=(640, 640)).imgsz == 640

    def test_numeric_string_preserved_as_compatibility_input(self):
        assert TrainConfig(imgsz="640").imgsz == 640

    def test_hxw_string_accepted(self):
        assert TrainConfig(imgsz="480x640").imgsz == (480, 640)

    def test_python_comma_string_rejected(self):
        with pytest.raises(ValueError):
            TrainConfig(imgsz="480,640")

    def test_wrong_arity_rejected(self):
        with pytest.raises(ValueError):
            TrainConfig(imgsz=(640, 640, 3))

    def test_nonpositive_rejected(self):
        with pytest.raises(ValueError):
            TrainConfig(imgsz=(0, 640))

    def test_non_int_rejected(self):
        with pytest.raises(TypeError):
            TrainConfig(imgsz=(640.0, 480))

    @pytest.mark.parametrize("bad", [640.0, True, "garbage", "0x640"])
    def test_invalid_scalar_rejected_at_config_boundary(self, bad):
        with pytest.raises((TypeError, ValueError)):
            TrainConfig(imgsz=bad)


# ---------------------------------------------------------------------------
# _input_size_hw helper
# ---------------------------------------------------------------------------


def test_input_size_hw_scalar():
    assert _input_size_hw(640) == (640, 640)


def test_input_size_hw_tuple():
    assert _input_size_hw((480, 640)) == (480, 640)


def test_input_size_hw_list():
    assert _input_size_hw([480, 640]) == (480, 640)


# ---------------------------------------------------------------------------
# Per-family preprocessing emits (C, H, W) with independent H and W
# ---------------------------------------------------------------------------


@pytest.fixture
def wide_image():
    # 100x500 BGR-ish uint8; content is irrelevant, shape flow is under test.
    return np.zeros((100, 500, 3), dtype=np.uint8)


class TestRectPreprocess:
    def test_yolox(self, wide_image):
        from libreyolo.models.yolox.utils import preprocess_numpy

        arr, ratio = preprocess_numpy(wide_image, input_size=RECT_HW)
        assert arr.shape == (3, *RECT_HW)
        assert ratio == pytest.approx(min(RECT_HW[0] / 100, RECT_HW[1] / 500))

    def test_yolo7(self, wide_image):
        from libreyolo.models.yolo7.utils import preprocess_numpy

        arr, _ = preprocess_numpy(wide_image, input_size=RECT_HW)
        assert arr.shape == (3, *RECT_HW)

    def test_rtmdet(self, wide_image):
        from libreyolo.models.rtmdet.utils import preprocess_numpy

        arr, _ = preprocess_numpy(wide_image, input_size=RECT_HW)
        assert arr.shape == (3, *RECT_HW)

    def test_picodet(self, wide_image):
        from libreyolo.models.picodet.utils import preprocess_numpy

        out = preprocess_numpy(wide_image, input_size=RECT_HW)
        arr = out[0] if isinstance(out, tuple) else out
        assert arr.shape == (3, *RECT_HW)

    def test_yolonas_rejects_rect(self, wide_image):
        from libreyolo.models.yolonas.utils import preprocess_numpy

        with pytest.raises(ValueError, match="rectangular"):
            preprocess_numpy(wide_image, input_size=RECT_HW)

    def test_yolonas_square_unchanged(self, wide_image):
        from libreyolo.models.yolonas.utils import preprocess_numpy

        arr, _ = preprocess_numpy(wide_image, input_size=640)
        assert arr.shape == (3, 640, 640)


# ---------------------------------------------------------------------------
# Postprocess coordinate handling
# ---------------------------------------------------------------------------


def test_picodet_clamp_uses_correct_axes():
    """Regression: canvas (h, w) axes were transposed, clamping x to the
    height and y to the width on rectangular input."""
    from libreyolo.postprocess.picodet import _per_level_filter_topk

    h, w = 32, 64
    cls_scores = [torch.full((1, 80, h // 8, w // 8), 5.0)]
    # Large positive logits decode to large distances -> boxes overflow the
    # canvas and must be clamped per-axis.
    bbox_preds = [torch.full((1, 32, h // 8, w // 8), 20.0)]
    _, _, boxes = _per_level_filter_topk(
        cls_scores,
        bbox_preds,
        strides=[8],
        reg_max=7,
        score_thr=0.1,
        nms_pre=100,
        canvas_size=(h, w),
    )
    assert boxes[:, 2].max().item() <= w
    assert boxes[:, 3].max().item() <= h
    # x2 must be able to exceed the height, otherwise the transposition is back.
    assert boxes[:, 2].max().item() > h


def test_postprocess_detections_rect_simple_resize():
    boxes = torch.tensor([[16.0, 8.0, 320.0, 64.0]])
    scores = torch.tensor([0.9])
    class_ids = torch.tensor([0])
    out = postprocess_detections(
        boxes.clone(),
        scores,
        class_ids,
        conf_thres=0.1,
        input_size=(64, 320),
        original_size=(500, 100),  # (w, h)
        letterbox=False,
    )
    got = out["boxes"][0]
    # x scaled by 500/320, y scaled by 100/64
    assert got[0] == pytest.approx(16.0 * 500 / 320)
    assert got[1] == pytest.approx(8.0 * 100 / 64)
    assert got[2] == pytest.approx(500.0)
    assert got[3] == pytest.approx(100.0)


def test_postprocess_detections_rect_letterbox_roundtrip():
    # A 500x100 image letterboxed into (64, 320): r = min(64/100, 320/500)
    r = min(64 / 100, 320 / 500)
    orig_box = [50.0, 20.0, 400.0, 90.0]
    boxes = torch.tensor([[v * r for v in orig_box]])
    out = postprocess_detections(
        boxes.clone(),
        torch.tensor([0.9]),
        torch.tensor([0]),
        conf_thres=0.1,
        input_size=(64, 320),
        original_size=(500, 100),
        letterbox=True,
    )
    got = out["boxes"][0]
    for i, v in enumerate(orig_box):
        assert got[i] == pytest.approx(v, rel=1e-5)


def test_rtmdet_mask_resize_is_per_axis(monkeypatch):
    """Regression: rectangular mask logits were resized with a single
    width-derived size, stretching the mask y-axis. The crop hides the shape
    difference, so this asserts mask *content*: a top-half mask must stay a
    top-half mask after the inverse-letterbox resize."""
    from libreyolo.postprocess import rtmdet as rtm

    def fake_decode_masks(mask_feat, kernels, priors):
        n = priors.shape[0]
        h, w = mask_feat.shape[-2:]
        logits = torch.full((n, h, w), -10.0)
        logits[:, : h // 2, :] = 10.0  # top half positive
        return logits

    monkeypatch.setattr(rtm, "_decode_masks", fake_decode_masks)

    gh, gw = 2, 10  # stride-8 grid for a (16, 80) canvas
    cls_scores = [torch.full((1, 1, gh, gw), -20.0)]
    cls_scores[0][0, 0, 0, 0] = 5.0  # one confident instance
    bbox_preds = [torch.full((1, 4, gh, gw), 4.0)]
    kernel_preds = [torch.zeros((1, 169, gh, gw))]
    mask_feats = torch.zeros((1, 8, gh, gw))

    out = rtm._postprocess_segment(
        (cls_scores, bbox_preds, kernel_preds, mask_feats),
        conf_thres=0.1,
        iou_thres=0.6,
        input_size=(16, 80),
        original_size=(160, 32),  # (w, h): the same 0.5 letterbox both axes
        ratio=0.5,
        max_det=10,
        strides=(8,),
        nms_pre=100,
    )
    masks = out["masks"]
    assert masks.shape[-2:] == (32, 160)
    # Correct per-axis resize keeps the top half positive and the bottom half
    # empty. The old width-derived square resize stretched the y-axis 5x, so
    # the crop kept only stretched top rows and the bottom row came out True.
    assert bool(masks[0, 0, :].any())
    assert not bool(masks[0, -1, :].any())


# ---------------------------------------------------------------------------
# Trainer gates
# ---------------------------------------------------------------------------


def test_allowlist_families_exist():
    from pathlib import Path

    models_dir = Path(__file__).resolve().parents[2] / "libreyolo" / "models"
    for family in RECTANGULAR_TRAINING_FAMILIES:
        assert (models_dir / family).is_dir(), f"allowlisted family {family} missing"


def test_allowlist_excludes_yolonas():
    assert "yolonas" not in RECTANGULAR_TRAINING_FAMILIES


def test_picodet_stride_is_64():
    assert RECTANGULAR_TRAINING_FAMILIES["picodet"] == 64


def _gate_check(family: str, imgsz, task: str = "detect"):
    """Run only the rectangular-imgsz validation block from BaseTrainer.setup."""
    if isinstance(imgsz, (list, tuple)) and int(imgsz[0]) != int(imgsz[1]):
        if family and family.lower() not in RECTANGULAR_TRAINING_FAMILIES:
            raise ValueError("family not supported")
        if task != "detect":
            raise ValueError("task not supported")
        stride = RECTANGULAR_TRAINING_FAMILIES.get(family.lower(), 32)
        h, w = int(imgsz[0]), int(imgsz[1])
        if h % stride != 0 or w % stride != 0:
            raise ValueError("stride")


class TestTrainerGateLogic:
    """The gate lives inline in BaseTrainer.setup(); these tests pin its
    contract via an extracted copy so regressions in the constants or the
    logic shape show up without building a full trainer."""

    def test_rfdetr_rejected(self):
        with pytest.raises(ValueError, match="family"):
            _gate_check("rfdetr", (480, 640))

    def test_yolox_detect_ok(self):
        _gate_check("yolox", (480, 640))

    def test_picodet_needs_64(self):
        with pytest.raises(ValueError, match="stride"):
            _gate_check("picodet", (480, 640))  # 480 % 64 != 0

    def test_picodet_64_multiple_ok(self):
        _gate_check("picodet", (448, 640))

    def test_non_detect_task_rejected(self):
        with pytest.raises(ValueError, match="task"):
            _gate_check("rtmdet", (480, 640), task="segment")


# ---------------------------------------------------------------------------
# Tiled inference guard
# ---------------------------------------------------------------------------


def test_yolonas_val_preprocessor_rejects_rect():
    from libreyolo.validation.preprocessors import YOLONASValPreprocessor

    pre = YOLONASValPreprocessor((64, 320))
    img = np.zeros((100, 500, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="rectangular"):
        pre(img, np.zeros((0, 5), dtype=np.float32), (64, 320))


# ---------------------------------------------------------------------------
# Checkpoint reload (#899)
# ---------------------------------------------------------------------------


def _family_checkpoint(tmp_path, family, **imgsz_meta):
    """Write a schema-v1 checkpoint for a tiny fresh model of ``family``."""
    from libreyolo.utils.serialization import wrap_libreyolo_checkpoint

    if family == "yolo9":
        from libreyolo.models.yolo9.model import LibreYOLO9 as cls

        size = "t"
    else:
        from libreyolo.models.yolox.model import LibreYOLOX as cls

        size = "n"
    fresh = cls(None, size=size, nb_classes=2, device="cpu")
    ckpt = wrap_libreyolo_checkpoint(
        fresh.model.state_dict(),
        model_family=family,
        size=size,
        task="detect",
        nc=2,
        names={0: "a", 1: "b"},
        **imgsz_meta,
    )
    path = tmp_path / f"{family}.pt"
    torch.save(ckpt, path)
    return cls, size, path


@pytest.mark.parametrize("family", ["yolo9", "yolox"])
def test_rect_checkpoint_restores_input_size(tmp_path, family):
    cls, size, path = _family_checkpoint(
        tmp_path, family, imgsz=320, imgsz_h=192, imgsz_w=320
    )
    model = cls(str(path), size=size, nb_classes=2, device="cpu")

    assert model.input_size == (192, 320)
    assert model._get_val_preprocessor().img_size == (192, 320)
    tensor, *_ = model._preprocess(np.zeros((1080, 1920, 3), dtype=np.uint8))
    assert tuple(tensor.shape[-2:]) == (192, 320)

    from libreyolo.utils.serialization import load_untrusted_torch_file

    saved = model.save(str(tmp_path / "resave.pt"))
    resaved = load_untrusted_torch_file(saved, map_location="cpu", context="test")
    assert (resaved["imgsz"], resaved["imgsz_h"], resaved["imgsz_w"]) == (320, 192, 320)
    assert cls(saved, size=size, nb_classes=2, device="cpu").input_size == (192, 320)


def test_square_checkpoint_keeps_native_input_size(tmp_path):
    cls, size, path = _family_checkpoint(tmp_path, "yolo9", imgsz=640)
    model = cls(str(path), size=size, nb_classes=2, device="cpu")

    assert model.input_size == 640
    assert model._get_val_preprocessor().img_size == (640, 640)


def test_half_rect_checkpoint_pair_is_rejected(tmp_path):
    cls, size, path = _family_checkpoint(tmp_path, "yolo9", imgsz=320, imgsz_h=192)
    with pytest.raises(RuntimeError, match="both imgsz_h and imgsz_w"):
        cls(str(path), size=size, nb_classes=2, device="cpu")


def test_square_checkpoint_after_rect_one_resets_input_size(tmp_path):
    cls, size, rect_path = _family_checkpoint(
        tmp_path, "yolox", imgsz=320, imgsz_h=192, imgsz_w=320
    )
    model = cls(str(rect_path), size=size, nb_classes=2, device="cpu")
    assert model.input_size == (192, 320)

    square_dir = tmp_path / "square"
    square_dir.mkdir()
    _cls, _size, square_path = _family_checkpoint(square_dir, "yolox", imgsz=640)
    model._load_weights(str(square_path))

    assert model.input_size == cls.INPUT_SIZES[size]


def test_square_family_default_pair_is_not_an_override(tmp_path):
    cls, size, path = _family_checkpoint(
        tmp_path, "yolo9", imgsz=640, imgsz_h=640, imgsz_w=640
    )
    model = cls(str(path), size=size, nb_classes=2, device="cpu")

    assert model.input_size == 640
    assert not model._input_size_from_checkpoint


def test_trained_input_size_matches_what_a_reload_gives():
    from libreyolo.models.yolox.model import LibreYOLOX

    model = LibreYOLOX(None, size="n", nb_classes=2, device="cpu")
    native = LibreYOLOX.INPUT_SIZES["n"]

    model._adopt_trained_input_size(640)  # square run on a fresh model: no-op
    assert model.input_size == native
    model._adopt_trained_input_size((192, 320))  # rectangular run
    assert model.input_size == (192, 320)
    model._adopt_trained_input_size("640")  # later square run drops it
    assert model.input_size == native


def test_natively_rect_family_ignores_trained_size_sync():
    from libreyolo.models.hrnet.model import LibreHRNet

    model = LibreHRNet(None, size="w32", device="cpu")
    native = model.input_size
    model._adopt_trained_input_size((64, 128))

    assert model.input_size == native


def test_yolox_postprocess_accepts_rect_size_at_unit_ratio():
    # The validator never passes ``ratio``; YOLOX recomputes it from the size.
    from libreyolo.models.yolox.model import LibreYOLOX

    model = LibreYOLOX(None, size="n", nb_classes=2, device="cpu")
    model.input_size = (192, 320)
    model.model.eval()
    result = model.predict(np.zeros((192, 320, 3), dtype=np.uint8), conf=0.99)

    assert result.orig_shape == (192, 320)


def test_tta_postprocess_uses_the_preprocessed_canvas(monkeypatch):
    from libreyolo.models.yolo9.model import LibreYOLO9

    model = LibreYOLO9(None, size="t", nb_classes=2, device="cpu")
    model.model.eval()
    seen = []
    original = model._postprocess

    def spy(*args, **kwargs):
        seen.append(kwargs.get("input_size"))
        return original(*args, **kwargs)

    monkeypatch.setattr(model, "_postprocess", spy)
    model.predict(np.zeros((120, 200, 3), dtype=np.uint8), imgsz=320, augment=True)

    assert seen and set(seen) == {320}


def test_yolo9_e2e_postprocess_defaults_to_model_input_size(monkeypatch):
    import libreyolo.models.yolo9_e2e.model as e2e_module

    model = e2e_module.LibreYOLO9E2E(None, size="t", nb_classes=2, device="cpu")
    model.input_size = (192, 320)
    seen = {}
    monkeypatch.setattr(
        e2e_module,
        "postprocess",
        lambda output, **kwargs: seen.update(kwargs) or {},
    )
    model._postprocess(None, 0.25, 0.45, (320, 192))

    assert seen["input_size"] == (192, 320)


def test_tiling_uses_long_side_for_restored_rect_size(tmp_path, monkeypatch):
    cls, size, path = _family_checkpoint(
        tmp_path, "yolo9", imgsz=320, imgsz_h=192, imgsz_w=320
    )
    model = cls(str(path), size=size, nb_classes=2, device="cpu")
    image = np.zeros((700, 900, 3), dtype=np.uint8)
    canvases = []
    original = model._preprocess

    def spy(*args, input_size=None, **kwargs):
        canvases.append(input_size)
        return original(*args, input_size=input_size, **kwargs)

    monkeypatch.setattr(model, "_preprocess", spy)

    assert model.predict(image, tiling=True, conf=0.99).orig_shape == (700, 900)
    # Each 320x320 tile is inferred at 320, not shrunk into the 192x320 canvas.
    assert canvases and set(canvases) == {320}
    with pytest.raises(ValueError, match="square imgsz"):
        model.predict(image, tiling=True, imgsz=(192, 320))


def test_tiling_small_image_uses_the_tile_canvas(tmp_path, monkeypatch):
    cls, size, path = _family_checkpoint(
        tmp_path, "yolo9", imgsz=320, imgsz_h=192, imgsz_w=320
    )
    model = cls(str(path), size=size, nb_classes=2, device="cpu")
    canvases = []
    original = model._preprocess

    def spy(*args, input_size=None, **kwargs):
        canvases.append(input_size)
        return original(*args, input_size=input_size, **kwargs)

    monkeypatch.setattr(model, "_preprocess", spy)
    model.predict(np.zeros((256, 256, 3), dtype=np.uint8), tiling=True, conf=0.99)

    assert canvases == [320]


@pytest.mark.parametrize("family", ["yolox", "rtmdet"])
def test_letterbox_gt_rescale_accepts_rect_size(family):
    # Validation plots and visualize=True invert the letterbox per image; the
    # shared default used to divide the (h, w) pair by an int (#899).
    from libreyolo.validation.preprocessors import (
        RTMDetValPreprocessor,
        YOLOXValPreprocessor,
    )

    cls = YOLOXValPreprocessor if family == "yolox" else RTMDetValPreprocessor
    r, off_x, off_y = cls(img_size=(192, 320)).letterbox_scale(1080, 1920, (192, 320))

    assert r == pytest.approx(min(192 / 1080, 320 / 1920))
    assert (off_x, off_y) == (0.0, 0.0)
