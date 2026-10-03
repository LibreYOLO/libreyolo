"""Class-agnostic NMS, ``agnostic_nms=True`` (#928).

It is one shared step applied to a family's finished detections, so these
tests drive it through every predict path and the detection validator with a
stub model: overlapping boxes of different classes collapse to the
highest-scoring one, nothing changes when the option is off, and the option
is never accepted and ignored.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from libreyolo.backends import base as backend_base
from libreyolo.models.base.inference import InferenceRunner
from libreyolo.models.base.model import BaseModel
from libreyolo.ops.agnostic_nms import (
    agnostic_nms_detections,
    agnostic_nms_keep,
    agnostic_rotated_nms_keep,
    top_detections,
)
from libreyolo.utils.image_loader import ImageLoader
from libreyolo.validation.config import ValidationConfig

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# The shared op
# ---------------------------------------------------------------------------


def _boxes(*rows):
    return torch.tensor(rows, dtype=torch.float32)


class TestKeep:
    def test_overlapping_boxes_keep_the_highest_score(self):
        boxes = _boxes([0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60])
        keep = agnostic_nms_keep(boxes, torch.tensor([0.5, 0.9, 0.3]), 0.5)
        assert keep.tolist() == [1, 2]

    def test_input_order_is_preserved(self):
        boxes = _boxes([50, 50, 60, 60], [0, 0, 10, 10], [80, 80, 90, 90])
        keep = agnostic_nms_keep(boxes, torch.tensor([0.1, 0.9, 0.5]), 0.5)
        assert keep.tolist() == [0, 1, 2]

    def test_threshold_is_exclusive_like_torchvision(self):
        # IoU of these two boxes is exactly 0.5.
        boxes = _boxes([0, 0, 10, 10], [0, 0, 10, 5])
        scores = torch.tensor([0.9, 0.8])
        assert agnostic_nms_keep(boxes, scores, 0.5).tolist() == [0, 1]
        assert agnostic_nms_keep(boxes, scores, 0.49).tolist() == [0]

    def test_identical_boxes_collapse_to_one(self):
        """The same box reported under two classes, as a set predictor can."""
        boxes = _boxes([5, 5, 25, 25], [5, 5, 25, 25])
        assert agnostic_nms_keep(boxes, torch.tensor([0.6, 0.7]), 0.99).tolist() == [1]

    @pytest.mark.parametrize("count", [0, 1])
    def test_fewer_than_two_boxes_are_untouched(self, count):
        boxes = torch.zeros((count, 4))
        keep = agnostic_nms_keep(boxes, torch.zeros(count), 0.5)
        assert keep.tolist() == list(range(count))
        assert keep.dtype == torch.long

    def test_non_finite_rows_pass_through(self):
        boxes = _boxes([0, 0, 10, 10], [float("nan"), 0, 1, 1], [0, 0, 10, 10])
        keep = agnostic_nms_keep(boxes, torch.tensor([0.9, 0.5, float("inf")]), 0.5)
        assert keep.tolist() == [0, 1, 2]

    def test_half_precision_and_int_inputs_are_accepted(self):
        boxes = _boxes([0, 0, 10, 10], [1, 1, 11, 11]).half()
        keep = agnostic_nms_keep(boxes, torch.tensor([0.5, 0.9]).half(), 0.5)
        assert keep.tolist() == [1]

    def test_rotated_boxes_use_rotated_iou(self):
        # Two thin boxes crossing at 90 degrees: their axis-aligned hulls are
        # identical-ish squares but the rectangles barely overlap.
        xywhr = torch.tensor(
            [[50.0, 50.0, 80.0, 6.0, 0.0], [50.0, 50.0, 80.0, 6.0, math.pi / 2]]
        )
        scores = torch.tensor([0.9, 0.8])
        assert agnostic_rotated_nms_keep(xywhr, scores, 0.3).tolist() == [0, 1]

        same = torch.tensor([[50.0, 50.0, 80.0, 6.0, 0.0], [51.0, 50.0, 80.0, 6.0, 0.0]])
        assert agnostic_rotated_nms_keep(same, scores, 0.3).tolist() == [0]


class TestTopDetections:
    def test_keeps_the_highest_scores_in_input_order(self):
        det = {
            "boxes": _boxes([0, 0, 1, 1], [2, 2, 3, 3], [4, 4, 5, 5]),
            "scores": torch.tensor([0.2, 0.9, 0.5]),
            "classes": torch.tensor([0, 1, 2]),
            "num_detections": 3,
        }
        out = top_detections(det, 2)
        assert out["classes"].tolist() == [1, 2]
        assert out["num_detections"] == 2
        assert top_detections(det, 3) is det
        assert top_detections(det, -1) is det


class TestDetections:
    def test_every_per_detection_entry_is_filtered_alike(self):
        det = {
            "boxes": _boxes([0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]),
            "scores": torch.tensor([0.5, 0.9, 0.3]),
            "classes": torch.tensor([0, 1, 2]),
            "masks": torch.arange(3).view(3, 1, 1).expand(3, 2, 2),
            "keypoints": torch.arange(3.0).view(3, 1, 1).expand(3, 4, 3),
            "num_detections": 3,
            "note": "kept",
        }
        out = agnostic_nms_detections(det, 0.5)

        assert out["classes"].tolist() == [1, 2]
        assert out["scores"].tolist() == pytest.approx([0.9, 0.3])
        assert out["masks"][:, 0, 0].tolist() == [1, 2]
        assert out["keypoints"][:, 0, 0].tolist() == [1.0, 2.0]
        assert out["num_detections"] == 2
        assert out["note"] == "kept"
        # The input is not modified.
        assert det["classes"].tolist() == [0, 1, 2]

    def test_nothing_to_suppress_returns_the_same_dict(self):
        det = {
            "boxes": _boxes([0, 0, 10, 10], [50, 50, 60, 60]),
            "scores": torch.tensor([0.5, 0.9]),
            "classes": torch.tensor([0, 1]),
        }
        assert agnostic_nms_detections(det, 0.5) is det


# ---------------------------------------------------------------------------
# predict(): the PyTorch runner
# ---------------------------------------------------------------------------

# Two classes on the same object, plus a separate object.
_DETECTIONS = {
    "boxes": [[2.0, 2.0, 12.0, 12.0], [3.0, 3.0, 13.0, 13.0], [20.0, 20.0, 28.0, 28.0]],
    "scores": [0.6, 0.9, 0.5],
    "classes": [0, 1, 2],
}


class _StubModel:
    """A BaseModel as InferenceRunner sees it.

    ``_postprocess`` has no ``**kwargs`` on purpose: if ``agnostic_nms`` ever
    leaked into a family postprocess, every test here would raise TypeError.
    """

    task = "detect"
    TTA_ENABLED = False
    SUPPORTS_BATCHED_PREDICT = True
    size = "n"
    names = {0: "a", 1: "b", 2: "c"}
    device = torch.device("cpu")
    model = SimpleNamespace(training=False)

    def _get_input_size(self):
        return 32

    def _get_model_name(self):
        return "stub"

    def _preprocess(self, image, color_format="auto", input_size=None):
        pil = ImageLoader.load(image, color_format=color_format)
        return torch.zeros(1, 3, 32, 32), pil, pil.size, 1.0

    def _forward(self, tensor):
        return tensor

    def _postprocess(
        self,
        output,
        conf,
        iou,
        original_size,
        max_det=300,
        ratio=1.0,
        classes=None,
        input_size=None,
    ):
        return {**_DETECTIONS, "num_detections": len(_DETECTIONS["boxes"])}


def _image(size=32):
    return np.zeros((size, size, 3), dtype=np.uint8)


def _classes(result):
    return sorted(int(c) for c in result.boxes.cls.tolist())


class TestRunner:
    def test_default_keeps_every_box(self):
        runner = InferenceRunner(_StubModel())
        assert _classes(runner(_image())) == [0, 1, 2]
        assert _classes(runner(_image(), agnostic_nms=False)) == [0, 1, 2]

    def test_single_image(self):
        result = InferenceRunner(_StubModel())(_image(), agnostic_nms=True)
        assert _classes(result) == [1, 2]
        assert result.boxes.conf.tolist() == pytest.approx([0.9, 0.5])

    def test_iou_sets_the_suppression_threshold(self):
        runner = InferenceRunner(_StubModel())
        # The two overlapping boxes have IoU 0.68.
        assert _classes(runner(_image(), agnostic_nms=True, iou=0.7)) == [0, 1, 2]
        assert _classes(runner(_image(), agnostic_nms=True, iou=0.6)) == [1, 2]

    def test_list_source(self):
        results = InferenceRunner(_StubModel())([_image(), _image()], agnostic_nms=True)
        assert [_classes(r) for r in results] == [[1, 2], [1, 2]]

    def test_batched_forward(self):
        results = InferenceRunner(_StubModel())(
            [_image(), _image(), _image()], batch=2, agnostic_nms=True
        )
        assert [_classes(r) for r in results] == [[1, 2]] * 3

    def test_stream(self):
        runner = InferenceRunner(_StubModel())
        stream = runner([_image(), _image()], stream=True, agnostic_nms=True)
        # A later call with the option off must not affect the open stream.
        assert _classes(runner(_image())) == [0, 1, 2]
        assert [_classes(r) for r in stream] == [[1, 2], [1, 2]]

    def test_video_frames(self):
        predict_frame = InferenceRunner(_StubModel())._frame_predictor(
            "clip.mp4", agnostic_nms=True
        )
        from PIL import Image

        assert _classes(predict_frame(Image.fromarray(_image()))) == [1, 2]

    def test_tiled(self):
        runner = InferenceRunner(_StubModel())
        default = runner(_image(64), tiling=True)
        agnostic = runner(_image(64), tiling=True, agnostic_nms=True)

        assert default.tiled and agnostic.tiled
        assert 0 in _classes(default)
        assert 0 not in _classes(agnostic)
        assert len(agnostic) < len(default)

    def test_classes_filter_runs_first(self):
        """A box of a class the caller filtered out suppresses nothing."""
        result = InferenceRunner(_StubModel())(
            _image(), classes=[0, 2], agnostic_nms=True
        )
        assert _classes(result) == [0, 2]

    def test_suppressed_slots_are_refilled_up_to_max_det(self):
        """max_det is cut after suppression, from a wider candidate budget."""

        class _HonorsMaxDet(_StubModel):
            def _postprocess(self, output, conf, iou, original_size, max_det=300, **kw):
                order = np.argsort(_DETECTIONS["scores"])[::-1][:max_det]
                return {
                    "boxes": [_DETECTIONS["boxes"][i] for i in order],
                    "scores": [_DETECTIONS["scores"][i] for i in order],
                    "classes": [_DETECTIONS["classes"][i] for i in order],
                    "num_detections": len(order),
                }

        runner = InferenceRunner(_HonorsMaxDet())
        # Without the wider budget the two top boxes (0.9, 0.6) overlap and
        # only one would be left.
        assert _classes(runner(_image(), max_det=2, agnostic_nms=True)) == [1, 2]
        assert _classes(runner(_image(), max_det=1, agnostic_nms=True)) == [1]
        # Default behavior is the family's own cut.
        assert _classes(runner(_image(), max_det=2)) == [0, 1]

    def test_masks_and_keypoints_stay_aligned(self):
        class _WithPayloads(_StubModel):
            def _postprocess(self, *args, **kwargs):
                det = _StubModel._postprocess(self, *args, **kwargs)
                det["masks"] = torch.arange(3).view(3, 1, 1).expand(3, 32, 32) > 100
                det["keypoints"] = torch.arange(3.0).view(3, 1, 1).expand(3, 2, 3)
                return det

        result = InferenceRunner(_WithPayloads())(_image(), agnostic_nms=True)
        assert len(result.masks) == len(result.keypoints.data) == len(result) == 2
        assert result.keypoints.data[:, 0, 0].tolist() == [1.0, 2.0]

    def test_obb_uses_rotated_boxes(self):
        class _Obb(_StubModel):
            task = "obb"

            def _postprocess(self, *args, **kwargs):
                det = _StubModel._postprocess(self, *args, **kwargs)
                # Crossing thin rectangles: same hull, almost no overlap.
                det["boxes"] = [[10.0, 10.0, 90.0, 90.0]] * 2 + [[0.0, 0.0, 5.0, 5.0]]
                det["obb"] = torch.tensor(
                    [
                        [50.0, 50.0, 80.0, 6.0, 0.0, 0.6, 0.0],
                        [50.0, 50.0, 80.0, 6.0, math.pi / 2, 0.9, 1.0],
                        [2.5, 2.5, 5.0, 5.0, 0.0, 0.5, 2.0],
                    ]
                )
                return det

        result = InferenceRunner(_Obb())(_image(128), agnostic_nms=True)
        assert len(result) == 3 and len(result.obb) == 3

    @pytest.mark.parametrize("task", ["classify", "semantic", "depth", "embed"])
    def test_tasks_without_boxes_reject_it(self, task):
        class _Other(_StubModel):
            pass

        _Other.task = task
        with pytest.raises(ValueError, match="agnostic_nms=True is not supported"):
            InferenceRunner(_Other())(_image(), agnostic_nms=True)

    def test_off_is_accepted_for_any_task(self):
        class _Classify(_StubModel):
            task = "classify"

            def _postprocess(self, *args, **kwargs):
                return {"probs": torch.tensor([0.2, 0.8])}

        result = InferenceRunner(_Classify())(_image(), agnostic_nms=False)
        assert result.probs.top1 == 1


class TestMergeTTA:
    def _dets(self):
        det = {**_DETECTIONS, "num_detections": 3}
        return [(det, (32, 32), False, 1.0)]

    def _merge(self, **kwargs):
        model = SimpleNamespace(names={0: "a", 1: "b", 2: "c"})
        return BaseModel._merge_tta(
            model,
            aug_dets=self._dets(),
            iou_thres=0.5,
            image_path=None,
            original_size=(32, 32),
            **kwargs,
        )

    def test_default_merge_is_class_aware(self):
        assert _classes(self._merge()) == [0, 1, 2]

    def test_agnostic_merge_drops_the_lower_scoring_class(self):
        result = self._merge(agnostic_nms=True)
        assert _classes(result) == [1, 2]
        assert result.boxes.conf.tolist() == pytest.approx([0.9, 0.5])

    def test_agnostic_then_max_det(self):
        result = self._merge(agnostic_nms=True, max_det=1)
        assert _classes(result) == [1]

    def test_runner_forwards_the_option_to_tta(self):
        seen = {}

        class _Tta(_StubModel):
            TTA_ENABLED = True

            def _predict_augment(self, image, **kwargs):
                seen.update(kwargs)
                return InferenceRunner(_StubModel())(image)

        InferenceRunner(_Tta())(_image(), augment=True, agnostic_nms=True)
        assert seen["agnostic_nms"] is True


# ---------------------------------------------------------------------------
# val(): the detection validator
# ---------------------------------------------------------------------------


class _RecordingEvaluator:
    def __init__(self):
        self.classes = []

    def update(self, pred, image_id):
        self.classes.append(sorted(int(c) for c in pred["classes"].tolist()))


def _validator(**config):
    from libreyolo.validation.detection_validator import DetectionValidator

    v = DetectionValidator.__new__(DetectionValidator)
    v.config = ValidationConfig(data="x", **config)
    v.nc = 3
    v.seen = 0
    v.coco_evaluator = _RecordingEvaluator()
    return v


def _val_preds():
    return [
        {
            "boxes": torch.tensor(_DETECTIONS["boxes"]),
            "scores": torch.tensor(_DETECTIONS["scores"]),
            "classes": torch.tensor(_DETECTIONS["classes"]),
            "masks": torch.zeros(3, 4, 4),
        }
    ]


class TestValidator:
    def test_default_scores_every_box(self):
        v = _validator()
        v._update_metrics(_val_preds(), None, [(32, 32)], [1])
        assert v.coco_evaluator.classes == [[0, 1, 2]]

    def test_agnostic_scores_the_surviving_boxes(self):
        v = _validator(agnostic_nms=True, iou_thres=0.5)
        preds = _val_preds()
        v._update_metrics(preds, None, [(32, 32)], [1])

        assert v.coco_evaluator.classes == [[1, 2]]
        # The batch is filtered in place, masks included, so the mask
        # evaluator of the segmentation validator sees the same detections.
        assert preds[0]["classes"].tolist() == [1, 2]
        assert preds[0]["masks"].shape[0] == 2

    def test_iou_thres_sets_the_threshold(self):
        v = _validator(agnostic_nms=True, iou_thres=0.7)
        v._update_metrics(_val_preds(), None, [(32, 32)], [1])
        assert v.coco_evaluator.classes == [[0, 1, 2]]

    def test_classes_filter_runs_first(self):
        v = _validator(agnostic_nms=True, iou_thres=0.5, classes=[0, 2])
        v._update_metrics(_val_preds(), None, [(32, 32)], [1])
        assert v.coco_evaluator.classes == [[0, 2]]

    def test_postprocess_budget_is_widened_then_cut_to_max_det(self):
        from libreyolo.utils.predict_args import DEFAULT_MAX_DET

        assert _validator(max_det=2)._postprocess_max_det() == 2
        v = _validator(agnostic_nms=True, iou_thres=0.5, max_det=2)
        assert v._postprocess_max_det() == DEFAULT_MAX_DET

        v._update_metrics(_val_preds(), None, [(32, 32)], [1])
        assert v.coco_evaluator.classes == [[1, 2]]

        one = _validator(agnostic_nms=True, iou_thres=0.5, max_det=1)
        preds = _val_preds()
        one._update_metrics(preds, None, [(32, 32)], [1])
        assert one.coco_evaluator.classes == [[1]]
        assert preds[0]["masks"].shape[0] == 1

    def test_config_round_trips_the_option(self, tmp_path):
        config = ValidationConfig(data="x", agnostic_nms=True)
        config.to_yaml(tmp_path / "config.yaml")
        assert ValidationConfig.from_yaml(tmp_path / "config.yaml").agnostic_nms is True
        assert ValidationConfig(data="x").agnostic_nms is False

    def test_validators_that_do_not_apply_it_reject_it(self):
        from libreyolo.validation.classify_validator import ClassifyValidator
        from libreyolo.validation.detection_validator import (
            DetectionValidator,
            SegmentationValidator,
        )
        from libreyolo.validation.obb_validator import OBBValidator
        from libreyolo.validation.pose_validator import PoseValidator

        assert DetectionValidator.supports_agnostic_nms
        assert SegmentationValidator.supports_agnostic_nms
        for cls in (ClassifyValidator, OBBValidator, PoseValidator):
            assert not cls.supports_agnostic_nms
            with pytest.raises(ValueError, match="agnostic_nms=True is not supported"):
                cls(model=SimpleNamespace(), config=ValidationConfig(data="x", agnostic_nms=True))


# ---------------------------------------------------------------------------
# Exported backends
# ---------------------------------------------------------------------------


def _backend(task="detect", family="rfdetr"):
    from libreyolo.backends.onnx import OnnxBackend

    backend = OnnxBackend.__new__(OnnxBackend)
    backend.task = task
    backend.model_family = family
    backend.names = {0: "a", 1: "b", 2: "c"}
    return backend


def _build(backend, **kwargs):
    return backend._build_result(
        np.array(_DETECTIONS["boxes"], dtype=np.float32),
        np.array(_DETECTIONS["scores"], dtype=np.float32),
        np.array(_DETECTIONS["classes"]),
        orig_shape=(32, 32),
        image_path=None,
        iou=0.5,
        classes=kwargs.pop("classes", None),
        max_det=kwargs.pop("max_det", 300),
        **kwargs,
    )


class TestBackend:
    def test_default_build_keeps_every_box(self):
        assert _classes(_build(_backend())) == [0, 1, 2]

    def test_agnostic_build_matches_native(self):
        token = backend_base._AGNOSTIC_NMS.set(True)
        try:
            result = _build(_backend())
        finally:
            backend_base._AGNOSTIC_NMS.reset(token)
        assert _classes(result) == [1, 2]
        assert result.boxes.conf.tolist() == pytest.approx([0.9, 0.5])

    def test_agnostic_build_filters_payloads_and_respects_classes(self):
        token = backend_base._AGNOSTIC_NMS.set(True)
        try:
            masks = np.zeros((3, 32, 32), dtype=bool)
            keypoints = np.arange(3, dtype=np.float32).reshape(3, 1, 1).repeat(3, axis=2)
            result = _build(_backend(), masks=masks, keypoints=keypoints)
            filtered = _build(_backend(), classes=[0, 2])
        finally:
            backend_base._AGNOSTIC_NMS.reset(token)
        assert len(result.masks) == 2
        assert result.keypoints.data[:, 0, 0].tolist() == [1.0, 2.0]
        assert _classes(filtered) == [0, 2]

    def test_numpy_keep_agrees_with_the_torch_op(self):
        rng = np.random.default_rng(928)
        for _ in range(200):
            count = int(rng.integers(2, 40))
            xy = rng.uniform(0, 100, (count, 2))
            boxes = np.concatenate([xy, xy + rng.uniform(1, 60, (count, 2))], axis=1)
            boxes = boxes.astype(np.float32)
            scores = rng.uniform(0, 1, count).astype(np.float32)
            iou = float(rng.uniform(0.1, 0.9))
            assert (
                backend_base._agnostic_nms_keep_numpy(boxes, scores, iou).tolist()
                == agnostic_nms_keep(
                    torch.from_numpy(boxes), torch.from_numpy(scores), iou
                ).tolist()
            )

    def test_call_scopes_the_option_to_that_call(self, monkeypatch):
        backend = _backend()
        seen = []

        def fake_single(image, **kwargs):
            seen.append(backend_base._AGNOSTIC_NMS.get())
            return "result"

        monkeypatch.setattr(backend, "_predict_single", fake_single, raising=False)
        backend.device = "cpu"

        assert backend(_image(), agnostic_nms=True) == "result"
        assert backend(_image()) == "result"
        assert seen == [True, False]
        assert backend_base._AGNOSTIC_NMS.get() is False

    def test_lazy_results_keep_the_option_until_consumed(self, monkeypatch):
        backend = _backend()
        backend.device = "cpu"

        def fake_stream(images, **kwargs):
            for _ in images:
                yield backend_base._AGNOSTIC_NMS.get()

        monkeypatch.setattr(backend, "_stream_in_batches", fake_stream, raising=False)

        stream = backend([_image(), _image()], stream=True, agnostic_nms=True)
        assert backend_base._AGNOSTIC_NMS.get() is False
        assert list(stream) == [True, True]
        assert backend_base._AGNOSTIC_NMS.get() is False

    def test_tasks_without_boxes_reject_it(self):
        backend = _backend(task="classify", family="resnet")
        backend.device = "cpu"
        with pytest.raises(ValueError, match="agnostic_nms=True is not supported"):
            backend(_image(), agnostic_nms=True)
