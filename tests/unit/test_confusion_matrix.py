"""Confusion matrix on the validation results (#928).

``val()`` returns ``results.confusion_matrix`` for detect, segment and
classify. The matrix is indexed ``[predicted, true]``, as in the ecosystem.
"""

from __future__ import annotations

import json
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from libreyolo.validation import ConfusionMatrix
from libreyolo.validation.base import (
    ClassifyMetrics,
    ValidationMetrics,
    with_image_metrics,
)
from libreyolo.validation.config import ValidationConfig

pytestmark = pytest.mark.unit

NAMES = ["cat", "dog", "fox"]
BACKGROUND = 3


def _box(x, y, size=10.0):
    return [x, y, x + size, y + size]


def _detect_fixture(**kwargs):
    """cat labelled dog, dog found, fox missed, one false alarm, one low score."""
    cm = ConfusionMatrix(nc=3, names=NAMES, **kwargs)
    cm.process_image(
        np.array([_box(0, 0), _box(20, 20), _box(80, 80), _box(50, 50)]),
        np.array([1, 1, 0, 2]),
        np.array([0.9, 0.8, 0.7, 0.1]),
        np.array([_box(0, 0), _box(20, 20), _box(50, 50)]),
        np.array([0, 1, 2]),
    )
    return cm


class TestDetection:
    def test_matrix_is_indexed_predicted_then_true(self):
        cm = _detect_fixture()

        expected = np.zeros((4, 4), dtype=np.int64)
        expected[1, 0] = 1  # predicted dog, truly cat
        expected[1, 1] = 1  # predicted dog, truly dog
        expected[BACKGROUND, 2] = 1  # fox missed
        expected[0, BACKGROUND] = 1  # cat predicted on nothing
        assert cm.matrix.tolist() == expected.tolist()

    def test_a_wrong_class_is_one_cell_not_a_miss_plus_a_false_alarm(self):
        cm = ConfusionMatrix(nc=2)
        cm.process_image(
            np.array([_box(0, 0)]), np.array([1]), np.array([0.9]),
            np.array([_box(0, 0)]), np.array([0]),
        )
        assert int(cm.matrix.sum()) == 1
        assert cm.matrix[1, 0] == 1

    def test_predictions_below_the_confidence_threshold_are_not_counted(self):
        low = _detect_fixture(conf_thres=0.05)
        # The 0.1 fox prediction now pairs with the fox ground truth.
        assert low.matrix[2, 2] == 1
        assert low.matrix[BACKGROUND, 2] == 0

        high = _detect_fixture(conf_thres=0.85)
        # Only the 0.9 prediction is left.
        assert int(high.matrix[:3].sum()) == 1
        assert high.matrix[BACKGROUND, :3].tolist() == [0, 1, 1]

    def test_pairing_needs_the_iou_threshold(self):
        cm = ConfusionMatrix(nc=1, iou_thres=0.5)
        cm.process_image(
            np.array([_box(6, 0)]), np.array([0]), np.array([0.9]),
            np.array([_box(0, 0)]), np.array([0]),
        )
        # IoU 0.25: a miss and a false alarm.
        assert cm.matrix.tolist() == [[0, 1], [1, 0]]

    def test_pairing_is_one_to_one_highest_iou_first(self):
        cm = ConfusionMatrix(nc=2)
        cm.process_image(
            np.array([_box(1, 0), _box(0, 0)]),
            np.array([1, 0]),
            np.array([0.9, 0.6]),
            np.array([_box(0, 0)]),
            np.array([0]),
        )
        # The exact box wins the ground truth; the other is a false alarm.
        assert cm.matrix[0, 0] == 1
        assert cm.matrix[1, 2] == 1
        assert int(cm.matrix.sum()) == 2

    def test_empty_images_and_empty_sides(self):
        cm = ConfusionMatrix(nc=2)
        empty_boxes = np.zeros((0, 4))
        empty = np.zeros(0)
        cm.process_image(empty_boxes, empty, empty, empty_boxes, empty)
        assert int(cm.matrix.sum()) == 0

        cm.process_image(empty_boxes, empty, empty, np.array([_box(0, 0)]), np.array([1]))
        cm.process_image(np.array([_box(0, 0)]), np.array([0]), np.array([0.9]), empty_boxes, empty)
        assert cm.matrix[2, 1] == 1
        assert cm.matrix[0, 2] == 1

    def test_classes_outside_the_matrix_are_ignored(self):
        cm = ConfusionMatrix(nc=2)
        cm.process_image(
            np.array([_box(0, 0), _box(30, 30)]),
            np.array([7, 0]),
            np.array([0.9, 0.9]),
            np.array([_box(30, 30), _box(60, 60)]),
            np.array([0, -1]),
        )
        assert cm.matrix[0, 0] == 1
        assert int(cm.matrix.sum()) == 1

    def test_counts_accumulate_over_images(self):
        cm = _detect_fixture()
        before = cm.matrix.copy()
        cm.process_image(
            np.array([_box(0, 0)]), np.array([0]), np.array([0.9]),
            np.array([_box(0, 0)]), np.array([0]),
        )
        assert cm.matrix[0, 0] == before[0, 0] + 1

    def test_tp_fp_excludes_background(self):
        tp, fp = _detect_fixture().tp_fp()
        assert tp.tolist() == [0, 1, 0]
        assert fp.tolist() == [1, 1, 0]

    def test_class_accuracy_is_over_found_objects(self):
        # cat was found but called dog; dog was found and right; fox was
        # missed, so it has no found instance and is left out.
        assert _detect_fixture().class_accuracy() == {"cat": 0.0, "dog": 1.0}


class TestExport:
    def test_summary_has_one_row_per_predicted_class(self):
        rows = _detect_fixture().summary()

        assert [row["Predicted"] for row in rows] == [*NAMES, "background"]
        assert list(rows[0]) == ["Predicted", *NAMES, "background"]
        assert rows[1] == {"Predicted": "dog", "cat": 1, "dog": 1, "fox": 0, "background": 0}
        assert all(isinstance(v, int) for row in rows for k, v in row.items() if k != "Predicted")

    def test_normalized_summary_gives_shares_of_each_true_class(self):
        cm = ConfusionMatrix(nc=2, names=["a", "b"], task="classify")
        cm.process_cls_preds([0, 0, 1], [0, 0, 0])
        rows = cm.summary(normalize=True, decimals=3)

        assert rows[0]["a"] == pytest.approx(0.667)
        assert rows[1]["a"] == pytest.approx(0.333)
        # A true class that never occurs has an all-zero column, not NaN.
        assert rows[0]["b"] == 0.0 and rows[1]["b"] == 0.0

    def test_normalized_columns_sum_to_one(self):
        normalized = _detect_fixture().normalized()
        assert normalized.sum(axis=0).tolist() == pytest.approx([1.0, 1.0, 1.0, 1.0])

    def test_json_and_csv_carry_the_summary(self):
        cm = _detect_fixture()

        assert json.loads(cm.to_json()) == cm.summary()
        lines = cm.to_csv().strip().split("\n")
        assert lines[0] == "Predicted,cat,dog,fox,background"
        assert lines[2] == "dog,1,1,0,0"

    def test_to_df_returns_a_polars_frame(self):
        pl = pytest.importorskip("polars")
        frame = _detect_fixture().to_df()

        assert isinstance(frame, pl.DataFrame)
        assert frame.columns == ["Predicted", *NAMES, "background"]
        assert frame["dog"].to_list() == [0, 1, 0, 0]

    def test_to_df_without_polars_says_how_to_get_it(self, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def no_polars(name, *args, **kwargs):
            if name == "polars":
                raise ImportError("no polars")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", no_polars)
        with pytest.raises(ImportError, match="pip install polars"):
            _detect_fixture().to_df()

    def test_duplicate_class_names_get_unique_labels(self):
        cm = ConfusionMatrix(nc=2, names=["crane", "crane"], task="classify")
        cm.process_cls_preds([0, 1, 1], [0, 1, 0])

        assert cm.labels == ["0: crane", "1: crane"]
        assert len(cm.summary()[0]) == 3
        assert cm.class_accuracy() == {"0: crane": 0.5, "1: crane": 1.0}

    @pytest.mark.parametrize("reserved", ["Predicted", "background"])
    def test_a_class_named_like_a_reserved_key_does_not_corrupt_rows(self, reserved):
        """The row-name key and the background label stay unambiguous."""
        cm = ConfusionMatrix(nc=2, names=["cat", reserved])
        cm.process_image(
            np.array([_box(0, 0)]), np.array([1]), np.array([0.9]),
            np.array([_box(0, 0)]), np.array([0]),
        )
        rows = cm.summary()

        assert cm.labels == ["0: cat", f"1: {reserved}", "background"]
        assert [row["Predicted"] for row in rows] == cm.labels
        assert all(len(row) == 4 for row in rows)
        assert rows[1]["0: cat"] == 1
        header = cm.to_csv().split("\n")[0].split(",")
        assert len(header) == len(set(header)) == 4
        assert json.loads(cm.to_json()) == rows

    def test_names_accept_a_dict_and_fill_gaps(self):
        cm = ConfusionMatrix(nc=3, names={0: "a", 2: "c"}, task="classify")
        assert cm.labels == ["a", "1", "c"]

    def test_plot_writes_the_ecosystem_file_names(self, tmp_path):
        pytest.importorskip("matplotlib")
        cm = _detect_fixture()

        assert cm.plot(normalize=False, save_dir=tmp_path).name == "confusion_matrix.png"
        normalized = cm.plot(normalize=True, save_dir=tmp_path)
        assert normalized.name == "confusion_matrix_normalized.png"
        assert (tmp_path / "confusion_matrix.png").stat().st_size > 0
        assert normalized.stat().st_size > 0

    def test_classify_plot_has_no_background(self, tmp_path):
        pytest.importorskip("matplotlib")
        cm = ConfusionMatrix(nc=3, names=NAMES, task="classify")
        cm.process_cls_preds([0, 1, 2], [0, 1, 1])
        assert cm.plot(save_dir=tmp_path).exists()


class TestClassification:
    def test_counts_top1_against_label(self):
        cm = ConfusionMatrix(nc=3, task="classify")
        cm.process_cls_preds(torch.tensor([0, 1, 1, 1]), torch.tensor([0, 0, 1, 1]))
        cm.process_cls_preds([torch.tensor([2, 0])], [torch.tensor([2, 2])])

        assert cm.matrix.shape == (3, 3)
        assert cm.matrix.tolist() == [[1, 0, 1], [1, 2, 0], [0, 0, 1]]
        assert cm.class_accuracy() == {"0": 0.5, "1": 1.0, "2": 0.5}

    def test_a_repeated_pair_counts_every_time(self):
        cm = ConfusionMatrix(nc=2, task="classify")
        cm.process_cls_preds([0, 0, 0], [0, 0, 0])
        cm.process_cls_preds([0], [0])
        assert cm.matrix[0, 0] == 4

    def test_storage_is_sparse_for_wide_heads(self):
        """No dense nc x nc array is held; ImageNet-21k would be gigabytes."""
        cm = ConfusionMatrix(nc=21_841, task="classify")
        cm.process_cls_preds([5, 9], [5, 7])

        assert not any(isinstance(v, np.ndarray) for v in vars(cm).values())
        tp, fp = cm.tp_fp()
        assert int(tp.sum()) == 1 and int(fp.sum()) == 1
        assert cm.class_accuracy() == {"5": 1.0, "7": 0.0}
        assert "total=2" in repr(cm)

    def test_imagenet_1k_is_within_the_dense_limit(self):
        cm = ConfusionMatrix(nc=1000, task="classify")
        cm.process_cls_preds([3], [3])
        assert cm.matrix.shape == (1000, 1000)

    @pytest.mark.parametrize("nc", [2_001, 21_841])
    def test_dense_accessors_refuse_heads_too_wide_to_build(self, nc):
        cm = ConfusionMatrix(nc=nc, task="classify")
        cm.process_cls_preds([5, 9, 9], [5, 7, 7])

        for dense in (lambda: cm.matrix, cm.normalized, cm.summary, cm.to_csv, cm.to_json):
            with pytest.raises(ValueError, match="nonzero"):
                dense()
        predicted, true, counts = cm.nonzero()
        assert (predicted.tolist(), true.tolist(), counts.tolist()) == ([5, 9], [5, 7], [1, 2])

    def test_nonzero_lists_the_counted_cells(self):
        predicted, true, counts = _detect_fixture().nonzero()
        assert list(zip(predicted.tolist(), true.tolist(), counts.tolist())) == [
            (0, BACKGROUND, 1),
            (1, 0, 1),
            (1, 1, 1),
            (BACKGROUND, 2, 1),
        ]

        cm = ConfusionMatrix(nc=3, task="classify")
        cm.process_cls_preds([2, 0, 0], [1, 0, 0])
        predicted, true, counts = cm.nonzero()
        assert list(zip(predicted.tolist(), true.tolist(), counts.tolist())) == [
            (0, 0, 2),
            (2, 1, 1),
        ]

    def test_misaligned_inputs_fail(self):
        cm = ConfusionMatrix(nc=2, task="classify")
        with pytest.raises(ValueError, match="same number"):
            cm.process_cls_preds([0, 1], [0])

    def test_wrong_task_methods_fail(self):
        with pytest.raises(ValueError, match="classification"):
            ConfusionMatrix(nc=2).process_cls_preds([0], [0])
        with pytest.raises(ValueError, match="detection"):
            ConfusionMatrix(nc=2, task="classify").process_image(
                np.zeros((0, 4)), np.zeros(0), np.zeros(0), np.zeros((0, 4)), np.zeros(0)
            )
        with pytest.raises(ValueError, match="task"):
            ConfusionMatrix(nc=2, task="pose")


def test_plotter_still_exports_the_class():
    """ConfusionMatrix used to live in val_plotter; that import keeps working."""
    from libreyolo.validation.val_plotter import ConfusionMatrix as legacy

    assert legacy is ConfusionMatrix


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------


def _detection_validator(**config):
    from libreyolo.validation.detection_validator import DetectionValidator

    v = DetectionValidator.__new__(DetectionValidator)
    v.config = ValidationConfig(data="x", **config)
    v.nc = 2
    v.class_names = ["a", "b"]
    v.seen = 0
    v.image_metrics = {}
    v.dataloader = SimpleNamespace(dataset=object())
    v._resolve_img_path = lambda dataset, idx, img_id: f"img{idx}.jpg"
    v.confusion_matrix = v._new_confusion_matrix()
    return v


def _score_one_image(v, classes, scores):
    """One 100x100 image: a class-1 object at (20..40) and predictions on it."""
    preds = [
        {
            "boxes": torch.tensor([[20.0, 20.0, 40.0, 40.0]] * len(classes)),
            "scores": torch.tensor(scores),
            "classes": torch.tensor(classes),
        }
    ]
    targets = torch.tensor([[[1.0, 0.3, 0.3, 0.2, 0.2]]])
    v._score_images(preds, targets, [(100, 100)], [0])
    return v.confusion_matrix


class TestDetectionValidator:
    def test_matrix_is_built_without_plots(self):
        v = _detection_validator()
        assert v.config.save_plots is False

        cm = _score_one_image(v, [1], [0.9])

        assert cm.names == {0: "a", 1: "b"}
        assert cm.matrix[1, 1] == 1
        assert int(cm.matrix.sum()) == 1

    def test_matrix_uses_the_visualize_thresholds(self):
        assert _detection_validator().confusion_matrix.conf_thres == 0.25
        assert _detection_validator().confusion_matrix.iou_thres == 0.5
        assert _detection_validator(conf_thres=0.6).confusion_matrix.conf_thres == 0.6
        assert _detection_validator(conf_thres=0.01).confusion_matrix.conf_thres == 0.25

    def test_matrix_and_image_metrics_agree(self):
        v = _detection_validator()
        cm = _score_one_image(v, [0], [0.9])  # found, wrong class

        assert cm.matrix[0, 1] == 1
        entry = v.image_metrics["img0.jpg"]
        assert (entry["tp"], entry["fp"], entry["fn"]) == (0, 1, 1)

    def test_classes_filter_applies_to_the_ground_truth(self):
        v = _detection_validator(classes=[0])
        cm = _score_one_image(v, [0], [0.9])

        # The class-1 object is not part of this run: no miss is recorded.
        assert cm.matrix[2, 1] == 0
        assert cm.matrix[0, 2] == 1

    def test_validator_built_without_init_is_safe(self):
        from libreyolo.validation.detection_validator import DetectionValidator

        v = DetectionValidator.__new__(DetectionValidator)
        v.config = ValidationConfig(data="x")
        v.nc = 2
        v.seen = 0
        v.dataloader = SimpleNamespace(dataset=object())
        v._resolve_img_path = lambda dataset, idx, img_id: None
        _score_one_image(v, [1], [0.9])
        assert v.confusion_matrix is None


def _classify_validator():
    from libreyolo.validation.classify_validator import ClassifyValidator

    validator = object.__new__(ClassifyValidator)
    validator.config = SimpleNamespace(visualize=False)
    validator.loss_adapter = None
    validator._autocast_context = nullcontext
    validator._init_metrics()
    return validator


def _one_hot_logits(preds, nc):
    out = torch.full((len(preds), nc), -5.0)
    for row, cls in enumerate(preds):
        out[row, cls] = 5.0
    return out


class TestClassifyValidator:
    def test_matrix_accumulates_over_batches(self):
        validator = _classify_validator()
        validator._update_metrics(
            _one_hot_logits([0, 1, 1, 1], 4), torch.tensor([0, 0, 1, 1]), None
        )
        validator._update_metrics(_one_hot_logits([2, 0], 4), torch.tensor([2, 2]), None)

        cm = validator.confusion_matrix
        assert cm.task == "classify" and cm.nc == 4
        assert cm.matrix.tolist() == [
            [1, 0, 1, 0],
            [1, 2, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 0],
        ]
        # The diagonal is the top-1 hit count.
        metrics = validator._compute_metrics()
        assert np.trace(cm.matrix) / cm.matrix.sum() == pytest.approx(
            metrics["metrics/accuracy_top1"]
        )

    def test_matrix_resets_between_runs(self):
        validator = _classify_validator()
        validator._update_metrics(_one_hot_logits([0], 2), torch.tensor([0]), None)
        validator._init_metrics()
        assert validator.confusion_matrix is None

    def test_plots_are_saved_with_save_plots(self, tmp_path):
        pytest.importorskip("matplotlib")
        validator = _classify_validator()
        validator.save_dir = tmp_path
        validator._update_metrics(_one_hot_logits([0, 1], 2), torch.tensor([0, 0]), None)

        validator._save_plots({})

        assert (tmp_path / "plots" / "confusion_matrix.png").exists()
        assert (tmp_path / "plots" / "confusion_matrix_normalized.png").exists()

    def test_plots_are_skipped_for_very_wide_heads(self, tmp_path):
        validator = _classify_validator()
        validator.save_dir = tmp_path
        validator.confusion_matrix = ConfusionMatrix(nc=5000, task="classify")

        validator._save_plots({})

        assert not (tmp_path / "plots").exists()


# ---------------------------------------------------------------------------
# What val() returns
# ---------------------------------------------------------------------------


class TestResults:
    def test_detection_results_carry_the_matrix_and_stay_a_plain_dict(self):
        cm = _detect_fixture()
        validator = SimpleNamespace(
            image_metrics={"a.jpg": {"tp": 1}},
            best_conf_per_class={"cat": 0.4},
            confidence_curves=None,
            confusion_matrix=cm,
            task="detect",
        )
        results = with_image_metrics({"metrics/mAP50": 0.5}, validator)

        assert isinstance(results, ValidationMetrics)
        assert dict(results) == {"metrics/mAP50": 0.5}
        assert results.confusion_matrix is cm
        assert results.box.image_metrics == {"a.jpg": {"tp": 1}}
        assert results.box.best_conf_per_class == {"cat": 0.4}
        assert json.loads(json.dumps(results)) == {"metrics/mAP50": 0.5}

    def test_classify_results_expose_top1_top5_and_the_matrix(self):
        cm = ConfusionMatrix(nc=2, task="classify")
        validator = SimpleNamespace(task="classify", confusion_matrix=cm)
        results = with_image_metrics(
            {"metrics/accuracy_top1": 0.75, "metrics/accuracy_top5": 1.0}, validator
        )

        assert isinstance(results, ClassifyMetrics)
        assert (results.top1, results.top5) == (0.75, 1.0)
        assert results.confusion_matrix is cm
        assert not hasattr(results, "box")

    def test_other_validators_pass_through(self):
        metrics = {"metrics/mIoU": 0.4}
        assert with_image_metrics(metrics, SimpleNamespace(task="semantic")) is metrics

    def test_legacy_positional_construction_still_works(self):
        results = ValidationMetrics({"k": 1.0}, {"a.jpg": {}}, {"cat": 0.3})
        assert results.box.best_conf_per_class == {"cat": 0.3}
        assert results.confusion_matrix is None
        assert results.box.p_curve.shape == (0, 0)
