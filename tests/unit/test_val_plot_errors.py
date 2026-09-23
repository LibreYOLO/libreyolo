"""Error-analysis validation plots (#887).

``plot_errors`` draws the first N incorrect validation images. It is a
plotting budget like ``plot_samples``: it must never change what is scored,
and an unsupported task must reject it instead of accepting and ignoring it.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from libreyolo.training.config import TrainConfig
from libreyolo.utils.plot_samples import (
    PLOT_ERRORS_TASKS,
    validate_plot_errors,
    wants_more_plot_errors,
)
from libreyolo.validation.base import BaseValidator
from libreyolo.validation.classify_validator import ClassifyValidator
from libreyolo.validation.config import ValidationConfig
from libreyolo.validation.detection_validator import DetectionValidator
from libreyolo.validation.val_plotter import (
    GT_CORRECT,
    GT_MISSED,
    GT_WRONG_CLASS,
    PRED_CORRECT,
    PRED_FALSE_POSITIVE,
    PRED_WRONG_CLASS,
    ValPlotter,
    detection_errors,
)

pytestmark = pytest.mark.unit

cv2 = pytest.importorskip("cv2")


# ---------------------------------------------------------------------------
# Option validation
# ---------------------------------------------------------------------------


class TestValidatePlotErrors:
    @pytest.mark.parametrize("value", [0, 1, 8, 500, -1])
    def test_accepted_with_save_plots(self, value):
        assert validate_plot_errors(value, save_plots=True) == value

    def test_zero_needs_nothing(self):
        assert validate_plot_errors(0, save_plots=False) == 0

    @pytest.mark.parametrize("value", [-2, "abc", None])
    def test_rejected(self, value):
        with pytest.raises(ValueError, match="plot_errors"):
            validate_plot_errors(value, save_plots=True)

    @pytest.mark.parametrize("value", [1, -1])
    def test_nonzero_without_save_plots_raises(self, value):
        """It would parse and draw nothing: accept-and-ignore is not allowed."""
        with pytest.raises(ValueError, match="save_plots"):
            validate_plot_errors(value, save_plots=False)

    def test_validation_config_default_is_off(self):
        assert ValidationConfig(data="x").plot_errors == 0

    def test_validation_config_checks_at_construction(self):
        assert ValidationConfig(data="x", save_plots=True, plot_errors=5).plot_errors == 5
        with pytest.raises(ValueError, match="save_plots"):
            ValidationConfig(data="x", plot_errors=5)

    def test_train_config_checks_at_construction(self):
        assert TrainConfig(data="x").plot_errors == 0
        assert TrainConfig(data="x", save_plots=True, plot_errors=-1).plot_errors == -1
        with pytest.raises(ValueError, match="save_plots"):
            TrainConfig(data="x", plot_errors=3)


class TestErrorBudget:
    @pytest.mark.parametrize(
        "budget,collected,expected",
        [(0, 0, False), (1, 0, True), (1, 1, False), (5, 4, True), (5, 5, False)],
    )
    def test_budget(self, budget, collected, expected):
        cfg = ValidationConfig(data="x", save_plots=True, plot_errors=budget)
        assert wants_more_plot_errors(cfg, collected) is expected

    def test_all_never_stops(self):
        cfg = ValidationConfig(data="x", save_plots=True, plot_errors=-1)
        assert wants_more_plot_errors(cfg, 10_000) is True

    def test_nothing_is_kept_when_plots_are_off(self):
        assert wants_more_plot_errors(SimpleNamespace(plot_errors=-1, save_plots=False), 0) is False

    def test_config_without_the_field_keeps_nothing(self):
        assert wants_more_plot_errors(SimpleNamespace(save_plots=True), 0) is False


# ---------------------------------------------------------------------------
# Detection matching
# ---------------------------------------------------------------------------


def _errors(pred, gt, **kw):
    pb = np.array([p[:4] for p in pred], dtype=np.float32).reshape(-1, 4)
    pc = np.array([p[4] for p in pred], dtype=int)
    ps = np.array([p[5] for p in pred], dtype=np.float32)
    gb = np.array([g[:4] for g in gt], dtype=np.float32).reshape(-1, 4)
    gc = np.array([g[4] for g in gt], dtype=int)
    return detection_errors(pb, pc, ps, gb, gc, **kw)


class TestDetectionErrors:
    def test_perfect_image_has_no_error(self):
        e = _errors([(10, 10, 50, 50, 1, 0.9)], [(10, 10, 50, 50, 1)])
        assert e["has_error"] is False
        assert list(e["gt_status"]) == [GT_CORRECT]
        assert list(e["pred_status"]) == [PRED_CORRECT]

    def test_empty_image_has_no_error(self):
        assert _errors([], [])["has_error"] is False

    def test_missed_ground_truth(self):
        e = _errors([], [(10, 10, 50, 50, 0)])
        assert e["has_error"] is True
        assert list(e["gt_status"]) == [GT_MISSED]

    def test_false_positive(self):
        e = _errors([(10, 10, 50, 50, 0, 0.9)], [])
        assert e["has_error"] is True
        assert list(e["pred_status"]) == [PRED_FALSE_POSITIVE]

    def test_wrong_class_pairs_by_iou(self):
        e = _errors([(10, 10, 50, 50, 2, 0.8)], [(11, 11, 50, 50, 0)])
        assert list(e["gt_status"]) == [GT_WRONG_CLASS]
        assert list(e["pred_status"]) == [PRED_WRONG_CLASS]
        assert list(e["pred_match"]) == [0]

    def test_low_confidence_predictions_are_ignored(self):
        """A 0.1 duplicate is not a false positive at the 0.25 threshold."""
        e = _errors(
            [(10, 10, 50, 50, 1, 0.9), (60, 60, 90, 90, 1, 0.1)],
            [(10, 10, 50, 50, 1)],
        )
        assert e["has_error"] is False
        assert list(e["keep"]) == [True, False]

    def test_below_iou_threshold_is_a_miss_and_a_false_positive(self):
        e = _errors([(40, 40, 80, 80, 1, 0.9)], [(10, 10, 50, 50, 1)])
        assert list(e["gt_status"]) == [GT_MISSED]
        assert list(e["pred_status"]) == [PRED_FALSE_POSITIVE]

    def test_matching_is_one_to_one(self):
        """A duplicate box on the same object is a false positive."""
        e = _errors(
            [(10, 10, 50, 50, 1, 0.9), (12, 12, 50, 50, 1, 0.8)],
            [(10, 10, 50, 50, 1)],
        )
        assert sorted(e["pred_status"]) == sorted([PRED_CORRECT, PRED_FALSE_POSITIVE])
        assert list(e["gt_status"]) == [GT_CORRECT]


# ---------------------------------------------------------------------------
# DetectionValidator collection
# ---------------------------------------------------------------------------


class _RecordingConfusionMatrix:
    def __init__(self):
        self.calls = []

    def process_image(self, pb, pc, ps, gt_boxes, gt_classes):
        self.calls.append((pb.tolist(), pc.tolist(), ps.tolist(), gt_boxes.tolist()))


def _run_detection_track(plot_errors, correct_mask):
    """Drive the real _track_plots_data; ``correct_mask[i]`` picks image i's pred."""
    v = DetectionValidator.__new__(DetectionValidator)
    v.config = ValidationConfig(
        data="x", save_plots=True, plot_samples=0, plot_errors=plot_errors
    )
    v.nc = 2
    v.seen = 0
    v._val_samples = []
    v._error_samples = []
    v._confusion_matrix = _RecordingConfusionMatrix()
    v.dataloader = SimpleNamespace(dataset=object())
    v._resolve_img_path = lambda dataset, idx, img_id: f"img{idx}.jpg"

    # GT: class 1 box at (20, 20, 40, 40) on a 100x100 image.
    preds = [
        {
            "boxes": torch.tensor([[20.0, 20.0, 40.0, 40.0]]),
            "scores": torch.tensor([0.9]),
            "classes": torch.tensor([1 if ok else 0]),
        }
        for ok in correct_mask
    ]
    targets = torch.tensor(
        [[[1.0, 0.3, 0.3, 0.2, 0.2]] for _ in correct_mask], dtype=torch.float32
    )
    n = len(correct_mask)
    v._track_plots_data(preds, targets, [(100, 100)] * n, list(range(n)))
    return v


class TestDetectionCollection:
    def test_keeps_only_incorrect_images_in_order(self):
        v = _run_detection_track(-1, [True, False, True, False, False])
        assert [s["img_path"] for s in v._error_samples] == [
            "img1.jpg",
            "img3.jpg",
            "img4.jpg",
        ]

    def test_budget_keeps_the_first_n(self):
        v = _run_detection_track(2, [False] * 6)
        assert [s["img_path"] for s in v._error_samples] == ["img0.jpg", "img1.jpg"]

    def test_off_keeps_nothing(self):
        v = _run_detection_track(0, [False] * 3)
        assert v._error_samples == []

    def test_scoring_input_is_identical_at_any_budget(self):
        mask = [True, False, False, True]
        baseline = _run_detection_track(0, mask)._confusion_matrix.calls
        for budget in (1, -1):
            assert _run_detection_track(budget, mask)._confusion_matrix.calls == baseline

    def test_validator_without_buffer_collects_nothing(self):
        """Narrow-scope validators built without _init_metrics keep working."""
        v = DetectionValidator.__new__(DetectionValidator)
        v.config = ValidationConfig(data="x", save_plots=True, plot_errors=-1)
        assert v._wants_more_error_samples() is False


# ---------------------------------------------------------------------------
# ClassifyValidator collection and output
# ---------------------------------------------------------------------------


def _classify_validator(tmp_path, plot_errors, n_images=4, names=("cat", "dog", "fox")):
    paths = []
    for i in range(n_images):
        path = tmp_path / f"img{i}.png"
        cv2.imwrite(str(path), np.full((32, 32, 3), 40 * i, dtype=np.uint8))
        paths.append(str(path))
    v = ClassifyValidator.__new__(ClassifyValidator)
    v.config = ValidationConfig(
        data="x", save_plots=plot_errors != 0, plot_errors=plot_errors
    )
    v.model = SimpleNamespace(names=dict(enumerate(names)), nb_classes=len(names))
    v.seen = 0
    v.save_dir = tmp_path / "run"
    v._error_samples = []
    impl = SimpleNamespace(samples=[(p, 0) for p in paths])
    v.dataloader = SimpleNamespace(dataset=SimpleNamespace(_impl=impl, classes=list(names)))
    return v


class TestClassifyCollection:
    def test_keeps_wrong_top1_with_label_prediction_and_score(self, tmp_path):
        v = _classify_validator(tmp_path, plot_errors=-1)
        logits = torch.tensor(
            [[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [5.0, 0.0, 0.0], [0.0, 0.0, 5.0]]
        )
        targets = torch.tensor([0, 0, 0, 1])
        top1 = logits.argmax(dim=1)
        v._track_errors(logits, targets, top1, top1 == targets)

        assert [(Path(s["img_path"]).name, s["target"], s["pred"]) for s in v._error_samples] == [
            ("img1.png", 0, 1),
            ("img3.png", 1, 2),
        ]
        expected = float(torch.softmax(logits[1], dim=0)[1])
        assert v._error_samples[0]["score"] == pytest.approx(expected)

    def test_budget_and_batch_offset(self, tmp_path):
        """The image index is offset by the images already seen."""
        v = _classify_validator(tmp_path, plot_errors=1)
        v.seen = 2
        logits = torch.tensor([[0.0, 5.0, 0.0], [0.0, 5.0, 0.0]])
        targets = torch.tensor([0, 0])
        top1 = logits.argmax(dim=1)
        v._track_errors(logits, targets, top1, top1 == targets)
        assert [Path(s["img_path"]).name for s in v._error_samples] == ["img2.png"]

    def test_off_keeps_nothing(self, tmp_path):
        v = _classify_validator(tmp_path, plot_errors=0)
        logits = torch.tensor([[0.0, 5.0, 0.0]])
        top1 = logits.argmax(dim=1)
        v._track_errors(logits, torch.tensor([0]), top1, top1 == torch.tensor([0]))
        assert v._error_samples == []

    def test_save_plots_writes_one_image_per_error(self, tmp_path):
        v = _classify_validator(tmp_path, plot_errors=-1)
        logits = torch.tensor([[0.0, 5.0, 0.0], [5.0, 0.0, 0.0], [0.0, 0.0, 5.0]])
        targets = torch.tensor([0, 0, 0])
        top1 = logits.argmax(dim=1)
        v._track_errors(logits, targets, top1, top1 == targets)
        v._save_plots({})

        written = sorted(p.name for p in (v.save_dir / "plots" / "errors").iterdir())
        assert written == ["error_000.jpg", "error_001.jpg"]

    def test_generic_model_names_fall_back_to_dataset_classes(self, tmp_path):
        v = _classify_validator(tmp_path, plot_errors=-1)
        v.model = SimpleNamespace(names={0: "class_0", 1: "class_1", 2: "class_2"}, nb_classes=3)
        assert v._class_display_name(1) == "dog"


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------


class TestDrawing:
    def test_classify_error_upscales_tiny_images_for_legible_text(self, tmp_path):
        out = tmp_path / "e.jpg"
        ValPlotter.plot_classify_error(
            np.zeros((32, 32, 3), np.uint8), "a very long class name", "other", 0.42, out
        )
        img = cv2.imread(str(out))
        assert img.shape[0] >= 320 and img.shape[1] >= 320

    def test_classify_error_keeps_large_images_at_size(self, tmp_path):
        out = tmp_path / "e.jpg"
        ValPlotter.plot_classify_error(np.zeros((480, 640, 3), np.uint8), "a", "b", 0.9, out)
        assert cv2.imread(str(out)).shape[:2] == (480, 640)

    def test_font_scale_fits_the_width(self):
        long = "x" * 200
        scale = ValPlotter._fit_font_scale(cv2, [long], 300, 1.0)
        (tw, _), _ = cv2.getTextSize(long, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        assert tw <= 300 or scale <= 0.25

    def test_detection_error_writes_header_plus_image(self, tmp_path):
        out = tmp_path / "d.jpg"
        ValPlotter.plot_detection_error(
            np.zeros((400, 600, 3), np.uint8),
            np.array([[10, 10, 100, 100], [200, 200, 300, 300]], np.float32),
            np.array([0, 1]),
            np.array([[12, 12, 100, 100], [450, 50, 590, 150]], np.float32),
            np.array([1, 0]),
            np.array([0.8, 0.6], np.float32),
            ["car", "person"],
            out,
        )
        img = cv2.imread(str(out))
        assert img.shape[1] == 600 and img.shape[0] > 400


# ---------------------------------------------------------------------------
# Unsupported tasks reject the option
# ---------------------------------------------------------------------------


class TestUnsupportedTasks:
    def test_supported_tasks(self):
        assert set(PLOT_ERRORS_TASKS) == {"detect", "segment", "classify"}

    def test_validators_that_draw_it_declare_so(self):
        from libreyolo.validation import SegmentationValidator

        assert DetectionValidator.supports_plot_errors
        assert SegmentationValidator.supports_plot_errors
        assert ClassifyValidator.supports_plot_errors
        assert BaseValidator.supports_plot_errors is False

    @pytest.mark.parametrize("name", ["PoseValidator", "OBBValidator", "SemanticValidator"])
    def test_other_validators_raise(self, name):
        import libreyolo.validation as validation

        cls = getattr(validation, name)
        assert cls.supports_plot_errors is False
        config = ValidationConfig(data="x", save_plots=True, plot_errors=3)
        with pytest.raises(ValueError, match="plot_errors is not supported"):
            cls(model=SimpleNamespace(device="cpu", nb_classes=1), config=config)

    def test_model_val_raises_for_other_tasks(self):
        from libreyolo.models.base.model import BaseModel

        fake = SimpleNamespace(task="pose", _get_input_size=lambda: 640)
        with pytest.raises(ValueError, match="task 'pose'"):
            BaseModel.val(fake, data="x", plots=True, plot_errors=2)

    def test_trainer_rejects_other_tasks_up_front(self):
        from libreyolo.training.trainer import BaseTrainer

        trainer = SimpleNamespace(
            wrapper_model=SimpleNamespace(task="obb"),
            config=SimpleNamespace(plot_errors=4),
        )
        with pytest.raises(ValueError, match="task 'obb'"):
            BaseTrainer.validate_plot_errors_config(trainer)
        trainer.wrapper_model.task = "segment"
        BaseTrainer.validate_plot_errors_config(trainer)
        trainer.wrapper_model.task = "obb"
        trainer.config.plot_errors = 0
        BaseTrainer.validate_plot_errors_config(trainer)


# ---------------------------------------------------------------------------
# CLI plumbing
# ---------------------------------------------------------------------------


def _invoke(cmd_name, cmd, args):
    import typer
    from typer.testing import CliRunner

    from libreyolo.cli.parsing import KeyValueCommand

    app = typer.Typer()
    app.command(cmd_name, cls=KeyValueCommand)(cmd)
    return CliRunner().invoke(app, args)


class TestCliPlumbing:
    def _val(self, monkeypatch, tmp_path, extra):
        from libreyolo.cli.commands.val import val_cmd

        captured = {}

        class _Detector:
            FAMILY = "yolo9"
            size = "t"
            task = "detect"
            device = "cpu"

            def val(self, **kwargs):
                captured["kwargs"] = kwargs
                return {"metrics/mAP50-95": 0.5, "metrics/mAP50": 0.6}

        monkeypatch.setattr(
            "libreyolo.cli.commands.val.load_model_or_exit", lambda *a, **k: _Detector()
        )
        result = _invoke(
            "val",
            val_cmd,
            ["data=coco8.yaml", "model=LibreYOLO9t.pt", f"project={tmp_path}",
             "exist_ok=true", "--json", *extra],
        )
        assert result.exit_code == 0, result.output
        return captured["kwargs"]

    def test_val_forwards_both_grammars(self, monkeypatch, tmp_path):
        kw = self._val(monkeypatch, tmp_path, ["save_plots=true", "plot_errors=12"])
        assert kw["plot_errors"] == 12
        kw = self._val(monkeypatch, tmp_path, ["--save-plots", "--plot-errors", "-1"])
        assert kw["plot_errors"] == -1

    def test_val_omits_the_default(self, monkeypatch, tmp_path):
        """Families whose val() rejects unknown kwargs keep working."""
        assert "plot_errors" not in self._val(monkeypatch, tmp_path, [])

    def test_train_forwards(self, monkeypatch, tmp_path):
        from libreyolo.cli.commands.train import train_cmd

        captured = {}

        class _Detector:
            FAMILY = "yolo9"
            device = "cpu"

            def train(self, data, **kwargs):
                captured["kwargs"] = kwargs
                return {"output_dir": str(tmp_path / "exp")}

        monkeypatch.setattr(
            "libreyolo.cli.commands.train.load_model_or_exit",
            lambda out, model, model_path, device: _Detector(),
        )
        result = _invoke(
            "train",
            train_cmd,
            ["data=coco8.yaml", "model=LibreYOLO9t.pt", "save_plots=true",
             "plot_errors=7", f"project={tmp_path}", "exist_ok=true", "--json"],
        )
        assert result.exit_code == 0, result.output
        assert captured["kwargs"]["plot_errors"] == 7

    def test_rfdetr_train_mapping_forwards_plot_options(self, tmp_path):
        from libreyolo.cli.config import _build_rfdetr_train_kwargs

        kwargs = _build_rfdetr_train_kwargs(
            {
                "project": str(tmp_path),
                "name": "exp",
                "exist_ok": True,
                "save_plots": True,
                "plot_samples": 4,
                "plot_errors": 9,
            }
        )
        assert kwargs["plot_samples"] == 4
        assert kwargs["plot_errors"] == 9
