"""``val(visualize=True)`` error-analysis images (#887).

Same name and meaning as the ecosystem's documented ``visualize`` validation
argument: every validated image is drawn with its true positives, false
positives and false negatives. It must never change what is scored, and a
task that cannot draw it must reject it instead of accepting and ignoring it.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from libreyolo.validation.base import BaseValidator
from libreyolo.validation.classify_validator import ClassifyValidator
from libreyolo.validation.config import VISUALIZE_TASKS, ValidationConfig
from libreyolo.validation.detection_validator import DetectionValidator
from libreyolo.validation.val_plotter import ValPlotter, match_detections

pytestmark = pytest.mark.unit

cv2 = pytest.importorskip("cv2")


class TestConfig:
    def test_defaults_match_the_ecosystem(self):
        cfg = ValidationConfig(data="x")
        assert cfg.visualize is False
        assert cfg.show_labels is True
        assert cfg.show_conf is True

    def test_does_not_need_save_plots(self):
        assert ValidationConfig(data="x", visualize=True).visualize is True


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------


def _match(pred, gt, **kw):
    pb = np.array([p[:4] for p in pred], dtype=np.float32).reshape(-1, 4)
    pc = np.array([p[4] for p in pred], dtype=int)
    ps = np.array([p[5] for p in pred], dtype=np.float32)
    gb = np.array([g[:4] for g in gt], dtype=np.float32).reshape(-1, 4)
    gc = np.array([g[4] for g in gt], dtype=int)
    return match_detections(pb, pc, ps, gb, gc, **kw)


class TestMatchDetections:
    def test_true_positive(self):
        m = _match([(10, 10, 50, 50, 1, 0.9)], [(10, 10, 50, 50, 1)])
        assert m["tp"].tolist() == [True]
        assert m["fn"].tolist() == [False]

    def test_empty(self):
        m = _match([], [])
        assert m["tp"].size == 0 and m["fn"].size == 0

    def test_false_negative(self):
        assert _match([], [(10, 10, 50, 50, 0)])["fn"].tolist() == [True]

    def test_false_positive(self):
        assert _match([(10, 10, 50, 50, 0, 0.9)], [])["tp"].tolist() == [False]

    def test_wrong_class_is_a_false_positive_and_a_false_negative(self):
        m = _match([(10, 10, 50, 50, 2, 0.8)], [(10, 10, 50, 50, 0)])
        assert m["tp"].tolist() == [False]
        assert m["fn"].tolist() == [True]

    def test_low_confidence_predictions_are_not_drawn(self):
        m = _match(
            [(10, 10, 50, 50, 1, 0.9), (60, 60, 90, 90, 1, 0.1)],
            [(10, 10, 50, 50, 1)],
        )
        assert m["keep"].tolist() == [True, False]
        assert m["tp"].tolist() == [True]

    def test_below_iou_threshold(self):
        m = _match([(40, 40, 80, 80, 1, 0.9)], [(10, 10, 50, 50, 1)])
        assert m["tp"].tolist() == [False]
        assert m["fn"].tolist() == [True]

    def test_one_to_one(self):
        """A duplicate box on one object is a false positive."""
        m = _match(
            [(10, 10, 50, 50, 1, 0.9), (12, 12, 50, 50, 1, 0.8)],
            [(10, 10, 50, 50, 1)],
        )
        assert sorted(m["tp"].tolist()) == [False, True]
        assert m["fn"].tolist() == [False]

    def test_same_class_match_wins_over_higher_iou_other_class(self):
        m = _match(
            [(10, 10, 50, 50, 2, 0.9), (13, 13, 50, 50, 1, 0.8)],
            [(10, 10, 50, 50, 1)],
        )
        assert m["tp"].tolist() == [False, True]
        assert m["fn"].tolist() == [False]


# ---------------------------------------------------------------------------
# DetectionValidator
# ---------------------------------------------------------------------------


class _RecordingEvaluator:
    def __init__(self):
        self.calls = []

    def update(self, pred, img_id):
        self.calls.append((pred["boxes"].tolist(), pred["classes"].tolist(), img_id))


def _detection_validator(tmp_path, visualize, n_images=3, **cfg):
    paths = []
    for i in range(n_images):
        path = tmp_path / f"img{i}.jpg"
        cv2.imwrite(str(path), np.full((100, 100, 3), 60, dtype=np.uint8))
        paths.append(path)
    v = DetectionValidator.__new__(DetectionValidator)
    v.config = ValidationConfig(data="x", visualize=visualize, **cfg)
    v.nc = 2
    v.seen = 0
    v.class_names = ["car", "person"]
    v.model = SimpleNamespace()
    v.save_dir = tmp_path / "run"
    v.coco_evaluator = _RecordingEvaluator()
    v.dataloader = SimpleNamespace(dataset=object())
    v._resolve_img_path = lambda dataset, idx, img_id: str(paths[idx])
    return v


def _batch(correct_mask):
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
    return preds, targets, [(100, 100)] * n, list(range(n))


class TestDetectionValidator:
    def test_draws_every_image_named_by_index_and_stem(self, tmp_path):
        v = _detection_validator(tmp_path, visualize=True)
        v._update_metrics(*_batch([True, False, True]))
        written = sorted(p.name for p in (v.save_dir / "visualize").iterdir())
        assert written == ["000000_img0.jpg", "000001_img1.jpg", "000002_img2.jpg"]

    def test_off_writes_nothing(self, tmp_path):
        v = _detection_validator(tmp_path, visualize=False)
        v._update_metrics(*_batch([False, False, False]))
        assert not (v.save_dir / "visualize").exists()

    def test_scoring_input_is_identical(self, tmp_path):
        mask = [True, False, True]
        (tmp_path / "a").mkdir()
        (tmp_path / "b").mkdir()
        off = _detection_validator(tmp_path / "a", visualize=False)
        on = _detection_validator(tmp_path / "b", visualize=True)
        off._update_metrics(*_batch(mask))
        on._update_metrics(*_batch(mask))
        assert off.coco_evaluator.calls == on.coco_evaluator.calls

    def test_a_broken_image_does_not_stop_validation(self, tmp_path):
        v = _detection_validator(tmp_path, visualize=True)
        v._resolve_img_path = lambda dataset, idx, img_id: str(tmp_path / "missing.jpg")
        v._update_metrics(*_batch([True]))
        assert len(v.coco_evaluator.calls) == 1


# ---------------------------------------------------------------------------
# ClassifyValidator
# ---------------------------------------------------------------------------


def _classify_validator(tmp_path, visualize, n_images=3, names=("cat", "dog", "fox"), **cfg):
    paths = []
    for i in range(n_images):
        path = tmp_path / f"img{i}.png"
        cv2.imwrite(str(path), np.full((32, 32, 3), 40 * i, dtype=np.uint8))
        paths.append(str(path))
    v = ClassifyValidator.__new__(ClassifyValidator)
    v.config = ValidationConfig(data="x", visualize=visualize, **cfg)
    v.model = SimpleNamespace(names=dict(enumerate(names)), nb_classes=len(names))
    v.seen = 0
    v.save_dir = tmp_path / "run"
    v._init_metrics()
    impl = SimpleNamespace(samples=[(p, 0) for p in paths])
    v.dataloader = SimpleNamespace(dataset=SimpleNamespace(_impl=impl, classes=list(names)))
    return v


class TestClassifyValidator:
    def test_draws_every_image(self, tmp_path):
        v = _classify_validator(tmp_path, visualize=True)
        logits = torch.tensor([[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]])
        v._update_metrics(logits, torch.tensor([0, 0, 2]), [{}] * 3)
        written = sorted(p.name for p in (v.save_dir / "visualize").iterdir())
        assert written == ["000000_img0.jpg", "000001_img1.jpg", "000002_img2.jpg"]
        assert v._top1_correct == 2

    def test_batch_offset(self, tmp_path):
        v = _classify_validator(tmp_path, visualize=True)
        v.seen = 2
        v._update_metrics(torch.tensor([[0.0, 5.0, 0.0]]), torch.tensor([0]), [{}])
        written = [p.name for p in (v.save_dir / "visualize").iterdir()]
        assert written == ["000002_img2.jpg"]

    def test_off_writes_nothing(self, tmp_path):
        v = _classify_validator(tmp_path, visualize=False)
        v._update_metrics(torch.tensor([[0.0, 5.0, 0.0]]), torch.tensor([0]), [{}])
        assert not (v.save_dir / "visualize").exists()

    def test_metrics_are_identical(self, tmp_path):
        logits = torch.tensor([[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]])
        targets = torch.tensor([0, 0, 2])
        (tmp_path / "a").mkdir()
        (tmp_path / "b").mkdir()
        off = _classify_validator(tmp_path / "a", visualize=False)
        on = _classify_validator(tmp_path / "b", visualize=True)
        off._update_metrics(logits, targets, [{}] * 3)
        on._update_metrics(logits, targets, [{}] * 3)
        assert off._compute_metrics() == on._compute_metrics()

    def test_generic_model_names_fall_back_to_dataset_classes(self, tmp_path):
        v = _classify_validator(tmp_path, visualize=True)
        v.model = SimpleNamespace(names={0: "class_0", 1: "class_1", 2: "class_2"}, nb_classes=3)
        assert v._class_display_name(1) == "dog"


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------


class TestDrawing:
    def test_classify_upscales_tiny_images(self, tmp_path):
        out = tmp_path / "e.jpg"
        ValPlotter.plot_classify_visualize(
            np.zeros((32, 32, 3), np.uint8), "a very long class name", "other", 0.42, out
        )
        img = cv2.imread(str(out))
        assert min(img.shape[:2]) >= 320

    def test_classify_keeps_large_images_at_size(self, tmp_path):
        out = tmp_path / "e.jpg"
        ValPlotter.plot_classify_visualize(np.zeros((480, 640, 3), np.uint8), "a", "b", 0.9, out)
        assert cv2.imread(str(out)).shape[:2] == (480, 640)

    @pytest.mark.parametrize("show_labels,show_conf", [(True, True), (False, True), (True, False), (False, False)])
    def test_text_toggles_draw(self, tmp_path, show_labels, show_conf):
        out = tmp_path / "d.jpg"
        ValPlotter.plot_detection_visualize(
            np.zeros((400, 600, 3), np.uint8),
            np.array([[10, 10, 100, 100], [200, 200, 300, 300]], np.float32),
            np.array([0, 1]),
            np.array([[12, 12, 100, 100], [450, 50, 590, 150]], np.float32),
            np.array([0, 0]),
            np.array([0.8, 0.6], np.float32),
            ["car", "person"],
            out,
            show_labels=show_labels,
            show_conf=show_conf,
        )
        img = cv2.imread(str(out))
        assert img.shape[1] == 600 and img.shape[0] > 400

    def test_text_is_off_when_both_toggles_are_off(self, tmp_path):
        a, b = tmp_path / "a.jpg", tmp_path / "b.jpg"
        args = (np.zeros((200, 200, 3), np.uint8), "cat", "dog", 0.7)
        ValPlotter.plot_classify_visualize(*args, a, show_labels=False, show_conf=False)
        ValPlotter.plot_classify_visualize(*args, b)
        assert not np.array_equal(cv2.imread(str(a)), cv2.imread(str(b)))

    def test_font_scale_fits_the_width(self):
        long = "x" * 200
        scale = ValPlotter._fit_font_scale(cv2, [long], 300, 1.0)
        (tw, _), _ = cv2.getTextSize(long, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        assert tw <= 300 or scale <= 0.25


# ---------------------------------------------------------------------------
# Unsupported tasks reject the flag
# ---------------------------------------------------------------------------


class TestUnsupportedTasks:
    def test_supported_tasks(self):
        assert set(VISUALIZE_TASKS) == {"detect", "segment", "classify"}

    def test_validators_declare_support(self):
        from libreyolo.validation import SegmentationValidator

        assert DetectionValidator.supports_visualize
        assert SegmentationValidator.supports_visualize
        assert ClassifyValidator.supports_visualize
        assert BaseValidator.supports_visualize is False

    @pytest.mark.parametrize("name", ["PoseValidator", "OBBValidator", "SemanticValidator"])
    def test_other_validators_raise(self, name):
        import libreyolo.validation as validation

        cls = getattr(validation, name)
        config = ValidationConfig(data="x", visualize=True)
        with pytest.raises(ValueError, match="visualize=True is not supported"):
            cls(model=SimpleNamespace(device="cpu", nb_classes=1), config=config)

    def test_model_val_raises_for_other_tasks(self):
        from libreyolo.models.base.model import BaseModel

        fake = SimpleNamespace(task="pose", _get_input_size=lambda: 640)
        with pytest.raises(ValueError, match="task 'pose'"):
            BaseModel.val(fake, data="x", visualize=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class TestCli:
    def _val(self, monkeypatch, tmp_path, extra):
        import typer
        from typer.testing import CliRunner

        from libreyolo.cli.commands.val import val_cmd
        from libreyolo.cli.parsing import KeyValueCommand

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
        app = typer.Typer()
        app.command("val", cls=KeyValueCommand)(val_cmd)
        result = CliRunner().invoke(
            app,
            ["data=coco8.yaml", "model=LibreYOLO9t.pt", f"project={tmp_path}",
             "exist_ok=true", "--json", *extra],
        )
        assert result.exit_code == 0, result.output
        return captured["kwargs"]

    def test_key_value_grammar(self, monkeypatch, tmp_path):
        kw = self._val(monkeypatch, tmp_path, ["visualize=true", "show_conf=false"])
        assert kw["visualize"] is True
        assert kw["show_conf"] is False
        assert "show_labels" not in kw

    def test_flag_grammar(self, monkeypatch, tmp_path):
        kw = self._val(monkeypatch, tmp_path, ["--visualize", "--no-show-labels"])
        assert kw["visualize"] is True
        assert kw["show_labels"] is False

    def test_defaults_are_not_forwarded(self, monkeypatch, tmp_path):
        """Families whose val() rejects unknown kwargs keep working."""
        kw = self._val(monkeypatch, tmp_path, [])
        assert not {"visualize", "show_labels", "show_conf"} & set(kw)


# ---------------------------------------------------------------------------
# Review fixes: effective confidence and reused run directories
# ---------------------------------------------------------------------------


class TestEffectiveConfidence:
    @pytest.mark.parametrize("configured,expected", [(0.001, 0.25), (0.25, 0.25), (0.5, 0.5), (None, 0.25)])
    def test_draws_at_the_higher_of_025_and_the_run_conf(self, configured, expected):
        from libreyolo.validation.val_plotter import visualize_conf_thres

        assert visualize_conf_thres(configured) == expected

    def test_validator_passes_the_run_conf(self, tmp_path, monkeypatch):
        seen = {}

        def _record(*args, **kwargs):
            seen["conf_thres"] = kwargs["conf_thres"]

        monkeypatch.setattr(ValPlotter, "plot_detection_visualize", staticmethod(_record))
        v = _detection_validator(tmp_path, visualize=True, n_images=1, conf_thres=0.6)
        v._update_metrics(*_batch([True]))
        assert seen["conf_thres"] == 0.6


class TestReusedRunDirectory:
    def test_only_visualize_images_from_an_earlier_run_are_removed(self, tmp_path):
        from libreyolo.validation.val_plotter import reset_visualize_dir

        out = tmp_path / "visualize"
        out.mkdir()
        (out / "000007_old.jpg").write_bytes(b"x")
        (out / "notes.txt").write_text("keep")
        (out / "mine.jpg").write_bytes(b"x")
        assert reset_visualize_dir(tmp_path) == out
        assert sorted(p.name for p in out.iterdir()) == ["mine.jpg", "notes.txt"]

    def test_missing_directory_is_fine(self, tmp_path):
        from libreyolo.validation.val_plotter import reset_visualize_dir

        assert not reset_visualize_dir(tmp_path / "new").exists()

    def test_classify_run_starts_clean(self, tmp_path):
        stale = tmp_path / "run" / "visualize" / "000099_old.jpg"
        stale.parent.mkdir(parents=True)
        stale.write_bytes(b"x")
        _classify_validator(tmp_path, visualize=True)
        assert not stale.exists()
