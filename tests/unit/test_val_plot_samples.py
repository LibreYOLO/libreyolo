"""Configurable validation sample-plot count (#830).

`plot_samples` is a plotting budget. The load-bearing property is that it
cannot change which images are scored or the metrics that come out.
"""

from __future__ import annotations

import pytest

from libreyolo.validation.config import (
    DEFAULT_PLOT_SAMPLES,
    PLOT_SAMPLES_ALL,
    ValidationConfig,
    validate_plot_samples,
)
from libreyolo.validation.detection_validator import DetectionValidator

pytestmark = pytest.mark.unit


class TestValidatePlotSamples:
    @pytest.mark.parametrize("value,expected", [(0, 0), (8, 8), (100, 100), (-1, -1)])
    def test_accepted(self, value, expected):
        assert validate_plot_samples(value) == expected

    @pytest.mark.parametrize("value", [-2, -10, "abc", None, 1.5j])
    def test_rejected(self, value):
        with pytest.raises(ValueError, match="plot_samples"):
            validate_plot_samples(value)

    def test_float_is_truncated_like_other_int_knobs(self):
        assert validate_plot_samples(8.0) == 8


class TestConfigDefault:
    def test_default_is_the_previous_hardcoded_value(self):
        assert DEFAULT_PLOT_SAMPLES == 8
        assert ValidationConfig(data="x").plot_samples == 8

    def test_invalid_value_fails_at_config_construction(self):
        with pytest.raises(ValueError, match="plot_samples"):
            ValidationConfig(data="x", plot_samples=-5)


class _Budget:
    """Minimal stand-in exercising only the sample-budget decision."""

    def __init__(self, budget, collected):
        self.config = ValidationConfig(data="x", plot_samples=budget)
        self._val_samples = [None] * collected

    _wants_more_val_samples = DetectionValidator._wants_more_val_samples


class TestSampleBudget:
    @pytest.mark.parametrize(
        "budget,collected,expected",
        [
            (8, 0, True),
            (8, 7, True),
            (8, 8, False),
            (8, 20, False),
            (0, 0, False),
            (1, 0, True),
            (1, 1, False),
            (64, 8, True),
        ],
    )
    def test_budget_is_respected(self, budget, collected, expected):
        assert _Budget(budget, collected)._wants_more_val_samples() is expected

    @pytest.mark.parametrize("collected", [0, 8, 5000])
    def test_all_never_stops_collecting(self, collected):
        assert _Budget(PLOT_SAMPLES_ALL, collected)._wants_more_val_samples() is True

    def test_zero_collects_nothing_at_any_point(self):
        """0 must disable collection outright, not keep one image."""
        assert _Budget(0, 0)._wants_more_val_samples() is False

    def test_missing_attribute_falls_back_to_the_old_default(self):
        """Validators built from a config without the field keep old behavior."""

        class _Legacy:
            config = object()
            _val_samples = [None] * 8
            _wants_more_val_samples = DetectionValidator._wants_more_val_samples

        assert _Legacy()._wants_more_val_samples() is False
        _Legacy._val_samples = [None] * 7
        assert _Legacy()._wants_more_val_samples() is True


# ---------------------------------------------------------------------------
# The load-bearing property: the budget cannot change what is scored
# ---------------------------------------------------------------------------


class _RecordingConfusionMatrix:
    def __init__(self):
        self.calls = []

    def process_image(self, pb, pc, ps, gt_boxes, gt_classes):
        self.calls.append(
            (pb.tolist(), pc.tolist(), ps.tolist(), gt_boxes.tolist(), gt_classes.tolist())
        )


def _run_track_plots(budget, n_images):
    """Drive the real _track_plots_data with a given sample budget."""
    from types import SimpleNamespace

    import torch

    v = DetectionValidator.__new__(DetectionValidator)
    v.config = ValidationConfig(data="x", plot_samples=budget)
    v.nc = 2
    v.seen = 0
    v._val_samples = []
    v._confusion_matrix = _RecordingConfusionMatrix()
    v.dataloader = SimpleNamespace(dataset=object())
    v._resolve_img_path = lambda dataset, idx, img_id: f"img{idx}.jpg"

    preds = [
        {
            "boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]]),
            "scores": torch.tensor([0.9]),
            "classes": torch.tensor([1]),
        }
        for _ in range(n_images)
    ]
    targets = torch.tensor(
        [[[1.0, 0.3, 0.3, 0.2, 0.2]] for _ in range(n_images)], dtype=torch.float32
    )
    img_info = [(100, 100)] * n_images
    img_ids = list(range(n_images))

    v._track_plots_data(preds, targets, img_info, img_ids)
    return v


class TestBudgetDoesNotAffectScoring:
    @pytest.mark.parametrize("budget", [0, 1, 8, 20, PLOT_SAMPLES_ALL])
    def test_confusion_matrix_sees_every_image_regardless_of_budget(self, budget):
        """Every image is still processed; only the plot buffer is capped."""
        v = _run_track_plots(budget, n_images=12)
        assert len(v._confusion_matrix.calls) == 12

    def test_confusion_matrix_input_is_identical_across_budgets(self):
        baseline = _run_track_plots(8, n_images=12)._confusion_matrix.calls
        for budget in (0, 1, 20, PLOT_SAMPLES_ALL):
            other = _run_track_plots(budget, n_images=12)._confusion_matrix.calls
            assert other == baseline

    @pytest.mark.parametrize(
        "budget,n_images,expected",
        [(0, 12, 0), (1, 12, 1), (8, 12, 8), (20, 12, 12), (PLOT_SAMPLES_ALL, 12, 12)],
    )
    def test_only_the_buffer_length_changes(self, budget, n_images, expected):
        v = _run_track_plots(budget, n_images)
        assert len(v._val_samples) == expected

    def test_default_budget_reproduces_the_old_hardcoded_cap(self):
        assert len(_run_track_plots(DEFAULT_PLOT_SAMPLES, 50)._val_samples) == 8


class TestCliPlumbing:
    """Both CLI surfaces must forward the value, not merely accept it."""

    def test_val_cli_forwards_plot_samples(self, monkeypatch, tmp_path):
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
            "libreyolo.cli.commands.val.load_model_or_exit",
            lambda *a, **k: _Detector(),
        )
        app = typer.Typer()
        app.command("val", cls=KeyValueCommand)(val_cmd)
        result = CliRunner().invoke(
            app,
            [
                "data=coco8.yaml",
                "model=LibreYOLO9t.pt",
                "plot_samples=-1",
                f"project={tmp_path}",
                "exist_ok=true",
                "--json",
            ],
        )
        assert result.exit_code == 0, result.output
        assert captured["kwargs"]["plot_samples"] == -1

    def test_train_cli_forwards_plot_samples(self, monkeypatch, tmp_path):
        import typer
        from typer.testing import CliRunner

        from libreyolo.cli.commands.train import train_cmd
        from libreyolo.cli.parsing import KeyValueCommand

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
        app = typer.Typer()
        app.command("train", cls=KeyValueCommand)(train_cmd)
        result = CliRunner().invoke(
            app,
            [
                "data=coco8.yaml",
                "model=LibreYOLO9t.pt",
                "plot_samples=32",
                f"project={tmp_path}",
                "exist_ok=true",
                "--json",
            ],
        )
        assert result.exit_code == 0, result.output
        assert captured["kwargs"]["plot_samples"] == 32


class TestSharedBudgetHelper:
    """`wants_more_plot_samples` is the one decision every validator uses."""

    @pytest.mark.parametrize(
        "budget,collected,expected",
        [(8, 7, True), (8, 8, False), (0, 0, False), (-1, 10_000, True), (3, 2, True)],
    )
    def test_decision(self, budget, collected, expected):
        from types import SimpleNamespace

        from libreyolo.utils.plot_samples import wants_more_plot_samples

        cfg = SimpleNamespace(plot_samples=budget)
        assert wants_more_plot_samples(cfg, collected) is expected

    def test_missing_attribute_uses_default(self):
        from libreyolo.utils.plot_samples import wants_more_plot_samples

        assert wants_more_plot_samples(object(), 7) is True
        assert wants_more_plot_samples(object(), 8) is False

    def test_validation_config_reexports_the_shared_names(self):
        from libreyolo.utils import plot_samples as shared
        from libreyolo.validation import config as val_config

        assert val_config.validate_plot_samples is shared.validate_plot_samples
        assert val_config.wants_more_plot_samples is shared.wants_more_plot_samples
        assert val_config.DEFAULT_PLOT_SAMPLES == shared.DEFAULT_PLOT_SAMPLES
        assert val_config.PLOT_SAMPLES_ALL == shared.PLOT_SAMPLES_ALL


class TestTrainConfigValidation:
    """An invalid budget must fail before training, not silently disable
    every scheduled validation."""

    def test_default_matches_validation_config(self):
        from libreyolo.training.config import TrainConfig

        assert TrainConfig().plot_samples == DEFAULT_PLOT_SAMPLES

    @pytest.mark.parametrize("value", [-2, -100, "abc"])
    def test_invalid_budget_rejected_at_construction(self, value):
        from libreyolo.training.config import TrainConfig

        with pytest.raises(ValueError, match="plot_samples"):
            TrainConfig(plot_samples=value)

    @pytest.mark.parametrize("value,expected", [(-1, -1), (0, 0), (20, 20), ("12", 12)])
    def test_valid_budget_normalized_to_int(self, value, expected):
        from libreyolo.training.config import TrainConfig

        cfg = TrainConfig(plot_samples=value)
        assert cfg.plot_samples == expected
        assert isinstance(cfg.plot_samples, int)

    def test_every_train_budget_is_a_valid_validation_budget(self):
        from libreyolo.training.config import TrainConfig

        for value in (-1, 0, 8, 50):
            ValidationConfig(data="x", plot_samples=TrainConfig(plot_samples=value).plot_samples)


class TestPoseValidatorBudget:
    def test_pose_sample_collection_uses_the_shared_budget(self):
        import inspect

        from libreyolo.validation.pose_validator import PoseValidator

        src = inspect.getsource(PoseValidator._predict_image)
        assert "wants_more_plot_samples(" in src
        assert "_val_sample_records" in src
        assert "< 8" not in src

    def test_no_validator_keeps_a_hardcoded_sample_cap(self):
        import re
        from pathlib import Path

        import libreyolo.validation as pkg

        offenders = []
        for path in Path(pkg.__file__).parent.glob("*.py"):
            for lineno, line in enumerate(path.read_text().splitlines(), 1):
                if re.search(r"_val_sample(s|_records)\)\s*<\s*\d", line):
                    offenders.append(f"{path.name}:{lineno}: {line.strip()}")
        assert offenders == []

    def test_rfdetr_pose_epoch_validation_forwards_the_budget(self):
        import inspect

        from libreyolo.models.rfdetr.trainer import RFDETRTrainer

        src = inspect.getsource(RFDETRTrainer)
        pose_block = src[src.index("PoseValidator, ValidationConfig"):]
        pose_block = pose_block[: pose_block.index("PoseValidator(model=")]
        assert "plot_samples=" in pose_block
