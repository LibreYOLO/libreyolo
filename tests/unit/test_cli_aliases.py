"""Tests for CLI alias resolution."""

import pytest

from libreyolo.cli.aliases import TRAIN_ALIASES, VAL_ALIASES, resolve_aliases

pytestmark = pytest.mark.unit


class TestTrainAliases:
    """Test train-mode alias resolution (CLI names → internal names)."""

    def test_mosaic_resolved(self):
        result = resolve_aliases({"mosaic": 0.8}, "train")
        assert result == {"mosaic_prob": 0.8}

    def test_mixup_resolved(self):
        result = resolve_aliases({"mixup": 0.5}, "train")
        assert result == {"mixup_prob": 0.5}

    def test_non_aliased_keys_pass_through(self):
        result = resolve_aliases({"epochs": 100, "batch": 16}, "train")
        assert result == {"epochs": 100, "batch": 16}

    def test_mixed_aliased_and_non_aliased(self):
        result = resolve_aliases({"mosaic": 0.8, "epochs": 100, "mixup": 0.5}, "train")
        assert result == {"mosaic_prob": 0.8, "epochs": 100, "mixup_prob": 0.5}


class TestValAliases:
    """Test val-mode alias resolution."""

    def test_all_val_aliases(self):
        result = resolve_aliases(
            {"batch": 32, "conf": 0.001, "iou": 0.6, "workers": 8}, "val"
        )
        assert result == {
            "batch_size": 32,
            "conf_thres": 0.001,
            "iou_thres": 0.6,
            "num_workers": 8,
        }


class TestPredictExportNoAliases:
    """Predict and export modes have no aliases — keys pass through."""

    def test_predict_passthrough(self):
        result = resolve_aliases({"conf": 0.25, "iou": 0.45, "batch": 1}, "predict")
        assert result == {"conf": 0.25, "iou": 0.45, "batch": 1}

    def test_export_passthrough(self):
        result = resolve_aliases({"batch": 1, "half": True}, "export")
        assert result == {"batch": 1, "half": True}

    def test_unknown_mode_passthrough(self):
        result = resolve_aliases({"foo": "bar"}, "nonexistent")
        assert result == {"foo": "bar"}


class TestEmptyInput:
    def test_empty_dict(self):
        assert resolve_aliases({}, "train") == {}
        assert resolve_aliases({}, "val") == {}
        assert resolve_aliases({}, "predict") == {}
