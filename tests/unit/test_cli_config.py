"""Tests for CLI config discovery and model name resolution."""

import pytest

from libreyolo.cli.config import (
    detect_family_from_name,
    get_family_defaults,
    get_train_config_class,
    resolve_model_name,
)
from libreyolo.training.config import TrainConfig, YOLOXConfig, YOLO9Config

pytestmark = pytest.mark.unit


class TestResolveModelName:
    """Test CLI model name → weight filename resolution."""

    def test_yolox_sizes(self):
        assert resolve_model_name("yolox-s") == "LibreYOLOXs.pt"
        assert resolve_model_name("yolox-n") == "LibreYOLOXn.pt"
        assert resolve_model_name("yolox-m") == "LibreYOLOXm.pt"

    def test_yolo9_sizes(self):
        assert resolve_model_name("yolo9-t") == "LibreYOLO9t.pt"
        assert resolve_model_name("yolo9-m") == "LibreYOLO9m.pt"

    def test_case_insensitive(self):
        assert resolve_model_name("YOLOX-S") == "LibreYOLOXs.pt"
        assert resolve_model_name("Yolo9-T") == "LibreYOLO9t.pt"

    def test_local_path_passthrough(self):
        assert resolve_model_name("best.pt") == "best.pt"
        assert resolve_model_name("runs/train/exp/weights/best.pt") == "runs/train/exp/weights/best.pt"
        assert resolve_model_name("model.onnx") == "model.onnx"

    def test_unknown_model_passthrough(self):
        assert resolve_model_name("unknown-model") == "unknown-model"


class TestDetectFamilyFromName:
    """Test family detection from CLI model names."""

    def test_yolox_family(self):
        assert detect_family_from_name("yolox-s") == "yolox"
        assert detect_family_from_name("yolox-n") == "yolox"

    def test_yolo9_family(self):
        assert detect_family_from_name("yolo9-m") == "yolo9"
        assert detect_family_from_name("yolo9-t") == "yolo9"

    def test_local_path_returns_none(self):
        assert detect_family_from_name("best.pt") is None
        assert detect_family_from_name("weights/model.pt") is None

    def test_unknown_returns_none(self):
        assert detect_family_from_name("unknown") is None


class TestGetTrainConfigClass:
    """Test auto-discovery of config classes from model registry."""

    def test_yolox_returns_yolox_config(self):
        assert get_train_config_class("yolox") is YOLOXConfig

    def test_yolo9_returns_yolo9_config(self):
        assert get_train_config_class("yolo9") is YOLO9Config

    def test_unknown_family_returns_base(self):
        assert get_train_config_class("nonexistent") is TrainConfig


class TestGetFamilyDefaults:
    """Test that family defaults are correctly diffed against base TrainConfig."""

    def test_yolox_momentum_differs(self):
        diffs = get_family_defaults("yolox")
        # YOLOXConfig.momentum = 0.9 vs TrainConfig.momentum = 0.937
        assert diffs["momentum"] == 0.9

    def test_yolo9_scheduler_differs(self):
        diffs = get_family_defaults("yolo9")
        # YOLO9Config.scheduler = "linear" vs TrainConfig.scheduler = "yoloxwarmcos"
        assert diffs["scheduler"] == "linear"

    def test_yolo9_mixup_prob_differs(self):
        diffs = get_family_defaults("yolo9")
        # YOLO9Config.mixup_prob = 0.0 vs TrainConfig.mixup_prob = 1.0
        assert diffs["mixup_prob"] == 0.0

    def test_yolox_only_has_differing_keys(self):
        diffs = get_family_defaults("yolox")
        # epochs is the same in both (300), should NOT be in diffs
        assert "epochs" not in diffs
        # batch is the same in both (16), should NOT be in diffs
        assert "batch" not in diffs

    def test_unknown_family_returns_empty(self):
        assert get_family_defaults("nonexistent") == {}
