"""Tests for CLI config discovery and model name resolution."""

import pytest

from dataclasses import fields as dc_fields

from libreyolo.cli.config import (
    build_train_kwargs,
    detect_family_from_name,
    get_cfg_defaults,
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


class TestBuildTrainKwargs:
    """Test auto-building train kwargs from CLI params via TrainConfig fields."""

    def test_aliases_resolved(self):
        """CLI names 'mosaic'/'mixup' map to internal 'mosaic_prob'/'mixup_prob'."""
        params = {"mosaic": 0.5, "mixup": 0.3, "epochs": 100}
        kwargs = build_train_kwargs(params)
        assert kwargs["mosaic_prob"] == 0.5
        assert kwargs["mixup_prob"] == 0.3
        assert "mosaic" not in kwargs
        assert "mixup" not in kwargs

    def test_excluded_fields(self):
        """size, num_classes, data, data_dir are never in output."""
        params = {"size": "m", "num_classes": 10, "data": "coco.yaml",
                  "data_dir": "/tmp", "epochs": 50}
        kwargs = build_train_kwargs(params)
        assert "size" not in kwargs
        assert "num_classes" not in kwargs
        assert "data" not in kwargs
        assert "data_dir" not in kwargs
        assert kwargs["epochs"] == 50

    def test_covers_all_config_fields(self):
        """Every non-excluded TrainConfig field is picked up when present."""
        excluded = {"size", "num_classes", "data", "data_dir"}
        from libreyolo.cli.aliases import TRAIN_ALIASES
        internal_to_cli = {v: k for k, v in TRAIN_ALIASES.items()}

        params = {}
        base = TrainConfig()
        for f in dc_fields(TrainConfig):
            if f.name in excluded:
                continue
            cli_name = internal_to_cli.get(f.name, f.name)
            params[cli_name] = getattr(base, f.name)

        kwargs = build_train_kwargs(params)
        expected_count = len(dc_fields(TrainConfig)) - len(excluded)
        assert len(kwargs) == expected_count

    def test_unknown_params_ignored(self):
        """Params not in TrainConfig are silently dropped."""
        params = {"epochs": 100, "pretrained": True, "val": True, "unknown": "x"}
        kwargs = build_train_kwargs(params)
        assert kwargs["epochs"] == 100
        assert "pretrained" not in kwargs
        assert "val" not in kwargs
        assert "unknown" not in kwargs


class TestGetCfgDefaults:
    """Test that cfg defaults are fully derived from dataclasses."""

    def test_has_all_sections(self):
        cfg = get_cfg_defaults()
        assert "train_defaults" in cfg
        assert "val_defaults" in cfg
        assert "predict_defaults" in cfg
        assert "family_overrides" in cfg

    def test_train_defaults_match_dataclass(self):
        """Train defaults should match TrainConfig() values."""
        cfg = get_cfg_defaults()
        base = TrainConfig()
        assert cfg["train_defaults"]["epochs"] == base.epochs
        assert cfg["train_defaults"]["lr0"] == base.lr0
        assert cfg["train_defaults"]["momentum"] == base.momentum
        assert cfg["train_defaults"]["mosaic"] == base.mosaic_prob
        assert cfg["train_defaults"]["mixup"] == base.mixup_prob

    def test_val_defaults_use_cli_names(self):
        """Val defaults should use aliased CLI names."""
        cfg = get_cfg_defaults()
        assert "batch" in cfg["val_defaults"]
        assert "conf" in cfg["val_defaults"]
        assert "iou" in cfg["val_defaults"]
        assert "workers" in cfg["val_defaults"]
        assert "batch_size" not in cfg["val_defaults"]
        assert "conf_thres" not in cfg["val_defaults"]

    def test_family_overrides_auto_discovered(self):
        """Family overrides should be discovered from model registry."""
        cfg = get_cfg_defaults()
        overrides = cfg["family_overrides"]
        assert overrides["yolox"]["momentum"] == 0.9
        assert overrides["yolo9"]["scheduler"] == "linear"
        assert overrides["yolo9"]["mixup"] == 0.0
        assert overrides["yolo9"]["workers"] == 8
        # These were missing from old hardcoded version but are real diffs
        assert overrides["yolo9"]["degrees"] == 0.0
        assert overrides["yolo9"]["shear"] == 0.0

    def test_no_excluded_fields_in_train(self):
        """Internal fields like size/num_classes should not appear."""
        cfg = get_cfg_defaults()
        assert "size" not in cfg["train_defaults"]
        assert "num_classes" not in cfg["train_defaults"]
        assert "device" not in cfg["train_defaults"]
