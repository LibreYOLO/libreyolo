"""Tests for CLI config discovery and model name resolution."""

import pytest

from dataclasses import fields as dc_fields

import libreyolo.cli.config as cli_config
from libreyolo.cli.config import (
    build_train_kwargs,
    detect_family_from_name,
    detect_family_from_model_ref,
    detect_family_from_weight_filename,
    get_cfg_defaults,
    get_family_defaults,
    get_train_config_class,
    is_known_weight_filename,
    resolve_model_name,
)
from libreyolo.training.config import TrainConfig, YOLOXConfig, YOLO9Config
from libreyolo.models.rtdetr.config import RTDETRConfig

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

    def test_deimv2_sizes(self):
        assert resolve_model_name("deimv2-atto") == "LibreDEIMv2atto.pt"
        assert resolve_model_name("deimv2-femto") == "LibreDEIMv2femto.pt"
        assert resolve_model_name("deimv2-pico") == "LibreDEIMv2pico.pt"
        assert resolve_model_name("deimv2-n") == "LibreDEIMv2n.pt"
        assert resolve_model_name("deimv2-x") == "LibreDEIMv2x.pt"

    def test_rfdetr_seg_sizes(self):
        assert resolve_model_name("rfdetr-n-seg") == "LibreRFDETRn-seg.pt"
        assert resolve_model_name("rfdetr-x-seg") == "LibreRFDETRx-seg.pt"
        assert resolve_model_name("rfdetr-xx-seg") == "LibreRFDETRxx-seg.pt"

    @pytest.mark.parametrize(
        ("short_name", "advertised_name", "weight_name"),
        [
            ("clip-b32", "clip-b32-cls", "LibreCLIPb32-cls.pt"),
            ("ddcolor-t", "ddcolor-t-restore", "LibreDDColort-restore.pt"),
            ("ddcolor-l", "ddcolor-l-restore", "LibreDDColorl-restore.pt"),
            (
                "depth_anything-s",
                "depth_anything-s-depth",
                "LibreDepthAnythingV2s-depth.pt",
            ),
            ("fomo-s", "fomo-s-point", "LibreFOMOs-point.pt"),
            (
                "hvi_cidnet-t",
                "hvi_cidnet-t-restore",
                "LibreHVICIDNett-restore.pt",
            ),
            ("lama-b", "lama-b-restore", "LibreLaMab-restore.pt"),
            ("midas-s", "midas-s-depth", "LibreMiDaSs-depth.pt"),
            ("siglip2-b16", "siglip2-b16-cls", "LibreSigLIP2b16-cls.pt"),
            ("vit-ti", "vit-ti-cls", "LibreViTti-cls.pt"),
            ("vitmatte-s", "vitmatte-s-matte", "LibreViTMattes-matte.pt"),
            ("zipdepth-b", "zipdepth-b-depth", "LibreZipDepthb-depth.pt"),
        ],
    )
    def test_non_detection_default_aliases_use_canonical_suffix(
        self, short_name, advertised_name, weight_name
    ):
        assert resolve_model_name(short_name) == weight_name
        assert resolve_model_name(advertised_name) == weight_name

    def test_suffix_optional_family_keeps_unsuffixed_checkpoint(self):
        assert resolve_model_name("l2cs-r50") == "LibreL2CSr50.pt"
        assert resolve_model_name("l2cs-r50-gaze") == "LibreL2CSr50.pt"

    def test_lazily_registered_family_receives_default_task_aliases(self, monkeypatch):
        import libreyolo.models as models
        from libreyolo.models.base.model import BaseModel

        class EagerFamily:
            FAMILY = "eager"
            FILENAME_PREFIX = "LibreEager"
            WEIGHT_EXT = ".pt"
            INPUT_SIZES = {"s": 64}
            TASK_INPUT_SIZES = {}
            SUPPORTED_TASKS = ("detect",)
            DEFAULT_TASK = "detect"
            REQUIRE_TASK_SUFFIX = False

        class LazyFamily:
            FAMILY = "lazy"
            FILENAME_PREFIX = "LibreLazy"
            WEIGHT_EXT = ".pt"
            INPUT_SIZES = {"s": 64}
            TASK_INPUT_SIZES = {}
            SUPPORTED_TASKS = ("semantic",)
            DEFAULT_TASK = "semantic"
            REQUIRE_TASK_SUFFIX = False

        registry = [EagerFamily]

        def register_lazy_family():
            registry.append(LazyFamily)
            return LazyFamily

        monkeypatch.setattr(BaseModel, "_registry", registry)
        monkeypatch.setattr(models, "try_ensure_rfdetr", register_lazy_family)
        monkeypatch.setattr(cli_config, "_CLI_NAME_TO_WEIGHTS", {})

        cli_config._build_name_map()

        assert cli_config.resolve_model_name("lazy-s") == "LibreLazys.pt"
        assert cli_config.resolve_model_name("lazy-s-sem") == "LibreLazys.pt"

    def test_case_insensitive(self):
        assert resolve_model_name("YOLOX-S") == "LibreYOLOXs.pt"
        assert resolve_model_name("Yolo9-T") == "LibreYOLO9t.pt"
        assert resolve_model_name("RFDETR-N-SEG") == "LibreRFDETRn-seg.pt"

    def test_local_path_passthrough(self):
        assert resolve_model_name("best.pt") == "best.pt"
        assert (
            resolve_model_name("runs/train/exp/weights/best.pt")
            == "runs/train/exp/weights/best.pt"
        )
        assert resolve_model_name("model.onnx") == "model.onnx"

    def test_unknown_model_passthrough(self):
        assert resolve_model_name("unknown-model") == "unknown-model"

    def test_known_weight_filename_detection(self):
        assert is_known_weight_filename("LibreYOLOXs.pt") is True
        assert is_known_weight_filename("weights/LibreYOLOXs.pt") is True
        assert is_known_weight_filename("LibreRFDETRn-seg.pt") is True
        assert is_known_weight_filename("weights/LibreRFDETRxx-seg.pt") is True
        assert is_known_weight_filename("not-a-real-model.pt") is False


class TestDetectFamilyFromName:
    """Test family detection from CLI model names."""

    def test_yolox_family(self):
        assert detect_family_from_name("yolox-s") == "yolox"
        assert detect_family_from_name("yolox-n") == "yolox"

    def test_yolo9_family(self):
        assert detect_family_from_name("yolo9-m") == "yolo9"
        assert detect_family_from_name("yolo9-t") == "yolo9"

    def test_task_alias_family_uses_real_model_family(self):
        assert detect_family_from_name("rfdetr-n-seg") == "rfdetr"

    def test_local_path_returns_none(self):
        assert detect_family_from_name("best.pt") is None
        assert detect_family_from_name("weights/model.pt") is None

    def test_unknown_returns_none(self):
        assert detect_family_from_name("unknown") is None


class TestDetectFamilyFromModelRef:
    """Test family detection across aliases, filenames, and paths."""

    def test_weight_filename_family(self):
        assert detect_family_from_weight_filename("LibreYOLOXs.pt") == "yolox"
        assert detect_family_from_weight_filename("weights/LibreYOLO9t.pt") == "yolo9"
        assert detect_family_from_weight_filename("LibreRTDETRr18.pt") == "rtdetr"
        assert detect_family_from_weight_filename("LibreDEIMv2atto.pt") == "deimv2"

    def test_model_ref_uses_alias_or_resolved_filename(self):
        assert (
            detect_family_from_model_ref("rtdetr-r18", "LibreRTDETRr18.pt") == "rtdetr"
        )
        assert detect_family_from_model_ref("LibreRTDETRr18.pt") == "rtdetr"
        assert detect_family_from_model_ref("/tmp/models/LibreRTDETRr18.pt") == "rtdetr"
        assert detect_family_from_model_ref("deimv2-atto") == "deimv2"

    def test_unknown_checkpoint_name_returns_none(self):
        assert detect_family_from_model_ref("best.pt") is None


class TestGetTrainConfigClass:
    """Test auto-discovery of config classes from model registry."""

    def test_yolox_returns_yolox_config(self):
        assert get_train_config_class("yolox") is YOLOXConfig

    def test_yolo9_returns_yolo9_config(self):
        assert get_train_config_class("yolo9") is YOLO9Config

    def test_rtdetr_returns_rtdetr_config(self):
        assert get_train_config_class("rtdetr") is RTDETRConfig

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

    def test_rtdetr_uses_family_specific_training_defaults(self):
        diffs = get_family_defaults("rtdetr")
        assert diffs["epochs"] == 72
        assert diffs["batch"] == 4
        assert diffs["optimizer"] == "adamw"
        assert diffs["lr0"] == 0.0001
        assert diffs["scheduler"] == "constant"
        assert diffs["mosaic_prob"] == 0.0
        assert diffs["hsv_prob"] == 0.5
        assert "lr_backbone" not in diffs


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
        params = {
            "size": "m",
            "num_classes": 10,
            "data": "coco.yaml",
            "data_dir": "/tmp",
            "epochs": 50,
        }
        kwargs = build_train_kwargs(params)
        assert "size" not in kwargs
        assert "num_classes" not in kwargs
        assert "data" not in kwargs
        assert "data_dir" not in kwargs
        assert kwargs["epochs"] == 50

    def test_covers_all_config_fields(self):
        """Every non-excluded TrainConfig field is picked up when present.

        Fields whose name is itself a CLI alias key (e.g. the classification
        ``mixup`` field, shadowed by the ``mixup`` -> ``mixup_prob`` detection
        alias) are API-only and intentionally skipped by the CLI builder.
        """
        excluded = {"size", "num_classes", "data", "data_dir"}
        from libreyolo.cli.aliases import TRAIN_ALIASES

        internal_to_cli = {v: k for k, v in TRAIN_ALIASES.items()}

        params = {}
        base = TrainConfig()
        alias_shadowed = 0
        for f in dc_fields(TrainConfig):
            if f.name in excluded:
                continue
            if f.name in TRAIN_ALIASES:
                alias_shadowed += 1
                continue
            cli_name = internal_to_cli.get(f.name, f.name)
            params[cli_name] = getattr(base, f.name)

        kwargs = build_train_kwargs(params)
        expected_count = len(dc_fields(TrainConfig)) - len(excluded) - alias_shadowed
        assert len(kwargs) == expected_count
        assert "mixup" not in kwargs

    def test_unknown_params_ignored(self):
        """Params not in TrainConfig are silently dropped."""
        params = {"epochs": 100, "pretrained": True, "val": True, "unknown": "x"}
        kwargs = build_train_kwargs(params)
        assert kwargs["epochs"] == 100
        assert "pretrained" not in kwargs
        assert "val" not in kwargs
        assert "unknown" not in kwargs

    def test_rfdetr_direct_mapping_keeps_single_cls(self, tmp_path):
        kwargs = cli_config._build_rfdetr_train_kwargs(
            {
                "project": str(tmp_path),
                "name": "single-cls",
                "exist_ok": True,
                "single_cls": True,
            },
            user_provided={"single_cls"},
        )

        assert kwargs["single_cls"] is True

    def test_rfdetr_cli_run_dir_is_not_incremented_twice(self, monkeypatch, tmp_path):
        """The CLI pre-increments and creates the run dir, so the wrapper must
        write into that exact dir instead of incrementing it again."""
        rfdetr_model = pytest.importorskip("libreyolo.models.rfdetr.model")
        from libreyolo.models.rfdetr.config import RFDETRConfig
        from libreyolo.training.trainer import BaseTrainer

        class _SaveDirTrainer:
            """Resolves save_dir exactly like BaseTrainer.setup() on rank 0."""

            def __init__(self, model, wrapper_model=None, **kwargs):
                self.config = RFDETRConfig(
                    data=kwargs["data"],
                    project=kwargs["project"],
                    name=kwargs["name"],
                    exist_ok=kwargs["exist_ok"],
                )

            def train(self):
                return {"save_dir": str(BaseTrainer._get_save_dir(self))}

        monkeypatch.setattr(rfdetr_model, "RFDETRTrainer", _SaveDirTrainer)

        wrapper = rfdetr_model.LibreRFDETR.__new__(rfdetr_model.LibreRFDETR)
        wrapper.model = object()
        wrapper.size = "n"
        wrapper.nb_classes = 2
        wrapper.input_size = 560
        wrapper.task = "detect"

        params = {"project": str(tmp_path), "name": "rfdetr_exp", "exist_ok": False}
        for expected in ("rfdetr_exp", "rfdetr_exp2"):
            kwargs = cli_config._build_rfdetr_train_kwargs(params)
            assert kwargs["output_dir"] == str(tmp_path / expected)
            result = wrapper.train(data="data.yaml", **kwargs)
            assert result["save_dir"] == str(tmp_path / expected)
        assert sorted(p.name for p in tmp_path.iterdir()) == [
            "rfdetr_exp",
            "rfdetr_exp2",
        ]


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
