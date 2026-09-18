"""Pins the declarative augmentation spec to the real pipeline plumbing.

``libreyolo/data/augment/spec.py`` claims, per family, which TrainConfig
augmentation knobs are used, mosaic-gated, or silently ignored. These tests
assert those claims against the actual trainers, transforms, and dataset
wrappers, so the spec (and everything built on it: CLI warnings, docs) cannot
drift from reality without a test failure.

No training runs here — we construct trainers with sentinel config values and
inspect what ``create_transforms()`` / the dataset wrappers actually receive.
"""

import dataclasses

import pytest
import torch.nn as nn

from libreyolo.data.augment.spec import (
    AUG_KNOBS,
    FAMILY_AUG_SUPPORT,
    FAMILY_DISPLAY_NAMES,
    FAMILY_EXTRA_AUG_KNOBS,
    GATED_BY_MOSAIC,
    IGNORED,
    USED,
    ignored_aug_params,
    mixup_gating_warning,
    uses_mosaic_gating,
)
from libreyolo.training.config import TrainConfig

pytestmark = pytest.mark.unit


# Sentinel values no family uses as a default, so a match proves plumbing.
FLIP = 0.123
HSV = 0.456
FLIPUD = 0.789
DEGREES = 1.5
TRANSLATE = 0.21
MOSAIC_SCALE = (0.4, 1.6)
MIXUP_SCALE = (0.6, 1.4)
SHEAR = 0.77
PERSPECTIVE = 0.0003


class _DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 3, 1)

    def forward(self, x):  # pragma: no cover - never called here
        return self.conv(x)


def _make_trainer(trainer_cls, **kwargs):
    kwargs.setdefault("device", "cpu")
    return trainer_cls(model=_DummyModel(), **kwargs)


class _DummyDataset:
    """Minimal stand-in for the wrapper constructors (never iterated)."""

    def __len__(self):
        return 4


# =========================================================================
# Spec self-consistency
# =========================================================================


def test_every_family_table_covers_every_knob_with_valid_statuses():
    valid = {USED, GATED_BY_MOSAIC, IGNORED}
    for family, table in FAMILY_AUG_SUPPORT.items():
        assert set(table) == set(AUG_KNOBS), (
            f"{family}: spec table must cover exactly the knobs in AUG_KNOBS"
        )
        for knob, support in table.items():
            assert support.status in valid, f"{family}.{knob}: {support.status}"


def test_aug_knobs_are_trainconfig_fields():
    field_names = {f.name for f in dataclasses.fields(TrainConfig)}
    missing = set(AUG_KNOBS) - field_names
    assert not missing, f"Spec knobs not on TrainConfig: {sorted(missing)}"


def test_spec_families_have_display_names():
    assert set(FAMILY_AUG_SUPPORT) == set(FAMILY_DISPLAY_NAMES)


def test_extra_knobs_exist_on_family_configs():
    from libreyolo.training.config import (
        DFINEConfig,
        ECConfig,
        YOLO9Config,
        YOLONASConfig,
    )
    from libreyolo.models.yolo9_e2e.config import YOLO9E2EConfig
    from libreyolo.models.yolo9_p2.config import YOLO9P2Config

    config_classes = {
        "yolo9": YOLO9Config,
        "yolo9_e2e": YOLO9E2EConfig,
        "yolo9_p2": YOLO9P2Config,
        "dfine": DFINEConfig,
        "yolonas": YOLONASConfig,
    }
    # EC's seg/pose knobs live on the task-specific config subclasses.
    from libreyolo.training.config import ECPoseConfig, ECSegConfig

    for family, extras in FAMILY_EXTRA_AUG_KNOBS.items():
        if family == "rfdetr":
            rf_config = pytest.importorskip("libreyolo.models.rfdetr.config")
            fields = {f.name for f in dataclasses.fields(rf_config.RFDETRConfig)}
        elif family == "ec":
            fields = {f.name for f in dataclasses.fields(ECConfig)}
            fields |= {f.name for f in dataclasses.fields(ECSegConfig)}
            fields |= {f.name for f in dataclasses.fields(ECPoseConfig)}
        elif family == "yolonas":
            from libreyolo.training.config import YOLONASPoseConfig

            fields = {f.name for f in dataclasses.fields(YOLONASConfig)}
            fields |= {f.name for f in dataclasses.fields(YOLONASPoseConfig)}
        else:
            fields = {f.name for f in dataclasses.fields(config_classes[family])}
        missing = set(extras) - fields
        assert not missing, f"{family}: extra knobs not on config: {sorted(missing)}"


# Families that are trainable but predate the TRAIN_CONFIG "machine-readable
# trainable flag" convention (their model classes don't declare it, so they
# cannot be discovered from the registry). If you set TRAIN_CONFIG on one of
# these, remove it here — a test below enforces that this list stays minimal.
_TRAINABLE_WITHOUT_TRAIN_CONFIG = {"dfine", "ec", "yolonas", "segformer"}


def test_spec_covers_every_trainable_family():
    """Every family the CLI can train must have a spec entry.

    Families are discovered through the same path the CLI uses
    (``cli.config._iter_model_classes``: the registry plus the lazy optional
    families), so anything the CLI can see is enforced here — including a
    future lazy ``_ensure_*`` family, which must be added to that iterator for
    the CLI to work and is then automatically required to have a spec entry.
    The required set is every discovered family whose model class declares
    TRAIN_CONFIG (the machine-readable trainable flag), plus the explicit
    ``_TRAINABLE_WITHOUT_TRAIN_CONFIG`` legacy list.
    """
    from libreyolo.cli.config import _iter_model_classes

    classes = list(_iter_model_classes())
    registered = {cls.FAMILY for cls in classes if cls.FAMILY}
    # rfdetr / dinov2 may be absent when their optional deps are not
    # installed; tolerate their spec entries in that case (this leniency is
    # scoped to exactly these known-optional families).
    known = registered | {"rfdetr", "dinov2"}
    unknown = set(FAMILY_AUG_SUPPORT) - known
    assert not unknown, f"Spec families not in the model registry: {sorted(unknown)}"

    declared_trainable = {
        cls.FAMILY
        for cls in classes
        if cls.FAMILY and getattr(cls, "TRAIN_CONFIG", None) is not None
    }
    required = declared_trainable | _TRAINABLE_WITHOUT_TRAIN_CONFIG
    missing = required - set(FAMILY_AUG_SUPPORT)
    assert not missing, (
        f"Trainable families missing from the augmentation spec: {sorted(missing)}. "
        "Classify their knobs in libreyolo/data/augment/spec.py."
    )

    # Self-cleaning: entries in the legacy list must actually lack
    # TRAIN_CONFIG. When a family gains the flag, drop it from the list so
    # coverage is registry-derived for it too.
    stale = _TRAINABLE_WITHOUT_TRAIN_CONFIG & declared_trainable
    assert not stale, (
        f"{sorted(stale)} now declare TRAIN_CONFIG; remove them from "
        "_TRAINABLE_WITHOUT_TRAIN_CONFIG."
    )


# =========================================================================
# Transform-level plumbing: knobs the spec claims USED must land on the
# transform; knobs claimed IGNORED must not.
# =========================================================================


def test_yolox_transform_plumbing():
    from libreyolo.data.augment.yolox import MosaicMixupDataset
    from libreyolo.models.yolox.trainer import YOLOXTrainer

    trainer = _make_trainer(
        YOLOXTrainer, flip_prob=FLIP, hsv_prob=HSV, flipud=FLIPUD
    )
    preproc, wrapper_cls = trainer.create_transforms()
    assert preproc.flip_prob == FLIP
    assert preproc.hsv_prob == HSV
    assert preproc.flipud == FLIPUD
    assert wrapper_cls is MosaicMixupDataset


def test_yolo9_transform_plumbing():
    from libreyolo.data.augment.yolo9 import YOLO9MosaicMixupDataset
    from libreyolo.models.yolo9.trainer import YOLO9Trainer

    trainer = _make_trainer(
        YOLO9Trainer, flip_prob=FLIP, hsv_prob=HSV, flipud=FLIPUD, rot90=0.25
    )
    preproc, wrapper_cls = trainer.create_transforms()
    assert preproc.flip_prob == FLIP
    assert preproc.hsv_prob == HSV
    assert preproc.vertical_flip_prob == FLIPUD
    assert preproc.rot90_prob == 0.25
    assert wrapper_cls is YOLO9MosaicMixupDataset


def test_yolo7_transform_plumbing():
    from libreyolo.data.augment.yolo9 import YOLO9MosaicMixupDataset
    from libreyolo.models.yolo7.trainer import YOLOv7Trainer

    trainer = _make_trainer(
        YOLOv7Trainer, flip_prob=FLIP, hsv_prob=HSV, flipud=FLIPUD
    )
    preproc, wrapper_cls = trainer.create_transforms()
    assert preproc.flip_prob == FLIP
    assert preproc.hsv_prob == HSV
    assert preproc.vertical_flip_prob == FLIPUD
    assert wrapper_cls is YOLO9MosaicMixupDataset


def test_yolonas_transform_plumbing():
    from libreyolo.data.augment.yolonas import YOLONASAffineMixupDataset
    from libreyolo.models.yolonas.trainer import YOLONASTrainer

    trainer = _make_trainer(
        YOLONASTrainer, flip_prob=FLIP, hsv_prob=HSV, flipud=FLIPUD
    )
    preproc, wrapper_cls = trainer.create_transforms()
    assert preproc.flip_prob == FLIP
    assert preproc.hsv_prob == HSV
    assert preproc.flipud == FLIPUD
    assert wrapper_cls is YOLONASAffineMixupDataset


def test_rtmdet_ignores_flipud_as_specced():
    from libreyolo.models.rtmdet.trainer import RTMDetTrainer

    assert FAMILY_AUG_SUPPORT["rtmdet"]["flipud"].status == IGNORED
    trainer = _make_trainer(
        RTMDetTrainer, flip_prob=FLIP, hsv_prob=HSV, flipud=FLIPUD
    )
    preproc, _ = trainer.create_transforms()
    assert preproc.flip_prob == FLIP
    assert preproc.hsv_prob == HSV
    # flipud never reaches the transform (stays at the class default 0.0).
    assert preproc.flipud == 0.0


def test_picodet_ignores_flipud_as_specced():
    from libreyolo.models.picodet.trainer import PICODETTrainer

    assert FAMILY_AUG_SUPPORT["picodet"]["flipud"].status == IGNORED
    trainer = _make_trainer(
        PICODETTrainer, flip_prob=FLIP, hsv_prob=HSV, flipud=FLIPUD
    )
    preproc, _ = trainer.create_transforms()
    assert preproc.flip_prob == FLIP
    assert preproc.hsv_prob == HSV
    assert not hasattr(preproc, "flipud")


def test_rtdetr_ignores_flipud_as_specced():
    from libreyolo.models.rtdetr.trainer import RTDETRTrainer

    assert FAMILY_AUG_SUPPORT["rtdetr"]["flipud"].status == IGNORED
    trainer = _make_trainer(
        RTDETRTrainer, flip_prob=FLIP, hsv_prob=HSV, flipud=FLIPUD
    )
    preproc, _ = trainer.create_transforms()
    assert preproc.flip_prob == FLIP
    assert preproc.hsv_prob == HSV
    # flipud never reaches the transform (stays at the class default 0.0).
    assert preproc.vertical_flip_prob == 0.0


@pytest.mark.parametrize("family", ["dfine", "deim", "deimv2", "ec"])
def test_detr_families_use_flip_and_ignore_hsv(family):
    trainer_cls = {
        "dfine": ("libreyolo.models.dfine.trainer", "DFINETrainer"),
        "deim": ("libreyolo.models.deim.trainer", "DEIMTrainer"),
        "deimv2": ("libreyolo.models.deimv2.trainer", "DEIMv2Trainer"),
        "ec": ("libreyolo.models.ec.trainer", "ECTrainer"),
    }[family]
    module = pytest.importorskip(trainer_cls[0])
    trainer = _make_trainer(
        getattr(module, trainer_cls[1]), flip_prob=FLIP, hsv_prob=HSV
    )
    preproc, wrapper_cls = trainer.create_transforms()
    assert preproc.flip_prob == FLIP
    # The spec says hsv_prob never reaches the DETR-style pipeline (except EC
    # pose, which is a different trainer): the transform has no such knob.
    assert not hasattr(preproc, "hsv_prob")

    # The pass-through wrapper drops every mosaic/affine knob it is handed.
    wrapper = wrapper_cls(
        dataset=_DummyDataset(),
        img_size=(64, 64),
        mosaic=True,
        preproc=preproc,
        degrees=DEGREES,
        translate=TRANSLATE,
        mosaic_scale=MOSAIC_SCALE,
        mixup_scale=MIXUP_SCALE,
        shear=SHEAR,
        enable_mixup=True,
        mosaic_prob=0.5,
        mixup_prob=0.5,
        perspective=PERSPECTIVE,
    )
    for knob in ("degrees", "translate", "shear", "perspective", "mixup_prob",
                 "mosaic_prob", "enable_mosaic", "enable_mixup"):
        assert not hasattr(wrapper, knob), f"{family} wrapper kept {knob}"


# =========================================================================
# Wrapper-level plumbing: mosaic gating vs always-on affine
# =========================================================================


def test_yolox_wrapper_receives_geometry_and_gating_knobs():
    from libreyolo.data.augment.yolox import MosaicMixupDataset

    wrapper = MosaicMixupDataset(
        dataset=_DummyDataset(),
        img_size=(64, 64),
        mosaic=True,
        preproc=object(),
        degrees=DEGREES,
        translate=TRANSLATE,
        mosaic_scale=MOSAIC_SCALE,
        mixup_scale=MIXUP_SCALE,
        shear=SHEAR,
        enable_mixup=True,
        mosaic_prob=0.5,
        mixup_prob=0.25,
        perspective=PERSPECTIVE,
    )
    assert wrapper.degrees == DEGREES
    assert wrapper.translate == TRANSLATE
    assert wrapper.scale == MOSAIC_SCALE
    assert wrapper.mixup_scale == MIXUP_SCALE
    assert wrapper.shear == SHEAR
    assert wrapper.perspective == PERSPECTIVE
    assert wrapper.enable_mosaic is True
    assert wrapper.mosaic_prob == 0.5
    assert wrapper.mixup_prob == 0.25


def test_yolo9_wrapper_receives_copy_paste():
    from libreyolo.data.augment.yolo9 import YOLO9MosaicMixupDataset

    wrapper = YOLO9MosaicMixupDataset(
        dataset=_DummyDataset(),
        img_size=(64, 64),
        preproc=object(),
        copy_paste=0.35,
        copy_paste_mode="mixup",
        perspective=PERSPECTIVE,
    )
    assert wrapper.copy_paste == 0.35
    assert wrapper.copy_paste_mode == "mixup"
    assert wrapper.perspective == PERSPECTIVE


def test_yolonas_wrapper_drops_mosaic_but_keeps_affine_and_mixup():
    from libreyolo.data.augment.yolonas import YOLONASAffineMixupDataset

    wrapper = YOLONASAffineMixupDataset(
        dataset=_DummyDataset(),
        img_size=(64, 64),
        mosaic=True,
        preproc=object(),
        degrees=DEGREES,
        translate=TRANSLATE,
        mosaic_scale=MOSAIC_SCALE,
        mixup_scale=MIXUP_SCALE,
        shear=SHEAR,
        enable_mixup=True,
        mosaic_prob=0.5,
        mixup_prob=0.25,
        perspective=PERSPECTIVE,
    )
    # Spec: mosaic_prob is IGNORED for YOLO-NAS (per-sample affine instead).
    assert not hasattr(wrapper, "mosaic_prob")
    assert not hasattr(wrapper, "enable_mosaic")
    # Spec: affine + mixup knobs are USED, independent of mosaic.
    assert wrapper.enable_affine is True
    assert wrapper.degrees == DEGREES
    assert wrapper.translate == TRANSLATE
    assert wrapper.scale == MOSAIC_SCALE
    assert wrapper.shear == SHEAR
    assert wrapper.perspective == PERSPECTIVE
    assert wrapper.enable_mixup is True
    assert wrapper.mixup_prob == 0.25


def test_mosaic_gating_flags_match_wrapper_classes():
    """Families claiming GATED_BY_MOSAIC must use a mosaic-branch wrapper."""
    from libreyolo.data.augment.yolo9 import YOLO9MosaicMixupDataset
    from libreyolo.data.augment.yolox import MosaicMixupDataset
    from libreyolo.models.rtdetr.trainer import RTDETRTrainer
    from libreyolo.models.rtmdet.trainer import RTMDetTrainer
    from libreyolo.models.yolo9.trainer import YOLO9Trainer
    from libreyolo.models.yolonas.trainer import YOLONASTrainer
    from libreyolo.models.yolox.trainer import YOLOXTrainer

    gated_wrappers = (MosaicMixupDataset, YOLO9MosaicMixupDataset)
    for family, trainer_cls in {
        "yolox": YOLOXTrainer,
        "yolo9": YOLO9Trainer,
        "rtdetr": RTDETRTrainer,
        "rtmdet": RTMDetTrainer,
    }.items():
        _, wrapper_cls = _make_trainer(trainer_cls).create_transforms()
        assert uses_mosaic_gating(family), family
        assert wrapper_cls in gated_wrappers, family

    _, nas_wrapper = _make_trainer(YOLONASTrainer).create_transforms()
    assert not uses_mosaic_gating("yolonas")
    assert nas_wrapper not in gated_wrappers


# =========================================================================
# Classification pack plumbing
# =========================================================================


def test_classify_pipeline_accepts_the_classification_pack():
    from libreyolo.data.classify_dataset import (
        build_classify_collate,
        build_classify_transforms,
        classify_collate_fn,
    )

    train_tf = build_classify_transforms(
        64, augment=True, auto_augment="randaugment", erasing=0.4
    )
    op_names = [type(op).__name__ for op in train_tf.transforms]
    assert "RandAugment" in op_names
    assert "RandomErasing" in op_names

    plain = build_classify_collate(10, mixup=0.0, cutmix=0.0)
    assert plain is classify_collate_fn
    mixer = build_classify_collate(10, mixup=0.3, cutmix=0.2)
    assert mixer is not classify_collate_fn


# =========================================================================
# Spec-driven CLI warnings
# =========================================================================


def test_get_unsupported_train_params_is_spec_driven():
    from libreyolo.cli.config import get_unsupported_train_params

    dfine = get_unsupported_train_params("dfine")
    assert {"mosaic", "mixup", "hsv_prob", "degrees", "translate", "shear",
            "mosaic_scale", "mixup_scale", "flipud", "perspective"} <= dfine
    assert "flip_prob" not in dfine
    assert "no_aug_epochs" not in dfine

    yolox = get_unsupported_train_params("yolox")
    assert "mosaic" not in yolox
    assert "mixup" not in yolox
    assert "flip_prob" not in yolox

    yolonas = get_unsupported_train_params("yolonas")
    assert "mosaic" in yolonas
    assert "mixup" not in yolonas

    resnet = get_unsupported_train_params("resnet")
    assert {"mosaic", "mixup", "flip_prob", "hsv_prob"} <= resnet

    assert get_unsupported_train_params(None) == set()
    assert get_unsupported_train_params("no_such_family") == set()


def test_get_unsupported_train_params_rfdetr_keeps_class_level_ignores():
    from libreyolo.cli.config import get_unsupported_train_params
    from libreyolo.models import try_ensure_rfdetr

    if try_ensure_rfdetr() is None:
        pytest.skip("RF-DETR optional dependencies not installed")
    rf = get_unsupported_train_params("rfdetr")
    # Spec-derived augmentation ignores plus the class-declared extras.
    assert {"mosaic", "mixup", "hsv_prob", "degrees", "translate", "shear",
            "mosaic_scale", "mixup_scale", "optimizer", "momentum",
            "nesterov"} <= rf
    assert "pretrained" not in rf
    assert "flip_prob" not in rf


def _make_train_app():
    import typer

    from libreyolo.cli.commands.train import train_cmd
    from libreyolo.cli.parsing import KeyValueCommand

    app = typer.Typer()
    app.command("train", cls=KeyValueCommand)(train_cmd)
    return app


def test_cli_warns_when_family_ignores_an_explicit_param(caplog):
    import logging

    from typer.testing import CliRunner

    with caplog.at_level(logging.INFO):
        result = CliRunner().invoke(
            _make_train_app(),
            ["data=coco8.yaml", "model=dfine-s", "mosaic=0.5", "--dry-run"],
        )
    assert result.exit_code == 0
    assert "D-FINE ignores these parameters" in caplog.text
    assert "mosaic" in caplog.text


def test_cli_warns_yolonas_ignores_mosaic_but_not_mixup(caplog):
    import logging

    from typer.testing import CliRunner

    with caplog.at_level(logging.INFO):
        result = CliRunner().invoke(
            _make_train_app(),
            ["data=coco8.yaml", "model=yolonas-s", "mosaic=0.5", "mixup=0.5", "--dry-run"],
        )
    assert result.exit_code == 0
    warn_lines = [
        line for line in caplog.text.splitlines()
        if "ignores these parameters" in line
    ]
    assert warn_lines, caplog.text
    assert "YOLO-NAS ignores these parameters: mosaic" in warn_lines[0]
    assert "mixup" not in warn_lines[0]


def test_cli_does_not_warn_when_family_uses_the_param(caplog):
    import logging

    from typer.testing import CliRunner

    with caplog.at_level(logging.INFO):
        result = CliRunner().invoke(
            _make_train_app(),
            ["data=coco8.yaml", "model=yolox-s", "mosaic=0.5", "mixup=0.5", "--dry-run"],
        )
    assert result.exit_code == 0
    assert "ignores these parameters" not in caplog.text


# =========================================================================
# Mosaic-gated mixup warning
# =========================================================================


def test_mixup_gating_warning_fires_only_when_it_should():
    # Gated family, mixup on, mosaic off -> warn.
    msg = mixup_gating_warning("yolox", mosaic_prob=0.0, mixup_prob=1.0)
    assert msg is not None and "mosaic_prob" in msg and "YOLOX" in msg
    # Mosaic on -> mixup can fire -> no warning.
    assert mixup_gating_warning("yolox", 1.0, 1.0) is None
    # Mixup off -> nothing to warn about.
    assert mixup_gating_warning("yolo9", 0.0, 0.0) is None
    # YOLO-NAS mixup is independent of mosaic -> never this warning.
    assert mixup_gating_warning("yolonas", 0.0, 1.0) is None
    # DETR-style families have no mixup at all -> never this warning.
    assert mixup_gating_warning("dfine", 0.0, 1.0) is None
    # Unknown family -> stay quiet.
    assert mixup_gating_warning("no_such_family", 0.0, 1.0) is None


def test_ignored_aug_params_returns_config_field_names():
    assert "mosaic_prob" in ignored_aug_params("dfine")
    assert "flip_prob" not in ignored_aug_params("dfine")
    assert ignored_aug_params("yolox") == {
        "auto_augment", "erasing", "mixup", "cutmix", "scale", "crop_pct"
    }
    assert ignored_aug_params(None) == set()


def test_spec_module_is_torch_free():
    """The spec must stay importable without torch (docs/tooling use it)."""
    import importlib
    import sys

    spec_module = importlib.import_module("libreyolo.data.augment.spec")
    source = open(spec_module.__file__, encoding="utf-8").read()
    assert "import torch" not in source
    assert "import cv2" not in source
    assert "libreyolo.data.augment.spec" in sys.modules
