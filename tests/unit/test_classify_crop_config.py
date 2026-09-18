"""Configurable classification crops (#878).

Training `RandomResizedCrop` area range and the deterministic eval
resize/center-crop ratio are user-settable, and default to exactly what they
were before.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image
from torchvision import transforms

from libreyolo.data.classify_dataset import (
    DEFAULT_CROP_PCT,
    DEFAULT_CROP_SCALE,
    build_classify_transforms,
    normalize_crop_scale,
)

pytestmark = pytest.mark.unit


def _op(compose, kind):
    return next(o for o in compose.transforms if isinstance(o, kind))


class TestNormalizeCropScale:
    @pytest.mark.parametrize(
        "value,expected",
        [
            (0.5, (0.5, 1.0)),
            (0.9, (0.9, 1.0)),
            ((0.3, 0.8), (0.3, 0.8)),
            ([0.2, 1.0], (0.2, 1.0)),
            (1.0, (1.0, 1.0)),
        ],
    )
    def test_accepted_forms(self, value, expected):
        assert normalize_crop_scale(value) == expected

    @pytest.mark.parametrize(
        "value", [0.0, -0.1, 1.5, (0.8, 0.3), (0.5, 1.5), (0.1, 0.2, 0.3), ()]
    )
    def test_rejected_forms(self, value):
        with pytest.raises(ValueError):
            normalize_crop_scale(value)


class TestTrainCrop:
    def test_default_is_unchanged(self):
        t = build_classify_transforms(224, augment=True)
        assert _op(t, transforms.RandomResizedCrop).scale == (0.5, 1.0)
        assert DEFAULT_CROP_SCALE == (0.5, 1.0)

    def test_float_is_the_lower_bound(self):
        t = build_classify_transforms(224, augment=True, scale=0.9)
        assert _op(t, transforms.RandomResizedCrop).scale == (0.9, 1.0)

    def test_explicit_pair(self):
        t = build_classify_transforms(224, augment=True, scale=(0.3, 0.8))
        assert _op(t, transforms.RandomResizedCrop).scale == (0.3, 0.8)

    def test_invalid_scale_raises_at_build_time(self):
        with pytest.raises(ValueError, match="0 < min <= max <= 1"):
            build_classify_transforms(224, augment=True, scale=2.0)

    def test_scale_does_not_touch_the_val_pipeline(self):
        t = build_classify_transforms(224, augment=False, scale=0.1)
        assert not any(
            isinstance(o, transforms.RandomResizedCrop) for o in t.transforms
        )


class TestEvalCrop:
    def test_default_ratio_is_unchanged(self):
        t = build_classify_transforms(224, augment=False)
        assert _op(t, transforms.Resize).size == int(224 / DEFAULT_CROP_PCT)
        assert tuple(_op(t, transforms.CenterCrop).size) == (224, 224)
        assert DEFAULT_CROP_PCT == 0.875

    @pytest.mark.parametrize("crop_pct,expected_resize", [(1.0, 224), (0.5, 448)])
    def test_crop_pct_drives_the_resize(self, crop_pct, expected_resize):
        t = build_classify_transforms(224, augment=False, crop_pct=crop_pct)
        assert _op(t, transforms.Resize).size == expected_resize
        assert tuple(_op(t, transforms.CenterCrop).size) == (224, 224)

    def test_crop_pct_one_keeps_the_whole_shorter_side(self):
        """crop_pct=1.0 is the "don't crop away context" case from the issue."""
        img = Image.fromarray(np.zeros((300, 600, 3), dtype=np.uint8))
        out = build_classify_transforms(224, augment=False, crop_pct=1.0)(img)
        assert out.shape == (3, 224, 224)


class TestConfigDefaultsStayInSync:
    def test_trainconfig_scale_matches_the_transform_default(self):
        """TrainConfig duplicates the literal to stay torchvision-free."""
        from libreyolo.training.config import TrainConfig

        assert tuple(TrainConfig(data="x").scale) == DEFAULT_CROP_SCALE

    def test_overrides_default_to_none_so_families_keep_their_value(self):
        from libreyolo.training.config import TrainConfig
        from libreyolo.validation.config import ValidationConfig

        assert TrainConfig(data="x").crop_pct is None
        assert ValidationConfig(data="x").crop_pct is None


# ---------------------------------------------------------------------------
# Plumbing: config -> trainer / validator -> transform
# ---------------------------------------------------------------------------


def _make_imagefolder(root, n_classes=2, n_per=4, size=64):
    for split in ("train", "val"):
        for ci in range(n_classes):
            d = root / split / f"c{ci}"
            d.mkdir(parents=True, exist_ok=True)
            base = np.zeros((size, size, 3), dtype=np.uint8)
            base[:, :, ci % 3] = 200
            for j in range(n_per):
                Image.fromarray(base).save(d / f"{j}.png")


class TestTrainerPlumbing:
    """The knobs must reach the built loader, not just the config object."""

    def _trainer(self, tmp_path, **cfg):
        from libreyolo import LibreMobileNetV4
        from libreyolo.models.mobilenetv4.trainer import MobileNetV4Trainer

        _make_imagefolder(tmp_path / "data")
        model = LibreMobileNetV4(size="s", device="cpu")
        trainer = MobileNetV4Trainer(
            model=model.model,
            wrapper_model=model,
            size="s",
            num_classes=model.nb_classes,
            data=str(tmp_path / "data"),
            imgsz=32,
            batch=2,
            workers=0,
            device="cpu",
            **cfg,
        )
        trainer._setup_classify_data()
        return trainer, model

    def test_scale_reaches_the_train_transform(self, tmp_path):
        trainer, _ = self._trainer(tmp_path, scale=(0.2, 0.7))
        compose = trainer.train_loader.dataset._impl.transform
        assert _op(compose, transforms.RandomResizedCrop).scale == (0.2, 0.7)

    def test_crop_pct_override_wins_over_the_family_value(self, tmp_path):
        trainer, model = self._trainer(tmp_path, crop_pct=1.0)
        assert trainer._effective_crop_pct(model) == 1.0

    def test_family_crop_pct_is_kept_when_unset(self, tmp_path):
        trainer, model = self._trainer(tmp_path)
        assert trainer._effective_crop_pct(model) == model.crop_pct

    def test_invalid_crop_pct_fails_at_data_setup(self, tmp_path):
        """Fail before the first epoch, not after a training run."""
        with pytest.raises(ValueError, match=r"crop_pct must be in \(0, 1\]"):
            self._trainer(tmp_path, crop_pct=0.0)


class TestValidatorPlumbing:
    def _validator(self, tmp_path, **cfg):
        from libreyolo.validation import ValidationConfig
        from libreyolo.validation.classify_validator import ClassifyValidator

        _make_imagefolder(tmp_path / "data")
        config = ValidationConfig(
            data_dir=str(tmp_path / "data"),
            save_dir=str(tmp_path / "runs"),
            verbose=False,
            **cfg,
        )

        class _Model:
            size = "s"
            crop_pct = 0.875
            interpolation = "bilinear"

            def _get_model_name(self):
                return "dummy"

        return ClassifyValidator(_Model(), config=config)

    def test_override_wins_over_the_family_value(self, tmp_path):
        v = self._validator(tmp_path, crop_pct=1.0)
        assert v._dataset_transform_kwargs()["crop_pct"] == 1.0

    def test_family_value_is_kept_when_unset(self, tmp_path):
        v = self._validator(tmp_path)
        assert v._dataset_transform_kwargs()["crop_pct"] == 0.875

    def test_invalid_override_raises(self, tmp_path):
        v = self._validator(tmp_path, crop_pct=1.5)
        with pytest.raises(ValueError, match=r"crop_pct must be in \(0, 1\]"):
            v._dataset_transform_kwargs()


class TestValCliPlumbing:
    """The val CLI must forward crop_pct, not just accept it (#878)."""

    def _run(self, monkeypatch, tmp_path, extra):
        import typer
        from typer.testing import CliRunner

        from libreyolo.cli.commands.val import val_cmd
        from libreyolo.cli.parsing import KeyValueCommand

        captured = {}

        class _ClassifyLike:
            FAMILY = "convnext"
            size = "t"
            task = "classify"
            device = "cpu"

            def val(self, **kwargs):
                captured["kwargs"] = kwargs
                return {"metrics/accuracy_top1": 1.0, "metrics/accuracy_top5": 1.0}

        monkeypatch.setattr(
            "libreyolo.cli.commands.val.load_model_or_exit",
            lambda *a, **k: _ClassifyLike(),
        )
        app = typer.Typer()
        app.command("val", cls=KeyValueCommand)(val_cmd)
        result = CliRunner().invoke(
            app,
            [
                "data=imagenette",
                "model=LibreConvNeXtt-cls.pt",
                *extra,
                f"project={tmp_path}",
                "exist_ok=true",
                "--json",
            ],
        )
        return result, captured

    def test_crop_pct_reaches_val(self, monkeypatch, tmp_path):
        result, captured = self._run(monkeypatch, tmp_path, ["crop_pct=1.0"])
        assert result.exit_code == 0, result.output
        assert captured["kwargs"]["crop_pct"] == pytest.approx(1.0)

    def test_default_is_none_so_the_family_value_is_kept(self, monkeypatch, tmp_path):
        result, captured = self._run(monkeypatch, tmp_path, [])
        assert result.exit_code == 0, result.output
        assert captured["kwargs"]["crop_pct"] is None
