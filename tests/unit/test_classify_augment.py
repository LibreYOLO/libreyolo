"""Classification augmentation recipe (``libreyolo/data/augment/classify.py``).

Pins the contract in ``docs/classification_augmentation.md``: knob names and
defaults, the train/eval op order, that the flip knobs are honoured, that one
object reads the knobs off the config, and that ``no_aug_epochs`` switches
the strong regularizers off through the same worker-safe path as detection.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from torchvision import transforms

from libreyolo.data.augment.classify import (
    DEFAULT_CROP_SCALE,
    DEFAULT_FLIP_PROB,
    ClassifyAugKnobs,
    ClassifyBatchMixer,
    build_classify_collate,
    build_classify_transforms,
    classify_collate_fn,
    normalize_auto_augment,
)

pytestmark = pytest.mark.unit


def _names(compose):
    return [type(op).__name__ for op in compose.transforms]


def _op(compose, kind):
    return next(op for op in compose.transforms if isinstance(op, kind))


def _make_imagefolder(root, n_classes=2, n_per=3, size=48):
    for split in ("train", "val"):
        for ci in range(n_classes):
            d = root / split / f"c{ci}"
            d.mkdir(parents=True, exist_ok=True)
            base = np.zeros((size, size, 3), dtype=np.uint8)
            base[:, :, ci % 3] = 200
            for j in range(n_per):
                Image.fromarray(base).save(d / f"{j}.png")


# ---------------------------------------------------------------------------
# Module home
# ---------------------------------------------------------------------------


class TestModuleHome:
    def test_dataset_module_reexports_the_recipe(self):
        import libreyolo.data.classify_dataset as ds
        from libreyolo import data
        from libreyolo.data.augment import classify as recipe

        for name in (
            "build_classify_transforms",
            "build_classify_collate",
            "classify_collate_fn",
            "normalize_crop_scale",
        ):
            assert getattr(ds, name) is getattr(recipe, name)
            assert getattr(data, name) is getattr(recipe, name)
        for name in ("AUTO_AUGMENT_POLICIES", "DEFAULT_CROP_PCT", "DEFAULT_CROP_SCALE", "ClassifyAugKnobs"):
            assert getattr(ds, name) is getattr(recipe, name)

    def test_augment_package_core_stays_torch_free(self):
        """The recipe imports torchvision, so the package __init__ must not pull it."""
        from libreyolo.data import augment

        src = Path(augment.__file__).read_text()
        assert "classify" not in src

    def test_dataset_module_owns_no_transform_code(self):
        import libreyolo.data.classify_dataset as ds

        src = Path(ds.__file__).read_text()
        assert "from torchvision import transforms" not in src
        assert "def build_classify_transforms" not in src
        assert "class _ClassifyBatchMixer" not in src


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------


class TestTrainTransform:
    def test_defaults_reproduce_the_historical_pipeline(self):
        assert _names(build_classify_transforms(224, True)) == [
            "RandomResizedCrop",
            "RandomHorizontalFlip",
            "ToTensor",
            "Normalize",
        ]
        t = build_classify_transforms(224, True)
        assert _op(t, transforms.RandomResizedCrop).scale == DEFAULT_CROP_SCALE
        assert _op(t, transforms.RandomHorizontalFlip).p == DEFAULT_FLIP_PROB == 0.5

    @pytest.mark.parametrize("p", [0.1, 0.5, 1.0])
    def test_flip_prob_reaches_the_op(self, p):
        t = build_classify_transforms(224, True, flip_prob=p)
        assert _op(t, transforms.RandomHorizontalFlip).p == p

    def test_flip_prob_zero_removes_the_op(self):
        assert "RandomHorizontalFlip" not in _names(
            build_classify_transforms(224, True, flip_prob=0.0)
        )

    def test_flipud_adds_vertical_flip_after_horizontal(self):
        names = _names(build_classify_transforms(224, True, flipud=0.3))
        assert names.index("RandomVerticalFlip") == names.index("RandomHorizontalFlip") + 1
        t = build_classify_transforms(224, True, flipud=0.3)
        assert _op(t, transforms.RandomVerticalFlip).p == 0.3

    def test_full_order(self):
        names = _names(
            build_classify_transforms(
                224, True, flipud=0.2, auto_augment="randaugment", erasing=0.4
            )
        )
        assert names == [
            "RandomResizedCrop",
            "RandomHorizontalFlip",
            "RandomVerticalFlip",
            "RandAugment",
            "ToTensor",
            "Normalize",
            "RandomErasing",
        ]

    @pytest.mark.parametrize("knob", ["flip_prob", "flipud"])
    @pytest.mark.parametrize("bad", [-0.1, 1.5, "x"])
    def test_invalid_flip_values_raise(self, knob, bad):
        with pytest.raises(ValueError, match=knob):
            build_classify_transforms(224, True, **{knob: bad})

    def test_eval_pipeline_ignores_the_train_knobs(self):
        t = build_classify_transforms(
            224, False, flip_prob=1.0, flipud=1.0, auto_augment="augmix", erasing=0.5
        )
        assert _names(t) == ["Resize", "CenterCrop", "ToTensor", "Normalize"]

    def test_eval_rejects_invalid_crop_pct(self):
        with pytest.raises(ValueError, match="crop_pct"):
            build_classify_transforms(224, False, crop_pct=0.0)


class TestNormalizeAutoAugment:
    @pytest.mark.parametrize("value", [None, "", "none", "None", "NULL", "false"])
    def test_off_spellings(self, value):
        assert normalize_auto_augment(value) is None

    @pytest.mark.parametrize("value,expected", [("RandAugment", "randaugment"), (" augmix ", "augmix")])
    def test_case_and_whitespace(self, value, expected):
        assert normalize_auto_augment(value) == expected

    def test_unknown_raises_with_the_valid_list(self):
        with pytest.raises(ValueError, match="randaugment, autoaugment, augmix"):
            normalize_auto_augment("cutout")


# ---------------------------------------------------------------------------
# One place reads the config
# ---------------------------------------------------------------------------


class TestClassifyAugKnobs:
    def test_defaults_are_the_off_values(self):
        k = ClassifyAugKnobs()
        assert k.scale == DEFAULT_CROP_SCALE
        assert k.flip_prob == DEFAULT_FLIP_PROB
        assert k.flipud == 0.0
        assert k.auto_augment is None
        assert k.erasing == 0.0 and k.mixup == 0.0 and k.cutmix == 0.0
        assert not k.has_strong

    def test_reads_every_knob_off_a_config(self):
        cfg = SimpleNamespace(
            scale=0.3,
            flip_prob=0.7,
            flipud=0.1,
            auto_augment="AutoAugment",
            erasing=0.2,
            mixup=0.4,
            cutmix=0.3,
        )
        k = ClassifyAugKnobs.from_config(cfg)
        assert k == ClassifyAugKnobs(
            scale=(0.3, 1.0),
            flip_prob=0.7,
            flipud=0.1,
            auto_augment="autoaugment",
            erasing=0.2,
            mixup=0.4,
            cutmix=0.3,
        )
        assert k.has_strong

    def test_reads_the_real_trainconfig(self):
        from libreyolo.training.config import TrainConfig

        k = ClassifyAugKnobs.from_config(TrainConfig(data="x"))
        assert k == ClassifyAugKnobs()
        k = ClassifyAugKnobs.from_config(
            TrainConfig(data="x", flipud=0.25, cutmix=0.5, scale=(0.2, 0.9))
        )
        assert (k.flipud, k.cutmix, k.scale) == (0.25, 0.5, (0.2, 0.9))

    def test_missing_attributes_fall_back(self):
        assert ClassifyAugKnobs.from_config(object()) == ClassifyAugKnobs()

    @pytest.mark.parametrize(
        "field,bad",
        [
            ("flip_prob", 2.0),
            ("flipud", -1),
            ("erasing", 1.0),
            ("mixup", 1.1),
            ("cutmix", "no"),
            ("scale", 3.0),
            ("auto_augment", "cutout"),
        ],
    )
    def test_invalid_values_fail_before_any_data_is_read(self, field, bad):
        with pytest.raises(ValueError, match=field.split("_")[0]):
            ClassifyAugKnobs.from_config(SimpleNamespace(**{field: bad}))

    def test_kwargs_split_between_transform_and_collate(self):
        k = ClassifyAugKnobs(flipud=0.1, auto_augment="augmix", erasing=0.3, mixup=0.2, cutmix=0.1)
        assert k.transform_kwargs() == {
            "scale": DEFAULT_CROP_SCALE,
            "flip_prob": DEFAULT_FLIP_PROB,
            "flipud": 0.1,
            "auto_augment": "augmix",
            "erasing": 0.3,
        }
        assert k.collate_kwargs() == {"mixup": 0.2, "cutmix": 0.1}
        # Everything the transform builder accepts on the train side is covered.
        accepted = set(inspect.signature(build_classify_transforms).parameters)
        assert set(k.transform_kwargs()) <= accepted

    def test_weak_switches_off_strong_and_keeps_geometry(self):
        k = ClassifyAugKnobs(
            scale=(0.3, 0.9), flip_prob=0.7, flipud=0.1,
            auto_augment="randaugment", erasing=0.4, mixup=0.5, cutmix=0.5,
        )
        w = k.weak()
        assert (w.scale, w.flip_prob, w.flipud) == ((0.3, 0.9), 0.7, 0.1)
        assert (w.auto_augment, w.erasing, w.mixup, w.cutmix) == (None, 0.0, 0.0, 0.0)
        assert not w.has_strong
        assert k.weak().weak() == k.weak()


# ---------------------------------------------------------------------------
# no_aug_epochs: dataset + collate switch-off
# ---------------------------------------------------------------------------


class TestCloseStrongAug:
    def test_dataset_drops_policy_and_erasing_but_keeps_geometry(self, tmp_path):
        from libreyolo.data.classify_dataset import ClassifyDataset

        _make_imagefolder(tmp_path)
        ds = ClassifyDataset(
            tmp_path, "train", imgsz=32, augment=True,
            transform_kwargs={
                "scale": (0.3, 0.9), "flip_prob": 0.7, "flipud": 0.2,
                "auto_augment": "randaugment", "erasing": 0.4,
            },
        )
        assert "RandAugment" in _names(ds.transform) and "RandomErasing" in _names(ds.transform)
        ds.close_strong_aug()
        assert _names(ds.transform) == [
            "RandomResizedCrop",
            "RandomHorizontalFlip",
            "RandomVerticalFlip",
            "ToTensor",
            "Normalize",
        ]
        assert _op(ds.transform, transforms.RandomResizedCrop).scale == (0.3, 0.9)
        assert _op(ds.transform, transforms.RandomHorizontalFlip).p == 0.7
        assert _op(ds.transform, transforms.RandomVerticalFlip).p == 0.2
        img, label = ds[0]
        assert img.shape == (3, 32, 32) and isinstance(label, int)

    def test_dataset_close_is_idempotent_and_a_noop_on_eval(self, tmp_path):
        from libreyolo.data.classify_dataset import ClassifyDataset

        _make_imagefolder(tmp_path)
        ev = ClassifyDataset(tmp_path, "val", imgsz=32, augment=False)
        before = _names(ev.transform)
        ev.close_strong_aug()
        assert _names(ev.transform) == before == ["Resize", "CenterCrop", "ToTensor", "Normalize"]
        tr = ClassifyDataset(tmp_path, "train", imgsz=32, augment=True, transform_kwargs={"erasing": 0.4})
        tr.close_strong_aug()
        once = _names(tr.transform)
        tr.close_strong_aug()
        assert _names(tr.transform) == once

    def test_mixer_produces_hard_labels_after_close(self):
        torch.manual_seed(0)
        batch = [(torch.rand(3, 8, 8), i % 3) for i in range(6)]
        mixer = build_classify_collate(3, mixup=1.0)
        assert isinstance(mixer, ClassifyBatchMixer)
        _, labels, _, _ = mixer(batch)
        assert labels.shape == (6, 3) and labels.dtype.is_floating_point
        mixer.close_strong_aug()
        imgs2, labels2, infos, ids = mixer(batch)
        p_imgs, p_labels, p_infos, p_ids = classify_collate_fn(batch)
        assert torch.equal(imgs2, p_imgs) and torch.equal(labels2, p_labels)
        assert labels2.dtype == torch.long and labels2.shape == (6,)
        assert infos == p_infos and ids == p_ids

    def test_plain_collate_has_nothing_to_close(self):
        assert not hasattr(classify_collate_fn, "close_strong_aug")

    def test_persistent_workers_are_refused_for_the_classification_hooks(self, tmp_path):
        from libreyolo.data.classify_dataset import ClassifyDataset
        from libreyolo.training.trainer import ensure_mutation_reaches_workers

        _make_imagefolder(tmp_path)
        ds = ClassifyDataset(tmp_path, "train", imgsz=32, augment=True)
        mixer = build_classify_collate(2, cutmix=0.5)
        bad = SimpleNamespace(dataset=ds, collate_fn=mixer, num_workers=2, persistent_workers=True)
        with pytest.raises(RuntimeError, match=r"ClassifyDataset\.close_strong_aug"):
            ensure_mutation_reaches_workers(bad, ds, "close_strong_aug")
        with pytest.raises(RuntimeError, match=r"ClassifyBatchMixer\.close_strong_aug"):
            ensure_mutation_reaches_workers(bad, mixer, "close_strong_aug")
        ok = SimpleNamespace(dataset=ds, collate_fn=mixer, num_workers=2, persistent_workers=False)
        ensure_mutation_reaches_workers(ok, ds, "close_strong_aug")
        ensure_mutation_reaches_workers(ok, mixer, "close_strong_aug")


# ---------------------------------------------------------------------------
# Trainer plumbing
# ---------------------------------------------------------------------------


class TestTrainerPlumbing:
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
        return trainer

    def test_knobs_are_read_once_and_reach_dataset_and_collate(self, tmp_path):
        trainer = self._trainer(
            tmp_path, flip_prob=0.9, flipud=0.3, auto_augment="randaugment",
            erasing=0.2, mixup=0.5, cutmix=0.25,
        )
        assert trainer._classify_aug == ClassifyAugKnobs.from_config(trainer.config)
        tf = trainer.train_loader.dataset.transform
        assert _op(tf, transforms.RandomHorizontalFlip).p == 0.9
        assert _op(tf, transforms.RandomVerticalFlip).p == 0.3
        assert "RandAugment" in _names(tf) and "RandomErasing" in _names(tf)
        collate = trainer.train_loader.collate_fn
        assert isinstance(collate, ClassifyBatchMixer)
        assert (collate._mixup_p, collate._cutmix_p) == (0.5, 0.25)

    def test_defaults_build_the_historical_loader(self, tmp_path):
        trainer = self._trainer(tmp_path)
        assert _names(trainer.train_loader.dataset.transform) == [
            "RandomResizedCrop", "RandomHorizontalFlip", "ToTensor", "Normalize",
        ]
        assert trainer.train_loader.collate_fn is classify_collate_fn

    def test_no_aug_hook_closes_strong_aug_for_classification(self, tmp_path):
        trainer = self._trainer(tmp_path, auto_augment="augmix", erasing=0.3, mixup=1.0)
        trainer.on_mosaic_disable()
        tf = trainer.train_loader.dataset.transform
        assert "AugMix" not in _names(tf) and "RandomErasing" not in _names(tf)
        assert trainer.train_loader.collate_fn.enabled is False
        _, labels, _, _ = next(iter(trainer.train_loader))
        assert labels.dtype == torch.long and labels.ndim == 1

    def test_no_aug_hook_is_safe_on_the_plain_loader(self, tmp_path):
        trainer = self._trainer(tmp_path)
        trainer.on_mosaic_disable()
        assert trainer.train_loader.collate_fn is classify_collate_fn

    def test_invalid_knob_fails_at_data_setup(self, tmp_path):
        with pytest.raises(ValueError, match="flipud"):
            self._trainer(tmp_path, flipud=1.5)

    def test_trainer_no_longer_hand_reads_the_knobs(self):
        from libreyolo.training.trainer import BaseTrainer

        src = inspect.getsource(BaseTrainer._setup_classify_data)
        assert "ClassifyAugKnobs.from_config" in src
        for knob in ("auto_augment", "erasing", "mixup", "cutmix", "scale"):
            assert f'getattr(self.config, "{knob}"' not in src

    def test_train_loop_fires_the_hook_at_the_no_aug_boundary(self):
        from libreyolo.training.trainer import BaseTrainer

        src = inspect.getsource(BaseTrainer.train)
        assert "no_aug_start = self.config.epochs - self.config.no_aug_epochs" in src
        assert "self.on_mosaic_disable()" in src


# ---------------------------------------------------------------------------
# Spec stays truthful
# ---------------------------------------------------------------------------


class TestSpecPins:
    @pytest.mark.parametrize("family", ["resnet", "convnext", "convnextv2", "mobilenetv4", "efficientnetv2"])
    def test_classification_families_declare_the_honoured_knobs(self, family):
        from libreyolo.data.augment.spec import FAMILY_AUG_SUPPORT, IGNORED, USED

        sup = FAMILY_AUG_SUPPORT[family]
        for knob in ("flip_prob", "flipud", "auto_augment", "erasing", "mixup", "cutmix", "no_aug_epochs"):
            assert sup[knob].status == USED, knob
        for knob in ("mosaic_prob", "mixup_prob", "hsv_prob", "degrees", "translate", "shear", "perspective"):
            assert sup[knob].status == IGNORED, knob

    def test_detection_families_still_ignore_the_classification_pack(self):
        from libreyolo.data.augment.spec import FAMILY_AUG_SUPPORT, IGNORED

        for knob in ("auto_augment", "erasing", "mixup", "cutmix"):
            assert FAMILY_AUG_SUPPORT["yolo9"][knob].status == IGNORED
