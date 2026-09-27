"""Tests for ``model.train(cfg=...)`` yaml loading."""

import pytest

from libreyolo.models.base.model import _wrap_train_with_cfg
from libreyolo.training.config import load_train_cfg

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# load_train_cfg
# ---------------------------------------------------------------------------


def test_load_train_cfg_basic(tmp_path):
    cfg = tmp_path / "train.yaml"
    cfg.write_text("epochs: 100\nbatch: 16\nlr0: 0.005\n")
    assert load_train_cfg(cfg) == {"epochs": 100, "batch": 16, "lr0": 0.005}


def test_load_train_cfg_passes_keys_through_unchanged(tmp_path):
    cfg = tmp_path / "train.yaml"
    cfg.write_text(
        "mosaic_prob: 1.0\nflip_prob: 0.5\nhsv_prob: 0.5\nmixup_prob: 0.0\n"
    )
    assert load_train_cfg(cfg) == {
        "mosaic_prob": 1.0,
        "flip_prob": 0.5,
        "hsv_prob": 0.5,
        "mixup_prob": 0.0,
    }


def test_load_train_cfg_empty_yaml(tmp_path):
    cfg = tmp_path / "empty.yaml"
    cfg.write_text("")
    assert load_train_cfg(cfg) == {}


def test_load_train_cfg_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="not found"):
        load_train_cfg(tmp_path / "does_not_exist.yaml")


def test_load_train_cfg_not_a_mapping(tmp_path):
    cfg = tmp_path / "list.yaml"
    cfg.write_text("- a\n- b\n")
    with pytest.raises(ValueError, match="must be a yaml mapping"):
        load_train_cfg(cfg)


def test_load_train_cfg_accepts_str_path(tmp_path):
    cfg = tmp_path / "train.yaml"
    cfg.write_text("epochs: 50\n")
    assert load_train_cfg(str(cfg)) == {"epochs": 50}


# ---------------------------------------------------------------------------
# _wrap_train_with_cfg
# ---------------------------------------------------------------------------


class _FakeWrapper:
    """Stand-in for a family wrapper class — we only need ``self`` shape."""

    pass


def _make_fake_train(captured: dict):
    def train(self, data, *, epochs=10, batch=8, **kwargs):
        captured["data"] = data
        captured["epochs"] = epochs
        captured["batch"] = batch
        captured["kwargs"] = dict(kwargs)
        return {"ok": True}

    return train


def test_wrapper_no_cfg_passes_through(tmp_path):
    captured = {}
    wrapped = _wrap_train_with_cfg(_make_fake_train(captured))
    wrapped(_FakeWrapper(), "data.yaml", epochs=42)
    assert captured["data"] == "data.yaml"
    assert captured["epochs"] == 42
    assert captured["batch"] == 8  # default


def test_wrapper_loads_cfg_yaml(tmp_path):
    cfg = tmp_path / "train.yaml"
    cfg.write_text("epochs: 100\nbatch: 32\n")
    captured = {}
    wrapped = _wrap_train_with_cfg(_make_fake_train(captured))
    wrapped(_FakeWrapper(), "data.yaml", cfg=str(cfg))
    assert captured["epochs"] == 100
    assert captured["batch"] == 32


def test_wrapper_user_kwargs_win_over_cfg(tmp_path):
    cfg = tmp_path / "train.yaml"
    cfg.write_text("epochs: 100\nbatch: 32\n")
    captured = {}
    wrapped = _wrap_train_with_cfg(_make_fake_train(captured))
    wrapped(_FakeWrapper(), "data.yaml", cfg=str(cfg), epochs=200)
    assert captured["epochs"] == 200  # user wins
    assert captured["batch"] == 32  # cfg fills the rest


def test_wrapper_unknown_keys_flow_through_kwargs(tmp_path):
    cfg = tmp_path / "train.yaml"
    cfg.write_text("epochs: 50\nmosaic_prob: 0.7\nlr0: 0.003\n")
    captured = {}
    wrapped = _wrap_train_with_cfg(_make_fake_train(captured))
    wrapped(_FakeWrapper(), "data.yaml", cfg=str(cfg))
    assert captured["epochs"] == 50
    assert captured["kwargs"] == {"mosaic_prob": 0.7, "lr0": 0.003}


def test_wrapper_drops_keys_consumed_positionally(tmp_path):
    """If user passes ``data`` positionally, cfg's ``data`` key must be dropped
    so the inner call doesn't raise ``TypeError: got multiple values``."""
    cfg = tmp_path / "train.yaml"
    cfg.write_text("data: from_cfg.yaml\nepochs: 99\n")
    captured = {}
    wrapped = _wrap_train_with_cfg(_make_fake_train(captured))
    wrapped(_FakeWrapper(), "from_arg.yaml", cfg=str(cfg))
    assert captured["data"] == "from_arg.yaml"
    assert captured["epochs"] == 99


def test_wrapper_marks_function_as_wrapped(tmp_path):
    captured = {}
    wrapped = _wrap_train_with_cfg(_make_fake_train(captured))
    assert getattr(wrapped, "_libreyolo_cfg_wrapped", False) is True


def test_wrapper_missing_cfg_file_raises(tmp_path):
    captured = {}
    wrapped = _wrap_train_with_cfg(_make_fake_train(captured))
    with pytest.raises(FileNotFoundError):
        wrapped(_FakeWrapper(), "data.yaml", cfg=str(tmp_path / "nope.yaml"))


def test_wrapper_drops_size_and_num_classes_from_cfg(tmp_path):
    """``size`` and ``num_classes`` come from the wrapper instance — not the
    yaml. Every family's train() spreads ``**kwargs`` into a trainer call that
    already has ``size=self.size`` and ``num_classes=self.nb_classes``, so
    forwarding these from cfg would raise ``TypeError: got multiple values``.

    Regression for: a yaml produced by ``TrainConfig().to_yaml(...)`` carrying
    ``size: s`` and ``num_classes: 80`` was crashing every family's train.
    """
    cfg = tmp_path / "train.yaml"
    cfg.write_text("size: s\nnum_classes: 80\nepochs: 50\n")

    def family_like_train(self, data, *, epochs=10, **kwargs):
        # Mirrors the shape of every family's train() body:
        #     trainer = FooTrainer(size=self.size, num_classes=self.nb_classes,
        #                          data=data, epochs=epochs, ..., **kwargs)
        return _trainer_call(
            size=self.size,
            num_classes=self.nb_classes,
            data=data,
            epochs=epochs,
            **kwargs,
        )

    def _trainer_call(**kw):
        return kw

    class _Wrapper:
        size = "m"
        nb_classes = 100

    wrapped = _wrap_train_with_cfg(family_like_train)
    out = wrapped(_Wrapper(), "data.yaml", cfg=str(cfg))
    # Wrapper instance state wins; cfg's size/num_classes were dropped.
    assert out["size"] == "m"
    assert out["num_classes"] == 100
    # Other cfg keys still flow through.
    assert out["epochs"] == 50
    assert out["data"] == "data.yaml"


# ---------------------------------------------------------------------------
# Auto-wrapping is applied to real family classes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "import_path,class_name",
    [
        ("libreyolo.models.yolox.model", "LibreYOLOX"),
        ("libreyolo.models.yolo9.model", "LibreYOLO9"),
        ("libreyolo.models.dfine.model", "LibreDFINE"),
        ("libreyolo.models.deim.model", "LibreDEIM"),
        ("libreyolo.models.deimv2.model", "LibreDEIMv2"),
        ("libreyolo.models.yolonas.model", "LibreYOLONAS"),
        ("libreyolo.models.ec.model", "LibreEC"),
        ("libreyolo.models.picodet.model", "LibrePICODET"),
        ("libreyolo.models.rtdetr.model", "LibreRTDETR"),
        ("libreyolo.models.yolo9_e2e.model", "LibreYOLO9E2E"),
    ],
)
def test_family_train_methods_are_auto_wrapped(import_path, class_name):
    """Every family's ``train`` is decorated by ``__init_subclass__``."""
    module = __import__(import_path, fromlist=[class_name])
    cls = getattr(module, class_name)
    assert getattr(cls.train, "_libreyolo_cfg_wrapped", False) is True, (
        f"{class_name}.train is not cfg-wrapped"
    )


def test_python_mosaic_sets_mosaic_prob_like_the_cli():
    """train(mosaic=0) warned 'Unknown training config keys (ignored)' and kept
    mosaic on, while the CLI maps mosaic to mosaic_prob."""
    import warnings

    from libreyolo.training.config import YOLO9Config

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert YOLO9Config.from_kwargs(mosaic=0).mosaic_prob == 0
    assert YOLO9Config.from_kwargs(mosaic=0.5, mosaic_prob=0.5).mosaic_prob == 0.5
    with pytest.raises(ValueError, match="Conflicting mosaic values"):
        YOLO9Config.from_kwargs(mosaic=0, mosaic_prob=1.0)


@pytest.fixture
def train_config_of(monkeypatch):
    """Run a real train() up to the trainer config, then stop."""
    from libreyolo.training.trainer import BaseTrainer

    class _Built(Exception):
        pass

    real_init = BaseTrainer.__init__

    def init(self, *args, **kwargs):
        real_init(self, *args, **kwargs)
        raise _Built(self.config)

    monkeypatch.setattr(BaseTrainer, "__init__", init)

    def run(model, **kwargs):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error")  # no "Unknown training config keys"
            with pytest.raises(_Built) as built:
                model.train(**kwargs)
        return built.value.args[0]

    return run


@pytest.fixture
def detect_yaml(tmp_path):
    import yaml
    from PIL import Image

    root = tmp_path / "data"
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
        Image.new("RGB", (32, 32)).save(root / "images" / split / "a.jpg")
        (root / "labels" / split / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    path = root / "data.yaml"
    path.write_text(yaml.safe_dump({"path": str(root), "train": "images/train",
                                    "val": "images/val", "names": {0: "a"}}))
    return str(path)


def test_python_train_takes_the_cli_augmentation_spellings(train_config_of, detect_yaml):
    """On detection, mixup set the classification MixUp field and fliplr was
    ignored with a warning; the CLI maps them to mixup_prob and flip_prob."""
    from libreyolo import LibreYOLO9

    config = train_config_of(
        LibreYOLO9(None, size="t", device="cpu"),
        data=detect_yaml, device="cpu", mixup=0.3, fliplr=0.2, mosaic=0,
    )

    assert (config.mixup_prob, config.flip_prob, config.mosaic_prob) == (0.3, 0.2, 0)
    assert config.mixup == 0.0


def test_classification_mixup_stays_the_batch_mixup_field(train_config_of, tmp_path):
    from libreyolo import LibreMobileNetV4

    config = train_config_of(
        LibreMobileNetV4(size="s", device="cpu"),
        data=str(tmp_path), device="cpu", mixup=0.3,
    )

    assert config.mixup == 0.3
    assert config.mixup_prob == LibreMobileNetV4.TRAIN_CONFIG().mixup_prob
