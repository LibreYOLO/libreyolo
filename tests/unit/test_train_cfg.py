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
