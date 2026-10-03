"""YOLO9 training stability (issue #927): gradient clipping and fp32 matching."""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.unit


def test_yolo9_family_configs_clip_gradient_norm_at_10():
    from libreyolo.models.yolo9_e2e.config import YOLO9E2EConfig
    from libreyolo.models.yolo9_p2.config import YOLO9P2Config
    from libreyolo.training.config import YOLO9Config

    for config in (YOLO9Config(), YOLO9P2Config(), YOLO9E2EConfig()):
        assert config.clip_max_norm == 10.0


@pytest.mark.parametrize(
    "kwargs,expected",
    [({}, 10.0), ({"clip_max_norm": 2.5}, 2.5), ({"clip_max_norm": 0}, 0.0)],
)
def test_yolo9_trainer_clip_max_norm_override(kwargs, expected):
    from libreyolo.models.yolo9.trainer import YOLO9Trainer
    from libreyolo.training.config import YOLO9Config

    trainer = object.__new__(YOLO9Trainer)
    trainer.config = YOLO9Config.from_kwargs(**kwargs)
    assert trainer._get_clip_max_norm() == expected
    assert trainer._should_clip_gradients() is (expected > 0)


def test_matcher_assigns_positives_with_fp16_background_logits():
    """Under fp16 the sigmoid of logits below about -17 is exactly 0; the
    matcher must still see fp32 scores and assign positives."""
    from libreyolo.models.yolo9.loss import YOLO9Loss

    assert torch.tensor(-20.0, dtype=torch.float16).sigmoid().item() == 0.0

    loss = YOLO9Loss(
        num_classes=2,
        reg_max=16,
        strides=[8, 16, 32],
        image_size=[64, 64],
        device=torch.device("cpu"),
        distributed_normalize=False,
    )
    raw = [torch.zeros(1, 4 * 16 + 2, s, s, dtype=torch.float16) for s in (8, 4, 2)]
    for level in raw:
        level[:, 4 * 16:] = -20.0
    targets = torch.tensor([[[1.0, 0.25, 0.25, 0.75, 0.75]]])

    seen = {}
    matcher = loss.matcher

    def _spy(target, predict):
        seen["dtypes"] = tuple(t.dtype for t in predict)
        return matcher(target, predict)

    loss.matcher = _spy
    out = loss(raw, targets)
    assert seen["dtypes"] == (torch.float32, torch.float32)
    assert float(out["num_fg"]) > 0
    assert torch.isfinite(out["total_loss"])


# -----------------------------------------------------------------------------
# PGI auxiliary branch defaults
# -----------------------------------------------------------------------------


def _save_yolo9t(tmp_path, *, with_aux, aux_class_neck=None):
    from libreyolo.models.yolo9.nn import YOLO9_CONFIGS, LibreYOLO9Model, YOLO9Head
    from libreyolo.utils.serialization import wrap_libreyolo_checkpoint

    raw = LibreYOLO9Model(config="t", nb_classes=2)
    if with_aux:
        raw.enable_aux(0.25)
        if aux_class_neck is not None:
            raw.aux_head = YOLO9Head(
                YOLO9_CONFIGS["t"]["head_channels"],
                2,
                reg_max=16,
                strides=(8, 16, 32),
                class_neck=aux_class_neck,
            )
    ckpt = wrap_libreyolo_checkpoint(
        raw.state_dict(),
        model_family="yolo9",
        size="t",
        task="detect",
        nc=2,
        names={0: "a", 1: "b"},
        imgsz=640,
    )
    path = tmp_path / ("with_aux.pt" if with_aux else "no_aux.pt")
    torch.save(ckpt, path)
    return path


def _aux_attached_when_training(model, monkeypatch, **train_kwargs):
    """Run ``train()`` up to trainer construction; report whether PGI is attached."""
    seen = {}

    class _Trainer:
        def __init__(self, **kwargs):
            seen["aux"] = kwargs["model"].aux is not None

        def train(self):
            return {}

    monkeypatch.setattr(
        "libreyolo.data.load_data_config",
        lambda *args, **kwargs: {"nc": 2, "names": ["a", "b"]},
    )
    monkeypatch.setattr(type(model), "_trainer_class", lambda self: _Trainer)
    model.train(data="dummy.yaml", device="cpu", **train_kwargs)
    return seen["aux"]


def test_weights_without_pgi_tensors_train_the_main_head_only(tmp_path, monkeypatch):
    from libreyolo.models.yolo9.model import LibreYOLO9

    path = _save_yolo9t(tmp_path, with_aux=False)
    model = LibreYOLO9(str(path), size="t", nb_classes=2, device="cpu")
    assert _aux_attached_when_training(model, monkeypatch) is False


def test_explicit_aux_weight_attaches_pgi_without_pretrained_tensors(tmp_path, monkeypatch):
    from libreyolo.models.yolo9.model import LibreYOLO9

    path = _save_yolo9t(tmp_path, with_aux=False)
    model = LibreYOLO9(str(path), size="t", nb_classes=2, device="cpu")
    assert _aux_attached_when_training(model, monkeypatch, aux_weight=0.25) is True


def test_weights_with_pgi_tensors_attach_pgi_by_default(tmp_path, monkeypatch):
    from libreyolo.models.yolo9.model import LibreYOLO9

    path = _save_yolo9t(tmp_path, with_aux=True)
    model = LibreYOLO9(str(path), size="t", nb_classes=2, device="cpu")
    assert _aux_attached_when_training(model, monkeypatch) is True


def test_from_scratch_training_attaches_pgi_by_default(monkeypatch):
    from libreyolo.models.yolo9.model import LibreYOLO9

    model = LibreYOLO9(None, size="t", nb_classes=2, device="cpu")
    assert _aux_attached_when_training(model, monkeypatch) is True


def test_pgi_head_takes_the_checkpoint_class_tower_width(tmp_path):
    """A PGI head whose class towers are wider than a fresh build loads fully."""
    from libreyolo.models.yolo9.model import LibreYOLO9

    path = _save_yolo9t(tmp_path, with_aux=True, aux_class_neck=128)
    saved = torch.load(path, map_location="cpu", weights_only=False)["model"]
    aux_keys = [k for k in saved if k.startswith(("aux.", "aux_head."))]

    model = LibreYOLO9(str(path), size="t", nb_classes=2, device="cpu")
    model.model.enable_aux(0.25)
    assert model.model.aux_head.class_neck != 128
    loaded = model._reload_aux_from_path(str(path))
    assert model.model.aux_head.class_neck == 128
    assert loaded == len(aux_keys)
