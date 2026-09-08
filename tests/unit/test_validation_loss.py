"""Unit coverage for opt-in validation loss across every trainable group."""

from __future__ import annotations

import logging
import math
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import pytest
import torch

from libreyolo.models.base.classify_validation_loss import ClassifyValidationLoss
from libreyolo.models.convnext.trainer import ConvNeXtTrainer
from libreyolo.models.efficientnetv2.trainer import EfficientNetV2Trainer
from libreyolo.models.mobilenetv4.trainer import MobileNetV4Trainer
from libreyolo.models.resnet.trainer import ResNetTrainer
from libreyolo.models.base.dense_head_validation_loss import DenseHeadValidationLoss
from libreyolo.models.base.detr_validation_loss import DETRValidationLoss
from libreyolo.models.base.semantic_validation_loss import SemanticValidationLoss
from libreyolo.models.base.validation_loss import (
    emit_loss_outputs,
    loss_output_modules,
    padded_targets_to_flat_pixels,
)
from libreyolo.models.deim.trainer import DEIMTrainer
from libreyolo.models.dfine.trainer import DFINETrainer
from libreyolo.models.rfdetr import loss as rfdetr_loss_module
from libreyolo.models.rfdetr.config import RFDETRConfig
from libreyolo.models.rfdetr.loss import SetCriterion
from libreyolo.models.rfdetr.trainer import RFDETRTrainer
from libreyolo.models.rfdetr.validation_loss import RFDETRValidationLoss
from libreyolo.models.rtdetr.trainer import RTDETRTrainer
from libreyolo.models.rtmdet.trainer import RTMDetTrainer
from libreyolo.models.yolo9 import loss as yolo9_loss_module
from libreyolo.models.yolo9.loss import YOLO9Loss
from libreyolo.models.yolo9.trainer import YOLO9Trainer
from libreyolo.models.yolo9.validation_loss import YOLO9ValidationLoss
from libreyolo.models.yolo9_e2e.trainer import YOLO9E2ETrainer
from libreyolo.models.yolo9_p2.trainer import YOLO9P2Trainer
from libreyolo.models.yolonas.trainer import YOLONASTrainer
from libreyolo.training.config import (
    DFINEConfig,
    TrainConfig,
    YOLO9Config,
    YOLONASConfig,
)
from libreyolo.training.loggers.base import epoch_metrics
from libreyolo.training.trainer import BaseTrainer
from libreyolo.ui.train_monitor_page import INDEX_HTML
from libreyolo.validation.classify_validator import ClassifyValidator
from libreyolo.validation.config import ValidationConfig
from libreyolo.validation.detection_validator import DetectionValidator
from libreyolo.validation.loss import ValidationLossMixin

pytestmark = pytest.mark.unit


class _Adapter:
    max_labels = 300

    def __init__(self, *, fail_on: int | None = None):
        self.calls = 0
        self.fail_on = fail_on
        self.image_sizes = []

    def __call__(self, predictions, targets, *, image_size):
        del predictions, targets
        self.calls += 1
        self.image_sizes.append(image_size)
        if self.calls == self.fail_on:
            raise RuntimeError("synthetic adapter failure")
        value = float(self.calls * 2 - 1)
        return {
            "loss": torch.tensor(value),
            "loss/box": value + 1.0,
        }


def _validator(adapter: _Adapter, *, augment: bool = False) -> DetectionValidator:
    model = SimpleNamespace(nb_classes=2)
    config = ValidationConfig(
        data_dir=".",
        device="cpu",
        verbose=False,
        augment=augment,
    )
    return DetectionValidator(model, config, loss_adapter=adapter)


def test_validation_loss_is_opt_in_by_default():
    assert TrainConfig().val_loss is False
    assert YOLO9Config().val_loss is False
    assert RFDETRConfig().val_loss is False
    assert DFINEConfig().val_loss is False
    assert YOLONASConfig().val_loss is False


@pytest.mark.parametrize(
    "trainer_class, match",
    [
        (YOLO9Trainer, "YOLO9 detection only"),
        (YOLO9P2Trainer, "YOLO9 detection only"),
        (YOLO9E2ETrainer, "YOLO9-E2E detection only"),
        (YOLONASTrainer, "YOLO-NAS detection only"),
        (DFINETrainer, "dfine detection only"),
        (DEIMTrainer, "deim detection only"),
        (RTDETRTrainer, "rtdetr detection only"),
    ],
)
def test_non_detection_tasks_reject_validation_loss(trainer_class, match):
    trainer = trainer_class.__new__(trainer_class)
    trainer.config = SimpleNamespace(val_loss=True)
    trainer.wrapper_model = SimpleNamespace(task="segment")
    trainer.model = SimpleNamespace()

    with pytest.raises(ValueError, match=match):
        trainer.validate_validation_loss_config()


def test_base_gate_rejects_a_family_that_has_not_implemented_val_loss():
    """Every trainable family overrides this now, so exercise the mechanism.

    A newly ported model inherits the base gate until it opts in, and must
    fail loudly rather than silently ignore the flag.
    """

    trainer = SimpleNamespace(
        config=SimpleNamespace(val_loss=True),
        get_model_family=lambda: "brandnew",
    )
    trainer.validate_validation_loss_config = (
        BaseTrainer.validate_validation_loss_config.__get__(trainer)
    )

    with pytest.raises(ValueError, match="not supported by brandnew training"):
        trainer.validate_validation_loss_config()


def test_rtmdet_accepts_detection_and_rejects_other_tasks():
    trainer = object.__new__(RTMDetTrainer)
    trainer.config = SimpleNamespace(val_loss=True)

    trainer.wrapper_model = SimpleNamespace(task="detect")
    trainer.validate_validation_loss_config()

    trainer.wrapper_model = SimpleNamespace(task="segment")
    with pytest.raises(ValueError, match="rtmdet detection only"):
        trainer.validate_validation_loss_config()


def test_yolo9_variant_guard_accepts_p2_and_rejects_e2e_head():
    from libreyolo.models.yolo9.nn import LibreYOLO9Model
    from libreyolo.models.yolo9_e2e.nn import LibreYOLO9E2EModel
    from libreyolo.models.yolo9_p2.nn import LibreYOLO9P2Model

    for model_class, size in (
        (LibreYOLO9Model, "t"),
        (LibreYOLO9P2Model, "t"),
    ):
        trainer = YOLO9Trainer.__new__(YOLO9Trainer)
        trainer.config = SimpleNamespace(val_loss=True)
        trainer.wrapper_model = SimpleNamespace(task="detect")
        trainer.model = model_class(config=size, nb_classes=2, img_size=64)
        trainer.validate_validation_loss_config()

    # The E2E model subclasses LibreYOLO9Model but swaps the head, so the
    # shared guard must not claim it.
    trainer = YOLO9Trainer.__new__(YOLO9Trainer)
    trainer.config = SimpleNamespace(val_loss=True)
    trainer.wrapper_model = SimpleNamespace(task="detect")
    trainer.model = LibreYOLO9E2EModel(config="t", nb_classes=2, img_size=64)
    with pytest.raises(ValueError, match="YOLO9 detection only"):
        trainer.validate_validation_loss_config()


def test_rfdetr_non_detection_task_rejects_validation_loss():
    trainer = RFDETRTrainer.__new__(RFDETRTrainer)
    trainer.config = SimpleNamespace(val_loss=True)
    trainer.wrapper_model = SimpleNamespace(task="segment")
    trainer.model = SimpleNamespace()

    with pytest.raises(ValueError, match="RF-DETR detection only"):
        trainer.validate_validation_loss_config()


def test_detection_validator_averages_loss_and_expands_target_capacity():
    adapter = _Adapter()
    validator = _validator(adapter)
    validator.val_preproc = SimpleNamespace(max_labels=120)
    validator._ensure_validation_loss_target_capacity()
    assert validator.val_preproc.max_labels == 300

    images = torch.zeros(2, 3, 32, 48)
    targets = torch.zeros(2, 4, 5)
    validator._update_batch_metrics({}, images, targets)
    validator._update_batch_metrics({}, images, targets)

    assert adapter.image_sizes == [(32, 48), (32, 48)]
    assert validator._validation_loss_metrics() == pytest.approx(
        {
            "metrics/loss": 2.0,
            "metrics/loss/box": 3.0,
        }
    )


def test_detection_validator_discards_partial_loss_after_adapter_failure(caplog):
    adapter = _Adapter(fail_on=2)
    validator = _validator(adapter)
    images = torch.zeros(1, 3, 16, 16)
    targets = torch.zeros(1, 2, 5)

    validator._update_batch_metrics({}, images, targets)
    validator._update_batch_metrics({}, images, targets)
    validator._update_batch_metrics({}, images, targets)

    assert adapter.calls == 2
    assert validator._validation_loss_metrics() == {}
    # The shared mixin names the failing validator's task, so the same
    # sentence serves detect, classify, semantic and restore.
    assert "detect metrics will continue" in caplog.text


def test_validation_loss_rejects_augmented_validation():
    with pytest.raises(ValueError, match="augmented validation"):
        _validator(_Adapter(), augment=True)


def test_yolo9_validation_target_conversion_compacts_and_normalizes():
    targets = torch.tensor(
        [
            [[20.0, 10.0, 60.0, 30.0, 2.0], [0.0, 0.0, 0.0, 0.0, 0.0]],
            [[50.0, 25.0, 150.0, 75.0, 1.0], [0.0, 0.0, 0.0, 0.0, 0.0]],
        ]
    )

    converted = YOLO9ValidationLoss._prepare_targets(
        targets,
        image_size=(100, 200),
        num_classes=3,
        device=torch.device("cpu"),
    )

    assert converted.shape == (2, 1, 5)
    assert converted[0, 0].tolist() == pytest.approx([2.0, 0.1, 0.1, 0.3, 0.3])
    assert converted[1, 0].tolist() == pytest.approx([1.0, 0.25, 0.25, 0.75, 0.75])


def test_yolo9_adapter_reuses_raw_eval_outputs():
    class _Loss:
        def __init__(self):
            self.image_size = None
            self.raw_outputs = None

        def update_anchors(self, image_size):
            self.image_size = image_size

        def __call__(self, raw_outputs, targets):
            self.raw_outputs = raw_outputs
            assert targets.shape == (1, 1, 5)
            return {
                "total_loss": torch.tensor(10.0),
                "box_loss": torch.tensor(4.0),
                "cls_loss": torch.tensor(3.0),
                "dfl_loss": torch.tensor(3.0),
            }

    adapter = object.__new__(YOLO9ValidationLoss)
    adapter.device = torch.device("cpu")
    adapter.num_classes = 2
    adapter.max_labels = 100
    adapter.loss = _Loss()
    raw_outputs = [torch.zeros(1, 66, 4, 4)]
    targets = torch.tensor([[[1.0, 2.0, 5.0, 6.0, 1.0]]])

    values = adapter(
        {"predictions": torch.empty(0), "raw_outputs": raw_outputs},
        targets,
        image_size=(8, 8),
    )

    assert adapter.loss.raw_outputs is raw_outputs
    assert adapter.loss.image_size == [8, 8]
    assert set(values) == {"loss", "loss/box", "loss/cls", "loss/dfl"}


def test_yolo9_rank_local_normalizer_skips_collective(monkeypatch):
    def _unexpected_collective(value):
        del value
        raise AssertionError("rank-local validation entered a collective")

    monkeypatch.setattr(
        yolo9_loss_module, "all_reduce_avg_scalar_tensor", _unexpected_collective
    )
    loss = YOLO9Loss(
        num_classes=2,
        reg_max=16,
        strides=[8, 16, 32],
        image_size=None,
        device=torch.device("cpu"),
        distributed_normalize=False,
    )

    assert loss._global_cls_norm(torch.tensor([2.0, 3.0])) == pytest.approx(5.0)
    assert loss._global_cls_norm(torch.tensor([0.0])) == pytest.approx(1.0)


def test_rfdetr_validation_target_conversion_to_normalized_cxcywh():
    targets = torch.tensor([[[20.0, 10.0, 60.0, 30.0, 2.0], [0.0, 0.0, 0.0, 0.0, 0.0]]])

    converted = RFDETRValidationLoss._prepare_targets(
        targets,
        image_size=(100, 200),
        num_classes=3,
        device=torch.device("cpu"),
    )

    assert len(converted) == 1
    assert converted[0]["labels"].tolist() == [2]
    assert converted[0]["boxes"][0].tolist() == pytest.approx([0.2, 0.2, 0.2, 0.2])


def test_rfdetr_adapter_uses_full_weighted_criterion_output():
    class _Criterion:
        weight_dict = {
            "loss_ce": 2.0,
            "loss_ce_0": 2.0,
            "loss_bbox": 5.0,
            "loss_giou": 2.0,
        }

        def __call__(self, predictions, targets):
            assert predictions["pred_logits"].shape == (1, 4, 3)
            assert targets[0]["labels"].tolist() == [1]
            return {
                "loss_ce": torch.tensor(1.0),
                "loss_ce_0": torch.tensor(2.0),
                "loss_bbox": torch.tensor(3.0),
                "loss_giou": torch.tensor(4.0),
            }

    adapter = object.__new__(RFDETRValidationLoss)
    adapter.device = torch.device("cpu")
    adapter.num_classes = 2
    adapter.criterion = _Criterion()
    predictions = {
        "pred_logits": torch.zeros(1, 4, 3),
        "pred_boxes": torch.zeros(1, 4, 4),
    }
    targets = torch.tensor([[[2.0, 2.0, 6.0, 6.0, 1.0]]])

    values = adapter(predictions, targets, image_size=(8, 8))

    assert float(values["loss"]) == pytest.approx(29.0)
    assert float(values["loss/ce"]) == pytest.approx(6.0)
    assert float(values["loss/bbox"]) == pytest.approx(15.0)
    assert float(values["loss/giou"]) == pytest.approx(8.0)
    assert float(values["loss/ce"] + values["loss/bbox"] + values["loss/giou"]) == (
        pytest.approx(float(values["loss"]))
    )


def _criterion(*, distributed_normalize: bool) -> SetCriterion:
    return SetCriterion(
        num_classes=2,
        matcher=None,
        weight_dict={},
        focal_alpha=0.25,
        losses=[],
        distributed_normalize=distributed_normalize,
    )


def test_rfdetr_rank_local_normalizer_skips_collective(monkeypatch):
    monkeypatch.setattr(
        rfdetr_loss_module, "is_dist_avail_and_initialized", lambda: True
    )
    monkeypatch.setattr(rfdetr_loss_module, "get_world_size", lambda: 4)

    def _unexpected_collective(value):
        del value
        raise AssertionError("rank-local validation entered a collective")

    monkeypatch.setattr(torch.distributed, "all_reduce", _unexpected_collective)
    criterion = _criterion(distributed_normalize=False)
    outputs = {"pred_logits": torch.zeros(1, 1, 3)}
    targets = [{"labels": torch.tensor([0, 1])}]

    assert criterion._box_count_normalizer(outputs, targets, 1) == pytest.approx(2.0)


def test_rfdetr_training_normalizer_keeps_global_average(monkeypatch):
    calls = []

    def _all_reduce(value):
        calls.append(True)
        value.add_(6.0)  # local 2 + other ranks 6 = global 8

    monkeypatch.setattr(
        rfdetr_loss_module, "is_dist_avail_and_initialized", lambda: True
    )
    monkeypatch.setattr(rfdetr_loss_module, "get_world_size", lambda: 4)
    monkeypatch.setattr(torch.distributed, "all_reduce", _all_reduce)
    criterion = _criterion(distributed_normalize=True)
    outputs = {"pred_logits": torch.zeros(1, 1, 3)}
    targets = [{"labels": torch.tensor([0, 1])}]

    assert criterion._box_count_normalizer(outputs, targets, 1) == pytest.approx(2.0)
    assert calls == [True]


def test_flat_pixel_target_conversion_drops_padding_and_keeps_pixels():
    targets = torch.tensor(
        [
            [[20.0, 10.0, 60.0, 30.0, 2.0], [0.0, 0.0, 0.0, 0.0, 0.0]],
            [[50.0, 25.0, 150.0, 75.0, 1.0], [0.0, 0.0, 0.0, 0.0, 0.0]],
        ]
    )

    converted = padded_targets_to_flat_pixels(
        targets,
        num_classes=3,
        device=torch.device("cpu"),
        family="YOLO-NAS",
    )

    assert converted.shape == (2, 6)
    assert converted[0].tolist() == pytest.approx([0.0, 2.0, 40.0, 20.0, 40.0, 20.0])
    assert converted[1].tolist() == pytest.approx([1.0, 1.0, 100.0, 50.0, 100.0, 50.0])


def test_flat_pixel_target_conversion_rejects_out_of_range_class():
    targets = torch.tensor([[[1.0, 1.0, 5.0, 5.0, 9.0]]])

    with pytest.raises(ValueError, match="outside"):
        padded_targets_to_flat_pixels(
            targets,
            num_classes=3,
            device=torch.device("cpu"),
            family="YOLO-NAS",
        )


class _Decoder(torch.nn.Module):
    def __init__(self, *, eval_idx=2, num_layers=3):
        super().__init__()
        self.emit_loss_outputs = False
        self.eval_idx = eval_idx
        self.num_layers = num_layers
        self.weight = torch.nn.Parameter(torch.zeros(1))


class _Model(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.decoder = _Decoder(**kwargs)


def test_emit_loss_outputs_sets_and_restores_the_flag():
    model = _Model()
    assert loss_output_modules(model) == [model.decoder]

    with emit_loss_outputs(model):
        assert model.decoder.emit_loss_outputs is True
    assert model.decoder.emit_loss_outputs is False


def test_emit_loss_outputs_restores_the_flag_after_an_error():
    model = _Model()

    with pytest.raises(RuntimeError):
        with emit_loss_outputs(model):
            raise RuntimeError("boom")

    assert model.decoder.emit_loss_outputs is False


def test_emit_loss_outputs_rejects_a_model_without_the_hook():
    with pytest.raises(TypeError, match="emit_loss_outputs"):
        with emit_loss_outputs(torch.nn.Linear(2, 2)):
            pass


def test_detr_adapter_rejects_decoder_whose_eval_layer_is_not_last():
    criterion = SimpleNamespace(num_classes=2, eval=lambda: None)

    with pytest.raises(TypeError, match="eval_idx=1 of 3 layers"):
        DETRValidationLoss(_Model(eval_idx=1, num_layers=3), criterion)


def _detr_adapter(loss_dict):
    class _Criterion:
        num_classes = 2

        def eval(self):
            return self

        def __call__(self, predictions, targets, **kwargs):
            del predictions, targets, kwargs
            return loss_dict

    adapter = object.__new__(DETRValidationLoss)
    adapter.model = _Model()
    adapter.criterion = _Criterion()
    adapter.device = torch.device("cpu")
    adapter.num_classes = 2
    return adapter


def _scoped_predictions():
    return {
        "pred_logits": torch.zeros(1, 4, 3),
        "pred_boxes": torch.zeros(1, 4, 4),
        "aux_outputs": [],
    }


def test_detr_adapter_groups_decorated_names_into_components():
    adapter = _detr_adapter(
        {
            "loss_vfl": torch.tensor(1.0),
            "loss_vfl_aux_0": torch.tensor(2.0),
            "loss_vfl_pre": torch.tensor(0.5),
            "loss_bbox": torch.tensor(3.0),
            "loss_bbox_enc_0": torch.tensor(1.5),
            "loss_giou_dn_1": torch.tensor(4.0),
        }
    )

    values = adapter(
        _scoped_predictions(),
        torch.tensor([[[2.0, 2.0, 6.0, 6.0, 1.0]]]),
        image_size=(8, 8),
    )

    assert float(values["loss"]) == pytest.approx(12.0)
    assert float(values["loss/vfl"]) == pytest.approx(3.5)
    assert float(values["loss/bbox"]) == pytest.approx(4.5)
    assert float(values["loss/giou"]) == pytest.approx(4.0)
    components = sum(float(v) for k, v in values.items() if k != "loss")
    assert components == pytest.approx(float(values["loss"]))


def test_detr_adapter_ignores_the_criterion_aggregate_key():
    adapter = _detr_adapter(
        {
            "loss_vfl": torch.tensor(1.0),
            "loss_bbox": torch.tensor(2.0),
            "total_loss": torch.tensor(3.0),
        }
    )

    values = adapter(_scoped_predictions(), torch.zeros(1, 1, 5), image_size=(8, 8))

    assert float(values["loss"]) == pytest.approx(3.0)
    assert set(values) == {"loss", "loss/vfl", "loss/bbox"}


def test_detr_adapter_requires_the_forward_scope_output():
    adapter = _detr_adapter({"loss_vfl": torch.tensor(1.0)})
    predictions = {
        "pred_logits": torch.zeros(1, 4, 3),
        "pred_boxes": torch.zeros(1, 4, 4),
    }

    with pytest.raises(ValueError, match="forward scope was not active"):
        adapter(predictions, torch.zeros(1, 1, 5), image_size=(8, 8))


def test_validator_enters_the_adapter_forward_scope():
    entered = []

    class _ScopedAdapter(_Adapter):
        def forward_scope(self):
            @contextmanager
            def scope():
                entered.append("in")
                try:
                    yield
                finally:
                    entered.append("out")

            return scope()

    validator = _validator(_ScopedAdapter())
    validator._active_loss_adapter = validator.loss_adapter
    with validator._validation_loss_scope():
        assert entered == ["in"]
    assert entered == ["in", "out"]


def test_validator_scope_is_a_noop_without_an_adapter_hook():
    validator = _validator(_Adapter())
    validator._active_loss_adapter = validator.loss_adapter
    with validator._validation_loss_scope():
        pass


def test_dfine_loss_scope_leaves_the_metric_predictions_untouched():
    """The mAP path must not notice the loss scope.

    ``eval_idx`` is the last decoder layer on every shipped size, so scoring
    the earlier layers only adds keys; ``pred_logits``/``pred_boxes`` stay the
    tensors the metrics already used.
    """
    from libreyolo.models.dfine.nn import LibreDFINEModel

    # 256px is the smallest square that still yields the decoder's 300 queries.
    model = LibreDFINEModel(
        config="n", nb_classes=2, eval_spatial_size=(256, 256)
    ).eval()
    images = torch.randn(1, 3, 256, 256)

    with torch.no_grad():
        plain = model(images)
        with emit_loss_outputs(model):
            scoped = model(images)
        after = model(images)

    assert set(plain) == set(after) == {"pred_logits", "pred_boxes"}
    assert {"aux_outputs", "enc_aux_outputs", "pre_outputs"} <= set(scoped)
    assert len(scoped["aux_outputs"]) == model.decoder.num_layers - 1
    for key in ("pred_logits", "pred_boxes"):
        assert torch.equal(plain[key], scoped[key])
        assert torch.equal(plain[key], after[key])


class _MixinHost(ValidationLossMixin):
    """Bare host for the shared plumbing, with no dataset or model."""

    task = "classify"

    def __init__(self, adapter):
        self.loss_adapter = adapter
        self._reset_validation_loss()

    def _autocast_context(self):
        return nullcontext()


def test_mixin_averages_scalars_over_batches():
    host = _MixinHost(lambda p, t, *, image_size: {"loss": p, "loss/ce": p})

    for value in (1.0, 2.0, 6.0):
        host._accumulate_validation_loss(value, None, image_size=None)

    assert host._validation_loss_metrics() == {
        "metrics/loss": pytest.approx(3.0),
        "metrics/loss/ce": pytest.approx(3.0),
    }


def test_mixin_discards_everything_when_an_adapter_fails_midway(caplog):
    calls = {"n": 0}

    def adapter(predictions, targets, *, image_size):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("synthetic failure")
        return {"loss": 1.0}

    host = _MixinHost(adapter)
    with caplog.at_level(logging.WARNING):
        for _ in range(3):
            host._accumulate_validation_loss(None, None, image_size=None)

    # A partial average would silently under-report, so nothing is published
    # and the adapter stays off for the rest of the pass.
    assert calls["n"] == 2
    assert host._validation_loss_metrics() == {}
    assert "classify metrics will continue" in caplog.text


def test_mixin_rejects_augmented_validation():
    host = _MixinHost(None)
    host.config = SimpleNamespace(augment=True)

    with pytest.raises(ValueError, match="augmented validation"):
        host._init_validation_loss(lambda *a, **k: {"loss": 1.0})


def test_classify_adapter_matches_cross_entropy():
    adapter = ClassifyValidationLoss(device=torch.device("cpu"), family="resnet")
    logits = torch.randn(4, 10)
    labels = torch.tensor([3, 1, 9, 0])

    values = adapter(logits, labels)

    expected = torch.nn.functional.cross_entropy(logits, labels)
    assert float(values["loss"]) == pytest.approx(float(expected))
    assert float(values["loss/ce"]) == pytest.approx(float(values["loss"]))


def test_classify_adapter_accepts_the_validators_dict_output():
    adapter = ClassifyValidationLoss(device=torch.device("cpu"), family="convnext")
    logits = torch.randn(2, 5)
    labels = torch.tensor([1, 4])

    assert float(adapter({"logits": logits}, labels)["loss"]) == pytest.approx(
        float(adapter(logits, labels)["loss"])
    )


def test_classify_adapter_rejects_a_label_outside_the_head():
    adapter = ClassifyValidationLoss(device=torch.device("cpu"), family="resnet")

    with pytest.raises(ValueError, match=r"outside \[0, 9\]"):
        adapter(torch.randn(2, 10), torch.tensor([1, 10]))


def test_classify_adapter_rejects_a_label_count_mismatch():
    adapter = ClassifyValidationLoss(device=torch.device("cpu"), family="resnet")

    with pytest.raises(ValueError, match="3 labels for 2 images"):
        adapter(torch.randn(2, 10), torch.tensor([1, 2, 3]))


def test_classify_validator_publishes_the_loss_next_to_accuracy():
    validator = object.__new__(ClassifyValidator)
    validator.loss_adapter = ClassifyValidationLoss(
        device=torch.device("cpu"), family="resnet"
    )
    validator._top1_correct = 1
    validator._top5_correct = 2
    validator._total = 2
    validator._reset_validation_loss()
    validator._autocast_context = nullcontext
    validator._accumulate_validation_loss(
        torch.zeros(2, 4), torch.tensor([0, 1]), image_size=None
    )

    metrics = validator._compute_metrics()

    assert metrics["metrics/accuracy_top1"] == pytest.approx(0.5)
    # Uniform logits over 4 classes: -log(1/4).
    assert metrics["metrics/loss"] == pytest.approx(math.log(4.0))
    assert metrics["metrics/loss/ce"] == pytest.approx(math.log(4.0))


def test_semantic_adapter_matches_the_family_criterion():
    adapter = SemanticValidationLoss(
        device=torch.device("cpu"), family="segformer", ignore_index=255
    )
    logits = torch.randn(2, 3, 8, 8)
    mask = torch.randint(0, 3, (2, 8, 8))
    mask[0, :2, :2] = 255

    values = adapter(logits, mask)

    expected = torch.nn.functional.cross_entropy(logits, mask, ignore_index=255)
    assert float(values["loss"]) == pytest.approx(float(expected))
    assert float(values["loss/sem"]) == pytest.approx(float(values["loss"]))


def test_semantic_adapter_upsamples_logits_to_the_mask():
    adapter = SemanticValidationLoss(
        device=torch.device("cpu"), family="segformer", ignore_index=255
    )
    logits = torch.randn(1, 3, 4, 4)
    mask = torch.randint(0, 3, (1, 16, 16))

    upsampled = torch.nn.functional.interpolate(
        logits, size=(16, 16), mode="bilinear", align_corners=False
    )
    expected = torch.nn.functional.cross_entropy(upsampled, mask, ignore_index=255)
    assert float(adapter(logits, mask)["loss"]) == pytest.approx(float(expected))


def test_semantic_adapter_returns_zero_when_every_pixel_is_ignored():
    adapter = SemanticValidationLoss(
        device=torch.device("cpu"), family="segformer", ignore_index=255
    )
    mask = torch.full((1, 8, 8), 255, dtype=torch.long)

    # cross_entropy is NaN here; a NaN would poison the epoch average.
    value = float(adapter(torch.randn(1, 3, 8, 8), mask)["loss"])
    assert value == 0.0


def test_dense_head_adapter_splits_padded_targets_and_labels_components():
    seen = {}

    class _Criterion(torch.nn.Module):
        def forward(self, cls_scores, bbox_preds, gt_boxes_list, gt_labels_list):
            seen["boxes"] = gt_boxes_list
            seen["labels"] = gt_labels_list
            return {
                "total_loss": torch.tensor(3.0),
                "loss_cls": torch.tensor(1.0),
                "loss_bbox": torch.tensor(2.0),
                "num_pos": 7.0,
            }

    adapter = DenseHeadValidationLoss(
        _Criterion(), num_classes=4, device=torch.device("cpu"), family="rtmdet"
    )
    targets = torch.tensor(
        [
            [[10.0, 20.0, 30.0, 60.0, 2.0], [0.0, 0.0, 0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]],
        ]
    )

    values = adapter(([torch.zeros(1)], [torch.zeros(1)]), targets)

    # Padding rows are dropped; boxes stay xyxy in pixels.
    assert seen["boxes"][0].tolist() == [[10.0, 20.0, 30.0, 60.0]]
    assert seen["labels"][0].tolist() == [2]
    assert seen["boxes"][1].shape == (0, 4)
    # num_pos is a diagnostic, not a weighted term.
    assert set(values) == {"loss", "loss/cls", "loss/bbox"}
    assert float(values["loss"]) == pytest.approx(3.0)


def test_dense_head_adapter_rejects_a_non_head_prediction():
    adapter = DenseHeadValidationLoss(
        torch.nn.Module(), num_classes=4, device=torch.device("cpu"), family="rtmdet"
    )

    with pytest.raises(TypeError, match="cls_scores"):
        adapter(torch.zeros(2, 3), torch.zeros(1, 1, 5))


def test_yolox_loss_scope_leaves_the_inference_output_untouched():
    """The mAP path must not notice the loss scope.

    YOLOX's eval branch sigmoids obj/cls and skips the grid bookkeeping the
    criterion needs, so the scope assembles a *second* set of tensors from the
    same conv outputs rather than changing the returned one.
    """
    from libreyolo.models.yolox.nn import LibreYOLOXModel
    from libreyolo.models.yolox.validation_loss import YOLOXValidationLoss

    model = LibreYOLOXModel(config="s", nb_classes=4).eval()
    images = torch.randn(2, 3, 256, 256)
    targets = torch.tensor(
        [
            [[20.0, 20.0, 100.0, 90.0, 1.0], [0.0, 0.0, 0.0, 0.0, 0.0]],
            [[50.0, 40.0, 180.0, 150.0, 3.0], [10.0, 10.0, 60.0, 60.0, 0.0]],
        ]
    )
    adapter = YOLOXValidationLoss(model, num_classes=4, device=torch.device("cpu"))

    with torch.no_grad():
        plain = model(images)
        with adapter.forward_scope():
            scoped = model(images)
            values = adapter(scoped, targets)
        after = model(images)

    for before, during, then in zip(plain, scoped, after):
        assert torch.equal(before, during)
        assert torch.equal(before, then)
    assert adapter.head.emit_loss_outputs is False
    assert adapter.head._loss_cache is None
    components = sum(float(v) for k, v in values.items() if k != "loss")
    assert components == pytest.approx(float(values["loss"]), rel=1e-4)


def test_yolox_adapter_requires_the_forward_scope():
    from libreyolo.models.yolox.nn import LibreYOLOXModel
    from libreyolo.models.yolox.validation_loss import YOLOXValidationLoss

    model = LibreYOLOXModel(config="s", nb_classes=4).eval()
    adapter = YOLOXValidationLoss(model, num_classes=4, device=torch.device("cpu"))

    with pytest.raises(ValueError, match="forward scope was not active"):
        adapter(None, torch.zeros(1, 1, 5))


@pytest.mark.parametrize(
    "trainer_cls, family",
    [
        (ResNetTrainer, "resnet"),
        (ConvNeXtTrainer, "convnext"),
        (MobileNetV4Trainer, "mobilenetv4"),
        (EfficientNetV2Trainer, "efficientnetv2"),
    ],
)
def test_classification_trainers_accept_val_loss(trainer_cls, family):
    trainer = object.__new__(trainer_cls)
    trainer.config = SimpleNamespace(val_loss=True)
    trainer.wrapper_model = SimpleNamespace(task="classify")
    trainer.device = torch.device("cpu")

    trainer.validate_validation_loss_config()
    adapter = trainer.build_validation_loss_adapter(torch.nn.Linear(2, 2))

    assert isinstance(adapter, ClassifyValidationLoss)
    assert adapter.family == family


def test_classification_trainer_rejects_a_non_classify_task():
    trainer = object.__new__(ResNetTrainer)
    trainer.config = SimpleNamespace(val_loss=True)
    trainer.wrapper_model = SimpleNamespace(task="detect")

    with pytest.raises(ValueError, match="classification only"):
        trainer.validate_validation_loss_config()


def test_fomo_accepts_val_loss_because_it_always_reports_one():
    from libreyolo.models.fomo.trainer import FOMOTrainer

    trainer = object.__new__(FOMOTrainer)
    trainer.config = SimpleNamespace(val_loss=True)

    # No adapter: FOMOValidator computes the loss unconditionally.
    trainer.validate_validation_loss_config()


def test_obb_and_point_validators_cannot_be_handed_a_loss_adapter():
    """Guards the one branch that picks a validator class by task.

    OBBValidator and PointValidator take no loss_adapter, so passing one
    would raise inside the validation try/except and cost the epoch its real
    metrics. The trainer gates on the mixin instead of on the task name.
    """
    from libreyolo.validation.obb_validator import OBBValidator
    from libreyolo.validation.point_validator import PointValidator
    from libreyolo.validation.detection_validator import SegmentationValidator

    assert not issubclass(OBBValidator, ValidationLossMixin)
    assert not issubclass(PointValidator, ValidationLossMixin)
    # Segmentation rides on DetectionValidator, so it does accept one.
    assert issubclass(SegmentationValidator, ValidationLossMixin)
    assert issubclass(DetectionValidator, ValidationLossMixin)


def test_semantic_trainer_rejects_a_non_semantic_task():
    from libreyolo.models.segformer.trainer import SegformerTrainer

    trainer = object.__new__(SegformerTrainer)
    trainer.config = SimpleNamespace(val_loss=True)
    trainer.wrapper_model = SimpleNamespace(task="detect")

    with pytest.raises(ValueError, match="semantic segmentation only"):
        trainer.validate_validation_loss_config()


#: Every family that has a trainer, and the group it is registered in. A new
#: family is free to land without validation loss, but one that has it must
#: not quietly lose it.
_VAL_LOSS_FAMILIES = {
    "libreyolo.models.rfdetr.trainer": "RFDETRTrainer",
    "libreyolo.models.yolo9.trainer": "YOLO9Trainer",
    "libreyolo.models.yolo9_e2e.trainer": "YOLO9E2ETrainer",
    "libreyolo.models.yolo9_p2.trainer": "YOLO9P2Trainer",
    "libreyolo.models.yolonas.trainer": "YOLONASTrainer",
    "libreyolo.models.rtdetr.trainer": "RTDETRTrainer",
    "libreyolo.models.rtdetrv2.trainer": "RTDETRv2Trainer",
    "libreyolo.models.rtdetrv4.trainer": "RTDETRv4Trainer",
    "libreyolo.models.dfine.trainer": "DFINETrainer",
    "libreyolo.models.domedetr.trainer": "DOMEDETRTrainer",
    "libreyolo.models.deim.trainer": "DEIMTrainer",
    "libreyolo.models.deimv2.trainer": "DEIMv2Trainer",
    "libreyolo.models.tinyformer.trainer": "TinyFormerTrainer",
    "libreyolo.models.ec.trainer": "ECTrainer",
    "libreyolo.models.rtmdet.trainer": "RTMDetTrainer",
    "libreyolo.models.picodet.trainer": "PICODETTrainer",
    "libreyolo.models.ppyoloe.trainer": "PPYOLOETrainer",
    "libreyolo.models.yolox.trainer": "YOLOXTrainer",
    "libreyolo.models.yolo7.trainer": "YOLOv7Trainer",
    "libreyolo.models.fomo.trainer": "FOMOTrainer",
    "libreyolo.models.resnet.trainer": "ResNetTrainer",
    "libreyolo.models.convnext.trainer": "ConvNeXtTrainer",
    "libreyolo.models.mobilenetv4.trainer": "MobileNetV4Trainer",
    "libreyolo.models.efficientnetv2.trainer": "EfficientNetV2Trainer",
    "libreyolo.models.ppliteseg.trainer": "PPLiteSegTrainer",
    "libreyolo.models.unet.trainer": "UNetTrainer",
    "libreyolo.models.segformer.trainer": "SegformerTrainer",
    "libreyolo.models.lingbotvision.trainer": "LingBotVisionTrainer",
    "libreyolo.models.dinov2.trainer": "DINOv2Trainer",
    "libreyolo.models.vjepa2.trainer": "VJEPA2Trainer",
    "libreyolo.models.nafnet.trainer": "NAFNetTrainer",
}


@pytest.mark.parametrize(
    "module_path, class_name", sorted(_VAL_LOSS_FAMILIES.items())
)
def test_every_trainable_family_implements_the_val_loss_gate(
    module_path, class_name
):
    """A family that inherits the base gate cannot support ``val_loss=True``.

    Overriding it is what distinguishes "implemented" from "rejected", so
    this catches a family losing support in a refactor.
    """
    import importlib

    trainer_cls = getattr(importlib.import_module(module_path), class_name)

    assert (
        trainer_cls.validate_validation_loss_config
        is not BaseTrainer.validate_validation_loss_config
    ), f"{class_name} no longer implements val_loss"


def test_the_covered_set_is_every_family_that_has_a_trainer():
    """Guards the docs claim that every trainable family is covered."""
    import pathlib

    import libreyolo.models

    models_dir = pathlib.Path(libreyolo.models.__file__).parent
    with_trainer = {
        path.parent.name for path in models_dir.glob("*/trainer.py")
    }
    covered = {path.split(".")[2] for path in _VAL_LOSS_FAMILIES}

    assert with_trainer == covered


def test_monitor_overlays_validation_loss_when_present():
    assert 'includes("metrics/loss")' in INDEX_HTML
    assert 'name: "val/loss"' in INDEX_HTML
    assert 'card("Val loss"' in INDEX_HTML


def test_experiment_loggers_emit_canonical_validation_loss_name():
    event = SimpleNamespace(
        train_loss=2.0,
        train_loss_items={},
        lr={},
        val_metrics={"metrics/loss": 1.5, "metrics/loss/box": 0.5},
        epoch_seconds=3.0,
    )

    metrics = epoch_metrics(event)

    assert metrics["val/loss"] == pytest.approx(1.5)
    assert metrics["val/loss/box"] == pytest.approx(0.5)
