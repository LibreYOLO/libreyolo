"""Unit tests for trainer layer freezing."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

pytestmark = pytest.mark.unit


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 4, 3, padding=1),
            nn.BatchNorm2d(4),
        )
        self.head = nn.Linear(4, 2)

    def forward(self, x):
        x = self.stem(x).mean(dim=(2, 3))
        return self.head(x)


def test_freeze_spec_normalization_matches_public_api():
    from libreyolo.training.freezing import (
        normalize_freeze_selectors,
        parse_freeze_spec,
    )

    assert normalize_freeze_selectors(3) == (0, 1, 2)
    assert normalize_freeze_selectors("[0, 3, 'head']") == (0, 3, "head")
    assert parse_freeze_spec("backbone,neck") == ["backbone", "neck"]
    assert normalize_freeze_selectors("backbone") == ("backbone",)
    assert normalize_freeze_selectors("") == ()


def test_apply_freeze_by_name_disables_params_and_bn_stats():
    from libreyolo.training.freezing import apply_freeze

    model = TinyModel().train()
    summary = apply_freeze(model, "stem")

    assert summary is not None
    assert summary.frozen_tensor_count == len(list(model.stem.parameters()))
    assert all(not p.requires_grad for p in model.stem.parameters())
    assert all(p.requires_grad for p in model.head.parameters())
    assert not model.stem[1].training


def test_base_optimizer_excludes_frozen_parameters():
    from libreyolo.training.trainer import BaseTrainer

    class DummyTrainer(BaseTrainer):
        def get_model_family(self):
            return "dummy"

        def get_model_tag(self):
            return "dummy"

        def create_transforms(self):
            return None, None

        def create_scheduler(self, iters_per_epoch):
            return None

        def get_loss_components(self, outputs):
            return {}

    model = TinyModel()
    for param in model.stem.parameters():
        param.requires_grad = False
    trainer = DummyTrainer(model=model, data=None, device="cpu", optimizer="sgd")

    optimizer = trainer._setup_optimizer()
    optimizer_param_ids = {
        id(param)
        for group in optimizer.param_groups
        for param in group["params"]
    }
    frozen_param_ids = {id(param) for param in model.stem.parameters()}
    trainable_param_ids = {id(param) for param in model.head.parameters()}

    assert not (optimizer_param_ids & frozen_param_ids)
    assert trainable_param_ids <= optimizer_param_ids


def test_yolo9_freeze_10_maps_to_complete_backbone():
    from libreyolo.models.yolo9.nn import LibreYOLO9Model
    from libreyolo.models.yolo9.trainer import YOLO9Trainer
    from libreyolo.training.freezing import apply_freeze

    model = LibreYOLO9Model(config="t", nb_classes=3)
    trainer = YOLO9Trainer.__new__(YOLO9Trainer)
    trainer.model = model

    groups = trainer.get_freeze_groups()
    assert [name for name, _module in groups[:10]] == [
        "backbone.conv0",
        "backbone.conv1",
        "backbone.elan1",
        "backbone.down2",
        "backbone.elan2",
        "backbone.down3",
        "backbone.elan3",
        "backbone.down4",
        "backbone.elan4",
        "backbone.spp",
    ]

    summary = apply_freeze(model, 10, freeze_groups=groups)
    assert summary is not None
    assert all(not p.requires_grad for p in model.backbone.parameters())
    assert any(p.requires_grad for p in model.neck.parameters())
    assert any(p.requires_grad for p in model.head.parameters())


class _FakeBackboneOwner(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(4, 4)
        self.projector = nn.Linear(4, 4)

    def get_named_param_lr_pairs(self, args, prefix: str):
        pairs = {}
        for name, param in self.named_parameters():
            if param.requires_grad:
                pairs[f"{prefix}.{name}"] = {
                    "params": param,
                    "lr": args.lr * 0.1,
                    "weight_decay": args.weight_decay,
                }
        return pairs


class _FakeLoRAEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Linear(4, 4)
        self.lora_A = nn.Parameter(torch.ones(2, 2))
        self.lora_B = nn.Parameter(torch.zeros(2, 2))


class _FakeTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = nn.Linear(4, 4)
        self.enc_output = nn.ModuleList([nn.Linear(4, 4)])
        self.enc_output_norm = nn.ModuleList([nn.LayerNorm(4)])
        self.enc_out_class_embed = nn.ModuleList([nn.Linear(4, 3)])
        self.enc_out_bbox_embed = nn.ModuleList([nn.Linear(4, 4)])


class _FakeDecoderWithSharedHead(nn.Module):
    def __init__(self, bbox_embed):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(4, 4)])
        self.ref_point_head = nn.Linear(4, 4)
        self.bbox_embed = bbox_embed


class _FakeRFDETRCore(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(_FakeBackboneOwner())
        self.transformer = _FakeTransformer()
        self.refpoint_embed = nn.Embedding(2, 4)
        self.query_feat = nn.Embedding(2, 4)
        self.class_embed = nn.Linear(4, 3)
        self.bbox_embed = nn.Linear(4, 4)
        self.angle_embed = None
        self.segmentation_head = None


class _FakeRFDETRWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _FakeRFDETRCore()


class _FakeRFDETRSharedHeadCore(_FakeRFDETRCore):
    def __init__(self):
        super().__init__()
        self.transformer.decoder = _FakeDecoderWithSharedHead(self.bbox_embed)


class _FakeRFDETRSharedHeadWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _FakeRFDETRSharedHeadCore()


class _FakeRFDETRClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = _FakeBackboneOwner()
        self.linear = nn.Linear(4, 3)


class _FakeRFDETRClassificationWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = _FakeRFDETRClassifier()
        self.model = None


def test_rfdetr_freeze_groups_support_backbone_and_head_selectors():
    from libreyolo.models.rfdetr.trainer import RFDETRTrainer
    from libreyolo.training.freezing import apply_freeze

    wrapper = _FakeRFDETRWrapper()
    trainer = RFDETRTrainer.__new__(RFDETRTrainer)
    trainer.model = wrapper

    groups = trainer.get_freeze_groups()
    assert [name for name, _module in groups] == [
        "backbone.encoder",
        "backbone.projector",
        "decoder",
        "queries",
        "transformer.encoder_output",
        "head",
    ]

    summary = apply_freeze(wrapper, "backbone", freeze_groups=groups)
    assert summary is not None
    backbone_owner = wrapper.model.backbone[0]
    assert all(not p.requires_grad for p in backbone_owner.encoder.parameters())
    assert all(not p.requires_grad for p in backbone_owner.projector.parameters())
    assert all(p.requires_grad for p in wrapper.model.transformer.decoder.parameters())
    assert all(p.requires_grad for p in wrapper.model.class_embed.parameters())

    wrapper = _FakeRFDETRWrapper()
    trainer.model = wrapper
    apply_freeze(wrapper, ["decoder", "head"], freeze_groups=trainer.get_freeze_groups())
    assert all(not p.requires_grad for p in wrapper.model.transformer.decoder.parameters())
    assert all(not p.requires_grad for p in wrapper.model.class_embed.parameters())
    assert all(not p.requires_grad for p in wrapper.model.bbox_embed.parameters())
    assert all(p.requires_grad for p in wrapper.model.backbone.parameters())


def test_rfdetr_decoder_freeze_does_not_freeze_shared_bbox_head():
    from libreyolo.models.rfdetr.trainer import RFDETRTrainer
    from libreyolo.training.freezing import apply_freeze

    wrapper = _FakeRFDETRSharedHeadWrapper()
    trainer = RFDETRTrainer.__new__(RFDETRTrainer)
    trainer.model = wrapper

    apply_freeze(wrapper, "decoder", freeze_groups=trainer.get_freeze_groups())

    assert all(not p.requires_grad for p in wrapper.model.transformer.decoder.layers.parameters())
    assert all(not p.requires_grad for p in wrapper.model.transformer.decoder.ref_point_head.parameters())
    assert all(p.requires_grad for p in wrapper.model.bbox_embed.parameters())
    assert all(p.requires_grad for p in wrapper.model.class_embed.parameters())


def test_rfdetr_classify_freeze_backbone_targets_classifier_backbone():
    from libreyolo.models.rfdetr.trainer import RFDETRTrainer
    from libreyolo.training.freezing import apply_freeze

    wrapper = _FakeRFDETRClassificationWrapper()
    trainer = RFDETRTrainer.__new__(RFDETRTrainer)
    trainer.model = wrapper

    groups = trainer.get_freeze_groups()
    assert [name for name, _module in groups] == [
        "backbone.encoder",
        "backbone.projector",
        "head",
    ]

    summary = apply_freeze(wrapper, "backbone", freeze_groups=groups)

    assert summary is not None
    assert all(not p.requires_grad for p in wrapper.classifier.backbone.parameters())
    assert all(p.requires_grad for p in wrapper.classifier.linear.parameters())


def test_rfdetr_lora_freeze_preserves_adapter_params():
    from libreyolo.models.rfdetr.trainer import RFDETRTrainer

    wrapper = _FakeRFDETRWrapper()
    wrapper.model.backbone[0].encoder = _FakeLoRAEncoder()
    trainer = RFDETRTrainer.__new__(RFDETRTrainer)
    trainer.model = wrapper
    trainer.config = SimpleNamespace(freeze="backbone", lora=True)

    trainer._apply_freeze_config()

    named_params = dict(wrapper.named_parameters())
    lora_params = {
        name: param for name, param in named_params.items() if "lora_" in name
    }
    frozen_backbone_base = {
        name: param
        for name, param in named_params.items()
        if name.startswith("model.backbone.0.encoder.base")
        or name.startswith("model.backbone.0.projector")
    }

    assert trainer.freeze_summary is not None
    assert lora_params
    assert frozen_backbone_base
    assert all(param.requires_grad for param in lora_params.values())
    assert all(not param.requires_grad for param in frozen_backbone_base.values())
    assert all(
        "lora_" not in name for name in trainer.freeze_summary.frozen_param_names
    )


def test_rfdetr_upstream_optimizer_groups_survive_frozen_backbone():
    from libreyolo.models.rfdetr.trainer import RFDETRTrainer

    wrapper = _FakeRFDETRWrapper()
    wrapper.args = SimpleNamespace(lr_component_decay=0.25)
    for param in wrapper.model.backbone.parameters():
        param.requires_grad = False

    trainer = RFDETRTrainer.__new__(RFDETRTrainer)
    trainer.model = wrapper
    trainer.config = SimpleNamespace(lr0=0.01, weight_decay=0.001)

    groups = trainer._setup_upstream_optimizer_groups()

    decoder_param_ids = {
        id(param) for param in wrapper.model.transformer.decoder.parameters()
    }
    grouped_param_ids = {id(group["params"]) for group in groups}
    decoder_groups = [
        group for group in groups if id(group["params"]) in decoder_param_ids
    ]

    assert groups
    assert decoder_param_ids <= grouped_param_ids
    assert all(group["lr"] == pytest.approx(0.0025) for group in decoder_groups)
    assert all(group["lr_mult"] == pytest.approx(0.25) for group in decoder_groups)
    assert not any(
        id(param) in grouped_param_ids for param in wrapper.model.backbone.parameters()
    )


def test_cli_train_kwargs_keep_freeze_for_generic_and_rfdetr(tmp_path):
    from libreyolo.cli.config import build_family_train_kwargs

    generic = build_family_train_kwargs({"freeze": "backbone"}, family="yolo9")
    assert generic["freeze"] == "backbone"

    rfdetr = build_family_train_kwargs(
        {
            "project": str(tmp_path),
            "name": "run",
            "exist_ok": True,
            "freeze": ["backbone", "head"],
        },
        family="rfdetr",
    )
    assert rfdetr["freeze"] == ["backbone", "head"]
