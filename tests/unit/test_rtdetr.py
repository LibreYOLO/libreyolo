"""Unit tests for RTDETR model.

Tests for RTDETR model registry, weight loading, class detection, and configuration.
"""

import numpy as np
import pytest
import torch

from libreyolo.models.dfine.model import LibreDFINE
from libreyolo.models.rtdetr.config import RTDETRConfig
from libreyolo.models.rtdetr.model import LibreRTDETR
from libreyolo.models.rtdetr.trainer import RTDETRTrainer
from libreyolo.training.scheduler import ConstantLRScheduler
from libreyolo.validation.preprocessors import RTDETRValPreprocessor

pytestmark = pytest.mark.unit


class TestRTDETRRegistry:
    """Test RTDETR model registration."""

    def test_rtdetr_is_registered(self):
        """LibreRTDETR should be in the BaseModel registry."""
        from libreyolo.models.base.model import BaseModel

        assert LibreRTDETR in BaseModel._registry or any(
            cls.__name__ == "LibreRTDETR" for cls in BaseModel._registry
        )


class TestRTDETRCanLoad:
    """Test RTDETR weight key detection."""

    def test_rtdetr_can_load_with_rtdetr_keys(self):
        """can_load() should return True for RTDETR-specific weight keys."""
        fake_weights = {
            "backbone.res_layers.0.blocks.0.conv1.weight": torch.zeros(64, 3, 3, 3),
            "encoder.input_proj.0.0.weight": torch.zeros(256, 512, 1, 1),
            "decoder.input_proj.0.conv.weight": torch.zeros(256, 256, 1, 1),
            "decoder.dec_score_head.0.weight": torch.zeros(80, 256),
            "decoder.dec_score_head.0.bias": torch.zeros(80),
        }
        assert LibreRTDETR.can_load(fake_weights) is True

    def test_rtdetr_can_load_with_hgnetv2_keys(self):
        """can_load() should return True for HGNetv2-backbone (L/X) weight keys."""
        fake_weights = {
            "backbone.stages.0.blocks.0.layers.0.conv1.conv.weight": torch.zeros(
                48, 48, 1, 1
            ),
            "encoder.input_proj.0.0.weight": torch.zeros(256, 512, 1, 1),
            "decoder.input_proj.0.conv.weight": torch.zeros(256, 256, 1, 1),
            "decoder.dec_score_head.0.weight": torch.zeros(80, 256),
        }
        assert LibreRTDETR.can_load(fake_weights) is True

    def test_rtdetr_does_not_claim_dfine_keys(self):
        """can_load() should not claim D-FINE checkpoints with overlapping keys."""
        fake_dfine_weights = {
            "backbone.stages.0.blocks.0.layers.0.conv1.conv.weight": torch.zeros(
                48, 48, 1, 1
            ),
            "encoder.input_proj.0.conv.weight": torch.zeros(128, 256, 1, 1),
            "decoder.pre_bbox_head.layers.0.weight": torch.zeros(256, 256),
            "decoder.dec_score_head.0.bias": torch.zeros(80),
            "decoder.dec_bbox_head.0.layers.0.weight": torch.zeros(256, 256),
        }
        assert LibreDFINE.can_load(fake_dfine_weights) is True
        assert LibreRTDETR.can_load(fake_dfine_weights) is False

    def test_dfine_does_not_claim_rtdetr_overlap_keys(self):
        """D-FINE should not match RT-DETR decoder heads that share names."""
        fake_rtdetr_weights = {
            "backbone.res_layers.0.blocks.0.conv1.weight": torch.zeros(64, 3, 3, 3),
            "encoder.input_proj.0.0.weight": torch.zeros(256, 512, 1, 1),
            "decoder.input_proj.0.conv.weight": torch.zeros(256, 256, 1, 1),
            "decoder.denoising_class_embed.weight": torch.zeros(81, 256),
            "decoder.enc_bbox_head.layers.0.weight": torch.zeros(256, 256),
            "decoder.dec_bbox_head.0.layers.0.weight": torch.zeros(256, 256),
            "decoder.dec_score_head.0.bias": torch.zeros(80),
        }
        assert LibreRTDETR.can_load(fake_rtdetr_weights) is True
        assert LibreDFINE.can_load(fake_rtdetr_weights) is False

    def test_rtdetr_cannot_load_rfdetr_keys(self):
        """can_load() should return False for RF-DETR weight keys."""
        fake_rfdetr_weights = {
            "model.backbone.0.body.layer1.0.conv1.weight": torch.zeros(64, 64, 1, 1),
            "model.encoder_projection.weight": torch.zeros(256, 512),
        }
        assert LibreRTDETR.can_load(fake_rfdetr_weights) is False


class TestRTDETRDetectSize:
    """Test RTDETR size detection from weights."""

    def test_detect_size_resnet18(self):
        """BasicBlock + two stage-0 blocks -> 'r18'."""
        weights = {
            "backbone.res_layers.0.blocks.0.branch2a.conv.weight": torch.zeros(
                64, 64, 3, 3
            ),
            "backbone.res_layers.0.blocks.1.branch2a.conv.weight": torch.zeros(
                64, 64, 3, 3
            ),
            "encoder.input_proj.0.0.weight": torch.zeros(256, 128, 1, 1),
        }
        assert LibreRTDETR.detect_size(weights) == "r18"

    def test_detect_size_resnet50(self):
        """Bottleneck + full-width encoder expansion -> 'r50'."""
        weights = {
            "backbone.res_layers.0.blocks.0.branch2c.conv.weight": torch.zeros(
                256, 64, 1, 1
            ),
            "encoder.input_proj.0.0.weight": torch.zeros(256, 512, 1, 1),
            "encoder.fpn_blocks.0.conv1.conv.weight": torch.zeros(256, 512, 1, 1),
        }
        assert LibreRTDETR.detect_size(weights) == "r50"

    def test_detect_size_resnet50m(self):
        """Bottleneck + half-width encoder expansion -> 'r50m'."""
        weights = {
            "backbone.res_layers.0.blocks.0.branch2c.conv.weight": torch.zeros(
                256, 64, 1, 1
            ),
            "encoder.input_proj.0.0.weight": torch.zeros(256, 512, 1, 1),
            "encoder.fpn_blocks.0.conv1.conv.weight": torch.zeros(128, 512, 1, 1),
        }
        assert LibreRTDETR.detect_size(weights) == "r50m"

    def test_detect_size_resnet101(self):
        """Hidden dim 384 identifies the R101 variant."""
        weights = {
            "backbone.res_layers.0.blocks.0.branch2c.conv.weight": torch.zeros(
                256, 64, 1, 1
            ),
            "encoder.input_proj.0.0.weight": torch.zeros(384, 512, 1, 1),
        }
        assert LibreRTDETR.detect_size(weights) == "r101"

    def test_detect_size_hgnetv2_l(self):
        """HGNetv2 + encoder hidden_dim 256 → 'l'."""
        weights = {
            "backbone.stages.0.blocks.0.layers.0.conv1.conv.weight": torch.zeros(
                48, 48, 1, 1
            ),
            "encoder.input_proj.0.0.weight": torch.zeros(256, 512, 1, 1),
        }
        assert LibreRTDETR.detect_size(weights) == "l"

    def test_detect_size_hgnetv2_x(self):
        """HGNetv2 + encoder hidden_dim 384 → 'x'."""
        weights = {
            "backbone.stages.0.blocks.0.layers.0.conv1.conv.weight": torch.zeros(
                64, 64, 1, 1
            ),
            "encoder.input_proj.0.0.weight": torch.zeros(384, 512, 1, 1),
        }
        assert LibreRTDETR.detect_size(weights) == "x"


class TestRTDETRDetectNbClasses:
    """Test RTDETR class count detection."""

    def test_rtdetr_detect_nb_classes(self):
        """detect_nb_classes() should read from dec_score_head bias."""
        fake_weights = {
            "decoder.dec_score_head.0.bias": torch.zeros(80),
            "decoder.dec_score_head.1.bias": torch.zeros(80),
            "decoder.dec_score_head.2.bias": torch.zeros(80),
        }
        assert LibreRTDETR.detect_nb_classes(fake_weights) == 80

    def test_rtdetr_detect_nb_classes_custom(self):
        """detect_nb_classes() should work with non-COCO class counts."""
        fake_weights = {
            "decoder.dec_score_head.0.bias": torch.zeros(10),
        }
        assert LibreRTDETR.detect_nb_classes(fake_weights) == 10


class TestRTDETRModelInstantiation:
    """Test RTDETR model creation."""

    def test_rtdetr_model_instantiation(self):
        """LibreRTDETR should be instantiatable from scratch."""
        model = LibreRTDETR(nb_classes=80, size="r18")
        assert model is not None
        assert model.nb_classes == 80
        assert model.size == "r18"


class TestRTDETRValPreprocessor:
    """Test RTDETR validation preprocessor."""

    def test_rtdetr_val_preprocessor(self):
        """RTDETRValPreprocessor should resize and normalize correctly."""
        preprocessor = RTDETRValPreprocessor(img_size=(640, 640))
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        # Empty targets for testing
        targets = np.zeros((0, 5), dtype=np.float32)
        result, padded_targets = preprocessor(img, targets, (640, 640))
        assert result.shape == (3, 640, 640)
        assert result.dtype == np.float32
        assert result.min() >= 0.0
        assert result.max() <= 1.0
        assert padded_targets.shape == (120, 5)  # max_labels, 5


class TestRTDETRExportMetadata:
    """Test RTDETR export metadata."""

    def test_rtdetr_export_metadata(self):
        """RTDETR model family should be 'rtdetr'."""
        model = LibreRTDETR(nc=80, size="r18")
        assert model.FAMILY == "rtdetr"

    def test_ncnn_export_is_blocked_for_rtdetr(self, tmp_path):
        """NCNN cannot run RT-DETR's DETR-style query selection."""
        model = LibreRTDETR(nc=80, size="r18", device="cpu")
        with pytest.raises(
            NotImplementedError, match="NCNN export is not supported for RT-DETR"
        ):
            model.export("ncnn", output_path=str(tmp_path / "rtdetr_ncnn"))


class TestRTDETRConfig:
    """Test RTDETR training configuration."""

    def test_rtdetr_config_defaults(self):
        """RTDETRConfig should have RTDETR-specific defaults."""
        config = RTDETRConfig(data="dummy.yaml")
        assert config.scheduler == "constant"
        assert config.lr_backbone == 0.000005
        assert config.betas == (0.9, 0.999)
        assert config.clip_max_norm == 0.1
        assert config.ema_decay == 0.9999
        assert config.mosaic_prob == 0.0
        assert config.hsv_prob == 0.5


def test_rtdetr_constant_scheduler_factory():
    trainer = RTDETRTrainer.__new__(RTDETRTrainer)
    trainer.config = RTDETRConfig(
        data="dummy.yaml",
        batch=16,
        lr0=0.001,
        warmup_epochs=2,
        warmup_lr_start=1e-6,
    )

    scheduler = RTDETRTrainer.create_scheduler(trainer, iters_per_epoch=10)

    assert isinstance(scheduler, ConstantLRScheduler)
    assert scheduler.update_lr(0) == pytest.approx(1e-6)
    assert scheduler.update_lr(20) == pytest.approx(0.001)
    assert scheduler.update_lr(100) == pytest.approx(0.001)
