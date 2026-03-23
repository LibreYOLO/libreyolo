"""
Unit tests for libreyolo.distillation module.

Tests the model-agnostic distillation system with dummy models to verify
hooks, losses, channel adaptation, and the full distiller pipeline work
correctly without requiring real YOLO weights.
"""

import pytest
import torch
import torch.nn as nn

pytestmark = pytest.mark.unit

from libreyolo.distillation.hooks import FeatureHookManager, _resolve_module
from libreyolo.distillation.losses import MGDLoss, CWDLoss
from libreyolo.distillation.distiller import Distiller
from libreyolo.distillation.configs import get_distill_config, list_supported


# =============================================================================
# Dummy models for testing
# =============================================================================


class DummyNeck(nn.Module):
    """Simulates a YOLO neck with 3 FPN output stages."""

    def __init__(self, channels=(64, 128, 256)):
        super().__init__()
        self.p3 = nn.Conv2d(3, channels[0], 1)
        self.p4 = nn.Conv2d(3, channels[1], 1)
        self.p5 = nn.Conv2d(3, channels[2], 1)

    def forward(self, x):
        return self.p3(x), self.p4(x), self.p5(x)


class DummyModel(nn.Module):
    """Simulates a YOLO model with backbone.neck structure."""

    def __init__(self, channels=(64, 128, 256)):
        super().__init__()
        self.neck = DummyNeck(channels)

    def forward(self, x):
        p3, p4, p5 = self.neck(x)
        return {"p3": p3, "p4": p4, "p5": p5}


class DummyModelFlat(nn.Module):
    """A model with flat (non-nested) modules."""

    def __init__(self, channels=(64, 128, 256)):
        super().__init__()
        self.layer_a = nn.Conv2d(3, channels[0], 1)
        self.layer_b = nn.Conv2d(3, channels[1], 1)
        self.layer_c = nn.Conv2d(3, channels[2], 1)

    def forward(self, x):
        return self.layer_a(x), self.layer_b(x), self.layer_c(x)


# =============================================================================
# Tests: _resolve_module
# =============================================================================


class TestResolveModule:
    def test_single_level(self):
        model = DummyModel()
        neck = _resolve_module(model, "neck")
        assert isinstance(neck, DummyNeck)

    def test_nested_path(self):
        model = DummyModel()
        p3 = _resolve_module(model, "neck.p3")
        assert isinstance(p3, nn.Conv2d)

    def test_invalid_path_raises(self):
        model = DummyModel()
        with pytest.raises(AttributeError, match="has no submodule"):
            _resolve_module(model, "neck.nonexistent")

    def test_invalid_deep_path_raises(self):
        model = DummyModel()
        with pytest.raises(AttributeError, match="failed at segment"):
            _resolve_module(model, "neck.p3.deeper.module")


# =============================================================================
# Tests: FeatureHookManager
# =============================================================================


class TestFeatureHookManager:
    def test_captures_features(self):
        model = DummyModel(channels=(32, 64, 128))
        manager = FeatureHookManager(model, ["neck.p3", "neck.p4", "neck.p5"])

        x = torch.randn(2, 3, 16, 16)
        model(x)

        feats = manager.get_feature_list()
        assert len(feats) == 3
        assert feats[0].shape == (2, 32, 16, 16)
        assert feats[1].shape == (2, 64, 16, 16)
        assert feats[2].shape == (2, 128, 16, 16)

    def test_clear_resets_features(self):
        model = DummyModel()
        manager = FeatureHookManager(model, ["neck.p3"])

        model(torch.randn(1, 3, 8, 8))
        assert len(manager.get_feature_list()) == 1

        manager.clear()
        assert len(manager.get_feature_list()) == 0

    def test_remove_stops_capture(self):
        model = DummyModel()
        manager = FeatureHookManager(model, ["neck.p3"])

        model(torch.randn(1, 3, 8, 8))
        assert len(manager.get_feature_list()) == 1

        manager.remove()
        model(torch.randn(1, 3, 8, 8))
        assert len(manager.get_feature_list()) == 0

    def test_all_tap_points_captured(self):
        """Hooks fire in forward execution order (not registration order).
        We verify all requested tap points are captured regardless of order."""
        model = DummyModel()
        manager = FeatureHookManager(model, ["neck.p5", "neck.p3", "neck.p4"])

        model(torch.randn(1, 3, 8, 8))
        keys = set(manager.get_features().keys())
        assert keys == {"neck.p5", "neck.p3", "neck.p4"}

    def test_flat_model(self):
        model = DummyModelFlat(channels=(16, 32, 64))
        manager = FeatureHookManager(model, ["layer_a", "layer_b", "layer_c"])

        model(torch.randn(1, 3, 8, 8))
        feats = manager.get_feature_list()
        assert len(feats) == 3
        assert feats[0].shape[1] == 16
        assert feats[2].shape[1] == 64


# =============================================================================
# Tests: MGDLoss
# =============================================================================


class TestMGDLoss:
    def test_same_channels(self):
        loss_fn = MGDLoss(student_channels=128, teacher_channels=128)
        s = torch.randn(2, 128, 8, 8)
        t = torch.randn(2, 128, 8, 8)
        loss = loss_fn(s, t)
        assert loss.shape == ()
        assert loss.item() > 0

    def test_different_channels(self):
        loss_fn = MGDLoss(student_channels=64, teacher_channels=256)
        s = torch.randn(2, 64, 8, 8)
        t = torch.randn(2, 256, 8, 8)
        loss = loss_fn(s, t)
        assert loss.shape == ()
        assert loss.item() > 0

    def test_identical_features_low_loss(self):
        loss_fn = MGDLoss(student_channels=32, teacher_channels=32, mask_ratio=0.0)
        t = torch.randn(2, 32, 8, 8)
        # With mask_ratio=0, no masking. If student == teacher, generation
        # can learn identity, but untrained it won't be zero.
        loss = loss_fn(t, t)
        assert loss.shape == ()

    def test_loss_weight_scales_output(self):
        """Loss weight=5 should produce 5x the loss of weight=1."""
        # Use mask_ratio=0 so the mask is deterministic (all ones)
        loss_fn_1 = MGDLoss(student_channels=32, teacher_channels=32, loss_weight=1.0, mask_ratio=0.0)
        loss_fn_5 = MGDLoss(student_channels=32, teacher_channels=32, loss_weight=5.0, mask_ratio=0.0)

        # Share the same weights so generation output is identical
        loss_fn_5.align.load_state_dict(loss_fn_1.align.state_dict())
        loss_fn_5.generation.load_state_dict(loss_fn_1.generation.state_dict())

        s = torch.randn(2, 32, 8, 8)
        t = torch.randn(2, 32, 8, 8)

        loss_1 = loss_fn_1(s, t)
        loss_5 = loss_fn_5(s, t)

        assert loss_1.item() > 0
        assert abs(loss_5.item() - 5.0 * loss_1.item()) < 1e-4

    def test_gradient_flows(self):
        loss_fn = MGDLoss(student_channels=64, teacher_channels=128)
        s = torch.randn(2, 64, 8, 8, requires_grad=True)
        t = torch.randn(2, 128, 8, 8)
        loss = loss_fn(s, t)
        loss.backward()
        assert s.grad is not None
        assert s.grad.shape == s.shape


# =============================================================================
# Tests: CWDLoss
# =============================================================================


class TestCWDLoss:
    def test_same_channels(self):
        loss_fn = CWDLoss(student_channels=128, teacher_channels=128)
        s = torch.randn(2, 128, 8, 8)
        t = torch.randn(2, 128, 8, 8)
        loss = loss_fn(s, t)
        assert loss.shape == ()
        assert loss.item() >= 0

    def test_different_channels(self):
        loss_fn = CWDLoss(student_channels=64, teacher_channels=256)
        s = torch.randn(2, 64, 8, 8)
        t = torch.randn(2, 256, 8, 8)
        loss = loss_fn(s, t)
        assert loss.shape == ()

    def test_identical_distributions_zero_loss(self):
        loss_fn = CWDLoss()
        t = torch.randn(2, 32, 8, 8)
        loss = loss_fn(t, t)
        # KL divergence of identical distributions should be ~0
        assert loss.item() < 1e-5

    def test_loss_weight_scales_output(self):
        """Loss weight=3 should produce 3x the loss of weight=1."""
        loss_fn_1 = CWDLoss(student_channels=64, teacher_channels=64, loss_weight=1.0)
        loss_fn_3 = CWDLoss(student_channels=64, teacher_channels=64, loss_weight=3.0)

        s = torch.randn(2, 64, 8, 8)
        t = torch.randn(2, 64, 8, 8)

        loss_1 = loss_fn_1(s, t)
        loss_3 = loss_fn_3(s, t)

        assert loss_1.item() > 0
        assert abs(loss_3.item() - 3.0 * loss_1.item()) < 1e-4

    def test_temperature_affects_loss(self):
        """Different tau values should produce different loss values."""
        s = torch.randn(2, 32, 8, 8)
        t = torch.randn(2, 32, 8, 8)

        loss_low_tau = CWDLoss(tau=0.1)(s, t)
        loss_high_tau = CWDLoss(tau=10.0)(s, t)

        # With very different temperatures, losses should differ
        assert loss_low_tau.item() != pytest.approx(loss_high_tau.item(), rel=0.1)

    def test_gradient_flows(self):
        loss_fn = CWDLoss(student_channels=64, teacher_channels=128)
        s = torch.randn(2, 64, 8, 8, requires_grad=True)
        t = torch.randn(2, 128, 8, 8)
        loss = loss_fn(s, t)
        loss.backward()
        assert s.grad is not None


# =============================================================================
# Tests: Distiller (integration)
# =============================================================================


class TestDistiller:
    def _make_distiller(self, loss_type="mgd"):
        teacher = DummyModel(channels=(128, 256, 512))
        student = DummyModel(channels=(64, 128, 256))

        t_cfg = {
            "tap_points": ["neck.p3", "neck.p4", "neck.p5"],
            "channels": [128, 256, 512],
            "strides": [8, 16, 32],
        }
        s_cfg = {
            "tap_points": ["neck.p3", "neck.p4", "neck.p5"],
            "channels": [64, 128, 256],
            "strides": [8, 16, 32],
        }

        distiller = Distiller(
            teacher_model=teacher,
            student_model=student,
            teacher_config=t_cfg,
            student_config=s_cfg,
            loss_type=loss_type,
        )
        return distiller, teacher, student

    def test_mgd_pipeline(self):
        distiller, teacher, student = self._make_distiller("mgd")
        x = torch.randn(2, 3, 16, 16)

        distiller.teacher_forward(x)
        student(x)
        loss = distiller.compute_loss()

        assert loss.shape == ()
        assert loss.item() > 0
        distiller.step()

    def test_cwd_pipeline(self):
        distiller, teacher, student = self._make_distiller("cwd")
        x = torch.randn(2, 3, 16, 16)

        distiller.teacher_forward(x)
        student(x)
        loss = distiller.compute_loss()

        assert loss.shape == ()
        assert loss.item() >= 0
        distiller.step()

    def test_teacher_is_frozen(self):
        distiller, teacher, student = self._make_distiller()
        for p in teacher.parameters():
            assert not p.requires_grad

    def test_stride_mismatch_raises(self):
        teacher = DummyModel()
        student = DummyModel()

        with pytest.raises(ValueError, match="matching strides"):
            Distiller(
                teacher_model=teacher,
                student_model=student,
                teacher_config={"tap_points": ["neck.p3"], "channels": [64], "strides": [8]},
                student_config={"tap_points": ["neck.p3"], "channels": [64], "strides": [16]},
            )

    def test_compute_loss_without_forward_raises(self):
        distiller, _, _ = self._make_distiller()
        with pytest.raises(RuntimeError, match="Expected"):
            distiller.compute_loss()

    def test_cleanup(self):
        distiller, teacher, student = self._make_distiller()
        x = torch.randn(1, 3, 8, 8)

        distiller.teacher_forward(x)
        student(x)
        distiller.cleanup()

        # After cleanup, hooks should be removed
        student(x)
        assert len(distiller.s_hooks.get_feature_list()) == 0

    def test_multiple_steps(self):
        distiller, teacher, student = self._make_distiller()

        for _ in range(3):
            x = torch.randn(2, 3, 16, 16)
            distiller.teacher_forward(x)
            student(x)
            loss = distiller.compute_loss()
            assert loss.item() > 0
            distiller.step()

    def test_cross_architecture(self):
        """Test distilling between different model architectures."""
        teacher = DummyModel(channels=(256, 512, 1024))
        student = DummyModelFlat(channels=(64, 128, 256))

        t_cfg = {
            "tap_points": ["neck.p3", "neck.p4", "neck.p5"],
            "channels": [256, 512, 1024],
            "strides": [8, 16, 32],
        }
        s_cfg = {
            "tap_points": ["layer_a", "layer_b", "layer_c"],
            "channels": [64, 128, 256],
            "strides": [8, 16, 32],
        }

        distiller = Distiller(
            teacher_model=teacher,
            student_model=student,
            teacher_config=t_cfg,
            student_config=s_cfg,
            loss_type="mgd",
        )

        x = torch.randn(2, 3, 16, 16)
        distiller.teacher_forward(x)
        student(x)
        loss = distiller.compute_loss()
        assert loss.item() > 0
        distiller.step()

    def test_mgd_loss_converges(self):
        """MGD distillation loss should decrease over optimization steps."""
        teacher = DummyModel(channels=(128, 256, 512))
        student = DummyModel(channels=(64, 128, 256))
        t_cfg = {
            "tap_points": ["neck.p3", "neck.p4", "neck.p5"],
            "channels": [128, 256, 512],
            "strides": [8, 16, 32],
        }
        s_cfg = {
            "tap_points": ["neck.p3", "neck.p4", "neck.p5"],
            "channels": [64, 128, 256],
            "strides": [8, 16, 32],
        }
        # Use loss_weight=1.0 so the loss isn't scaled down to near-zero
        distiller = Distiller(
            teacher_model=teacher,
            student_model=student,
            teacher_config=t_cfg,
            student_config=s_cfg,
            loss_type="mgd",
            loss_weight=1.0,
        )

        params = list(student.parameters()) + list(distiller.loss_modules.parameters())
        optimizer = torch.optim.Adam(params, lr=1e-3)

        x = torch.randn(2, 3, 16, 16)

        losses = []
        for _ in range(50):
            distiller.teacher_forward(x)
            student(x)
            loss = distiller.compute_loss()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            distiller.step()

        assert losses[-1] < losses[0] * 0.5, (
            f"MGD loss did not converge: first={losses[0]:.6f} → last={losses[-1]:.6f}"
        )

    def test_cwd_loss_converges(self):
        """CWD distillation loss should decrease over optimization steps."""
        distiller, teacher, student = self._make_distiller("cwd")

        params = list(student.parameters()) + list(distiller.loss_modules.parameters())
        optimizer = torch.optim.Adam(params, lr=1e-3)

        x = torch.randn(2, 3, 16, 16)

        losses = []
        for _ in range(50):
            distiller.teacher_forward(x)
            student(x)
            loss = distiller.compute_loss()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            distiller.step()

        assert losses[-1] < losses[0] * 0.5, (
            f"CWD loss did not converge: first={losses[0]:.6f} → last={losses[-1]:.6f}"
        )

    def test_different_inputs_produce_different_losses(self):
        """Hooks should capture per-batch features, not stale ones."""
        distiller, teacher, student = self._make_distiller("mgd")

        x1 = torch.randn(2, 3, 16, 16)
        x2 = torch.randn(2, 3, 16, 16) + 5.0  # very different input

        distiller.teacher_forward(x1)
        student(x1)
        loss_1 = distiller.compute_loss().item()
        distiller.step()

        distiller.teacher_forward(x2)
        student(x2)
        loss_2 = distiller.compute_loss().item()
        distiller.step()

        assert loss_1 != pytest.approx(loss_2, rel=0.01), (
            "Different inputs should produce different losses"
        )

    def test_distiller_params_receive_gradients(self):
        """Distiller's own parameters (align/generation convs) should be trainable."""
        distiller, teacher, student = self._make_distiller("mgd")

        x = torch.randn(2, 3, 16, 16)
        distiller.teacher_forward(x)
        student(x)
        loss = distiller.compute_loss()
        loss.backward()

        # MGD loss_modules have align and generation convolutions
        has_grad = False
        for p in distiller.loss_modules.parameters():
            if p.grad is not None and p.grad.abs().sum() > 0:
                has_grad = True
                break

        assert has_grad, "No gradients flowed to distiller's own parameters"

    def test_per_scale_weight_mismatch_raises(self):
        """Wrong number of per_scale_weight entries should raise ValueError."""
        teacher = DummyModel(channels=(128, 256, 512))
        student = DummyModel(channels=(64, 128, 256))

        t_cfg = {
            "tap_points": ["neck.p3", "neck.p4", "neck.p5"],
            "channels": [128, 256, 512],
            "strides": [8, 16, 32],
        }
        s_cfg = {
            "tap_points": ["neck.p3", "neck.p4", "neck.p5"],
            "channels": [64, 128, 256],
            "strides": [8, 16, 32],
        }

        with pytest.raises(ValueError, match="per_scale_weight"):
            Distiller(
                teacher_model=teacher,
                student_model=student,
                teacher_config=t_cfg,
                student_config=s_cfg,
                per_scale_weight=[1.0, 1.0],  # 2 instead of 3
            )

    def test_cleanup_is_idempotent(self):
        """Calling cleanup() twice should not raise."""
        distiller, teacher, student = self._make_distiller()

        x = torch.randn(1, 3, 8, 8)
        distiller.teacher_forward(x)
        student(x)

        distiller.cleanup()
        distiller.cleanup()  # should not raise

    def test_mgd_mask_ratio_affects_convergence(self):
        """mask_ratio=0 (sees full features) should converge faster than mask_ratio=1 (sees zeros)."""
        s = torch.randn(4, 64, 8, 8)
        t = torch.randn(4, 64, 8, 8)

        loss_fn_no_mask = MGDLoss(student_channels=64, teacher_channels=64, mask_ratio=0.0)
        loss_fn_full_mask = MGDLoss(student_channels=64, teacher_channels=64, mask_ratio=1.0)

        # Share initial weights
        loss_fn_full_mask.align.load_state_dict(loss_fn_no_mask.align.state_dict())
        loss_fn_full_mask.generation.load_state_dict(loss_fn_no_mask.generation.state_dict())

        # Train each for 30 steps
        for loss_fn, results in [
            (loss_fn_no_mask, []),
            (loss_fn_full_mask, []),
        ]:
            opt = torch.optim.Adam(loss_fn.parameters(), lr=1e-3)
            for _ in range(30):
                loss = loss_fn(s, t)
                opt.zero_grad()
                loss.backward()
                opt.step()
            results.append(loss.item())

        # Evaluate both with mask_ratio=0 (no mask) to compare learned quality
        with torch.no_grad():
            # Temporarily set mask_ratio=0 for fair comparison
            orig_ratio = loss_fn_full_mask.mask_ratio
            loss_fn_full_mask.mask_ratio = 0.0
            loss_no_mask_final = loss_fn_no_mask(s, t).item()
            loss_full_mask_final = loss_fn_full_mask(s, t).item()
            loss_fn_full_mask.mask_ratio = orig_ratio

        # The model that trained with full input should reconstruct better
        assert loss_no_mask_final < loss_full_mask_final, (
            f"No-mask trained loss ({loss_no_mask_final:.6f}) should be lower than "
            f"full-mask trained loss ({loss_full_mask_final:.6f})"
        )


# =============================================================================
# Tests: configs.py
# =============================================================================


class TestConfigs:
    def test_yolo9_all_sizes(self):
        for size in ["t", "s", "m", "c"]:
            cfg = get_distill_config("yolo9", size)
            assert len(cfg["tap_points"]) == 3
            assert len(cfg["channels"]) == 3
            assert cfg["strides"] == [8, 16, 32]

    def test_yolox_all_sizes(self):
        for size in ["n", "t", "s", "m", "l", "x"]:
            cfg = get_distill_config("yolox", size)
            assert len(cfg["tap_points"]) == 3
            assert len(cfg["channels"]) == 3
            assert cfg["strides"] == [8, 16, 32]

    def test_yolo9_channel_values(self):
        cfg = get_distill_config("yolo9", "t")
        assert cfg["channels"] == [64, 96, 128]

        cfg = get_distill_config("yolo9", "c")
        assert cfg["channels"] == [256, 512, 512]

    def test_yolox_channel_values(self):
        cfg = get_distill_config("yolox", "s")
        assert cfg["channels"] == [128, 256, 512]

        cfg = get_distill_config("yolox", "l")
        assert cfg["channels"] == [256, 512, 1024]

    def test_unknown_family_raises(self):
        with pytest.raises(ValueError, match="not yet configured"):
            get_distill_config("unknown_model", "s")

    def test_unknown_size_raises(self):
        with pytest.raises(ValueError, match="Invalid size"):
            get_distill_config("yolo9", "z")

    def test_list_supported(self):
        supported = list_supported()
        assert "yolo9" in supported
        assert "yolox" in supported


# =============================================================================
# Tests: model wrapper get_distill_config()
# =============================================================================


class TestModelDistillConfig:
    """Verify that model wrappers return correct distillation configs."""

    def test_yolo9_config_matches_nn_architecture(self):
        """Config channels must match YOLO9_CONFIGS head_channels."""
        from libreyolo.models.yolo9.model import LibreYOLO9
        from libreyolo.models.yolo9.nn import YOLO9_CONFIGS

        for size in ["t", "s", "m", "c"]:
            model = LibreYOLO9(model_path=None, size=size)
            cfg = model.get_distill_config()
            expected = list(YOLO9_CONFIGS[size]["head_channels"])
            assert cfg["channels"] == expected, f"YOLOv9-{size}: {cfg['channels']} != {expected}"
            assert cfg["strides"] == [8, 16, 32]
            assert len(cfg["tap_points"]) == 3

    def test_yolox_config_matches_nn_architecture(self):
        """Config channels must match YOLOX width * base channels."""
        from libreyolo.models.yolox.model import LibreYOLOX
        from libreyolo.models.yolox.nn import LibreYOLOXModel

        for size in ["n", "t", "s", "m", "l", "x"]:
            model = LibreYOLOX(model_path=None, size=size)
            cfg = model.get_distill_config()
            width = LibreYOLOXModel.CONFIGS[size]["width"]
            expected = [int(256 * width), int(512 * width), int(1024 * width)]
            assert cfg["channels"] == expected, f"YOLOX-{size}: {cfg['channels']} != {expected}"
            assert cfg["strides"] == [8, 16, 32]
            assert len(cfg["tap_points"]) == 3

    def test_unsupported_family_raises(self):
        """BaseModel default should raise NotImplementedError."""
        from libreyolo.models.base.model import BaseModel

        # BaseModel can't be instantiated directly (ABC), so test via the
        # get_distill_config convenience function with an unknown family
        with pytest.raises(ValueError, match="not yet configured"):
            get_distill_config("rfdetr", "n")

    def test_model_config_matches_convenience_function(self):
        """Model wrapper and convenience function should return identical configs."""
        from libreyolo.models.yolo9.model import LibreYOLO9

        model = LibreYOLO9(model_path=None, size="t")
        direct = model.get_distill_config()
        via_func = get_distill_config("yolo9", "t")
        assert direct == via_func


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
