"""Unit tests for LibreEC segmentation support.

Mirrors test_ec_pose.py with segmentation-specific bits. Covers:
- filename ``-seg`` suffix resolves to ``task='segment'``
- pose vs detect vs segment checkpoint discrimination
- explicit-vs-checkpoint task conflicts raise clearly
- forward emits pred_masks; postprocess emits masks
- detect path still wires through (no regression)
"""

from __future__ import annotations

import pytest
import torch

from libreyolo.models.ec.model import LibreEC
from libreyolo.models.ec.nn import LibreECSegModel
from libreyolo.models.ec.postprocess import postprocess_seg
from libreyolo.training.config import ECSegConfig

pytestmark = [pytest.mark.unit, pytest.mark.ec]


class TestFilenameTaskResolution:
    def test_seg_suffix_resolves_to_segment_task(self):
        assert LibreEC.detect_task_from_filename("LibreECs-seg.pt") == "segment"
        assert LibreEC.detect_task_from_filename("LibreECl-seg.pt") == "segment"

    def test_seg_in_supported_tasks(self):
        assert "segment" in LibreEC.SUPPORTED_TASKS

    def test_size_detection_for_seg_filenames(self):
        for size in ("s", "m", "l", "x"):
            assert LibreEC.detect_size_from_filename(f"LibreEC{size}-seg.pt") == size


class TestSegCheckpointDiscrimination:
    def test_seg_state_dict_detected(self):
        sd = {"decoder.decoder.segmentation_head.bias": torch.zeros(1)}
        assert LibreEC.is_seg_state_dict(sd) is True
        assert LibreEC.detect_task_from_state_dict(sd) == "segment"

    def test_pose_state_dict_not_seg(self):
        sd = {"decoder.keypoint_embedding.weight": torch.zeros(17, 192)}
        assert LibreEC.is_seg_state_dict(sd) is False
        assert LibreEC.detect_task_from_state_dict(sd) == "pose"

    def test_detect_state_dict_neither(self):
        sd = {"decoder.dec_score_head.0.bias": torch.zeros(80)}
        assert LibreEC.is_seg_state_dict(sd) is False
        assert LibreEC.is_pose_state_dict(sd) is False
        assert LibreEC.detect_task_from_state_dict(sd) is None


class TestSegFamilyClassWiring:
    def test_seg_init_sets_task_and_metadata(self):
        m = LibreEC(model_path=None, size="s", task="segment")
        assert m.task == "segment"
        assert m.family == "ec"
        assert isinstance(m.model, LibreECSegModel)

    def test_train_seg_requires_allow_experimental(self):
        # Seg training is implemented but gated behind the experimental flag.
        m = LibreEC(model_path=None, size="s", task="segment")
        with pytest.raises(RuntimeError, match="experimental"):
            m.train(data="dummy.yaml")

    def test_train_seg_selects_seg_trainer(self):
        # With the flag set, the seg task dispatches to ECSegTrainer (it fails
        # only later, on the missing dummy dataset — not with NotImplementedError).
        m = LibreEC(model_path=None, size="s", task="segment")
        with pytest.raises(FileNotFoundError):
            m.train(data="definitely_missing.yaml", allow_experimental=True)


class TestSegForwardAndPostprocess:
    @pytest.fixture(scope="class")
    def seg_model(self):
        m = LibreEC(model_path=None, size="s", task="segment")
        m.model.eval()
        return m

    def test_forward_output_shape(self, seg_model):
        x = torch.randn(1, 3, 640, 640).to(seg_model.device)
        with torch.no_grad():
            out = seg_model._forward(x)
        assert "pred_masks" in out
        assert out["pred_logits"].shape == (1, 300, 80)
        assert out["pred_boxes"].shape == (1, 300, 4)
        # mask resolution = input / mask_downsample_ratio (4) = 160x160
        assert out["pred_masks"].shape == (1, 300, 160, 160)

    def test_postprocess_emits_masks(self, seg_model):
        x = torch.randn(1, 3, 640, 640).to(seg_model.device)
        with torch.no_grad():
            raw = seg_model._forward(x)
        det = postprocess_seg(
            raw,
            conf_thres=0.0,
            iou_thres=0.0,
            original_size=(800, 600),
            max_det=20,
        )
        assert "masks" in det
        # Masks resampled to original (H, W).
        assert det["masks"].shape[-2:] == (600, 800)
        assert det["masks"].shape[0] == det["boxes"].shape[0]
        assert det["masks"].dtype == torch.bool

    def test_full_predict_pipeline(self, seg_model):
        from PIL import Image

        img = Image.new("RGB", (320, 240), color=(127, 127, 127))
        result = seg_model(img, conf=0.0, max_det=10)
        assert result.masks is not None
        # masks (N, H, W) tensor; boxes share the same N
        assert len(result) == result.masks.data.shape[0]


class TestDetectPathUnchanged:
    """Sanity: enabling segment in the family should not break detect mode."""

    def test_detect_init_unchanged(self):
        m = LibreEC(model_path=None, size="s")
        assert m.task == "detect"
        assert m.nb_classes == 80

    def test_detect_forward_no_pred_masks(self):
        m = LibreEC(model_path=None, size="s")
        m.model.eval()
        x = torch.randn(1, 3, 640, 640).to(m.device)
        with torch.no_grad():
            out = m._forward(x)
        assert "pred_masks" not in out
        assert "pred_logits" in out and "pred_boxes" in out


class TestSegTrainingStep:
    """One forward+loss+backward step exercises the deferred-mask training path."""

    def _criterion(self):
        from libreyolo.models.ec.seg_loss import ECSegCriterion, ECSegHungarianMatcher

        return ECSegCriterion(
            matcher=ECSegHungarianMatcher(
                weight_dict={
                    "cost_class": 2.0,
                    "cost_bbox": 1.0,
                    "cost_giou": 1.0,
                    "cost_mask_ce": 5.0,
                    "cost_mask_dice": 5.0,
                },
                use_focal_loss=True,
                alpha=0.25,
                gamma=2.0,
                mask_point_sample_ratio=16,
            ),
            weight_dict={
                "loss_mal": 2.0, "loss_bbox": 1.0, "loss_giou": 1.0,
                "loss_fgl": 0.15, "loss_ddf": 1.5,
                "loss_mask_ce": 5.0, "loss_mask_dice": 5.0,
            },
            losses=["mal", "boxes", "local", "masks"],
            num_classes=2, alpha=0.75, gamma=1.5, reg_max=32,
        )

    def test_seg_config_size_specific_optimizer_defaults(self):
        cfg_l = ECSegConfig.from_kwargs(size="l")
        assert cfg_l.backbone_lr_mult == pytest.approx(0.005)
        assert cfg_l.weight_decay == pytest.approx(1.25e-4)

        cfg_x = ECSegConfig.from_kwargs(
            size="x", backbone_lr_mult=0.2, weight_decay=9e-4
        )
        assert cfg_x.backbone_lr_mult == pytest.approx(0.2)
        assert cfg_x.weight_decay == pytest.approx(9e-4)

    def test_seg_criterion_uses_upstream_mal_gamma(self):
        assert self._criterion().gamma == pytest.approx(1.5)

    def test_seg_train_forward_emits_deferred_masks(self):
        torch.manual_seed(0)
        model = LibreECSegModel(config="s", nb_classes=2, eval_spatial_size=(128, 128))
        model.train()
        masks = torch.zeros(2, 128, 128, dtype=torch.bool)
        masks[0, 20:60, 20:60] = True
        masks[1, 70:110, 60:115] = True
        targets = [{
            "labels": torch.tensor([0, 1]),
            "boxes": torch.tensor([[0.31, 0.31, 0.31, 0.31], [0.68, 0.70, 0.43, 0.31]]),
            "masks": masks,
        }]
        out = model(torch.randn(1, 3, 128, 128), targets=targets)
        # Training masks come back in the memory-efficient deferred form.
        assert isinstance(out["pred_masks"], dict)
        assert {"spatial_features", "query_features", "bias"} <= set(out["pred_masks"])
        assert "pred_masks" in out["aux_outputs"][0]
        assert "pred_masks" in out["pre_outputs"]
        assert "pred_masks" in out["dn_pre_outputs"]

    def test_seg_loss_backward_reaches_mask_head(self):
        torch.manual_seed(0)
        model = LibreECSegModel(config="s", nb_classes=2, eval_spatial_size=(128, 128))
        model.train()

        def make_target():
            masks = torch.zeros(2, 128, 128, dtype=torch.bool)
            masks[0, 20:60, 20:60] = True
            masks[1, 70:110, 60:115] = True
            return {
                "labels": torch.tensor([0, 1]),
                "boxes": torch.tensor([[0.31, 0.31, 0.31, 0.31], [0.68, 0.70, 0.43, 0.31]]),
                "masks": masks,
            }

        # second image has no instances → exercises the empty-match branch
        targets = [make_target(), {
            "labels": torch.zeros(0, dtype=torch.long),
            "boxes": torch.zeros(0, 4),
            "masks": torch.zeros(0, 128, 128, dtype=torch.bool),
        }]
        out = model(torch.randn(2, 3, 128, 128), targets=targets)
        losses = self._criterion()(out, targets)
        assert any(k.startswith("loss_mask") for k in losses)
        assert "loss_mask_ce_pre" in losses
        assert "loss_mask_dice_pre" in losses
        assert "loss_mask_ce_dn_pre" in losses
        assert "loss_mask_dice_dn_pre" in losses
        total = sum(losses.values())
        assert torch.isfinite(total)
        total.backward()
        seg_grad = sum(
            1 for n, p in model.named_parameters()
            if "segmentation_head" in n and p.grad is not None and p.grad.abs().sum() > 0
        )
        assert seg_grad > 0, "segmentation head received no gradient"
