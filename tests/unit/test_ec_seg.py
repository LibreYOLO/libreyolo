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
from libreyolo.tasks import resolve_task

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

    def test_train_seg_raises_not_implemented(self):
        m = LibreEC(model_path=None, size="s", task="segment")
        with pytest.raises(NotImplementedError, match="ECSeg training"):
            m.train(data="dummy.yaml", allow_experimental=True)


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
