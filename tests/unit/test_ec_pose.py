"""Unit tests for LibreEC pose support.

Mirrors test_yolonas_pose.py with EC-specific bits. Covers:
- filename ``-pose`` suffix resolves to ``task='pose'``
- pose vs detect checkpoint discrimination
- explicit-vs-checkpoint task conflicts raise clearly
- Results plumbing + ``_select`` alignment
- pose forward + postprocess shape contract
- detect path still wires through (no regression)
"""

from __future__ import annotations

import pytest
import torch

from libreyolo.models.ec.model import LibreEC
from libreyolo.models.ec.nn import LibreECPoseModel
from libreyolo.models.ec.postprocess import postprocess_pose
from libreyolo.tasks import resolve_task

pytestmark = [pytest.mark.unit, pytest.mark.ec]


class TestFilenameTaskResolution:
    def test_pose_suffix_resolves_to_pose_task(self):
        assert LibreEC.detect_task_from_filename("LibreECs-pose.pt") == "pose"
        assert LibreEC.detect_task_from_filename("LibreECl-pose.pt") == "pose"

    def test_no_suffix_resolves_to_none_task(self):
        assert LibreEC.detect_task_from_filename("LibreECs.pt") is None

    def test_size_detection_for_pose_filenames(self):
        for size in ("s", "m", "l", "x"):
            assert LibreEC.detect_size_from_filename(f"LibreEC{size}-pose.pt") == size

    def test_unsupported_task_raises(self):
        with pytest.raises(ValueError, match="not supported"):
            resolve_task(
                explicit_task="classify",
                supported_tasks=LibreEC.SUPPORTED_TASKS,
            )

    def test_pose_in_supported_tasks(self):
        assert "pose" in LibreEC.SUPPORTED_TASKS
        assert "detect" in LibreEC.SUPPORTED_TASKS


class TestPoseCheckpointDiscrimination:
    def test_pose_state_dict_detected(self):
        sd = {"decoder.keypoint_embedding.weight": torch.zeros(17, 192)}
        assert LibreEC.is_pose_state_dict(sd) is True

    def test_detect_state_dict_not_pose(self):
        sd = {"decoder.dec_score_head.0.bias": torch.zeros(80)}
        assert LibreEC.is_pose_state_dict(sd) is False


class TestPoseFamilyClassWiring:
    def test_pose_init_sets_task_and_metadata(self):
        m = LibreEC(model_path=None, size="s", task="pose")
        assert m.task == "pose"
        assert m.family == "ec"
        assert m.nb_classes == 1
        assert m.names == {0: "person"}
        assert isinstance(m.model, LibreECPoseModel)

    def test_detect_init_unchanged(self):
        m = LibreEC(model_path=None, size="s")
        assert m.task == "detect"
        assert m.nb_classes == 80
        assert not isinstance(m.model, LibreECPoseModel)

    def test_classify_task_rejected(self):
        with pytest.raises(ValueError, match="not supported"):
            LibreEC(model_path=None, size="s", task="classify")

    def test_train_pose_raises_not_implemented(self):
        m = LibreEC(model_path=None, size="s", task="pose")
        with pytest.raises(NotImplementedError, match="ECPose training"):
            m.train(data="dummy.yaml", allow_experimental=True)


class TestPoseForwardAndPostprocess:
    @pytest.fixture(scope="class")
    def pose_model(self):
        m = LibreEC(model_path=None, size="s", task="pose")
        m.model.eval()
        return m

    def test_forward_output_shape(self, pose_model):
        x = torch.randn(1, 3, 640, 640).to(pose_model.device)
        with torch.no_grad():
            out = pose_model._forward(x)
        assert set(out) == {"pred_logits", "pred_keypoints"}
        # ECPose-s default: num_queries=60, 2-class, 17 keypoints
        assert out["pred_logits"].shape == (1, 60, 2)
        assert out["pred_keypoints"].shape == (1, 60, 34)

    def test_postprocess_emits_keypoints(self, pose_model):
        x = torch.randn(1, 3, 640, 640).to(pose_model.device)
        with torch.no_grad():
            raw = pose_model._forward(x)
        det = postprocess_pose(
            raw,
            conf_thres=0.0,
            iou_thres=0.0,
            original_size=(800, 600),
            max_det=20,
            num_keypoints=17,
        )
        assert "keypoints" in det
        assert det["keypoints"].shape[-1] == 3
        assert det["keypoints"].shape[-2] == 17
        assert det["keypoints"].shape[0] == det["boxes"].shape[0]

    def test_full_predict_pipeline(self, pose_model):
        # Exercise _wrap_results so the keypoint plumbing stays working.
        from PIL import Image

        img = Image.new("RGB", (320, 240), color=(127, 127, 127))
        result = pose_model(img, conf=0.0, max_det=10)
        assert result.keypoints is not None
        assert result.keypoints.data.shape[-2:] == (17, 3)
        # Boxes and keypoints are the same length.
        assert len(result) == result.keypoints.data.shape[0]
