"""Unit tests for LibreYOLONAS pose support.

Covers the precedent-setting pieces the
``libreyolo-add-native-pose-model`` skill calls out:

- filename ``-pose`` suffix resolves to ``task='pose'``
- explicit-vs-checkpoint task conflicts raise clearly
- pose checkpoints route to the pose model, detection checkpoints don't
- ``Results(boxes=, keypoints=)`` exposes ``result.keypoints``
- ``_select`` preserves the box ↔ keypoint alignment after filtering
- pose forward + postprocess produce a valid ``Results`` on synthetic input
- detection-mode YOLO-NAS still wires through cleanly (no regression)
"""

from __future__ import annotations

import pytest
import torch

from libreyolo.models.yolonas.model import LibreYOLONAS
from libreyolo.models.yolonas.nn import LibreYOLONASPoseModel
from libreyolo.models.yolonas.utils import postprocess_pose
from libreyolo.tasks import resolve_task
from libreyolo.utils.results import Boxes, Keypoints, Results

pytestmark = [pytest.mark.unit, pytest.mark.yolonas]


# ---------------------------------------------------------------------------
# Filename suffix + task resolution
# ---------------------------------------------------------------------------


class TestFilenameTaskResolution:
    def test_pose_suffix_resolves_to_pose_task(self):
        assert LibreYOLONAS.detect_task_from_filename("LibreYOLONASn-pose.pt") == "pose"
        assert LibreYOLONAS.detect_task_from_filename("LibreYOLONASs-pose.pt") == "pose"

    def test_no_suffix_resolves_to_none_task(self):
        # detect_task_from_filename returns None when no recognized suffix.
        assert LibreYOLONAS.detect_task_from_filename("LibreYOLONASs.pt") is None

    def test_size_detection_for_pose_filenames(self):
        # All pose sizes including 'n' (which is pose-only, not in detect INPUT_SIZES).
        for size in ("n", "s", "m", "l"):
            fn = f"LibreYOLONAS{size}-pose.pt"
            assert LibreYOLONAS.detect_size_from_filename(fn) == size

    def test_size_detection_for_detect_filenames(self):
        for size in ("s", "m", "l"):
            fn = f"LibreYOLONAS{size}.pt"
            assert LibreYOLONAS.detect_size_from_filename(fn) == size

    def test_unsupported_task_raises(self):
        with pytest.raises(ValueError, match="not supported"):
            resolve_task(
                explicit_task="classify",
                supported_tasks=LibreYOLONAS.SUPPORTED_TASKS,
            )

    def test_pose_in_supported_tasks(self):
        assert "pose" in LibreYOLONAS.SUPPORTED_TASKS
        assert "detect" in LibreYOLONAS.SUPPORTED_TASKS


# ---------------------------------------------------------------------------
# Pose checkpoint discrimination
# ---------------------------------------------------------------------------


class TestPoseCheckpointDiscrimination:
    def test_pose_state_dict_detected(self):
        sd = {"heads.head1.pose_pred.weight": torch.zeros(34, 48, 1, 1)}
        assert LibreYOLONAS.is_pose_state_dict(sd) is True

    def test_detect_state_dict_not_pose(self):
        sd = {"heads.head1.cls_pred.weight": torch.zeros(80, 64, 1, 1)}
        assert LibreYOLONAS.is_pose_state_dict(sd) is False

    def test_size_detection_from_pose_state_dict(self):
        # Pose-n: bbox_inter=48 (width_mult 0.33 of 128), cls_pred outputs 1+17=18
        sd = {
            "heads.head1.cls_pred.weight": torch.zeros(18, 48, 1, 1),
            "heads.head1.pose_pred.weight": torch.zeros(34, 48, 1, 1),
        }
        assert LibreYOLONAS.detect_size(sd) == "n"
        # Pose head implies single-class detection.
        assert LibreYOLONAS.detect_nb_classes(sd) == 1

    def test_size_detection_from_detect_state_dict(self):
        # Detect-s: bbox_inter=64, cls_pred outputs 80
        sd = {"heads.head1.cls_pred.weight": torch.zeros(80, 64, 1, 1)}
        assert LibreYOLONAS.detect_size(sd) == "s"
        assert LibreYOLONAS.detect_nb_classes(sd) == 80


# ---------------------------------------------------------------------------
# Family-class wiring (random weights — no checkpoint download required)
# ---------------------------------------------------------------------------


class TestPoseFamilyClassWiring:
    def test_pose_init_sets_task_and_metadata(self):
        model = LibreYOLONAS(model_path=None, size="n", task="pose")
        assert model.task == "pose"
        assert model.family == "yolonas"
        assert model.nb_classes == 1
        assert model.names == {0: "person"}
        assert isinstance(model.model, LibreYOLONASPoseModel)

    def test_detect_init_unchanged(self):
        model = LibreYOLONAS(model_path=None, size="s")
        assert model.task == "detect"
        assert model.nb_classes == 80
        assert not isinstance(model.model, LibreYOLONASPoseModel)

    def test_pose_n_size_only_valid_for_pose(self):
        with pytest.raises(ValueError, match="size"):
            LibreYOLONAS(model_path=None, size="n")

    def test_classify_task_rejected(self):
        with pytest.raises(ValueError, match="not supported"):
            LibreYOLONAS(model_path=None, size="s", task="classify")


# ---------------------------------------------------------------------------
# Forward + postprocess shape contract
# ---------------------------------------------------------------------------


class TestPoseForwardAndPostprocess:
    @pytest.fixture(scope="class")
    def pose_model(self):
        m = LibreYOLONAS(model_path=None, size="n", task="pose")
        m.model.eval()
        return m

    def test_forward_output_shape(self, pose_model):
        x = torch.randn(1, 3, 640, 640).to(pose_model.device)
        with torch.no_grad():
            out = pose_model._forward(x)
        assert set(out) == {"boxes", "scores", "keypoints_xy", "keypoints_conf"}
        assert out["boxes"].shape == (1, 8400, 4)
        assert out["scores"].shape == (1, 8400, 1)
        assert out["keypoints_xy"].shape == (1, 8400, 17, 2)
        assert out["keypoints_conf"].shape == (1, 8400, 17)

    def test_postprocess_emits_keypoints(self, pose_model):
        x = torch.randn(1, 3, 640, 640).to(pose_model.device)
        with torch.no_grad():
            raw = pose_model._forward(x)
        det = postprocess_pose(
            raw,
            conf_thres=0.0,
            iou_thres=0.7,
            input_size=640,
            original_size=(800, 600),
            post_nms_max_predictions=20,
        )
        assert "keypoints" in det
        assert det["keypoints"].shape[-1] == 3  # x, y, conf
        assert det["keypoints"].shape[-2] == 17  # COCO keypoints
        # Boxes and keypoints have matching first dim.
        assert det["keypoints"].shape[0] == det["boxes"].shape[0]


# ---------------------------------------------------------------------------
# Results plumbing
# ---------------------------------------------------------------------------


class TestResultsKeypointsPlumbing:
    @staticmethod
    def _make_results(n: int) -> Results:
        boxes_t = torch.rand((n, 4)) * 100
        boxes_t[:, 2:] += boxes_t[:, :2]  # ensure x2 > x1, y2 > y1
        conf = torch.rand((n,))
        cls = torch.zeros((n,))
        kpts = torch.rand((n, 17, 3)) * 100
        return Results(
            boxes=Boxes(boxes_t, conf, cls),
            orig_shape=(640, 640),
            keypoints=Keypoints(kpts, (640, 640)),
        )

    def test_keypoints_attribute_present(self):
        r = self._make_results(3)
        assert r.keypoints is not None
        assert r.keypoints.data.shape == (3, 17, 3)
        # Convenience accessors
        assert r.keypoints.xy.shape == (3, 17, 2)
        assert r.keypoints.conf.shape == (3, 17)
        assert r.keypoints.has_visible.shape == (3, 17)

    def test_select_preserves_box_keypoint_alignment(self):
        r = self._make_results(5)
        idx = torch.tensor([0, 2, 4])
        sub = r._select(idx)
        assert len(sub) == 3
        assert sub.keypoints is not None
        assert sub.keypoints.data.shape == (3, 17, 3)
        # Boxes and keypoints should reference the same instances.
        torch.testing.assert_close(
            sub.boxes.xyxy, r.boxes.xyxy[idx]
        )
        torch.testing.assert_close(
            sub.keypoints.data, r.keypoints.data[idx]
        )

    def test_unsupported_slots_default_to_none(self):
        r = self._make_results(1)
        assert r.masks is None
        assert r.probs is None
        assert r.obb is None

    def test_xyn_normalization(self):
        r = self._make_results(2)
        xyn = r.keypoints.xyn
        h, w = r.orig_shape
        torch.testing.assert_close(
            xyn, r.keypoints.xy / torch.tensor([w, h], dtype=xyn.dtype)
        )


# ---------------------------------------------------------------------------
# Class filter alignment (boxes ↔ keypoints)
# ---------------------------------------------------------------------------


class TestApplyClassesFilterPreservesKeypoints:
    def test_filter_keeps_keypoints_aligned(self):
        from libreyolo.models.base.inference import InferenceRunner

        boxes = torch.tensor(
            [[0, 0, 10, 10], [20, 20, 30, 30], [40, 40, 50, 50]],
            dtype=torch.float32,
        )
        conf = torch.tensor([0.9, 0.8, 0.7])
        cls = torch.tensor([0.0, 1.0, 0.0])
        kpts = torch.arange(3 * 17 * 3, dtype=torch.float32).reshape(3, 17, 3)

        out_boxes, _, _, masks_out, kpts_out = InferenceRunner._apply_classes_filter(
            boxes, conf, cls, [0], None, kpts
        )
        assert masks_out is None
        assert len(out_boxes) == 2
        # Class 0 was at indices 0 and 2 — those keypoints must survive.
        torch.testing.assert_close(kpts_out, kpts[[0, 2]])
