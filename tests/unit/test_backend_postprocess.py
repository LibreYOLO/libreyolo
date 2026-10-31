from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from PIL import Image

import libreyolo.backends.base as backend_base
from libreyolo.backends.base import BaseBackend

pytestmark = pytest.mark.unit


class _DummyBackend(BaseBackend):
    def __init__(
        self,
        model_family: str,
        task: str | None = None,
        supported_tasks=("detect",),
        model_size: str | None = None,
        imgsz=640,
        **pose_metadata,
    ):
        super().__init__(
            model_path="dummy",
            nb_classes=2,
            device="cpu",
            imgsz=imgsz,
            model_family=model_family,
            model_size=model_size,
            names={0: "class_0", 1: "class_1"},
            task=task,
            supported_tasks=supported_tasks,
            **pose_metadata,
        )

    def _run_inference(self, blob: np.ndarray) -> list:
        raise NotImplementedError


def test_matte_backend_decodes_logits_and_resizes():
    backend = _DummyBackend(
        "birefnet", task="matte", supported_tasks=("matte",), imgsz=16
    )
    logits = np.zeros((1, 1, 16, 16), dtype=np.float32)
    logits[..., 4:12, 4:12] = 20.0
    result = backend._build_matte_result(
        [logits], orig_shape=(8, 10), original_size=(10, 8), image_path=None
    )

    assert result.matte.data.shape == (8, 10)
    assert float(result.matte.data.max()) > 0.99
    assert float(result.matte.data.min()) == pytest.approx(0.5)


def test_gaze_backend_decodes_head_logits_for_full_crop():
    backend = _DummyBackend(
        "l2cs",
        task="gaze",
        supported_tasks=("gaze",),
        imgsz=448,
        num_bins=90,
        bin_width_deg=4.0,
        offset_deg=-180.0,
    )
    yaw = np.full((1, 90), -50.0, dtype=np.float32)
    pitch = np.full((1, 90), -50.0, dtype=np.float32)
    yaw[0, 80] = 50.0
    pitch[0, 10] = 50.0
    result = backend._build_gaze_result(
        [yaw, pitch], orig_shape=(64, 80), image_path=None
    )

    assert result.boxes.xyxy.tolist() == [[0.0, 0.0, 80.0, 64.0]]
    assert float(result.gaze.pitch_deg[0]) == pytest.approx(-140.0, abs=1e-4)
    assert float(result.gaze.yaw_deg[0]) == pytest.approx(140.0, abs=1e-4)


def test_embedding_backend_normalizes_vectors_and_builds_results():
    backend = _DummyBackend(
        "dinov2",
        task="embed",
        supported_tasks=("embed",),
        imgsz=224,
    )
    result = backend._build_embedding_result(
        [np.array([[3.0, 4.0]], dtype=np.float32)],
        orig_shape=(80, 120),
        image_path=None,
    )

    assert result.boxes is None
    assert result.embeddings.data.shape == (1, 2)
    torch.testing.assert_close(
        result.embeddings.data,
        torch.tensor([[0.6, 0.8]], dtype=torch.float32),
    )


def test_depth_anything3_backend_applies_sky_correction_and_inverse_depth():
    depth = np.full((1, 1, 8, 8), 2.0, dtype=np.float32)
    sky = np.zeros_like(depth)
    depth[..., :4, :] = 100.0
    sky[..., :4, :] = 1.0

    parsed = BaseBackend._parse_depth_anything3_output(
        [depth, sky], original_size=(8, 8)
    )

    torch.testing.assert_close(parsed, torch.full((8, 8), 0.5))


def test_removed_family_export_is_rejected():
    """A removed-family (DAMO-YOLO) exported artifact must fail loudly instead of
    silently falling through to YOLO9 preprocessing/parsing."""
    with pytest.raises(ValueError, match="no longer supported"):
        _DummyBackend("damoyolo")


def test_dfine_backend_skips_generic_nms():
    backend = _DummyBackend("dfine")

    boxes = np.array([[0, 0, 10, 10], [0, 0, 10, 10]], dtype=np.float32)
    scores = np.array([0.9, 0.8], dtype=np.float32)
    classes = np.array([0, 1], dtype=np.int64)

    result = backend._build_result(
        boxes,
        scores,
        classes,
        orig_shape=(10, 10),
        image_path=None,
        iou=0.45,
        classes=None,
        max_det=300,
    )

    assert len(result.boxes) == 2


def test_rfdetr_backend_skips_generic_nms():
    backend = _DummyBackend("rfdetr")

    boxes = np.array([[0, 0, 10, 10], [0, 0, 10, 10]], dtype=np.float32)
    scores = np.array([0.9, 0.8], dtype=np.float32)
    classes = np.array([0, 1], dtype=np.int64)

    result = backend._build_result(
        boxes,
        scores,
        classes,
        orig_shape=(10, 10),
        image_path=None,
        iou=0.45,
        classes=None,
        max_det=300,
    )

    assert len(result.boxes) == 2


def test_yolo9_e2e_backend_skips_generic_nms():
    backend = _DummyBackend("yolo9_e2e")

    boxes = np.array([[0, 0, 10, 10], [0, 0, 10, 10]], dtype=np.float32)
    scores = np.array([0.9, 0.8], dtype=np.float32)
    classes = np.array([0, 0], dtype=np.int64)

    result = backend._build_result(
        boxes,
        scores,
        classes,
        orig_shape=(10, 10),
        image_path=None,
        iou=0.45,
        classes=None,
        max_det=300,
    )

    assert len(result.boxes) == 2


def test_yolo9_e2e_backend_uses_native_topk_anchor_ranking():
    backend = _DummyBackend("yolo9_e2e")
    output = np.zeros((1, 6, 3), dtype=np.float32)
    output[0, :4, :] = np.array(
        [
            [0, 10, 20],
            [0, 10, 20],
            [5, 15, 25],
            [5, 15, 25],
        ],
        dtype=np.float32,
    )
    output[0, 4:, :] = np.array(
        [
            [0.89, 0.92, 0.50],
            [0.91, 0.10, 0.99],
        ],
        dtype=np.float32,
    )

    boxes, scores, classes = backend._parse_yolo9(
        [output],
        effective_imgsz=100,
        orig_w=100,
        orig_h=100,
        conf=0.001,
        max_det=1,
    )

    assert len(boxes) == 1
    assert classes.tolist() == [1]
    np.testing.assert_allclose(scores, [0.99])
    np.testing.assert_allclose(boxes[0], [20, 20, 25, 25])


def test_yolox_backend_recomputes_validation_letterbox_ratio():
    backend = _DummyBackend("yolox")
    # Canvas-space xywh for an original 200x100 image letterboxed to 100x100.
    output = np.array([[[50.0, 25.0, 20.0, 10.0, 0.9, 0.8, 0.1]]], dtype=np.float32)

    boxes, scores, classes = backend._parse_yolox(
        [output],
        effective_imgsz=100,
        orig_w=200,
        orig_h=100,
        conf=0.5,
        ratio=1.0,
    )

    np.testing.assert_array_equal(classes, [0])
    np.testing.assert_allclose(scores, [0.72], rtol=1e-6)
    np.testing.assert_allclose(boxes, [[80.0, 40.0, 120.0, 60.0]])


def test_rtmdet_backend_keeps_multilabel_candidates_and_class_aware_nms():
    backend = _DummyBackend("rtmdet")
    output = np.array(
        [[[10.0, 10.0, 30.0, 30.0, 0.95, 0.9]]],
        dtype=np.float32,
    )

    boxes, scores, classes = backend._parse_rtmdet(
        [output],
        effective_imgsz=64,
        orig_w=64,
        orig_h=64,
        conf=0.5,
    )
    result = backend._build_result(
        boxes,
        scores,
        classes,
        orig_shape=(64, 64),
        image_path=None,
        iou=0.45,
        classes=None,
        max_det=300,
    )

    assert len(result.boxes) == 2
    np.testing.assert_array_equal(result.boxes.cls.numpy().astype(np.int64), [0, 1])


def test_yolonas_backend_undoes_center_padding():
    backend = _DummyBackend("yolonas")
    # Original 1000x500 image: resize to 636x318, center-pad in 640 canvas.
    boxes_out = np.array([[[65.6, 192.8, 129.2, 256.4]]], dtype=np.float32)
    scores_out = np.array([[[0.9, 0.1]]], dtype=np.float32)

    boxes, scores, classes = backend._parse_yolonas(
        [boxes_out, scores_out],
        effective_imgsz=640,
        orig_w=1000,
        orig_h=500,
        conf=0.5,
    )

    np.testing.assert_array_equal(classes, [0])
    np.testing.assert_allclose(scores, [0.9], rtol=1e-6)
    np.testing.assert_allclose(boxes, [[100.0, 50.0, 200.0, 150.0]], atol=1e-4)


def test_rfdetr_backend_uses_topk_over_queries_and_classes():
    backend = _DummyBackend("rfdetr")

    boxes = np.array(
        [[[0.5, 0.5, 0.25, 0.25], [0.25, 0.25, 0.1, 0.1]]],
        dtype=np.float32,
    )
    logits = np.array([[[10.0, 9.0], [-10.0, -10.0]]], dtype=np.float32)

    parsed_boxes, scores, classes, masks = backend._parse_rfdetr(
        [boxes, logits],
        orig_w=100,
        orig_h=100,
        conf=0.5,
    )

    assert masks is None
    assert len(parsed_boxes) == 2
    assert classes.tolist() == [0, 1]
    assert scores[0] > scores[1] > 0.5
    np.testing.assert_allclose(parsed_boxes[0], [37.5, 37.5, 62.5, 62.5])
    np.testing.assert_allclose(parsed_boxes[1], [37.5, 37.5, 62.5, 62.5])


def test_rfdetr_obb_backend_parses_angle_output():
    backend = _DummyBackend(
        "rfdetr",
        task="obb",
        supported_tasks=("detect", "segment", "obb"),
    )
    boxes = np.array([[[0.5, 0.25, 0.2, 0.1]]], dtype=np.float32)
    logits = np.array([[[0.0, 10.0]]], dtype=np.float32)
    angles = np.array([[[0.3]]], dtype=np.float32)

    parsed_boxes, scores, classes, masks, obb = backend._parse_rfdetr(
        [boxes, logits, angles],
        orig_w=200,
        orig_h=100,
        conf=0.5,
    )

    assert masks is None
    assert classes.tolist() == [1]
    np.testing.assert_allclose(parsed_boxes[0], [80.0, 20.0, 120.0, 30.0])
    np.testing.assert_allclose(
        obb[0],
        [100.0, 25.0, 40.0, 10.0, 0.3, scores[0], 1.0],
        rtol=1e-6,
        atol=1e-6,
    )


def test_rfdetr_pose_backend_parses_keypoints_not_masks():
    backend = _DummyBackend(
        "rfdetr",
        task="pose",
        supported_tasks=("detect", "pose"),
    )
    boxes = np.array(
        [[[0.5, 0.5, 0.2, 0.4], [0.25, 0.25, 0.1, 0.1]]],
        dtype=np.float32,
    )
    logits = np.array([[[10.0], [-10.0]]], dtype=np.float32)
    keypoints = np.zeros((1, 2, 2, 3), dtype=np.float32)
    keypoints[0, 0, :, 0] = [0.25, 0.75]
    keypoints[0, 0, :, 1] = [0.5, 0.25]
    keypoints[0, 0, :, 2] = [2.0, -2.0]

    parsed_boxes, scores, classes, masks, obb, parsed_keypoints = backend._parse_rfdetr(
        [boxes, logits, keypoints],
        orig_w=200,
        orig_h=100,
        conf=0.5,
    )

    assert masks is None
    assert obb is None
    assert classes.tolist() == [0]
    assert parsed_boxes.shape == (1, 4)
    assert parsed_keypoints.shape == (1, 2, 3)
    np.testing.assert_allclose(parsed_keypoints[0, :, 0], [50.0, 150.0])
    np.testing.assert_allclose(parsed_keypoints[0, :, 1], [50.0, 25.0])
    np.testing.assert_allclose(
        parsed_keypoints[0, :, 2],
        [0.880797, 0.119203],
        rtol=1e-5,
    )
    assert scores[0] > 0.99


def test_rfdetr_pose_backend_decodes_grouppose_keypoint_slots():
    backend = _DummyBackend(
        "rfdetr",
        task="pose",
        supported_tasks=("detect", "pose"),
        num_keypoints_per_class=[0, 17],
    )
    backend.nb_classes = 1
    backend.names = {0: "person"}
    boxes = np.array([[[0.5, 0.5, 0.2, 0.4]]], dtype=np.float32)
    logits = np.array([[[0.0, 10.0]]], dtype=np.float32)
    keypoints = np.zeros((1, 1, 34, 8), dtype=np.float32)
    keypoints[0, 0, 17, :7] = [0.25, 0.5, 2.0, 0.0, 0.0, 1.0, 0.0]
    keypoints[0, 0, 18, :7] = [0.75, 0.25, -2.0, 0.0, 0.0, 1.0, 0.0]

    parsed_boxes, scores, classes, masks, obb, parsed_keypoints = backend._parse_rfdetr(
        [boxes, logits, keypoints],
        orig_w=200,
        orig_h=100,
        conf=0.5,
    )

    assert masks is None
    assert obb is None
    assert classes.tolist() == [0]
    assert parsed_boxes.shape == (1, 4)
    assert parsed_keypoints.shape == (1, 17, 3)
    np.testing.assert_allclose(parsed_keypoints[0, :2, 0], [50.0, 150.0])
    np.testing.assert_allclose(parsed_keypoints[0, :2, 1], [50.0, 25.0])
    np.testing.assert_allclose(
        parsed_keypoints[0, :2, 2],
        [0.880797, 0.119203],
        rtol=1e-5,
    )
    assert 0.7 < scores[0] < 1.0


def test_rfdetr_grouppose_backend_honors_requested_max_det():
    backend = _DummyBackend(
        "rfdetr",
        task="pose",
        supported_tasks=("detect", "pose"),
        model_size="x",
        num_keypoints_per_class=[0, 17],
    )
    backend.nb_classes = 1
    backend.names = {0: "person"}
    boxes = np.array([[[0.5, 0.5, 0.2, 0.4]]], dtype=np.float32)
    logits = np.array([[[10.0, 9.0]]], dtype=np.float32)
    keypoints = np.zeros((1, 1, 34, 8), dtype=np.float32)
    keypoints[0, 0, 17, :7] = [0.25, 0.5, 2.0, 0.0, 0.0, 1.0, 0.0]

    parsed_boxes, scores, classes, masks, obb, parsed_keypoints = (
        backend._parse_outputs(
            [boxes, logits, keypoints],
            64,
            (200, 100),
            conf=0.0,
            max_det=1,
        )
    )

    assert masks is None
    assert obb is None
    assert parsed_boxes.shape == (0, 4)
    assert scores.shape == (0,)
    assert classes.shape == (0,)
    assert parsed_keypoints.shape == (0, 17, 3)


def test_rfdetr_pose_backend_uses_exported_grouppose_schema():
    backend = _DummyBackend(
        "rfdetr",
        task="pose",
        supported_tasks=("detect", "pose"),
    )
    backend.nb_classes = 2
    backend.names = {0: "person", 1: "tool"}
    backend.num_keypoints_per_class = [0, 17, 4]

    boxes = np.array([[[0.5, 0.5, 0.2, 0.4]]], dtype=np.float32)
    logits = np.array([[[-10.0, -10.0, 10.0]]], dtype=np.float32)
    keypoints = np.zeros((1, 1, 51, 8), dtype=np.float32)
    class_offset = 2 * 17
    keypoints[0, 0, class_offset : class_offset + 4, :7] = np.array(
        [
            [0.25, 0.50, 2.0, 0.0, 0.0, 0.0, 0.0],
            [0.75, 0.25, 2.0, 0.0, 0.0, 0.0, 0.0],
            [0.50, 0.75, 2.0, 0.0, 0.0, 0.0, 0.0],
            [0.10, 0.10, 2.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    keypoints[0, 0, class_offset + 4 : class_offset + 17, 2] = 10.0
    keypoints[0, 0, class_offset + 4 : class_offset + 17, 4] = -10.0
    keypoints[0, 0, class_offset + 4 : class_offset + 17, 6] = -10.0

    parsed_boxes, scores, classes, masks, obb, parsed_keypoints = backend._parse_rfdetr(
        [boxes, logits, keypoints],
        orig_w=200,
        orig_h=100,
        conf=0.5,
    )

    assert masks is None
    assert obb is None
    assert classes.tolist() == [1]
    assert parsed_boxes.shape == (1, 4)
    assert parsed_keypoints.shape == (1, 17, 3)
    assert scores[0] > 0.5
    np.testing.assert_allclose(parsed_keypoints[0, :4, 0], [50.0, 150.0, 100.0, 20.0])
    np.testing.assert_allclose(parsed_keypoints[0, 4:, :], 0.0)


def test_rfdetr_pose_backend_postprocess_returns_keypoints():
    backend = _DummyBackend(
        "rfdetr",
        task="pose",
        supported_tasks=("detect", "pose"),
    )
    boxes = np.array([[[0.5, 0.5, 0.2, 0.4]]], dtype=np.float32)
    logits = np.array([[[10.0]]], dtype=np.float32)
    keypoints = np.array(
        [[[[0.25, 0.5, 2.0], [0.75, 0.25, -2.0]]]],
        dtype=np.float32,
    )

    out = backend._postprocess(
        [boxes, logits, keypoints],
        conf_thres=0.5,
        iou_thres=0.5,
        original_size=(200, 100),
        input_size=64,
    )

    assert out["num_detections"] == 1
    assert "masks" not in out
    assert "keypoints" in out
    assert out["keypoints"].shape == (1, 2, 3)
    np.testing.assert_allclose(out["keypoints"][0, :, 0].numpy(), [50.0, 150.0])
    np.testing.assert_allclose(out["keypoints"][0, :, 1].numpy(), [50.0, 25.0])


def test_rfdetr_seg_backend_uses_variant_num_select():
    backend = _DummyBackend(
        "rfdetr",
        task="segment",
        supported_tasks=("segment",),
        model_size="n",
    )
    num_queries = 150
    boxes = np.tile(
        np.array([[0.5, 0.5, 0.25, 0.25]], dtype=np.float32),
        (1, num_queries, 1),
    )
    logits = np.linspace(10.0, 1.0, num_queries, dtype=np.float32).reshape(
        1, num_queries, 1
    )
    masks = np.ones((1, num_queries, 4, 4), dtype=np.float32)

    parsed_boxes, scores, classes, parsed_masks = backend._parse_rfdetr(
        [boxes, logits, masks],
        orig_w=16,
        orig_h=16,
        conf=0.5,
    )

    assert len(parsed_boxes) == 100
    assert len(scores) == 100
    assert classes.tolist() == [0] * 100
    assert parsed_masks.shape == (100, 16, 16)


def test_dfine_segment_backend_parses_probability_masks():
    backend = _DummyBackend(
        "dfine",
        task="segment",
        supported_tasks=("detect", "segment"),
    )
    logits = np.array([[[10.0], [-10.0]]], dtype=np.float32)
    boxes = np.array(
        [[[0.5, 0.5, 0.5, 0.5], [0.1, 0.1, 0.1, 0.1]]],
        dtype=np.float32,
    )
    masks = np.ones((1, 2, 4, 4), dtype=np.float32)

    parsed_boxes, scores, classes, parsed_masks = backend._parse_outputs(
        [logits, boxes, masks],
        effective_imgsz=64,
        original_size=(8, 8),
        conf=0.5,
        max_det=2,
    )

    assert parsed_boxes.shape == (1, 4)
    np.testing.assert_allclose(parsed_boxes[0], [2.0, 2.0, 6.0, 6.0])
    assert scores[0] > 0.99
    np.testing.assert_array_equal(classes, [0])
    assert parsed_masks.shape == (1, 8, 8)
    assert parsed_masks[0, 4, 4]
    assert not parsed_masks[0, 0, 0]


def test_dfine_segment_backend_uses_input_size_mask_resize_path():
    backend = _DummyBackend(
        "dfine",
        task="segment",
        supported_tasks=("detect", "segment"),
        imgsz=6,
    )
    logits = np.array([[[10.0]]], dtype=np.float32)
    boxes = np.array([[[0.5, 0.5, 1.0, 1.0]]], dtype=np.float32)
    masks = np.array(
        [
            [
                [
                    [0.4963, 0.7682, 0.0885],
                    [0.1320, 0.3074, 0.6341],
                    [0.4901, 0.8964, 0.4556],
                ]
            ]
        ],
        dtype=np.float32,
    )

    _, _, _, parsed_masks = backend._parse_outputs(
        [logits, boxes, masks],
        effective_imgsz=6,
        original_size=(5, 7),
        conf=0.5,
        max_det=1,
    )

    mask_t = torch.from_numpy(masks)
    two_step = F.interpolate(
        F.interpolate(mask_t, size=(6, 6), mode="bilinear", align_corners=False),
        size=(7, 5),
        mode="bilinear",
        align_corners=False,
    )[0, 0].numpy()
    direct = F.interpolate(
        mask_t,
        size=(7, 5),
        mode="bilinear",
        align_corners=False,
    )[0, 0].numpy()

    assert not np.array_equal(two_step >= 0.5, direct >= 0.5)
    np.testing.assert_array_equal(parsed_masks[0], two_step >= 0.5)


def test_yolonas_pose_backend_uses_pose_preprocessor(monkeypatch):
    backend = _DummyBackend(
        "yolonas",
        task="pose",
        supported_tasks=("detect", "pose"),
    )
    calls = []

    def fake_pose_preprocess(image, input_size, color_format):
        calls.append((image, input_size, color_format))
        return "tensor", "image", (10, 20), 1.0

    monkeypatch.setattr(
        backend_base, "yolonas_preprocess_pose_image", fake_pose_preprocess
    )

    out = backend._preprocess("source.jpg", 640, "auto")

    assert out == ("tensor", "image", (10, 20), 1.0)
    assert calls == [("source.jpg", 640, "auto")]


def test_deimv2_backend_uses_dino_val_preprocessor_for_dino_sizes():
    from libreyolo.validation.preprocessors import (
        DEIMv2DINOValPreprocessor,
        DEIMv2ValPreprocessor,
    )

    dino_backend = _DummyBackend("deimv2", model_size="s")
    hgnet_backend = _DummyBackend("deimv2", model_size="n")

    assert isinstance(
        dino_backend._get_val_preprocessor(img_size=640), DEIMv2DINOValPreprocessor
    )
    assert isinstance(
        hgnet_backend._get_val_preprocessor(img_size=640), DEIMv2ValPreprocessor
    )
    assert not isinstance(
        hgnet_backend._get_val_preprocessor(img_size=640),
        DEIMv2DINOValPreprocessor,
    )


def test_ec_segment_backend_parses_masks():
    backend = _DummyBackend(
        "ec",
        task="segment",
        supported_tasks=("detect", "segment"),
    )
    logits = np.array([[[-10.0, 10.0], [-10.0, -10.0]]], dtype=np.float32)
    boxes = np.array([[[0.5, 0.5, 0.25, 0.5], [0.1, 0.1, 0.1, 0.1]]], dtype=np.float32)
    masks = np.array(
        [[[[1.0, 1.0], [1.0, 1.0]], [[-1.0, -1.0], [-1.0, -1.0]]]],
        dtype=np.float32,
    )

    parsed_boxes, scores, classes, parsed_masks = backend._parse_outputs(
        [logits, boxes, masks], 64, (200, 100), conf=0.5
    )

    assert parsed_boxes.shape == (1, 4)
    np.testing.assert_allclose(parsed_boxes[0], [75.0, 25.0, 125.0, 75.0])
    assert scores[0] > 0.99
    np.testing.assert_array_equal(classes, [1])
    assert parsed_masks.shape == (1, 100, 200)
    assert parsed_masks[0].all()


def test_ec_segment_backend_does_not_clip_boxes():
    backend = _DummyBackend(
        "ec",
        task="segment",
        supported_tasks=("detect", "segment"),
    )
    logits = np.array([[[10.0]]], dtype=np.float32)
    boxes = np.array([[[0.05, 0.5, 0.3, 0.5]]], dtype=np.float32)
    masks = np.ones((1, 1, 2, 2), dtype=np.float32)

    parsed_boxes, scores, classes, parsed_masks = backend._parse_outputs(
        [logits, boxes, masks], 64, (200, 100), conf=0.5
    )

    np.testing.assert_allclose(parsed_boxes, [[-20.0, 25.0, 40.0, 75.0]])
    assert scores[0] > 0.99
    np.testing.assert_array_equal(classes, [0])
    assert parsed_masks.shape == (1, 100, 200)


@pytest.mark.parametrize("family", ("dfine", "rtdetrv4", "rtdetr", "rtdetrv2"))
def test_detr_detection_backend_does_not_clip_boxes(family):
    backend = _DummyBackend(family)
    logits = np.array([[[10.0]]], dtype=np.float32)
    boxes = np.array([[[0.05, 0.5, 0.3, 0.5]]], dtype=np.float32)

    parsed_boxes, scores, classes, masks = backend._parse_outputs(
        [logits, boxes], 64, (200, 100), conf=0.5
    )

    np.testing.assert_allclose(parsed_boxes, [[-20.0, 25.0, 40.0, 75.0]])
    assert scores[0] > 0.99
    np.testing.assert_array_equal(classes, [0])
    assert masks is None


def test_ec_segment_backend_honors_max_det():
    backend = _DummyBackend(
        "ec",
        task="segment",
        supported_tasks=("detect", "segment"),
    )
    logits = np.array([[[10.0], [9.0], [8.0]]], dtype=np.float32)
    boxes = np.array(
        [[[0.1, 0.1, 0.1, 0.1], [0.5, 0.5, 0.2, 0.2], [0.8, 0.8, 0.1, 0.1]]],
        dtype=np.float32,
    )
    masks = np.ones((1, 3, 2, 2), dtype=np.float32)

    parsed_boxes, scores, classes, parsed_masks = backend._parse_outputs(
        [logits, boxes, masks], 64, (100, 100), conf=0.5, max_det=1
    )

    assert parsed_boxes.shape == (1, 4)
    assert scores.shape == (1,)
    assert classes.tolist() == [0]
    assert parsed_masks.shape == (1, 100, 100)


def test_ec_pose_backend_parses_flattened_keypoints():
    backend = _DummyBackend(
        "ec",
        task="pose",
        supported_tasks=("detect", "pose"),
    )
    # person is the LAST logit (index 1) of the 2-class ECPose head.
    logits = np.array([[[-10.0, 10.0], [-10.0, -10.0]]], dtype=np.float32)
    keypoints = np.array(
        [[[0.25, 0.5, 0.75, 0.25], [0.0, 0.0, 0.0, 0.0]]],
        dtype=np.float32,
    )

    boxes, scores, classes, masks, obb, parsed_keypoints = backend._parse_outputs(
        [logits, keypoints], 64, (200, 100), conf=0.5
    )

    assert masks is None
    assert obb is None
    np.testing.assert_array_equal(classes, [0])
    np.testing.assert_allclose(boxes, [[50.0, 25.0, 150.0, 50.0]])
    np.testing.assert_allclose(
        parsed_keypoints[0, :, :2], [[50.0, 50.0], [150.0, 25.0]]
    )
    np.testing.assert_allclose(parsed_keypoints[0, :, 2], [1.0, 1.0])
    assert scores[0] > 0.99


def test_ec_pose_backend_uses_exported_boxes_when_present():
    backend = _DummyBackend(
        "ec",
        task="pose",
        supported_tasks=("detect", "pose"),
    )
    logits = np.array([[[-10.0, 10.0]]], dtype=np.float32)  # person == last logit
    boxes = np.array([[[0.5, 0.5, 0.8, 0.6]]], dtype=np.float32)
    keypoints = np.array([[[0.45, 0.45, 0.55, 0.55]]], dtype=np.float32)

    parsed_boxes, scores, classes, masks, obb, parsed_keypoints = (
        backend._parse_outputs([logits, boxes, keypoints], 64, (200, 100), conf=0.5)
    )

    assert scores.shape == (1,)
    assert classes.tolist() == [0]
    assert masks is None
    assert obb is None
    np.testing.assert_allclose(parsed_boxes, [[20.0, 20.0, 180.0, 80.0]])
    np.testing.assert_allclose(
        parsed_keypoints[0, :, :2], [[90.0, 45.0], [110.0, 55.0]]
    )


def test_ec_pose_backend_does_not_clip_keypoints():
    backend = _DummyBackend(
        "ec",
        task="pose",
        supported_tasks=("detect", "pose"),
    )
    logits = np.array([[[-10.0, 10.0]]], dtype=np.float32)  # person == last logit
    keypoints = np.array([[[-0.25, 1.25, 0.5, 0.5]]], dtype=np.float32)

    boxes, _, _, _, _, parsed_keypoints = backend._parse_outputs(
        [logits, keypoints], 64, (200, 100), conf=0.5
    )

    np.testing.assert_allclose(
        parsed_keypoints[0, :, :2], [[-50.0, 125.0], [100.0, 50.0]]
    )
    np.testing.assert_allclose(boxes, [[-50.0, 50.0, 100.0, 125.0]])


def test_ec_pose_backend_honors_max_det():
    backend = _DummyBackend(
        "ec",
        task="pose",
        supported_tasks=("detect", "pose"),
    )
    logits = np.array([[[10.0], [9.0], [8.0]]], dtype=np.float32)
    keypoints = np.array(
        [[[0.1, 0.1, 0.2, 0.2], [0.3, 0.3, 0.4, 0.4], [0.5, 0.5, 0.6, 0.6]]],
        dtype=np.float32,
    )

    boxes, scores, classes, masks, obb, parsed_keypoints = backend._parse_outputs(
        [logits, keypoints], 64, (100, 100), conf=0.0, max_det=2
    )

    assert boxes.shape == (2, 4)
    assert scores.shape == (2,)
    assert classes.tolist() == [0, 0]
    assert masks is None
    assert obb is None
    assert parsed_keypoints.shape == (2, 2, 3)


def test_ec_pose_backend_caps_default_topk_to_query_count():
    backend = _DummyBackend(
        "ec",
        task="pose",
        supported_tasks=("detect", "pose"),
    )
    logits = np.ones((1, 60, 2), dtype=np.float32) * 10.0
    keypoints = np.zeros((1, 60, 34), dtype=np.float32)

    boxes, scores, classes, masks, obb, parsed_keypoints = backend._parse_outputs(
        [logits, keypoints], 64, (100, 100), conf=0.0, max_det=300
    )

    assert boxes.shape == (60, 4)
    assert scores.shape == (60,)
    assert classes.tolist() == [0] * 60
    assert masks is None
    assert obb is None
    assert parsed_keypoints.shape == (60, 17, 3)


def test_ec_pose_backend_selects_unique_queries_before_collapsing_classes():
    backend = _DummyBackend(
        "ec",
        task="pose",
        supported_tasks=("detect", "pose"),
    )
    logits = np.array([[[10.0, 9.0], [8.0, -10.0]]], dtype=np.float32)
    keypoints = np.array(
        [[[0.1, 0.1, 0.2, 0.2], [0.7, 0.7, 0.8, 0.8]]],
        dtype=np.float32,
    )

    boxes, scores, classes, masks, obb, parsed_keypoints = backend._parse_outputs(
        [logits, keypoints], 64, (100, 100), conf=0.0, max_det=2
    )

    assert boxes.shape == (2, 4)
    assert scores.shape == (2,)
    assert classes.tolist() == [0, 0]
    assert masks is None
    assert obb is None
    np.testing.assert_allclose(parsed_keypoints[:, 0, :2], [[10.0, 10.0], [70.0, 70.0]])


def test_ec_pose_backend_scores_person_logit_only():
    backend = _DummyBackend(
        "ec",
        task="pose",
        supported_tasks=("detect", "pose"),
    )
    # Person is the LAST logit (index 1): query 0 wins on its person logit
    # (10.0) over query 1 (-10.0); query 1's high index-0 logit (9.0) is ignored.
    logits = np.array([[[0.0, 10.0], [9.0, -10.0]]], dtype=np.float32)
    keypoints = np.array(
        [[[0.1, 0.1, 0.2, 0.2], [0.7, 0.7, 0.8, 0.8]]],
        dtype=np.float32,
    )

    boxes, scores, classes, masks, obb, parsed_keypoints = backend._parse_outputs(
        [logits, keypoints], 64, (100, 100), conf=0.0, max_det=1
    )

    assert boxes.shape == (1, 4)
    assert scores.shape == (1,)
    assert classes.tolist() == [0]
    assert masks is None
    assert obb is None
    np.testing.assert_allclose(parsed_keypoints[0, :, :2], [[10.0, 10.0], [20.0, 20.0]])


def test_ec_pose_backend_does_not_hard_cap_queries_at_sixty():
    backend = _DummyBackend(
        "ec",
        task="pose",
        supported_tasks=("detect", "pose"),
    )
    logits = np.linspace(10.0, 1.0, 70, dtype=np.float32).reshape(1, 70, 1)
    keypoints = np.zeros((1, 70, 34), dtype=np.float32)

    boxes, scores, classes, masks, obb, parsed_keypoints = backend._parse_outputs(
        [logits, keypoints], 64, (100, 100), conf=0.0, max_det=70
    )

    assert boxes.shape == (70, 4)
    assert scores.shape == (70,)
    assert classes.shape == (70,)
    assert masks is None
    assert obb is None
    assert parsed_keypoints.shape == (70, 17, 3)


def test_yolonas_pose_backend_parses_keypoints_and_bottom_right_letterbox():
    backend = _DummyBackend(
        "yolonas",
        task="pose",
        supported_tasks=("detect", "pose"),
    )
    boxes = np.array(
        [[[32.0, 32.0, 160.0, 128.0], [0.0, 0.0, 5.0, 5.0]]],
        dtype=np.float32,
    )
    scores = np.array([[[0.9], [0.1]]], dtype=np.float32)
    keypoints_xy = np.array(
        [[[[32.0, 64.0], [96.0, 128.0]], [[0.0, 0.0], [1.0, 1.0]]]],
        dtype=np.float32,
    )
    keypoints_conf = np.array([[[0.8, 0.7], [0.1, 0.1]]], dtype=np.float32)

    parsed_boxes, parsed_scores, classes, masks, obb, keypoints = (
        backend._parse_outputs(
            [boxes, scores, keypoints_xy, keypoints_conf],
            100,
            (200, 100),
            conf=0.5,
            ratio=None,
        )
    )

    assert masks is None
    assert obb is None
    np.testing.assert_array_equal(classes, [0])
    np.testing.assert_allclose(parsed_boxes, [[10.0, 10.0, 50.0, 40.0]])
    np.testing.assert_allclose(parsed_scores, [0.9])
    np.testing.assert_allclose(
        keypoints,
        [[[10.0, 20.0, 0.8], [30.0, 40.0, 0.7]]],
    )


def test_yolonas_pose_backend_does_not_clip_keypoints():
    backend = _DummyBackend(
        "yolonas",
        task="pose",
        supported_tasks=("detect", "pose"),
    )
    boxes = np.array([[[32.0, 32.0, 160.0, 128.0]]], dtype=np.float32)
    scores = np.array([[[0.9]]], dtype=np.float32)
    keypoints_xy = np.array([[[[-32.0, 64.0], [96.0, 800.0]]]], dtype=np.float32)
    keypoints_conf = np.array([[[0.8, 0.7]]], dtype=np.float32)

    _, _, _, _, _, keypoints = backend._parse_outputs(
        [boxes, scores, keypoints_xy, keypoints_conf],
        100,
        (200, 100),
        conf=0.5,
        ratio=None,
    )

    np.testing.assert_allclose(
        keypoints,
        [[[-10.0, 20.0, 0.8], [30.0, 250.0, 0.7]]],
    )


def test_yolonas_pose_backend_preselects_requested_max_det():
    backend = _DummyBackend(
        "yolonas",
        task="pose",
        supported_tasks=("detect", "pose"),
    )
    count = 1001
    boxes = np.tile(
        np.array([[32.0, 32.0, 160.0, 128.0]], dtype=np.float32),
        (1, count, 1),
    )
    scores = np.linspace(1.0, 0.1, count, dtype=np.float32).reshape(1, count, 1)
    keypoints_xy = np.zeros((1, count, 2, 2), dtype=np.float32)
    keypoints_conf = np.ones((1, count, 2), dtype=np.float32)

    parsed_boxes, parsed_scores, classes, masks, obb, keypoints = (
        backend._parse_outputs(
            [boxes, scores, keypoints_xy, keypoints_conf],
            100,
            (200, 100),
            conf=0.0,
            ratio=None,
            max_det=count,
        )
    )

    assert parsed_boxes.shape == (count, 4)
    assert parsed_scores.shape == (count,)
    assert classes.shape == (count,)
    assert masks is None
    assert obb is None
    assert keypoints.shape == (count, 2, 3)


def test_yolonas_backend_uses_preprocess_ratio_for_large_canvas():
    backend = _DummyBackend("yolonas")
    boxes_out = np.array([[[384.0, 512.0, 448.0, 576.0]]], dtype=np.float32)
    scores_out = np.array([[[0.9, 0.1]]], dtype=np.float32)

    boxes, scores, classes = backend._parse_yolonas(
        [boxes_out, scores_out],
        effective_imgsz=1280,
        orig_w=1000,
        orig_h=500,
        conf=0.5,
        ratio=0.64,
    )

    np.testing.assert_array_equal(classes, [0])
    np.testing.assert_allclose(scores, [0.9], rtol=1e-6)
    np.testing.assert_allclose(boxes, [[100.0, 50.0, 200.0, 150.0]], atol=1e-4)


def test_yolonas_backend_recomputes_ratio_when_validation_omits_it():
    backend = _DummyBackend("yolonas")
    ratio = 636.0 / 1000.0
    offset_x = 2.0
    offset_y = 161.0
    boxes_out = np.array(
        [
            [
                [
                    100.0 * ratio + offset_x,
                    50.0 * ratio + offset_y,
                    200.0 * ratio + offset_x,
                    150.0 * ratio + offset_y,
                ]
            ]
        ],
        dtype=np.float32,
    )
    scores_out = np.array([[[0.9, 0.1]]], dtype=np.float32)

    result = backend._postprocess(
        [boxes_out, scores_out],
        conf_thres=0.5,
        iou_thres=0.6,
        original_size=(1000, 500),
        input_size=640,
        letterbox=True,
        max_det=300,
    )

    assert result["num_detections"] == 1
    np.testing.assert_allclose(
        result["boxes"].numpy(),
        [[100.0, 50.0, 200.0, 150.0]],
        atol=1e-4,
    )
    np.testing.assert_allclose(result["scores"].numpy(), [0.9], rtol=1e-6)
    np.testing.assert_array_equal(result["classes"].numpy(), [0])


def test_yolo9_pose_backend_parses_keypoints():
    backend = _DummyBackend(
        "yolo9",
        task="pose",
        supported_tasks=("detect", "pose"),
    )
    pred = np.zeros((1, 5, 1), dtype=np.float32)
    pred[0, :4, 0] = [10.0, 20.0, 50.0, 60.0]
    pred[0, 4, 0] = 0.9
    keypoints = np.array([[[[10.0, 20.0, 0.8], [50.0, 60.0, 0.7]]]], dtype=np.float32)

    boxes, scores, classes, masks, obb, parsed_keypoints = backend._parse_outputs(
        [pred, keypoints], 100, (100, 100), conf=0.5
    )

    assert masks is None
    assert obb is None
    np.testing.assert_array_equal(classes, [0])
    np.testing.assert_allclose(boxes, [[10.0, 20.0, 50.0, 60.0]])
    np.testing.assert_allclose(scores, [0.9])
    np.testing.assert_allclose(parsed_keypoints, keypoints[0])


def test_rfdetr_seg_backend_uses_detected_size_for_num_select_without_metadata():
    backend = _DummyBackend(
        "rfdetr",
        task="segment",
        supported_tasks=("segment",),
        model_size=None,
    )
    backend.size = "n"
    num_queries = 150
    boxes = np.tile(
        np.array([[0.5, 0.5, 0.25, 0.25]], dtype=np.float32),
        (1, num_queries, 1),
    )
    logits = np.linspace(10.0, 1.0, num_queries, dtype=np.float32).reshape(
        1, num_queries, 1
    )
    masks = np.ones((1, num_queries, 4, 4), dtype=np.float32)

    parsed_boxes, scores, classes, parsed_masks = backend._parse_rfdetr(
        [boxes, logits, masks],
        orig_w=16,
        orig_h=16,
        conf=0.5,
    )

    assert len(parsed_boxes) == 100
    assert len(scores) == 100
    assert classes.tolist() == [0] * 100
    assert parsed_masks.shape == (100, 16, 16)


def test_rfdetr_x_pose_backend_uses_configured_num_select():
    backend = _DummyBackend(
        "rfdetr",
        task="pose",
        supported_tasks=("pose",),
        model_size="x",
    )
    boxes = np.tile(
        np.array([[0.5, 0.5, 0.25, 0.25]], dtype=np.float32),
        (1, 100, 1),
    )
    logits = np.ones((1, 100, 2), dtype=np.float32) * 10.0

    parsed_boxes, scores, classes, masks, obb, keypoints = backend._parse_outputs(
        [boxes, logits], 64, (100, 100), conf=0.5
    )

    assert parsed_boxes.shape == (100, 4)
    assert scores.shape == (100,)
    assert classes.shape == (100,)
    assert masks is None
    assert obb is None
    assert keypoints is None


def test_rfdetr_pose_backend_reshapes_batched_flattened_keypoints():
    backend = _DummyBackend(
        "rfdetr",
        task="pose",
        supported_tasks=("pose",),
        model_size="n",
    )
    backend.keypoint_dim = 3
    boxes = np.array(
        [[[0.5, 0.5, 0.2, 0.2], [0.25, 0.25, 0.1, 0.1]]],
        dtype=np.float32,
    )
    logits = np.array([[[10.0], [9.0]]], dtype=np.float32)
    keypoints = np.array(
        [
            [
                [0.1, 0.2, 2.0, 0.3, 0.4, 4.0],
                [0.5, 0.6, 0.0, 0.7, 0.8, -2.0],
            ]
        ],
        dtype=np.float32,
    )

    parsed_boxes, scores, classes, masks, obb, parsed_keypoints = (
        backend._parse_outputs([boxes, logits, keypoints], 64, (100, 200), conf=0.5)
    )

    assert parsed_boxes.shape == (2, 4)
    assert scores.shape == (2,)
    assert classes.tolist() == [0, 0]
    assert masks is None
    assert obb is None
    assert parsed_keypoints.shape == (2, 2, 3)
    np.testing.assert_allclose(parsed_keypoints[0, :, :2], [[10.0, 40.0], [30.0, 80.0]])
    np.testing.assert_allclose(
        parsed_keypoints[0, :, 2],
        [1.0 / (1.0 + np.exp(-2.0)), 1.0 / (1.0 + np.exp(-4.0))],
    )


def test_rfdetr_pose_backend_accepts_xy_only_keypoints():
    backend = _DummyBackend(
        "rfdetr",
        task="pose",
        supported_tasks=("pose",),
        model_size="n",
    )
    backend.keypoint_dim = 2
    boxes = np.array([[[0.5, 0.5, 0.2, 0.2]]], dtype=np.float32)
    logits = np.array([[[10.0]]], dtype=np.float32)
    keypoints = np.array([[[0.1, 0.2, 0.3, 0.4]]], dtype=np.float32)

    parsed_boxes, scores, classes, masks, obb, parsed_keypoints = (
        backend._parse_outputs([boxes, logits, keypoints], 64, (100, 200), conf=0.5)
    )

    assert parsed_boxes.shape == (1, 4)
    assert scores.shape == (1,)
    assert classes.tolist() == [0]
    assert masks is None
    assert obb is None
    assert parsed_keypoints.shape == (1, 2, 3)
    np.testing.assert_allclose(parsed_keypoints[0, :, :2], [[10.0, 40.0], [30.0, 80.0]])
    np.testing.assert_allclose(parsed_keypoints[0, :, 2], [1.0, 1.0])


def test_backend_sets_pose_metadata_attributes():
    assert backend_base._read_pose_metadata(
        {"num_keypoints_per_class": "[0, 17, 4]"}
    ) == {"num_keypoints_per_class": [0, 17, 4]}
    assert backend_base._read_pose_metadata(
        {"num_keypoints": "17", "keypoint_dim": "3", "num_keypoints_per_class": [0, 17]}
    ) == {
        "num_keypoints": 17,
        "keypoint_dim": 3,
        "num_keypoints_per_class": [0, 17],
    }
    backend = _DummyBackend(
        "rfdetr",
        task="pose",
        supported_tasks=("pose",),
        num_keypoints=17,
        keypoint_dim=3,
        num_keypoints_per_class=[0, 17, 4],
    )

    assert backend.num_keypoints == 17
    assert backend.keypoint_dim == 3
    assert backend.num_keypoints_per_class == [0, 17, 4]


def test_rfdetr_classic_pose_slices_background_logits_before_topk():
    backend = _DummyBackend(
        "rfdetr",
        task="pose",
        supported_tasks=("pose",),
        model_size="n",
        keypoint_dim=2,
    )
    backend.nb_classes = 1
    backend.names = {0: "person"}
    boxes = np.array(
        [[[0.5, 0.5, 0.2, 0.2], [0.25, 0.25, 0.1, 0.1]]],
        dtype=np.float32,
    )
    logits = np.array([[[-10.0, 12.0], [9.0, -12.0]]], dtype=np.float32)
    keypoints = np.array(
        [[[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]], dtype=np.float32
    )

    parsed_boxes, scores, classes, masks, obb, parsed_keypoints = (
        backend._parse_outputs([boxes, logits, keypoints], 64, (100, 200), conf=0.6)
    )

    assert parsed_boxes.shape == (1, 4)
    assert scores.shape == (1,)
    assert classes.tolist() == [0]
    assert masks is None
    assert obb is None
    np.testing.assert_allclose(
        parsed_keypoints[0, :, :2], [[50.0, 120.0], [70.0, 160.0]]
    )


def test_rfdetr_pose_backend_rejects_invalid_grouppose_schema():
    backend = _DummyBackend(
        "rfdetr",
        task="pose",
        supported_tasks=("pose",),
        model_size="n",
    )
    backend.num_keypoints_per_class = [0, 17, 4]
    boxes = np.array([[[0.5, 0.5, 0.2, 0.2]]], dtype=np.float32)
    logits = np.array([[[10.0, 9.0]]], dtype=np.float32)
    keypoints = np.zeros((1, 1, 34, 8), dtype=np.float32)

    with pytest.raises(ValueError, match="GroupPose"):
        backend._parse_outputs([boxes, logits, keypoints], 64, (100, 100), conf=0.5)


def test_yolo_backend_still_applies_nms():
    backend = _DummyBackend("yolo9")

    boxes = np.array([[0, 0, 10, 10], [0, 0, 10, 10]], dtype=np.float32)
    scores = np.array([0.9, 0.8], dtype=np.float32)
    classes = np.array([0, 0], dtype=np.int64)

    result = backend._build_result(
        boxes,
        scores,
        classes,
        orig_shape=(10, 10),
        image_path=None,
        iou=0.45,
        classes=None,
        max_det=300,
    )

    assert len(result.boxes) == 1


def test_yolo9_backend_parse_uses_letterbox_inverse():
    backend = _DummyBackend("yolo9")
    pred = np.zeros((1, 6, 1), dtype=np.float32)
    pred[0, :4, 0] = [0.0, 0.0, 320.0, 320.0]
    pred[0, 4, 0] = 0.9

    boxes, scores, classes, masks = backend._parse_outputs(
        [pred], 640, (1280, 960), conf=0.25
    )

    assert masks is None
    np.testing.assert_allclose(boxes, [[0.0, 0.0, 640.0, 640.0]])
    np.testing.assert_allclose(scores, [0.9])
    np.testing.assert_array_equal(classes, [0])


def test_yolo9_backend_parse_accepts_rectangular_imgsz():
    backend = _DummyBackend("yolo9")
    pred = np.zeros((1, 6, 1), dtype=np.float32)
    pred[0, :4, 0] = [0.0, 0.0, 320.0, 320.0]
    pred[0, 4, 0] = 0.9

    boxes, scores, classes, masks = backend._parse_outputs(
        [pred], (320, 640), (1280, 960), conf=0.25
    )

    assert masks is None
    np.testing.assert_allclose(boxes, [[0.0, 0.0, 960.0, 960.0]])
    np.testing.assert_allclose(scores, [0.9])
    np.testing.assert_array_equal(classes, [0])


def test_embedded_nms_backend_parse_drops_boxes_collapsed_by_clipping():
    backend = _DummyBackend("yolo9")
    backend.embedded_nms = True
    det = np.array(
        [
            [
                [-20.0, -20.0, -1.0, -1.0, 0.9, 1.0],
                [10.0, 20.0, 30.0, 40.0, 0.8, 0.0],
                [0.0, 0.0, 10.0, 10.0, 0.1, 0.0],
            ]
        ],
        dtype=np.float32,
    )

    boxes, scores, classes, masks = backend._parse_outputs(
        [det], 100, (100, 100), conf=0.25
    )

    assert masks is None
    np.testing.assert_allclose(boxes, [[10.0, 20.0, 30.0, 40.0]])
    np.testing.assert_allclose(scores, [0.8])
    np.testing.assert_array_equal(classes, [0])


def test_yolo9_backend_parse_drops_boxes_collapsed_by_clipping():
    backend = _DummyBackend("yolo9")
    pred = np.zeros((1, 5, 2), dtype=np.float32)
    pred[0, :4, 0] = [-20.0, -20.0, -1.0, -1.0]
    pred[0, :4, 1] = [10.0, 20.0, 30.0, 40.0]
    pred[0, 4, :] = [0.9, 0.8]

    boxes, scores, classes, masks = backend._parse_outputs(
        [pred], 100, (100, 100), conf=0.25
    )

    assert masks is None
    np.testing.assert_allclose(boxes, [[10.0, 20.0, 30.0, 40.0]])
    np.testing.assert_allclose(scores, [0.8])
    np.testing.assert_array_equal(classes, [0])


def test_embedded_nms_backend_applies_post_clip_nms():
    backend = _DummyBackend("yolo9")
    backend.embedded_nms = True
    det = np.array(
        [
            [
                [0.0, 0.0, 100.0, 100.0, 0.9, 0.0],
                [0.0, 0.0, 100.0, 40.0, 0.8, 0.0],
            ]
        ],
        dtype=np.float32,
    )

    boxes, scores, classes, masks = backend._parse_outputs(
        [det], 100, (100, 50), conf=0.25
    )
    result = backend._build_result(
        boxes,
        scores,
        classes,
        orig_shape=(50, 100),
        image_path=None,
        iou=0.45,
        classes=None,
        max_det=300,
    )

    assert masks is None
    assert len(result.boxes) == 1
    np.testing.assert_allclose(result.boxes.xyxy.numpy(), [[0.0, 0.0, 100.0, 50.0]])
    np.testing.assert_allclose(result.boxes.conf.numpy(), [0.9])


def test_backend_rejects_rectangular_imgsz_for_non_yolo9_family():
    backend = _DummyBackend("yolox")

    with pytest.raises(NotImplementedError, match="YOLO9-family"):
        backend._resolve_predict_imgsz((320, 640))


def test_backend_rectangular_imgsz_guard_normalizes_family_name():
    backend = _DummyBackend("YOLO9", imgsz=(320, 640))

    assert backend._resolve_predict_imgsz() == (320, 640)


def test_classify_backend_postprocess_returns_probs():
    backend = _DummyBackend(
        "yolo9",
        task="classify",
        supported_tasks=("detect", "classify"),
        imgsz=8,
    )
    logits = np.array([[1.0, 3.0]], dtype=np.float32)

    det = backend._postprocess(
        [logits],
        conf_thres=0.25,
        iou_thres=0.5,
        original_size=(12, 10),
        input_size=8,
    )

    assert set(det) == {"probs"}
    assert det["probs"].shape == (2,)
    assert det["probs"].argmax().item() == 1


def test_classify_backend_predict_returns_probs_and_saves_original(
    tmp_path, monkeypatch
):
    backend = _DummyBackend(
        "yolo9",
        task="classify",
        supported_tasks=("detect", "classify"),
        imgsz=8,
    )
    captured = {}

    def run_inference(blob):
        captured["shape"] = tuple(blob.shape)
        return [np.array([[1.0, 3.0]], dtype=np.float32)]

    monkeypatch.setattr(backend, "_run_inference", run_inference)
    output_path = tmp_path / "classified.jpg"

    result = backend._predict_single(
        np.zeros((10, 12, 3), dtype=np.uint8),
        save=True,
        output_path=str(output_path),
    )

    assert captured["shape"] == (1, 3, 8, 8)
    assert result.boxes is None
    assert result.probs is not None
    assert result.probs.top1 == 1
    assert len(result) == 1
    assert output_path.exists()
    assert result.saved_path == str(output_path)


def test_classify_validator_accepts_backend_single_output_list():
    from libreyolo.validation.classify_validator import ClassifyValidator

    validator = object.__new__(ClassifyValidator)
    logits = np.array([[1.0, 3.0]], dtype=np.float32)

    preds = validator._postprocess_predictions([logits], batch=None)

    np.testing.assert_allclose(preds, logits)


def test_backend_metadata_rejects_rectangular_non_yolo9_family():
    from libreyolo.backends.base import _read_metadata_imgsz

    with pytest.raises(NotImplementedError, match="YOLO9-family"):
        _read_metadata_imgsz(
            {"imgsz": "640", "imgsz_h": "320", "imgsz_w": "640"},
            "yolox",
            artifact="test metadata",
        )


@pytest.mark.parametrize(
    "metadata",
    [
        {"imgsz": "640", "imgsz_h": "320"},
        {"imgsz": "640", "imgsz_w": "640"},
    ],
)
def test_backend_metadata_rejects_partial_rectangular_imgsz(metadata):
    from libreyolo.backends.base import _read_metadata_imgsz

    with pytest.raises(ValueError, match="both imgsz_h and imgsz_w"):
        _read_metadata_imgsz(metadata, "yolo9", artifact="test metadata")


def test_yolo9_backend_predict_uses_rectangular_default_imgsz(monkeypatch):
    backend = _DummyBackend("yolo9", imgsz=(16, 32))
    captured = {}

    def run_inference(blob):
        captured["shape"] = tuple(blob.shape)
        return [np.zeros((1, 6, 0), dtype=np.float32)]

    monkeypatch.setattr(backend, "_run_inference", run_inference)

    result = backend._predict_single(np.zeros((8, 16, 3), dtype=np.uint8))

    assert captured["shape"] == (1, 3, 16, 32)
    assert result.orig_shape == (8, 16)
    assert len(result) == 0


def test_yolo9_backend_parse_detection_is_multilabel():
    backend = _DummyBackend("yolo9")
    pred = np.zeros((1, 6, 1), dtype=np.float32)
    pred[0, :4, 0] = [0.0, 0.0, 100.0, 100.0]
    pred[0, 4:, 0] = [0.9, 0.8]

    boxes, scores, classes, masks = backend._parse_outputs(
        [pred], 100, (100, 100), conf=0.25
    )

    assert masks is None
    np.testing.assert_allclose(boxes, [[0.0, 0.0, 100.0, 100.0]] * 2)
    np.testing.assert_allclose(np.sort(scores), [0.8, 0.9])
    np.testing.assert_array_equal(np.sort(classes), [0, 1])


def test_yolo9_backend_parse_caps_multilabel_candidates(monkeypatch):
    monkeypatch.setattr(backend_base, "_YOLO9_NMS_PRE", 3)
    backend = _DummyBackend("yolo9")
    pred = np.zeros((1, 6, 4), dtype=np.float32)
    pred[0, :4] = np.array(
        [
            [0.0, 20.0, 40.0, 60.0],
            [0.0, 0.0, 0.0, 0.0],
            [10.0, 30.0, 50.0, 70.0],
            [10.0, 10.0, 10.0, 10.0],
        ],
        dtype=np.float32,
    )
    pred[0, 4:] = np.array(
        [[0.1, 0.9, 0.7, 0.5], [0.8, 0.2, 0.6, 0.4]], dtype=np.float32
    )

    boxes, scores, classes, masks = backend._parse_outputs(
        [pred], 80, (80, 80), conf=0.01
    )

    assert masks is None
    assert boxes.shape[0] == 8
    np.testing.assert_allclose(
        np.sort(scores), [0.1, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], rtol=0, atol=1e-6
    )
    np.testing.assert_array_equal(classes, [0, 1, 0, 1, 0, 1, 0, 1])


def test_yolo9_obb_backend_parse_outputs_obb_payload():
    backend = _DummyBackend(
        "yolo9",
        task="obb",
        supported_tasks=("detect", "segment", "obb"),
    )
    pred = np.zeros((1, 7, 2), dtype=np.float32)
    pred[0, :4] = np.array(
        [
            [10.0, 10.0],
            [20.0, 20.0],
            [50.0, 50.0],
            [40.0, 40.0],
        ],
        dtype=np.float32,
    )
    pred[0, 4] = 0.25
    pred[0, 5:] = np.array([[0.9, 0.8], [0.1, 0.2]], dtype=np.float32)

    boxes, scores, classes, masks, obb = backend._parse_outputs(
        [pred], 64, (64, 64), conf=0.25, iou=0.5, max_det=1
    )

    assert masks is None
    assert boxes.shape == (1, 4)
    np.testing.assert_allclose(scores, [0.9])
    np.testing.assert_array_equal(classes, [0])
    assert obb.shape == (1, 7)
    np.testing.assert_allclose(obb[0, :5], [30.0, 30.0, 40.0, 20.0, 0.25])


def test_yolo9_obb_backend_parse_uses_letterbox_inverse_for_non_square_images():
    backend = _DummyBackend(
        "yolo9",
        task="obb",
        supported_tasks=("detect", "segment", "obb"),
    )
    pred = np.zeros((1, 7, 1), dtype=np.float32)
    pred[0, :4, 0] = [100.0, 50.0, 200.0, 150.0]
    pred[0, 4, 0] = 0.25
    pred[0, 5:, 0] = [0.9, 0.1]

    boxes, scores, classes, masks, obb = backend._parse_outputs(
        [pred],
        640,
        (1280, 960),
        conf=0.25,
        iou=0.5,
        max_det=300,
    )

    angle = 0.25
    envelope = 100.0 * 2.0 * (np.cos(angle) + np.sin(angle))
    half_envelope = envelope / 2.0

    assert masks is None
    np.testing.assert_allclose(
        boxes,
        [
            [
                300.0 - half_envelope,
                200.0 - half_envelope,
                300.0 + half_envelope,
                200.0 + half_envelope,
            ]
        ],
        rtol=1e-6,
        atol=1e-5,
    )
    np.testing.assert_allclose(scores, [0.9])
    np.testing.assert_array_equal(classes, [0])
    assert obb.shape == (1, 7)
    np.testing.assert_allclose(obb[0, :5], [300.0, 200.0, 200.0, 200.0, 0.25])


def test_yolo9_obb_backend_postprocess_returns_obb_tensor():
    backend = _DummyBackend(
        "yolo9",
        task="obb",
        supported_tasks=("detect", "segment", "obb"),
    )
    pred = np.zeros((1, 7, 1), dtype=np.float32)
    pred[0, :4, 0] = [10.0, 20.0, 50.0, 40.0]
    pred[0, 4, 0] = 0.25
    pred[0, 5:, 0] = [0.9, 0.1]

    out = backend._postprocess(
        [pred],
        conf_thres=0.25,
        iou_thres=0.5,
        original_size=(64, 64),
        input_size=64,
    )

    assert out["num_detections"] == 1
    assert "obb" in out
    np.testing.assert_allclose(
        out["obb"][0, :5].numpy(), [30.0, 30.0, 40.0, 20.0, 0.25]
    )


def test_obb_backend_class_filter_preserves_obb_alignment():
    backend = _DummyBackend(
        "yolo9",
        task="obb",
        supported_tasks=("detect", "segment", "obb"),
    )
    boxes = np.array(
        [[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 40.0, 40.0]],
        dtype=np.float32,
    )
    scores = np.array([0.9, 0.8], dtype=np.float32)
    classes = np.array([0, 1], dtype=np.int64)
    obb = np.array(
        [
            [5.0, 5.0, 10.0, 10.0, 0.1, 0.9, 0.0],
            [30.0, 30.0, 20.0, 20.0, 0.2, 0.8, 1.0],
        ],
        dtype=np.float32,
    )

    result = backend._build_result(
        boxes,
        scores,
        classes,
        obb=obb,
        orig_shape=(80, 100),
        image_path=None,
        iou=0.5,
        classes=[1],
        max_det=300,
    )

    assert len(result.boxes) == 1
    assert result.obb is not None
    assert result.boxes.cls.tolist() == [1.0]
    assert result.obb.cls.tolist() == [1.0]
    np.testing.assert_allclose(
        result.obb.xywhr.numpy(), [[30.0, 30.0, 20.0, 20.0, 0.2]]
    )


def test_backend_save_annotated_accepts_directory_output_path(tmp_path):
    backend = _DummyBackend("yolo9")
    image_path = tmp_path / "source.jpg"
    output_dir = tmp_path / "predictions"
    result = backend._build_result(
        np.empty((0, 4), dtype=np.float32),
        np.empty((0,), dtype=np.float32),
        np.empty((0,), dtype=np.int64),
        orig_shape=(8, 8),
        image_path=image_path,
        iou=0.5,
        classes=None,
        max_det=300,
    )

    backend._save_annotated(
        result,
        Image.new("RGB", (8, 8)),
        image_path,
        str(output_dir),
    )

    expected = output_dir / "source.jpg"
    assert expected.exists()
    assert result.saved_path == str(expected)


@pytest.mark.parametrize("task", ["semantic", "matte", "point"])
def test_backend_save_annotated_handles_boxless_dense_results(tmp_path, task):
    from libreyolo.utils.results import Matte, Points, Results, SemanticMask

    backend = _DummyBackend(
        "fomo" if task == "point" else "test",
        task=task,
        supported_tasks=(task,),
        imgsz=8,
    )
    payload = {}
    if task == "semantic":
        payload["semantic_mask"] = SemanticMask(torch.zeros(8, 8), (8, 8))
    elif task == "matte":
        payload["matte"] = Matte(torch.ones(8, 8), (8, 8))
    else:
        payload["points"] = Points(
            torch.tensor([[4.0, 4.0, 0.0, 0.9]]),
            (8, 8),
        )
    result = Results(
        boxes=None,
        orig_shape=(8, 8),
        names={0: "object"},
        **payload,
    )
    output = tmp_path / f"{task}.jpg"

    backend._save_annotated(
        result,
        Image.new("RGB", (8, 8), "white"),
        output.name,
        str(output),
    )

    expected = output.with_suffix(".png") if task == "matte" else output
    assert expected.exists()
    assert result.saved_path == str(expected)


def test_tensorrt_backend_detects_obb_task_from_filename():
    from libreyolo.backends.tensorrt import TensorRTBackend

    backend = object.__new__(TensorRTBackend)
    backend.model_path = "weights/LibreYOLO9t-obb.engine"

    assert backend._detect_task_from_filename() == "obb"


def test_yolo9_segment_backend_is_rejected():
    with pytest.raises(NotImplementedError, match="YOLO9 segmentation"):
        _DummyBackend("yolo9", task="segment", supported_tasks=("detect", "segment"))


def test_backend_call_accepts_device_kwarg(monkeypatch):
    backend = _DummyBackend("yolo9")
    monkeypatch.setattr(backend, "_predict_single", lambda source, **kwargs: "ok")

    assert backend("image.jpg", device="cpu") == "ok"


def test_backend_rejects_unsupported_explicit_task():
    with pytest.raises(ValueError, match="not supported"):
        _DummyBackend("yolo9", task="segment", supported_tasks=("detect",))


def test_backend_accepts_fomo_point_task_with_shared_parser():
    backend = _DummyBackend("fomo", task="point", supported_tasks=("point",))
    assert backend.task == "point"
