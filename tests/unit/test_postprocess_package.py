"""Centralized-postprocess package: move integrity + extraction parity.

Postprocessing was moved verbatim from ``models/*/utils.py`` (and
``models/ec/postprocess.py`` / ``utils/general.py``) to
``libreyolo/postprocess/``; the old import paths re-export the same objects.
These tests pin:
  (a) old-path / new-path identity (proves the move is a pure re-export),
  (b) the DEIM -> D-FINE dedup (historical copy was code-identical),
  (c) numeric parity for the one real extraction (RT-DETR's inline method),
  (d) the lazy ``box_cxcywh_to_xyxy`` imports actually resolve at call time.
"""

import importlib

import pytest
import torch

pytestmark = pytest.mark.unit


def test_yolo9_reexports_are_same_objects():
    old = importlib.import_module("libreyolo.models.yolo9.utils")
    new = importlib.import_module("libreyolo.postprocess.yolo9")

    assert old.postprocess is new.postprocess
    assert old._nms_keep_indices is new._nms_keep_indices
    assert old._rotated_nms_keep_indices is new._rotated_nms_keep_indices
    assert old._obb_prefilter_keep_indices is new._obb_prefilter_keep_indices
    assert old._process_masks is new._process_masks
    assert old._input_size_hw is new._input_size_hw
    assert old._YOLO9_MAX_NMS_CANDIDATES == new._YOLO9_MAX_NMS_CANDIDATES == 30000
    assert old._YOLO9_OBB_MAX_NMS_CANDIDATES == new._YOLO9_OBB_MAX_NMS_CANDIDATES


def test_shared_tail_reexport_is_same_object():
    old = importlib.import_module("libreyolo.utils.general")
    new = importlib.import_module("libreyolo.postprocess.common")

    assert old.postprocess_detections is new.postprocess_detections


def test_family_reexports_are_same_objects():
    pairs = [
        ("libreyolo.models.yolox.utils", "libreyolo.postprocess.yolox",
         ["postprocess", "decode_outputs", "make_grids"]),
        ("libreyolo.models.rtmdet.utils", "libreyolo.postprocess.rtmdet",
         ["postprocess", "_distance2bbox", "_make_grid_priors"]),
        ("libreyolo.models.picodet.utils", "libreyolo.postprocess.picodet",
         ["postprocess", "_per_level_filter_topk", "_grid_centers"]),
        ("libreyolo.models.yolonas.utils", "libreyolo.postprocess.yolonas",
         ["postprocess", "postprocess_pose", "_undo_letterbox_xyxy",
          "_undo_letterbox_xy", "_extract_decoded_predictions"]),
        ("libreyolo.models.damoyolo.utils", "libreyolo.postprocess.damoyolo",
         ["postprocess_predictions", "multiclass_nms"]),
        ("libreyolo.models.yolo9_e2e.utils", "libreyolo.postprocess.yolo9_e2e",
         ["postprocess", "_scale_and_clip_boxes"]),
        ("libreyolo.models.ec.postprocess", "libreyolo.postprocess.ec",
         ["postprocess", "postprocess_seg", "postprocess_pose"]),
        ("libreyolo.models.dfine.utils", "libreyolo.postprocess.dfine",
         ["postprocess"]),
    ]
    for old_path, new_path, names in pairs:
        old = importlib.import_module(old_path)
        new = importlib.import_module(new_path)
        for name in names:
            assert getattr(old, name) is getattr(new, name), (
                f"{old_path}.{name} is not the object from {new_path}"
            )


def test_yolonas_constants_preserved():
    new = importlib.import_module("libreyolo.postprocess.yolonas")
    old = importlib.import_module("libreyolo.models.yolonas.utils")

    assert old.YOLO_NAS_RESIZE_SIZE == new.YOLO_NAS_RESIZE_SIZE == 636
    assert old.YOLO_NAS_POSE_RESIZE_SIZE == new.YOLO_NAS_POSE_RESIZE_SIZE == 640
    assert old.YOLO_NAS_PRE_NMS_TOP_K == new.YOLO_NAS_PRE_NMS_TOP_K == 1000
    assert old.YOLO_NAS_POSE_PAD_VALUE == 127  # preprocess-only; stays in utils


def test_deim_consumes_dfine_postprocess():
    dfine_new = importlib.import_module("libreyolo.postprocess.dfine").postprocess
    deim_new = importlib.import_module("libreyolo.postprocess.deim").postprocess
    deim_old = importlib.import_module("libreyolo.models.deim.utils").postprocess
    deimv2_old = importlib.import_module("libreyolo.models.deimv2.utils").postprocess

    assert deim_new is dfine_new
    assert deim_old is dfine_new
    assert deimv2_old is dfine_new


def test_rfdetr_reexport_is_same_object():
    # models/rfdetr/utils.py needs no optional deps (transformers is only
    # required by model.py via the lazy registry), so a plain import is safe.
    new = importlib.import_module("libreyolo.postprocess.rfdetr")
    old = importlib.import_module("libreyolo.models.rfdetr.utils")

    assert old.postprocess is new.postprocess


def test_rtdetr_extracted_postprocess_matches_reference():
    postprocess = importlib.import_module("libreyolo.postprocess.rtdetr").postprocess

    g = torch.Generator().manual_seed(0)
    logits = torch.randn(1, 30, 5, generator=g)
    boxes_cxcywh = torch.rand(1, 30, 4, generator=g) * 0.5 + 0.25
    output = {"pred_logits": logits, "pred_boxes": boxes_cxcywh}

    result = postprocess(
        output, conf_thres=0.3, iou_thres=0.45, original_size=(320, 240), max_det=50
    )

    # Reference decode — the exact math that lived inline in
    # LibreRTDETR._postprocess before the extraction.
    scores_per_class = torch.sigmoid(logits[0])
    num_classes = scores_per_class.shape[-1]
    flat = scores_per_class.flatten()
    k = min(50, flat.numel())
    topk_scores, topk_indices = torch.topk(flat, k)
    query_idx = topk_indices // num_classes
    class_idx = topk_indices % num_classes
    ref = boxes_cxcywh[0][query_idx]
    cx, cy, w, h = ref.unbind(-1)
    ref_boxes = torch.stack(
        [(cx - w / 2) * 320, (cy - h / 2) * 240, (cx + w / 2) * 320, (cy + h / 2) * 240],
        dim=-1,
    )
    mask = topk_scores > 0.3

    assert torch.equal(result["boxes"], ref_boxes[mask])
    assert torch.equal(result["scores"], topk_scores[mask])
    assert torch.equal(result["classes"], class_idx[mask])
    assert result["num_detections"] == int(mask.sum())


def test_rtdetr_model_method_delegates_to_package():
    model_mod = importlib.import_module("libreyolo.models.rtdetr.model")
    package_fn = importlib.import_module("libreyolo.postprocess.rtdetr").postprocess

    assert model_mod.rtdetr_postprocess is package_fn


def test_dfine_postprocess_smoke_lazy_import():
    # Exercises the function-body import of models.dfine.box_ops (kept lazy
    # to avoid a circular import through the eager models registry).
    postprocess = importlib.import_module("libreyolo.postprocess.dfine").postprocess

    g = torch.Generator().manual_seed(1)
    output = {
        "pred_logits": torch.randn(1, 20, 4, generator=g),
        "pred_boxes": torch.rand(1, 20, 4, generator=g) * 0.4 + 0.3,
    }
    result = postprocess(output, conf_thres=0.05, original_size=(640, 480), max_det=10)

    assert set(result) >= {"boxes", "scores", "classes", "num_detections"}
    assert result["num_detections"] == len(result["boxes"])
    assert len(result["boxes"]) <= 10


def test_ec_postprocess_smoke_lazy_import():
    ec = importlib.import_module("libreyolo.postprocess.ec")

    g = torch.Generator().manual_seed(2)
    det = ec.postprocess(
        {
            "pred_logits": torch.randn(1, 15, 3, generator=g),
            "pred_boxes": torch.rand(1, 15, 4, generator=g) * 0.4 + 0.3,
        },
        conf_thres=0.05,
        original_size=(320, 320),
        max_det=10,
    )
    assert det["num_detections"] == len(det["boxes"])

    pose = ec.postprocess_pose(
        {
            "pred_logits": torch.randn(1, 15, 1, generator=g),
            "pred_keypoints": torch.rand(1, 15, 34, generator=g),
        },
        conf_thres=0.05,
        original_size=(320, 320),
        max_det=5,
    )
    assert pose["num_detections"] == len(pose["boxes"])
    assert pose["keypoints"].shape[1:] == (17, 3)
