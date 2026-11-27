"""Hermetic CPU unit tests for LibreLoopDVR (LoopDet, LoopSeg, LoopPose)."""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

import libreyolo
from libreyolo import LibreLoopDVR, LibreYOLO
from libreyolo.models.loopdvr.nn import (
    LibreLoopDVRModel,
    LibreLoopDVRPoseModel,
    LibreLoopDVRSegModel,
)
from libreyolo.postprocess.loopdvr import (
    postprocess,
    postprocess_pose,
    postprocess_seg,
)
from libreyolo.utils.serialization import (
    validate_checkpoint_metadata,
    wrap_libreyolo_checkpoint,
)

pytestmark = [pytest.mark.unit, pytest.mark.loopdvr]


def _loopdvr_minimal_state(
    size: str = "s",
    task: str = "detect",
    nc: int = 80,
) -> dict[str, torch.Tensor]:
    dims = {
        "s": (192, 192, 768, 704),
        "m": (256, 256, 1024, 704),
        "l": (384, 256, 1536, 704),
        "x": (384, 256, 2304, 896),
    }
    embed_dim, proj_dim, mlp_hidden, cv4_in = dims[size]
    state: dict[str, torch.Tensor] = {
        "looped_vit.register_token": torch.zeros(1, 4, embed_dim),
        "looped_vit.blocks.0.loop_adapters.0.coefficient1.bias": torch.zeros(embed_dim),
        "looped_vit.blocks.0.mlp.fc1.weight": torch.zeros(mlp_hidden, embed_dim),
        "looped_vit.projector.0.conv.weight": torch.zeros(proj_dim, embed_dim, 1, 1),
        "encoder.fpn_blocks.0.cv4.conv.weight": torch.zeros(proj_dim, cv4_in, 1, 1),
        "decoder.dec_score_head.0.bias": torch.zeros(2 if task == "pose" else nc),
    }
    if task == "pose":
        state["decoder.keypoint_embedding.weight"] = torch.zeros(17, proj_dim)
    elif task == "segment":
        state["decoder.decoder.segmentation_head.bias"] = torch.zeros(proj_dim)
    return state


def test_top_level_export_and_registry():
    assert LibreLoopDVR is libreyolo.LibreLoopDVR
    assert LibreLoopDVR.FAMILY == "loopdvr"
    assert LibreLoopDVR.FILENAME_PREFIX == "LibreLoopDVR"
    assert LibreLoopDVR.SUPPORTED_TASKS == ("detect", "pose", "segment")
    assert LibreLoopDVR.DEFAULT_TASK == "detect"
    assert LibreLoopDVR.TRAIN_CONFIG is None


@pytest.mark.parametrize(
    ("filename", "expected_size", "expected_task"),
    [
        ("LibreLoopDVRs.pt", "s", None),
        ("LibreLoopDVRm.pt", "m", None),
        ("LibreLoopDVRl.pt", "l", None),
        ("LibreLoopDVRx.pt", "x", None),
        ("LibreLoopDVRs-seg.pt", "s", "segment"),
        ("LibreLoopDVRm-pose.pt", "m", "pose"),
        ("loopdet-s.pth", "s", "detect"),
        ("loopseg-l.pth", "l", "segment"),
        ("looppose-x.pth", "x", "pose"),
        ("LibreECs.pt", None, None),
    ],
)
def test_filename_detection(filename, expected_size, expected_task):
    assert LibreLoopDVR.detect_size_from_filename(filename) == expected_size
    assert LibreLoopDVR.detect_task_from_filename(filename) == expected_task


@pytest.mark.parametrize("size", ["s", "m", "l", "x"])
@pytest.mark.parametrize("task", ["detect", "segment", "pose"])
def test_state_dict_detection(size, task):
    state = _loopdvr_minimal_state(size=size, task=task, nc=12)
    assert LibreLoopDVR.can_load(state) is True
    assert LibreLoopDVR.detect_size(state) == size
    expected_ckpt_task = None if task == "detect" else task
    assert LibreLoopDVR.detect_checkpoint_task(state) == expected_ckpt_task
    assert LibreLoopDVR.detect_nb_classes(state) == (1 if task == "pose" else 12)


@pytest.mark.parametrize(
    ("sibling_name", "sibling_state"),
    [
        ("LibreEC", {"backbone.backbone.register_token": torch.zeros(1, 4, 192)}),
        ("LibreDFINE", {"decoder.pre_bbox_head.layers.0.weight": torch.zeros(4, 4)}),
        ("LibreDEIM", {"decoder.pre_bbox_head.layers.0.weight": torch.zeros(4, 4)}),
        (
            "LibreRTDETR",
            {
                "backbone.res_layers.0.blocks.0.conv1.weight": torch.zeros(4, 4, 3, 3),
                "encoder.input_proj.0.0.weight": torch.zeros(4, 4, 1, 1),
                "decoder.input_proj.0.conv.weight": torch.zeros(4, 4, 1, 1),
                "decoder.dec_score_head.0.weight": torch.zeros(80, 4),
            },
        ),
        (
            "LibreDETR",
            {
                "query_embed.weight": torch.zeros(100, 256),
                "transformer.decoder.layers.0.multihead_attn.in_proj_weight": torch.zeros(
                    768, 256
                ),
                "backbone.0.body.conv1.weight": torch.zeros(64, 3, 7, 7),
                "class_embed.weight": torch.zeros(92, 256),
            },
        ),
        ("LibreYOLOX", {"backbone.backbone.stem.conv.weight": torch.zeros(4, 3, 3, 3)}),
    ],
)
def test_can_load_rejects_siblings(sibling_name, sibling_state):
    sibling = getattr(libreyolo, sibling_name)
    assert sibling.can_load(sibling_state) is True
    assert LibreLoopDVR.can_load(sibling_state) is False
    loopdvr_state = _loopdvr_minimal_state("s", "detect")
    assert LibreLoopDVR.can_load(loopdvr_state) is True
    assert sibling.can_load(loopdvr_state) is False


def test_train_raises_not_implemented():
    model = LibreLoopDVR(None, size="s", nb_classes=5, device="cpu", task="detect")
    with pytest.raises(NotImplementedError, match="inference-only"):
        model.train(data="coco8.yaml", epochs=1)


def test_forward_shapes_all_tasks():
    x = torch.zeros(1, 3, 640, 640)

    det = LibreLoopDVRModel(config="s", nb_classes=5).eval()
    with torch.no_grad():
        det_out = det(x)
    assert det_out["pred_logits"].shape == (1, 300, 5)
    assert det_out["pred_boxes"].shape == (1, 300, 4)

    seg = LibreLoopDVRSegModel(config="s", nb_classes=5).eval()
    with torch.no_grad():
        seg_out = seg(x)
    assert seg_out["pred_logits"].shape == (1, 300, 5)
    assert seg_out["pred_boxes"].shape == (1, 300, 4)
    assert seg_out["pred_masks"].shape == (1, 300, 160, 160)

    pose = LibreLoopDVRPoseModel(config="s").eval()
    with torch.no_grad():
        pose_out = pose(x)
    assert pose_out["pred_logits"].shape == (1, 60, 2)
    assert pose_out["pred_keypoints"].shape == (1, 60, 34)


def test_pose_m_uses_512_decoder_ffn():
    pose_m = LibreLoopDVRPoseModel(config="m")
    assert pose_m.decoder.decoder.layers[0].linear1.out_features == 512
    det_m = LibreLoopDVRModel(config="m", nb_classes=80)
    assert det_m.decoder.decoder.layers[0].linear1.out_features == 1024


def test_postprocess_contracts():
    det_logits = torch.full((1, 5, 3), -10.0)
    det_logits[0, 2, 1] = 5.0
    det_boxes = torch.tensor([[[0.5, 0.5, 0.2, 0.4]] * 5])
    det_res = postprocess(
        {"pred_logits": det_logits, "pred_boxes": det_boxes},
        conf_thres=0.5,
        original_size=(100, 200),
    )
    assert det_res["num_detections"] == 1
    assert int(det_res["classes"][0]) == 1
    assert np.allclose(det_res["boxes"][0], [40.0, 60.0, 60.0, 140.0], atol=1e-4)

    seg_masks = torch.full((1, 5, 16, 16), -5.0)
    seg_masks[0, 2, 4:12, 4:12] = 5.0
    seg_res = postprocess_seg(
        {"pred_logits": det_logits, "pred_boxes": det_boxes, "pred_masks": seg_masks},
        conf_thres=0.5,
        original_size=(100, 200),
    )
    assert seg_res["num_detections"] == 1
    assert seg_res["masks"].shape == (1, 200, 100)
    assert seg_res["masks"].dtype == torch.bool

    pose_logits = torch.full((1, 4, 2), -10.0)
    pose_logits[0, 1, 1] = 4.0  # person is column 1
    pose_kpts = torch.full((1, 4, 34), 0.5)
    pose_res = postprocess_pose(
        {"pred_logits": pose_logits, "pred_keypoints": pose_kpts},
        conf_thres=0.5,
        original_size=(100, 200),
    )
    assert pose_res["num_detections"] == 1
    assert pose_res["keypoints"].shape == (1, 17, 3)
    assert np.allclose(pose_res["keypoints"][0, :, 0], 50.0)
    assert np.allclose(pose_res["keypoints"][0, :, 1], 100.0)
    assert np.allclose(pose_res["keypoints"][0, :, 2], 1.0)


@pytest.mark.parametrize("task", ["detect", "segment", "pose"])
def test_checkpoint_roundtrip_and_autoconvert(tmp_path, task):
    nb_classes = 1 if task == "pose" else 5
    src_model = LibreLoopDVR(
        None, size="s", nb_classes=nb_classes, device="cpu", task=task
    )
    sd = src_model.model.state_dict()

    # 1. Native v1.0 checkpoint round-trip
    extra = {"num_keypoints": 17, "keypoint_dim": 3} if task == "pose" else {}
    names = {0: "person"} if task == "pose" else {i: f"cls_{i}" for i in range(nb_classes)}
    wrapped = wrap_libreyolo_checkpoint(
        sd,
        model_family="loopdvr",
        size="s",
        task=task,
        nc=nb_classes,
        names=names,
        **extra,
    )
    validate_checkpoint_metadata(wrapped, strict=True)
    suffix = {"detect": "", "segment": "-seg", "pose": "-pose"}[task]
    ckpt_path = tmp_path / f"LibreLoopDVRs{suffix}.pt"
    torch.save(wrapped, ckpt_path)

    loaded = LibreYOLO(str(ckpt_path), device="cpu")
    assert isinstance(loaded, LibreLoopDVR)
    assert loaded.FAMILY == "loopdvr"
    assert loaded.size == "s"
    assert loaded.task == task
    assert loaded.nb_classes == nb_classes

    # 2. Upstream raw {"model_id", "state_dict"} runtime auto-conversion
    prefix = {"detect": "loopdet", "segment": "loopseg", "pose": "looppose"}[task]
    raw_path = tmp_path / f"{prefix}-s.pth"
    torch.save({"model_id": f"{prefix}-s", "state_dict": sd}, raw_path)

    auto_loaded = LibreYOLO(str(raw_path), device="cpu")
    assert isinstance(auto_loaded, LibreLoopDVR)
    assert auto_loaded.FAMILY == "loopdvr"
    assert auto_loaded.size == "s"
    assert auto_loaded.task == task
    assert auto_loaded.nb_classes == nb_classes


@pytest.mark.parametrize("task", ["detect", "segment", "pose"])
def test_export_onnx_and_torchscript_parity(tmp_path, task):
    pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")

    from libreyolo.models.loopdvr.nn import LoopDVRExportWrapper

    torch.manual_seed(0)
    nb_classes = 1 if task == "pose" else 4
    model = LibreLoopDVR(None, size="s", nb_classes=nb_classes, device="cpu", task=task)
    model.model.eval()

    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, size=(320, 480, 3), dtype=np.uint8)
    tensor_in, _, _, _ = model._preprocess(img)

    with torch.no_grad():
        ref_model = copy.deepcopy(model.model).eval().deploy()
        ref_tensors = LoopDVRExportWrapper(ref_model, task=task)(tensor_in)

    # ONNX export + graph metadata + output parity + backend round-trip
    onnx_path = model.export(
        format="onnx",
        output_path=str(tmp_path / f"loopdvr_s_{task}.onnx"),
        simplify=False,
    )
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    ort_outs = sess.run(None, {sess.get_inputs()[0].name: tensor_in.numpy()})
    assert len(ort_outs) == len(ref_tensors)
    for ref_t, ort_a in zip(ref_tensors, ort_outs):
        assert ort_a.shape == tuple(ref_t.shape)
        # Untrained random weights have near-identical encoder top-k scores, so
        # a single query slot may swap between PyTorch and ORT MLAS kernels;
        # check the 95th percentile of absolute error across all elements.
        assert float(np.percentile(np.abs(ref_t.numpy() - ort_a), 95)) < 1e-3

    onnx_backend = LibreYOLO(onnx_path, device="cpu")
    assert onnx_backend.model_family == "loopdvr"
    assert onnx_backend.task == task

    r_pt = model.predict(img, conf=0.0, max_det=10)
    r_onnx = onnx_backend.predict(img, conf=0.0, max_det=10)
    assert len(r_pt.boxes) == len(r_onnx.boxes) == 10
    assert np.allclose(
        r_pt.boxes.conf.numpy(), r_onnx.boxes.conf.numpy(), atol=1e-3
    )
    if task == "segment":
        assert r_onnx.masks is not None
        assert r_onnx.masks.data.shape == r_pt.masks.data.shape
    if task == "pose":
        assert r_onnx.keypoints is not None
        assert r_onnx.keypoints.data.shape == r_pt.keypoints.data.shape

    # TorchScript export + raw tensor parity + backend round-trip
    ts_path = model.export(
        format="torchscript",
        output_path=str(tmp_path / f"loopdvr_s_{task}.torchscript"),
    )
    ts_mod = torch.jit.load(ts_path, map_location="cpu")
    with torch.no_grad():
        ts_outs = ts_mod(tensor_in)
    assert len(ts_outs) == len(ref_tensors)
    for ref_t, ts_t in zip(ref_tensors, ts_outs):
        assert torch.allclose(ref_t, ts_t, atol=1e-4, rtol=1e-4)

    ts_backend = LibreYOLO(ts_path, device="cpu")
    assert ts_backend.model_family == "loopdvr"
    assert ts_backend.task == task

    r_ts = ts_backend.predict(img, conf=0.0, max_det=10)
    assert len(r_ts.boxes) == 10
    assert np.allclose(r_pt.boxes.conf.numpy(), r_ts.boxes.conf.numpy(), atol=1e-3)
