"""Hermetic CPU unit tests for LibreLoopDVR (LoopDet, LoopSeg, LoopPose)."""

from __future__ import annotations

import copy
from pathlib import Path

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
    from libreyolo.models.loopdvr.trainer import LoopDVRConfig

    assert LibreLoopDVR is libreyolo.LibreLoopDVR
    assert LibreLoopDVR.FAMILY == "loopdvr"
    assert LibreLoopDVR.FILENAME_PREFIX == "LibreLoopDVR"
    assert LibreLoopDVR.SUPPORTED_TASKS == ("detect", "pose", "segment")
    assert LibreLoopDVR.DEFAULT_TASK == "detect"
    assert LibreLoopDVR.TRAIN_CONFIG is LoopDVRConfig
    assert LibreLoopDVR.RESUME_RESTORES_TRAIN_ARGS is True


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


def test_optimizer_param_groups_for_looped_vit():
    from libreyolo.models.loopdvr.trainer import (
        LoopDVRConfig,
        LoopDVRPoseConfig,
        LoopDVRSegConfig,
        _build_loopdvr_optimizer,
    )

    model = LibreLoopDVRModel(config="s", nb_classes=4)
    opt = _build_loopdvr_optimizer(model, lr=5e-4, wd=1e-4, bb_mult=0.05)
    assert len(opt.param_groups) == 4

    param_to_group = {}
    for g in opt.param_groups:
        for p in g["params"]:
            param_to_group[id(p)] = g

    named = dict(model.named_parameters())
    assert param_to_group[id(named["looped_vit.blocks.0.attn.qkv.weight"])]["lr_mult"] == 0.05
    assert param_to_group[id(named["looped_vit.blocks.0.attn.qkv.weight"])]["weight_decay"] == 1e-4
    assert param_to_group[id(named["looped_vit.register_token"])]["lr_mult"] == 0.05
    assert param_to_group[id(named["looped_vit.register_token"])]["weight_decay"] == 0.0
    assert param_to_group[id(named["decoder.dec_score_head.0.weight"])]["lr_mult"] == 1.0
    assert param_to_group[id(named["decoder.dec_score_head.0.bias"])]["weight_decay"] == 0.0

    # Size-specific L/X backbone LR multipliers.
    assert LoopDVRConfig.from_kwargs(size="l").backbone_lr_mult == 0.01
    assert LoopDVRSegConfig.from_kwargs(size="x").backbone_lr_mult == 0.005
    assert LoopDVRPoseConfig.from_kwargs(size="l").backbone_lr_mult == 0.005


@pytest.mark.parametrize("task", ["detect", "segment", "pose"])
def test_training_loss_and_backward_all_tasks(task):
    from libreyolo.models.dfine.matcher import HungarianMatcher
    from libreyolo.models.ec.loss import ECCriterion
    from libreyolo.models.ec.pose_loss import (
        ECPoseCriterion,
        PoseHungarianMatcher,
        default_oks_sigmas,
    )
    from libreyolo.models.ec.seg_loss import ECSegCriterion, ECSegHungarianMatcher

    torch.manual_seed(0)
    x = torch.randn(2, 3, 320, 320)

    if task == "detect":
        model = LibreLoopDVRModel(config="s", nb_classes=4).train()
        targets = [
            {
                "labels": torch.tensor([0, 2], dtype=torch.int64),
                "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.3], [0.3, 0.4, 0.15, 0.2]]),
            },
            {
                "labels": torch.tensor([1], dtype=torch.int64),
                "boxes": torch.tensor([[0.6, 0.6, 0.25, 0.25]]),
            },
        ]
        criterion = ECCriterion(
            matcher=HungarianMatcher(
                weight_dict={"cost_class": 2.0, "cost_bbox": 5.0, "cost_giou": 2.0},
                use_focal_loss=True,
                alpha=0.25,
                gamma=2.0,
            ),
            weight_dict={
                "loss_mal": 1.0,
                "loss_bbox": 5.0,
                "loss_giou": 2.0,
                "loss_fgl": 0.15,
                "loss_ddf": 1.5,
            },
            losses=["mal", "boxes", "local"],
            num_classes=4,
            distributed_normalize=False,
        )
        out = model(x, targets=targets)
        assert "dn_outputs" in out and "aux_outputs" in out
    elif task == "segment":
        model = LibreLoopDVRSegModel(config="s", nb_classes=4).train()
        m1 = torch.zeros(2, 80, 80, dtype=torch.bool)
        m1[0, 20:50, 20:50] = True
        m1[1, 10:30, 10:30] = True
        m2 = torch.zeros(1, 80, 80, dtype=torch.bool)
        m2[0, 30:60, 30:60] = True
        targets = [
            {
                "labels": torch.tensor([0, 2], dtype=torch.int64),
                "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.3], [0.3, 0.4, 0.15, 0.2]]),
                "masks": m1,
            },
            {
                "labels": torch.tensor([1], dtype=torch.int64),
                "boxes": torch.tensor([[0.6, 0.6, 0.25, 0.25]]),
                "masks": m2,
            },
        ]
        criterion = ECSegCriterion(
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
                mask_point_sample_ratio=8,
            ),
            weight_dict={
                "loss_mal": 2.0,
                "loss_bbox": 1.0,
                "loss_giou": 1.0,
                "loss_fgl": 0.15,
                "loss_ddf": 1.5,
                "loss_mask_ce": 5.0,
                "loss_mask_dice": 5.0,
            },
            losses=["mal", "boxes", "local", "masks"],
            num_classes=4,
            mask_point_sample_ratio=8,
        )
        out = model(x, targets=targets)
        assert isinstance(out["pred_masks"], dict)
        assert "dn_outputs" in out
    else:
        model = LibreLoopDVRPoseModel(config="s").train()
        kpts = torch.cat([torch.full((1, 34), 0.5), torch.ones(1, 17)], dim=1)
        targets = [
            {
                "labels": torch.tensor([1], dtype=torch.int64),
                "boxes": torch.tensor([[0.25, 0.25, 0.75, 0.75]]),
                "keypoints": kpts,
                "area": torch.tensor([0.25]),
            },
            {
                "labels": torch.tensor([1], dtype=torch.int64),
                "boxes": torch.tensor([[0.3, 0.3, 0.7, 0.7]]),
                "keypoints": kpts,
                "area": torch.tensor([0.16]),
            },
        ]
        sigmas = default_oks_sigmas(17)
        criterion = ECPoseCriterion(
            matcher=PoseHungarianMatcher(num_keypoints=17, sigmas=sigmas),
            num_keypoints=17,
            num_classes=2,
            weight_dict={"loss_vfl": 2.0, "loss_keypoints": 10.0, "loss_oks": 4.0},
            losses=("vfl", "keypoints"),
            sigmas=sigmas,
        )
        out = model(x, targets=targets)
        assert "dn_aux_outputs" in out and "aux_interm_outputs" in out

    losses = criterion(out, targets)
    total = sum(losses.values())
    assert torch.isfinite(total) and total.item() > 0.0
    total.backward()

    # Gradients must flow through the looped ViT backbone, adapters, and decoder.
    assert model.looped_vit.blocks[0].attn.qkv.weight.grad is not None
    assert torch.isfinite(model.looped_vit.blocks[0].attn.qkv.weight.grad).all()
    assert model.looped_vit.blocks[0].loop_adapters[0].mlp.fc1.weight.grad is not None
    assert model.decoder.decoder.loop_adapters[0].mlp.fc1.weight.grad is not None


def test_validation_loss_adapter_detect():
    from libreyolo.models.dfine.matcher import HungarianMatcher
    from libreyolo.models.ec.loss import ECCriterion
    from libreyolo.models.loopdvr.trainer import LoopDVRValidationLoss

    model = LibreLoopDVRModel(config="s", nb_classes=4).eval()
    criterion = ECCriterion(
        matcher=HungarianMatcher(
            weight_dict={"cost_class": 2.0, "cost_bbox": 5.0, "cost_giou": 2.0},
            use_focal_loss=True,
            alpha=0.25,
            gamma=2.0,
        ),
        weight_dict={
            "loss_mal": 1.0,
            "loss_bbox": 5.0,
            "loss_giou": 2.0,
            "loss_fgl": 0.15,
            "loss_ddf": 1.5,
        },
        losses=["mal", "boxes", "local"],
        num_classes=4,
        distributed_normalize=False,
    )
    adapter = LoopDVRValidationLoss(model, criterion)
    x = torch.randn(1, 3, 320, 320)
    targets = torch.zeros(1, 5, 5)
    targets[0, 0] = torch.tensor([1.0, 160.0, 160.0, 64.0, 64.0])
    with torch.no_grad(), adapter.forward_scope():
        preds = model(x)
        report = adapter(preds, targets, image_size=(320, 320))
    assert "loss" in report
    assert torch.isfinite(torch.as_tensor(report["loss"]))


@pytest.mark.parametrize("task", ["detect", "segment", "pose"])
def test_train_end_to_end_smoke_all_tasks(tmp_path, task):
    import yaml
    from PIL import Image

    ds = tmp_path / f"ds_{task}"
    for split in ("train", "val"):
        (ds / "images" / split).mkdir(parents=True)
        (ds / "labels" / split).mkdir(parents=True)
        for i in range(2):
            img = Image.fromarray(
                np.full((64, 64, 3), 100 + i * 30, dtype=np.uint8)
            )
            img.save(ds / "images" / split / f"img_{i}.jpg")
            if task == "detect":
                label = "0 0.5 0.5 0.3 0.3\n"
            elif task == "segment":
                label = "0 0.35 0.35 0.65 0.35 0.65 0.65 0.35 0.65\n"
            else:
                kpts = " ".join("0.5 0.5 2" for _ in range(17))
                label = f"0 0.5 0.5 0.4 0.4 {kpts}\n"
            (ds / "labels" / split / f"img_{i}.txt").write_text(label)

    cfg = {
        "path": str(ds),
        "train": "images/train",
        "val": "images/val",
        "nc": 1 if task == "pose" else 2,
        "names": {0: "person"} if task == "pose" else {0: "cat", 1: "dog"},
    }
    if task == "pose":
        cfg["kpt_shape"] = [17, 3]
    yaml_path = ds / "data.yaml"
    yaml_path.write_text(yaml.safe_dump(cfg))

    model = LibreLoopDVR(
        None, size="s", nb_classes=1 if task == "pose" else 2, device="cpu", task=task
    )
    imgsz = 320 if task == "detect" else 640
    results = model.train(
        data=str(yaml_path),
        epochs=1,
        batch=2,
        imgsz=imgsz,
        device="cpu",
        workers=0,
        amp=False,
        ema=False,
        eval_interval=1,
        project=str(tmp_path / "runs"),
        name=f"smoke_{task}",
        exist_ok=True,
    )
    last_ckpt = results.get("last_checkpoint")
    assert last_ckpt and Path(last_ckpt).exists()

    reloaded = LibreYOLO(last_ckpt, device="cpu")
    assert isinstance(reloaded, LibreLoopDVR)
    assert reloaded.task == task
    assert reloaded.size == "s"


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
    # Under untrained random weights, masked-out border cells (all zeros) can
    # share an identical constant score above interior cells, causing PyTorch
    # and ONNX Runtime TopK tie-breaking to select different query indices.
    with torch.no_grad():
        if hasattr(model.model.decoder, "enc_output"):
            model.model.decoder.enc_output.bias.zero_()
            model.model.decoder.enc_output_norm.bias.zero_()
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
        # adjacent query slots may swap between PyTorch and ORT MLAS kernels;
        # check both median and 95th-percentile absolute error.
        diff = np.abs(ref_t.numpy() - ort_a)
        assert float(np.median(diff)) < 1e-3
        assert float(np.percentile(diff, 95)) < 2e-2

    onnx_backend = LibreYOLO(onnx_path, device="cpu")
    assert onnx_backend.model_family == "loopdvr"
    assert onnx_backend.task == task

    r_pt = model.predict(img, conf=0.0, max_det=10)
    r_onnx = onnx_backend.predict(img, conf=0.0, max_det=10)
    assert len(r_pt.boxes) == len(r_onnx.boxes) == 10
    assert np.allclose(
        r_pt.boxes.conf.numpy(), r_onnx.boxes.conf.numpy(), atol=5e-3
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


@pytest.mark.parametrize("task", ["detect", "segment", "pose"])
def test_get_available_layer_names_all_tasks(task):
    nb_classes = 1 if task == "pose" else 4
    model = LibreLoopDVR(None, size="s", nb_classes=nb_classes, device="cpu", task=task)
    names = model.get_available_layer_names()
    assert "backbone" in names
    assert "looped_vit" in names
    assert "encoder" in names
    assert "decoder" in names
    layers = model._get_available_layers()
    assert set(names) == set(layers)
    if task == "pose":
        assert "pose_embed" in layers
        assert "class_embed" in layers
    else:
        assert "dec_bbox_head" in layers
        assert "dec_score_head" in layers
        if task == "segment":
            assert "segmentation_head" in layers


@pytest.mark.parametrize(
    ("task", "extra_cfg"),
    [
        ("detect", {"backbone_lr_mult": 0.25}),
        ("segment", {"mask_ce_loss_weight": 3.5, "mask_dice_loss_weight": 2.5}),
        ("pose", {"dn_number": 12, "oks_loss_weight": 5.5, "oks_sigmas": [0.1] * 17}),
    ],
)
def test_resume_preserves_task_specific_train_config(tmp_path, task, extra_cfg):
    nb_classes = 1 if task == "pose" else 4
    model = LibreLoopDVR(None, size="s", nb_classes=nb_classes, device="cpu", task=task)
    ckpt_path = tmp_path / "run" / "weights" / "last.pt"
    ckpt_path.parent.mkdir(parents=True)
    torch.save(
        {
            "model": model.model.state_dict(),
            "config": {"epochs": 10, "batch": 8, **extra_cfg},
        },
        ckpt_path,
    )
    restored = model._resume_train_args(ckpt_path, given=set())
    for key, value in extra_cfg.items():
        assert restored[key] == value


@pytest.mark.parametrize("task", ["detect", "segment", "pose"])
def test_freeze_backbone_and_looped_vit_selectors(task):
    from libreyolo.models.loopdvr.trainer import (
        LoopDVRPoseTrainer,
        LoopDVRSegTrainer,
        LoopDVRTrainer,
    )
    from libreyolo.training.freezing import apply_freeze

    trainer_cls = {
        "detect": LoopDVRTrainer,
        "segment": LoopDVRSegTrainer,
        "pose": LoopDVRPoseTrainer,
    }[task]
    nb_classes = 1 if task == "pose" else 4
    for selector in ("backbone", "looped_vit", 1):
        wrapper = LibreLoopDVR(
            None, size="s", nb_classes=nb_classes, device="cpu", task=task
        )
        trainer = trainer_cls.__new__(trainer_cls)
        trainer.model = wrapper.model
        groups = trainer.get_freeze_groups()
        summary = apply_freeze(wrapper.model, selector, freeze_groups=groups)
        assert summary is not None
        assert summary.frozen_param_count > 0
        assert all(not p.requires_grad for p in wrapper.model.looped_vit.parameters())
        assert any(p.requires_grad for p in wrapper.model.decoder.parameters())

