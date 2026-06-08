"""Unit tests for RF-DETR pose support."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn

pytestmark = pytest.mark.unit


def test_rfdetr_pose_download_url_and_notice():
    from libreyolo.models.rfdetr.model import LibreRFDETR

    url = LibreRFDETR.get_download_url("LibreRFDETRn-pose.pt")

    assert (
        url
        == "https://huggingface.co/LibreYOLO/LibreRFDETRn-pose/resolve/main/LibreRFDETRn-pose.pt"
    )
    assert "EXTREMELY experimental" in LibreRFDETR.get_download_notice(
        "LibreRFDETRn-pose.pt",
        url,
    )
    assert "EXTREMELY experimental" in LibreRFDETR.get_download_notice(
        "LibreRFDETRs-pose.pt",
        LibreRFDETR.get_download_url("LibreRFDETRs-pose.pt"),
    )
    assert "EXTREMELY experimental" in LibreRFDETR.get_download_notice(
        "LibreRFDETRm-pose.pt",
        LibreRFDETR.get_download_url("LibreRFDETRm-pose.pt"),
    )
    assert "EXTREMELY experimental" in LibreRFDETR.get_download_notice(
        "LibreRFDETRl-pose.pt",
        LibreRFDETR.get_download_url("LibreRFDETRl-pose.pt"),
    )
    assert LibreRFDETR.get_download_notice("LibreRFDETRn.pt", url) is None


def test_rfdetr_pose_transform_square_resizes_boxes_and_keypoints():
    from libreyolo.models.rfdetr.pose_transforms import RFDETRPoseTransform

    image = np.zeros((20, 40, 3), dtype=np.uint8)
    boxes = np.array([[0.5, 0.5, 0.5, 0.5]], dtype=np.float32)
    cls = np.array([0], dtype=np.float32)
    kpts = np.array([[[0.25, 0.5, 2.0], [0.75, 0.25, 1.0]]], dtype=np.float32)
    transform = RFDETRPoseTransform(2, max_labels=4, flip_prob=0.0, imgsz=80)

    img, target = transform(image, boxes, cls, kpts, (80, 80))

    assert img.shape == (3, 80, 80)
    assert target.shape == (4, 11)
    assert target[0, :5].tolist() == pytest.approx([0.0, 40.0, 40.0, 40.0, 40.0])
    assert target[0, 5:11].tolist() == pytest.approx([20.0, 40.0, 2.0, 60.0, 20.0, 1.0])


def test_rfdetr_pose_transform_zeroes_visibility_for_outside_keypoints():
    from libreyolo.models.rfdetr.pose_transforms import RFDETRPoseTransform

    image = np.zeros((20, 40, 3), dtype=np.uint8)
    boxes = np.array([[0.5, 0.5, 0.5, 0.5]], dtype=np.float32)
    cls = np.array([0], dtype=np.float32)
    kpts = np.array([[[0.5, 0.5, 2.0], [1.25, -0.25, 1.0]]], dtype=np.float32)
    transform = RFDETRPoseTransform(2, max_labels=4, flip_prob=0.0, imgsz=80)

    _img, target = transform(image, boxes, cls, kpts, (80, 80))

    assert target[0, 5:11].tolist() == pytest.approx([40.0, 40.0, 2.0, 80.0, 0.0, 0.0])


def test_rfdetr_pose_transform_flips_and_reindexes_keypoints(monkeypatch):
    from libreyolo.models.rfdetr.pose_transforms import RFDETRPoseTransform

    monkeypatch.setattr("libreyolo.models.rfdetr.pose_transforms.random.random", lambda: 0.0)
    image = np.zeros((20, 40, 3), dtype=np.uint8)
    boxes = np.array([[0.25, 0.5, 0.25, 0.5]], dtype=np.float32)
    cls = np.array([0], dtype=np.float32)
    kpts = np.array([[[0.25, 0.5, 2.0], [0.75, 0.5, 2.0]]], dtype=np.float32)
    transform = RFDETRPoseTransform(
        2,
        flip_idx=[1, 0],
        max_labels=4,
        flip_prob=1.0,
        imgsz=80,
    )

    _img, target = transform(image, boxes, cls, kpts, (80, 80))

    assert target[0, :5].tolist() == pytest.approx([0.0, 60.0, 40.0, 20.0, 40.0])
    assert target[0, 5:11].tolist() == pytest.approx([20.0, 40.0, 2.0, 60.0, 40.0, 2.0])


def test_rfdetr_trainer_converts_pose_targets_to_normalized_keypoints():
    from libreyolo.models.rfdetr.config import RFDETRConfig
    from libreyolo.models.rfdetr.trainer import RFDETRTrainer

    trainer = RFDETRTrainer.__new__(RFDETRTrainer)
    trainer.wrapper_model = SimpleNamespace(task="pose")
    trainer.config = RFDETRConfig(num_keypoints=2)
    trainer.device = torch.device("cpu")
    targets = torch.zeros(1, 3, 11)
    targets[0, 0] = torch.tensor(
        [0.0, 16.0, 32.0, 8.0, 10.0, 12.0, 20.0, 2.0, 40.0, 24.0, 2.0]
    )

    target_list = trainer._targets_to_rfdetr_list(targets, height=64, width=32)

    assert target_list[0]["boxes"].shape == (1, 4)
    assert target_list[0]["boxes"][0].tolist() == pytest.approx([0.5, 0.5, 0.25, 0.15625])
    assert target_list[0]["keypoints"].shape == (1, 2, 3)
    assert target_list[0]["keypoints"][0].reshape(-1).tolist() == pytest.approx(
        [0.375, 0.3125, 2.0, 1.0, 0.375, 0.0]
    )


def test_rfdetr_pose_loss_runs_and_no_match_zero_is_connected():
    from libreyolo.models.rfdetr.loss import SetCriterion

    criterion = SetCriterion(
        num_classes=1,
        matcher=None,
        weight_dict={},
        focal_alpha=0.25,
        losses=["keypoints"],
        num_keypoints=2,
    )
    pred_keypoints = torch.tensor(
        [[[[0.5, 0.5, 0.1], [0.25, 0.25, -0.1]]]],
        dtype=torch.float32,
        requires_grad=True,
    )
    targets = [
        {
            "labels": torch.tensor([0], dtype=torch.long),
            "boxes": torch.tensor([[0.5, 0.5, 0.5, 0.5]], dtype=torch.float32),
            "keypoints": torch.tensor(
                [[[0.5, 0.5, 2.0], [0.25, 0.25, 0.0]]],
                dtype=torch.float32,
            ),
        }
    ]

    losses = criterion.loss_keypoints(
        {"pred_keypoints": pred_keypoints},
        targets,
        [(torch.tensor([0]), torch.tensor([0]))],
        num_boxes=1.0,
    )

    assert set(losses) == {
        "loss_keypoints_l1",
        "loss_keypoints_oks",
        "loss_keypoints_vis",
    }
    assert all(torch.isfinite(v) for v in losses.values())

    empty_losses = criterion.loss_keypoints(
        {"pred_keypoints": pred_keypoints},
        [{"labels": torch.zeros(0, dtype=torch.long), "boxes": torch.zeros(0, 4), "keypoints": torch.zeros(0, 2, 3)}],
        [(torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long))],
        num_boxes=1.0,
    )
    zero = sum(empty_losses.values())
    zero.backward()
    assert pred_keypoints.grad is not None
    assert pred_keypoints.grad.abs().sum().item() == 0.0


def test_rfdetr_pose_l1_loss_is_area_normalized():
    from libreyolo.models.rfdetr.loss import SetCriterion

    criterion = SetCriterion(
        num_classes=1,
        matcher=None,
        weight_dict={},
        focal_alpha=0.25,
        losses=["keypoints"],
        num_keypoints=1,
    )
    outputs = {
        "pred_keypoints": torch.tensor(
            [[[[0.6, 0.5, 1.0]], [[0.6, 0.5, 1.0]]]],
            dtype=torch.float32,
        )
    }
    targets = [
        {
            "labels": torch.tensor([0, 0], dtype=torch.long),
            "boxes": torch.tensor(
                [[0.5, 0.5, 0.1, 0.1], [0.5, 0.5, 1.0, 1.0]],
                dtype=torch.float32,
            ),
            "keypoints": torch.tensor(
                [[[0.5, 0.5, 2.0]], [[0.5, 0.5, 2.0]]],
                dtype=torch.float32,
            ),
        }
    ]

    losses = criterion.loss_keypoints(
        outputs,
        targets,
        [(torch.tensor([0, 1]), torch.tensor([0, 1]))],
        num_boxes=2.0,
    )

    assert losses["loss_keypoints_l1"].item() == pytest.approx(0.55, rel=1e-5)


def test_lwdetr_keypoint_decode_mirrors_bbox_reparam_without_sigmoid():
    from libreyolo.models.rfdetr.lwdetr import LWDETR

    class _Head:
        def __call__(self, hs):
            raw = torch.tensor([0.5, -0.5, 3.0], dtype=hs.dtype, device=hs.device)
            return raw.view(1, 1, 1, 3).expand(*hs.shape[:-1], 1, 3).reshape(*hs.shape[:-1], 3)

    model = object.__new__(LWDETR)
    model.keypoint_head = _Head()
    model.num_keypoints = 1
    model.bbox_reparam = True
    hs = torch.zeros(1, 1, 1, 4)
    reference = torch.tensor([[[[0.25, 0.75, 0.20, 0.40]]]], dtype=torch.float32)

    decoded = LWDETR._decode_keypoints(model, hs, reference)

    assert decoded[0, 0, 0, 0].tolist() == pytest.approx([0.35, 0.55, 3.0])


def test_rfdetr_postprocess_filters_keypoints_in_topk_lockstep():
    from libreyolo.models.rfdetr.utils import postprocess

    outputs = {
        "pred_logits": torch.tensor([[[5.0], [4.0], [-1.0]]]),
        "pred_boxes": torch.tensor([[[0.5, 0.5, 0.2, 0.2], [0.2, 0.2, 0.1, 0.1], [0.8, 0.8, 0.1, 0.1]]]),
        "pred_keypoints": torch.zeros(1, 3, 2, 3),
    }
    outputs["pred_keypoints"][0, :, :, 0] = torch.tensor([0.1, 0.2, 0.3]).view(3, 1)
    outputs["pred_keypoints"][0, :, :, 1] = 0.5
    outputs["pred_keypoints"][0, :, :, 2] = 2.0

    result = postprocess(outputs, torch.tensor([[100.0, 200.0]]), num_select=2)[0]

    assert result["keypoints"].shape == (2, 2, 3)
    assert result["keypoints"][0, :, 0].tolist() == pytest.approx([20.0, 20.0])
    assert result["keypoints"][1, :, 0].tolist() == pytest.approx([40.0, 40.0])


def test_rfdetr_pose_fresh_model_uses_one_class_logit():
    from libreyolo.models.rfdetr.model import LibreRFDETR

    model = LibreRFDETR({}, size="n", task="pose", device="cpu")

    assert model.nb_classes == 1
    assert model.model.nb_classes == 1
    assert model.model.args.num_classes == 0
    assert model.model.model.class_embed.out_features == 1
    assert model.names == {0: "person"}


def test_rfdetr_pose_none_builds_scratch_without_default_detect_load(monkeypatch):
    from libreyolo.models.rfdetr.model import LibreRFDETR

    def fail_load(self, model_path):
        raise AssertionError("pose scratch construction should not load detect weights")

    monkeypatch.setattr(LibreRFDETR, "_load_weights", fail_load)

    model = LibreRFDETR(None, size="n", task="pose", device="cpu")

    assert model.task == "pose"
    assert model.nb_classes == 1
    assert model.names == {0: "person"}


def test_rfdetr_pose_direct_load_rejects_detect_checkpoint():
    from libreyolo.models.rfdetr.model import LibreRFDETR

    with pytest.raises(ValueError, match="task='detect'"):
        LibreRFDETR(
            {
                "model_family": "rfdetr",
                "task": "detect",
                "model": {},
            },
            size="n",
            task="pose",
            device="cpu",
        )


def test_rfdetr_pose_detect_transfer_resizes_to_one_class_logit():
    from libreyolo.models.rfdetr.model import LibreRFDETR

    model = LibreRFDETR({}, size="n", task="pose", device="cpu")
    model._allow_detect_to_pose_transfer = True
    det_model = LibreRFDETR({}, size="n", task="detect", device="cpu")
    detect_state = {
        key: value.detach().clone()
        for key, value in det_model.model.state_dict().items()
    }

    model._load_weights(
        {
            "model_family": "rfdetr",
            "task": "detect",
            "nc": 80,
            "names": {i: f"class_{i}" for i in range(80)},
            "model": detect_state,
        }
    )

    assert model.nb_classes == 1
    assert model.model.nb_classes == 1
    assert model.model.args.num_classes == 0
    assert model.model.model.class_embed.out_features == 1
    assert model.names == {0: "person"}


def test_rfdetr_pose_load_weights_rejects_detect_without_transfer_flag():
    from libreyolo.models.rfdetr.model import LibreRFDETR

    model = LibreRFDETR({}, size="n", task="pose", device="cpu")
    det_model = LibreRFDETR({}, size="n", task="detect", device="cpu")
    detect_state = {
        key: value.detach().clone()
        for key, value in det_model.model.state_dict().items()
    }

    with pytest.raises(RuntimeError, match="task='detect'"):
        model._load_weights(
            {
                "model_family": "rfdetr",
                "task": "detect",
                "nc": 80,
                "names": {i: f"class_{i}" for i in range(80)},
                "model": detect_state,
            }
        )


def test_rfdetr_detect_nb_classes_keeps_raw_pose_at_one_class():
    from libreyolo.models.rfdetr.model import LibreRFDETR

    state = {
        "class_embed.bias": torch.zeros(1),
        "keypoint_head.layers.2.weight": torch.zeros(6, 32),
    }

    assert LibreRFDETR.detect_nb_classes(state) == 1


def test_rfdetr_pose_checkpoint_requires_keypoint_head_weights():
    from libreyolo.models.rfdetr.model import LibreRFDETR

    model = LibreRFDETR({}, size="n", task="pose", device="cpu")
    det_model = LibreRFDETR({}, size="n", task="detect", device="cpu")
    detect_state = {
        key: value.detach().clone()
        for key, value in det_model.model.state_dict().items()
    }

    with pytest.raises(RuntimeError, match="keypoint_head"):
        model._load_weights(
            {
                "model_family": "rfdetr",
                "task": "pose",
                "nc": 1,
                "names": {0: "person"},
                "model": detect_state,
            }
        )


def test_rfdetr_pose_load_restores_keypoint_count_from_head_weights():
    from libreyolo.models.rfdetr.model import LibreRFDETR

    model = LibreRFDETR({}, size="n", task="pose", num_keypoints=17, device="cpu")
    pose_model = LibreRFDETR({}, size="n", task="pose", num_keypoints=2, device="cpu")
    pose_state = {
        key: value.detach().clone()
        for key, value in pose_model.model.state_dict().items()
    }

    model._load_weights(
        {
            "model_family": "rfdetr",
            "task": "pose",
            "nc": 1,
            "names": {0: "person"},
            "model": pose_state,
        }
    )

    assert model.num_keypoints == 2


def test_rfdetr_pose_postprocess_trims_legacy_extra_class_column():
    from libreyolo.models.rfdetr.model import LibreRFDETR

    model = LibreRFDETR({}, size="n", task="pose", device="cpu")
    output = {
        "pred_logits": torch.tensor([[[0.1, 10.0], [5.0, -10.0]]]),
        "pred_boxes": torch.tensor([[[0.1, 0.1, 0.1, 0.1], [0.5, 0.5, 0.2, 0.2]]]),
        "pred_keypoints": torch.zeros(1, 2, 2, 3),
    }

    result = model._postprocess(
        output,
        conf_thres=0.01,
        iou_thres=0.5,
        original_size=(100, 100),
        max_det=1,
    )

    assert result["classes"] == [0]
    assert result["boxes"][0] == pytest.approx([40.0, 40.0, 60.0, 60.0])


def test_rfdetr_pose_ddp_uses_find_unused_not_static_graph():
    from libreyolo.models.rfdetr.trainer import RFDETRTrainer

    trainer = RFDETRTrainer.__new__(RFDETRTrainer)
    trainer.wrapper_model = SimpleNamespace(task="pose")

    kwargs = trainer._ddp_kwargs()

    assert kwargs["find_unused_parameters"] is True
    assert kwargs["static_graph"] is False


def test_rfdetr_pose_ddp_validation_skips_rank_zero_criterion_collective():
    from libreyolo.models.rfdetr.trainer import RFDETRTrainer

    class _DummyModel(nn.Module):
        def forward(self, imgs, targets=None):
            raise AssertionError("criterion validation forward should be skipped under DDP")

    class _DummyCriterion:
        weight_dict = {}

        def __call__(self, outputs, targets):
            raise AssertionError("criterion should be skipped under DDP")

    trainer = RFDETRTrainer.__new__(RFDETRTrainer)
    trainer.wrapper_model = SimpleNamespace(task="pose")
    trainer.model = _DummyModel()
    trainer.ema_model = None
    trainer.criterion = _DummyCriterion()
    trainer.val_loader = [(torch.zeros(1, 3, 16, 16), torch.zeros(1, 1, 11))]
    trainer.device = torch.device("cpu")
    trainer.is_distributed = True
    trainer._run_pose_metric_validation = lambda *args, **kwargs: {
        "metrics/keypoints_mAP50-95": 0.25,
        "metrics/keypoints_mAP50": 0.5,
    }

    result = trainer._run_validation(0)

    assert result["best_metric"] == pytest.approx(0.25)
    assert result["metrics"]["loss/val"] == pytest.approx(0.0)


def test_rfdetr_pose_train_resolves_kpt_shape_and_person_class(monkeypatch, tmp_path):
    from libreyolo.models.rfdetr import model as rfdetr_model

    data_yaml = tmp_path / "pose.yaml"
    data_yaml.write_text(
        "path: .\ntrain: images/train\nval: images/val\nnc: 1\nnames: [person]\nkpt_shape: [2, 3]\n",
        encoding="utf-8",
    )
    captured = {}

    class _Inner:
        def __init__(self):
            self.reinitialized = None

        def reinitialize_keypoint_head(self, num_keypoints):
            self.reinitialized = num_keypoints

    class _Model:
        def __init__(self):
            self.num_keypoints = 17
            self.model = _Inner()
            self.args = SimpleNamespace(num_keypoints=17)

    class _DummyTrainer:
        def __init__(self, model, wrapper_model=None, **kwargs):
            captured["model"] = model
            captured["wrapper"] = wrapper_model
            captured["kwargs"] = kwargs

        def train(self):
            return {"save_dir": str(tmp_path / "run")}

    wrapper = rfdetr_model.LibreRFDETR.__new__(rfdetr_model.LibreRFDETR)
    wrapper.task = "pose"
    wrapper.model = _Model()
    wrapper.size = "n"
    wrapper.nb_classes = 80
    wrapper.names = {i: f"class_{i}" for i in range(80)}
    wrapper.input_size = 384
    wrapper.device = torch.device("cpu")
    monkeypatch.setattr(rfdetr_model, "RFDETRTrainer", _DummyTrainer)
    monkeypatch.setattr(wrapper, "_restore_after_training", lambda _result: None)

    wrapper.train(data=str(data_yaml), epochs=1, batch_size=2, lr=1e-4)

    assert wrapper.model.num_keypoints == 2
    assert wrapper.model.model.reinitialized == 2
    assert wrapper.nb_classes == 1
    assert wrapper.names == {0: "person"}
    assert captured["kwargs"]["num_keypoints"] == 2
    assert captured["kwargs"]["keypoint_dim"] == 3
    assert captured["kwargs"]["num_classes"] == 1


def test_rfdetr_pose_onnx_uses_keypoint_output_name(tmp_path):
    onnx = pytest.importorskip("onnx")
    from libreyolo.export.onnx import export_onnx

    class _TinyRFDETRPose(nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = nn.Parameter(torch.zeros(()))

        def forward(self, x):
            batch = x.shape[0]
            signal = x.mean(dim=(1, 2, 3), keepdim=True) + self.anchor
            boxes = signal.reshape(batch, 1, 1).expand(batch, 3, 4)
            logits = signal.reshape(batch, 1, 1).expand(batch, 3, 2)
            keypoints = signal.reshape(batch, 1, 1, 1).expand(batch, 3, 2, 3)
            return boxes, logits, keypoints

    output_path = tmp_path / "rfdetr-pose.onnx"
    export_onnx(
        _TinyRFDETRPose(),
        torch.zeros(1, 3, 32, 32),
        output_path=str(output_path),
        opset=17,
        simplify=False,
        dynamic=False,
        half=False,
        metadata={"model_family": "rfdetr", "task": "pose", "segmentation": "false"},
    )

    proto = onnx.load(output_path)
    assert [i.name for i in proto.graph.input] == ["input"]
    assert [o.name for o in proto.graph.output] == ["dets", "labels", "keypoints"]
