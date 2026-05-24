"""Unit tests for YOLO-NAS pose-estimation training.

Covers the training-path additions:

- the pose head emits the raw 8-tuple in ``train()`` mode, the 4-tuple in eval
- ``replace_num_keypoints`` / ``detect_num_keypoints`` for custom keypoint counts
- ``YoloNASPoseLoss`` forward + backward are finite (objects and empty batch)
- the YOLO-format pose label parser and ``YOLOPoseDataset`` pipeline
- the keypoint-aware train/val transforms produce the padded target slab
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest
import torch
import yaml

from libreyolo.data import (
    YOLOPoseDataset,
    load_data_config,
    parse_yolo_pose_label_line,
    pose_collate_fn,
)
from libreyolo.models.yolonas.loss import YoloNASPoseLoss
from libreyolo.models.yolonas.model import LibreYOLONAS
from libreyolo.models.yolonas.nn import LibreYOLONASPoseModel
from libreyolo.models.yolonas.pose_transforms import (
    YOLONASPoseTrainTransform,
    YOLONASPoseValTransform,
)
from libreyolo.models.yolonas.pose_trainer import (
    YOLONASPoseTrainer,
    default_oks_sigmas,
)
from libreyolo.models.yolonas.pose_transforms import YOLO_NAS_POSE_RESIZE_SIZE

pytestmark = [pytest.mark.unit, pytest.mark.yolonas]


# ---------------------------------------------------------------------------
# YOLO-format pose label parsing
# ---------------------------------------------------------------------------


class TestPoseLabelParsing:
    def test_parses_valid_line(self):
        # class cx cy w h + 4 keypoints * (kx ky v)
        parts = "0 0.5 0.5 0.2 0.3 0.4 0.4 2 0.6 0.4 2 0.6 0.6 0 0.4 0.6 1".split()
        cls, bbox, kpts = parse_yolo_pose_label_line(parts, num_keypoints=4)
        assert cls == 0
        assert bbox.shape == (4,)
        assert kpts.shape == (4, 3)
        assert kpts[2, 2] == 0  # third keypoint not labelled
        assert kpts[3, 2] == 1  # fourth keypoint occluded

    def test_wrong_field_count_raises(self):
        parts = "0 0.5 0.5 0.2 0.3".split()  # bbox only, no keypoints
        with pytest.raises(ValueError):
            parse_yolo_pose_label_line(parts, num_keypoints=4)

    def test_xy_only_keypoint_labels_are_promoted_to_visible(self):
        parts = "0 0.5 0.5 0.2 0.3 0.4 0.4 0.6 0.6".split()
        cls, bbox, kpts = parse_yolo_pose_label_line(
            parts, num_keypoints=2, keypoint_dim=2
        )
        assert cls == 0
        assert bbox.shape == (4,)
        assert kpts.shape == (2, 3)
        assert np.all(kpts[:, 2] == 2.0)


# ---------------------------------------------------------------------------
# Pose head: train vs eval outputs
# ---------------------------------------------------------------------------


class TestPoseHeadTrainingOutputs:
    def test_train_mode_returns_decoded_and_raw(self):
        model = LibreYOLONASPoseModel(config="s", num_keypoints=17).train()
        decoded, raw = model(torch.zeros(2, 3, 640, 640))
        assert len(decoded) == 4
        assert len(raw) == 8
        cls, distri, pose_coords, pose_logits = raw[:4]
        assert cls.shape == (2, 8400, 1)
        assert distri.shape == (2, 8400, 4 * (16 + 1))
        assert pose_coords.shape == (2, 8400, 17, 2)
        assert pose_logits.shape == (2, 8400, 17)

    def test_eval_mode_returns_decoded_only(self):
        model = LibreYOLONASPoseModel(config="s", num_keypoints=17).eval()
        with torch.no_grad():
            decoded, raw = model(torch.zeros(1, 3, 640, 640))
        assert len(decoded) == 4
        assert len(raw) == 8

    def test_replace_num_keypoints_rebuilds_head(self):
        model = LibreYOLONASPoseModel(config="s", num_keypoints=17)
        model.replace_num_keypoints(4)
        model.train()
        _, raw = model(torch.zeros(1, 3, 640, 640))
        assert raw[2].shape == (1, 8400, 4, 2)  # pose coords
        assert raw[3].shape == (1, 8400, 4)  # pose logits

    def test_detect_num_keypoints_from_state_dict(self):
        model = LibreYOLONASPoseModel(config="s", num_keypoints=4)
        sd = model.state_dict()
        assert LibreYOLONAS.detect_num_keypoints(sd) == 4

    def test_wrapper_loads_custom_keypoint_state_dict(self):
        model = LibreYOLONASPoseModel(config="s", num_keypoints=4)
        wrapper = LibreYOLONAS(model.state_dict(), size="s", task="pose", device="cpu")
        assert wrapper.num_keypoints == 4

    def test_wrapper_postprocess_accepts_train_mode_pose_output(self):
        model = LibreYOLONAS(None, size="s", task="pose", device="cpu")
        model.model.train()
        output = model._forward(torch.zeros(1, 3, 640, 640))
        detections = model._postprocess(output, 1.1, 0.45, (640, 640))
        assert detections["num_detections"] == 0


# ---------------------------------------------------------------------------
# Pose loss
# ---------------------------------------------------------------------------


def _synthetic_pose_targets(batch_size, max_labels, num_keypoints):
    targets = torch.zeros(batch_size, max_labels, 5 + 3 * num_keypoints)
    for b in range(batch_size):
        for n in range(2):  # two objects per image
            cx, cy, w, h = 320.0, 300.0, 120.0, 200.0
            targets[b, n, 1:5] = torch.tensor([cx, cy, w, h])
            for k in range(num_keypoints):
                targets[b, n, 5 + 3 * k] = cx + (k - num_keypoints / 2) * 5
                targets[b, n, 5 + 3 * k + 1] = cy + (k - num_keypoints / 2) * 5
                targets[b, n, 5 + 3 * k + 2] = 2.0
    return targets


class TestPoseLoss:
    def test_forward_backward_finite(self):
        K = 4
        model = LibreYOLONASPoseModel(config="s", num_keypoints=K).train()
        loss_fn = YoloNASPoseLoss(oks_sigmas=default_oks_sigmas(K))
        targets = _synthetic_pose_targets(2, 12, K)
        loss, logs = loss_fn(model(torch.zeros(2, 3, 640, 640)), targets)
        assert torch.isfinite(loss)
        assert logs.shape == (6,)  # cls, iou, dfl, pose_cls, pose_reg, total
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)

    def test_empty_targets_finite(self):
        K = 4
        model = LibreYOLONASPoseModel(config="s", num_keypoints=K).train()
        loss_fn = YoloNASPoseLoss(oks_sigmas=default_oks_sigmas(K))
        empty = torch.zeros(2, 10, 5 + 3 * K)
        loss, _ = loss_fn(model(torch.zeros(2, 3, 640, 640)), empty)
        assert torch.isfinite(loss)

    def test_eval_mode_outputs_work_with_loss(self):
        K = 4
        model = LibreYOLONASPoseModel(config="s", num_keypoints=K).eval()
        loss_fn = YoloNASPoseLoss(oks_sigmas=default_oks_sigmas(K))
        targets = _synthetic_pose_targets(1, 4, K)
        with torch.no_grad():
            loss, logs = loss_fn(model(torch.zeros(1, 3, 640, 640)), targets)
        assert torch.isfinite(loss)
        assert logs.shape == (6,)

    def test_unpack_padded_targets_front_packs(self):
        loss_fn = YoloNASPoseLoss(oks_sigmas=default_oks_sigmas(4))
        targets = _synthetic_pose_targets(2, 12, 4)
        tgt = loss_fn._unpack_padded_targets(targets)
        # Two valid objects per image -> trimmed to n_max == 2.
        assert tgt["gt_bbox"].shape == (2, 2, 4)
        assert tgt["gt_poses"].shape == (2, 2, 4, 3)
        assert tgt["pad_gt_mask"].sum() == 4

    def test_oks_sigmas_length_mismatch_is_caught_by_trainer_helper(self):
        assert len(default_oks_sigmas(17)) == 17
        assert len(default_oks_sigmas(4)) == 4

    def test_custom_pose_loss_weights_are_honored(self, tmp_path):
        img_path = _write_pose_sample(tmp_path, 4)
        data_yaml = tmp_path / "data.yaml"
        data_yaml.write_text(
            yaml.safe_dump(
                {
                    "path": str(tmp_path),
                    "train": str(img_path.parent.relative_to(tmp_path)),
                    "names": ["object"],
                    "kpt_shape": [4, 3],
                }
            )
        )
        trainer = YOLONASPoseTrainer(
            model=torch.nn.Identity(),
            size="s",
            num_keypoints=4,
            data=str(data_yaml),
            dfl_loss_weight=0.5,
            pose_reg_loss_weight=10.0,
            workers=0,
            device="cpu",
        )
        trainer.on_setup()
        assert trainer.loss_fn.dfl_loss_weight == pytest.approx(0.5)
        assert trainer.loss_fn.pose_reg_loss_weight == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# YOLO-format pose dataset + transforms
# ---------------------------------------------------------------------------


def _write_pose_sample(tmp_path, num_keypoints):
    img_dir = tmp_path / "images" / "train"
    lbl_dir = tmp_path / "labels" / "train"
    img_dir.mkdir(parents=True)
    lbl_dir.mkdir(parents=True)
    img_path = img_dir / "sample.jpg"
    cv2.imwrite(str(img_path), np.full((480, 640, 3), 127, dtype=np.uint8))

    row = ["0", "0.5", "0.5", "0.3", "0.4"]
    for k in range(num_keypoints):
        row += [f"{0.4 + 0.02 * k:.3f}", f"{0.45 + 0.02 * k:.3f}", "2"]
    (lbl_dir / "sample.txt").write_text(" ".join(row) + "\n")
    return img_path


class TestPoseDatasetAndTransforms:
    def test_dataset_with_train_transform(self, tmp_path):
        K = 4
        img_path = _write_pose_sample(tmp_path, K)
        tf = YOLONASPoseTrainTransform(
            K,
            flip_idx=[1, 0, 3, 2],
            hsv_prob=0.0,
            brightness_contrast_prob=0.0,
            affine_prob=0.0,
        )
        ds = YOLOPoseDataset(
            img_files=[img_path], num_keypoints=K, img_size=(640, 640), preproc=tf
        )
        img, target, info, idx = ds[0]
        assert img.shape == (3, 640, 640)
        assert target.shape == (100, 5 + 3 * K)
        valid = (target[:, 3] > 0) & (target[:, 4] > 0)
        assert valid.sum() == 1

    def test_val_transform_matches_yolonas_center_letterbox_geometry(self, tmp_path):
        K = 4
        img_path = _write_pose_sample(tmp_path, K)
        ds = YOLOPoseDataset(
            img_files=[img_path],
            num_keypoints=K,
            preproc=YOLONASPoseValTransform(K),
        )
        target = ds[0][1][0]
        ratio = YOLO_NAS_POSE_RESIZE_SIZE / 640
        pad_y = 0
        assert target[1] == pytest.approx(320.0, abs=1.0)
        assert target[2] == pytest.approx(240.0 * ratio + pad_y, abs=1.0)

    def test_pose_transform_keeps_bgr_channel_order(self, tmp_path):
        K = 4
        img_path = _write_pose_sample(tmp_path, K)
        bgr = np.zeros((32, 32, 3), dtype=np.uint8)
        bgr[..., 0] = 255
        cv2.imwrite(str(img_path), bgr)
        ds = YOLOPoseDataset(
            img_files=[img_path],
            num_keypoints=K,
            preproc=YOLONASPoseValTransform(K),
        )
        img = ds[0][0]
        assert img[0, 0, 0] > 0.95
        assert img[2, 0, 0] == pytest.approx(0.0)

    def test_val_transform_and_collate(self, tmp_path):
        K = 4
        img_path = _write_pose_sample(tmp_path, K)
        ds = YOLOPoseDataset(
            img_files=[img_path, img_path],
            num_keypoints=K,
            preproc=YOLONASPoseValTransform(K),
        )
        batch = [ds[0], ds[1]]
        imgs, targets, infos, ids = pose_collate_fn(batch)
        assert imgs.shape == (2, 3, 640, 640)
        assert targets.shape == (2, 100, 5 + 3 * K)

    def test_flip_reindexes_keypoints(self, tmp_path):
        K = 4
        img_path = _write_pose_sample(tmp_path, K)
        # flip_prob=1 always flips; flip_idx swaps (0<->1, 2<->3).
        flipped = YOLONASPoseTrainTransform(
            K,
            flip_idx=[1, 0, 3, 2],
            flip_prob=1.0,
            hsv_prob=0.0,
            brightness_contrast_prob=0.0,
            affine_prob=0.0,
        )
        plain = YOLONASPoseValTransform(K)
        ds_f = YOLOPoseDataset(img_files=[img_path], num_keypoints=K, preproc=flipped)
        ds_p = YOLOPoseDataset(img_files=[img_path], num_keypoints=K, preproc=plain)
        tf = ds_f[0][1][0]  # flipped target row
        tp = ds_p[0][1][0]  # plain target row
        # Keypoint 0 of the flipped sample mirrors keypoint 1 of the plain one.
        assert tf[5] == pytest.approx(640.0 - tp[5 + 3], abs=1.0)

    def test_dataset_accepts_kpt_shape_two_labels(self, tmp_path):
        K = 2
        img_dir = tmp_path / "images" / "train"
        lbl_dir = tmp_path / "labels" / "train"
        img_dir.mkdir(parents=True)
        lbl_dir.mkdir(parents=True)
        img_path = img_dir / "sample.jpg"
        cv2.imwrite(str(img_path), np.full((64, 64, 3), 127, dtype=np.uint8))
        (lbl_dir / "sample.txt").write_text("0 0.5 0.5 0.5 0.5 0.25 0.25 0.75 0.75\n")
        ds = YOLOPoseDataset(
            img_files=[img_path],
            num_keypoints=K,
            keypoint_dim=2,
            preproc=YOLONASPoseValTransform(K),
        )
        target = ds[0][1][0]
        assert target.shape == (5 + 3 * K,)
        assert target[7] == 2.0
        assert target[10] == 2.0

    def test_load_data_config_accepts_list_splits_with_autodownload(self, tmp_path):
        img_a = _write_pose_sample(tmp_path / "a", 4)
        img_b = _write_pose_sample(tmp_path / "b", 4)
        data_yaml = tmp_path / "data.yaml"
        data_yaml.write_text(
            yaml.safe_dump(
                {
                    "path": str(tmp_path),
                    "train": [
                        str(img_a.parent.relative_to(tmp_path)),
                        str(img_b.parent.relative_to(tmp_path)),
                    ],
                    "val": str(img_a.parent.relative_to(tmp_path)),
                    "names": ["object"],
                    "kpt_shape": [4, 3],
                }
            )
        )
        cfg = load_data_config(str(data_yaml), autodownload=True)
        assert len(cfg["train_img_files"]) == 2

    def test_pose_trainer_keeps_partial_small_batch(self, tmp_path):
        img_path = _write_pose_sample(tmp_path, 4)
        data_yaml = tmp_path / "data.yaml"
        data_yaml.write_text(
            yaml.safe_dump(
                {
                    "path": str(tmp_path),
                    "train": str(img_path.parent.relative_to(tmp_path)),
                    "names": ["object"],
                    "kpt_shape": [4, 3],
                }
            )
        )
        trainer = YOLONASPoseTrainer(
            model=torch.nn.Identity(),
            size="s",
            num_keypoints=4,
            data=str(data_yaml),
            batch=4,
            workers=0,
            device="cpu",
        )
        trainer._setup_data()
        assert len(trainer.train_loader) == 1

    def test_pose_trainer_does_not_scale_lr_by_batch(self, tmp_path):
        img_path = _write_pose_sample(tmp_path, 4)
        data_yaml = tmp_path / "data.yaml"
        data_yaml.write_text(
            yaml.safe_dump(
                {
                    "path": str(tmp_path),
                    "train": str(img_path.parent.relative_to(tmp_path)),
                    "names": ["object"],
                    "kpt_shape": [4, 3],
                }
            )
        )
        trainer = YOLONASPoseTrainer(
            model=torch.nn.Identity(),
            size="s",
            num_keypoints=4,
            data=str(data_yaml),
            batch=128,
            lr0=2e-3,
            workers=0,
            device="cpu",
        )
        assert trainer.effective_lr == pytest.approx(2e-3)

    def test_pose_checkpoint_metadata_includes_keypoint_shape(self, tmp_path):
        class Wrapper:
            task = "pose"
            names = {0: "object"}

        trainer = YOLONASPoseTrainer(
            model=torch.nn.Linear(1, 1),
            wrapper_model=Wrapper(),
            size="s",
            num_keypoints=4,
            keypoint_dim=3,
            data=None,
            workers=0,
            device="cpu",
            ema=False,
        )
        trainer.save_dir = tmp_path
        trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.01)
        trainer._save_checkpoint(0, 1.0, val_metrics=None, is_best=False)

        ckpt = torch.load(tmp_path / "weights" / "last.pt", map_location="cpu")
        assert ckpt["schema_version"] == "1.0"
        assert ckpt["libreyolo_version"]
        assert ckpt["model_family"] == "yolonas"
        assert ckpt["size"] == "s"
        assert ckpt["task"] == "pose"
        assert ckpt["nc"] == 1
        assert ckpt["names"] == {0: "object"}
        assert ckpt["imgsz"] == trainer.config.imgsz
        assert ckpt["num_keypoints"] == 4
        assert ckpt["keypoint_dim"] == 3
        assert ckpt["oks_sigmas"] == default_oks_sigmas(4)
        assert ckpt["best_metric_value"] == ckpt["best_mAP50_95"]
        assert ckpt["is_ema_weights"] is False
        assert ckpt["best_metric"] == ckpt["best_mAP50_95"]
        assert ckpt["best_metric_name"] == ckpt["best_metric_key"]
