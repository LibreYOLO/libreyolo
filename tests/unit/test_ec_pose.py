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

import numpy as np
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

    def test_train_pose_requires_allow_experimental(self):
        # Pose training is implemented but gated behind the experimental flag.
        m = LibreEC(model_path=None, size="s", task="pose")
        with pytest.raises(RuntimeError, match="experimental"):
            m.train(data="dummy.yaml")

    def test_train_pose_selects_pose_trainer(self):
        # With the flag set, the pose task dispatches to the pose path (it fails
        # only later, on the missing dummy dataset — not with NotImplementedError).
        m = LibreEC(model_path=None, size="s", task="pose")
        with pytest.raises(FileNotFoundError):
            m.train(data="definitely_missing.yaml", allow_experimental=True)

    def test_train_pose_rejects_multiclass_yaml(self, tmp_path):
        data_yaml = tmp_path / "pose.yaml"
        data_yaml.write_text(
            "\n".join(
                [
                    "path: .",
                    "train: images/train",
                    "nc: 2",
                    "names: [person, dog]",
                    "kpt_shape: [17, 3]",
                ]
            ),
            encoding="utf-8",
        )

        m = LibreEC(model_path=None, size="s", task="pose")
        with pytest.raises(ValueError, match="single-class pose datasets"):
            m.train(data=str(data_yaml), allow_experimental=True)

    def test_train_pose_rejects_non_native_imgsz(self):
        m = LibreEC(model_path=None, size="s", task="pose")
        with pytest.raises(ValueError, match="imgsz=640"):
            m.train(data="dummy.yaml", allow_experimental=True, imgsz=320)

    def test_train_pose_explicit_flip_idx_overrides_yaml(self, tmp_path, monkeypatch):
        data_yaml = tmp_path / "pose.yaml"
        data_yaml.write_text(
            "\n".join(
                [
                    "path: .",
                    "train: images/train",
                    "nc: 1",
                    "names: [person]",
                    "kpt_shape: [17, 3]",
                    "flip_idx: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]",
                ]
            ),
            encoding="utf-8",
        )
        captured = {}

        class DummyTrainer:
            def __init__(self, **kwargs):
                captured.update(kwargs)

            def train(self):
                return {}

        monkeypatch.setattr("libreyolo.models.ec.pose_trainer.ECPoseTrainer", DummyTrainer)
        explicit_flip_idx = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]

        m = LibreEC(model_path=None, size="s", task="pose")
        m.train(
            data=str(data_yaml),
            allow_experimental=True,
            workers=0,
            flip_idx=explicit_flip_idx,
        )

        assert captured["flip_idx"] == explicit_flip_idx


class TestPoseTransforms:
    def test_ec_pose_val_transform_direct_resizes_rectangular_targets(self):
        from libreyolo.models.ec.pose_transforms import ECPoseValTransform

        img = np.zeros((100, 200, 3), dtype=np.uint8)
        bboxes = np.array([[0.5, 0.5, 0.2, 0.4]], dtype=np.float32)
        cls = np.array([0], dtype=np.float32)
        kpts = np.zeros((1, 17, 3), dtype=np.float32)
        kpts[..., 0] = 0.25
        kpts[..., 1] = 0.5
        kpts[..., 2] = 2.0

        out_img, target = ECPoseValTransform(
            17, imagenet_norm=False, to_rgb=False
        )(img, bboxes, cls, kpts, (64, 64))

        assert out_img.shape == (3, 64, 64)
        assert target[0, 1:5] == pytest.approx([32.0, 32.0, 12.8, 25.6])
        assert target[0, 5:8] == pytest.approx([16.0, 32.0, 2.0])


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


class TestPoseTrainingStep:
    """One forward+loss+backward step exercises the DETRPose training path
    (deep supervision + contrastive denoising)."""

    K = 17

    def _make_targets(self):
        def tgt(n):
            xy = torch.rand(n, self.K, 2).reshape(n, 2 * self.K)
            vis = torch.ones(n, self.K)
            return {
                "labels": torch.zeros(n, dtype=torch.long),
                "boxes": torch.tensor([[0.2, 0.2, 0.7, 0.8]]).repeat(n, 1),
                "keypoints": torch.cat([xy, vis], dim=1),  # (n, 2K + K)
                "area": torch.full((n,), 0.3),
            }

        # one populated image + one empty (exercises the no-match branch)
        return [tgt(2), {
            "labels": torch.zeros(0, dtype=torch.long),
            "boxes": torch.zeros(0, 4),
            "keypoints": torch.zeros(0, 3 * self.K),
            "area": torch.zeros(0),
        }]

    def test_pose_train_forward_exposes_aux_interm_pre_and_dn(self):
        torch.manual_seed(0)
        model = LibreECPoseModel(config="s", eval_spatial_size=(128, 128))
        model.train()
        targets = self._make_targets()
        out = model(torch.randn(2, 3, 128, 128), targets=targets)
        # DETRPose-faithful deep supervision + denoising structure
        assert "aux_outputs" in out
        assert "aux_interm_outputs" in out
        assert "aux_pre_outputs" in out
        assert "dn_aux_outputs" in out and "dn_meta" in out
        assert out["pred_logits"].shape == (2, 60, 2)
        # training keypoints are flattened (B, Q, 2K)
        assert out["pred_keypoints"].shape == (2, 60, 2 * self.K)

    def test_pose_loss_backward_reaches_decoder_and_dn_embeddings(self):
        from libreyolo.models.ec.pose_loss import ECPoseCriterion, PoseHungarianMatcher

        torch.manual_seed(0)
        model = LibreECPoseModel(config="s", eval_spatial_size=(128, 128))
        model.train()
        targets = self._make_targets()
        out = model(torch.randn(2, 3, 128, 128), targets=targets)
        crit = ECPoseCriterion(
            matcher=PoseHungarianMatcher(num_keypoints=self.K), num_keypoints=self.K, num_classes=2
        )
        losses = crit(out, targets)
        assert {"loss_vfl", "loss_keypoints", "loss_oks"} <= set(losses)
        # denoising losses are present
        assert any(k.endswith("_dn_0") for k in losses)
        total = sum(losses.values())
        assert torch.isfinite(total)
        total.backward()
        dec_grad = sum(
            1 for n, p in model.named_parameters()
            if "decoder" in n and p.grad is not None and p.grad.abs().sum() > 0
        )
        assert dec_grad > 0, "pose decoder received no gradient"
        # Denoising must actually train the label/pose embeddings.
        for name in ("decoder.label_enc.weight", "decoder.pose_enc.weight"):
            p = dict(model.named_parameters())[name]
            assert p.grad is not None and p.grad.abs().sum() > 0, f"{name} got no gradient (DN not wired)"

    def test_pose_custom_oks_sigmas_reach_matcher_and_criterion(self):
        from libreyolo.models.ec.pose_loss import ECPoseCriterion, PoseHungarianMatcher

        sigmas = [0.01 + i * 0.001 for i in range(self.K)]
        matcher = PoseHungarianMatcher(num_keypoints=self.K, sigmas=sigmas)
        criterion = ECPoseCriterion(
            matcher=matcher,
            num_keypoints=self.K,
            num_classes=2,
            sigmas=sigmas,
        )
        assert matcher.sigmas.tolist() == pytest.approx(sigmas)
        assert criterion.oks.sigmas.tolist() == pytest.approx(sigmas)

    def test_pose_default_oks_sigmas_keep_ec_tables(self):
        from libreyolo.models.ec.pose_loss import default_oks_sigmas

        assert default_oks_sigmas(14) == pytest.approx(
            [0.079, 0.079, 0.072, 0.072, 0.062, 0.062, 0.107, 0.107,
             0.087, 0.087, 0.089, 0.089, 0.079, 0.079]
        )

    def test_pose_custom_oks_sigmas_reach_denoising(self):
        from libreyolo.models.ec.pose_denoising import prepare_for_cdn

        K = self.K
        target = {
            "labels": torch.zeros(1, dtype=torch.long),
            "boxes": torch.tensor([[0.2, 0.2, 0.7, 0.8]]),
            "keypoints": torch.cat(
                [torch.full((1, 2 * K), 0.5), torch.ones(1, K)], dim=1
            ),
            "area": torch.full((1,), 0.3),
        }
        label_enc = torch.nn.Embedding(81, 8)
        pose_enc = torch.nn.Embedding(K, 8)
        kwargs = dict(
            dn_args=([target], 20, 0.5),
            training=True,
            num_queries=60,
            num_classes=80,
            num_keypoints=K,
            hidden_dim=8,
            label_enc=label_enc,
            pose_enc=pose_enc,
            img_dim=(128, 128),
            device=torch.device("cpu"),
        )

        torch.manual_seed(123)
        default_pose = prepare_for_cdn(**kwargs)[1]
        torch.manual_seed(123)
        custom_pose = prepare_for_cdn(**kwargs, sigmas=[0.2] * K)[1]

        assert not torch.allclose(default_pose, custom_pose)
