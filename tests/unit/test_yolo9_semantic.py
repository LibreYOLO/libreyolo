"""Unit tests for YOLO9 semantic segmentation."""

import numpy as np
import pytest
import torch
from PIL import Image

from libreyolo import LibreYOLO9
from libreyolo.models.yolo9.nn import LibreYOLO9Model, SemanticDecoder
from libreyolo.models.yolo9.utils import postprocess_semantic
from libreyolo.utils.serialization import wrap_libreyolo_checkpoint

pytestmark = [pytest.mark.unit, pytest.mark.yolo9]


def _save_rgb(path, width, height):
    Image.new("RGB", (width, height), color=(40, 80, 120)).save(path)
    return str(path)


class TestSemanticNN:
    def test_decoder_forward_shapes(self):
        model = LibreYOLO9Model(config="t", nb_classes=3, semantic=True)
        model.eval()
        x = torch.rand(1, 3, 64, 64)

        with torch.no_grad():
            logits = model(x)

        assert isinstance(model.head, SemanticDecoder)
        assert logits.shape == (1, 3, 64, 64)

    def test_training_loss_and_backward(self):
        model = LibreYOLO9Model(config="t", nb_classes=3, semantic=True)
        model.train()
        x = torch.rand(2, 3, 64, 64)
        targets = torch.randint(0, 3, (2, 64, 64))
        targets[:, :4, :] = 255  # ignore region must not break the loss

        out = model(x, targets=targets)

        assert set(out) == {"total_loss", "sem"}
        assert torch.isfinite(out["total_loss"])
        out["total_loss"].backward()
        grad = model.head.predict.weight.grad
        assert grad is not None and torch.isfinite(grad).all()

    def test_one_task_head_at_a_time(self):
        with pytest.raises(ValueError, match="one task head"):
            LibreYOLO9Model(config="t", semantic=True, segmentation=True)


class TestSemanticWrapper:
    def test_forward_and_class_rebuild(self):
        m = LibreYOLO9(None, size="t", task="semantic", nb_classes=4, device="cpu")
        assert m.task == "semantic"

        x = torch.rand(1, 3, 64, 64)
        m.model.eval()
        with torch.no_grad():
            logits = m._forward(x)
        assert logits.shape == (1, 4, 64, 64)

        m._rebuild_for_new_classes(7)
        with torch.no_grad():
            logits = m._forward(x)
        assert logits.shape == (1, 7, 64, 64)

    def test_predict_returns_semantic_mask(self, tmp_path):
        img_path = _save_rgb(tmp_path / "img.jpg", 96, 48)
        m = LibreYOLO9(None, size="t", task="semantic", nb_classes=3, device="cpu")

        result = m.predict(img_path, imgsz=64)

        assert result.boxes is None
        assert result.semantic_mask is not None
        assert tuple(result.semantic_mask.data.shape) == (48, 96)
        assert result.semantic_mask.orig_shape == (48, 96)
        rows = result.summary()
        assert all({"class", "pixel_count", "pixel_fraction"} <= set(r) for r in rows)

    def test_predict_save_draws_overlay(self, tmp_path):
        img_path = _save_rgb(tmp_path / "img.jpg", 64, 64)
        save_path = tmp_path / "out.jpg"
        m = LibreYOLO9(None, size="t", task="semantic", nb_classes=2, device="cpu")

        m.predict(img_path, save=True, output_path=str(save_path), imgsz=64)

        assert save_path.exists()

    def test_tta_rejected(self, tmp_path):
        img_path = _save_rgb(tmp_path / "img.jpg", 64, 64)
        m = LibreYOLO9(None, size="t", task="semantic", nb_classes=2, device="cpu")

        with pytest.raises(ValueError, match="semantic"):
            m.predict(img_path, augment=True)


class TestSemanticCheckpoints:
    def test_metadata_round_trip_without_task_arg(self, tmp_path):
        m = LibreYOLO9(None, size="t", task="semantic", nb_classes=3, device="cpu")
        ckpt = wrap_libreyolo_checkpoint(
            m.model.state_dict(),
            model_family="yolo9",
            size="t",
            task="semantic",
            nc=3,
            names={0: "road", 1: "sky", 2: "tree"},
            imgsz=640,
        )
        path = tmp_path / "LibreYOLO9t-sem.pt"
        torch.save(ckpt, path)

        reloaded = LibreYOLO9(str(path), size="t", device="cpu")

        assert reloaded.task == "semantic"
        assert reloaded.nb_classes == 3
        assert reloaded.names[1] == "sky"

    def test_detection_weights_rejected_as_semantic(self, tmp_path):
        detect = LibreYOLO9(None, size="t", task="detect", nb_classes=3, device="cpu")
        ckpt = wrap_libreyolo_checkpoint(
            detect.model.state_dict(),
            model_family="yolo9",
            size="t",
            task="semantic",
            nc=3,
            names={0: "a", 1: "b", 2: "c"},
            imgsz=640,
        )
        path = tmp_path / "bad-sem.pt"
        torch.save(ckpt, path)

        with pytest.raises(RuntimeError, match="head.predict"):
            LibreYOLO9(str(path), size="t", task="semantic", device="cpu")


class TestSemanticPostprocess:
    def test_rectangular_unletterbox(self):
        # 96x48 original at input 64: content occupies the top-left
        # (h=32, w=63) region of the letterboxed square.
        nc = 2
        logits = torch.zeros(1, nc, 64, 64)
        logits[:, 1, :16, :] = 10.0  # top quarter of the input -> class 1

        out = postprocess_semantic(logits, input_size=64, original_size=(96, 48))
        semantic = out["semantic"]

        assert semantic.shape == (48, 96)
        assert int(semantic[0, 0]) == 1  # top of the canvas is class 1
        assert int(semantic[40, 0]) == 0  # bottom is class 0

    def test_dict_output_and_3d_logits_accepted(self):
        logits = torch.zeros(2, 32, 32)
        out = postprocess_semantic(
            {"predictions": logits}, input_size=32, original_size=(32, 32)
        )
        assert out["semantic"].shape == (32, 32)
