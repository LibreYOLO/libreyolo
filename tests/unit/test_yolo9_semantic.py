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


def _make_semantic_yaml(root, n_images=8, size=64, split_names=("train", "val")):
    """Trivially-separable synthetic set: left half class 0, right half class 1."""
    import yaml as _yaml

    for split in split_names:
        for i in range(n_images):
            img_dir = root / "images" / split
            img_dir.mkdir(parents=True, exist_ok=True)
            arr = np.zeros((size, size, 3), dtype=np.uint8)
            arr[:, : size // 2] = (200, 40, 40)
            arr[:, size // 2 :] = (40, 40, 200)
            Image.fromarray(arr).save(img_dir / f"img{i}.jpg")
            mask = np.zeros((size, size), dtype=np.uint8)
            mask[:, size // 2 :] = 1
            mask_dir = root / "masks" / split
            mask_dir.mkdir(parents=True, exist_ok=True)
            Image.fromarray(mask, mode="L").save(mask_dir / f"img{i}.png")
    yaml_path = root / "data.yaml"
    yaml_path.write_text(
        _yaml.safe_dump(
            {
                "path": str(root),
                "train": "images/train",
                "val": "images/val",
                "masks_dir": "masks",
                "nc": 2,
                "names": {0: "left", 1: "right"},
            }
        )
    )
    return yaml_path


def test_yolo9_semantic_train_smoke(tmp_path):
    """A few epochs on the synthetic set run end-to-end and reduce loss."""
    yaml_path = _make_semantic_yaml(tmp_path)
    m = LibreYOLO9(None, size="t", task="semantic", nb_classes=2, device="cpu")

    res = m.train(
        data=str(yaml_path),
        epochs=3,
        batch=4,
        imgsz=64,
        optimizer="adamw",
        lr0=2e-3,
        workers=0,
        eval_interval=1,
        project=str(tmp_path / "runs"),
        name="sem_smoke",
        exist_ok=True,
        amp=False,
        ema=False,
        warmup_epochs=0,
    )

    losses = res["epoch_losses"]
    assert len(losses) == 3
    assert all(np.isfinite(losses))
    assert losses[-1] < losses[0]
    assert (
        res["epoch_metrics"][-1]["val_metrics"].get("metrics/mIoU") is not None
    )

    ckpt = res.get("last_checkpoint") or res.get("best_checkpoint")
    assert ckpt is not None
    reloaded = LibreYOLO9(ckpt, size="t", device="cpu")  # no task= passed
    assert reloaded.task == "semantic"
    assert reloaded.nb_classes == 2


def test_yolo9_semantic_polygon_fallback_appends_background(tmp_path):
    """Training from polygon labels grows the head by one background class."""
    import yaml as _yaml

    for split in ("train", "val"):
        img_dir = tmp_path / "images" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        label_dir = tmp_path / "labels" / split
        label_dir.mkdir(parents=True, exist_ok=True)
        for i in range(4):
            arr = np.full((64, 64, 3), 120, dtype=np.uint8)
            arr[:, :32] = (220, 60, 60)
            Image.fromarray(arr).save(img_dir / f"img{i}.jpg")
            # Class-0 polygon over the left half.
            (label_dir / f"img{i}.txt").write_text(
                "0 0.0 0.0 0.5 0.0 0.5 1.0 0.0 1.0\n"
            )
    yaml_path = tmp_path / "data.yaml"
    yaml_path.write_text(
        _yaml.safe_dump(
            {
                "path": str(tmp_path),
                "train": "images/train",
                "val": "images/val",
                "nc": 1,
                "names": {0: "object"},
            }
        )
    )

    m = LibreYOLO9(None, size="t", task="semantic", nb_classes=1, device="cpu")
    res = m.train(
        data=str(yaml_path),
        epochs=1,
        batch=4,
        imgsz=64,
        optimizer="adamw",
        lr0=1e-3,
        workers=0,
        eval_interval=0,
        project=str(tmp_path / "runs"),
        name="sem_poly",
        exist_ok=True,
        amp=False,
        ema=False,
        warmup_epochs=0,
    )

    assert m.nb_classes == 2  # object + background
    assert m.names[1] == "background"
    assert np.isfinite(res["epoch_losses"][0])
