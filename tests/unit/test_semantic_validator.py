"""Unit tests for the semantic-segmentation validator."""

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image

from libreyolo.validation import SemanticValidator, ValidationConfig

pytestmark = pytest.mark.unit

IMGSZ = 32


def _write_image(path: Path, width: int, height: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (width, height), color=(30, 60, 90)).save(path)


def _write_mask(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array.astype(np.uint8), mode="L").save(path)


def _make_dataset_yaml(root: Path, n_images: int = 2) -> Path:
    """Square dataset: left half class 0, right half class 1."""
    for i in range(n_images):
        _write_image(root / "images" / "val" / f"img{i}.jpg", IMGSZ, IMGSZ)
        mask = np.zeros((IMGSZ, IMGSZ), dtype=np.uint8)
        mask[:, IMGSZ // 2 :] = 1
        _write_mask(root / "masks" / "val" / f"img{i}.png", mask)
    yaml_path = root / "data.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {root.as_posix()}",
                "val: images/val",
                "masks_dir: masks",
                "nc: 2",
                "names:",
                "  0: left",
                "  1: right",
                "",
            ]
        )
    )
    return yaml_path


class _StubSemanticModel:
    """Minimal model double satisfying the BaseValidator contract."""

    size = "t"
    nb_classes = 2
    names = {0: "left", 1: "right"}
    semantic_resize_mode = "letterbox"

    def __init__(self, prediction: str = "perfect"):
        self.model = nn.Identity()
        self._prediction = prediction

    def _get_model_name(self) -> str:
        return "stub"

    def _forward(self, images: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = images.shape
        logits = torch.zeros((batch, self.nb_classes, height, width))
        if self._prediction == "perfect":
            logits[:, 0, :, : width // 2] = 10.0
            logits[:, 1, :, width // 2 :] = 10.0
        else:  # everything predicted as class 0
            logits[:, 0] = 10.0
        return logits


def _run_validator(tmp_path, prediction: str, **overrides):
    yaml_path = _make_dataset_yaml(tmp_path)
    config = ValidationConfig(
        data=str(yaml_path),
        imgsz=IMGSZ,
        batch_size=2,
        device="cpu",
        num_workers=0,
        verbose=False,
        save_dir=str(tmp_path / "runs"),
        **overrides,
    )
    validator = SemanticValidator(_StubSemanticModel(prediction), config)
    return validator.run()


def test_perfect_predictions_score_full_miou(tmp_path):
    metrics = _run_validator(tmp_path, prediction="perfect")

    assert metrics["metrics/mIoU"] == pytest.approx(1.0)
    assert metrics["metrics/pixel_accuracy"] == pytest.approx(1.0)
    assert metrics["fitness"] == pytest.approx(1.0)


def test_single_class_collapse_scores_half(tmp_path):
    # Predicting class 0 everywhere: IoU(left)=0.5, IoU(right)=0 -> mIoU 0.25,
    # pixel accuracy 0.5.
    metrics = _run_validator(tmp_path, prediction="all_zero")

    assert metrics["metrics/mIoU"] == pytest.approx(0.25)
    assert metrics["metrics/pixel_accuracy"] == pytest.approx(0.5)


def test_class_count_mismatch_raises(tmp_path):
    yaml_path = _make_dataset_yaml(tmp_path)
    config = ValidationConfig(
        data=str(yaml_path),
        imgsz=IMGSZ,
        device="cpu",
        num_workers=0,
        verbose=False,
        save_dir=str(tmp_path / "runs"),
    )
    model = _StubSemanticModel()
    model.nb_classes = 5

    with pytest.raises(ValueError, match="matching dataset/checkpoint"):
        SemanticValidator(model, config).run()


def test_low_resolution_logits_are_upsampled(tmp_path):
    class _StrideFourModel(_StubSemanticModel):
        def _forward(self, images: torch.Tensor) -> torch.Tensor:
            batch, _, height, width = images.shape
            logits = torch.zeros((batch, self.nb_classes, height // 4, width // 4))
            logits[:, 0, :, : width // 8] = 10.0
            logits[:, 1, :, width // 8 :] = 10.0
            return logits

    yaml_path = _make_dataset_yaml(tmp_path)
    config = ValidationConfig(
        data=str(yaml_path),
        imgsz=IMGSZ,
        device="cpu",
        num_workers=0,
        verbose=False,
        save_dir=str(tmp_path / "runs"),
    )
    metrics = SemanticValidator(_StrideFourModel(), config).run()

    assert metrics["metrics/mIoU"] == pytest.approx(1.0)


def test_ignore_pixels_are_excluded(tmp_path):
    # Mask is half class 0, half ignore; a model predicting class 1 on the
    # ignored half must still score perfectly.
    _write_image(tmp_path / "images" / "val" / "a.jpg", IMGSZ, IMGSZ)
    mask = np.full((IMGSZ, IMGSZ), 255, dtype=np.uint8)
    mask[:, : IMGSZ // 2] = 0
    _write_mask(tmp_path / "masks" / "val" / "a.png", mask)
    yaml_path = tmp_path / "data.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {tmp_path.as_posix()}",
                "val: images/val",
                "masks_dir: masks",
                "nc: 2",
                "names:",
                "  0: left",
                "  1: right",
                "",
            ]
        )
    )

    config = ValidationConfig(
        data=str(yaml_path),
        imgsz=IMGSZ,
        device="cpu",
        num_workers=0,
        verbose=False,
        save_dir=str(tmp_path / "runs"),
    )
    metrics = SemanticValidator(_StubSemanticModel("perfect"), config).run()

    assert metrics["metrics/pixel_accuracy"] == pytest.approx(1.0)
    assert metrics["metrics/mIoU"] == pytest.approx(1.0)
