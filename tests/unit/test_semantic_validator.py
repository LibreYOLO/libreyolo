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


class _ContentAwareSemanticModel(_StubSemanticModel):
    """Predicts per-pixel class from image intensity, not tensor shape.

    ``_StubSemanticModel._forward`` reads off tensor shape alone, so it
    can't tell a flipped batch from an unflipped one — flip-TTA would be a
    no-op either way. This stub actually looks at pixel intensity so
    flipping the input tensor flips the raw prediction too, letting the
    flip-back-and-average round trip in ``_run_validation_augmented`` mean
    something.
    """

    def _forward(self, images: torch.Tensor) -> torch.Tensor:
        intensity = images.mean(dim=1, keepdim=True)
        dark = intensity < 0.5
        logits = torch.zeros((images.shape[0], self.nb_classes, images.shape[2], images.shape[3]))
        logits[:, 0:1][dark] = 10.0
        logits[:, 1:2][~dark] = 10.0
        return logits


def _write_two_tone_image(path: Path, width: int, height: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (width, height), color=(0, 0, 0))
    pixels = img.load()
    for y in range(height):
        for x in range(width // 2, width):
            pixels[x, y] = (255, 255, 255)
    img.save(path)


def test_flip_content_leaves_padding_in_place_and_mirrors_content_only():
    """torch.flip on the whole padded canvas would move letterbox padding to
    the wrong side for the flipped view; _flip_content must mirror only the
    real (unpadded) content and leave padding exactly where SemanticDataset
    put it (bottom/right — see valid_content_hw)."""
    from types import SimpleNamespace

    # Content occupies columns [0:2]; columns [2:4] are letterbox padding
    # (marker value 99, distinct from any real content value used here).
    tensor = torch.tensor([[[[1.0, 2.0, 99.0, 99.0]]]])
    img_info = [{"orig_shape": (1, 2), "ratio": 1.0}]
    fake_self = SimpleNamespace(_resize_mode="letterbox")

    flipped = SemanticValidator._flip_content(fake_self, tensor, img_info)

    assert flipped.tolist() == [[[[2.0, 1.0, 99.0, 99.0]]]]


def test_flip_content_stretch_mode_flips_whole_canvas():
    """stretch resize mode has no padding, so the whole canvas is content."""
    from types import SimpleNamespace

    tensor = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]])
    fake_self = SimpleNamespace(_resize_mode="stretch")

    flipped = SemanticValidator._flip_content(fake_self, tensor, img_info=[{}])

    assert flipped.tolist() == [[[[4.0, 3.0, 2.0, 1.0]]]]


def test_flip_content_rescale_crop_mode_flips_whole_canvas():
    """rescale_crop validates by direct resize (SemanticDataset._resize), so the
    canvas is all content and flip TTA is the stretch geometry."""
    from types import SimpleNamespace

    tensor = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]])
    fake_self = SimpleNamespace(_resize_mode="rescale_crop")

    flipped = SemanticValidator._flip_content(fake_self, tensor, img_info=[{}])

    assert flipped.tolist() == [[[[4.0, 3.0, 2.0, 1.0]]]]


def test_valid_content_hw_matches_real_dataset_letterbox_boundary(tmp_path):
    """valid_content_hw must locate exactly the boundary SemanticDataset
    itself pads at, on a real non-square image — the historical bug only
    manifests when padding is asymmetric (a non-square source image), since
    a square imgsz x imgsz image has no padding to misplace."""
    from libreyolo.data.semantic_dataset import (
        _PAD_COLOR,
        SemanticDataset,
        resolve_semantic_data,
        valid_content_hw,
    )

    orig_w, orig_h = 16, IMGSZ  # portrait: letterboxes into IMGSZ x IMGSZ
    # with padding on the width axis only (the axis flip-TTA operates on).
    _write_image(tmp_path / "images" / "val" / "img0.jpg", orig_w, orig_h)
    mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
    _write_mask(tmp_path / "masks" / "val" / "img0.png", mask)
    yaml_path = tmp_path / "data.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {tmp_path.as_posix()}",
                "val: images/val",
                "masks_dir: masks",
                "nc: 1",
                "names:",
                "  0: bg",
                "",
            ]
        )
    )
    data_config = resolve_semantic_data(str(yaml_path))
    dataset = SemanticDataset(data_config, split="val", imgsz=IMGSZ, augment=False)
    img_tensor, _, img_info, _ = dataset[0]

    new_h, new_w = valid_content_hw(
        img_info["orig_shape"], img_info["ratio"], (IMGSZ, IMGSZ)
    )

    assert new_h == IMGSZ
    assert new_w == orig_w
    img_uint8 = (img_tensor * 255.0).round()
    # Last real-content column must not be the pad color; first padding
    # column must be exactly the pad color.
    assert not torch.allclose(img_uint8[:, 0, new_w - 1], torch.full((3,), float(_PAD_COLOR)))
    assert torch.allclose(img_uint8[:, 0, new_w], torch.full((3,), float(_PAD_COLOR)))


def test_augmented_validation_matches_non_augmented_on_content_aware_model(tmp_path):
    for i in range(2):
        _write_two_tone_image(tmp_path / "images" / "val" / f"img{i}.jpg", IMGSZ, IMGSZ)
        mask = np.zeros((IMGSZ, IMGSZ), dtype=np.uint8)
        mask[:, IMGSZ // 2 :] = 1
        _write_mask(tmp_path / "masks" / "val" / f"img{i}.png", mask)
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
        batch_size=2,
        device="cpu",
        num_workers=0,
        verbose=False,
        save_dir=str(tmp_path / "runs"),
        augment=True,
    )
    metrics = SemanticValidator(_ContentAwareSemanticModel(), config).run()

    assert metrics["metrics/mIoU"] == pytest.approx(1.0)
    assert metrics["metrics/pixel_accuracy"] == pytest.approx(1.0)


def test_augmented_validation_keeps_padding_on_the_right_for_nonsquare_input(tmp_path):
    """End-to-end guard for the letterbox-padding bug on a NON-SQUARE image.

    A pixelwise stub cannot catch a naive whole-canvas flip end-to-end (flip
    then flip-back cancels for any permutation-equivariant model), and a square
    image has no padding to misplace — so neither existing test would notice if
    ``_flip_content`` regressed to ``tensor.flip(-1)``. Instead, capture the
    tensors the model is actually fed and assert the flipped view still has its
    letterbox padding on the RIGHT, where every non-TTA code path puts it. A
    whole-canvas flip would relocate it to the left and fail here.
    """
    from libreyolo.data.semantic_dataset import _PAD_COLOR

    content_w = IMGSZ // 2  # portrait -> pads on the width axis, which flip touches

    class _RecordingModel(_ContentAwareSemanticModel):
        def __init__(self):
            super().__init__()
            self.seen_inputs = []

        def _forward(self, images: torch.Tensor) -> torch.Tensor:
            self.seen_inputs.append(images.clone())
            return super()._forward(images)

    for i in range(2):
        img = Image.new("RGB", (content_w, IMGSZ), color=(30, 60, 90))  # dark -> class 0
        pixels = img.load()
        for y in range(IMGSZ):
            for x in range(content_w // 2, content_w):
                pixels[x, y] = (255, 255, 255)  # light -> class 1
        path = tmp_path / "images" / "val" / f"img{i}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        img.save(path)

        mask = np.zeros((IMGSZ, content_w), dtype=np.uint8)
        mask[:, content_w // 2 :] = 1
        _write_mask(tmp_path / "masks" / "val" / f"img{i}.png", mask)

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
        batch_size=2,
        device="cpu",
        num_workers=0,
        verbose=False,
        save_dir=str(tmp_path / "runs"),
        augment=True,
    )
    model = _RecordingModel()
    metrics = SemanticValidator(model, config).run()

    # BaseValidator._warmup_model fires dummy forwards first; the real batch is
    # the last two calls (one original view + one flipped view).
    assert len(model.seen_inputs) >= 2
    original, flipped = model.seen_inputs[-2:]
    pad_value = _PAD_COLOR / 255.0

    # Sanity: the un-augmented view is letterboxed with padding on the right.
    assert torch.allclose(
        original[:, :, :, content_w:], torch.full_like(original[:, :, :, content_w:], pad_value), atol=1e-2
    )

    # The contract: the flipped view's padding is STILL on the right. A naive
    # tensor.flip(-1) would put pad_value in columns [0:content_w) instead.
    assert torch.allclose(
        flipped[:, :, :, content_w:], torch.full_like(flipped[:, :, :, content_w:], pad_value), atol=1e-2
    )
    # ...and its real content is the mirror of the original's real content.
    assert torch.allclose(
        flipped[:, :, :, :content_w], original[:, :, :, :content_w].flip(-1)
    )

    assert metrics["metrics/mIoU"] == pytest.approx(1.0)


def test_flip_content_rejects_unknown_resize_mode():
    """Modes whose padding this window math does not model must fail loudly
    (EoMT's "split", say) rather than silently mirror the wrong region."""
    from types import SimpleNamespace

    fake_self = SimpleNamespace(_resize_mode="split")
    with pytest.raises(ValueError, match="split"):
        SemanticValidator._flip_content(
            fake_self, torch.zeros(1, 1, 1, 4), [{"orig_shape": (1, 2), "ratio": 1.0}]
        )


def test_extract_logits_upcasts_half_precision_at_target_resolution():
    """Under half=True a model whose logits already sit at target_hw must not
    carry fp16 into the softmax merge."""
    from types import SimpleNamespace

    half_logits = torch.zeros(1, 2, 4, 4, dtype=torch.float16)
    out = SemanticValidator._extract_logits(SimpleNamespace(), half_logits, (4, 4))

    assert out.dtype == torch.float32


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


def test_imgsz_divisor_mismatch_raises(tmp_path):
    yaml_path = _make_dataset_yaml(tmp_path)
    config = ValidationConfig(
        data=str(yaml_path),
        imgsz=IMGSZ,  # 32, not divisible by 14
        device="cpu",
        num_workers=0,
        verbose=False,
        save_dir=str(tmp_path / "runs"),
    )
    model = _StubSemanticModel()
    model.semantic_imgsz_divisor = 14

    with pytest.raises(ValueError, match="divisible by 14"):
        SemanticValidator(model, config).run()
