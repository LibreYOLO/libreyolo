"""Hermetic tests for the MiDaS relative-depth family."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
import yaml
from PIL import Image

from libreyolo.models.autoconvert import autoconvert_upstream_checkpoint
from libreyolo.models.midas.convert import (
    UPSTREAM_URLS,
    verify_and_wrap_download,
    wrap_upstream_state_dict,
)
from libreyolo.models.midas.model import LibreMiDaS
from libreyolo.models.midas.utils import _resize_shape, preprocess_numpy
from libreyolo.utils.serialization import validate_checkpoint_metadata

pytestmark = [pytest.mark.unit, pytest.mark.midas]


@pytest.fixture(scope="module")
def midas_small():
    pytest.importorskip("timm")
    return LibreMiDaS(model_path=None, size="s", task="depth", device="cpu")


def _midas_signature(size: str) -> dict[str, torch.Tensor]:
    common = {
        "scratch.refinenet1.resConfUnit1.conv1.weight": torch.zeros(1, 1, 1, 1),
        "scratch.output_conv.4.weight": torch.zeros(1, 1, 1, 1),
    }
    if size == "s":
        return {
            **common,
            "pretrained.layer1.0.weight": torch.zeros(32, 3, 3, 3),
            "pretrained.layer1.3.0.conv_dw.weight": torch.zeros(32, 1, 3, 3),
        }
    if size == "l":
        return {
            **common,
            "pretrained.model.cls_token": torch.zeros(1, 1, 1024),
        }
    raise ValueError(size)


def _sibling_depth_signatures() -> list[tuple[type, dict[str, torch.Tensor]]]:
    from libreyolo.models.depth_anything.model import LibreDepthAnythingV2
    from libreyolo.models.depth_anything3.model import LibreDepthAnything3
    from libreyolo.models.zipdepth.model import LibreZipDepth

    return [
        (
            LibreDepthAnythingV2,
            {
                "pretrained.cls_token": torch.zeros(1, 1, 384),
                "depth_head.scratch.output_conv2.0.weight": torch.zeros(1),
            },
        ),
        (
            LibreDepthAnything3,
            {
                "backbone.pretrained.cls_token": torch.zeros(1, 1, 1024),
                "head.scratch.output_conv2.0.weight": torch.zeros(1),
            },
        ),
        (
            LibreZipDepth,
            {
                "encoder.stem_half.conv.weight": torch.zeros(24, 3, 3, 3),
                "decoder.convex_up.weight": torch.zeros(1),
            },
        ),
    ]


def test_family_metadata_and_registration():
    from libreyolo.models.base import BaseModel

    assert LibreMiDaS.FAMILY == "midas"
    assert LibreMiDaS.FILENAME_PREFIX == "LibreMiDaS"
    assert LibreMiDaS.INPUT_SIZES == {"s": 256, "l": 384}
    assert LibreMiDaS.SUPPORTED_TASKS == ("depth",)
    assert LibreMiDaS.DEFAULT_TASK == "depth"
    assert LibreMiDaS.REQUIRE_TASK_SUFFIX is True
    assert LibreMiDaS in BaseModel._registry


@pytest.mark.parametrize("size", ["s", "l"])
def test_can_load_and_detect_size(size: str):
    state = _midas_signature(size)
    assert LibreMiDaS.can_load(state)
    assert LibreMiDaS.detect_size(state) == size
    assert LibreMiDaS.detect_nb_classes(state) == 1


def test_can_load_is_bidirectionally_exclusive_with_depth_siblings():
    for sibling, sibling_state in _sibling_depth_signatures():
        assert sibling.can_load(sibling_state)
        assert not LibreMiDaS.can_load(sibling_state)
        for size in ("s", "l"):
            midas_state = _midas_signature(size)
            assert LibreMiDaS.can_load(midas_state)
            assert not sibling.can_load(midas_state)


def test_filename_and_download_routing():
    assert LibreMiDaS.detect_size_from_filename("LibreMiDaSs-depth.pt") == "s"
    assert LibreMiDaS.detect_size_from_filename("LibreMiDaSl-depth.pt") == "l"
    assert LibreMiDaS.detect_task_from_filename("LibreMiDaSl-depth.pt") == "depth"
    assert LibreMiDaS.detect_size_from_filename("LibreMiDaSs.pt") is None
    # Mirrored sizes resolve to the LibreYOLO org; anything not mirrored still
    # falls back to the official upstream asset.
    for size in ("s", "l"):
        url = LibreMiDaS.get_download_url(f"LibreMiDaS{size}-depth.pt")
        assert url == (
            f"https://huggingface.co/LibreYOLO/LibreMiDaS{size}-depth"
            f"/resolve/main/LibreMiDaS{size}-depth.pt"
        )
        assert url != UPSTREAM_URLS[size]
    assert LibreMiDaS.get_download_url("LibreMiDaSs.pt") is None


@pytest.mark.parametrize(
    "size,expected",
    [("s", (256, 192)), ("l", (512, 384))],
)
def test_resize_geometry_matches_official_rules(size: str, expected: tuple[int, int]):
    assert _resize_shape(640, 480, LibreMiDaS.INPUT_SIZES[size], size) == expected


@pytest.mark.parametrize("size", ["s", "l"])
@pytest.mark.parametrize("width,height", [(10_000, 1), (1, 10_000)])
def test_resize_geometry_keeps_extreme_aspect_ratio_sides_nonzero(
    size: str,
    width: int,
    height: int,
):
    new_width, new_height = _resize_shape(
        width,
        height,
        LibreMiDaS.INPUT_SIZES[size],
        size,
    )
    assert min(new_width, new_height) >= LibreMiDaS.depth_imgsz_divisor

    image = np.zeros((height, width, 3), dtype=np.uint8)
    chw, _ = preprocess_numpy(image, LibreMiDaS.INPUT_SIZES[size], size)
    assert min(chw.shape[1:]) >= LibreMiDaS.depth_imgsz_divisor


@pytest.mark.parametrize("size", ["s", "l"])
def test_preprocess_is_rgb_float_and_multiple_of_32(size: str):
    image = np.random.default_rng(7).integers(
        0, 256, size=(321, 517, 3), dtype=np.uint8
    )
    chw, ratio = preprocess_numpy(image, LibreMiDaS.INPUT_SIZES[size], size)

    assert chw.dtype == np.float32
    assert chw.shape[0] == 3
    assert chw.shape[1] % 32 == 0
    assert chw.shape[2] % 32 == 0
    assert np.isfinite(chw).all()
    assert ratio == 1.0


@pytest.mark.parametrize(
    ("family", "size", "canvas"),
    [("midas", "s", 64), ("midas", "l", 96), ("depth_anything", "s", 70)],
)
def test_exported_backend_resizes_square_images_like_native(family, size, canvas):
    # A square image needs no aspect approximation, so the exported input must
    # be the native one, including the bicubic float resize.
    from libreyolo.backends.base import BaseBackend
    from libreyolo.models.depth_anything.utils import (
        preprocess_numpy as depth_anything_preprocess,
    )

    class StubBackend(BaseBackend):
        def _run_inference(self, blob):
            return []

    backend = StubBackend(
        model_path="fixture.onnx",
        nb_classes=1,
        device="cpu",
        imgsz=canvas,
        model_family=family,
        model_size=size,
        names={0: "depth"},
        task="depth",
        supported_tasks=("depth",),
        default_task="depth",
    )
    image = np.random.default_rng(11).integers(0, 256, (150, 150, 3), dtype=np.uint8)
    tensor, _, original_size, _ = backend._preprocess(image, canvas, "rgb")
    if family == "midas":
        native, _ = preprocess_numpy(image, canvas, size)
    else:
        native, _ = depth_anything_preprocess(image, canvas)

    assert original_size == (150, 150)
    assert tuple(tensor.shape) == (1, *native.shape)
    np.testing.assert_allclose(tensor[0].numpy(), native, atol=1e-6)


def test_upstream_wrap_has_strict_depth_metadata():
    checkpoint = wrap_upstream_state_dict(_midas_signature("s"), "s")
    assert validate_checkpoint_metadata(checkpoint, strict=True) == []
    assert checkpoint["model_family"] == "midas"
    assert checkpoint["size"] == "s"
    assert checkpoint["task"] == "depth"
    assert checkpoint["nc"] == 1
    assert checkpoint["names"] == {0: "depth"}
    assert checkpoint["imgsz"] == 256


def test_upstream_wrap_rejects_contradictory_size():
    with pytest.raises(ValueError, match="expected 'l'"):
        wrap_upstream_state_dict(_midas_signature("s"), "l")


def test_download_verification_rejects_tampering(tmp_path: Path):
    checkpoint = tmp_path / "download.pt.part"
    checkpoint.write_bytes(b"tampered")

    with pytest.raises(RuntimeError, match="Checksum mismatch"):
        verify_and_wrap_download(str(checkpoint), UPSTREAM_URLS["s"])


def test_raw_upstream_state_autoconverts_with_depth_name(tmp_path: Path):
    source = tmp_path / "midas_v21_small_256.pt"
    torch.save(_midas_signature("s"), source)

    converted = autoconvert_upstream_checkpoint(str(source))

    assert converted is not None
    assert Path(converted).name == ("midas_v21_small_256-LibreMiDaSs-depth.pt")
    checkpoint = torch.load(converted, map_location="cpu", weights_only=True)
    assert validate_checkpoint_metadata(checkpoint, strict=True) == []
    assert checkpoint["model_family"] == "midas"
    assert checkpoint["names"] == {0: "depth"}


def test_small_forward_contract(midas_small):
    model = midas_small.model.eval()
    assert "pixel_mean" not in model.state_dict()
    assert "pixel_std" not in model.state_dict()
    with torch.inference_mode():
        output = model(torch.rand(1, 3, 64, 64))
    assert output.shape == (1, 1, 64, 64)
    assert float(output.min()) >= 0.0


def test_bound_export_preprocessor_matches_family_utility(midas_small):
    image = np.random.default_rng(12).integers(
        0, 256, size=(83, 117, 3), dtype=np.uint8
    )
    expected = preprocess_numpy(image, 64, "s")
    actual = midas_small._get_preprocess_numpy()(image, 64)
    np.testing.assert_array_equal(actual[0], expected[0])
    assert actual[1] == expected[1]


@pytest.mark.parametrize(
    "wrapped",
    [
        lambda value: value,
        lambda value: value[:, 0],
        lambda value: value[0, 0],
        lambda value: {"depth": value},
        lambda value: {"predictions": value[:, 0]},
        lambda value: (value,),
    ],
)
def test_postprocess_restores_original_size(wrapped):
    model = LibreMiDaS.__new__(LibreMiDaS)
    raw = torch.arange(24, dtype=torch.float32).reshape(1, 1, 4, 6)
    parsed = model._postprocess(
        wrapped(raw),
        conf_thres=0.25,
        iou_thres=0.45,
        original_size=(11, 7),
    )
    expected = F.interpolate(
        raw,
        size=(7, 11),
        mode="bilinear",
        align_corners=True,
    )[0, 0]
    assert set(parsed) == {"depth"}
    torch.testing.assert_close(parsed["depth"], expected)


@pytest.mark.parametrize(
    "output",
    [{}, [], torch.zeros(1, 2, 4, 4), torch.zeros(1, 1, 1, 4, 4)],
)
def test_postprocess_rejects_malformed_outputs(output):
    model = LibreMiDaS.__new__(LibreMiDaS)
    with pytest.raises(ValueError):
        model._postprocess(
            output,
            conf_thres=0.25,
            iou_thres=0.45,
            original_size=(11, 7),
        )


def test_predict_returns_relative_inverse_depth_result(midas_small):
    image = np.random.default_rng(22).integers(0, 256, size=(80, 90, 3), dtype=np.uint8)

    result = midas_small.predict(image, imgsz=64)

    assert result.boxes is None
    assert result.depth_map is not None
    assert tuple(result.depth_map.data.shape) == (80, 90)
    assert torch.isfinite(result.depth_map.data).all()
    assert result.names == {0: "depth"}


def test_onnx_auto_opset_is_17():
    from libreyolo.export.onnx import _requires_onnx_opset17

    assert _requires_onnx_opset17("midas")


@pytest.mark.parametrize("format", ["torchscript", "onnx"])
def test_small_exported_depth_parity(midas_small, tmp_path: Path, format: str):
    if format == "onnx":
        pytest.importorskip("onnx")
        pytest.importorskip("onnxruntime")
    from libreyolo import LibreYOLO

    y, x = np.mgrid[0:64, 0:64]
    image = np.stack((x * 4, y * 4, (x + y) * 2), axis=-1).astype(np.uint8)
    native = midas_small.predict(image, imgsz=64).depth_map.data.numpy()
    artifact = midas_small.export(
        format=format,
        output_path=str(tmp_path / f"midas_s.{format}"),
        imgsz=64,
        dynamic=False,
        simplify=False,
    )
    backend = LibreYOLO(artifact, device="cpu")
    actual = backend.predict(image).depth_map.data.numpy()

    mse = float(np.mean((native - actual) ** 2))
    peak = max(float(np.max(np.abs(native))), 1e-6)
    psnr = float("inf") if mse == 0 else 20.0 * np.log10(peak / np.sqrt(mse))
    assert psnr > 40.0
    assert backend.family == "midas"
    assert backend.size == "s"
    assert backend.task == "depth"
    assert backend.names == {0: "depth"}


def _make_depth_yaml(root: Path) -> Path:
    for split in ("train", "val"):
        image_dir = root / "images" / split
        depth_dir = root / "depths" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        depth_dir.mkdir(parents=True, exist_ok=True)
        for index in range(2):
            image = np.zeros((64, 64, 3), dtype=np.uint8)
            image[:, :32] = (200, 40 + index, 40)
            image[:, 32:] = (40, 40, 200 - index)
            Image.fromarray(image).save(image_dir / f"image{index}.jpg")
            depth = np.full((64, 64), 8.0, dtype=np.float32)
            depth[:, :32] = 2.0
            encoded = np.rint(depth * 256.0).astype(np.uint16)
            Image.fromarray(encoded).save(depth_dir / f"image{index}.png")
    config = root / "depth.yaml"
    config.write_text(
        yaml.safe_dump(
            {"path": str(root), "train": "images/train", "val": "images/val"}
        ),
        encoding="utf-8",
    )
    return config


def test_zero_shot_depth_validation_runs(midas_small, tmp_path: Path):
    metrics = midas_small.val(
        data=str(_make_depth_yaml(tmp_path)),
        imgsz=64,
        batch=1,
        workers=0,
        verbose=False,
    )
    assert np.isfinite(metrics["metrics/abs_rel"])
    assert np.isfinite(metrics["metrics/rmse"])
    assert "metrics/delta1" in metrics
    assert metrics["fitness"] == metrics["metrics/delta1"]


def test_bad_imgsz_and_training_fail_explicitly():
    model = LibreMiDaS.__new__(LibreMiDaS)
    with pytest.raises(ValueError, match="divisible by 32"):
        model._preprocess(np.zeros((64, 64, 3), dtype=np.uint8), input_size=100)
    with pytest.raises(NotImplementedError, match="not implemented"):
        model.train(data="depth.yaml")
