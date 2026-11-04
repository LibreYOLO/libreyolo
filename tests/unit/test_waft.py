"""LibreWAFT: checkpoint recognition, filenames and conversion metadata."""

from __future__ import annotations

import gc
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import torch

from libreyolo import LibreWAFT, LibreYOLO
from libreyolo.models.depth_anything._vendor.dpt import DepthAnythingV2
from libreyolo.models.waft._vendor.waft import WAFTA1, WAFTA2
from libreyolo.utils.serialization import validate_checkpoint_metadata

_WEIGHTS = Path(__file__).resolve().parents[2] / "weights"
sys.path.insert(0, str(_WEIGHTS))
import convert_waft_weights  # noqa: E402

pytestmark = pytest.mark.unit

_NETS = {
    "t": lambda: WAFTA2("dinov3"),
    "s": lambda: WAFTA2("dav2"),
    "m": lambda: WAFTA1(),
    "l": lambda: WAFTA2("twins"),
}
# Tensor counts of the released upstream checkpoints (a2 dinov3/dav2/twins
# zero-shot.pth, a1 tar-c-t.pth); the vendored graph must match them exactly.
_UPSTREAM_KEYS = {"t": 521, "s": 508, "m": 604, "l": 699}


def _meta_state_dict(build) -> dict:
    """Keys and shapes of a net without allocating its weights.

    The four WAFT graphs hold about 314 M parameters; building them for real
    inside the fast unit gate costs 1.25 GB. Recognition and size detection
    only look at key names, so the nets are constructed on the meta device.
    The vendored DINOv2 derives its drop-path schedule from ``torch.linspace``
    with ``.item()``, which meta tensors refuse, so that one factory call is
    pinned to the CPU while the net is built.
    """
    real_linspace = torch.linspace

    def cpu_linspace(*args, **kwargs):
        kwargs.setdefault("device", "cpu")
        return real_linspace(*args, **kwargs)

    with torch.device("meta"), mock.patch("torch.linspace", cpu_linspace):
        return build().state_dict()


@pytest.fixture(scope="module")
def state_dicts():
    return {size: _meta_state_dict(build) for size, build in _NETS.items()}


def _small_model() -> LibreWAFT:
    """A real size-s model (57 M parameters, 230 MB); built per test and freed."""
    torch.manual_seed(0)
    return LibreWAFT(None, size="s", device="cpu")


def _fingerprint(state_dict: dict) -> dict:
    """Shape, dtype and float64 sum per tensor, plus full copies of the small ones.

    Lets the round-trip test compare the loaded weights against the source
    after the source model has been freed, so only one 230 MB model is alive.
    """
    out = {}
    for key, value in state_dict.items():
        small = value.clone() if value.numel() <= 1024 else None
        out[key] = (tuple(value.shape), value.dtype, float(value.double().sum()), small)
    return out


@pytest.mark.parametrize("size", sorted(_NETS))
def test_can_load_and_detect_size(state_dicts, size):
    sd = state_dicts[size]
    assert len(sd) == _UPSTREAM_KEYS[size]
    assert LibreWAFT.can_load(sd)
    assert LibreWAFT.detect_size(sd) == size
    assert LibreWAFT.detect_nb_classes(sd) == 1


def test_rejects_foreign_state_dicts():
    dav2 = _meta_state_dict(lambda: DepthAnythingV2("vits", 64, [48, 96, 192, 384]))
    assert not LibreWAFT.can_load(dav2)
    assert not LibreWAFT.can_load({"warp_linear.weight": torch.zeros(1)})
    assert not LibreWAFT.can_load({})


@pytest.mark.parametrize(
    ("filename", "size", "variant"),
    [
        ("LibreWAFTt-flow-dinov3.pt", "t", "dinov3"),
        ("LibreWAFTs-flow-dav2.pt", "s", "dav2"),
        ("LibreWAFTm-flow.pt", "m", None),
        ("LibreWAFTl-flow-twins.pt", "l", "twins"),
    ],
)
def test_canonical_filenames(filename, size, variant):
    assert LibreWAFT.detect_size_from_filename(filename) == size
    assert LibreWAFT.detect_task_from_filename(filename) == "flow"
    assert LibreWAFT.detect_variant_from_filename(filename) == variant
    stem = filename[:-3]
    assert LibreWAFT.get_download_url(filename) == (
        f"https://huggingface.co/LibreYOLO/{stem}/resolve/main/{filename}"
    )
    assert convert_waft_weights.canonical_filename(size) == filename


@pytest.mark.parametrize(
    "filename",
    ["LibreWAFTm-flow-dav2.pt", "LibreWAFTs-flow.pt", "LibreWAFTt-flow-twins.pt", "LibreWAFTl-flow.pt"],
)
def test_unpublished_combinations_raise(filename):
    with pytest.raises(FileNotFoundError, match="published only as"):
        LibreWAFT.get_download_url(filename)


def test_conversion_metadata_and_load(state_dicts, tmp_path):
    source_model = _small_model()
    source = source_model.model.state_dict()
    assert source.keys() == state_dicts["s"].keys()
    ckpt = convert_waft_weights.convert_state_dict(source)
    assert validate_checkpoint_metadata(ckpt, strict=True) == []
    assert (ckpt["model_family"], ckpt["size"], ckpt["task"]) == ("waft", "s", "flow")
    assert ckpt["nc"] == 1 and ckpt["names"] == {0: "flow"}
    assert ckpt["variant"] == "dav2"
    assert ckpt["libreyolo_version"]

    path = tmp_path / "LibreWAFTs-flow-dav2.pt"
    torch.save(ckpt, path)
    expected = _fingerprint(source)
    del source_model, source, ckpt
    gc.collect()

    model = LibreYOLO(str(path), device="cpu")
    path.unlink()
    assert isinstance(model, LibreWAFT)
    assert (model.size, model.task, model.names) == ("s", "flow", {0: "flow"})
    loaded = model.model.state_dict()
    assert loaded.keys() == expected.keys()
    for key, value in loaded.items():
        shape, dtype, total, small = expected[key]
        assert (tuple(value.shape), value.dtype) == (shape, dtype), key
        assert float(value.double().sum()) == total, key
        if small is not None:
            assert torch.equal(value, small), key


def test_conversion_rejects_size_mismatch(state_dicts):
    with pytest.raises(ValueError, match="does not match"):
        convert_waft_weights.convert_state_dict(state_dicts["m"], size="l")
    with pytest.raises(ValueError, match="WAFT checkpoint"):
        convert_waft_weights.convert_state_dict({"foo": torch.zeros(1)})


def test_pair_predict_shapes_and_single_image_error():
    model = _small_model()
    rng = np.random.default_rng(0)
    frame = (rng.random((70, 90, 3)) * 255).astype(np.uint8)
    results = model([frame, np.roll(frame, 2, axis=1), np.roll(frame, 4, axis=1)])
    assert len(results) == 2
    for r in results:
        assert r.boxes is None
        assert tuple(r.flow.data.shape) == (70, 90, 2)
    with pytest.raises(ValueError):
        model(frame)


def test_imgsz_caps_the_long_side():
    """An explicit imgsz downscales the frames; the field returns to the original canvas."""
    model = _small_model()
    frame = np.zeros((70, 90, 3), dtype=np.uint8)
    tensor, _, size, ratio = model._preprocess(frame, input_size=48)
    assert tuple(tensor.shape) == (1, 3, 37, 48) and size == (90, 70)
    assert ratio == (90 / 48, 70 / 37)
    native, _, _, ratio_native = model._preprocess(frame)
    assert tuple(native.shape) == (1, 3, 70, 90) and ratio_native == (1.0, 1.0)
    seen = []
    forward_flow = model._forward_flow

    def record(window):
        seen.append(tuple(window.shape))
        return forward_flow(window)

    model._forward_flow = record
    rng = np.random.default_rng(0)
    frame = (rng.random((70, 90, 3)) * 255).astype(np.uint8)
    results = model([frame, np.roll(frame, 2, axis=1)], imgsz=64)
    assert seen == [(2, 3, 50, 64)]
    assert tuple(results[0].flow.data.shape) == (70, 90, 2)
    with pytest.raises(ValueError, match="single int"):
        model([frame, frame], imgsz=(64, 64))


def test_train_is_rejected():
    with pytest.raises(NotImplementedError):
        _small_model().train(data="x.yaml")
