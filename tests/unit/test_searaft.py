"""Unit tests for the LibreSEARAFT optical flow family (CPU, random weights)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from libreyolo import LibreYOLO
from libreyolo.models.searaft.model import LibreSEARAFT
from libreyolo.models.searaft.nn import info_to_confidence
from libreyolo.models.searaft.utils import compute_padding, preprocess_numpy
from libreyolo.utils.serialization import validate_checkpoint_metadata

pytestmark = [pytest.mark.unit, pytest.mark.searaft]


def _net_state_dict(size: str) -> dict:
    torch.manual_seed(0)
    return LibreSEARAFT(None, size=size, device="cpu").model.state_dict()


def _upstream_like(size: str) -> dict:
    """The upstream safetensors layout: shared ``downsample.1`` aliases dropped."""
    return {k: v.clone() for k, v in _net_state_dict(size).items() if ".downsample.1." not in k}


def _load_converter():
    path = Path(__file__).resolve().parents[2] / "weights" / "convert_searaft_weights.py"
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("convert_searaft_weights", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _pair(w: int = 160, h: int = 128, shift: int = 2):
    rng = np.random.default_rng(0)
    a = rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8)
    b = np.roll(a, shift, axis=1)
    return Image.fromarray(a), Image.fromarray(b)


# --------------------------------------------------------------------------
# can_load / detect_size
# --------------------------------------------------------------------------


@pytest.mark.parametrize("size,expected", [("s", "s"), ("m", "m"), ("l", "m")])
def test_can_load_and_detect_size(size, expected):
    sd = _net_state_dict(size)
    assert LibreSEARAFT.can_load(sd)
    assert LibreSEARAFT.detect_size(sd) == expected
    assert LibreSEARAFT.can_load(_upstream_like(size))


def test_can_load_rejects_raft_style_and_unrelated_dicts():
    sd = _net_state_dict("s")
    # RAFT's context encoder reads one frame (3 channels) and has no init_conv.
    raft_like = dict(sd)
    raft_like["cnet.conv1.weight"] = torch.zeros(64, 3, 7, 7)
    assert not LibreSEARAFT.can_load(raft_like)
    no_init = {k: v for k, v in sd.items() if not k.startswith("init_conv.")}
    assert not LibreSEARAFT.can_load(no_init)
    assert not LibreSEARAFT.can_load({"pretrained.cls_token": torch.zeros(1, 1, 384)})
    assert not LibreSEARAFT.can_load({})


def test_restore_shared_keys_makes_upstream_layout_load_strictly():
    sd = LibreSEARAFT.restore_shared_keys(_upstream_like("m"))
    model = LibreSEARAFT(None, size="m", device="cpu")
    model.model.load_state_dict(sd, strict=True)
    assert LibreSEARAFT.convert_upstream_state_dict({"foo": torch.zeros(1)}) is None


def test_download_url_is_one_repo_per_file():
    for size in "sml":
        name = f"LibreSEARAFT{size}-flow"
        assert LibreSEARAFT.get_download_url(f"{name}.pt") == (
            f"https://huggingface.co/LibreYOLO/{name}/resolve/main/{name}.pt"
        )


# --------------------------------------------------------------------------
# conversion
# --------------------------------------------------------------------------


@pytest.mark.parametrize("size", ["s", "m", "l"])
def test_convert_safetensors_writes_valid_flow_checkpoint(tmp_path, size):
    from safetensors.torch import save_file

    conv = _load_converter()
    src = tmp_path / "model.safetensors"
    save_file(_upstream_like("s" if size == "s" else "m"), str(src))
    out = tmp_path / f"LibreSEARAFT{size}-flow.pt"
    conv.convert_weights(str(src), str(out), size=size)

    ckpt = torch.load(out, map_location="cpu", weights_only=True)
    assert validate_checkpoint_metadata(ckpt, strict=True) == []
    assert ckpt["model_family"] == "searaft"
    assert ckpt["size"] == size
    assert ckpt["task"] == "flow"
    assert ckpt["nc"] == 1
    assert ckpt["names"] == {0: "flow"}
    assert ckpt["libreyolo_version"]

    model = LibreYOLO(str(out), device="cpu")
    assert isinstance(model, LibreSEARAFT)
    assert (model.size, model.task, model.names) == (size, "flow", {0: "flow"})
    ref = _net_state_dict("s" if size == "s" else "m")
    for key, value in model.model.state_dict().items():
        assert torch.equal(value, ref[key]), key


def test_convert_rejects_wrong_encoder_and_foreign_checkpoints(tmp_path):
    from safetensors.torch import save_file

    conv = _load_converter()
    src = tmp_path / "s.safetensors"
    save_file(_upstream_like("s"), str(src))
    with pytest.raises(ValueError, match="SEA-RAFT-M"):
        conv.convert_weights(str(src), str(tmp_path / "x.pt"), size="l")
    bad = tmp_path / "bad.pth"
    torch.save({"pretrained.cls_token": torch.zeros(1, 1, 384)}, bad)
    with pytest.raises(ValueError, match="SEA-RAFT"):
        conv.convert_weights(str(bad), str(tmp_path / "y.pt"))


# --------------------------------------------------------------------------
# preprocess / postprocess geometry
# --------------------------------------------------------------------------


def test_padding_matches_upstream_input_padder():
    assert compute_padding(498, 640) == [0, 0, 3, 3]
    assert compute_padding(141, 203) == [2, 3, 1, 2]
    assert compute_padding(128, 160) == [0, 0, 0, 0]


def test_preprocess_range_and_long_side_cap():
    img = np.full((200, 300, 3), 255, dtype=np.uint8)
    x, meta = preprocess_numpy(img, None)
    assert x.shape == (1, 3, 200, 304) and float(x.max()) == 1.0
    assert meta["work_hw"] == (200, 300)
    # frames under the cap run at native resolution
    _, meta = preprocess_numpy(img, 960)
    assert meta["work_hw"] == (200, 300)
    big = np.zeros((600, 1200, 3), dtype=np.uint8)
    x, meta = preprocess_numpy(big, 960)
    assert meta["work_hw"] == (480, 960) and x.shape[-2:] == (480, 960)
    with pytest.raises(ValueError, match="at least 128"):
        preprocess_numpy(np.zeros((64, 64, 3), dtype=np.uint8), None)


def test_postprocess_rescales_vectors_to_original_canvas():
    model = LibreSEARAFT(None, size="s", device="cpu")
    big = np.zeros((600, 1200, 3), dtype=np.uint8)
    _, meta = preprocess_numpy(big, 960)  # 0.8 downscale, no padding
    flow = torch.ones(1, 2, 480, 960)
    info = torch.zeros(1, 4, 480, 960)
    out = model._postprocess((flow, info), 0.0, 0.0, (1200, 600), ratio=meta)
    assert out["flow"].shape == (600, 1200, 2)
    assert torch.allclose(out["flow"][..., 0], torch.full((600, 1200), 1200 / 960))
    assert torch.allclose(out["flow"][..., 1], torch.full((600, 1200), 600 / 480))
    assert out["flow_confidence"].shape == (600, 1200)


def test_info_to_confidence_bounds():
    info = torch.zeros(1, 4, 8, 8)
    assert torch.allclose(info_to_confidence(info), torch.ones(1, 8, 8))
    info[:, 0] = 50.0  # all mass on the wide component
    info[:, 2] = 100.0  # log scale beyond var_max is clamped
    assert torch.allclose(info_to_confidence(info), torch.zeros(1, 8, 8), atol=1e-6)


# --------------------------------------------------------------------------
# predict through the flow runner
# --------------------------------------------------------------------------


def test_predict_pair_returns_flow_on_original_canvas():
    model = LibreSEARAFT(None, size="s", device="cpu")
    a, b = _pair(w=203, h=141)
    results = model([a, b])
    assert len(results) == 1
    r = results[0]
    assert r.boxes is None
    assert tuple(r.flow.data.shape) == (141, 203, 2)
    conf = torch.as_tensor(r.flow.confidence)
    assert conf.shape == (141, 203)
    assert float(conf.min()) >= 0.0 and float(conf.max()) <= 1.0
    assert torch.isfinite(torch.as_tensor(r.flow.data)).all()


def test_forward_flow_hook_matches_direct_pair_forward():
    model = LibreSEARAFT(None, size="s", device="cpu")
    a, b = _pair()
    t1, *_ = model._preprocess(a)
    t2, *_ = model._preprocess(b)
    with torch.no_grad():
        f_hook, _ = model._forward_flow(torch.cat([t1, t2], dim=0))
        f_pair, _ = model._forward((t1, t2))
    assert torch.equal(f_hook, f_pair)


def test_single_image_and_training_are_rejected():
    model = LibreSEARAFT(None, size="s", device="cpu")
    a, _ = _pair()
    with pytest.raises(ValueError):
        model(a)
    with pytest.raises(NotImplementedError):
        model.train(data="x.yaml")


def test_preprocess_matches_upstream_cuda_normalisation_bit_for_bit():
    # Upstream divides by 255.0 on the GPU, which CUDA computes as a multiply
    # by the float32 reciprocal; parity on the clip depends on matching it.
    img = np.arange(256, dtype=np.uint8).repeat(3 * 128).reshape(128, 256, 3)
    x, _ = preprocess_numpy(img, None)
    ref = torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1)[None] * (
        torch.tensor(1.0, dtype=torch.float32) / torch.tensor(255.0, dtype=torch.float32)
    )
    assert torch.equal(x, ref)
