"""Depth Pro 2 inference parity against Apple's pinned ml-depth-pro-2 implementation.

Recipe:

1. Clone ``https://github.com/apple-aiml-research/ml-depth-pro-2`` at commit
   ``aa389cbb0b74341664ed663a0966c859905cd6c0`` and set ``DEPTHPRO2_UPSTREAM_DIR``
   to the repository root.
2. Optionally set ``DEPTHPRO2_CHECKPOINT`` (path to an official safetensors or
   ``.pt`` depth checkpoint, e.g. ``apple/DepthPro2-S/model.safetensors``) and
   ``DEPTHPRO2_FOCAL_CHECKPOINT`` (path to ``apple/DepthPro2-focal/model.safetensors``)
   plus ``DEPTHPRO2_SIZE`` (default ``s``). When checkpoint paths are omitted,
   seeded weights from ``build_depthpro2_model`` are cross-loaded with
   ``strict=True`` into upstream ``DepthPro2`` so every layer, RoPE buffer,
   decoder fusion, focal estimator, buffered cross-fade, and unbuffered KV-cache
   path is verified bitwise in FP32.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch

UPSTREAM_DIR = os.environ.get("DEPTHPRO2_UPSTREAM_DIR")
CHECKPOINT_PATH = os.environ.get("DEPTHPRO2_CHECKPOINT")
FOCAL_CHECKPOINT_PATH = os.environ.get("DEPTHPRO2_FOCAL_CHECKPOINT")
MODEL_SIZE = os.environ.get("DEPTHPRO2_SIZE", "s").lower()

pytestmark = [
    pytest.mark.depthpro2,
    pytest.mark.external_data,
    pytest.mark.skipif(
        not UPSTREAM_DIR or not Path(UPSTREAM_DIR).is_dir(),
        reason=(
            "Set DEPTHPRO2_UPSTREAM_DIR to a checkout of "
            "apple-aiml-research/ml-depth-pro-2 at aa389cbb"
        ),
    ),
]


def _load_weights(path: str) -> dict[str, torch.Tensor]:
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file

        return dict(load_file(path, device="cpu"))
    raw = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(raw, dict) and "model" in raw and isinstance(raw["model"], dict):
        return raw["model"]
    return dict(raw)


def _import_upstream_depthpro2():
    """Import upstream ``DepthPro2`` without requiring upstream's video/HEIF IO extras."""
    import importlib.util
    import types

    stubbed: list[str] = []
    if importlib.util.find_spec("av") is None and "av" not in sys.modules:
        sys.modules["av"] = types.ModuleType("av")
        stubbed.append("av")
    if (
        importlib.util.find_spec("pillow_heif") is None
        and "pillow_heif" not in sys.modules
    ):
        heif_stub = types.ModuleType("pillow_heif")
        heif_stub.register_heif_opener = lambda: None  # type: ignore[attr-defined]
        sys.modules["pillow_heif"] = heif_stub
        stubbed.append("pillow_heif")

    src_root = str(Path(UPSTREAM_DIR) / "src")
    sys.path.insert(0, src_root)
    try:
        from depth_pro_2.model import PRESETS, DepthPro2
    finally:
        sys.path.pop(0)
        for mod_name in stubbed:
            sys.modules.pop(mod_name, None)
    return PRESETS, DepthPro2


def _build_pair(size: str):
    pytest.importorskip("timm")
    PRESETS, DepthPro2 = _import_upstream_depthpro2()

    from libreyolo.models.depthpro2.convert import has_fov_weights, merge_fov_state_dict
    from libreyolo.models.depthpro2.nn import build_depthpro2_model

    preset = PRESETS[f"depthpro2-{size}"]

    state_dict: dict[str, torch.Tensor] | None = None
    use_fov = True
    if CHECKPOINT_PATH:
        state_dict = _load_weights(CHECKPOINT_PATH)
        if FOCAL_CHECKPOINT_PATH:
            state_dict = merge_fov_state_dict(
                state_dict, _load_weights(FOCAL_CHECKPOINT_PATH)
            )
        use_fov = has_fov_weights(state_dict)

    torch.manual_seed(0)
    ours = build_depthpro2_model(size, fov=use_fov, precision=torch.float32)
    upstream = DepthPro2(
        backbone_preset=preset.backbone_preset,
        reassembled_blocks=preset.reassembled_blocks,
        cross_frame_blocks=preset.cross_frame_blocks,
        dims_reassemble=preset.dims_reassemble,
        dim_decoder=preset.dim_decoder,
        fov=use_fov,
        precision=torch.float32,
    )

    if state_dict is None:
        # Stabilize random-weight ReLU inverse-depth head so depth = f / (W * inv_d) is finite.
        with torch.no_grad():
            ours.head[4].bias.fill_(0.25)
        state_dict = ours.state_dict()
    else:
        ours.load_state_dict(state_dict, strict=True)

    upstream.load_state_dict(state_dict, strict=True)
    return upstream.eval(), ours.eval()


@pytest.fixture(scope="module")
def parity_pair():
    return _build_pair(MODEL_SIZE)


def _assert_pred_equal(got, ref, *, label: str):
    assert torch.equal(got["inverse_depth"], ref["inverse_depth"]), (
        f"{label} inverse_depth max_abs_diff="
        f"{(got['inverse_depth'] - ref['inverse_depth']).abs().max().item()}"
    )
    if ref["depth"] is None:
        assert got["depth"] is None
        assert got["focallength_px"] is None
    else:
        assert got["depth"] is not None
        assert torch.equal(got["depth"], ref["depth"]), (
            f"{label} depth max_abs_diff="
            f"{(got['depth'] - ref['depth']).abs().max().item()}"
        )
        assert got["focallength_px"] is not None
        assert torch.equal(got["focallength_px"], ref["focallength_px"])


def test_depthpro2_single_image_parity(parity_pair):
    upstream, ours = parity_pair
    torch.manual_seed(11)
    img = torch.randn(3, 54, 74)

    with torch.no_grad():
        ref = upstream.infer(img)
        got = ours.infer(img)

    _assert_pred_equal(got, ref, label="single_image")


def test_depthpro2_buffered_video_parity(parity_pair):
    upstream, ours = parity_pair
    torch.manual_seed(22)
    frames = [torch.randn(3, 52, 68) for _ in range(5)]

    with torch.no_grad():
        ref_preds = list(upstream.infer_video(frames, window=3, overlap=1, mode="buffered"))
        got_preds = list(ours.infer_video(frames, window=3, overlap=1, mode="buffered"))

    assert len(got_preds) == len(ref_preds) == len(frames)
    for idx, (got, ref) in enumerate(zip(got_preds, ref_preds, strict=True)):
        _assert_pred_equal(got, ref, label=f"buffered[{idx}]")


def test_depthpro2_unbuffered_kv_cache_video_parity(parity_pair):
    upstream, ours = parity_pair
    torch.manual_seed(33)
    frames = [torch.randn(3, 52, 68) for _ in range(4)]

    # Upstream's online unbuffered session (DepthStream) vs our infer_video(mode="unbuffered")
    # which runs without lookahead.
    with torch.no_grad():
        up_stream = upstream.stream(window=3, overlap=0, mode="unbuffered")
        ref_preds = []
        for frame in frames:
            ref_preds.extend(up_stream.step(frame))
        ref_preds.extend(up_stream.flush())

        got_preds = list(ours.infer_video(frames, window=3, overlap=0, mode="unbuffered"))

    assert len(got_preds) == len(ref_preds) == len(frames)
    for idx, (got, ref) in enumerate(zip(got_preds, ref_preds, strict=True)):
        _assert_pred_equal(got, ref, label=f"unbuffered[{idx}]")
