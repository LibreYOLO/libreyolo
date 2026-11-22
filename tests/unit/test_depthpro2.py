"""Hermetic unit tests for the Apple Depth Pro 2 family (random weights, CPU only)."""

from __future__ import annotations

import builtins
import io
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

from libreyolo import LibreDepthPro2, LibreYOLO
from libreyolo.models.autoconvert import autoconvert_upstream_checkpoint
from libreyolo.models.depthpro2.convert import (
    EMBED_DIM_TO_SIZE,
    FOV_UPSTREAM_SHA256,
    FOV_UPSTREAM_URL,
    MIRROR_SHA256,
    UPSTREAM_SHA256,
    UPSTREAM_URLS,
    has_fov_weights,
    merge_fov_state_dict,
    verify_and_wrap_download,
    wrap_upstream_state_dict,
)
from libreyolo.models.depthpro2.encoder import KVCache, _require_timm
from libreyolo.models.depthpro2.fov import crop_angles_to_full, median_focal_length_px
from libreyolo.models.depthpro2.streaming import blend_ramp
from libreyolo.models.depthpro2.utils import (
    extract_exif_f_px,
    fpx_from_f35,
    normalize_rgb,
    pixel_intrinsics,
    preprocess_numpy,
)
from libreyolo.utils.serialization import validate_checkpoint_metadata

pytestmark = [pytest.mark.unit, pytest.mark.depthpro2]

REPO_ROOT = Path(__file__).resolve().parents[2]


def _dp2_signature(embed_dim: int = 384, *, with_fov: bool = True) -> dict[str, torch.Tensor]:
    state = {
        "encoder.patch_embed.proj.weight": torch.zeros(embed_dim, 3, 16, 16),
        "encoder.cls_token": torch.zeros(1, 1, embed_dim),
        "encoder.reg_token": torch.zeros(1, 4, embed_dim),
        "encoder.reassemble.0.0.weight": torch.zeros(48, embed_dim, 1, 1),
        "decoder.fusions.0.deconv.weight": torch.zeros(64, 64, 2, 2),
        "head.1.weight": torch.zeros(32, 32, 2, 2),
    }
    if with_fov:
        state["fov.trunk.cls_token"] = torch.zeros(1, 1, 384)
        state["fov.head.out.weight"] = torch.zeros(2, 248)
    return state


@pytest.fixture(scope="module")
def dp2_small():
    pytest.importorskip("timm")
    torch.manual_seed(1)
    model = LibreDepthPro2(model_path=None, size="s", device="cpu")
    with torch.no_grad():
        # Depth head ends with ReLU -> 1/x; keep the random-weight unit-test
        # bias positive so DepthValidator never sees an all-inf dead-ReLU map.
        model.model.head[4].bias.fill_(0.25)
    return model


@pytest.fixture(scope="module")
def converter():
    sys.path.insert(0, str(REPO_ROOT / "weights"))
    try:
        import convert_depthpro2_weights
    finally:
        sys.path.remove(str(REPO_ROOT / "weights"))
    return convert_depthpro2_weights


def _images(n: int, size: tuple[int, int] = (74, 54)) -> list[Image.Image]:
    rng = np.random.default_rng(7)
    w, h = size
    return [
        Image.fromarray(rng.integers(0, 255, (h, w, 3), dtype=np.uint8))
        for _ in range(n)
    ]


# ---------------------------------------------------------------- metadata & licensing


def test_family_metadata_and_registration():
    from libreyolo.models.base import BaseModel
    from libreyolo.models.registry import MODEL_GROUPS

    assert LibreDepthPro2.FAMILY == "depthpro2"
    assert LibreDepthPro2.FILENAME_PREFIX == "LibreDepthPro2"
    assert LibreDepthPro2.INPUT_SIZES == {"s": 512, "b": 512, "l": 512, "h": 512}
    assert LibreDepthPro2.SUPPORTED_TASKS == ("depth",)
    assert LibreDepthPro2.DEFAULT_TASK == "depth"
    assert LibreDepthPro2.REQUIRE_TASK_SUFFIX is True
    assert LibreDepthPro2.SEQUENCE_INPUT is True
    assert LibreDepthPro2.SUPPORTS_BATCHED_PREDICT is False
    assert LibreDepthPro2.TRAIN_CONFIG is None
    assert LibreDepthPro2.depth_imgsz_divisor == 32
    assert LibreDepthPro2 in BaseModel._registry
    assert MODEL_GROUPS["depthpro2"] == "g3"


def test_license_and_notice_files_ship_in_package():
    pkg_dir = REPO_ROOT / "libreyolo" / "models" / "depthpro2"
    for name in ("LICENSE", "LICENSE_MODEL", "ACKNOWLEDGEMENTS", "NOTICE"):
        path = pkg_dir / name
        assert path.exists() and path.stat().st_size > 100
    assert "Apple Inc." in (pkg_dir / "LICENSE").read_text(encoding="utf-8")
    assert "apple-amlr" in (pkg_dir / "NOTICE").read_text(encoding="utf-8")


# ---------------------------------------------------------------- recognition & routing


@pytest.mark.parametrize(("embed_dim", "size"), list(EMBED_DIM_TO_SIZE.items()))
def test_can_load_and_detect_size(embed_dim: int, size: str):
    state = _dp2_signature(embed_dim, with_fov=True)
    assert LibreDepthPro2.can_load(state)
    assert LibreDepthPro2.detect_size(state) == size
    assert LibreDepthPro2.detect_nb_classes(state) == 1
    assert has_fov_weights(state)

    state_no_fov = _dp2_signature(embed_dim, with_fov=False)
    assert LibreDepthPro2.can_load(state_no_fov)
    assert LibreDepthPro2.detect_size(state_no_fov) == size
    assert not has_fov_weights(state_no_fov)


def test_only_depthpro2_claims_depthpro2_signature():
    from libreyolo.models.base import BaseModel

    state = _dp2_signature(384, with_fov=True)
    claimants = [cls for cls in BaseModel._registry if cls.can_load(state)]
    assert claimants == [LibreDepthPro2]


def test_depthpro2_rejects_sibling_depth_families():
    siblings = [
        # MiDaS DPT-Large
        {
            "pretrained.model.cls_token": torch.zeros(1, 1, 1024),
            "scratch.refinenet1.resConfUnit1.conv1.weight": torch.zeros(1),
            "scratch.output_conv.4.weight": torch.zeros(1),
        },
        # Depth Anything V2
        {
            "pretrained.cls_token": torch.zeros(1, 1, 384),
            "depth_head.scratch.output_conv2.0.weight": torch.zeros(1),
        },
        # Video Depth Anything
        {
            "pretrained.cls_token": torch.zeros(1, 1, 384),
            "head.motion_modules.0.weight": torch.zeros(1),
            "head.scratch.output_conv2.0.weight": torch.zeros(1),
        },
        # MoGe-3
        {
            "encoder.backbone.cls_token": torch.zeros(1, 1, 1024),
            "neck.input_blocks.0.weight": torch.zeros(1),
            "refiner.input_proj.weight": torch.zeros(32, 3),
        },
        # ZipDepth
        {
            "encoder.stem_half.conv.weight": torch.zeros(24, 3, 3, 3),
            "decoder.convex_up.weight": torch.zeros(1),
        },
    ]
    for sibling in siblings:
        assert not LibreDepthPro2.can_load(sibling)


# ---------------------------------------------------------------- filenames, URLs & verification


@pytest.mark.parametrize("size", ["s", "b", "l", "h"])
def test_filenames_urls_checksums_and_notice(size: str):
    canonical = f"LibreDepthPro2{size}-depth.pt"
    expected_url = (
        f"https://huggingface.co/LibreYOLO/LibreDepthPro2{size}-depth/"
        f"resolve/main/{canonical}"
    )
    assert LibreDepthPro2.detect_size_from_filename(canonical) == size
    assert LibreDepthPro2.detect_task_from_filename(canonical) == "depth"
    assert LibreDepthPro2.get_download_url(canonical) == expected_url
    assert len(MIRROR_SHA256[size]) == 64
    assert len(UPSTREAM_SHA256[size]) == 64
    # Task suffix is mandatory.
    assert LibreDepthPro2.get_download_url(f"LibreDepthPro2{size}.pt") is None

    notice = LibreDepthPro2.get_download_notice(canonical, expected_url)
    assert notice is not None
    assert "apple-amlr" in notice
    assert "non-commercial" in notice.lower()
    assert "Apple Machine Learning Research Model License Agreement" in notice
    assert "Apple Inc." in notice


def test_fov_upstream_pin_and_verification_rejects_bad_downloads(tmp_path):
    assert FOV_UPSTREAM_URL.startswith("https://huggingface.co/apple/DepthPro2-focal/resolve/")
    assert len(FOV_UPSTREAM_SHA256) == 64

    bad_file = tmp_path / "LibreDepthPro2s-depth.pt"
    bad_file.write_bytes(b"corrupted bytes")
    mirror_url = LibreDepthPro2.get_download_url("LibreDepthPro2s-depth.pt")
    assert mirror_url is not None
    with pytest.raises(RuntimeError, match="Checksum mismatch"):
        verify_and_wrap_download(str(bad_file), mirror_url)
    with pytest.raises(RuntimeError, match="unpinned"):
        verify_and_wrap_download(
            str(bad_file),
            "https://huggingface.co/LibreYOLO/LibreDepthPro2z-depth/resolve/main/LibreDepthPro2z-depth.pt",
        )
    with pytest.raises(RuntimeError, match="Checksum mismatch"):
        verify_and_wrap_download(str(bad_file), UPSTREAM_URLS["s"])
    with pytest.raises(RuntimeError, match="unpinned"):
        verify_and_wrap_download(
            str(bad_file),
            "https://huggingface.co/apple/DepthPro2-S/resolve/main/model.safetensors",
        )


# ---------------------------------------------------------------- utilities & math


def test_preprocess_and_normalize_match_imagenet_stats():
    img = np.array([[[0, 128, 255]]], dtype=np.uint8)
    chw, ratio = preprocess_numpy(img)
    assert ratio == 1.0
    assert chw.shape == (3, 1, 1) and chw.dtype == np.float32
    np.testing.assert_allclose(chw[:, 0, 0], [0.0, 128.0 / 255.0, 1.0], rtol=1e-6)

    norm = normalize_rgb(torch.from_numpy(chw))
    expected = (torch.tensor([0.0, 128.0 / 255.0, 1.0]) - torch.tensor([0.485, 0.456, 0.406])) / torch.tensor([0.229, 0.224, 0.225])
    torch.testing.assert_close(norm[:, 0, 0], expected, rtol=1e-6, atol=1e-6)


def test_exif_focal_length_and_pixel_intrinsics():
    f_px = fpx_from_f35(4000, 3000, 35.0)
    assert f_px == pytest.approx(35.0 * 5000.0 / np.hypot(36.0, 24.0))

    # Build a tiny JPEG carrying FocalLengthIn35mmFilm = 50 in its EXIF IFD.
    pil = Image.fromarray(np.zeros((60, 80, 3), dtype=np.uint8))
    exif = pil.getexif()
    exif_ifd = exif.get_ifd(0x8769)
    exif_ifd[41989] = 50
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", exif=exif.tobytes())
    buf.seek(0)
    with Image.open(buf) as loaded:
        parsed = extract_exif_f_px(loaded, 80, 60)
    assert parsed == pytest.approx(fpx_from_f35(80, 60, 50.0))

    K = pixel_intrinsics(torch.tensor(512.5), width=80, height=60)
    assert K.shape == (3, 3) and K.dtype == torch.float32
    expected_K = torch.tensor(
        [[512.5, 0.0, 39.5], [0.0, 512.5, 29.5], [0.0, 0.0, 1.0]], dtype=torch.float32
    )
    torch.testing.assert_close(K, expected_K)


def test_kv_cache_ring_buffer_and_blend_ramp_and_fov_math():
    cache = KVCache(window=2)
    p0 = torch.ones(1, 5, 8)
    p1 = torch.full((1, 5, 8), 3.0)
    p2 = torch.full((1, 5, 8), 5.0)
    assert torch.allclose(cache.update_prefix(0, p0), p0)
    k0, _v0 = cache.update(0, torch.zeros(1, 2, 4, 4), torch.zeros(1, 2, 4, 4), torch.ones(1, 2, 5, 4), torch.ones(1, 2, 5, 4))
    assert k0.shape == (1, 2, 5 + 4, 4)

    assert torch.allclose(cache.update_prefix(0, p1), torch.full((1, 5, 8), 2.0))
    k1, _ = cache.update(0, torch.ones(1, 2, 4, 4), torch.ones(1, 2, 4, 4), torch.ones(1, 2, 5, 4), torch.ones(1, 2, 5, 4))
    assert k1.shape == (1, 2, 5 + 8, 4)

    # Third frame evicts frame 0; mean of {p1=3, p2=5} is 4.
    assert torch.allclose(cache.update_prefix(0, p2), torch.full((1, 5, 8), 4.0))
    k2, _ = cache.update(0, torch.full((1, 2, 4, 4), 2.0), torch.full((1, 2, 4, 4), 2.0), torch.ones(1, 2, 5, 4), torch.ones(1, 2, 5, 4))
    assert k2.shape == (1, 2, 5 + 8, 4)

    ramp = blend_ramp(3, torch.zeros(3, 2, 2))
    torch.testing.assert_close(ramp.squeeze(), torch.tensor([0.75, 0.5, 0.25]))

    # FoV focal math consistency
    f_med = median_focal_length_px(torch.tensor([[90.0, 60.0], [90.0, 90.0]]), height=100, width=200)
    assert f_med.item() == pytest.approx(50.0 / np.tan(np.deg2rad(60.0) / 2.0), rel=1e-5)

    # Center crop conversion round-trips a known pinhole camera
    angles = torch.tensor([[60.0, 40.0]])
    full = crop_angles_to_full(angles, crop_hw=(100.0, 160.0), full_hw=(100.0, 200.0))
    assert full[0, 1].item() == pytest.approx(40.0, rel=1e-5)
    assert full[0, 0].item() > 60.0


def test_missing_timm_error_mentions_depthpro2_extra(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "timm" or name.startswith("timm."):
            raise ImportError("no timm")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ModuleNotFoundError, match=r"libreyolo\[depthpro2\]"):
        _require_timm()


# ---------------------------------------------------------------- conversion & autoconvert


def test_wrap_and_merge_helpers():
    depth_sd = _dp2_signature(768, with_fov=False)
    fov_sd = {
        "trunk.cls_token": torch.zeros(1, 1, 384),
        "head.out.weight": torch.zeros(2, 248),
    }
    merged = merge_fov_state_dict(depth_sd, fov_sd)
    assert has_fov_weights(merged)
    wrapped = wrap_upstream_state_dict(merged, "b")
    assert validate_checkpoint_metadata(wrapped, strict=True) == []
    assert wrapped["model_family"] == "depthpro2"
    assert wrapped["size"] == "b"
    assert wrapped["task"] == "depth"


def test_converter_and_autoconvert_end_to_end(tmp_path, converter, dp2_small):
    from safetensors.torch import save_file

    full_sd = dp2_small.model.state_dict()
    depth_sd = {k: v for k, v in full_sd.items() if not k.startswith("fov.")}
    fov_sd = {k[len("fov.") :]: v for k, v in full_sd.items() if k.startswith("fov.")}

    depth_sft = tmp_path / "depthpro2_s.safetensors"
    fov_sft = tmp_path / "depthpro2_focal.safetensors"
    save_file(depth_sd, str(depth_sft))
    save_file(fov_sd, str(fov_sft))

    # 1. Bundled conversion (depth + fov)
    out_bundled = tmp_path / "bundled" / "LibreDepthPro2s-depth.pt"
    wrapped = converter.convert_weights(
        str(depth_sft), str(out_bundled), fov_path=str(fov_sft)
    )
    assert validate_checkpoint_metadata(wrapped, strict=True) == []
    loaded = LibreYOLO(str(out_bundled), device="cpu")
    assert isinstance(loaded, LibreDepthPro2)
    assert loaded.model.fov is not None
    assert loaded.depth_encoding == "depth"

    # 2. Depth-only conversion (no fov)
    out_nofov = tmp_path / "nofov" / "LibreDepthPro2s-depth.pt"
    converter.convert_weights(str(depth_sft), str(out_nofov))
    loaded_nofov = LibreYOLO(str(out_nofov), device="cpu")
    assert isinstance(loaded_nofov, LibreDepthPro2)
    assert loaded_nofov.model.fov is None
    assert loaded_nofov.depth_encoding == "inverse_depth"

    # 3. Runtime auto-conversion of raw upstream .pt
    raw_pt = tmp_path / "raw_upstream.pt"
    torch.save(depth_sd, raw_pt)
    converted_path = autoconvert_upstream_checkpoint(str(raw_pt))
    assert converted_path is not None
    assert converted_path.endswith("LibreDepthPro2s-depth.pt")


# ---------------------------------------------------------------- inference & validation


def test_single_image_and_direct_forward_parity(dp2_small):
    img = _images(1, size=(74, 54))[0]
    result = dp2_small.predict(img)
    assert not isinstance(result, list)
    assert result.boxes is None
    assert result.depth_map is not None
    assert result.depth_map.metric is True
    assert result.depth_map.encoding == "depth"
    assert tuple(result.depth_map.data.shape) == (54, 74)
    assert result.depth_map.intrinsics is not None
    assert tuple(result.depth_map.intrinsics.shape) == (3, 3)

    # Matches the independent-image _forward path used by DepthValidator.
    tensor, _, _, _ = dp2_small._preprocess(img)
    with torch.no_grad():
        direct = dp2_small._forward(tensor)
    assert tuple(direct.shape) == (1, 1, 54, 74)
    assert torch.equal(result.depth_map.data, direct[0, 0].cpu())


def test_buffered_and_unbuffered_video_sequence(dp2_small):
    frames = _images(4, size=(68, 52))

    # Buffered mode with overlapping windows
    dp2_small.mode = "buffered"
    dp2_small.window = 3
    dp2_small.overlap = 1
    buf_results = dp2_small.predict(frames)
    assert isinstance(buf_results, list) and len(buf_results) == 4
    focals = [float(r.depth_map.intrinsics[0, 0]) for r in buf_results]
    assert len(set(focals)) == 1
    for res in buf_results:
        assert res.depth_map.metric is True
        assert res.depth_map.encoding == "depth"
        assert tuple(res.depth_map.data.shape) == (52, 68)

    # Unbuffered causal KV-cache mode
    dp2_small.mode = "unbuffered"
    dp2_small.window = 2
    unbuf_results = dp2_small.predict(frames)
    assert isinstance(unbuf_results, list) and len(unbuf_results) == 4
    for res in unbuf_results:
        assert res.depth_map.metric is True
        assert tuple(res.depth_map.data.shape) == (52, 68)

    # Restore defaults on shared fixture
    dp2_small.mode = "buffered"
    dp2_small.window = 16
    dp2_small.overlap = 4


def test_unbuffered_mode_yields_without_lookahead(dp2_small):
    frames = _images(4, size=(64, 48))
    norm_frames = [
        normalize_rgb(torch.from_numpy(preprocess_numpy(np.asarray(im))[0]))
        for im in frames
    ]

    # 1. DepthPro2Net.infer_video(mode="unbuffered") must yield frame i after reading
    # only frame i, never looking ahead `window` frames for focal estimation.
    pulled = 0

    def tracked_frames():
        nonlocal pulled
        for f in norm_frames:
            pulled += 1
            yield f

    unbuf_preds = []
    for idx, pred in enumerate(
        dp2_small.model.infer_video(tracked_frames(), window=4, overlap=0, mode="unbuffered")
    ):
        assert pulled == idx + 1, f"expected no lookahead on frame {idx}, pulled={pulled}"
        unbuf_preds.append(pred)
    assert len(unbuf_preds) == len(norm_frames)

    # Output must match driving DepthStream directly frame-by-frame.
    stream = dp2_small.model.stream(window=4, overlap=0, mode="unbuffered")
    direct_preds = []
    for f in norm_frames:
        direct_preds.extend(stream.step(f))
    direct_preds.extend(stream.flush())
    assert len(direct_preds) == len(unbuf_preds)
    for got, ref in zip(unbuf_preds, direct_preds, strict=True):
        assert torch.equal(got["inverse_depth"], ref["inverse_depth"])
        assert got["depth"] is not None and ref["depth"] is not None
        assert torch.equal(got["depth"], ref["depth"])
        assert got["focallength_px"] is not None and ref["focallength_px"] is not None
        assert torch.equal(got["focallength_px"], ref["focallength_px"])

    # 2. LibreDepthPro2._forward_sequence in unbuffered mode must also yield frame 0
    # before pulling frame 1.
    prev_mode, prev_window = dp2_small.mode, dp2_small.window
    try:
        dp2_small.mode = "unbuffered"
        dp2_small.window = 4
        seq_pulled = 0

        def tracked_seq():
            nonlocal seq_pulled
            for im in frames:
                seq_pulled += 1
                chw, _ = preprocess_numpy(np.asarray(im))
                yield torch.from_numpy(chw).unsqueeze(0), im.size

        for idx, pred in enumerate(dp2_small._forward_sequence(tracked_seq())):
            assert seq_pulled == idx + 1, (
                f"_forward_sequence looked ahead on frame {idx}: seq_pulled={seq_pulled}"
            )
            assert pred["depth"] is not None
    finally:
        dp2_small.mode = prev_mode
        dp2_small.window = prev_window


def test_without_fov_returns_inverse_depth_and_f_px_override_restores_metric():
    pytest.importorskip("timm")
    model = LibreDepthPro2(None, size="s", device="cpu", fov=False)
    assert model.model.fov is None
    assert model.depth_encoding == "inverse_depth"

    img = _images(1, size=(64, 64))[0]
    res_inv = model.predict(img)
    assert res_inv.depth_map.metric is False
    assert res_inv.depth_map.encoding == "inverse_depth"
    assert res_inv.depth_map.intrinsics is None

    model_fpx = LibreDepthPro2(None, size="s", device="cpu", fov=False, f_px=500.0)
    assert model_fpx.depth_encoding == "depth"
    res_metric = model_fpx.predict(img)
    assert res_metric.depth_map.metric is True
    assert res_metric.depth_map.encoding == "depth"
    assert res_metric.depth_map.intrinsics is not None
    assert float(res_metric.depth_map.intrinsics[0, 0]) == pytest.approx(500.0)


def test_depth_validator_zero_shot_val(tmp_path, dp2_small):
    img_dir = tmp_path / "images" / "val"
    dep_dir = tmp_path / "depths" / "val"
    img_dir.mkdir(parents=True)
    dep_dir.mkdir(parents=True)

    rng = np.random.default_rng(1)
    for idx in range(2):
        rgb = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
        Image.fromarray(rgb).save(img_dir / f"sample_{idx}.png")
        gt = np.linspace(1.0, 10.0, 64 * 64, dtype=np.float32).reshape(64, 64)
        np.save(dep_dir / f"sample_{idx}.npy", gt)

    yaml_path = tmp_path / "depth.yaml"
    yaml_path.write_text(
        yaml.safe_dump({"path": str(tmp_path), "val": "images/val"}),
        encoding="utf-8",
    )

    metrics = dp2_small.val(data=str(yaml_path), imgsz=64, batch=2, workers=0)
    for key in (
        "metrics/abs_rel",
        "metrics/rmse",
        "metrics/delta1",
        "metrics/delta2",
        "metrics/delta3",
        "fitness",
    ):
        assert key in metrics
        assert np.isfinite(metrics[key])


def test_invalid_arguments_and_out_of_scope_methods(dp2_small):
    with pytest.raises(ValueError, match="mode"):
        LibreDepthPro2(None, size="s", device="cpu", mode="invalid")
    with pytest.raises(ValueError, match="window"):
        LibreDepthPro2(None, size="s", device="cpu", window=0)
    with pytest.raises(ValueError, match="overlap"):
        LibreDepthPro2(None, size="s", device="cpu", window=4, overlap=4)
    with pytest.raises(ValueError, match="precision"):
        LibreDepthPro2(None, size="s", device="cpu", precision="int8")

    with pytest.raises(NotImplementedError):
        dp2_small.train(data="x.yaml")
    with pytest.raises(NotImplementedError):
        dp2_small.export(format="onnx")
