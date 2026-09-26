"""YOLOv9 letterbox stamp + PGI aux: no silent 1.5 → 1.6 geometry flip."""

from __future__ import annotations

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.unit

from libreyolo.preprocess.letterbox import (
    DEFAULT_LETTERBOX_PAD,
    LETTERBOX_CENTER,
    LETTERBOX_TOPLEFT,
    apply_letterbox_hwc,
    letterbox_geometry,
    normalize_letterbox_pad,
    unletterbox_xyxy,
)
from libreyolo.preprocess.yolo9 import preprocess_numpy
from libreyolo.postprocess.yolo9 import postprocess
from libreyolo.validation.preprocessors import YOLO9ValPreprocessor


def _uint8_gradient(h=40, w=80):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :, 0] = np.linspace(0, 255, w, dtype=np.uint8)[None, :]
    img[:, :, 1] = np.linspace(0, 255, h, dtype=np.uint8)[:, None]
    img[:, :, 2] = 40
    return img


def test_unmarked_letterbox_pad_defaults_to_topleft():
    assert normalize_letterbox_pad(None) == LETTERBOX_TOPLEFT
    assert normalize_letterbox_pad("") == LETTERBOX_TOPLEFT
    assert normalize_letterbox_pad("nope") == LETTERBOX_TOPLEFT
    assert DEFAULT_LETTERBOX_PAD == LETTERBOX_TOPLEFT


def test_topleft_geometry_has_zero_pad_offsets():
    ratio, new_h, new_w, pad_left, pad_top = letterbox_geometry(40, 80, 64, 64, "topleft")
    assert pad_left == 0
    assert pad_top == 0
    assert new_w == 64
    assert new_h == 32
    assert ratio == pytest.approx(0.8)


def test_center_geometry_splits_pad():
    _ratio, _nh, _nw, pad_left, pad_top = letterbox_geometry(40, 80, 64, 64, "center")
    assert pad_left == 0
    assert pad_top == 16


def test_topleft_preprocess_is_bit_identical_to_historical_placement():
    img = _uint8_gradient()
    historical = np.full((64, 64, 3), 114, dtype=np.uint8)
    import cv2

    resized = cv2.resize(img, (64, 32), interpolation=cv2.INTER_LINEAR)
    historical[:32, :64] = resized

    canvas, ratio, pad_left, pad_top = apply_letterbox_hwc(img, 64, 64, pad="topleft")
    assert pad_left == 0
    assert pad_top == 0
    np.testing.assert_array_equal(canvas, historical)
    assert ratio == pytest.approx(0.8)


def test_default_preprocess_numpy_matches_topleft():
    img = _uint8_gradient()
    default, _ = preprocess_numpy(img, 64)
    topleft, _ = preprocess_numpy(img, 64, letterbox_pad="topleft")
    center, _ = preprocess_numpy(img, 64, letterbox_pad="center")
    np.testing.assert_array_equal(default, topleft)
    assert not np.array_equal(default, center)


def test_val_preprocessor_default_matches_predict_topleft():
    img = _uint8_gradient()
    tensor, _ = preprocess_numpy(img, 64, letterbox_pad=None)
    val, _ = YOLO9ValPreprocessor((64, 64), max_labels=1)(
        img[:, :, ::-1].copy(),
        np.zeros((0, 5), dtype=np.float32),
        (64, 64),
    )
    np.testing.assert_allclose(tensor, val, atol=1e-6)


def test_val_preprocessor_center_moves_content():
    img = _uint8_gradient()
    topleft, _ = YOLO9ValPreprocessor((64, 64), letterbox_pad="topleft")(
        img[:, :, ::-1].copy(),
        np.zeros((0, 5), dtype=np.float32),
        (64, 64),
    )
    center, _ = YOLO9ValPreprocessor((64, 64), letterbox_pad="center")(
        img[:, :, ::-1].copy(),
        np.zeros((0, 5), dtype=np.float32),
        (64, 64),
    )
    assert not np.array_equal(topleft, center)
    r, off_x, off_y = YOLO9ValPreprocessor((64, 64), letterbox_pad="center").letterbox_scale(
        40, 80, 64
    )
    assert off_x == 0.0
    assert off_y == 16.0
    assert r == pytest.approx(0.8)


def test_center_val_targets_get_pad_so_gt_undo_recovers_original():
    """Dataset hands a ratio-scaled, unpadded frame. Center pad must be
    added to GT or the validator subtracts an offset that was never applied
    (62.5px on a 500x375 → 640 letterbox)."""
    orig_w, orig_h, imgsz = 500, 375, 640
    r, off_x, off_y = YOLO9ValPreprocessor(
        (imgsz, imgsz), letterbox_pad="center"
    ).letterbox_scale(orig_h, orig_w, imgsz)
    assert r == pytest.approx(min(imgsz / orig_h, imgsz / orig_w))
    assert off_x == 0.0
    assert off_y == 80.0

    # Dataset contract: image already resized, boxes already * r.
    resized_box = np.array([[100.0 * r, 50.0 * r, 200.0 * r, 150.0 * r, 0.0]], dtype=np.float32)
    resized_h = int(orig_h * r)
    resized_w = int(orig_w * r)
    frame = np.zeros((resized_h, resized_w, 3), dtype=np.uint8)
    _img, targets = YOLO9ValPreprocessor(
        (imgsz, imgsz), max_labels=4, letterbox_pad="center"
    )(frame, resized_box, (imgsz, imgsz))
    # Validator undo: subtract pad, divide by r.
    recovered = targets[0, :4].copy()
    recovered[[0, 2]] = (recovered[[0, 2]] - off_x) / r
    recovered[[1, 3]] = (recovered[[1, 3]] - off_y) / r
    np.testing.assert_allclose(recovered, [100.0, 50.0, 200.0, 150.0], atol=1e-4)


def test_topleft_val_targets_stay_unshifted():
    orig_w, orig_h, imgsz = 500, 375, 640
    r = min(imgsz / orig_h, imgsz / orig_w)
    resized_box = np.array([[100.0 * r, 50.0 * r, 200.0 * r, 150.0 * r, 0.0]], dtype=np.float32)
    frame = np.zeros((int(orig_h * r), int(orig_w * r), 3), dtype=np.uint8)
    _img, targets = YOLO9ValPreprocessor(
        (imgsz, imgsz), max_labels=4, letterbox_pad="topleft"
    )(frame, resized_box, (imgsz, imgsz))
    np.testing.assert_allclose(targets[0, :4], resized_box[0, :4], atol=1e-5)


def test_postprocess_topleft_undo_matches_historical_divide_by_ratio():
    pred = torch.zeros(1, 6, 1)
    pred[0, :4, 0] = torch.tensor([0.0, 0.0, 320.0, 320.0])
    pred[0, 4, 0] = 0.9

    out = postprocess(
        {"predictions": pred},
        input_size=640,
        original_size=(1280, 960),
    )
    torch.testing.assert_close(
        torch.as_tensor(out["boxes"]),
        torch.tensor([[0.0, 0.0, 640.0, 640.0]]),
    )

    explicit = postprocess(
        {"predictions": pred.clone()},
        input_size=640,
        original_size=(1280, 960),
        letterbox_pad="topleft",
    )
    torch.testing.assert_close(
        torch.as_tensor(out["boxes"]),
        torch.as_tensor(explicit["boxes"]),
    )


def test_postprocess_center_subtracts_pad():
    pred = torch.zeros(1, 6, 1)
    # Box sitting in the padded canvas at y=16..336 on a 64-tall content
    # after center pad of 16 on a 40x80 → 64x64 letterbox (ratio 0.8).
    pred[0, :4, 0] = torch.tensor([0.0, 16.0, 64.0, 48.0])
    pred[0, 4, 0] = 0.9
    out = postprocess(
        {"predictions": pred},
        input_size=64,
        original_size=(80, 40),
        letterbox_pad="center",
    )
    torch.testing.assert_close(
        torch.as_tensor(out["boxes"]),
        torch.tensor([[0.0, 0.0, 80.0, 40.0]]),
    )


def test_unletterbox_topleft_is_pure_divide():
    boxes = torch.tensor([[8.0, 4.0, 16.0, 12.0]])
    out = unletterbox_xyxy(boxes, orig_w=80, orig_h=40, input_h=64, input_w=64, pad="topleft")
    torch.testing.assert_close(out, torch.tensor([[10.0, 5.0, 20.0, 15.0]]))


def test_unmarked_yolo9_checkpoint_stays_topleft(tmp_path):
    from libreyolo.models.yolo9.model import LibreYOLO9
    from libreyolo.utils.serialization import wrap_libreyolo_checkpoint
    from libreyolo.models.yolo9.nn import LibreYOLO9Model

    raw = LibreYOLO9Model(config="t", nb_classes=2)
    ckpt = wrap_libreyolo_checkpoint(
        raw.state_dict(),
        model_family="yolo9",
        size="t",
        task="detect",
        nc=2,
        names={0: "a", 1: "b"},
        imgsz=640,
    )
    path = tmp_path / "old.pt"
    torch.save(ckpt, path)

    loaded = LibreYOLO9(str(path), size="t", nb_classes=2, device="cpu")
    assert loaded.letterbox_pad == LETTERBOX_TOPLEFT
    assert loaded.model.aux is None


def test_stamped_center_checkpoint_is_honored(tmp_path):
    from libreyolo.models.yolo9.model import LibreYOLO9
    from libreyolo.utils.serialization import wrap_libreyolo_checkpoint
    from libreyolo.models.yolo9.nn import LibreYOLO9Model

    raw = LibreYOLO9Model(config="t", nb_classes=2)
    ckpt = wrap_libreyolo_checkpoint(
        raw.state_dict(),
        model_family="yolo9",
        size="t",
        task="detect",
        nc=2,
        names={0: "a", 1: "b"},
        imgsz=640,
        letterbox_pad="center",
    )
    path = tmp_path / "official.pt"
    torch.save(ckpt, path)

    loaded = LibreYOLO9(str(path), size="t", nb_classes=2, device="cpu")
    assert loaded.letterbox_pad == LETTERBOX_CENTER
    saved = loaded.save(str(tmp_path / "resave.pt"))
    from libreyolo.utils.serialization import load_untrusted_torch_file

    resaved = load_untrusted_torch_file(saved, map_location="cpu", context="resave")
    assert resaved["letterbox_pad"] == LETTERBOX_CENTER


def test_aux_keys_are_ignored_at_inference(tmp_path):
    from libreyolo.models.yolo9.model import LibreYOLO9
    from libreyolo.utils.serialization import wrap_libreyolo_checkpoint
    from libreyolo.models.yolo9.nn import LibreYOLO9Model

    raw = LibreYOLO9Model(config="t", nb_classes=2)
    raw.enable_aux(0.25)
    ckpt = wrap_libreyolo_checkpoint(
        raw.state_dict(),
        model_family="yolo9",
        size="t",
        task="detect",
        nc=2,
        names={0: "a", 1: "b"},
        imgsz=640,
        letterbox_pad="center",
    )
    path = tmp_path / "with_aux.pt"
    torch.save(ckpt, path)

    loaded = LibreYOLO9(str(path), size="t", nb_classes=2, device="cpu")
    assert loaded.model.aux is None
    assert all(not k.startswith("aux.") for k in loaded.model.state_dict())


def test_aux_head_convert_prefix_loads_into_enabled_branch():
    from libreyolo.models.yolo9.convert import convert_state_dict
    from libreyolo.models.yolo9.nn import LibreYOLO9Model

    sd = {
        "0.conv.weight": torch.zeros(16, 3, 3, 3),
        "30.heads.0.class_conv.2.weight": torch.zeros(5, 16, 1, 1),
    }
    converted, _stats = convert_state_dict(sd, "t")
    assert "aux_head.cv3.0.2.weight" in converted
    model = LibreYOLO9Model(config="t", nb_classes=5)
    model.enable_aux(0.25)
    assert "aux_head.cv3.0.2.weight" in model.state_dict()


def test_fine_tune_reloads_aux_after_inference_strip(tmp_path):
    from libreyolo.models.yolo9.model import LibreYOLO9
    from libreyolo.models.yolo9.nn import LibreYOLO9Model
    from libreyolo.utils.serialization import wrap_libreyolo_checkpoint

    raw = LibreYOLO9Model(config="t", nb_classes=2)
    raw.enable_aux(0.25)
    marker = torch.arange(raw.aux_head.cv3[0][2].weight.numel(), dtype=torch.float32)
    raw.aux_head.cv3[0][2].weight.data.copy_(
        marker.reshape_as(raw.aux_head.cv3[0][2].weight)
    )
    ckpt = wrap_libreyolo_checkpoint(
        raw.state_dict(),
        model_family="yolo9",
        size="t",
        task="detect",
        nc=2,
        names={0: "a", 1: "b"},
        imgsz=640,
        letterbox_pad="center",
    )
    path = tmp_path / "official_aux.pt"
    torch.save(ckpt, path)

    loaded = LibreYOLO9(str(path), size="t", nb_classes=2, device="cpu")
    assert loaded.model.aux is None
    loaded.model.enable_aux(0.25)
    n = loaded._reload_aux_from_path(str(path))
    assert n > 0
    torch.testing.assert_close(
        loaded.model.aux_head.cv3[0][2].weight.detach().cpu().flatten(),
        marker,
    )


def test_enable_aux_is_training_only_and_idempotent():
    from libreyolo.models.yolo9.nn import LibreYOLO9Model

    model = LibreYOLO9Model(config="t", nb_classes=2)
    assert model.aux is None
    model.enable_aux(0.25)
    first = model.aux
    model.enable_aux(0.25)
    assert model.aux is first
    model.eval()
    with torch.no_grad():
        out = model(torch.zeros(1, 3, 64, 64))
    assert "predictions" in out


def test_aux_training_forward_combines_losses():
    from libreyolo.models.yolo9.nn import LibreYOLO9Model

    model = LibreYOLO9Model(config="t", nb_classes=2)
    model.enable_aux(0.25)
    model.train()
    targets = torch.zeros(1, 4, 5)
    targets[0, 0] = torch.tensor([0.0, 0.2, 0.2, 0.4, 0.4])
    out = model(torch.zeros(1, 3, 64, 64), targets=targets)
    assert torch.isfinite(out["total_loss"])
    bare = LibreYOLO9Model(config="t", nb_classes=2)
    bare.train()
    main_only = bare(torch.zeros(1, 3, 64, 64), targets=targets)
    # Combined PGI loss is a different tensor than the single-head path.
    assert out["total_loss"].shape == main_only["total_loss"].shape


def test_model_deepcopies_after_pgi_training_step():
    """EMA, AutoBatch and a second train() deep-copy the live model after a
    graph-building forward; nothing from that forward may stay on the module."""
    import copy

    from libreyolo.models.yolo9.nn import LibreYOLO9Model

    model = LibreYOLO9Model(config="t", nb_classes=2)
    model.enable_aux(0.25)
    model.train()
    targets = torch.zeros(1, 4, 5)
    targets[0, 0] = torch.tensor([0.0, 0.2, 0.2, 0.4, 0.4])
    model(torch.zeros(1, 3, 64, 64), targets=targets)["total_loss"].backward()
    copy.deepcopy(model)
    model(torch.zeros(1, 3, 64, 64))
    copy.deepcopy(model)


def test_backbone_returns_b5_only_on_request():
    from libreyolo.models.yolo9.nn import LibreYOLO9Model

    model = LibreYOLO9Model(config="t", nb_classes=2).eval()
    x = torch.zeros(1, 3, 64, 64)
    with torch.no_grad():
        plain = model.backbone(x)
        with_b5 = model.backbone(x, return_b5=True)
    assert len(plain) == 3 and len(with_b5) == 4
    assert torch.equal(plain[2], with_b5[2])
    assert not hasattr(model.backbone, "last_b5")


def test_yolo9_config_recipe_defaults():
    from libreyolo.training.config import YOLO9Config

    cfg = YOLO9Config()
    assert cfg.max_labels == 300
    assert cfg.aux_weight == 0.25
    assert cfg.warmup_momentum == 0.8


def test_linear_scheduler_momentum_warmup():
    from libreyolo.training.scheduler import LinearLRScheduler

    sched = LinearLRScheduler(
        lr=0.01,
        iters_per_epoch=10,
        total_epochs=10,
        warmup_epochs=3,
        warmup_momentum=0.8,
        momentum=0.937,
    )
    assert sched.update_momentum(0) == pytest.approx(0.8)
    assert sched.update_momentum(30) == pytest.approx(0.937)
    mid = sched.update_momentum(15)
    assert 0.8 < mid < 0.937

    plain = LinearLRScheduler(lr=0.01, iters_per_epoch=10, total_epochs=10)
    assert plain.update_momentum(1) is None


def test_rebuild_for_new_classes_updates_aux_head():
    from libreyolo.models.yolo9.model import LibreYOLO9

    wrapper = LibreYOLO9(None, size="t", nb_classes=80, device="cpu")
    wrapper.model.enable_aux(0.25)
    assert wrapper.model.aux_head.nc == 80
    wrapper._rebuild_for_new_classes(5)
    assert wrapper.model.head.nc == 5
    assert wrapper.model.aux_head.nc == 5
    assert wrapper.model.aux_head.cv3[0][-1].out_channels == 5


def test_ddp_bootstrap_merges_aux_and_keeps_flat_dict(tmp_path):
    from libreyolo.models.yolo9.nn import LibreYOLO9Model
    from libreyolo.training.ddp_spawn import (
        _bootstrap_state_dict,
        _build_init_kw,
    )
    from libreyolo.utils.serialization import wrap_libreyolo_checkpoint

    raw = LibreYOLO9Model(config="t", nb_classes=2)
    raw.enable_aux(0.25)
    marker = raw.aux.spp.cv1.conv.weight.detach().clone()
    ckpt = wrap_libreyolo_checkpoint(
        raw.state_dict(),
        model_family="yolo9",
        size="t",
        task="detect",
        nc=2,
        names={0: "a", 1: "b"},
        imgsz=640,
        letterbox_pad="center",
    )
    path = tmp_path / "official.pt"
    torch.save(ckpt, path)

    infer = LibreYOLO9Model(config="t", nb_classes=2)
    class _Wrap:
        pass

    parent = _Wrap()
    parent.model = infer
    parent.model_path = str(path)
    parent.letterbox_pad = "center"
    parent.size = "t"
    parent.nb_classes = 2
    parent.task = "detect"
    parent.reg_max = 16

    state = _bootstrap_state_dict(parent)
    assert all(torch.is_tensor(v) for v in state.values())
    assert "model" not in state
    assert "aux.spp.cv1.conv.weight" in state
    torch.testing.assert_close(state["aux.spp.cv1.conv.weight"], marker.cpu())

    parent.__class__ = type("LibreYOLO9", (), {"__init__": lambda self, *a, **k: None})
    # _build_init_kw inspects the real class; use a tiny stand-in with **kwargs.
    class _Fake:
        def __init__(self, model_path, size="t", **kwargs):
            pass

    fake = _Fake(None)
    fake.size = "t"
    fake.nb_classes = 2
    fake.task = "detect"
    fake.reg_max = 16
    fake.letterbox_pad = "center"
    init_kw = _build_init_kw(fake)
    assert init_kw["letterbox_pad"] == "center"


def test_resolve_aux_weight_keeps_zero():
    from libreyolo.models.yolo9.model import resolve_aux_weight

    assert resolve_aux_weight(None) == 0.25
    assert resolve_aux_weight(0) == 0.0
    assert resolve_aux_weight(0.0) == 0.0
    assert resolve_aux_weight(0.25) == 0.25


def test_aux_weight_zero_does_not_attach_pgi(tmp_path):
    from libreyolo.models.yolo9.model import LibreYOLO9
    from libreyolo.models.yolo9.nn import LibreYOLO9Model
    from libreyolo.utils.serialization import wrap_libreyolo_checkpoint

    raw = LibreYOLO9Model(config="t", nb_classes=2)
    raw.enable_aux(0.25)
    ckpt = wrap_libreyolo_checkpoint(
        raw.state_dict(),
        model_family="yolo9",
        size="t",
        task="detect",
        nc=2,
        names={0: "a", 1: "b"},
        imgsz=640,
    )
    path = tmp_path / "with_aux.pt"
    torch.save(ckpt, path)

    loaded = LibreYOLO9(str(path), size="t", nb_classes=2, device="cpu")
    assert loaded.model.aux is None
    n = loaded._maybe_enable_aux_from_path(str(path), aux_weight=0)
    assert n == 0
    assert loaded.model.aux is None


def test_disable_aux_drops_the_branch():
    from libreyolo.models.yolo9.nn import LibreYOLO9Model

    model = LibreYOLO9Model(config="t", nb_classes=2).enable_aux(0.25)
    model.disable_aux()
    assert model.aux is None and model.aux_head is None and model.aux_weight == 0.0
    assert not any(k.startswith(("aux.", "aux_head.")) for k in model.state_dict())


def test_cuda_graph_spec_covers_pgi_when_attached():
    from types import SimpleNamespace

    from libreyolo.models.yolo9.nn import LibreYOLO9Model
    from libreyolo.models.yolo9.trainer import YOLO9Trainer

    model = LibreYOLO9Model(config="t", nb_classes=2)
    host = SimpleNamespace(
        wrapper_model=SimpleNamespace(task="detect"), model=model
    )
    plain = YOLO9Trainer.cuda_graph_train_spec(host)
    assert plain is not None
    model.enable_aux(0.25)
    pgi = YOLO9Trainer.cuda_graph_train_spec(host)
    assert pgi is not None and pgi.network.module is not model
