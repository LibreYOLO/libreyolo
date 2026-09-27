"""Unit tests for the PP-LiteSeg semantic family (no weights required)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from libreyolo import LibrePPLiteSeg
from libreyolo.models.ppliteseg.model import CITYSCAPES_NAMES, preprocess_numpy
from libreyolo.models.ppliteseg.nn import SIZE_CONFIGS, STRIDE, LibrePPLiteSegNet

pytestmark = [pytest.mark.unit, pytest.mark.ppliteseg]

ALL_SIZES = ("t50", "b50", "t75", "b75")


def _tiny_net(size: str = "t50", nc: int = 3) -> LibrePPLiteSegNet:
    return LibrePPLiteSegNet(size=size, num_classes=nc, use_aux_heads=True)


@pytest.mark.parametrize("size", ALL_SIZES)
def test_size_configs_are_rectangular_and_stride_aligned(size):
    height, width = SIZE_CONFIGS[size]["imgsz"]
    assert (height, width) == LibrePPLiteSeg.INPUT_SIZES[size]
    assert height % STRIDE == 0 and width % STRIDE == 0
    assert width > height, "every PP-LiteSeg canvas is landscape; H and W must not swap"
    expected = (512, 1024) if size.endswith("50") else (768, 1536)
    assert (height, width) == expected


def test_train_and_val_geometries_stay_distinct():
    # The 75 recipe trains on a square crop and validates on a rectangle.
    for size in ("t50", "b50"):
        assert SIZE_CONFIGS[size]["train_crop"] == (512, 1024)
        assert SIZE_CONFIGS[size]["imgsz"] == (512, 1024)
    for size in ("t75", "b75"):
        assert SIZE_CONFIGS[size]["train_crop"] == (768, 768)
        assert SIZE_CONFIGS[size]["imgsz"] == (768, 1536)


@pytest.mark.parametrize("size", ALL_SIZES)
def test_forward_shapes_and_aux_gating(size):
    net = _tiny_net(size)
    x = torch.zeros(1, 3, 128, 256)
    net.eval()
    with torch.no_grad():
        out = net(x)
    assert torch.is_tensor(out), "eval forward must return main logits only"
    assert out.shape == (1, 3, 128, 256)

    net.train()
    with torch.no_grad():
        out = net(torch.zeros(2, 3, 128, 256))
    assert isinstance(out, tuple) and len(out) == 4
    for tensor in out:
        assert tensor.shape == (2, 3, 128, 256)


def test_architecture_channel_contract():
    tiny = _tiny_net("t50")
    base = _tiny_net("b50")
    assert tiny.encoder.backbone.out_widths == [256, 512, 1024]
    assert [c.seq.conv.out_channels for c in tiny.encoder.proj_convs] == [64, 128, 128]
    assert [c.seq.conv.out_channels for c in base.encoder.proj_convs] == [96, 128, 128]
    assert tiny.encoder.context_module.pool_sizes == [1, 2, 4]
    assert tiny.encoder.context_module.out_channels == 128
    # UAFM spatial attention is a 4 -> 2 -> 1 stack on the four reduced maps.
    atten = tiny.decoder.up_stages[0].conv_atten
    assert atten[0].seq.conv.in_channels == 4 and atten[0].seq.conv.out_channels == 2
    assert atten[1].seq.conv.out_channels == 1
    assert not hasattr(atten[1].seq, "act"), "the second attention conv has no ReLU"
    assert [s.conv_out.seq.conv.out_channels for s in tiny.decoder.up_stages] == [128, 64, 32]
    assert [s.conv_out.seq.conv.out_channels for s in base.decoder.up_stages] == [128, 96, 64]


def test_stdc_block_counts_differ_by_backbone():
    tiny = _tiny_net("t50").encoder.backbone
    base = _tiny_net("b50").encoder.backbone
    assert len(tiny.stages["block_s8"]) == 2
    assert len(base.stages["block_s8"]) == 4
    assert len(base.stages["block_s16"]) == 5
    assert len(base.stages["block_s32"]) == 3


def test_checkpoint_detection_is_conservative():
    state = _tiny_net("t50", nc=19).state_dict()
    assert LibrePPLiteSeg.can_load(state)
    assert LibrePPLiteSeg.detect_backbone(state) == "stdc1"
    assert LibrePPLiteSeg.detect_backbone(_tiny_net("b50", nc=19).state_dict()) == "stdc2"
    assert LibrePPLiteSeg.detect_nb_classes(state) == 19
    # The 50/75 recipes are architecturally identical, so size must not be guessed.
    assert LibrePPLiteSeg.detect_size(state) is None
    # A DDP-prefixed dict is recognized without pre-stripping.
    prefixed = {f"module.{k}": v for k, v in state.items()}
    assert LibrePPLiteSeg.can_load(prefixed)
    assert LibrePPLiteSeg.detect_backbone(prefixed) == "stdc1"


def test_can_load_rejects_foreign_checkpoints():
    assert not LibrePPLiteSeg.can_load({"backbone.conv1.weight": torch.zeros(1)})
    assert not LibrePPLiteSeg.can_load({"decode_head.classifier.weight": torch.zeros(1)})
    # A partial PP-LiteSeg-shaped dict without the UAFM/SPPM evidence is refused.
    state = _tiny_net("t50").state_dict()
    partial = {k: v for k, v in state.items() if not k.startswith("decoder.")}
    assert not LibrePPLiteSeg.can_load(partial)


def test_detect_nb_classes_rejects_disagreeing_heads():
    state = dict(_tiny_net("t50", nc=19).state_dict())
    state["aux_heads.1.0.seg_head.2.weight"] = torch.zeros(7, 64, 1, 1)
    with pytest.raises(RuntimeError, match="inconsistent"):
        LibrePPLiteSeg.detect_nb_classes(state)


def test_preprocess_direct_resizes_without_padding():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (300, 700, 3), dtype=np.uint8)
    chw, ratio = preprocess_numpy(img, (512, 1024))
    assert chw.shape == (3, 512, 1024)
    assert chw.dtype == np.float32
    assert 0.0 <= chw.min() and chw.max() <= 1.0
    # Direct resize leaves no padded region, so nothing has to be trimmed later.
    assert ratio == 1.0


def test_preprocess_rejects_off_stride_canvas():
    model = LibrePPLiteSeg(size="t50", device="cpu")
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="divisible"):
        model._preprocess(img, input_size=(500, 1000))


def test_normalization_is_applied_once_inside_forward():
    net = _tiny_net("t50")
    x = torch.rand(1, 3, 64, 128)
    normalized = net.normalize(x)
    expected = (x - torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)) / torch.tensor(
        [0.229, 0.224, 0.225]
    ).view(1, 3, 1, 1)
    assert torch.equal(normalized, expected)
    # The mean/std buffers are non-persistent: they must not appear in a
    # checkpoint, or strict loading of an upstream state dict would break.
    assert "pixel_mean" not in net.state_dict()
    assert "pixel_std" not in net.state_dict()


def test_replace_num_classes_rebuilds_main_and_every_aux_head():
    net = _tiny_net("t50", nc=19)
    net.replace_num_classes(5)
    state = net.state_dict()
    assert state["seg_head.0.seg_head.2.weight"].shape[0] == 5
    for index in range(3):
        assert state[f"aux_heads.{index}.0.seg_head.2.weight"].shape[0] == 5
    assert net.num_classes == 5


def test_wrapper_rebuild_for_new_classes_updates_names_and_heads():
    model = LibrePPLiteSeg(size="t50", device="cpu")
    assert model.names == CITYSCAPES_NAMES
    model._rebuild_for_new_classes(4)
    assert model.nb_classes == 4
    assert len(model.names) == 4
    state = model.model.state_dict()
    assert state["seg_head.0.seg_head.2.weight"].shape[0] == 4


def test_postprocess_restores_original_canvas_and_emits_no_ignore():
    model = LibrePPLiteSeg(size="t50", nb_classes=19, device="cpu")
    logits = torch.randn(1, 19, 64, 128)
    result = model._postprocess(logits, 0.25, 0.45, original_size=(333, 211))
    mask = result["semantic"]
    assert mask.shape == (211, 333)  # original_size is (w, h)
    assert int(mask.min()) >= 0 and int(mask.max()) < 19


def test_postprocess_accepts_the_training_tuple():
    model = LibrePPLiteSeg(size="t50", nb_classes=19, device="cpu")
    main = torch.randn(1, 19, 32, 64)
    aux = [torch.randn(1, 19, 32, 64) for _ in range(3)]
    from_tuple = model._postprocess(tuple([main] + aux), 0.25, 0.45, original_size=(64, 32))
    from_main = model._postprocess(main, 0.25, 0.45, original_size=(64, 32))
    assert torch.equal(from_tuple["semantic"], from_main["semantic"])


def test_all_one_class_and_tied_logits_are_deterministic():
    model = LibrePPLiteSeg(size="t50", nb_classes=19, device="cpu")
    single = torch.full((1, 19, 16, 32), -5.0)
    single[:, 7] = 5.0
    mask = model._postprocess(single, 0.25, 0.45, original_size=(32, 16))["semantic"]
    assert torch.all(mask == 7)
    tied = torch.zeros(1, 19, 16, 32)
    tied_mask = model._postprocess(tied, 0.25, 0.45, original_size=(32, 16))["semantic"]
    assert torch.all(tied_mask == 0), "argmax breaks ties toward the lowest class ID"


def test_export_copy_drops_aux_heads_without_touching_the_live_model():
    model = LibrePPLiteSeg(size="t50", device="cpu")
    export_model = model._export_model((512, 1024))
    assert not hasattr(export_model, "aux_heads")
    assert hasattr(model.model, "aux_heads"), "the trainable model must stay intact"
    export_model.eval()
    with torch.no_grad():
        out = export_model(torch.zeros(1, 3, 512, 1024))
    assert torch.is_tensor(out) and out.shape == (1, 19, 512, 1024)


def test_export_copy_caches_sppm_kernels_per_rectangle():
    model = LibrePPLiteSeg(size="t50", device="cpu")
    small = model._export_model((512, 1024))
    large = model._export_model((768, 1536))
    small_kernels = [branch[0].kernel_size for branch in small.encoder.context_module.branches]
    large_kernels = [branch[0].kernel_size for branch in large.encoder.context_module.branches]
    assert small_kernels == [[16, 32], [8, 16], [4, 8]]
    assert large_kernels == [[24, 48], [12, 24], [6, 12]]
    assert small_kernels != large_kernels, "a 512x1024 kernel must not be reused at 768x1536"
    # And the live model still holds adaptive pooling, unmodified.
    assert isinstance(
        model.model.encoder.context_module.branches[0][0], torch.nn.AdaptiveAvgPool2d
    )


def test_sppm_fixed_pooling_matches_adaptive_pooling():
    net = _tiny_net("t50")
    net.eval()
    feature = torch.randn(1, 1024, 16, 32)
    with torch.no_grad():
        adaptive = net.encoder.context_module(feature)
    net.encoder.context_module.prep_model_for_conversion((512, 1024))
    with torch.no_grad():
        fixed = net.encoder.context_module(feature)
    assert torch.allclose(adaptive, fixed, atol=1e-6)


def test_semantic_recipe_attributes_follow_the_size():
    t50 = LibrePPLiteSeg(size="t50", device="cpu")
    t75 = LibrePPLiteSeg(size="t75", device="cpu")
    assert t50.semantic_scale_jitter == (0.125, 1.5)
    assert t75.semantic_scale_jitter == (0.25, 1.75)
    assert t50.semantic_train_imgsz == (512, 1024)
    assert t75.semantic_train_imgsz == (768, 768)
    assert t75.semantic_val_imgsz == (768, 1536)
    assert LibrePPLiteSeg.semantic_resize_mode == "rescale_crop"
    assert LibrePPLiteSeg.semantic_hsv_prob == 0.0


def test_family_metadata_and_task_contract():
    assert LibrePPLiteSeg.FAMILY == "ppliteseg"
    assert LibrePPLiteSeg.FILENAME_PREFIX == "LibrePPLiteSeg"
    assert LibrePPLiteSeg.SUPPORTED_TASKS == ("semantic",)
    assert LibrePPLiteSeg.DEFAULT_TASK == "semantic"
    assert LibrePPLiteSeg.REQUIRE_TASK_SUFFIX is True
    with pytest.raises(ValueError, match="semantic"):
        LibrePPLiteSeg(size="t50", task="detect", device="cpu")


def test_download_notice_names_the_non_commercial_restriction():
    notice = LibrePPLiteSeg.get_download_notice("LibrePPLiteSegt50-sem.pt", "https://example")
    assert "NON-COMMERCIAL" in notice
    assert "cityscapes-dataset.com/license" in notice
    assert "not to LibreYOLO's MIT code" in notice


def test_family_is_enrolled_in_the_model_registry():
    from libreyolo.models.registry import MODEL_GROUPS

    assert MODEL_GROUPS["ppliteseg"] == "g2"
