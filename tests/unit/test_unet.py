"""Unit tests for the U-Net semantic family (no weights required)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from libreyolo import LibreUNet
from libreyolo.models.unet.loss import UNetLoss
from libreyolo.models.unet.model import CITYSCAPES_NAMES
from libreyolo.models.unet.nn import SIZE_CONFIGS, STRIDE, LibreUNetNet
from libreyolo.models.unet.utils import preprocess_numpy
from libreyolo.training.config import UNetConfig

pytestmark = [pytest.mark.unit, pytest.mark.unet]


def _tiny_net(nc: int = 3) -> LibreUNetNet:
    return LibreUNetNet(size="s", num_classes=nc)


def test_size_config_is_rectangular_and_stride_aligned():
    # Evaluation canvas is the whole Cityscapes frame (mmseg test pipeline
    # Resize(2048, 1024) + mode='whole'); 512x1024 is only the train crop.
    height, width = SIZE_CONFIGS["s"]["imgsz"]
    assert (height, width) == LibreUNet.INPUT_SIZES["s"] == (1024, 2048)
    assert height % STRIDE == 0 and width % STRIDE == 0
    assert width > height
    assert SIZE_CONFIGS["s"]["train_crop"] == (512, 1024)
    assert SIZE_CONFIGS["s"]["rescale_range"] == (0.5, 2.0)
    assert SIZE_CONFIGS["s"]["base_channels"] == 64


def test_recipe_accessors_split_train_crop_from_eval_canvas():
    model = LibreUNet(size="s", device="cpu")
    assert model.semantic_train_imgsz == (512, 1024)
    assert model.semantic_val_imgsz == (1024, 2048)
    assert model.semantic_scale_jitter == (0.5, 2.0)
    assert model._get_input_size() == (1024, 2048)


def test_forward_shapes_and_aux_gating():
    net = _tiny_net()
    x = torch.zeros(1, 3, 64, 128)
    net.eval()
    with torch.no_grad():
        out = net(x)
    assert torch.is_tensor(out)
    assert out.shape == (1, 3, 64, 128)

    net.train()
    with torch.no_grad():
        out = net(torch.zeros(2, 3, 64, 128))
    assert isinstance(out, tuple) and len(out) == 2
    assert out[0].shape == (2, 3, 64, 128)
    assert out[1].shape == (2, 3, 64, 128)


def test_architecture_channel_contract():
    net = _tiny_net(nc=19)
    assert net.decode_head.conv_seg.out_channels == 19
    assert net.auxiliary_head.conv_seg.out_channels == 19
    assert net.decode_head.convs[0].conv.in_channels == 64
    assert net.auxiliary_head.convs[0].conv.in_channels == 128
    deepest = net.backbone.encoder[4][1].convs[1].conv
    assert deepest.out_channels == 1024
    stem = net.backbone.encoder[0][0].convs[0].conv
    assert stem.in_channels == 3 and stem.out_channels == 64


def test_checkpoint_detection():
    state = _tiny_net(nc=19).state_dict()
    assert LibreUNet.can_load(state)
    assert LibreUNet.detect_size(state) == "s"
    assert LibreUNet.detect_nb_classes(state) == 19
    prefixed = {f"module.{key}": value for key, value in state.items()}
    assert LibreUNet.can_load(prefixed)
    assert LibreUNet.detect_size(prefixed) == "s"
    assert LibreUNet.detect_nb_classes(prefixed) == 19


def test_can_load_rejects_foreign_and_partial_checkpoints():
    assert not LibreUNet.can_load({"backbone.conv1.weight": torch.zeros(1)})
    assert not LibreUNet.can_load({"decode_head.conv_seg.weight": torch.zeros(19, 64, 1, 1)})
    state = _tiny_net(nc=19).state_dict()
    partial = {key: value for key, value in state.items() if not key.startswith("backbone.decoder.")}
    assert not LibreUNet.can_load(partial)


def test_detect_nb_classes_rejects_disagreeing_heads():
    state = dict(_tiny_net(nc=19).state_dict())
    state["auxiliary_head.conv_seg.weight"] = torch.zeros(7, 64, 1, 1)
    with pytest.raises(RuntimeError, match="inconsistent"):
        LibreUNet.detect_nb_classes(state)


def test_preprocess_stretch_resizes_without_padding():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (300, 700, 3), dtype=np.uint8)
    chw, ratio = preprocess_numpy(img, (512, 1024))
    assert chw.shape == (3, 512, 1024)
    assert chw.dtype == np.float32
    assert 0.0 <= chw.min() and chw.max() <= 1.0
    assert ratio == 1.0
    default_chw, _ = preprocess_numpy(img)
    assert default_chw.shape == (3, 1024, 2048)


def test_preprocess_is_identity_on_a_native_cityscapes_frame():
    """Whole-frame inference must hand the network the source pixels unchanged,
    exactly as the upstream test pipeline does on 2048x1024 Cityscapes images."""
    rng = np.random.default_rng(1)
    img = rng.integers(0, 256, (1024, 2048, 3), dtype=np.uint8)
    chw, _ = preprocess_numpy(img, (1024, 2048))
    assert np.array_equal(chw, img.astype(np.float32).transpose(2, 0, 1) / 255.0)


def test_internal_standardization_matches_mmseg_on_uint8_values():
    """(x / 255) * 255 must round-trip every uint8 value exactly so our
    [0, 1] input contract yields the same standardized tensor mmseg computes
    from 0-255 pixels."""
    net = _tiny_net()
    values = torch.arange(256, dtype=torch.float32)
    x01 = (values / 255.0).view(1, 1, 16, 16).expand(1, 3, 16, 16)
    mean = net._mean
    std = net._std
    expected = (values.view(1, 1, 16, 16).expand(1, 3, 16, 16) - mean) / std
    assert torch.equal(net._normalize(x01), expected)


def test_preprocess_rejects_off_stride_canvas():
    model = LibreUNet(size="s", device="cpu")
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="divisible"):
        model._preprocess(img, input_size=(500, 1000))


def test_normalization_buffers_are_non_persistent():
    net = _tiny_net()
    state = net.state_dict()
    assert "_mean" not in state
    assert "_std" not in state


def test_replace_num_classes_rebuilds_both_heads():
    net = _tiny_net(nc=19)
    net.replace_num_classes(5)
    state = net.state_dict()
    assert state["decode_head.conv_seg.weight"].shape[0] == 5
    assert state["auxiliary_head.conv_seg.weight"].shape[0] == 5
    assert net.num_classes == 5


def test_wrapper_rebuild_for_new_classes_updates_names_and_heads():
    model = LibreUNet(size="s", device="cpu")
    assert model.names == CITYSCAPES_NAMES
    model._rebuild_for_new_classes(4)
    assert model.nb_classes == 4
    assert len(model.names) == 4
    state = model.model.state_dict()
    assert state["decode_head.conv_seg.weight"].shape[0] == 4


def test_postprocess_restores_original_canvas():
    model = LibreUNet(size="s", nb_classes=19, device="cpu")
    logits = torch.randn(1, 19, 64, 128)
    result = model._postprocess(logits, 0.25, 0.45, original_size=(333, 211))
    mask = result["semantic"]
    assert mask.shape == (211, 333)
    assert int(mask.min()) >= 0 and int(mask.max()) < 19


def test_postprocess_accepts_the_training_tuple():
    model = LibreUNet(size="s", nb_classes=19, device="cpu")
    main = torch.randn(1, 19, 32, 64)
    aux = torch.randn(1, 19, 32, 64)
    from_tuple = model._postprocess((main, aux), 0.25, 0.45, original_size=(64, 32))
    from_main = model._postprocess(main, 0.25, 0.45, original_size=(64, 32))
    assert torch.equal(from_tuple["semantic"], from_main["semantic"])


def test_all_one_class_and_tied_logits_are_deterministic():
    model = LibreUNet(size="s", nb_classes=19, device="cpu")
    single = torch.full((1, 19, 16, 32), -5.0)
    single[:, 7] = 5.0
    mask = model._postprocess(single, 0.25, 0.45, original_size=(32, 16))["semantic"]
    assert torch.all(mask == 7)
    tied = torch.zeros(1, 19, 16, 32)
    tied_mask = model._postprocess(tied, 0.25, 0.45, original_size=(32, 16))["semantic"]
    assert torch.all(tied_mask == 0)


def test_loss_weights_aux_at_point_four():
    criterion = UNetLoss(aux_weight=0.4)
    main = torch.zeros(1, 3, 8, 8)
    aux = torch.zeros(1, 3, 8, 8)
    main[:, 0] = 10.0
    aux[:, 1] = 10.0
    target = torch.zeros(1, 8, 8, dtype=torch.long)
    parts = criterion((main, aux), target)
    assert parts["loss_ce"].item() < 1e-4
    assert parts["loss_aux"].item() > 1.0
    assert parts["loss"].item() == pytest.approx(
        parts["loss_ce"].item() + 0.4 * parts["loss_aux"].item(), rel=1e-5
    )


def test_config_matches_the_source_recipe():
    config = UNetConfig()
    assert config.optimizer == "sgd"
    assert config.lr0 == 0.01
    assert config.momentum == 0.9
    assert config.weight_decay == 5e-4
    assert config.nesterov is False
    assert config.scheduler == "poly"
    assert config.aux_weight == 0.4
    assert config.epochs == 160
    assert config.batch == 4
    assert config.amp is False
    assert config.imgsz == (512, 1024)


def test_family_metadata_and_task_contract():
    assert LibreUNet.FAMILY == "unet"
    assert LibreUNet.FILENAME_PREFIX == "LibreUNet"
    assert LibreUNet.SUPPORTED_TASKS == ("semantic",)
    assert LibreUNet.DEFAULT_TASK == "semantic"
    assert LibreUNet.REQUIRE_TASK_SUFFIX is True
    assert LibreUNet.semantic_resize_mode == "rescale_crop"
    assert LibreUNet.semantic_imgsz_divisor == 16
    with pytest.raises(ValueError, match="semantic"):
        LibreUNet(size="s", task="detect", device="cpu")


def test_download_notice_names_the_non_commercial_restriction():
    notice = LibreUNet.get_download_notice("LibreUNets-sem.pt", "https://example")
    assert "NON-COMMERCIAL" in notice
    assert "cityscapes-dataset.com/license" in notice
    assert "not to LibreYOLO's MIT code" in notice


def test_family_is_enrolled_in_the_model_registry():
    from libreyolo.models.registry import MODEL_GROUPS

    assert MODEL_GROUPS["unet"] == "g2"


def test_cli_alias_resolves_to_the_suffixed_filename():
    from libreyolo.cli.config import is_known_weight_filename, resolve_model_name

    assert resolve_model_name("unet-s") == "LibreUNets-sem.pt"
    assert resolve_model_name("unet-s-sem") == "LibreUNets-sem.pt"
    assert is_known_weight_filename("LibreUNets-sem.pt") is True


def test_download_url_keeps_the_task_suffix():
    assert LibreUNet.get_download_url("LibreUNets-sem.pt") == (
        "https://huggingface.co/LibreYOLO/LibreUNets-sem/resolve/main/LibreUNets-sem.pt"
    )
