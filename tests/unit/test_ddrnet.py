"""Unit tests for the DDRNet semantic family (no weights or network required)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch

from libreyolo import LibreDDRNet, LibrePIDNet
from libreyolo.models.ddrnet.convert import (
    IMAGENET_BACKBONES,
    convert_upstream,
    imagenet_backbone_state,
    is_ddrnet_state_dict,
)
from libreyolo.models.ddrnet.loss import DDRNetLoss, OhemCrossEntropy
from libreyolo.models.ddrnet.model import CITYSCAPES_NAMES, preprocess_numpy
from libreyolo.models.ddrnet.nn import (
    ARCH_CONFIGS,
    SIZE_CONFIGS,
    STRIDE,
    LibreDDRNetNet,
)
from libreyolo.models.ddrnet.trainer import DDRNetTrainer
from libreyolo.models.pidnet.nn import LibrePIDNetNet
from libreyolo.training.config import DDRNetConfig
from libreyolo.training.scheduler import PolyLRScheduler

pytestmark = [pytest.mark.unit, pytest.mark.ddrnet]

SIZES = ("s", "m", "l")

# sha256 over the sorted "key (shape)" manifest of the upstream layout:
# s/m from the official Cityscapes checkpoints (``model.`` prefix stripped,
# ``loss.criterion.weight`` dropped), l from the official DDRNet_39
# ``DualResNet(..., augment=True)`` graph at commit de0db317. Pinning upstream's
# layout, not our own, keeps a key rename from staying self-consistent while
# breaking strict loads of the official files.
UPSTREAM_MANIFESTS = {
    "s": (346, "a0fa9b6692377dd0891782b05affbaa1316fa2ebb132dd9dab91833cdda5b405"),
    "m": (346, "d12ca50b1dbb2123099606efa613148e41b7868e040eaf718017f06dcea1a649"),
    "l": (514, "000f7d629ed1ad6f8847771159101cac35971e5f3de89421951b210d48929f2b"),
}


def _manifest(state: dict) -> tuple[int, str]:
    text = "\n".join(f"{key} {tuple(value.shape)}" for key, value in sorted(state.items()))
    return len(state), hashlib.sha256(text.encode()).hexdigest()


def _imagenet_like_state(size: str) -> dict:
    """The official ImageNet classifier layout: shared tensors + its own head."""
    state = {
        key: value.clone()
        for key, value in LibreDDRNetNet(size, 19).state_dict().items()
        if not key.startswith(("spp.", "final_layer.", "seghead_extra."))
    }
    state["down5.0.weight"] = torch.zeros(1)
    state["last_layer.0.weight"] = torch.zeros(1)
    state["linear.weight"] = torch.zeros(1000, 8)
    state["linear.bias"] = torch.zeros(1000)
    return state


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("size", SIZES)
def test_state_dict_layout_matches_the_upstream_manifest(size):
    assert _manifest(LibreDDRNetNet(size, 19).state_dict()) == UPSTREAM_MANIFESTS[size]


@pytest.mark.parametrize("size", SIZES)
def test_forward_shapes_and_aux_gating(size):
    net = LibreDDRNetNet(size, num_classes=5).eval()
    with torch.no_grad():
        out = net(torch.zeros(1, 3, 64, 128))
    assert torch.is_tensor(out)
    assert out.shape == (1, 5, 8, 16)

    net.train()
    with torch.no_grad():
        out = net(torch.rand(2, 3, 64, 128))
    assert isinstance(out, tuple) and len(out) == 2
    assert out[0].shape == out[1].shape == (2, 5, 8, 16)


def test_size_configs_match_the_upstream_factories():
    assert ARCH_CONFIGS["s"].planes == 32 and ARCH_CONFIGS["s"].head_planes == 64
    assert ARCH_CONFIGS["m"].planes == 64 and ARCH_CONFIGS["m"].head_planes == 128
    assert ARCH_CONFIGS["l"].layers == (3, 4, 6, 3) and ARCH_CONFIGS["l"].split_stage3
    for size in SIZES:
        assert SIZE_CONFIGS[size]["imgsz"] == LibreDDRNet.INPUT_SIZES[size] == (1024, 2048)
        assert SIZE_CONFIGS[size]["train_crop"] == (1024, 1024)
        assert SIZE_CONFIGS[size]["rescale_range"] == (0.5, 2.0)


@pytest.mark.parametrize("size,millions", [("s", 5.7), ("m", 20.3), ("l", 32.7)])
def test_parameter_counts_match_the_paper_plus_aux_head(size, millions):
    count = sum(p.numel() for p in LibreDDRNetNet(size, 19).parameters()) / 1e6
    assert count == pytest.approx(millions, abs=0.1)


def test_normalization_buffers_are_non_persistent():
    state = LibreDDRNetNet("s", 19).state_dict()
    assert "pixel_mean" not in state and "pixel_std" not in state


def test_replace_num_classes_rebuilds_both_heads_and_keeps_the_backbone():
    net = LibreDDRNetNet("s", 19)
    before = {k: v.clone() for k, v in net.state_dict().items() if not k.endswith("conv2.weight")}
    net.replace_num_classes(4)
    state = net.state_dict()
    assert state["final_layer.conv2.weight"].shape[0] == 4
    assert state["seghead_extra.conv2.weight"].shape[0] == 4
    assert torch.equal(state["conv1.0.weight"], before["conv1.0.weight"])
    assert torch.equal(state["spp.scale0.2.weight"], before["spp.scale0.2.weight"])


# ---------------------------------------------------------------------------
# Checkpoint recognition
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("size", SIZES)
def test_checkpoint_detection_on_native_and_upstream_layouts(size):
    state = LibreDDRNetNet(size, 19).state_dict()
    upstream = {f"model.{k}": v for k, v in state.items()}
    upstream["loss.criterion.weight"] = torch.ones(19)
    for layout in (state, upstream, {f"module.{k}": v for k, v in state.items()}):
        assert LibreDDRNet.can_load(layout)
        assert LibreDDRNet.detect_size(layout) == size
        assert LibreDDRNet.detect_nb_classes(layout) == 19


def test_convert_upstream_strips_the_wrapper_and_drops_the_loss_buffer():
    state = LibreDDRNetNet("s", 19).state_dict()
    upstream = {f"model.{k}": v for k, v in state.items()}
    upstream["loss.criterion.weight"] = torch.ones(19)
    converted = convert_upstream(upstream)
    assert set(converted) == set(state)
    assert LibreDDRNet.convert_upstream_state_dict(upstream).keys() == state.keys()


@pytest.mark.parametrize("size", SIZES)
def test_ddrnet_and_pidnet_reject_each_other(size):
    """Both families share the stem, residual stages, DAPPM and head names."""
    ddrnet = LibreDDRNetNet(size, 19).state_dict()
    pidnet = LibrePIDNetNet(size, 19).state_dict()
    assert LibreDDRNet.can_load(ddrnet) and not LibrePIDNet.can_load(ddrnet)
    assert LibrePIDNet.can_load(pidnet) and not LibreDDRNet.can_load(pidnet)
    assert LibreDDRNet.convert_upstream_state_dict(pidnet) is None


def test_imagenet_classifier_is_not_a_semantic_checkpoint():
    assert not is_ddrnet_state_dict(_imagenet_like_state("s"))
    assert not LibreDDRNet.can_load({"conv1.0.weight": torch.zeros(1)})


def test_detect_nb_classes_rejects_disagreeing_heads():
    state = dict(LibreDDRNetNet("s", 19).state_dict())
    state["seghead_extra.conv2.weight"] = torch.zeros(7, 64, 1, 1)
    with pytest.raises(RuntimeError, match="inconsistent"):
        LibreDDRNet.detect_nb_classes(state)


def test_unofficial_files_get_no_weight_license(tmp_path):
    path = tmp_path / "fine_tune.pth"
    path.write_bytes(b"not an official artifact")
    metadata = LibreDDRNet.upstream_checkpoint_metadata({}, source=path)
    assert "weight_license" not in metadata
    assert (metadata["imgsz_h"], metadata["imgsz_w"]) == (1024, 2048)


def test_upstream_checkpoint_loads_and_round_trips(tmp_path):
    """A raw upstream file loads through the wrapper; a converted one too."""
    net = LibreDDRNetNet("s", 19)
    upstream = {f"model.{k}": v for k, v in net.state_dict().items()}
    upstream["loss.criterion.weight"] = torch.ones(19)
    raw_path = tmp_path / "best_val_smaller.pth"
    torch.save(upstream, raw_path)

    model = LibreDDRNet(str(raw_path), device="cpu")
    assert model.size == "s" and model.nb_classes == 19
    assert model.names == CITYSCAPES_NAMES
    assert model.weight_license is None  # not an official digest
    for key, value in net.state_dict().items():
        assert torch.equal(model.model.state_dict()[key], value)


def test_weights_for_l_are_reported_unpublished():
    with pytest.raises(FileNotFoundError, match="DDRNet-39"):
        LibreDDRNet.get_download_url("LibreDDRNetl-sem.pt")
    assert LibreDDRNet.get_download_url("LibreDDRNets-sem.pt").endswith(
        "LibreYOLO/LibreDDRNets-sem/resolve/main/LibreDDRNets-sem.pt"
    )


# ---------------------------------------------------------------------------
# ImageNet backbone initialization
# ---------------------------------------------------------------------------


def test_imagenet_backbone_state_takes_exactly_the_shared_tensors():
    target = set(LibreDDRNetNet("s", 19).state_dict())
    taken = imagenet_backbone_state(_imagenet_like_state("s"), target)
    assert not any(k.startswith(("spp.", "final_layer.", "seghead_extra.")) for k in taken)
    assert not any(k.startswith(("down5.", "last_layer.", "linear.")) for k in taken)
    assert len(taken) == len(target) - sum(
        k.startswith(("spp.", "final_layer.", "seghead_extra.")) for k in target
    )


def test_imagenet_backbone_of_the_wrong_size_fails_loudly():
    target = set(LibreDDRNetNet("l", 19).state_dict())
    with pytest.raises(RuntimeError, match="ImageNet backbone"):
        imagenet_backbone_state(_imagenet_like_state("m"), target)


def test_pretrained_backbone_initializes_only_shared_tensors(tmp_path):
    source = _imagenet_like_state("s")
    source["conv1.0.weight"] = torch.full_like(source["conv1.0.weight"], 0.25)
    path = tmp_path / "DDRNet23s_imagenet.pth"
    torch.save(source, path)

    model = LibreDDRNet(size="s", device="cpu", pretrained_backbone=str(path))
    assert model._imagenet_init_pending is False
    assert torch.all(model.model.conv1[0].weight == 0.25)


def test_save_keeps_weight_terms(tmp_path):
    model = LibreDDRNet(size="s", device="cpu")
    model.weight_license = "MIT"
    model.weight_dataset = "Cityscapes"
    path = tmp_path / "LibreDDRNets-sem.pt"
    model.save(str(path))
    reopened = LibreDDRNet(str(path), device="cpu")
    assert reopened.weight_license == "MIT"
    assert reopened.weight_dataset == "Cityscapes"


def test_grayscale_checkpoint_reopens(tmp_path):
    from libreyolo.utils.input_channels import configure_channels, model_channels

    model = LibreDDRNet(size="s", device="cpu")
    configure_channels(model, 1, pretrained=True)
    path = tmp_path / "gray.pt"
    model.save(str(path))
    reopened = LibreDDRNet(str(path), device="cpu")
    assert model_channels(reopened) == 1
    assert torch.equal(reopened.model.conv1[0].weight, model.model.conv1[0].weight)


def test_grayscale_model_folds_the_imagenet_stem(tmp_path):
    from libreyolo.utils.input_channels import configure_channels

    source = _imagenet_like_state("s")
    path = tmp_path / "DDRNet23s_imagenet.pth"
    torch.save(source, path)
    model = LibreDDRNet(size="s", device="cpu")
    configure_channels(model, 1)
    model.load_imagenet_backbone(path)
    assert torch.allclose(
        model.model.conv1[0].weight, source["conv1.0.weight"].sum(dim=1, keepdim=True)
    )


def test_model_path_and_pretrained_backbone_are_mutually_exclusive(tmp_path):
    with pytest.raises(ValueError, match="not both"):
        LibreDDRNet(str(tmp_path / "x.pt"), pretrained_backbone=str(tmp_path / "y.pth"))


def test_imagenet_mirror_pins_upstream_files():
    assert {spec["filename"] for spec in IMAGENET_BACKBONES.values()} == {
        "DDRNet23s_imagenet.pth",
        "DDRNet23_imagenet.pth",
        "DDRNet39_imagenet.pth",
    }
    assert all(len(spec["sha256"]) == 64 for spec in IMAGENET_BACKBONES.values())


@pytest.fixture
def stub_training(monkeypatch):
    """Stop DDRNetTrainer before any real work."""
    calls = []

    def init(self, *args, **kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(DDRNetTrainer, "__init__", init)
    monkeypatch.setattr(DDRNetTrainer, "train", lambda self: {})
    return calls


def test_fresh_model_loads_the_imagenet_backbone_when_training_starts(
    tmp_path, monkeypatch, stub_training
):
    path = tmp_path / "DDRNet23s_imagenet.pth"
    torch.save(_imagenet_like_state("s"), path)
    requested = []

    def fake_path(cls, size, weights_dir="weights"):
        requested.append(size)
        return path

    monkeypatch.setattr(LibreDDRNet, "imagenet_backbone_path", classmethod(fake_path))
    model = LibreDDRNet(size="s", device="cpu")
    assert model._imagenet_init_pending is True
    model.train(data="unused.yaml", epochs=1, device="cpu")
    assert requested == ["s"]
    assert model._imagenet_init_pending is False
    assert stub_training and stub_training[-1]["imgsz"] == (1024, 1024)


def test_loaded_or_scratch_models_do_not_fetch_the_backbone(tmp_path, monkeypatch, stub_training):
    def fail(*args, **kwargs):
        raise AssertionError("ImageNet backbone must not be fetched")

    monkeypatch.setattr(LibreDDRNet, "imagenet_backbone_path", classmethod(fail))
    net = LibreDDRNetNet("s", 19)
    path = tmp_path / "ckpt.pth"
    torch.save({f"model.{k}": v for k, v in net.state_dict().items()}, path)
    LibreDDRNet(str(path), device="cpu").train(data="unused.yaml", epochs=1, device="cpu")

    fresh = LibreDDRNet(size="s", device="cpu")
    fresh.train(data="unused.yaml", epochs=1, device="cpu", pretrained=False)
    assert fresh._imagenet_init_pending is False


# ---------------------------------------------------------------------------
# Preprocess / postprocess
# ---------------------------------------------------------------------------


def test_preprocess_is_identity_on_a_native_cityscapes_frame():
    rng = np.random.default_rng(1)
    img = rng.integers(0, 256, (1024, 2048, 3), dtype=np.uint8)
    chw, ratio = preprocess_numpy(img)
    assert ratio == 1.0
    assert np.array_equal(chw, img.astype(np.float32).transpose(2, 0, 1) / 255.0)


def test_preprocess_stretch_resizes_without_padding():
    img = np.zeros((300, 700, 3), dtype=np.uint8)
    chw, _ = preprocess_numpy(img, (512, 1024))
    assert chw.shape == (3, 512, 1024) and chw.dtype == np.float32


def test_preprocess_rejects_off_stride_canvas():
    model = LibreDDRNet(size="s", device="cpu")
    with pytest.raises(ValueError, match="divisible"):
        model._preprocess(np.zeros((64, 64, 3), dtype=np.uint8), input_size=(500, 1000))


def test_postprocess_restores_original_canvas_and_ignores_the_aux_head():
    model = LibreDDRNet(size="s", device="cpu")
    main = torch.randn(1, 19, 16, 32)
    result = model._postprocess(main, 0.25, 0.45, original_size=(333, 211))
    assert result["semantic"].shape == (211, 333)
    from_tuple = model._postprocess((main, torch.randn_like(main)), 0.25, 0.45, original_size=(333, 211))
    assert torch.equal(from_tuple["semantic"], result["semantic"])


def test_family_metadata_and_task_contract():
    assert LibreDDRNet.FAMILY == "ddrnet"
    assert LibreDDRNet.FILENAME_PREFIX == "LibreDDRNet"
    assert LibreDDRNet.SUPPORTED_TASKS == ("semantic",)
    assert LibreDDRNet.REQUIRE_TASK_SUFFIX is True
    assert LibreDDRNet.semantic_resize_mode == "rescale_crop"
    assert LibreDDRNet.semantic_imgsz_divisor == STRIDE == 8
    model = LibreDDRNet(size="m", device="cpu")
    assert model.semantic_train_imgsz == (1024, 1024)
    assert model.semantic_val_imgsz == (1024, 2048)
    assert model.semantic_scale_jitter == (0.5, 2.0)
    assert model.names == CITYSCAPES_NAMES


def test_wrapper_rebuild_for_new_classes_keeps_the_backbone():
    model = LibreDDRNet(size="s", device="cpu")
    stem = model.model.conv1[0].weight.detach().clone()
    model._rebuild_for_new_classes(3)
    assert model.nb_classes == 3 and len(model.names) == 3
    assert torch.equal(model.model.conv1[0].weight, stem)


# ---------------------------------------------------------------------------
# Loss / trainer
# ---------------------------------------------------------------------------


def test_loss_weights_aux_at_point_four_and_upsamples_logits():
    criterion = DDRNetLoss(aux_weight=0.4, ohem=False)
    main = torch.zeros(1, 3, 4, 4)
    aux = torch.zeros(1, 3, 4, 4)
    main[:, 0] = 10.0
    aux[:, 1] = 10.0
    target = torch.zeros(1, 32, 32, dtype=torch.long)
    parts = criterion((main, aux), target)
    assert parts["loss_ce"].item() < 1e-4
    assert parts["loss_aux"].item() > 1.0
    assert parts["loss"].item() == pytest.approx(
        parts["loss_ce"].item() + 0.4 * parts["loss_aux"].item(), rel=1e-5
    )


def test_ohem_drops_easy_pixels_but_keeps_min_kept():
    logits = torch.zeros(1, 2, 1, 4)
    logits[0, 0, 0] = torch.tensor([10.0, 10.0, -10.0, 0.0])  # easy, easy, hard, 0.5
    target = torch.zeros(1, 1, 4, dtype=torch.long)
    per_pixel = torch.nn.functional.cross_entropy(logits, target, reduction="none").reshape(-1)

    loss = OhemCrossEntropy(thresh=0.9, min_kept=1)(logits, target)
    assert loss.item() == pytest.approx(per_pixel[2:].mean().item(), rel=1e-6)

    keep_all = OhemCrossEntropy(thresh=0.9, min_kept=4)(logits, target)
    assert keep_all.item() == pytest.approx(per_pixel.mean().item(), rel=1e-6)


def test_ohem_excludes_ignore_pixels_and_survives_an_all_ignore_batch():
    logits = torch.randn(1, 3, 4, 4, requires_grad=True)
    target = torch.full((1, 4, 4), 255, dtype=torch.long)
    loss = OhemCrossEntropy()(logits, target)
    assert loss.item() == 0.0
    loss.backward()
    assert logits.grad is not None

    target[0, 0, 0] = 2
    plain = torch.nn.functional.cross_entropy(logits, target, ignore_index=255)
    assert OhemCrossEntropy(min_kept=10)(logits, target).item() == pytest.approx(plain.item())


def test_config_matches_the_paper_recipe():
    config = DDRNetConfig()
    assert config.optimizer == "sgd"
    assert config.lr0 == 0.01
    assert config.momentum == 0.9
    assert config.weight_decay == 5e-4
    assert config.nesterov is False
    assert config.scheduler == "poly" and config.poly_power == 0.9
    assert config.warmup_epochs == 0
    assert config.aux_weight == 0.4
    assert config.ohem is True and config.ohem_thresh == 0.9 and config.ohem_min_kept == 131072
    assert config.epochs == 484 and config.batch == 12
    assert config.imgsz == 1024
    assert config.sync_bn is True
    assert config.ema is False and config.amp is False


def _trainer(model: LibreDDRNet, **overrides) -> DDRNetTrainer:
    trainer = DDRNetTrainer.__new__(DDRNetTrainer)
    trainer.model = model.model
    trainer.wrapper_model = model
    trainer.device = torch.device("cpu")
    trainer.num_classes = model.nb_classes
    trainer.config = DDRNetConfig(size=model.size, num_classes=model.nb_classes, **overrides)
    return trainer


def test_trainer_identity_scheduler_and_forward():
    model = LibreDDRNet(size="s", nb_classes=3, device="cpu")
    trainer = _trainer(model, ohem_min_kept=16)
    assert trainer.get_model_family() == "ddrnet"
    assert trainer.get_model_tag() == "LibreDDRNet-s"
    assert trainer.best_metric_key == "metrics/mIoU"
    assert isinstance(trainer.create_scheduler(10), PolyLRScheduler)
    assert trainer.criterion.criterion.min_kept == 16

    model.model.train()
    out = trainer.on_forward(torch.rand(2, 3, 64, 64), torch.randint(0, 3, (2, 64, 64)))
    assert {"total_loss", "loss_ce", "loss_aux"} <= set(out)
    out["total_loss"].backward()
    assert model.model.final_layer.conv2.weight.grad is not None
    assert model.model.seghead_extra.conv2.weight.grad is not None


def test_trainer_rejects_a_non_poly_schedule():
    trainer = _trainer(LibreDDRNet(size="s", device="cpu"), scheduler="cos")
    with pytest.raises(ValueError, match="polynomial"):
        trainer.create_scheduler(10)


def test_train_rejects_an_off_stride_imgsz(monkeypatch, stub_training):
    model = LibreDDRNet(size="s", device="cpu")
    model._imagenet_init_pending = False
    with pytest.raises(ValueError, match="divisible"):
        model.train(data="unused.yaml", imgsz=(100, 200), device="cpu")


def test_parity_and_converter_scripts_exist():
    root = Path(__file__).resolve().parents[2]
    assert (root / "weights" / "convert_ddrnet_weights.py").is_file()
    assert (root / "weights" / "parity_ddrnet.py").is_file()
