"""Unit tests for the native YOLOv9-P2 (stride-4 small-object) family."""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.unit


def test_yolo9_p2_is_registered_and_detects_filename():
    from libreyolo import LibreYOLO9P2
    from libreyolo.models.base.model import BaseModel

    assert any(cls.__name__ == "LibreYOLO9P2" for cls in BaseModel._registry)
    assert LibreYOLO9P2.FAMILY == "yolo9_p2"
    assert LibreYOLO9P2.SUPPORTED_TASKS == ("detect",)
    assert LibreYOLO9P2.detect_size_from_filename("LibreYOLO9P2t.pt") == "t"
    assert LibreYOLO9P2.detect_size_from_filename("LibreYOLO9P2s.pt") == "s"


def test_yolo9_p2_filename_does_not_match_base_family():
    """LibreYOLO9P2t.pt must not be claimed by the base yolo9 filename regex
    (and vice versa)."""
    from libreyolo import LibreYOLO9, LibreYOLO9P2

    assert LibreYOLO9.detect_size_from_filename("LibreYOLO9P2t.pt") is None
    assert LibreYOLO9P2.detect_size_from_filename("LibreYOLO9t.pt") is None


def test_yolo9_p2_eval_forward_shapes():
    """Inference returns the standard yolo9 prediction tensor with the extra
    stride-4 level: 160^2 + 80^2 + 40^2 + 20^2 = 34000 anchors at 640."""
    from libreyolo import LibreYOLO9P2

    model = LibreYOLO9P2(None, size="t", device="cpu")
    model.model.eval()
    with torch.no_grad():
        out = model.model(torch.zeros(1, 3, 640, 640))

    assert isinstance(out, dict)
    assert out["predictions"].shape == (1, 84, 34000)
    assert len(out["raw_outputs"]) == 4
    assert out["x4"]["features"].shape[-2:] == (160, 160)
    assert out["x8"]["features"].shape[-2:] == (80, 80)


def test_yolo9_p2_train_forward_shapes():
    """Training mode without targets returns four raw FPN maps of shape
    (B, 4*reg_max + nc, H, W) at strides 4/8/16/32."""
    from libreyolo import LibreYOLO9P2

    model = LibreYOLO9P2(None, size="t", device="cpu")
    model.model.train()
    out = model.model(torch.zeros(1, 3, 640, 640))

    expected = [
        (1, 144, 160, 160),
        (1, 144, 80, 80),
        (1, 144, 40, 40),
        (1, 144, 20, 20),
    ]
    assert [tuple(t.shape) for t in out] == expected


def test_yolo9_p2_loss_runs_and_produces_finite_total():
    """End-to-end train forward with targets exercises the 4-scale anchor
    generation (stride 4 included) and returns finite losses."""
    from libreyolo import LibreYOLO9P2

    model = LibreYOLO9P2(None, size="t", device="cpu")
    model.model.train()

    targets = torch.zeros(2, 30, 5)
    targets[0, 0] = torch.tensor([3.0, 320.0, 240.0, 12.0, 10.0])  # tiny box
    targets[0, 1] = torch.tensor([5.0, 100.0, 100.0, 200.0, 150.0])
    targets[1, 0] = torch.tensor([1.0, 400.0, 320.0, 6.0, 8.0])  # tiny box

    out = model.model(torch.randn(2, 3, 640, 640), targets=targets)
    assert torch.isfinite(out["total_loss"])
    for key in ("box", "cls", "dfl"):
        assert torch.isfinite(torch.as_tensor(out[key]))


def test_yolo9_p2_can_load_discriminators():
    """Checkpoint routing must stay unambiguous in all directions across the
    yolo9 / yolo9_e2e / yolo9_p2 trio."""
    from libreyolo import LibreYOLO9, LibreYOLO9E2E, LibreYOLO9P2

    p2_sd = LibreYOLO9P2(None, size="t", device="cpu").model.state_dict()
    reg_sd = LibreYOLO9(None, size="t", device="cpu").model.state_dict()
    e2e_sd = LibreYOLO9E2E(None, size="t", device="cpu").model.state_dict()

    assert LibreYOLO9P2.can_load(p2_sd) is True
    assert LibreYOLO9P2.can_load(reg_sd) is False
    assert LibreYOLO9P2.can_load(e2e_sd) is False
    assert LibreYOLO9.can_load(p2_sd) is False
    assert LibreYOLO9E2E.can_load(p2_sd) is False
    assert LibreYOLO9.can_load(reg_sd) is True


def test_yolo9_p2_factory_resolves_p2_checkpoint(tmp_path):
    from libreyolo import LibreYOLO, LibreYOLO9P2

    src = LibreYOLO9P2(None, size="t", device="cpu")
    ckpt = tmp_path / "LibreYOLO9P2t.pt"
    torch.save({"model": src.model.state_dict()}, ckpt)

    loaded = LibreYOLO(str(ckpt), device="cpu")
    assert loaded.FAMILY == "yolo9_p2"
    assert loaded.size == "t"


def test_yolo9_p2_detect_nb_classes():
    from libreyolo import LibreYOLO9P2

    sd = LibreYOLO9P2(None, size="t", device="cpu").model.state_dict()
    assert LibreYOLO9P2.detect_nb_classes(sd) == 80


def test_yolo9_p2_transfer_remap_from_base_yolo9():
    """Every tensor of a base yolo9-t checkpoint must land in the P2 model:
    backbone/neck as-is, head towers shifted to indices 1..3. The stride-4
    tower (index 0) and the new neck modules stay freshly initialized."""
    from libreyolo import LibreYOLO9, LibreYOLO9P2

    base_sd = LibreYOLO9(None, size="t", device="cpu").model.state_dict()
    p2 = LibreYOLO9P2(None, size="t", device="cpu")

    prepared = p2._prepare_state_dict(dict(base_sd))
    current = p2.model.state_dict()

    unmatched = [
        key
        for key, value in prepared.items()
        if key not in current or current[key].shape != value.shape
    ]
    assert unmatched == []

    # Towers shifted by one: base tower 0 (P3) now sits at index 1.
    assert "head.cv2.1.0.conv.weight" in prepared
    assert not any(key.startswith("head.cv2.0.") for key in prepared)
    assert any(key.startswith("head.cv2.3.") for key in prepared)

    # Fresh modules receive nothing from the checkpoint.
    leftover = set(current) - set(prepared)
    assert any(key.startswith("neck.elan_up3.") for key in leftover)
    assert any(key.startswith("neck.elan_down0.") for key in leftover)
    assert any(key.startswith("head.cv2.0.") for key in leftover)
    assert any(key.startswith("head.cv3.0.") for key in leftover)


def test_yolo9_p2_prepare_state_dict_passes_p2_checkpoints_through():
    """P2 checkpoints (containing the new neck keys) must not be re-shifted."""
    from libreyolo import LibreYOLO9P2

    p2 = LibreYOLO9P2(None, size="t", device="cpu")
    own_sd = p2.model.state_dict()
    prepared = p2._prepare_state_dict(dict(own_sd))
    assert set(prepared) == set(own_sd)


def test_yolo9_p2_visdrone_variant_filename_routing():
    """The published VisDrone research-preview weight must route like any
    other weight file: size detected, base family not claiming it, and the
    download URL pointing at the variant's own HF repo."""
    from libreyolo import LibreYOLO9, LibreYOLO9P2

    name = "LibreYOLO9P2s-visdrone.pt"
    assert LibreYOLO9P2.detect_size_from_filename(name) == "s"
    assert LibreYOLO9P2.detect_variant_from_filename(name) == "visdrone"
    assert LibreYOLO9.detect_size_from_filename(name) is None

    url = LibreYOLO9P2.get_download_url(name)
    assert url == (
        "https://huggingface.co/LibreYOLO/LibreYOLO9P2s-visdrone/"
        "resolve/main/LibreYOLO9P2s-visdrone.pt"
    )

    notice = LibreYOLO9P2.get_download_notice(name, url)
    assert notice is not None
    assert "NON-COMMERCIAL" in notice
    assert "CC BY-NC-SA" in notice


def test_yolo9_p2_plain_filenames_have_no_variant_or_notice():
    from libreyolo import LibreYOLO9P2

    name = "LibreYOLO9P2t.pt"
    assert LibreYOLO9P2.detect_size_from_filename(name) == "t"
    assert LibreYOLO9P2.detect_variant_from_filename(name) is None
    url = LibreYOLO9P2.get_download_url(name)
    assert url == (
        "https://huggingface.co/LibreYOLO/LibreYOLO9P2t/"
        "resolve/main/LibreYOLO9P2t.pt"
    )
    assert LibreYOLO9P2.get_download_notice(name, url) is None


def test_yolo9_p2_loads_transfer_trained_checkpoint(tmp_path):
    """A checkpoint fine-tuned after transfer init keeps the SOURCE family's
    class-tower hidden width, which no fresh P2 build reproduces (e.g. the
    VisDrone release model: stock yolo9-s width 128 vs P2 nc=10 width 64).
    Direct construction from such a checkpoint must rebuild the towers to the
    checkpoint geometry instead of failing with a shape mismatch."""
    from libreyolo import LibreYOLO9P2

    donor = LibreYOLO9P2(None, size="t", device="cpu")
    default_hidden = donor.model.state_dict()["head.cv3.0.0.conv.weight"].shape[0]
    transfer_hidden = default_hidden + 16

    donor.nb_classes = 10
    donor._align_class_towers_for_transfer(
        {"head.cv3.0.0.conv.weight": torch.zeros(transfer_hidden, 8, 3, 3)}
    )
    ckpt = tmp_path / "LibreYOLO9P2t-visdrone.pt"
    torch.save({"model": donor.model.state_dict(), "nc": 10}, ckpt)

    loaded = LibreYOLO9P2(str(ckpt), size="t", device="cpu")
    assert loaded.nb_classes == 10
    hidden = loaded.model.state_dict()["head.cv3.0.0.conv.weight"].shape[0]
    assert hidden == transfer_hidden

    loaded.model.eval()
    with torch.no_grad():
        out = loaded.model(torch.zeros(1, 3, 640, 640))
    assert out["predictions"].shape == (1, 14, 34000)


def test_yolo9_p2_trainer_metadata():
    from libreyolo.models.yolo9_p2.config import YOLO9P2Config
    from libreyolo.models.yolo9_p2.trainer import YOLO9P2Trainer

    assert YOLO9P2Trainer._config_class() is YOLO9P2Config
    assert YOLO9P2Trainer.artifact_model_families == ("yolo9_p2",)
    cfg = YOLO9P2Config(size="t")
    assert cfg.max_labels == 300
    assert "elan_up3" in YOLO9P2Trainer._NECK_FREEZE_MODULES
