"""Unit tests for the native DEIM family."""

from __future__ import annotations

import torch
import pytest


pytestmark = pytest.mark.unit


def test_deim_is_registered_and_detects_upstream_filename():
    from libreyolo import LibreDEIM
    from libreyolo.models.base.model import BaseModel

    assert any(cls.__name__ == "LibreDEIM" for cls in BaseModel._registry)
    assert LibreDEIM.FAMILY == "deim"
    assert LibreDEIM.detect_size_from_filename("LibreDEIMn.pt") == "n"
    assert LibreDEIM.detect_size_from_filename("deim_hgnetv2_n_coco.pth") == "n"


def test_deim_forward_shapes():
    from libreyolo import LibreDEIM

    model = LibreDEIM(None, size="n", device="cpu")
    model.model.eval()
    with torch.no_grad():
        out = model.model(torch.zeros(1, 3, 640, 640))

    assert out["pred_logits"].shape == (1, 300, 80)
    assert out["pred_boxes"].shape == (1, 300, 4)


def test_deim_filename_hint_wins_over_dfine_for_ambiguous_architecture(tmp_path):
    from libreyolo import LibreDEIM, LibreYOLO

    src = LibreDEIM(None, size="n", device="cpu")
    ckpt = tmp_path / "deim_hgnetv2_n_coco.pth"
    torch.save({"model": src.model.state_dict()}, ckpt)

    loaded = LibreYOLO(str(ckpt), device="cpu")
    assert loaded.FAMILY == "deim"
    assert loaded.size == "n"


def test_dfine_filename_hint_wins_over_deim_for_ambiguous_architecture(tmp_path):
    from libreyolo import LibreDFINE, LibreYOLO

    src = LibreDFINE(None, size="n", device="cpu")
    ckpt = tmp_path / "dfine_hgnetv2_n_coco.pth"
    torch.save({"model": src.model.state_dict()}, ckpt)

    loaded = LibreYOLO(str(ckpt), device="cpu")
    assert loaded.FAMILY == "dfine"
    assert loaded.size == "n"


def test_generic_dfine_deim_checkpoint_requires_family_hint(tmp_path):
    from libreyolo import LibreDEIM, LibreYOLO

    src = LibreDEIM(None, size="n", device="cpu")
    ckpt = tmp_path / "ambiguous.pth"
    torch.save({"model": src.model.state_dict()}, ckpt)

    with pytest.raises(ValueError, match="Ambiguous D-FINE/DEIM checkpoint"):
        LibreYOLO(str(ckpt), device="cpu")


def test_ecdet_checkpoint_does_not_trip_dfine_deim_ambiguity(tmp_path):
    """ECDet's decoder also has ``decoder.pre_bbox_head.*`` keys, so DFINE and
    DEIM ``can_load`` both return True alongside ECDet's. The factory must
    still resolve to ECDet via the more-specific ``register_token`` match
    instead of raising the D-FINE/DEIM ambiguity error."""
    from libreyolo import LibreECDET, LibreYOLO

    src = LibreECDET(None, size="s", device="cpu")
    ckpt = tmp_path / "ecdet_s_coco.pth"
    torch.save({"model": src.model.state_dict()}, ckpt)

    loaded = LibreYOLO(str(ckpt), device="cpu")
    assert loaded.FAMILY == "ecdet"
    assert loaded.size == "s"


def test_deim_metadata_hint_wins_over_dfine_for_libreyolo_checkpoint(tmp_path):
    from libreyolo import LibreDEIM, LibreYOLO

    src = LibreDEIM(None, size="n", device="cpu")
    ckpt = tmp_path / "ambiguous.pt"
    torch.save(
        {
            "model": src.model.state_dict(),
            "model_family": "deim",
            "size": "n",
            "nc": 80,
        },
        ckpt,
    )

    loaded = LibreYOLO(str(ckpt), device="cpu")
    assert loaded.FAMILY == "deim"
    assert loaded.size == "n"


def test_deim_mal_loss_direct_call():
    from libreyolo.models.deim.loss import DEIMCriterion
    from libreyolo.models.deim.matcher import HungarianMatcher

    criterion = DEIMCriterion(
        matcher=HungarianMatcher(
            weight_dict={"cost_class": 2.0, "cost_bbox": 5.0, "cost_giou": 2.0},
            use_focal_loss=True,
        ),
        weight_dict={"loss_mal": 1.0},
        losses=["mal"],
        num_classes=3,
        gamma=1.5,
    )
    outputs = {
        "pred_logits": torch.tensor([[[2.0, -1.0, 0.5], [0.1, 0.2, -0.5]]]),
        "pred_boxes": torch.tensor([[[0.5, 0.5, 0.2, 0.2], [0.2, 0.2, 0.1, 0.1]]]),
    }
    targets = [
        {
            "labels": torch.tensor([0]),
            "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]]),
        }
    ]
    indices = [(torch.tensor([0]), torch.tensor([0]))]

    loss = criterion.loss_labels_mal(outputs, targets, indices, num_boxes=1)
    assert set(loss) == {"loss_mal"}
    assert torch.isfinite(loss["loss_mal"])


def test_deim_export_wrapper_returns_tuple():
    from libreyolo import LibreDEIM
    from libreyolo.models.deim.nn import DEIMExportWrapper

    wrapper = LibreDEIM(None, size="n", device="cpu")
    export_model = DEIMExportWrapper(wrapper.model)
    export_model.eval()

    with torch.no_grad():
        out = export_model(torch.zeros(1, 3, 640, 640))

    assert isinstance(out, tuple)
    assert len(out) == 2
    assert out[0].shape == (1, 300, 80)
    assert out[1].shape == (1, 300, 4)
