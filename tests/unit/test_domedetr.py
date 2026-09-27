"""LibreDOMEDETR unit suite: routing, shapes, and the MWAS reformulation.

The upstream-parity proof lives in ``weights/parity_domedetr.py`` (it needs the
upstream checkout and the published checkpoints, so it is not a CI test).
``tests/e2e/test_val_coco128.py``'s mAP gate does not apply to this family:
Dome-DETR has no COCO checkpoint, only AI-TOD-V2 and VisDrone.
"""

from __future__ import annotations

import pytest
import torch

from libreyolo.models.deim.model import LibreDEIM
from libreyolo.models.deimv2.model import LibreDEIMv2
from libreyolo.models.dfine.model import LibreDFINE
from libreyolo.models.domedetr.model import LibreDOMEDETR
from libreyolo.models.domedetr.nn import DEC_NUM_LAYERS, LibreDOMEDETRModel
from libreyolo.models.rtdetrv4.model import LibreRTDETRv4


pytestmark = [pytest.mark.unit, pytest.mark.domedetr]


def _domedetr_state_dict(size: str = "s", variant: str = "aitod") -> dict:
    nc = 9 if variant == "aitod" else 12
    return LibreDOMEDETRModel(config=size, nb_classes=nc, variant=variant).state_dict()


def _lazy_rfdetr():
    """RF-DETR registers lazily behind the transformers dep.

    Its discriminator is deliberately broad and matches Dome-DETR's
    ``decoder.denoising_class_embed.weight``, so the rejection has to be
    explicit rather than left to registry order.
    """
    from libreyolo.models.rfdetr.model import LibreRFDETR

    return LibreRFDETR


def _dfine_like_state_dict() -> dict:
    """A D-FINE-lineage state dict: carries pre_bbox_head but no DeFE."""
    from libreyolo.models.dfine.nn import LibreDFINEModel

    return LibreDFINEModel(config="s", nb_classes=80).state_dict()


# -- routing ------------------------------------------------------------------


def test_can_load_accepts_domedetr():
    assert LibreDOMEDETR.can_load(_domedetr_state_dict()) is True


def test_can_load_rejects_dfine():
    assert LibreDOMEDETR.can_load(_dfine_like_state_dict()) is False


@pytest.mark.parametrize(
    "sibling",
    [LibreDFINE, LibreDEIM, LibreDEIMv2, LibreRTDETRv4, _lazy_rfdetr],
    ids=["dfine", "deim", "deimv2", "rtdetrv4", "rfdetr"],
)
def test_dfine_lineage_rejects_domedetr(sibling):
    """The whole D-FINE lineage must refuse Dome-DETR checkpoints.

    Dome-DETR is a D-FINE derivative and carries ``decoder.pre_bbox_head.``,
    the key D-FINE discriminates on. Without an explicit rejection LibreDFINE
    would claim these files and load a subset of their tensors.
    """
    cls = sibling() if callable(sibling) and not hasattr(sibling, "can_load") else sibling
    assert cls.can_load(_domedetr_state_dict()) is False


def test_only_domedetr_claims_a_domedetr_checkpoint():
    """Exactly one registered family may claim these tensors.

    Registry order cannot be the safeguard here: importing LibreDOMEDETR pulls
    in ``models.dfine`` for the shared decoder stack, so LibreDFINE registers
    first no matter where the import sits. The bidirectional ``can_load``
    rejection is what actually decides it, so assert the outcome directly.
    """
    import libreyolo.models  # noqa: F401  (registers every family)
    from libreyolo.models.base import BaseModel

    state_dict = _domedetr_state_dict()
    claimants = {
        cls.FAMILY for cls in BaseModel._registry if cls.can_load(state_dict)
    }
    assert claimants == {"domedetr"}


def test_filename_detection_with_variant_suffix():
    assert LibreDOMEDETR.detect_size_from_filename("LibreDOMEDETRs-visdrone.pt") == "s"
    assert LibreDOMEDETR.detect_size_from_filename("LibreDOMEDETRl-aitod.pt") == "l"
    # Detect has no task suffix, so the variant must not be read as one.
    assert LibreDOMEDETR.detect_task_from_filename("LibreDOMEDETRm-aitod.pt") in (
        None,
        "detect",
    )


@pytest.mark.parametrize(
    ("filename", "nb_classes", "expected"),
    [
        ("LibreDOMEDETRs-visdrone.pt", 80, "visdrone"),
        ("LibreDOMEDETRs-aitod.pt", 80, "aitod"),
        (None, 12, "visdrone"),
        (None, 9, "aitod"),
    ],
)
def test_weight_variant_resolution(filename, nb_classes, expected):
    resolved = LibreDOMEDETR._resolve_weight_variant(
        explicit=None, model_path=filename, nb_classes=nb_classes
    )
    assert resolved == expected


def test_weight_variant_rejects_unknown():
    with pytest.raises(ValueError, match="Unknown weight_variant"):
        LibreDOMEDETR._resolve_weight_variant(
            explicit="coco", model_path=None, nb_classes=9
        )


# -- shapes -------------------------------------------------------------------


@pytest.mark.parametrize("size", ["s", "m", "l"])
def test_size_detection_from_state_dict(size):
    assert LibreDOMEDETR.detect_size(_domedetr_state_dict(size)) == size


@pytest.mark.parametrize(("variant", "nc"), [("aitod", 9), ("visdrone", 12)])
def test_nb_classes_detection(variant, nc):
    assert LibreDOMEDETR.detect_nb_classes(_domedetr_state_dict("s", variant)) == nc


def test_decoder_depth_is_per_size_and_variant():
    """L is 4 decoder layers on AI-TOD-V2 but 6 on VisDrone.

    Keying depth off the size alone silently builds the wrong model, and the
    state dict then loads non-strictly with two layers left at init.
    """
    assert DEC_NUM_LAYERS[("l", "aitod")] == 4
    assert DEC_NUM_LAYERS[("l", "visdrone")] == 6

    aitod = LibreDOMEDETRModel(config="l", nb_classes=9, variant="aitod")
    visdrone = LibreDOMEDETRModel(config="l", nb_classes=12, variant="visdrone")
    assert len(aitod.decoder.decoder.layers) == 4
    assert len(visdrone.decoder.decoder.layers) == 6


def test_forward_shape_and_query_budget():
    torch.manual_seed(0)
    model = LibreDOMEDETRModel(config="s", nb_classes=12, variant="visdrone").eval()
    with torch.no_grad():
        out = model(torch.randn(1, 3, 800, 800))

    assert set(out) >= {"pred_logits", "pred_boxes"}
    n_queries = out["pred_logits"].shape[1]
    assert out["pred_logits"].shape == (1, n_queries, 12)
    assert out["pred_boxes"].shape == (1, n_queries, 4)
    # PAQI keeps the top min_num_select unconditionally and never exceeds
    # max_num_select, whatever the density map says.
    assert 250 <= n_queries <= 500
    assert out["batch_queries_num"] == [n_queries]


def test_batch_padding_uses_negative_logits():
    """Short rows in a mixed batch must not surface as 0.5-confidence boxes."""
    torch.manual_seed(0)
    model = LibreDOMEDETRModel(config="s", nb_classes=12, variant="visdrone").eval()
    with torch.no_grad():
        out = model(torch.randn(2, 3, 800, 800))

    counts = out["batch_queries_num"]
    n_queries = out["pred_logits"].shape[1]
    assert n_queries == max(counts)
    for b, count in enumerate(counts):
        if count < n_queries:
            padded = out["pred_logits"][b, count:]
            assert torch.all(padded < -1e30), "padding must be far below any real logit"


# -- MWAS reformulation -------------------------------------------------------


def test_mwas_static_path_matches_gather_path():
    """The static path is algebraically the same computation, not a fallback.

    It keeps every window in the tensor and hides the empty ones from the
    cross-window attention with a key-padding mask instead of gathering the
    occupied ones. Softmax over the padded key set reassociates the
    floating-point sums, so this pins the gap at ~1e-5 rather than asserting
    bit-equality.
    """
    torch.manual_seed(0)
    model = LibreDOMEDETRModel(config="s", nb_classes=9, variant="aitod").eval()
    x = torch.randn(1, 3, 800, 800)
    processor = model.encoder.mwas_processor

    with torch.no_grad():
        processor.force_static_path = False
        gather = model.encoder(model.backbone(x), img_inputs=x)
        processor.force_static_path = True
        static = model.encoder(model.backbone(x), img_inputs=x)

    assert torch.equal(
        gather["defe"]["defe_window_mask"], static["defe"]["defe_window_mask"]
    )
    worst = max(
        (a - b).abs().max().item()
        for a, b in zip(gather["feats"], static["feats"])
    )
    assert worst < 1e-4, f"static MWAS path diverged by {worst}"


# -- scope --------------------------------------------------------------------


def test_train_config_is_wired():
    from libreyolo.training.config import DOMEDETRConfig

    assert LibreDOMEDETR.TRAIN_CONFIG is DOMEDETRConfig
    cfg = DOMEDETRConfig(num_classes=12)
    assert cfg.imgsz == 800
    # MWAS needs the stride-8 map divisible by the window size, so a random
    # per-batch resize would break the forward outright.
    assert cfg.multi_scale is False
    # Upstream runs the backbone at 2e-5 against a 2e-4 base.
    assert cfg.backbone_lr_mult == pytest.approx(0.1)


def test_training_forward_emits_full_objective():
    """A training step must produce every term the criterion consumes."""
    torch.manual_seed(0)
    model = LibreDOMEDETRModel(config="s", nb_classes=12, variant="visdrone").train()
    targets = [
        {"labels": torch.randint(0, 12, (5,)), "boxes": torch.rand(5, 4) * 0.2 + 0.4},
        {"labels": torch.randint(0, 12, (9,)), "boxes": torch.rand(9, 4) * 0.2 + 0.4},
    ]
    out = model(torch.randn(2, 3, 800, 800), targets=targets)

    for key in ("pred_logits", "pred_boxes", "pred_corners", "aux_outputs",
                "dn_outputs", "dn_meta", "enc_aux_outputs", "defe"):
        assert key in out, f"missing {key}"
    # The density target DeFE is supervised against.
    assert out["defe"]["gt_density_map"].shape[-2:] == (800, 800)
    # The criterion reads these back to build the count-regression target.
    assert out["defe"]["min_num_select"] == 250
    assert out["defe"]["max_num_select"] == 500


def test_criterion_emits_defe_losses():
    """Both DeFE terms must be live, not silently absent."""
    from libreyolo.models.dfine.matcher import HungarianMatcher
    from libreyolo.models.domedetr.loss import DomeCriterion

    torch.manual_seed(0)
    model = LibreDOMEDETRModel(config="s", nb_classes=12, variant="visdrone").train()
    targets = [{"labels": torch.randint(0, 12, (7,)), "boxes": torch.rand(7, 4) * 0.2 + 0.4}]
    out = model(torch.randn(1, 3, 800, 800), targets=targets)

    criterion = DomeCriterion(
        matcher=HungarianMatcher(
            weight_dict={"cost_class": 2.0, "cost_bbox": 5.0, "cost_giou": 2.0},
            use_focal_loss=True,
            alpha=0.25,
            gamma=2.0,
        ),
        weight_dict={"loss_vfl": 1.0, "loss_bbox": 5.0, "loss_giou": 2.0,
                     "loss_fgl": 0.15, "loss_ddf": 1.5},
        losses=["vfl", "boxes", "local"],
        alpha=0.75,
        gamma=2.0,
        num_classes=12,
        reg_max=32,
    )
    losses = {k: v.detach() for k, v in criterion(out, targets).items()}

    assert "loss_defe_density" in losses
    assert "loss_defe_reg" in losses
    assert float(losses["loss_defe_density"]) > 0
    assert torch.isfinite(losses["loss_defe_density"])
    # Randomly-initialised reg head sits near sigmoid(0)=0.5 against a target
    # of 0, so this must be clearly non-zero on a fresh model.
    assert float(losses["loss_defe_reg"]) > 0
    assert all(torch.isfinite(v).all() for v in losses.values())


def test_denoising_mask_isolates_each_image_padding():
    """Padded queries must not exchange attention with real ones, per image."""
    from libreyolo.models.domedetr.denoising import (
        get_contrastive_denoising_training_group,
    )

    torch.manual_seed(0)
    num_classes, num_queries, num_heads = 12, 40, 8
    embed = torch.nn.Embedding(num_classes + 1, 16, padding_idx=num_classes)
    targets = [
        {"labels": torch.randint(0, num_classes, (3,)), "boxes": torch.rand(3, 4) * 0.2 + 0.4},
        {"labels": torch.randint(0, num_classes, (2,)), "boxes": torch.rand(2, 4) * 0.2 + 0.4},
    ]
    batch_queries_num = [40, 25]  # image 1 is padded from 25 up to 40

    _, _, attn_mask, dn_meta = get_contrastive_denoising_training_group(
        targets, num_classes, num_queries, embed,
        num_denoising=20, batch_queries_num=batch_queries_num, num_heads=num_heads,
    )

    bs = len(batch_queries_num)
    dn_len = int(dn_meta["dn_num_split"][0])
    assert attn_mask.shape == (bs * num_heads, dn_len + num_queries, dn_len + num_queries)

    # Image 0 is full width: no padding to mask beyond D-FINE's own blocks.
    pad_start = dn_len + batch_queries_num[1]
    head0, head1 = 0, num_heads
    assert bool(attn_mask[head1:, :pad_start, pad_start:].all()), "real -> pad not masked"
    assert bool(attn_mask[head1:, pad_start:, :pad_start].all()), "pad -> real not masked"
    # A fully masked row makes softmax NaN, so padded rows keep seeing themselves.
    assert not bool(attn_mask[head1:, pad_start:, pad_start:].all())


def test_bare_filename_download_explains_itself():
    """No COCO checkpoint exists, so the bare name must say so, not 404.

    Left to the base implementation this retries a never-to-exist HF repo
    three times and ends in a generic "file not found", which sends people
    hunting for a network fault.
    """
    with pytest.raises(FileNotFoundError) as excinfo:
        LibreDOMEDETR.get_download_url("LibreDOMEDETRs.pt")
    message = str(excinfo.value)
    assert "no dataset suffix" in message
    assert "LibreDOMEDETRs-aitod.pt" in message
    assert "LibreDOMEDETRs-visdrone.pt" in message


def test_suffixed_filename_download_uses_libreyolo_mirror():
    filename = "LibreDOMEDETRm-visdrone.pt"
    assert LibreDOMEDETR.get_download_url(filename) == (
        "https://huggingface.co/LibreYOLO/LibreDOMEDETRm-visdrone/"
        f"resolve/main/{filename}"
    )


def test_download_notice_discloses_restricted_weight_terms():
    notice = LibreDOMEDETR.get_download_notice(
        "LibreDOMEDETRs-aitod.pt", "https://example.invalid/checkpoint.pt"
    )
    assert "ACADEMIC RESEARCH PURPOSES ONLY" in notice
    assert "NOT covered by LibreYOLO's MIT license" in notice
    assert "Apache-2.0" in notice
    assert "responsible" in notice


@pytest.mark.parametrize(
    "foreign",
    ["LibreTEEDt-edge.pt", "LibreDexiNedb-edge.pt", "LibreDFINEs.pt",
     "LibreYOLO9t.pt", "LibreRFDETRs.pt"],
)
def test_download_hook_ignores_other_families(foreign):
    """The hook must claim only Dome-DETR filenames.

    ``download_weights`` asks every registered family in turn and takes the
    first non-None answer, so a hook that raises unconditionally hijacks every
    other family's download with the wrong error. Returning None hands the
    filename to its real owner.
    """
    assert LibreDOMEDETR.get_download_url(foreign) is None


def test_group_is_supporting_trainable():
    from libreyolo.models.registry import MODEL_GROUPS

    assert MODEL_GROUPS["domedetr"] == "g2"


@pytest.mark.parametrize(
    "fmt",
    ["onnx", "torchscript", "tensorrt", "openvino", "ncnn", "tflite",
     "coreml", "coreai", "paddle", "mnn", "executorch", "rknn"],
)
def test_every_export_format_is_blocked_with_a_reason(fmt):
    """Better a clear refusal than a graph that is only valid for one image.

    Parametrised over every registered format, and the guard sits in
    ``BaseExporter.create`` rather than in the per-format preflight: formats
    with an optional toolchain (ExecuTorch) raise ImportError from their own
    constructor otherwise, telling the user to install a dependency that
    would not help.
    """
    model = LibreDOMEDETR(model_path=None, size="s", nb_classes=9)
    with pytest.raises(NotImplementedError, match="query count per image"):
        model.export(format=fmt)


def test_support_registry_agrees_with_the_runtime_block():
    """The published support table must not advertise what export() refuses.

    These are two independent sources: the guard in BaseExporter.create and
    the tiers in libreyolo/export/support.py that generate
    docs/export_support.md. Registering only the first left the docs claiming
    ONNX/TorchScript/TensorRT/OpenVINO were available for this family.
    """
    from libreyolo.export.support import EXPORT_FORMATS, get_support

    for fmt in EXPORT_FORMATS:
        entry = get_support("domedetr", "detect", fmt)
        assert entry.tier == "blocked", f"{fmt} advertised as {entry.tier}"
        assert "query count" in entry.reason


def test_export_registry_has_no_domedetr_support_recorded():
    """The published inventory must not advertise an export this cannot do."""
    import json
    from pathlib import Path as _Path

    inventory = json.loads(
        (_Path(__file__).resolve().parents[2] / "reports" / "export_inventory.json")
        .read_text(encoding="utf-8")
    )
    assert inventory["domedetr"]["export_override"] == "none"


def test_is_nms_free_family():
    """PAQI's NMS runs inside the decoder; backends must not re-suppress."""
    from libreyolo.backends.base import _is_nms_free_family

    assert _is_nms_free_family("domedetr") is True
