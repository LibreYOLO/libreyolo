"""Factory and checkpoint-recognition tests for LibreFCN."""

from __future__ import annotations

import pytest
import torch
from PIL import Image
from torchvision.models.segmentation import fcn_resnet50

from libreyolo import LibreFCN
from libreyolo.models.fcn.nn import LibreFCNModel
from libreyolo.models.registry import group_of

pytestmark = pytest.mark.unit


def _fcn_state(depth: int = 50, nc: int = 21) -> dict[str, torch.Tensor]:
    last_block = 5 if depth == 50 else 22
    return {
        "backbone.conv1.weight": torch.empty(64, 3, 7, 7),
        "backbone.layer4.0.conv2.weight": torch.empty(512, 512, 3, 3),
        f"backbone.layer3.{last_block}.conv3.weight": torch.empty(1024, 256, 1, 1),
        "classifier.0.weight": torch.empty(512, 2048, 3, 3),
        "classifier.1.running_mean": torch.empty(512),
        "classifier.4.weight": torch.empty(nc, 512, 1, 1),
        "aux_classifier.0.weight": torch.empty(256, 1024, 3, 3),
        "aux_classifier.4.weight": torch.empty(nc, 256, 1, 1),
    }


def test_fcn_public_factory():
    model = LibreFCN(size="r50", nb_classes=21, device="cpu")
    assert model.family == "fcn"
    assert model.task == "semantic"
    assert model.input_size == 520
    assert model.names[0] == "__background__"
    assert model.names[20] == "tvmonitor"
    assert group_of("fcn") == "g3"


def test_fcn_native_graph_matches_torchvision_with_shared_weights():
    torch.manual_seed(0)
    reference = fcn_resnet50(weights=None, weights_backbone=None, aux_loss=True).eval()
    actual = LibreFCNModel(size="r50", normalize_input=False).eval()
    actual.load_state_dict(reference.state_dict(), strict=True)
    image = torch.rand(1, 3, 32, 32)

    with torch.inference_mode():
        expected_output = reference(image)
        actual_output = actual(image)

    assert tuple(actual_output) == ("out", "aux")
    torch.testing.assert_close(
        actual_output["out"], expected_output["out"], rtol=0, atol=0
    )
    torch.testing.assert_close(
        actual_output["aux"], expected_output["aux"], rtol=0, atol=0
    )


@pytest.mark.parametrize(
    ("size", "parameters"), [("r50", 35_322_218), ("r101", 54_314_346)]
)
def test_fcn_published_parameter_counts(size, parameters):
    model = LibreFCNModel(size=size)
    assert sum(parameter.numel() for parameter in model.parameters()) == parameters


@pytest.mark.parametrize(("depth", "size"), [(50, "r50"), (101, "r101")])
def test_fcn_checkpoint_recognition(depth, size):
    state = _fcn_state(depth)
    assert LibreFCN.can_load(state)
    assert LibreFCN.detect_size(state) == size
    assert LibreFCN.detect_nb_classes(state) == 21


def test_fcn_canonical_default_task_filename_is_suffixless():
    assert LibreFCN.detect_size_from_filename("LibreFCNr50.pt") == "r50"
    assert LibreFCN.get_download_url("LibreFCNr50.pt") == (
        "https://huggingface.co/LibreYOLO/LibreFCNr50/resolve/main/LibreFCNr50.pt"
    )


def test_fcn_rejects_generic_resnet_backbone():
    state = {
        "backbone.conv1.weight": torch.empty(64, 3, 7, 7),
        "classifier.4.weight": torch.empty(21, 512, 1, 1),
    }
    assert not LibreFCN.can_load(state)


def test_fcn_checkpoint_fingerprint_is_bidirectionally_disjoint():
    from libreyolo.models.dinov2.model import LibreDINOv2
    from libreyolo.models.eomt.model import LibreEoMT
    from libreyolo.models.resnet.model import LibreResNet
    from libreyolo.models.rtdetr.model import LibreRTDETR

    tensor = torch.empty(1)
    resnet = {"conv1.weight": tensor, "fc.weight": tensor}
    for layer, blocks in ((1, 2), (2, 2), (3, 2), (4, 2)):
        for block in range(blocks):
            resnet[f"layer{layer}.{block}.conv1.weight"] = tensor
    foreign_states = [
        (LibreResNet, resnet),
        (
            LibreDINOv2,
            {"backbone.encoder.proj.weight": tensor, "predict.weight": tensor},
        ),
        (
            LibreEoMT,
            {
                "query.weight": tensor,
                "mask_head.fc1.weight": tensor,
                "mask_head.fc2.weight": tensor,
                "mask_head.fc3.weight": tensor,
                "class_predictor.weight": tensor,
                "embeddings.patch_embeddings.projection.weight": tensor,
            },
        ),
        (
            LibreRTDETR,
            {
                "backbone.res_layers.0.blocks.0.conv1.weight": tensor,
                "encoder.input_proj.0.conv.weight": tensor,
                "decoder.input_proj.0.conv.weight": tensor,
                "decoder.dec_score_head.0.weight": tensor,
            },
        ),
    ]
    fcn = _fcn_state(50)

    for foreign_class, foreign_state in foreign_states:
        assert foreign_class.can_load(foreign_state)
        assert not LibreFCN.can_load(foreign_state)
        assert not foreign_class.can_load(fcn)


def test_fcn_training_is_explicitly_out_of_scope():
    model = LibreFCN(size="r50", device="cpu")
    with pytest.raises(NotImplementedError, match="inference-only"):
        model.train()


def test_fcn_postprocess_uses_primary_logits_and_restores_source_shape():
    model = object.__new__(LibreFCN)
    primary = torch.zeros(1, 3, 8, 8)
    primary[:, 2] = 1
    auxiliary = torch.zeros_like(primary)
    auxiliary[:, 1] = 10

    result = model._postprocess(
        {"out": primary, "aux": auxiliary},
        conf_thres=0.25,
        iou_thres=0.45,
        original_size=(13, 7),
    )

    assert tuple(result["semantic"].shape) == (7, 13)
    assert torch.all(result["semantic"] == 2)


def test_fcn_predict_returns_semantic_result(tmp_path, monkeypatch):
    class Stub(torch.nn.Module):
        def forward(self, image):
            logits = torch.zeros(image.shape[0], 3, image.shape[-2], image.shape[-1])
            logits[:, 1] = 1
            return {"out": logits, "aux": torch.zeros_like(logits)}

    image_path = tmp_path / "image.png"
    Image.new("RGB", (19, 11), color=(30, 80, 120)).save(image_path)
    monkeypatch.setattr(LibreFCN, "_init_model", lambda self: Stub())
    model = LibreFCN(size="r50", nb_classes=3, device="cpu")

    result = model.predict(str(image_path), imgsz=32)

    assert result.boxes is None
    assert result.semantic_mask is not None
    assert tuple(result.semantic_mask.data.shape) == (11, 19)


def test_shared_semantic_surfaces_extract_primary_fcn_output():
    from libreyolo.export.exporter import _SemanticExportWrapper
    from libreyolo.validation.semantic_validator import SemanticValidator

    class Stub(torch.nn.Module):
        def forward(self, image):
            return {"out": image[:, :2], "aux": image[:, 1:3]}

    image = torch.rand(1, 3, 8, 8)
    exported = _SemanticExportWrapper(Stub())(image)
    validated = SemanticValidator._extract_logits(
        object(), {"out": image[:, :2], "aux": image[:, 1:3]}, (8, 8)
    )

    torch.testing.assert_close(exported, image[:, :2])
    torch.testing.assert_close(validated, image[:, :2])


@pytest.mark.parametrize("size", ["r50", "r101"])
def test_fcn_hosted_spellings_resolve_to_suffixless_repos(size):
    """Both accepted suffixless names must point at the repos that exist.

    The hosted repos are ``LibreYOLO/LibreFCNr50`` and
    ``LibreYOLO/LibreFCNr101``, without the ``-sem`` suffix its semantic
    siblings use.
    """
    assert LibreFCN.get_download_url(f"LibreFCN{size}.pt") == (
        f"https://huggingface.co/LibreYOLO/LibreFCN{size}"
        f"/resolve/main/LibreFCN{size}.pt"
    )


@pytest.mark.parametrize("size", ["r50", "r101"])
def test_fcn_suffixed_spelling_is_rejected_with_the_canonical_name(size):
    """``LibreFCN<size>-sem.pt`` names a repo that does not exist.

    The filename regex accepts the ``-sem`` spelling, so left to the base
    implementation it builds a dead URL and the download fails with an HTTP
    error that reads like an outage. The resolver must instead reject it
    before any URL is built and name the hosted spelling.
    """
    with pytest.raises(FileNotFoundError) as excinfo:
        LibreFCN.get_download_url(f"LibreFCN{size}-sem.pt")
    message = str(excinfo.value)
    assert f"LibreFCN{size}.pt" in message
    assert "huggingface" not in message.lower()


def test_fcn_download_guard_ignores_foreign_filenames():
    """``download_weights`` polls every family; never hijack another's name."""
    assert LibreFCN.get_download_url("LibreYOLOXs.pt") is None
    assert LibreFCN.get_download_url("LibreDeepLabv3mv3-sem.pt") is None


def test_fcn_cli_names_resolve_to_the_hosted_spelling():
    """CLI aliases and weight-name checks must agree with the resolver."""
    from libreyolo.cli.config import is_known_weight_filename, resolve_model_name

    assert resolve_model_name("fcn-r50") == "LibreFCNr50.pt"
    assert resolve_model_name("fcn-r50-sem") == "LibreFCNr50.pt"
    assert is_known_weight_filename("LibreFCNr50.pt") is True
    # The suffixed file spelling stays rejected on both surfaces: the CLI
    # treats it as unknown and the resolver raises before building a URL.
    assert is_known_weight_filename("LibreFCNr50-sem.pt") is False
