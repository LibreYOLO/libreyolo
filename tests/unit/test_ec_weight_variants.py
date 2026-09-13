"""EC O365 checkpoints are opt-in; COCO defaults retain their URLs and terms."""

import importlib.util
from pathlib import Path

import pytest
import torch

from libreyolo.models.ec.model import LibreEC
from libreyolo.utils.serialization import validate_checkpoint_metadata
from libreyolo.utils.general import COCO_CLASSES

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("size", ["s", "m", "l", "x"])
@pytest.mark.parametrize(
    "suffix,task", [("", None), ("-seg", "segment"), ("-pose", "pose")]
)
def test_variant_download_and_coco_default(size, suffix, task):
    plain = f"LibreEC{size}{suffix}"
    for variant in ("", "-obj2coco"):
        name = plain + variant
        filename = name + ".pt"
        url = LibreEC.get_download_url(filename)
        assert url == f"https://huggingface.co/LibreYOLO/{name}/resolve/main/{filename}"
        assert LibreEC.detect_size_from_filename(filename) == size
        assert LibreEC.detect_task_from_filename(filename) == task
        notice = LibreEC.get_download_notice(filename, url)
        if variant:
            assert "NON-COMMERCIAL" in notice
        else:
            assert notice is None
    assert LibreEC.get_download_url(plain + "-unknown.pt") is None


@pytest.fixture
def converter(monkeypatch):
    folder = Path(__file__).resolve().parents[2] / "weights"
    monkeypatch.syspath_prepend(str(folder))
    spec = importlib.util.spec_from_file_location(
        "convert_ec_weights", folder / "convert_ec_weights.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _state(nc=80):
    return {
        "backbone.backbone.register_token": torch.zeros(1, 4, 192),
        "backbone.projector.0.conv.weight": torch.zeros(192, 1, 1, 1),
        "decoder.dec_score_head.0.bias": torch.zeros(nc),
    }


@pytest.mark.parametrize(
    "task,suffix", [("detect", ""), ("segment", "-seg"), ("pose", "-pose")]
)
def test_conversion_preserves_ema_and_marks_variant(converter, tmp_path, task, suffix):
    source = tmp_path / "source.pth"
    ema = _state(2 if task == "pose" else 80)
    if task == "pose":
        ema["decoder.keypoint_embedding.weight"] = torch.zeros(17, 192)
    elif task == "segment":
        ema["decoder.decoder.segmentation_head.bias"] = torch.zeros(1)
    torch.save({"model": {"unused": torch.ones(1)}, "ema": {"module": ema}}, source)
    target = tmp_path / f"LibreECs{suffix}-obj2coco.pt"
    converter.convert_weights(
        str(source), str(target), "s", task=task, variant="obj2coco"
    )
    result = torch.load(target, weights_only=True)
    validate_checkpoint_metadata(result, strict=True)
    assert result["task"] == task
    assert result["nc"] == (1 if task == "pose" else 80)
    if task == "pose":
        assert result["names"] == {0: "person"}
        assert result["num_keypoints"] == 17
        assert result["keypoint_dim"] == 3
    else:
        assert result["names"] == dict(enumerate(COCO_CLASSES))
        assert "num_keypoints" not in result
        assert "keypoint_dim" not in result
    assert result["weight_variant"] == "obj2coco"
    assert result["license"] == "edgecrafter-non-commercial"
    assert len(result["source_sha256"]) == 64
    assert result["model"].keys() == ema.keys()
    assert all(torch.equal(result["model"][key], value) for key, value in ema.items())


def test_raw_o365_head_cannot_be_labelled_coco(converter, tmp_path):
    source = tmp_path / "source.pth"
    torch.save({"model": _state(365)}, source)
    with pytest.raises(ValueError, match="head"):
        converter.convert_weights(
            str(source), str(tmp_path / "LibreECs-obj2coco.pt"), "s", variant="obj2coco"
        )


def test_variant_cannot_overwrite_default_name(converter, tmp_path):
    with pytest.raises(ValueError, match="filename"):
        converter.convert_weights(
            "unused", str(tmp_path / "LibreECs.pt"), "s", variant="obj2coco"
        )


def test_coco_conversion_has_no_variant_restriction(converter, tmp_path):
    source = tmp_path / "ecdet_s.pth"
    torch.save({"model": _state()}, source)
    result = converter.convert_weights(str(source), str(tmp_path / "LibreECs.pt"), "s")
    validate_checkpoint_metadata(result, strict=True)
    assert "weight_variant" not in result
    assert "license" not in result
    assert result["nc"] == 80


def test_variant_filename_task_must_match(converter, tmp_path):
    with pytest.raises(ValueError, match="filename"):
        converter.convert_weights(
            "unused",
            str(tmp_path / "LibreECs-seg-obj2coco.pt"),
            "s",
            variant="obj2coco",
        )
