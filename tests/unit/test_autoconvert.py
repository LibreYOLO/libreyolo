"""Unit tests for upstream flagship weight auto-conversion.

Covers the pure YOLO9 key-remapping logic and the factory-facing orchestration
in :mod:`libreyolo.models.autoconvert`, using tiny synthetic state dicts so the
tests stay fast and need no external weights.
"""

import argparse
from pathlib import Path

import pytest
import torch

from libreyolo.models import autoconvert as autoconvert_module
from libreyolo.models.yolo9.convert import (
    convert_key,
    convert_state_dict,
    infer_config,
    infer_nb_classes,
    is_upstream_state_dict,
)
from libreyolo.models.autoconvert import autoconvert_upstream_checkpoint
from libreyolo.utils.serialization import (
    load_untrusted_torch_file,
    validate_checkpoint_metadata,
    wrap_libreyolo_checkpoint,
)

pytestmark = [pytest.mark.unit, pytest.mark.yolo9]


class TestYolo9ConvertKey:
    """Upstream numbered-index keys map to LibreYOLO semantic names."""

    @pytest.mark.parametrize(
        "upstream,config,expected",
        [
            ("0.conv.weight", "t", "backbone.conv0.conv.weight"),
            ("1.bn.bias", "t", "backbone.conv1.bn.bias"),
            ("22.heads.0.class_conv.2.weight", "t", "head.cv3.0.2.weight"),
            ("22.heads.1.anchor_conv.2.bias", "t", "head.cv2.1.2.bias"),
            ("3.conv.conv.weight", "t", "backbone.down2.cv.conv.weight"),  # AConv
            ("3.conv1.conv.weight", "c", "backbone.down2.cv1.conv.weight"),  # ADown
            ("9.conv1.conv.weight", "t", "backbone.spp.cv1.conv.weight"),
        ],
    )
    def test_maps_known_keys(self, upstream, config, expected):
        out, ok = convert_key(upstream, config)
        assert ok and out == expected

    def test_anc2vec_is_dropped(self):
        out, ok = convert_key("22.heads.0.anc2vec.anc2vec.weight", "t")
        assert ok is False

    def test_auxiliary_head_not_converted(self):
        out, ok = convert_key("23.heads.0.class_conv.2.weight", "t")
        assert ok is False


class TestYolo9Inference:
    """Config and class-count inference from upstream tensor shapes."""

    def test_is_upstream_state_dict(self):
        assert is_upstream_state_dict({"22.heads.0.class_conv.2.weight": torch.zeros(1)})
        assert not is_upstream_state_dict({"backbone.conv0.conv.weight": torch.zeros(1)})

    @pytest.mark.parametrize(
        "stem_ch,block_ch,expected",
        [(16, None, "t"), (64, None, "c"), (32, 64, "s"), (32, 128, "m")],
    )
    def test_infer_config(self, stem_ch, block_ch, expected):
        sd = {"0.conv.weight": torch.zeros(stem_ch, 3, 3, 3)}
        if block_ch is not None:
            sd["2.conv1.conv.weight"] = torch.zeros(block_ch, 16, 1, 1)
        assert infer_config(sd) == expected

    def test_infer_nb_classes_reads_head_width(self):
        sd = {"22.heads.0.class_conv.2.weight": torch.zeros(7, 16, 1, 1)}
        assert infer_nb_classes(sd) == 7

    def test_infer_nb_classes_ignores_hidden_class_tower_width(self):
        sd = {
            "22.heads.0.class_conv.0.weight": torch.zeros(80, 16, 3, 3),
            "22.heads.0.class_conv.2.weight": torch.zeros(3, 80, 1, 1),
        }

        assert infer_nb_classes(sd) == 3

    def test_convert_state_dict_drops_aux_and_anc2vec(self):
        sd = {
            "0.conv.weight": torch.zeros(16, 3, 3, 3),
            "22.heads.0.class_conv.2.weight": torch.zeros(5, 16, 1, 1),
            "22.heads.0.anc2vec.anc2vec.weight": torch.zeros(1, 16, 1, 1, 1),
            "23.heads.0.class_conv.2.weight": torch.zeros(5, 16, 1, 1),
        }
        converted, stats = convert_state_dict(sd, "t")
        assert "backbone.conv0.conv.weight" in converted
        assert "head.cv3.0.2.weight" in converted
        assert stats["skipped"] == 1  # the aux head (layer 23)
        assert stats["failed"] == 1  # anc2vec


def _synthetic_upstream_yolo9(nc: int) -> dict:
    """Minimal upstream-shaped YOLO9 (config t) state dict with class count nc."""
    return {
        "0.conv.weight": torch.zeros(16, 3, 3, 3),
        "0.bn.weight": torch.zeros(16),
        "22.heads.0.class_conv.2.weight": torch.zeros(nc, 16, 1, 1),
        "22.heads.0.class_conv.2.bias": torch.zeros(nc),
    }


class TestAutoconvertOrchestration:
    def test_converts_upstream_yolo9_with_custom_nc(self, tmp_path):
        src = tmp_path / "v9-t.pt"
        torch.save(
            {
                "model": _synthetic_upstream_yolo9(nc=3),
                "names": ["bolt", "nut", "washer"],
            },
            src,
        )

        out = autoconvert_upstream_checkpoint(str(src))

        assert out is not None
        out_path = Path(out)
        assert out_path.name == "v9-t-LibreYOLO9t.pt"
        assert out_path.parent == tmp_path  # written beside source
        ckpt = torch.load(out_path, map_location="cpu", weights_only=False)
        assert validate_checkpoint_metadata(ckpt, strict=False) == []
        assert ckpt["model_family"] == "yolo9"
        assert ckpt["size"] == "t"
        assert ckpt["nc"] == 3
        assert ckpt["names"] == {0: "bolt", 1: "nut", 2: "washer"}
        assert "head.cv3.0.2.weight" in ckpt["model"]

    def test_autoconvert_does_not_overwrite_canonical_checkpoint(self, tmp_path):
        src = tmp_path / "v9-t.pt"
        canonical = tmp_path / "LibreYOLO9t.pt"
        torch.save({"model": _synthetic_upstream_yolo9(nc=2)}, src)
        torch.save({"sentinel": torch.tensor([1.0])}, canonical)

        out = autoconvert_upstream_checkpoint(str(src))

        assert out is not None
        assert Path(out).name == "v9-t-LibreYOLO9t.pt"
        loaded_canonical = torch.load(canonical, map_location="cpu", weights_only=True)
        assert torch.equal(loaded_canonical["sentinel"], torch.tensor([1.0]))

    def test_uses_safe_loader_when_checkpoint_is_not_preloaded(
        self,
        tmp_path,
        monkeypatch,
    ):
        src = tmp_path / "maybe-rfdetr.pth"
        src.write_bytes(b"not used")
        calls = {}

        def fake_safe_load(path, **kwargs):
            calls["path"] = path
            calls["kwargs"] = kwargs
            return {"not": "upstream"}

        monkeypatch.setattr(
            autoconvert_module,
            "load_untrusted_torch_file",
            fake_safe_load,
        )
        monkeypatch.setattr(autoconvert_module, "_try_yolo9", lambda loaded: None)
        monkeypatch.setattr(autoconvert_module, "_try_rfdetr", lambda loaded: None)

        assert autoconvert_upstream_checkpoint(str(src)) is None
        assert calls["path"] == str(src)
        assert calls["kwargs"]["context"] == "upstream weights"
        assert argparse.Namespace in calls["kwargs"]["safe_globals"]

    def test_checkpoint_names_reads_args_class_names(self):
        loaded = {"args": argparse.Namespace(class_names=["bolt", "nut", "washer"])}

        assert autoconvert_module._checkpoint_names(loaded, nc=2) == ["bolt", "nut"]

    def test_checkpoint_args_are_safe_loader_compatible(self):
        loaded = {
            "args": argparse.Namespace(
                class_names=["bolt", "nut"],
                num_queries=100,
                group_detr=13,
                unsafe=object(),
            )
        }

        assert autoconvert_module._checkpoint_args(loaded) == {
            "class_names": ["bolt", "nut"],
            "num_queries": 100,
            "group_detr": 13,
        }

    def test_preserved_args_remain_weights_only_loadable(self, tmp_path):
        args = autoconvert_module._checkpoint_args(
            {
                "args": argparse.Namespace(
                    class_names=["bolt", "nut"],
                    num_queries=100,
                    group_detr=13,
                )
            }
        )
        ckpt = wrap_libreyolo_checkpoint(
            {"class_embed.bias": torch.zeros(3)},
            model_family="rfdetr",
            size="n",
            task="detect",
            nc=2,
            names=["bolt", "nut"],
            imgsz=384,
            args=args,
        )
        path = tmp_path / "rfdetr-converted.pt"
        torch.save(ckpt, path)

        loaded = load_untrusted_torch_file(path)

        assert loaded["args"] == {
            "class_names": ["bolt", "nut"],
            "num_queries": 100,
            "group_detr": 13,
        }

    def test_rfdetr_custom_90_class_names_are_not_coerced_to_coco(self):
        names = [f"custom_{i}" for i in range(90)]

        assert autoconvert_module._rfdetr_class_metadata(
            {"args": argparse.Namespace(class_names=names)},
            90,
        ) == (90, names)

    def test_rfdetr_coco_metadata_maps_90_arch_classes_to_coco80(self):
        assert autoconvert_module._rfdetr_class_metadata(
            {"args": argparse.Namespace(dataset_file="coco")},
            90,
        )[0] == 80

    def test_returns_none_for_non_upstream_file(self, tmp_path):
        src = tmp_path / "random.pt"
        torch.save({"some.random.tensor": torch.zeros(4)}, src)
        assert autoconvert_upstream_checkpoint(str(src)) is None

    def test_returns_none_for_valid_libreyolo_checkpoint(self, tmp_path):
        from libreyolo.utils.serialization import wrap_libreyolo_checkpoint

        wrapped = wrap_libreyolo_checkpoint(
            {"head.cv3.0.2.weight": torch.zeros(80, 16, 1, 1)},
            model_family="yolo9",
            size="t",
            task="detect",
            nc=80,
        )
        src = tmp_path / "LibreYOLO9t.pt"
        torch.save(wrapped, src)
        assert autoconvert_upstream_checkpoint(str(src)) is None

    def test_returns_none_for_missing_file(self, tmp_path):
        assert autoconvert_upstream_checkpoint(str(tmp_path / "nope.pt")) is None
