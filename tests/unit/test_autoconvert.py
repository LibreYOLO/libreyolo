"""Unit tests for upstream flagship weight auto-conversion.

Covers the pure YOLO9 key-remapping logic and the factory-facing orchestration
in :mod:`libreyolo.models.autoconvert`, using tiny synthetic state dicts so the
tests stay fast and need no external weights.
"""

import argparse
import os
import stat
import threading
from concurrent.futures import ThreadPoolExecutor
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

    def test_auxiliary_spp_and_head_convert(self):
        out, ok = convert_key("23.conv1.weight", "t")
        assert ok and out == "aux.spp.cv1.weight"
        out, ok = convert_key("30.heads.0.class_conv.2.weight", "t")
        assert ok and out == "aux_head.cv3.0.2.weight"

    def test_unknown_aux_leftover_not_converted(self):
        out, ok = convert_key("24.heads.0.class_conv.2.weight", "t")
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

    def test_convert_state_dict_keeps_aux_and_drops_anc2vec(self):
        sd = {
            "0.conv.weight": torch.zeros(16, 3, 3, 3),
            "22.heads.0.class_conv.2.weight": torch.zeros(5, 16, 1, 1),
            "22.heads.0.anc2vec.anc2vec.weight": torch.zeros(1, 16, 1, 1, 1),
            "23.conv1.weight": torch.zeros(16, 16, 1, 1),
            "30.heads.0.class_conv.2.weight": torch.zeros(5, 16, 1, 1),
            "30.heads.0.anc2vec.anc2vec.weight": torch.zeros(1, 16, 1, 1, 1),
        }
        converted, stats = convert_state_dict(sd, "t")
        assert "backbone.conv0.conv.weight" in converted
        assert "head.cv3.0.2.weight" in converted
        assert "aux.spp.cv1.weight" in converted
        assert "aux_head.cv3.0.2.weight" in converted
        assert stats["failed"] == 1  # layer-22 anc2vec
        assert stats["skipped"] == 1  # layer-30 anc2vec


def _synthetic_upstream_yolo9(nc: int) -> dict:
    """Minimal upstream-shaped YOLO9 (config t) state dict with class count nc."""
    return {
        "0.conv.weight": torch.zeros(16, 3, 3, 3),
        "0.bn.weight": torch.zeros(16),
        "22.heads.0.class_conv.2.weight": torch.zeros(nc, 16, 1, 1),
        "22.heads.0.class_conv.2.bias": torch.zeros(nc),
    }


class TestAutoconvertOrchestration:
    def test_publishes_conversion_atomically(self, tmp_path, monkeypatch):
        src = tmp_path / "v9-t.pt"
        torch.save({"model": _synthetic_upstream_yolo9(nc=2)}, src)
        expected = tmp_path / "v9-t-LibreYOLO9t.pt"
        real_save = torch.save
        observed = {}

        def inspect_save(value, destination, *args, **kwargs):
            destination = Path(destination)
            observed["temporary"] = destination
            assert destination != expected
            assert destination.parent == expected.parent
            assert not expected.exists()
            real_save(value, destination, *args, **kwargs)
            assert not expected.exists()

        monkeypatch.setattr(autoconvert_module.torch, "save", inspect_save)

        out = autoconvert_upstream_checkpoint(str(src))

        assert out == str(expected)
        assert expected.exists()
        assert not observed["temporary"].exists()

    def test_atomic_save_preserves_existing_file_on_failure(
        self, tmp_path, monkeypatch
    ):
        destination = tmp_path / "converted.pt"
        destination.write_bytes(b"complete-old-checkpoint")

        def fail_after_partial_write(_value, temporary):
            Path(temporary).write_bytes(b"partial")
            raise RuntimeError("injected save failure")

        monkeypatch.setattr(autoconvert_module.torch, "save", fail_after_partial_write)

        with pytest.raises(RuntimeError, match="injected save failure"):
            autoconvert_module._atomic_torch_save({}, destination)

        assert destination.read_bytes() == b"complete-old-checkpoint"
        assert list(tmp_path.glob(f".{destination.name}.*.tmp")) == []

    def test_concurrent_writers_use_private_staging_files(
        self, tmp_path, monkeypatch
    ):
        destination = tmp_path / "converted.pt"
        barrier = threading.Barrier(2)
        real_save = torch.save
        staged_paths = []
        staged_paths_lock = threading.Lock()

        def synchronized_save(value, temporary, *args, **kwargs):
            with staged_paths_lock:
                staged_paths.append(Path(temporary))
            barrier.wait(timeout=5)
            real_save(value, temporary, *args, **kwargs)

        monkeypatch.setattr(autoconvert_module.torch, "save", synchronized_save)

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(
                    autoconvert_module._atomic_torch_save,
                    {"writer": torch.tensor([writer])},
                    destination,
                )
                for writer in range(2)
            ]
            for future in futures:
                future.result(timeout=10)

        assert len(set(staged_paths)) == 2
        assert all(path.parent == destination.parent for path in staged_paths)
        assert all(not path.exists() for path in staged_paths)
        assert torch.load(destination, weights_only=True)["writer"].item() in {0, 1}

    @pytest.mark.skipif(os.name == "nt", reason="POSIX file modes")
    def test_first_conversion_inherits_source_mode(self, tmp_path):
        src = tmp_path / "v9-t.pt"
        torch.save({"model": _synthetic_upstream_yolo9(nc=2)}, src)
        src.chmod(0o640)

        out = Path(autoconvert_upstream_checkpoint(str(src)))

        assert stat.S_IMODE(out.stat().st_mode) == 0o640

    @pytest.mark.skipif(os.name == "nt", reason="POSIX file modes")
    def test_reconversion_preserves_destination_mode(self, tmp_path):
        src = tmp_path / "v9-t.pt"
        destination = tmp_path / "v9-t-LibreYOLO9t.pt"
        torch.save({"model": _synthetic_upstream_yolo9(nc=2)}, src)
        destination.write_bytes(b"old conversion")
        destination.chmod(0o660)

        out = Path(autoconvert_upstream_checkpoint(str(src)))

        assert out == destination
        assert stat.S_IMODE(out.stat().st_mode) == 0o660

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

    def test_checkpoint_args_normalizes_dict_class_names_for_rfdetr_loader(self):
        loaded = {
            "args": argparse.Namespace(
                class_names={"0": "bolt", "1": "nut"},
                num_queries=100,
                group_detr=13,
            )
        }

        assert autoconvert_module._checkpoint_args(loaded) == {
            "class_names": ["bolt", "nut"],
            "num_queries": 100,
            "group_detr": 13,
        }

    def test_checkpoint_args_omits_sparse_dict_class_names(self):
        loaded = {
            "args": argparse.Namespace(
                class_names={0: "bolt", 2: "washer"},
                num_queries=100,
                group_detr=13,
            )
        }

        assert autoconvert_module._checkpoint_args(loaded) == {
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

    def test_rfdetr_single_logit_head_is_one_class(self):
        # A one-category upstream dataset yields a single-output head with no
        # spare slot (raw_nc 0); it must not fall back to COCO's 80 classes.
        assert autoconvert_module._rfdetr_class_metadata(
            {"args": argparse.Namespace(num_classes=1, class_names=["object"])},
            0,
        ) == (1, ["object"])

    def test_rfdetr_coco_metadata_maps_90_arch_classes_to_coco80(self):
        assert autoconvert_module._rfdetr_class_metadata(
            {"args": argparse.Namespace(dataset_file="coco")},
            90,
        )[0] == 80

    def test_rfdetr_bare_90_class_state_dict_maps_to_coco80(self):
        # A metadata-less upstream RF-DETR state_dict (no names, no dataset
        # hint) is the canonical Roboflow COCO-pretrained checkpoint, so its
        # 90 arch-classes normalize to LibreYOLO's COCO-80.
        assert autoconvert_module._rfdetr_class_metadata({}, 90)[0] == 80

    def test_rfdetr_90_class_with_non_coco_dataset_hint_not_coerced(self):
        # A non-COCO dataset hint (even without a names list) must NOT trigger
        # the bare-checkpoint COCO fallback; the custom 90-class head stands.
        assert autoconvert_module._rfdetr_class_metadata(
            {"args": argparse.Namespace(dataset_file="custom90")},
            90,
        )[0] == 90

    def test_rfdetr_90_class_with_non_string_dataset_hint_not_coerced(self):
        # A non-string dataset hint (e.g. a data-config dict) is still a hint;
        # the bare-checkpoint COCO fallback must NOT fire.
        assert autoconvert_module._rfdetr_class_metadata(
            {"data": {"path": "/datasets/custom90"}},
            90,
        )[0] == 90

    def test_rfdetr_90_class_with_explicit_num_classes_not_coerced(self):
        # Explicit class-count metadata marks the class space as declared, so
        # even without names it is not a bare checkpoint -> stays nc=90.
        assert autoconvert_module._rfdetr_class_metadata(
            {"args": argparse.Namespace(num_classes=90)},
            90,
        )[0] == 90

    def test_rfdetr_90_class_with_empty_dataset_container_is_bare_coco(self):
        # Empty placeholders ({} / []) are not real dataset hints, so an
        # otherwise-bare checkpoint still normalizes to COCO-80.
        assert autoconvert_module._rfdetr_class_metadata({"data": {}}, 90)[0] == 80
        assert autoconvert_module._rfdetr_class_metadata({"data": []}, 90)[0] == 80

    def test_rfdetr_explicit_nc80_is_honored_as_coco(self):
        # A checkpoint that explicitly declares 80 classes is COCO.
        assert autoconvert_module._rfdetr_class_metadata({"nc": 80}, 90)[0] == 80

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
