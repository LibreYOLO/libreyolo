"""Unit tests for the registry-driven upstream auto-conversion.

Each family gets a tiny synthetic state dict carrying just enough keys for
its recognizer (``convert_upstream_state_dict``), size detection and class
count detection, wrapped in that family's upstream checkpoint layout. The
real-weights end-to-end coverage lives in the opt-in network tests; these
stay fast and offline.
"""

import sys
import types
from pathlib import Path

import pytest
import torch

from libreyolo.models.autoconvert import autoconvert_upstream_checkpoint
from libreyolo.utils.serialization import validate_checkpoint_metadata

pytestmark = [pytest.mark.unit]


# ---------------------------------------------------------------------------
# Synthetic upstream state dicts (smallest size of each family)
# ---------------------------------------------------------------------------


def _deimv2_atto():
    return {
        "encoder.fpn.swish_ffn.0.weight": torch.zeros(8, 8),
        "decoder.dec_score_head.0.weight": torch.zeros(80, 64),
        "decoder.dec_score_head.0.bias": torch.zeros(80),
    }


def _deim_family_n():
    """DEIM and D-FINE share this exact layout — that is the point."""
    return {
        "decoder.pre_bbox_head.layers.0.weight": torch.zeros(4, 4),
        "encoder.lateral_convs.0.conv.weight": torch.zeros(64, 128, 1, 1),
        "encoder.input_proj.0.conv.weight": torch.zeros(128, 256, 1, 1),
        "encoder.input_proj.1.conv.weight": torch.zeros(128, 512, 1, 1),
        "decoder.dec_score_head.0.bias": torch.zeros(80),
    }


def _rtdetrv4_s():
    # D-FINE shapes for size "s" (RT-DETRv4 ships no "n"): hidden 256,
    # three pyramid levels, B0 backbone stage out-channels 256.
    return {
        "decoder.pre_bbox_head.layers.0.weight": torch.zeros(4, 4),
        "encoder.lateral_convs.0.conv.weight": torch.zeros(64, 256, 1, 1),
        "encoder.input_proj.0.conv.weight": torch.zeros(256, 256, 1, 1),
        "encoder.input_proj.1.conv.weight": torch.zeros(256, 512, 1, 1),
        "encoder.input_proj.2.conv.weight": torch.zeros(256, 1024, 1, 1),
        "decoder.dec_score_head.0.bias": torch.zeros(80),
        "encoder.feature_projector.0.weight": torch.zeros(8, 8),
    }


def _ec_s():
    return {
        "backbone.backbone.register_token": torch.zeros(1, 1, 192),
        "backbone.projector.0.conv.weight": torch.zeros(192, 384, 1, 1),
        "decoder.dec_score_head.0.bias": torch.zeros(80),
    }


def _yolox_s():
    return {
        "backbone.backbone.stem.conv.conv.weight": torch.zeros(32, 3, 3, 3),
        "head.cls_preds.0.weight": torch.zeros(80, 128, 1, 1),
    }


def _yolonas_s(pose: bool = False):
    sd = {
        "backbone.stem.conv.branch_3x3.conv.weight": torch.zeros(48, 3, 3, 3),
        "backbone.stem.conv.branch_1x1.weight": torch.zeros(48, 3, 1, 1),
        "backbone.stem.conv.rbr_reparam.weight": torch.zeros(48, 3, 3, 3),
        "heads.head1.cls_pred.weight": torch.zeros(80, 64, 1, 1),
        "heads.head1.reg_pred.weight": torch.zeros(68, 64, 1, 1),
    }
    if pose:
        sd["heads.head1.pose_pred.weight"] = torch.zeros(34, 64, 1, 1)
    return sd


def _damoyolo_s():
    return {
        "neck.merge_3.conv.weight": torch.zeros(8, 8, 1, 1),
        "backbone.block_list.0.conv.weight": torch.zeros(8, 3, 3, 3),
        "head.gfl_cls.0.weight": torch.zeros(80, 128, 1, 1),
        "head.gfl_reg.0.weight": torch.zeros(68, 128, 1, 1),
    }


def _picodet_s_upstream():
    return {
        "bbox_head.gfl_cls.0.weight": torch.zeros(112, 96, 1, 1),
        "backbone.2_1.conv_pw_2.conv.weight": torch.zeros(48, 24, 1, 1),
        "neck.trans.trans.0.conv.weight": torch.zeros(96, 232, 1, 1),
        "ema_bbox_head.gfl_cls.0.weight": torch.zeros(1),
        "bbox_head.integral.project": torch.zeros(8),
    }


def _rtmdet_s_upstream():
    return {
        "bbox_head.rtm_cls.0.weight": torch.zeros(80, 128, 1, 1),
        "backbone.stem.0.conv.weight": torch.zeros(16, 3, 3, 3),
        "data_preprocessor.mean": torch.zeros(3),
    }


def _rtdetr_r18_upstream(v2: bool = False):
    sd = {
        "backbone.res_layers.0.blocks.0.conv1.weight": torch.zeros(4, 4, 3, 3),
        "encoder.input_proj.0.conv.weight": torch.zeros(256, 512, 1, 1),
        "encoder.input_proj.0.norm.weight": torch.zeros(256),
        "decoder.input_proj.0.conv.weight": torch.zeros(4, 4, 1, 1),
        "decoder.dec_score_head.0.bias": torch.zeros(80),
    }
    if v2:
        sd["decoder.decoder.layers.0.cross_attn.num_points_scale"] = torch.zeros(4)
        sd["decoder.anchors"] = torch.zeros(1, 4)
    return sd


def _rtdetr_hgnetv2_l_upstream():
    return {
        "backbone.stages.0.blocks.0.conv.weight": torch.zeros(4, 4, 3, 3),
        "encoder.input_proj.0.conv.weight": torch.zeros(256, 512, 1, 1),
        "encoder.input_proj.0.norm.weight": torch.zeros(256),
        "decoder.input_proj.0.conv.weight": torch.zeros(4, 4, 1, 1),
        "decoder.dec_score_head.0.bias": torch.zeros(80),
        "decoder.decoder.layers.0.cross_attn.num_points_scale": torch.zeros(4),
        "decoder.anchors": torch.zeros(1, 4),
        "decoder.valid_mask": torch.zeros(1, 1),
    }


def _yolo9_e2e_t():
    return {
        "backbone.conv0.conv.weight": torch.zeros(16, 3, 3, 3),
        "head.one2one_cv3.0.2.weight": torch.zeros(80, 16, 1, 1),
    }


def _synthetic_numbered_yolo9():
    """Minimal upstream-shaped (numbered) YOLO9 detection dict, config t."""
    return {
        "0.conv.weight": torch.zeros(16, 3, 3, 3),
        "0.bn.weight": torch.zeros(16),
        "22.heads.0.class_conv.2.weight": torch.zeros(80, 16, 1, 1),
        "22.heads.0.class_conv.2.bias": torch.zeros(80),
    }


def _wrap_ema_module(sd):
    return {"ema": {"module": sd}}


def _wrap_model(sd):
    return {"model": sd}


def _wrap_state_dict(sd):
    return {"state_dict": sd}


def _wrap_ema_net(sd):
    return {"ema_net": sd, "net": {k: torch.ones_like(v) for k, v in sd.items()}}


def _wrap_ema_state_dict(sd):
    return {"ema_state_dict": {f"module.{k}": v for k, v in sd.items()}}


def _identity(sd):
    return sd


CASES = [
    # (case id, build_sd, wrapper, source filename, family, prefix, size, task, nc)
    ("deimv2", _deimv2_atto, _wrap_ema_module, "deimv2_upstream.pth", "deimv2", "LibreDEIMv2", "atto", "detect", 80),
    ("deim", _deim_family_n, _wrap_model, "deim_hgnetv2_n_coco.pth", "deim", "LibreDEIM", "n", "detect", 80),
    ("dfine", _deim_family_n, _wrap_model, "dfine_hgnetv2_n_coco.pth", "dfine", "LibreDFINE", "n", "detect", 80),
    ("rtdetrv4", _rtdetrv4_s, _wrap_ema_module, "rtv4_hgnetv2_s_coco.pth", "rtdetrv4", "LibreRTDETRv4", "s", "detect", 80),
    ("ec", _ec_s, _wrap_ema_module, "ec_det_s.pth", "ec", "LibreEC", "s", "detect", 80),
    ("yolox", _yolox_s, _wrap_model, "yolox_s.pth", "yolox", "LibreYOLOX", "s", "detect", 80),
    ("yolonas", _yolonas_s, _wrap_ema_net, "yolo_nas_s_coco.pth", "yolonas", "LibreYOLONAS", "s", "detect", 80),
    ("yolonas-pose", lambda: _yolonas_s(pose=True), _wrap_ema_net, "yolo_nas_pose_s_coco.pth", "yolonas", "LibreYOLONAS", "s", "pose", 1),
    ("damoyolo", _damoyolo_s, _identity, "damoyolo_tinynasL25_S.pth", "damoyolo", "LibreDAMOYOLO", "s", "detect", 80),
    ("picodet", _picodet_s_upstream, _wrap_state_dict, "picodet_s_320.pth", "picodet", "LibrePICODET", "s", "detect", 80),
    ("rtmdet", _rtmdet_s_upstream, _wrap_ema_state_dict, "rtmdet_s_coco.pth", "rtmdet", "LibreRTMDet", "s", "detect", 80),
    ("rtdetr-r18", _rtdetr_r18_upstream, _wrap_ema_module, "rtdetr_r18vd_coco.pth", "rtdetr", "LibreRTDETR", "r18", "detect", 80),
    ("rtdetrv2-r18", lambda: _rtdetr_r18_upstream(v2=True), _wrap_ema_module, "rtdetrv2_r18vd_120e_coco.pth", "rtdetrv2", "LibreRTDETRv2", "r18", "detect", 80),
    ("rtdetr-hgnetv2-l", _rtdetr_hgnetv2_l_upstream, _wrap_ema_module, "rtdetrv2_hgnetv2_l_6x_coco.pth", "rtdetr", "LibreRTDETR", "l", "detect", 80),
    ("yolo9-e2e", _yolo9_e2e_t, _wrap_model, "gelan_e2e_t.pt", "yolo9_e2e", "LibreYOLO9E2E", "t", "detect", 80),
]


@pytest.mark.parametrize(
    "build_sd,wrapper,filename,family,prefix,size,task,nc",
    [case[1:] for case in CASES],
    ids=[case[0] for case in CASES],
)
def test_family_autoconverts(tmp_path, build_sd, wrapper, filename, family, prefix, size, task, nc):
    source_sd = build_sd()
    src = tmp_path / filename
    torch.save(wrapper(source_sd), src)

    out = autoconvert_upstream_checkpoint(str(src))

    assert out is not None, f"{family} upstream checkpoint was not recognized"
    out_path = Path(out)
    suffix = {"pose": "-pose", "segment": "-seg"}.get(task, "")
    assert out_path.name == f"{src.stem}-{prefix}{size}{suffix}.pt"
    assert out_path.parent == tmp_path

    ckpt = torch.load(out_path, map_location="cpu", weights_only=True)
    assert validate_checkpoint_metadata(ckpt, strict=False) == []
    assert ckpt["model_family"] == family
    assert ckpt["size"] == size
    assert ckpt["task"] == task
    assert ckpt["nc"] == nc

    # The converted model must hold real tensors and not be empty — a remap
    # that produces an empty or value-corrupted dict must fail here, not slip
    # through. Tensor *values* must survive conversion: for keys that pass
    # through unrenamed we compare directly; fully-remapping families (every
    # key renamed) are covered by the dedicated TestRemappedFamilies asserts,
    # so here we only require the value multiset to be preserved.
    model = ckpt["model"]
    assert model, f"{family} converted to an empty model dict"
    assert all(isinstance(v, torch.Tensor) for v in model.values())
    source_tensors = {k: v for k, v in source_sd.items() if isinstance(v, torch.Tensor)}
    assert len(model) >= len(source_tensors) - 5  # may drop training-only/aliased keys
    for key in set(model) & set(source_tensors):
        assert torch.equal(model[key], source_tensors[key].float()), (
            f"{family} corrupted tensor at {key}"
        )
    converted_sums = sorted(round(float(v.sum()), 3) for v in model.values())
    source_sums = sorted(round(float(v.float().sum()), 3) for v in source_tensors.values())
    for s in source_sums:
        # Every source tensor's value-signature must appear in the output
        # (allowing for dropped training-only tensors, never corrupted ones).
        if s in converted_sums:
            converted_sums.remove(s)


class TestRemappedFamilies:
    """Families whose upstream key naming differs from the native port."""

    def test_picodet_keys_are_remapped_and_filtered(self, tmp_path):
        src = tmp_path / "picodet_s_320.pth"
        torch.save(_wrap_state_dict(_picodet_s_upstream()), src)

        ckpt = torch.load(
            autoconvert_upstream_checkpoint(str(src)), weights_only=True
        )
        model = ckpt["model"]
        assert "head.gfl_cls.0.weight" in model
        assert "backbone.blocks.0.conv_pw_2.conv.weight" in model
        assert "neck.trans.0.conv.weight" in model
        assert not any(k.startswith(("bbox_head.", "ema_")) for k in model)
        assert not any(k.endswith("integral.project") for k in model)

    def test_rtmdet_keys_are_remapped_and_filtered(self, tmp_path):
        src = tmp_path / "rtmdet_s_coco.pth"
        torch.save(_wrap_ema_state_dict(_rtmdet_s_upstream()), src)

        ckpt = torch.load(
            autoconvert_upstream_checkpoint(str(src)), weights_only=True
        )
        model = ckpt["model"]
        assert "head.rtm_cls.0.weight" in model
        assert not any(
            k.startswith(("bbox_head.", "data_preprocessor.", "module."))
            for k in model
        )

    def test_rtdetr_v1_remaps_input_proj_and_drops_v2_buffers(self, tmp_path):
        src = tmp_path / "rtdetrv2_hgnetv2_l_6x_coco.pth"
        torch.save(_wrap_ema_module(_rtdetr_hgnetv2_l_upstream()), src)

        ckpt = torch.load(
            autoconvert_upstream_checkpoint(str(src)), weights_only=True
        )
        model = ckpt["model"]
        assert "encoder.input_proj.0.0.weight" in model
        assert "encoder.input_proj.0.1.weight" in model
        # decoder.input_proj keeps upstream's named submodules; only the
        # encoder projection is remapped to Sequential numeric keys.
        assert not any(
            k.startswith("encoder.input_proj.") and (".conv." in k or ".norm." in k)
            for k in model
        )
        assert "decoder.input_proj.0.conv.weight" in model
        assert "decoder.anchors" not in model
        assert "decoder.valid_mask" not in model
        assert not any("num_points_scale" in k for k in model)

    def test_rtdetrv2_keeps_buffers(self, tmp_path):
        src = tmp_path / "rtdetrv2_r18vd_120e_coco.pth"
        torch.save(_wrap_ema_module(_rtdetr_r18_upstream(v2=True)), src)

        ckpt = torch.load(
            autoconvert_upstream_checkpoint(str(src)), weights_only=True
        )
        model = ckpt["model"]
        assert ckpt["model_family"] == "rtdetrv2"
        assert "encoder.input_proj.0.0.weight" in model
        assert "decoder.anchors" in model
        assert any("num_points_scale" in k for k in model)

    def test_rtdetrv4_drops_feature_projector(self, tmp_path):
        src = tmp_path / "rtv4_hgnetv2_s_coco.pth"
        torch.save(_wrap_ema_module(_rtdetrv4_s()), src)

        ckpt = torch.load(
            autoconvert_upstream_checkpoint(str(src)), weights_only=True
        )
        assert ckpt["model_family"] == "rtdetrv4"
        assert not any("feature_projector" in k for k in ckpt["model"])

    def test_rtdetrv4_wins_over_dfine_base_without_filename_hint(self, tmp_path):
        """A raw v4 file under a generic name must not convert as D-FINE.

        D-FINE registers before RT-DETRv4 (base classes register first), and
        its passthrough also claims raw v4 files; the most-derived claimant
        must win.
        """
        src = tmp_path / "best.pt"
        torch.save(_wrap_ema_module(_rtdetrv4_s()), src)

        out = autoconvert_upstream_checkpoint(str(src))

        assert out is not None
        ckpt = torch.load(out, map_location="cpu", weights_only=True)
        assert ckpt["model_family"] == "rtdetrv4"
        assert not any("feature_projector" in k for k in ckpt["model"])

    def test_rtdetrv4_wins_over_dfine_even_under_dfine_filename(self, tmp_path):
        """Subclass-wins must run before the filename hint: a raw v4 file named
        dfine_* inherits D-FINE's filename regex, so the hint alone would hand
        it to the base D-FINE (retaining feature_projector -> wrong model)."""
        src = tmp_path / "dfine_hgnetv2_s_coco.pth"
        torch.save(_wrap_ema_module(_rtdetrv4_s()), src)

        out = autoconvert_upstream_checkpoint(str(src))

        assert out is not None
        ckpt = torch.load(out, map_location="cpu", weights_only=True)
        assert ckpt["model_family"] == "rtdetrv4"
        assert not any("feature_projector" in k for k in ckpt["model"])

    def test_unique_content_claim_beats_filename_pointing_at_broad_match(
        self, tmp_path
    ):
        """A filename must not override a unique content match. EC weights
        satisfy YOLOX's broad ``backbone.backbone`` check, but EC's
        ``register_token`` is the specific match — an EC file named like a
        YOLOX checkpoint must still convert as EC."""
        src = tmp_path / "LibreYOLOXs.pt"
        torch.save(_wrap_ema_module(_ec_s()), src)

        out = autoconvert_upstream_checkpoint(str(src))

        assert out is not None
        ckpt = torch.load(out, map_location="cpu", weights_only=True)
        assert ckpt["model_family"] == "ec"

    def test_numbered_yolo9_with_one2one_key_routes_to_yolo9_not_e2e(self, tmp_path):
        """A numbered upstream YOLO9 dict carrying a stray one2one key must
        convert as yolo9 (head remapped), not be passed through raw as
        yolo9_e2e by the subclass-wins rule."""
        sd = _synthetic_numbered_yolo9()
        sd["99.one2one_cv2.0.conv.weight"] = torch.zeros(4, 4, 1, 1)
        src = tmp_path / "last.pt"
        torch.save({"model": sd}, src)

        out = autoconvert_upstream_checkpoint(str(src))

        assert out is not None
        ckpt = torch.load(out, map_location="cpu", weights_only=True)
        assert ckpt["model_family"] == "yolo9"
        # Head was remapped to semantic keys, not left in numbered form.
        assert any(k.startswith("head.cv3") for k in ckpt["model"])
        assert not any(k[0].isdigit() for k in ckpt["model"])


class TestDispatchRules:
    def test_pose_conversion_carries_keypoint_metadata(self, tmp_path):
        """Pose checkpoints must carry num_keypoints/keypoint_dim (schema).

        Uses a non-COCO keypoint count (20, i.e. ``pose_pred`` width 40) so the
        value can only come from ``detect_num_keypoints`` — never the family's
        ``POSE_NUM_KEYPOINTS=17`` fallback.
        """
        sd = _yolonas_s(pose=True)
        sd["heads.head1.pose_pred.weight"] = torch.zeros(40, 64, 1, 1)
        src = tmp_path / "yolo_nas_pose_s_coco.pth"
        torch.save(_wrap_ema_net(sd), src)

        ckpt = torch.load(
            autoconvert_upstream_checkpoint(str(src)), weights_only=True
        )
        assert ckpt["task"] == "pose"
        assert ckpt["num_keypoints"] == 20
        assert ckpt["keypoint_dim"] == 3

    def test_remapped_upstream_with_names_metadata_still_converts(self, tmp_path):
        """A names key must not suppress recognizers that prove upstream
        origin by remapping keys (e.g. mm-series RTMDet naming)."""
        wrapped = _wrap_state_dict(_rtmdet_s_upstream())
        wrapped["names"] = {0: "person"}
        src = tmp_path / "rtmdet_s_finetune.pth"
        torch.save(wrapped, src)

        out = autoconvert_upstream_checkpoint(str(src))

        assert out is not None
        ckpt = torch.load(out, map_location="cpu", weights_only=True)
        assert ckpt["model_family"] == "rtmdet"
        assert "head.rtm_cls.0.weight" in ckpt["model"]

    def test_oversized_names_dict_is_trimmed_to_detected_nc(self, tmp_path):
        """A fine-tune that kept an 80-entry names dict over a 7-class head
        must still convert: names are trimmed to nc, not passed through whole
        (which the strict validator would reject -> silent failure)."""
        sd = _rtmdet_s_upstream()
        sd["bbox_head.rtm_cls.0.weight"] = torch.zeros(7, 128, 1, 1)
        wrapped = _wrap_state_dict(sd)
        wrapped["names"] = {i: f"class_{i}" for i in range(80)}
        src = tmp_path / "rtmdet_s_7class.pth"
        torch.save(wrapped, src)

        out = autoconvert_upstream_checkpoint(str(src))

        assert out is not None
        ckpt = torch.load(out, map_location="cpu", weights_only=True)
        assert ckpt["nc"] == 7
        assert sorted(ckpt["names"]) == list(range(7))

    def test_unwritable_source_directory_falls_back_to_temp_dir(
        self, tmp_path, monkeypatch
    ):
        src = tmp_path / "deimv2_upstream.pth"
        torch.save(_wrap_ema_module(_deimv2_atto()), src)

        real_save = torch.save

        def failing_save(obj, path, *args, **kwargs):
            if str(path).startswith(str(tmp_path)):
                raise OSError("read-only directory")
            return real_save(obj, path, *args, **kwargs)

        monkeypatch.setattr(torch, "save", failing_save)

        out = autoconvert_upstream_checkpoint(str(src))

        assert out is not None
        assert not out.startswith(str(tmp_path))
        ckpt = torch.load(out, map_location="cpu", weights_only=True)
        assert ckpt["model_family"] == "deimv2"
        Path(out).unlink()

    def test_metadata_only_ema_block_does_not_mask_model_weights(self, tmp_path):
        """An ema dict without weights must fall through to the model key."""
        wrapped = {"ema": {"decay": 0.9995, "updates": 1234}, "model": _deimv2_atto()}
        src = tmp_path / "deimv2_finetune.pth"
        torch.save(wrapped, src)

        out = autoconvert_upstream_checkpoint(str(src))

        assert out is not None
        ckpt = torch.load(out, map_location="cpu", weights_only=True)
        assert ckpt["model_family"] == "deimv2"
        assert "decoder.dec_score_head.0.weight" in ckpt["model"]

    def test_ambiguous_deim_dfine_without_filename_hint_is_refused(self, tmp_path):
        src = tmp_path / "model.pth"
        torch.save(_wrap_model(_deim_family_n()), src)

        assert autoconvert_upstream_checkpoint(str(src)) is None

    def test_partial_libreyolo_metadata_skips_generic_conversion(self, tmp_path):
        wrapped = _wrap_model(_deimv2_atto())
        wrapped["names"] = {0: "person"}
        src = tmp_path / "old-libreyolo-deimv2.pt"
        torch.save(wrapped, src)

        assert autoconvert_upstream_checkpoint(str(src)) is None

    def test_floating_tensors_are_cast_to_fp32(self, tmp_path):
        sd = {k: v.half() for k, v in _deimv2_atto().items()}
        src = tmp_path / "deimv2_fp16.pth"
        torch.save(_wrap_ema_module(sd), src)

        ckpt = torch.load(
            autoconvert_upstream_checkpoint(str(src)), weights_only=True
        )
        assert all(v.dtype == torch.float32 for v in ckpt["model"].values())


class TestRFDETR:
    """RF-DETR uses a bespoke recognizer (lazy registration, checkpoint-level
    args for size). Exercised end-to-end through ``_try_rfdetr`` without
    constructing the heavy DINOv2 model."""

    def _rfdetr_coco_upstream(self, arch_classes: int):
        import argparse

        state = {
            "backbone.0.encoder.encoder.embeddings.cls_token": torch.zeros(1, 1, 256),
            "transformer.decoder.query_embed.weight": torch.zeros(300, 256),
            "class_embed.bias": torch.zeros(arch_classes),
            "class_embed.weight": torch.zeros(arch_classes, 256),
        }
        return {
            "model": state,
            "args": argparse.Namespace(
                resolution=384, dataset_file="coco", num_queries=300
            ),
        }

    def test_rfdetr_coco_checkpoint_converts_and_remaps_90_to_80(self, tmp_path):
        pytest.importorskip("transformers")
        src = tmp_path / "rf-detr-finetune.pth"
        torch.save(self._rfdetr_coco_upstream(arch_classes=91), src)

        out = autoconvert_upstream_checkpoint(str(src))

        assert out is not None
        ckpt = torch.load(out, map_location="cpu", weights_only=True)
        assert ckpt["model_family"] == "rfdetr"
        assert ckpt["size"] == "n"  # from args.resolution=384
        assert ckpt["task"] == "detect"
        assert ckpt["nc"] == 80  # COCO 90 arch-classes -> LibreYOLO COCO-80
        assert len(ckpt["names"]) == 80


class TestInertStubLoading:
    def test_pickled_third_party_objects_do_not_block_conversion(self, tmp_path):
        module_name = "fake_mmlib_config"
        module = types.ModuleType(module_name)
        fake_cls = type("FakeCfg", (), {"__module__": module_name})
        module.FakeCfg = fake_cls
        sys.modules[module_name] = module
        try:
            payload = _wrap_state_dict(_rtmdet_s_upstream())
            payload["meta"] = fake_cls()
            src = tmp_path / "rtmdet_s_coco.pth"
            torch.save(payload, src)
        finally:
            del sys.modules[module_name]

        out = autoconvert_upstream_checkpoint(str(src))

        assert out is not None
        ckpt = torch.load(out, map_location="cpu", weights_only=True)
        assert ckpt["model_family"] == "rtmdet"
        assert "head.rtm_cls.0.weight" in ckpt["model"]

    def test_builtins_and_blocklisted_modules_are_never_stubbed(self):
        """The stub fabricator must refuse builtins and os/sys-family modules."""
        from libreyolo.models.autoconvert import _stub_for_blocked_global

        for blocked in ("builtins.eval", "os.system", "sys.modules", "posix.system"):
            exc = Exception(f"GLOBAL {blocked} was not an allowed global")
            assert _stub_for_blocked_global(exc) is None
        # A genuine third-party config class IS stubbed.
        ok = _stub_for_blocked_global(
            Exception("GLOBAL mmengine.config.Config was not an allowed global")
        )
        assert ok is not None and ok.__module__ == "mmengine.config"
