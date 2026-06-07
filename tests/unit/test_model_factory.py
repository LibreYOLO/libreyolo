"""Unit tests for model factory heuristics."""

import torch
import pytest

from libreyolo import LibreYOLO
from libreyolo.models import _needs_rfdetr_registration
from libreyolo.models.base.model import BaseModel
from libreyolo.models.yolo9.model import LibreYOLO9
from libreyolo.models.yolo9.nn import LibreYOLO9Model
from libreyolo.utils.serialization import wrap_libreyolo_checkpoint

pytestmark = pytest.mark.unit


def test_rfdetr_lazy_registration_detects_enc_out_markers():
    weights_dict = {
        "transformer.enc_out_class_embed.0.weight": object(),
        "transformer.enc_out_bbox_embed.0.layers.0.weight": object(),
    }

    assert _needs_rfdetr_registration(weights_dict) is True


def test_rfdetr_lazy_registration_detects_classifier_signature():
    weights_dict = {
        "backbone.encoder.encoder.embeddings.position_embeddings": object(),
        "linear.weight": object(),
    }

    assert _needs_rfdetr_registration(weights_dict) is True


def test_rfdetr_lazy_registration_ignores_rtdetr_signature():
    weights_dict = {
        "backbone.stages.0.conv.weight": object(),
        "encoder.input_proj.0.0.weight": object(),
        "decoder.input_proj.0.conv.weight": object(),
        "decoder.dec_score_head.0.weight": object(),
    }

    assert _needs_rfdetr_registration(weights_dict) is False


@pytest.mark.parametrize("device_arg", ["0", 0])
def test_base_model_normalizes_bare_numeric_device(device_arg):
    class _NoopModule(torch.nn.Module):
        def to(self, device):
            self.moved_to = device
            return self

    class _DeviceDummy(BaseModel):
        FAMILY = "_device_dummy"
        FILENAME_PREFIX = "_DeviceDummy"
        INPUT_SIZES = {"n": 32}

        def _init_model(self):
            self.inner = _NoopModule()
            return self.inner

        def _get_available_layers(self):
            return {}

        @staticmethod
        def _get_preprocess_numpy():
            return None

        def _preprocess(self, image, color_format="auto", input_size=None):
            raise NotImplementedError

        def _forward(self, input_tensor):
            raise NotImplementedError

        def _postprocess(
            self,
            output,
            conf_thres,
            iou_thres,
            original_size,
            max_det=300,
            ratio=1.0,
            **kwargs,
        ):
            raise NotImplementedError

    try:
        model = _DeviceDummy(None, size="n", device=device_arg)

        assert str(model.device) == "cuda:0"
        assert model.inner.moved_to == torch.device("cuda:0")
    finally:
        if _DeviceDummy in BaseModel._registry:
            BaseModel._registry.remove(_DeviceDummy)


def test_factory_loads_yolo9_t_metadata_checkpoint_with_coco_class_width(tmp_path):
    model = LibreYOLO9Model(config="t", nb_classes=80)

    # Mimic a fine-tuned checkpoint saved from a COCO-width YOLO9-t model:
    # only the final class conv is rebuilt to 2 classes, while the class
    # branch hidden width stays at 80.
    for seq in model.head.cv3:
        in_channels = seq[-1].weight.shape[1]
        seq[-1] = torch.nn.Conv2d(in_channels, 2, 1)

    ckpt_path = tmp_path / "yolo9_t_best.pt"
    torch.save(
        wrap_libreyolo_checkpoint(
            model.state_dict(),
            model_family="yolo9",
            size="t",
            task="detect",
            nc=2,
            names={0: "red", 1: "white"},
            imgsz=640,
        ),
        ckpt_path,
    )

    loaded = LibreYOLO(str(ckpt_path), size="t", device="cpu")

    assert loaded.nb_classes == 2
    assert loaded.names == {0: "red", 1: "white"}
    assert loaded.model.head.cv3[0][0].conv.weight.shape[0] == 80


def test_factory_loads_raw_yolo9_pose_checkpoint_with_custom_keypoints(tmp_path):
    model = LibreYOLO9Model(config="t", nb_classes=1, pose=True, num_keypoints=5)
    ckpt_path = tmp_path / "LibreYOLO9t-pose.pt"
    torch.save(model.state_dict(), ckpt_path)

    loaded = LibreYOLO(str(ckpt_path), device="cpu")

    assert loaded.task == "pose"
    assert loaded.nb_classes == 1
    assert loaded.num_keypoints == 5
    assert loaded.model.head.num_keypoints == 5


def test_yolo9_obb_transfer_accepts_same_family_detect_checkpoint(tmp_path):
    detect_model = LibreYOLO9Model(config="t", nb_classes=80)
    ckpt_path = tmp_path / "LibreYOLO9t.pt"
    torch.save(
        wrap_libreyolo_checkpoint(
            detect_model.state_dict(),
            model_family="yolo9",
            size="t",
            task="detect",
            nc=80,
            imgsz=640,
        ),
        ckpt_path,
    )

    target = LibreYOLO9(None, size="t", task="obb", nb_classes=6, device="cpu")
    stats = target._load_transfer_weights(ckpt_path)

    assert target.task == "obb"
    assert target.nb_classes == 6
    assert target.model.head.cv3[0][0].conv.weight.shape[0] == 80
    assert hasattr(target.model.head, "cv4")
    assert stats["loaded"] > 0
    assert stats["skipped"] > 0
    torch.testing.assert_close(
        target.model.state_dict()["backbone.conv0.conv.weight"],
        detect_model.state_dict()["backbone.conv0.conv.weight"],
    )


def test_yolo9_obb_direct_load_rejects_detect_checkpoint(tmp_path):
    detect_model = LibreYOLO9Model(config="t", nb_classes=80)
    ckpt_path = tmp_path / "LibreYOLO9t.pt"
    torch.save(
        wrap_libreyolo_checkpoint(
            detect_model.state_dict(),
            model_family="yolo9",
            size="t",
            task="detect",
            nc=80,
            imgsz=640,
        ),
        ckpt_path,
    )

    with pytest.raises(RuntimeError, match="task='detect'"):
        LibreYOLO9(str(ckpt_path), size="t", task="obb", device="cpu")


def test_yolo9_train_pretrained_true_uses_default_transfer_after_data_classes(
    tmp_path, monkeypatch
):
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                f"path: {tmp_path.as_posix()}",
                "train: images/train",
                "val: images/val",
                "nc: '6'",
                "names: [bike, bus, car, other_vehicle, taxi, truck]",
            ]
        ),
        encoding="utf-8",
    )

    model = LibreYOLO9(None, size="t", task="obb", device="cpu")
    transfer_call = {}

    def fake_load_transfer(weights):
        transfer_call["weights"] = weights
        transfer_call["classes"] = model.nb_classes
        return {"loaded": 10, "skipped": 2}

    monkeypatch.setattr(model, "_load_transfer_weights", fake_load_transfer)

    trainer_init = {}

    class FakeTrainer:
        def __init__(self, **kwargs):
            trainer_init.update(kwargs)

        def train(self):
            return {"final_loss": 0.0}

    import libreyolo.models.yolo9.trainer as trainer_module

    monkeypatch.setattr(trainer_module, "YOLO9Trainer", FakeTrainer)

    result = model.train(data=str(data_yaml), pretrained=True, epochs=1)

    assert result == {"final_loss": 0.0}
    assert transfer_call == {"weights": "LibreYOLO9t.pt", "classes": 6}
    assert model.nb_classes == 6
    assert trainer_init["num_classes"] == 6
    assert trainer_init["wrapper_model"] is model


def test_factory_loads_yolo9_obb_scratch_checkpoint_with_custom_class_width(tmp_path):
    model = LibreYOLO9Model(config="t", nb_classes=1, obb=True)

    ckpt_path = tmp_path / "LibreYOLO9t-obb.pt"
    torch.save(
        wrap_libreyolo_checkpoint(
            model.state_dict(),
            model_family="yolo9",
            size="t",
            task="obb",
            nc=1,
            names={0: "ship"},
            imgsz=64,
        ),
        ckpt_path,
    )

    loaded = LibreYOLO(str(ckpt_path), device="cpu")

    assert loaded.FAMILY == "yolo9"
    assert loaded.task == "obb"
    assert loaded.nb_classes == 1
    assert loaded.names == {0: "ship"}
    assert loaded.model.training is False
    assert loaded.model.head.cv3[0][0].conv.weight.shape[0] == 64


def test_factory_resolves_metadata_less_yolo9_obb_from_filename(tmp_path):
    model = LibreYOLO9Model(config="t", nb_classes=1, obb=True)
    ckpt_path = tmp_path / "LibreYOLO9t-obb.pt"
    torch.save(model.state_dict(), ckpt_path)

    loaded = LibreYOLO(str(ckpt_path), device="cpu")

    assert loaded.FAMILY == "yolo9"
    assert loaded.task == "obb"
    assert loaded.nb_classes == 1


def test_factory_resolves_metadata_less_yolo9_obb_from_angle_head(tmp_path):
    model = LibreYOLO9Model(config="t", nb_classes=1, obb=True)
    ckpt_path = tmp_path / "best.pt"
    torch.save(model.state_dict(), ckpt_path)

    loaded = LibreYOLO(str(ckpt_path), size="t", device="cpu")

    assert loaded.FAMILY == "yolo9"
    assert loaded.task == "obb"
    assert loaded.nb_classes == 1


def test_factory_resolves_metadata_less_yolo9_segment_from_proto_head(tmp_path):
    model = LibreYOLO9Model(config="t", nb_classes=1, segmentation=True)
    ckpt_path = tmp_path / "best.pt"
    torch.save(model.state_dict(), ckpt_path)

    loaded = LibreYOLO(str(ckpt_path), size="t", device="cpu")

    assert loaded.FAMILY == "yolo9"
    assert loaded.task == "segment"
    assert loaded.nb_classes == 1


def test_factory_warns_for_legacy_libreyolo_metadata_checkpoint(tmp_path, caplog):
    model = LibreYOLO9Model(config="t", nb_classes=80)
    ckpt_path = tmp_path / "LibreYOLO9t.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "nc": 80,
            "names": {i: f"class_{i}" for i in range(80)},
            "model_family": "yolo9",
            "size": "t",
        },
        ckpt_path,
    )

    loaded = LibreYOLO(str(ckpt_path), size="t", device="cpu")

    assert loaded.nb_classes == 80
    assert "legacy compatibility path" in caplog.text


def test_factory_warns_for_foreign_metadata_less_checkpoint(tmp_path, caplog):
    model = LibreYOLO9Model(config="t", nb_classes=80)
    ckpt_path = tmp_path / "upstream_yolo9.pt"
    torch.save(model.state_dict(), ckpt_path)

    loaded = LibreYOLO(str(ckpt_path), size="t", device="cpu")

    assert loaded.FAMILY == "yolo9"
    assert "LibreYOLO metadata was not found" in caplog.text


def test_factory_rejects_unsupported_explicit_task_from_filename():
    with pytest.raises(ValueError, match="not supported"):
        LibreYOLO("LibreYOLOXs.pt", task="segment")
