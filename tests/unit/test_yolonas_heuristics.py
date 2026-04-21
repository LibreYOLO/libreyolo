"""Unit tests for YOLO-NAS checkpoint heuristics and native model smoke paths."""

from pathlib import Path

import pytest
import torch

from libreyolo.models import LibreYOLO, _unwrap_state_dict
from libreyolo.models.yolonas.loss import PPYoloELoss
from libreyolo.models.yolonas.model import LibreYOLONAS
from libreyolo.models.yolonas.nn import LibreYOLONASModel
from libreyolo.models.yolox.model import LibreYOLOX
from libreyolo.models.yolo9.model import LibreYOLO9
from libreyolo.models.yolonas.utils import unwrap_yolonas_checkpoint

pytestmark = pytest.mark.unit

OFFICIAL_YOLONAS_S = Path("downloads/yolonas/yolo_nas_s_coco.pth")


def _make_yolonas_state_dict(width: int, num_classes: int = 80):
    return {
        "backbone.stem.conv.branch_3x3.conv.weight": torch.zeros(48, 3, 3, 3),
        "backbone.stem.conv.branch_1x1.weight": torch.zeros(48, 3, 1, 1),
        "backbone.stem.conv.rbr_reparam.weight": torch.zeros(48, 3, 3, 3),
        "heads.head1.cls_pred.weight": torch.zeros(num_classes, width, 1, 1),
        "heads.head1.reg_pred.weight": torch.zeros(68, width, 1, 1),
    }


class TestCheckpointUnwrap:
    def test_global_unwrap_supports_sg_net(self):
        state = {"net": {"foo": torch.tensor(1.0)}}
        assert _unwrap_state_dict(state) == state["net"]

    def test_global_unwrap_prefers_sg_ema_net(self):
        state = {
            "net": {"foo": torch.tensor(1.0)},
            "ema_net": {"bar": torch.tensor(2.0)},
        }
        assert _unwrap_state_dict(state) == state["ema_net"]

    def test_local_yolonas_unwrap_prefers_ema_net(self):
        state = {
            "net": {"foo": torch.tensor(1.0)},
            "ema_net": {"bar": torch.tensor(2.0)},
        }
        assert unwrap_yolonas_checkpoint(state) == state["ema_net"]


class TestYOLONASHeuristics:
    def test_can_load_positive(self):
        assert LibreYOLONAS.can_load(_make_yolonas_state_dict(width=64))

    def test_can_load_negative_yolox(self):
        yolox_state = {
            "backbone.backbone.stem.conv.conv.weight": torch.zeros(32, 3, 3, 3),
            "head.cls_preds.0.weight": torch.zeros(80, 32, 1, 1),
        }
        assert LibreYOLOX.can_load(yolox_state)
        assert not LibreYOLONAS.can_load(yolox_state)

    def test_can_load_negative_yolo9(self):
        yolo9_state = {
            "backbone.conv0.conv.weight": torch.zeros(32, 3, 3, 3),
            "backbone.elan1.cv1.conv.weight": torch.zeros(64, 32, 1, 1),
            "head.cv3.0.2.weight": torch.zeros(80, 64, 1, 1),
        }
        assert LibreYOLO9.can_load(yolo9_state)
        assert not LibreYOLONAS.can_load(yolo9_state)

    @pytest.mark.parametrize(
        ("width", "expected_size"),
        [(64, "s"), (96, "m"), (128, "l")],
    )
    def test_detect_size_from_head_width(self, width, expected_size):
        assert (
            LibreYOLONAS.detect_size(_make_yolonas_state_dict(width=width))
            == expected_size
        )

    def test_detect_size_missing_key(self):
        assert LibreYOLONAS.detect_size({}) is None

    def test_detect_nb_classes(self):
        assert LibreYOLONAS.detect_nb_classes(_make_yolonas_state_dict(64, 17)) == 17


class TestYOLONASNativeModel:
    def test_native_model_forward_shapes(self):
        model = LibreYOLONASModel("s")
        model.eval()

        with torch.no_grad():
            output = model(torch.randn(1, 3, 640, 640))

        decoded_boxes, decoded_scores = output[0]
        assert tuple(decoded_boxes.shape) == (1, 8400, 4)
        assert tuple(decoded_scores.shape) == (1, 8400, 80)

    @pytest.mark.skipif(
        not OFFICIAL_YOLONAS_S.exists(),
        reason="Official YOLO-NAS checkpoint not present in local downloads/",
    )
    def test_official_checkpoint_loads_cleanly(self):
        model = LibreYOLONASModel("s")
        ckpt = torch.load(OFFICIAL_YOLONAS_S, map_location="cpu", weights_only=False)[
            "net"
        ]
        missing, unexpected = model.load_state_dict(ckpt, strict=False)
        assert missing == []
        assert unexpected == []

    @pytest.mark.skipif(
        not OFFICIAL_YOLONAS_S.exists(),
        reason="Official YOLO-NAS checkpoint not present in local downloads/",
    )
    def test_factory_detects_official_checkpoint(self):
        model = LibreYOLO(str(OFFICIAL_YOLONAS_S), device="cpu")
        assert isinstance(model, LibreYOLONAS)
        assert model.size == "s"
        assert model.nb_classes == 80

    def test_rebuild_for_new_classes_replaces_class_heads(self):
        model = LibreYOLONAS(model_path=None, size="s", nb_classes=80, device="cpu")
        assert model.model.heads.head1.cls_pred.weight.shape[0] == 80

        model._rebuild_for_new_classes(5)

        assert model.nb_classes == 5
        assert model.model.heads.head1.cls_pred.weight.shape[0] == 5
        assert model.model.heads.head2.cls_pred.weight.shape[0] == 5
        assert model.model.heads.head3.cls_pred.weight.shape[0] == 5

    def test_loss_backward_with_synthetic_targets(self):
        model = LibreYOLONASModel("s", nb_classes=80)
        model.train()
        loss_fn = PPYoloELoss(num_classes=80)

        imgs = torch.randn(1, 3, 64, 64)
        targets = torch.zeros(1, 10, 5)
        targets[0, 0] = torch.tensor([0.0, 32.0, 32.0, 16.0, 20.0])

        outputs = model(imgs)
        loss, logs = loss_fn(outputs, targets)
        loss.backward()

        assert torch.isfinite(loss)
        assert tuple(logs.shape) == (4,)
        assert model.heads.head1.cls_pred.weight.grad is not None

    def test_factory_detects_local_training_checkpoint(self, tmp_path):
        wrapper = LibreYOLONAS(model_path=None, size="s", nb_classes=80, device="cpu")
        ckpt_path = tmp_path / "yolonas_train.pt"
        torch.save(
            {
                "model": wrapper.model.state_dict(),
                "nc": 80,
                "size": "s",
                "model_family": "yolonas",
                "names": {i: f"class_{i}" for i in range(80)},
            },
            ckpt_path,
        )

        loaded = LibreYOLO(str(ckpt_path), device="cpu")
        assert isinstance(loaded, LibreYOLONAS)
        assert loaded.size == "s"
        assert loaded.nb_classes == 80
