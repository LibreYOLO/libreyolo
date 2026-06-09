"""Unit tests for RF-DETR semantic segmentation.

Structural tests run against a lightweight fake backbone (monkeypatched
``build_backbone``) so they stay hermetic; one real-backbone forward test is
network-marked for nightly runs.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

pytestmark = pytest.mark.unit


def _fake_backbone_factory(hidden_dim: int, num_levels: int):
    from libreyolo.models.rfdetr.nn import NestedTensor

    class _FakeBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Conv2d(3, hidden_dim, 14, stride=14)

        def forward(self, nested):
            x = self.proj(nested.tensors)
            levels = [x]
            for _ in range(num_levels - 1):
                x = F.max_pool2d(x, 2, ceil_mode=True)
                levels.append(x)
            return [NestedTensor(t, None) for t in levels]

    return _FakeBackbone()


@pytest.fixture
def fake_backbone(monkeypatch):
    """Replace the DINOv2 backbone build with a tiny conv pyramid."""
    import libreyolo.models.rfdetr.nn as rfdetr_nn

    def _build(load_dinov2_weights=True, **kwargs):
        backbone = _fake_backbone_factory(
            kwargs["hidden_dim"], len(kwargs["projector_scale"])
        )
        return nn.Sequential(backbone, nn.Identity())

    monkeypatch.setattr(rfdetr_nn, "build_backbone", _build)
    return _build


class TestSemanticMetadata:
    def test_task_registration(self):
        from libreyolo.models.rfdetr.model import LibreRFDETR

        assert "semantic" in LibreRFDETR.SUPPORTED_TASKS
        assert LibreRFDETR.TASK_INPUT_SIZES["semantic"]["n"] == 518
        assert LibreRFDETR.semantic_resize_mode == "stretch"

    def test_can_load_recognizes_semantic_signature(self):
        from libreyolo.models.rfdetr.model import LibreRFDETR

        state = {
            "backbone.encoder.proj.weight": torch.zeros(1),
            "predict.weight": torch.zeros(3, 8, 1, 1),
        }
        assert LibreRFDETR.can_load(state)

    def test_detect_task_from_source_prefers_metadata_then_signature(self):
        from libreyolo.models.rfdetr.model import LibreRFDETR

        assert (
            LibreRFDETR._detect_task_from_source({"task": "semantic", "model": {}})
            == "semantic"
        )
        signature_ckpt = {
            "model": {
                "backbone.encoder.proj.weight": torch.zeros(1),
                "predict.weight": torch.zeros(3, 8, 1, 1),
            }
        }
        assert LibreRFDETR._detect_task_from_source(signature_ckpt) == "semantic"


class TestSemanticSegmenter:
    def test_forward_loss_and_eval_shapes(self, fake_backbone):
        from libreyolo.models.rfdetr.nn import RFDETRSemanticSegmenter

        model = RFDETRSemanticSegmenter(config="n", nb_classes=3)
        x = torch.rand(2, 3, 70, 70)

        model.train()
        targets = torch.randint(0, 3, (2, 70, 70))
        targets[:, :8, :] = 255
        out = model(x, targets=targets)
        assert set(out) == {"total_loss", "sem"}
        assert torch.isfinite(out["total_loss"])
        out["total_loss"].backward()
        assert model.predict.weight.grad is not None

        model.eval()
        with torch.no_grad():
            logits = model(x)
        assert logits.shape == (2, 3, 70, 70)

    def test_one_task_head_at_a_time(self, fake_backbone):
        from libreyolo.models.rfdetr.nn import LibreRFDETRModel

        with pytest.raises(ValueError, match="one task head"):
            LibreRFDETRModel(config="n", semantic=True, classification=True)

    def test_wrapper_predict_returns_semantic_mask(self, fake_backbone, tmp_path):
        from libreyolo.models.rfdetr.model import LibreRFDETR

        img_path = tmp_path / "img.jpg"
        Image.new("RGB", (90, 45), color=(50, 90, 130)).save(img_path)

        m = LibreRFDETR(
            model_path=None, size="n", task="semantic", nb_classes=3, device="cpu"
        )
        assert m.task == "semantic"
        assert m.input_size == 518

        result = m.predict(str(img_path), imgsz=70)

        assert result.boxes is None
        assert result.semantic_mask is not None
        assert tuple(result.semantic_mask.data.shape) == (45, 90)

    def test_wrapper_class_rebuild(self, fake_backbone):
        from libreyolo.models.rfdetr.model import LibreRFDETR

        m = LibreRFDETR(
            model_path=None, size="n", task="semantic", nb_classes=3, device="cpu"
        )
        m._rebuild_for_new_classes(5)

        m.model.eval()
        with torch.no_grad():
            logits = m.model(torch.rand(1, 3, 70, 70))
        assert logits.shape == (1, 5, 70, 70)


def _make_semantic_yaml(root, n_images=4, size=70):
    import yaml as _yaml

    for split in ("train", "val"):
        for i in range(n_images):
            img_dir = root / "images" / split
            img_dir.mkdir(parents=True, exist_ok=True)
            arr = np.zeros((size, size, 3), dtype=np.uint8)
            arr[:, : size // 2] = (200, 40, 40)
            arr[:, size // 2 :] = (40, 40, 200)
            Image.fromarray(arr).save(img_dir / f"img{i}.jpg")
            mask = np.zeros((size, size), dtype=np.uint8)
            mask[:, size // 2 :] = 1
            mask_dir = root / "masks" / split
            mask_dir.mkdir(parents=True, exist_ok=True)
            Image.fromarray(mask, mode="L").save(mask_dir / f"img{i}.png")
    yaml_path = root / "data.yaml"
    yaml_path.write_text(
        _yaml.safe_dump(
            {
                "path": str(root),
                "train": "images/train",
                "val": "images/val",
                "masks_dir": "masks",
                "nc": 2,
                "names": {0: "left", 1: "right"},
            }
        )
    )
    return yaml_path


def test_rfdetr_semantic_train_smoke(fake_backbone, tmp_path):
    """One epoch through the shared trainer with the stub backbone."""
    from libreyolo.models.rfdetr.model import LibreRFDETR

    yaml_path = _make_semantic_yaml(tmp_path)
    m = LibreRFDETR(
        model_path=None, size="n", task="semantic", nb_classes=2, device="cpu"
    )

    res = m.train(
        data=str(yaml_path),
        epochs=1,
        batch=2,
        imgsz=70,
        workers=0,
        eval_interval=1,
        project=str(tmp_path / "runs"),
        name="sem_smoke",
        exist_ok=True,
        amp=False,
        ema=False,
        warmup_epochs=0,
    )

    assert np.isfinite(res["epoch_losses"][0])
    assert res["epoch_metrics"][-1]["val_metrics"].get("metrics/mIoU") is not None


@pytest.mark.external_data
@pytest.mark.network
@pytest.mark.slow
def test_rfdetr_semantic_forward_real_backbone():
    """RF-DETR semantic build + forward (DINOv2 backbone; random-init if offline)."""
    from libreyolo import LibreRFDETR

    m = LibreRFDETR(
        model_path=None, size="n", task="semantic", nb_classes=4, device="cpu"
    )
    assert m.task == "semantic"
    assert m.input_size == 518
    assert m.model.semantic

    x = torch.rand(1, 3, 518, 518)
    m.model.train()
    out = m.model(x, targets=torch.randint(0, 4, (1, 518, 518)))
    assert "total_loss" in out

    m.model.eval()
    with torch.no_grad():
        assert m.model(x).shape == (1, 4, 518, 518)
