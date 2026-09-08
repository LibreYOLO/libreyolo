"""Unit tests for LibreDINOv2 semantic segmentation.

Structural tests run against a lightweight fake backbone (monkeypatched
``build_backbone``) so they stay hermetic; one real-backbone forward test is
network-marked for nightly runs.
"""

from __future__ import annotations

import os

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


class TestDINOv2Metadata:
    def test_task_registration(self):
        from libreyolo.models.dinov2.model import LibreDINOv2

        assert "semantic" in LibreDINOv2.SUPPORTED_TASKS
        assert "embed" in LibreDINOv2.SUPPORTED_TASKS
        assert LibreDINOv2.INPUT_SIZES["n"] == 518
        assert LibreDINOv2.TASK_INPUT_SIZES["embed"]["n"] == 224
        assert LibreDINOv2.detect_size_from_filename("LibreDINOv2n-embed.pt") is None
        assert LibreDINOv2.semantic_resize_mode == "stretch"

    def test_family_is_dinov2(self):
        from libreyolo.models.dinov2.model import LibreDINOv2

        assert LibreDINOv2.FAMILY == "dinov2"
        assert LibreDINOv2.FILENAME_PREFIX == "LibreDINOv2"

    def test_can_load_recognizes_semantic_signature(self):
        from libreyolo.models.dinov2.model import LibreDINOv2

        state = {
            "backbone.encoder.proj.weight": torch.zeros(1),
            "predict.weight": torch.zeros(3, 8, 1, 1),
        }
        assert LibreDINOv2.can_load(state)

    def test_can_load_rejects_detection_signature(self):
        from libreyolo.models.dinov2.model import LibreDINOv2

        state = {
            "backbone.encoder.proj.weight": torch.zeros(1),
            "class_embed.bias": torch.zeros(81),
        }
        assert not LibreDINOv2.can_load(state)

    def test_rfdetr_no_longer_claims_semantic(self):
        """LibreRFDETR.can_load must return False for semantic-only key sets."""
        from libreyolo.models.rfdetr.model import LibreRFDETR

        state = {
            "backbone.encoder.proj.weight": torch.zeros(1),
            "predict.weight": torch.zeros(3, 8, 1, 1),
        }
        assert not LibreRFDETR.can_load(state)

    def test_rfdetr_supported_tasks_excludes_semantic(self):
        from libreyolo.models.rfdetr.model import LibreRFDETR

        assert "semantic" not in LibreRFDETR.SUPPORTED_TASKS


class TestDINOv2SemanticSegmenter:
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

    def test_wrapper_predict_returns_semantic_mask(self, fake_backbone, tmp_path):
        from libreyolo.models.dinov2.model import LibreDINOv2

        img_path = tmp_path / "img.jpg"
        Image.new("RGB", (90, 45), color=(50, 90, 130)).save(img_path)

        m = LibreDINOv2(
            model_path=None, size="n", task="semantic", nb_classes=3, device="cpu"
        )
        assert m.task == "semantic"
        assert m.input_size == 518

        result = m.predict(str(img_path), imgsz=70)

        assert result.boxes is None
        assert result.semantic_mask is not None
        assert tuple(result.semantic_mask.data.shape) == (45, 90)

    def test_wrapper_predict_augment_returns_semantic_mask(
        self, fake_backbone, tmp_path
    ):
        from libreyolo.models.dinov2.model import LibreDINOv2

        img_path = tmp_path / "img.jpg"
        Image.new("RGB", (90, 45), color=(50, 90, 130)).save(img_path)

        m = LibreDINOv2(
            model_path=None, size="n", task="semantic", nb_classes=3, device="cpu"
        )
        result = m.predict(str(img_path), imgsz=70, augment=True)

        assert result.boxes is None
        assert result.semantic_mask is not None
        assert tuple(result.semantic_mask.data.shape) == (45, 90)

    def test_wrapper_class_rebuild(self, fake_backbone):
        from libreyolo.models.dinov2.model import LibreDINOv2

        m = LibreDINOv2(
            model_path=None, size="n", task="semantic", nb_classes=3, device="cpu"
        )
        m._rebuild_for_new_classes(5)

        m.model.eval()
        with torch.no_grad():
            logits = m.model(torch.rand(1, 3, 70, 70))
        assert logits.shape == (1, 5, 70, 70)

    def test_wrong_task_raises(self):
        from libreyolo.models.dinov2.model import LibreDINOv2

        with pytest.raises(ValueError, match="semantic"):
            LibreDINOv2(
                model_path=None, size="n", task="detect", nb_classes=3, device="cpu"
            )

    @pytest.mark.parametrize("format", ["onnx", "torchscript", "openvino"])
    def test_exported_semantic_parity(self, fake_backbone, tmp_path, format):
        if format == "onnx":
            pytest.importorskip("onnx")
            pytest.importorskip("onnxruntime")
        if format == "openvino":
            pytest.importorskip("openvino")

        from libreyolo import LibreYOLO
        from libreyolo.models.dinov2.model import LibreDINOv2

        model = LibreDINOv2(
            model_path=None, size="n", task="semantic", nb_classes=3, device="cpu"
        )
        model.model.eval()
        image = np.random.default_rng(13).integers(
            0, 256, size=(70, 70, 3), dtype=np.uint8
        )
        native = model.predict(image, imgsz=70).semantic_mask.data
        artifact = tmp_path / {
            "onnx": "dinov2_semantic.onnx",
            "torchscript": "dinov2_semantic.torchscript",
            "openvino": "dinov2_semantic_openvino",
        }[format]
        model.export(
            format=format,
            output_path=str(artifact),
            imgsz=70,
            dynamic=False,
            simplify=False,
        )
        exported = LibreYOLO(str(artifact), device="cpu").predict(image)
        agreement = (native == exported.semantic_mask.data).float().mean().item()
        assert agreement > 0.95


def test_dinov2_classify_torchscript_predict_parity(fake_backbone, tmp_path):
    from libreyolo import LibreYOLO
    from libreyolo.models.dinov2.model import LibreDINOv2

    torch.manual_seed(7)
    model = LibreDINOv2(
        model_path=None,
        size="n",
        task="classify",
        nb_classes=3,
        device="cpu",
    )
    image = np.random.default_rng(7).integers(
        0, 256, size=(180, 260, 3), dtype=np.uint8
    )
    native = model.predict(image, imgsz=224)
    artifact = tmp_path / "dinov2_classify.torchscript"

    model.export(
        format="torchscript",
        output_path=str(artifact),
        imgsz=224,
    )
    exported = LibreYOLO(str(artifact), device="cpu").predict(image)

    torch.testing.assert_close(
        exported.probs.data,
        native.probs.data,
        rtol=1e-5,
        atol=1e-6,
    )


class TestDINOv2Embed:
    def test_wrapper_uses_final_cls_token(self, monkeypatch):
        from types import SimpleNamespace

        import libreyolo.models.rfdetr.nn as rfdetr_nn
        from libreyolo.models.dinov2.model import _DINOv2EmbedderWrapper

        class _Embeddings(nn.Module):
            def forward(self, images):
                batch = images.shape[0]
                tokens = torch.full((batch, 3, 4), 10.0)
                tokens[:, 0] = torch.tensor([1.0, 2.0, 3.0, 4.0])
                return tokens

        class _Encoder(nn.Module):
            def forward(self, hidden, **_):
                return (hidden + 1.0,)

        class _Dino(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = SimpleNamespace(hidden_size=4)
                self.embeddings = _Embeddings()
                self.encoder = _Encoder()
                self.layernorm = nn.Identity()

        class _DinoContainer(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = _Dino()

        class _Backbone(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = _DinoContainer()
                self.projector = nn.Linear(4, 4)

        class _Classifier:
            resolution = 224
            patch_size = 14
            num_windows = 1

            def __init__(self, **_):
                self.backbone = _Backbone()

        monkeypatch.setattr(rfdetr_nn, "RFDETRClassifier", _Classifier)
        wrapper = _DINOv2EmbedderWrapper("n")
        output = wrapper(torch.zeros(2, 3, 16, 16))

        assert wrapper.embedding_dim == 4
        assert output.shape == (2, 4)
        assert not any("projector" in key for key in wrapper.state_dict())
        assert torch.equal(
            output,
            torch.tensor([[2.0, 3.0, 4.0, 5.0]]).repeat(2, 1),
        )

    def test_wrapper_matches_canonical_hf_forward(self):
        """The hand-rolled embeddings->encoder->layernorm->CLS path must stay
        equivalent to the canonical model forward at num_windows=1."""
        from libreyolo.models.dinov2.model import _DINOv2EmbedderWrapper
        from libreyolo.models.rfdetr.dinov2 import (
            WindowedDinov2WithRegistersBackbone,
            WindowedDinov2WithRegistersConfig,
            WindowedDinov2WithRegistersModel,
        )

        config = WindowedDinov2WithRegistersConfig(
            image_size=28,
            patch_size=14,
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_register_tokens=2,
            num_windows=1,
            out_indices=[1],
        )
        torch.manual_seed(0)
        backbone = WindowedDinov2WithRegistersBackbone(config).eval()
        canonical = WindowedDinov2WithRegistersModel(config).eval()
        canonical.load_state_dict(backbone.state_dict())

        wrapper = object.__new__(_DINOv2EmbedderWrapper)
        nn.Module.__init__(wrapper)
        container = nn.Module()
        container.encoder = backbone
        wrapper.backbone = nn.Module()
        wrapper.backbone.encoder = container

        x = torch.randn(2, 3, 28, 28)
        with torch.no_grad():
            ours = wrapper(x)
            expected = canonical(x).pooler_output
        assert ours.shape == (2, 16)
        torch.testing.assert_close(ours, expected)

    def test_predict_and_embed_verb_contract(self, monkeypatch):
        from libreyolo.models.dinov2.model import LibreDINOv2

        class _FakeEmbedder(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = nn.Conv2d(3, 6, 1, bias=False)

            def forward(self, images):
                return self.backbone(images).mean(dim=(2, 3))

        monkeypatch.setattr(
            LibreDINOv2,
            "_init_model",
            lambda self: _FakeEmbedder(),
        )
        model = LibreDINOv2(
            model_path=None,
            size="n",
            task="embed",
            device="cpu",
        )
        image_a = Image.new("RGB", (40, 30), color=(120, 30, 10))
        image_b = Image.new("RGB", (40, 30), color=(10, 30, 120))

        result = model(image_a)
        assert result.boxes is None
        assert result.embeddings.data.shape == (1, 6)
        assert torch.allclose(
            result.embeddings.data.norm(dim=-1),
            torch.ones(1),
            atol=1e-5,
        )
        assert model.embed([image_a, image_b]).shape == (2, 6)
        assert not hasattr(model, "embed_text")

    @pytest.mark.parametrize("format", ["onnx", "tflite"])
    def test_train_val_are_out_of_scope_and_embed_export_routes(
        self, monkeypatch, format
    ):
        from libreyolo.models.base.model import BaseModel
        from libreyolo.models.dinov2.model import LibreDINOv2

        model = object.__new__(LibreDINOv2)
        model.task = "embed"

        with pytest.raises(NotImplementedError, match="training is not implemented"):
            model.train(data="unused")
        with pytest.raises(NotImplementedError, match="retrieval validation"):
            model.val(data="unused")

        captured = {}

        def fake_export(self, format="onnx", **kwargs):
            captured.update(format=format, **kwargs)
            return f"dinov2-embed.{format}"

        monkeypatch.setattr(BaseModel, "export", fake_export)
        assert model.export(format=format, dynamic=False) == (
            f"dinov2-embed.{format}"
        )
        assert captured == {"format": format, "opset": 17, "dynamic": False}

    def test_classify_unsupported_export_keeps_classify_error(self):
        from libreyolo.models.dinov2.model import LibreDINOv2

        model = object.__new__(LibreDINOv2)
        model.task = "classify"
        with pytest.raises(NotImplementedError, match="classify export"):
            model.export(format="tflite")


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


def test_dinov2_semantic_train_smoke(fake_backbone, tmp_path):
    """One epoch through the shared trainer with the stub backbone."""
    from libreyolo.models.dinov2.model import LibreDINOv2

    yaml_path = _make_semantic_yaml(tmp_path)
    m = LibreDINOv2(
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


def test_dinov2_checkpoint_family_is_dinov2(fake_backbone, tmp_path):
    """Trainer must save model_family='dinov2' (not 'rfdetr')."""
    from libreyolo.models.dinov2.model import LibreDINOv2
    from libreyolo.utils.serialization import load_trusted_torch_file

    yaml_path = _make_semantic_yaml(tmp_path)
    m = LibreDINOv2(
        model_path=None, size="n", task="semantic", nb_classes=2, device="cpu"
    )
    res = m.train(
        data=str(yaml_path),
        epochs=1,
        batch=2,
        imgsz=70,
        workers=0,
        eval_interval=0,
        project=str(tmp_path / "runs"),
        name="ckpt_family",
        exist_ok=True,
        amp=False,
        ema=False,
        warmup_epochs=0,
    )
    ckpt_path = res.get("best_checkpoint") or res.get("last_checkpoint")
    assert ckpt_path is not None
    ckpt = load_trusted_torch_file(ckpt_path, map_location="cpu", context="test")
    assert ckpt.get("model_family") == "dinov2"


def test_dinov2_semantic_resume_continues_same_run_dir(fake_backbone, tmp_path):
    """resume=True must actually invoke the trainer's resume lifecycle and
    keep writing into the same run_dir, not increment away from it."""
    from libreyolo.models.dinov2.model import LibreDINOv2

    yaml_path = _make_semantic_yaml(tmp_path)
    runs_root = tmp_path / "runs"
    run_dir = runs_root / "resume_test"

    m1 = LibreDINOv2(
        model_path=None, size="n", task="semantic", nb_classes=2, device="cpu"
    )
    res1 = m1.train(
        data=str(yaml_path),
        epochs=1,
        batch=2,
        imgsz=70,
        workers=0,
        eval_interval=0,
        project=str(runs_root),
        name="resume_test",
        amp=False,
        ema=False,
        warmup_epochs=0,
    )
    assert run_dir.exists()
    assert res1.get("last_checkpoint") is not None

    m2 = LibreDINOv2(
        model_path=None, size="n", task="semantic", nb_classes=2, device="cpu"
    )
    res2 = m2.train(
        data=str(yaml_path),
        epochs=2,
        batch=2,
        imgsz=70,
        workers=0,
        eval_interval=0,
        project=str(runs_root),
        name="resume_test",
        resume=True,
        amp=False,
        ema=False,
        warmup_epochs=0,
    )

    # exist_ok must have been forced True for the resumed run: no sibling
    # "resume_test2" directory should exist alongside the original.
    assert not (runs_root / "resume_test2").exists()
    # The resume lifecycle actually ran: start_epoch was restored to 1, so
    # only epoch index 1 trained, not epochs 0 and 1 from scratch.
    assert len(res2["epoch_losses"]) == 1


@pytest.mark.external_data
@pytest.mark.network
@pytest.mark.slow
def test_dinov2_semantic_forward_real_backbone():
    """LibreDINOv2 build + forward (DINOv2 backbone; random-init if offline)."""
    from libreyolo.models.dinov2.model import LibreDINOv2

    m = LibreDINOv2(
        model_path=None, size="n", task="semantic", nb_classes=4, device="cpu"
    )
    assert m.task == "semantic"
    assert m.input_size == 518

    x = torch.rand(1, 3, 518, 518)
    m.model.train()
    out = m.model(x, targets=torch.randint(0, 4, (1, 518, 518)))
    assert "total_loss" in out

    m.model.eval()
    with torch.no_grad():
        assert m.model(x).shape == (1, 4, 518, 518)


@pytest.mark.external_data
@pytest.mark.network
@pytest.mark.slow
@pytest.mark.skipif(
    os.environ.get("LIBREYOLO_RUN_REAL_EXPORT_PARITY") != "1",
    reason="set LIBREYOLO_RUN_REAL_EXPORT_PARITY=1 for real DINOv2 export parity",
)
@pytest.mark.parametrize(("task", "imgsz"), [("semantic", 518), ("classify", 224)])
@pytest.mark.parametrize("format", ["onnx", "torchscript"])
def test_dinov2_real_export_raw_parity(tmp_path, task, imgsz, format):
    if format == "onnx":
        pytest.importorskip("onnx")
        pytest.importorskip("onnxruntime")

    from libreyolo import LibreDINOv2, LibreYOLO
    from libreyolo.export.exporter import OnnxExporter

    torch.manual_seed(0)
    model = LibreDINOv2(
        model_path=None, size="n", task=task, nb_classes=3, device="cpu"
    )
    model.model.eval()
    tensor = torch.rand(1, 3, imgsz, imgsz)
    exporter = OnnxExporter(model)
    with exporter._model_context("cpu", False, False, 1, (imgsz, imgsz)) as (
        wrapped,
        _,
    ):
        with torch.no_grad():
            expected = wrapped(tensor)
    if isinstance(expected, torch.Tensor):
        expected = (expected,)

    artifact = model.export(
        format=format,
        imgsz=imgsz,
        dynamic=False,
        simplify=False,
        output_path=str(tmp_path / f"dinov2-{task}.{format}"),
    )
    actual = LibreYOLO(artifact, device="cpu")._run_inference(tensor.numpy())

    assert len(actual) == len(expected)
    rtol, atol = (2e-3, 2e-2) if format == "onnx" else (1e-3, 1e-3)
    for actual_output, expected_output in zip(actual, expected):
        np.testing.assert_allclose(
            actual_output,
            expected_output.detach().cpu().numpy(),
            rtol=rtol,
            atol=atol,
        )


def test_all_ignore_targets_yield_finite_zero_loss(fake_backbone):
    from libreyolo.models.rfdetr.nn import RFDETRSemanticSegmenter

    model = RFDETRSemanticSegmenter(config="n", nb_classes=3)
    model.train()
    out = model(
        torch.rand(1, 3, 70, 70),
        targets=torch.full((1, 70, 70), 255, dtype=torch.long),
    )

    assert torch.isfinite(out["total_loss"])
    assert float(out["total_loss"]) == 0.0


def test_dinov2_semantic_predict_rejects_non_patch_imgsz(fake_backbone, tmp_path):
    from libreyolo.models.dinov2.model import LibreDINOv2

    img_path = tmp_path / "img.jpg"
    Image.new("RGB", (64, 64), color=(10, 20, 30)).save(img_path)
    m = LibreDINOv2(
        model_path=None, size="n", task="semantic", nb_classes=2, device="cpu"
    )

    with pytest.raises(ValueError, match="divisible by 14"):
        m.predict(str(img_path), imgsz=100)


def test_dinov2_semantic_train_rejects_non_patch_imgsz(
    fake_backbone, tmp_path, monkeypatch
):
    from libreyolo.models.dinov2.model import LibreDINOv2

    monkeypatch.chdir(tmp_path)
    yaml_path = _make_semantic_yaml(tmp_path)
    m = LibreDINOv2(
        model_path=None, size="n", task="semantic", nb_classes=2, device="cpu"
    )

    with pytest.raises(ValueError, match="divisible by 14"):
        m.train(
            data=str(yaml_path),
            epochs=1,
            batch=2,
            imgsz=64,
            workers=0,
            eval_interval=0,
            project=str(tmp_path / "runs"),
            name="bad_imgsz",
            exist_ok=True,
            amp=False,
            ema=False,
            warmup_epochs=0,
        )


def test_dinov2_semantic_rejects_lora(fake_backbone, tmp_path, monkeypatch):
    from libreyolo.models.dinov2.model import LibreDINOv2

    monkeypatch.chdir(tmp_path)
    yaml_path = _make_semantic_yaml(tmp_path)
    m = LibreDINOv2(
        model_path=None, size="n", task="semantic", nb_classes=2, device="cpu"
    )

    with pytest.raises(ValueError, match="lora"):
        m.train(
            data=str(yaml_path),
            epochs=1,
            batch=2,
            imgsz=70,
            workers=0,
            eval_interval=0,
            project=str(tmp_path / "runs"),
            name="lora_reject",
            exist_ok=True,
            amp=False,
            ema=False,
            warmup_epochs=0,
            lora=True,
        )
