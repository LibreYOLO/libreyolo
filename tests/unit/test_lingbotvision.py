"""Unit tests for LibreLingBotVision semantic segmentation.

The family is fully native (no optional runtime dependency), so these run the
real architecture at a small imgsz on CPU. Size ``s`` is used for
instantiation-heavy tests; the larger sizes are covered by fabricated state
dicts (size detection is a cls_token shape read) — ``g`` is 1.1B parameters
and has no place in the unit tier.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

pytestmark = pytest.mark.unit

ALL_SIZES = ("s", "b", "l", "g")
EMBED_DIMS = {"s": 384, "b": 768, "l": 1024, "g": 1536}


def _make_semantic_yaml(root, n_images=4, size=64):
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


def _fake_state(size: str, nc: int = 150) -> dict:
    dim = EMBED_DIMS[size]
    return {
        "backbone.cls_token": torch.zeros(1, 1, dim),
        "backbone.storage_tokens": torch.zeros(1, 4, dim),
        # All sizes use head_dim 64, so the periods buffer is 64 // 4 = 16.
        "backbone.rope_embed.periods": torch.zeros(16),
        "predict.weight": torch.zeros(nc, dim, 1, 1),
        "predict.bias": torch.zeros(nc),
    }


class TestLingBotVisionMetadata:
    def test_family_and_tasks(self):
        from libreyolo.models.lingbotvision.model import LibreLingBotVision

        assert LibreLingBotVision.FAMILY == "lingbotvision"
        assert LibreLingBotVision.FILENAME_PREFIX == "LibreLingBotVision"
        assert LibreLingBotVision.SUPPORTED_TASKS == ("semantic",)
        assert LibreLingBotVision.DEFAULT_TASK == "semantic"
        assert set(LibreLingBotVision.INPUT_SIZES) == set(ALL_SIZES)
        assert all(v == 512 for v in LibreLingBotVision.INPUT_SIZES.values())

    def test_get_download_url_keeps_the_sem_suffix(self):
        from libreyolo.models.lingbotvision.model import LibreLingBotVision

        url = LibreLingBotVision.get_download_url("LibreLingBotVisions-sem.pt")
        assert url == (
            "https://huggingface.co/LibreYOLO/LibreLingBotVisions-sem/resolve/main/"
            "LibreLingBotVisions-sem.pt"
        )

    def test_can_load_detects_size_and_classes_real_net(self):
        from libreyolo.models.lingbotvision.model import LibreLingBotVision
        from libreyolo.models.lingbotvision.nn import LingBotVisionSemanticSegmenter

        net = LingBotVisionSemanticSegmenter(size="s", num_classes=7)
        state = net.state_dict()
        assert LibreLingBotVision.can_load(state)
        assert LibreLingBotVision.detect_size(state) == "s"
        assert LibreLingBotVision.detect_nb_classes(state) == 7

    @pytest.mark.parametrize("size", ALL_SIZES)
    def test_detect_size_from_fabricated_state(self, size):
        from libreyolo.models.lingbotvision.model import LibreLingBotVision

        state = _fake_state(size)
        assert LibreLingBotVision.can_load(state)
        assert LibreLingBotVision.detect_size(state) == size
        assert LibreLingBotVision.detect_nb_classes(state) == 150

    def test_can_load_rejects_dinov2_semantic_signature(self):
        from libreyolo.models.lingbotvision.model import LibreLingBotVision

        state = {
            "backbone.encoder.embeddings.cls_token": torch.zeros(1, 1, 384),
            "predict.weight": torch.zeros(19, 8, 1, 1),
        }
        assert not LibreLingBotVision.can_load(state)

    def test_can_load_rejects_segformer_signature(self):
        from libreyolo.models.lingbotvision.model import LibreLingBotVision

        state = {
            "decode_head.linear_fuse.weight": torch.zeros(256, 1024, 1, 1),
            "decode_head.classifier.weight": torch.zeros(150, 256, 1, 1),
            "encoder.stages.0.patch_embeddings.proj.weight": torch.zeros(32, 3, 7, 7),
        }
        assert not LibreLingBotVision.can_load(state)

    def test_can_load_rejects_upstream_backbone_only_checkpoint(self):
        """An upstream model.pt has no dense head; the factory must not claim
        it (the converter is the entry path for upstream weights)."""
        from libreyolo.models.lingbotvision.model import LibreLingBotVision

        state = _fake_state("s")
        del state["predict.weight"], state["predict.bias"]
        assert not LibreLingBotVision.can_load(state)

    def test_registered_families_reject_lingbotvision_signature(self):
        from libreyolo.models.base.model import BaseModel
        from libreyolo.models.lingbotvision.model import LibreLingBotVision

        state = _fake_state("s")
        for cls in BaseModel._registry:
            if cls is LibreLingBotVision:
                continue
            assert not cls.can_load(state), f"{cls.__name__} incorrectly claims LingBot-Vision weights"

    def test_dinov2_rejects_lingbotvision_signature_when_available(self):
        """LibreDINOv2's can_load matches backbone.* + predict.weight; the RoPE
        buffer must exclude us even when dinov2 registers (transformers dep)."""
        pytest.importorskip("transformers")
        from libreyolo.models import try_ensure_rfdetr

        try_ensure_rfdetr()
        from libreyolo.models.dinov2.model import LibreDINOv2

        assert not LibreDINOv2.can_load(_fake_state("s"))


class TestLingBotVisionForward:
    def test_eval_shape(self):
        from libreyolo.models.lingbotvision.nn import LingBotVisionSemanticSegmenter

        net = LingBotVisionSemanticSegmenter(size="s", num_classes=5).eval()
        with torch.no_grad():
            out = net(torch.rand(2, 3, 64, 64))
        assert out["semantic_logits"].shape == (2, 5, 4, 4)

    def test_train_loss_and_backward_head_only(self):
        from libreyolo.models.lingbotvision.nn import LingBotVisionSemanticSegmenter

        net = LingBotVisionSemanticSegmenter(size="s", num_classes=3).train()
        for p in net.backbone.parameters():
            p.requires_grad_(False)
        targets = torch.randint(0, 3, (2, 64, 64))
        out = net(torch.rand(2, 3, 64, 64), targets=targets)
        assert torch.isfinite(out["total_loss"])
        out["total_loss"].backward()
        assert net.predict.weight.grad is not None
        assert net.backbone.patch_embed.proj.weight.grad is None

    def test_all_ignored_targets_yield_finite_zero_loss(self):
        from libreyolo.models.lingbotvision.nn import (
            IGNORE_INDEX,
            LingBotVisionSemanticSegmenter,
        )

        net = LingBotVisionSemanticSegmenter(size="s", num_classes=3).train()
        targets = torch.full((1, 64, 64), IGNORE_INDEX)
        out = net(torch.rand(1, 3, 64, 64), targets=targets)
        assert torch.isfinite(out["total_loss"])
        assert float(out["total_loss"]) == 0.0
        assert out["total_loss"].grad_fn is not None

    def test_input_not_divisible_by_patch_raises(self):
        from libreyolo.models.lingbotvision.nn import LingBotVisionBackbone

        net = LingBotVisionBackbone(size="s").eval()
        with pytest.raises(ValueError, match="divisible"):
            net(torch.rand(1, 3, 60, 60))


class TestLingBotVisionWrapper:
    def test_wrapper_predict_returns_semantic_mask(self, tmp_path):
        from libreyolo.models.lingbotvision.model import LibreLingBotVision

        img_path = tmp_path / "img.jpg"
        Image.fromarray(np.random.randint(0, 255, (96, 128, 3), dtype=np.uint8)).save(img_path)

        model = LibreLingBotVision(size="s", nb_classes=4, device="cpu")
        results = model(str(img_path), imgsz=64)
        assert results.semantic_mask is not None
        assert tuple(results.semantic_mask.data.shape) == (96, 128)

    def test_wrapper_rejects_non_patch_imgsz(self, tmp_path):
        from libreyolo.models.lingbotvision.model import LibreLingBotVision

        img_path = tmp_path / "img.jpg"
        Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8)).save(img_path)
        model = LibreLingBotVision(size="s", nb_classes=4, device="cpu")
        with pytest.raises(ValueError, match="divisible"):
            model(str(img_path), imgsz=60)

    def test_wrapper_class_rebuild_only_touches_head(self):
        from libreyolo.models.lingbotvision.model import LibreLingBotVision

        model = LibreLingBotVision(size="s", nb_classes=10, device="cpu")
        backbone_before = model.model.backbone.patch_embed.proj.weight.clone()
        model._rebuild_for_new_classes(3)
        assert model.model.predict.out_channels == 3
        assert model.nb_classes == 3
        assert torch.equal(model.model.backbone.patch_embed.proj.weight, backbone_before)

    def test_wrong_task_raises(self):
        from libreyolo.models.lingbotvision.model import LibreLingBotVision

        with pytest.raises(ValueError, match="semantic"):
            LibreLingBotVision(size="s", task="detect")

    def test_openvino_semantic_predict_parity(self, tmp_path):
        pytest.importorskip("openvino")
        from libreyolo import LibreYOLO
        from libreyolo.models.lingbotvision.model import LibreLingBotVision

        torch.manual_seed(0)
        model = LibreLingBotVision(size="s", nb_classes=3, device="cpu")
        model.model.eval()
        image = np.random.default_rng(54).integers(
            0, 256, size=(64, 64, 3), dtype=np.uint8
        )
        native = model.predict(image, imgsz=64).semantic_mask.data
        artifact = model.export(
            format="openvino",
            output_path=str(tmp_path / "lingbotvision_openvino"),
            imgsz=64,
            dynamic=False,
            simplify=False,
        )
        actual = LibreYOLO(artifact, device="cpu").predict(image).semantic_mask.data

        agreement = (native == actual).float().mean().item()
        assert agreement > 0.95


def test_lingbotvision_checkpoint_round_trip(tmp_path):
    """A metadata-wrapped checkpoint loads back through the factory as the
    right family, size, and class count."""
    import libreyolo
    from libreyolo.models.lingbotvision.nn import LingBotVisionSemanticSegmenter
    from libreyolo.utils.serialization import wrap_libreyolo_checkpoint

    net = LingBotVisionSemanticSegmenter(size="s", num_classes=3)
    ckpt = wrap_libreyolo_checkpoint(
        net.state_dict(),
        model_family="lingbotvision",
        size="s",
        task="semantic",
        nc=3,
        names={0: "a", 1: "b", 2: "c"},
        imgsz=512,
    )
    path = tmp_path / "LibreLingBotVisions-sem.pt"
    torch.save(ckpt, path)

    model = libreyolo.LibreYOLO(str(path))
    assert model.__class__.__name__ == "LibreLingBotVision"
    assert model.size == "s"
    assert model.nb_classes == 3
    assert model.names[1] == "b"


@pytest.mark.slow
def test_lingbotvision_train_smoke(tmp_path):
    """One epoch on a tiny synthetic dataset: the shared semantic path wires
    up, the head trains (backbone frozen by default), a checkpoint lands."""
    from libreyolo.models.lingbotvision.model import LibreLingBotVision

    yaml_path = _make_semantic_yaml(tmp_path / "data")
    model = LibreLingBotVision(size="s", nb_classes=2, device="cpu")
    result = model.train(
        data=str(yaml_path),
        epochs=1,
        batch=2,
        imgsz=64,
        workers=0,
        project=str(tmp_path / "runs"),
        name="smoke",
        exist_ok=True,
        amp=False,
    )
    assert result.get("best_checkpoint") or result.get("last_checkpoint")


def test_unpublished_size_download_explains_itself():
    """``lingbotvision-g`` has no hosted checkpoint, so say so, do not 404.

    Sizes s/b/l are mirrored under ``LibreYOLO/``; g is the 1.1B teacher and
    never was. Left to the base implementation the user gets an HTTP failure
    against a repo that does not exist, which reads like an outage rather
    than a checkpoint that was never published.
    """
    from libreyolo.models.lingbotvision.model import LibreLingBotVision

    with pytest.raises(FileNotFoundError) as excinfo:
        LibreLingBotVision.get_download_url("LibreLingBotVisiong-sem.pt")
    message = str(excinfo.value)
    assert "publishes no weights" in message
    assert "lingbotvision-s" in message
    assert "lingbotvision-l" in message
    assert "lingbotvision-g" not in message.split("published sizes:")[-1]


@pytest.mark.parametrize("size", ["s", "b", "l"])
def test_published_sizes_still_resolve(size):
    """The guard must not swallow the sizes that are actually hosted."""
    from libreyolo.models.lingbotvision.model import LibreLingBotVision

    url = LibreLingBotVision.get_download_url(f"LibreLingBotVision{size}-sem.pt")
    assert url == (
        f"https://huggingface.co/LibreYOLO/LibreLingBotVision{size}-sem"
        f"/resolve/main/LibreLingBotVision{size}-sem.pt"
    )


@pytest.mark.parametrize(
    "foreign",
    ["LibreSegformerb0-sem.pt", "LibreDINOv2s-sem.pt", "LibreYOLO9t.pt", "LibreDOMEDETRs-aitod.pt"],
)
def test_download_hook_ignores_other_families(foreign):
    """The hook must claim only LingBot-Vision filenames.

    ``download_weights`` asks every registered family in turn and takes the
    first non-None answer, so a hook that raised on a foreign name would
    hijack that family's download with the wrong error.
    """
    from libreyolo.models.lingbotvision.model import LibreLingBotVision

    assert LibreLingBotVision.get_download_url(foreign) is None
