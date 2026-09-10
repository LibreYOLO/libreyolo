"""Marigold checkpoint routing and original-canvas output contracts."""

import numpy as np
import pytest
import torch
from torch import nn

from libreyolo import LibreYOLO
from libreyolo.models.marigold_v2 import model as module
from libreyolo.models.marigold_v2.config import VARIANTS, canonical_filename
from libreyolo.models.marigold_v2.convert import PROMPT_KEYS, VARIANT_KEY
from libreyolo.models.marigold_v2.nn import BASE_REPO, BASE_REVISION
from libreyolo.utils.serialization import wrap_libreyolo_checkpoint

pytestmark = pytest.mark.unit


class _Adapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.lora_A = nn.ModuleDict({"default": nn.Linear(3, 2, bias=False)})
        self.lora_B = nn.ModuleDict({"default": nn.Linear(2, 3, bias=False)})


def _components(*args, **kwargs):
    vae = nn.Module()
    vae.decoder = nn.Identity()
    transformer = nn.Module()
    block = nn.Module()
    block.attn = nn.Module()
    block.attn.to_q = _Adapter()
    transformer.transformer_blocks = nn.ModuleList([block])
    return vae, transformer


def packet(variant="log-stage2"):
    task = VARIANTS[variant].task
    state = {
        "Diffuser.transformer_blocks.0.attn.to_q.lora_A.default.weight": torch.ones(
            2, 3
        ),
        "Diffuser.transformer_blocks.0.attn.to_q.lora_B.default.weight": torch.ones(
            3, 2
        )
        * 2,
        PROMPT_KEYS[0]: torch.zeros(1, 4, 8),
        PROMPT_KEYS[1]: torch.ones(1, 4, dtype=torch.int64),
        VARIANT_KEY: torch.tensor(tuple(VARIANTS).index(variant)),
    }
    return wrap_libreyolo_checkpoint(
        state,
        model_family="marigold_v2",
        size="b",
        task=task,
        nc=1,
        names={0: task},
        imgsz=1024,
        variant=variant,
        base_model=BASE_REPO,
        base_revision=BASE_REVISION,
    )


def test_generic_factory_default_class_count_and_save_roundtrip(monkeypatch, tmp_path):
    monkeypatch.setattr(module, "build_components", _components)
    original_init = module.LibreMarigoldV2.__init__
    received = []

    def cpu_init(self, *args, **kwargs):
        received.append(kwargs["nb_classes"])
        original_init(self, *args, quantization="none", **kwargs)

    monkeypatch.setattr(module.LibreMarigoldV2, "__init__", cpu_init)
    source = tmp_path / "LibreMarigoldV2b-depth.pt"
    original_packet = packet()
    provenance = {
        "upstream_repo": "huawei-bayerlab/marigold-v2-0",
        "upstream_revision": "6fd6d1ca246c9d2d99a4d8ac375a4eccc87178ad",
        "omitted_training_tensors": [
            "iREPAStudentProjector.out__qwen_dit_hidden_state_-1.weight"
        ],
    }
    original_packet.update(provenance)
    torch.save(original_packet, source)
    model = LibreYOLO(str(source), device="cpu")
    assert received == [80]  # factory's legacy constructor default
    assert model.nb_classes == 1
    assert model.depth_encoding == "log_depth"
    assert model.variant == "log-stage2"
    destination = tmp_path / "renamed.pt"
    model.save(destination)
    saved = torch.load(destination, weights_only=True)
    assert {key: saved[key] for key in provenance} == provenance
    restored = LibreYOLO(str(destination), device="cpu")
    for name, value in model.model.trainable_state_dict().items():
        torch.testing.assert_close(value, restored.model.trainable_state_dict()[name])
    with pytest.raises(NotImplementedError, match="fine-tuning"):
        model.train(data="unused.yaml")


@pytest.mark.parametrize("variant", VARIANTS)
def test_variant_task_and_filename_agree(variant):
    cls = module.LibreMarigoldV2
    state = packet(variant)["model"]
    assert cls.can_load(state)
    assert cls.detect_size(state) == "b"
    assert cls.detect_checkpoint_task(state) == VARIANTS[variant].task
    filename = canonical_filename(variant)
    assert cls.detect_task_from_filename(filename) == VARIANTS[variant].task
    assert cls.detect_size_from_filename(filename) == "b"
    assert cls.get_download_url(filename)
    assert not cls.can_load({"backbone.weight": torch.zeros(2, 3)})


def test_reject_mismatched_checkpoint_before_loading_base(monkeypatch):
    def should_not_load(*args, **kwargs):
        raise AssertionError("base loading should not be reached")

    monkeypatch.setattr(module, "build_components", should_not_load)
    with pytest.raises(ValueError, match="requested task"):
        module.LibreMarigoldV2(
            packet(), task="normal", device="cpu", quantization="none"
        )
    wrong = packet()
    wrong["base_revision"] = "untrusted"
    with pytest.raises(ValueError, match="unpinned base"):
        module.LibreMarigoldV2(wrong, device="cpu", quantization="none")
    wrong = packet()
    wrong["variant"] = "log-stage1"
    with pytest.raises(ValueError, match="marker disagree"):
        module.LibreMarigoldV2(wrong, device="cpu", quantization="none")


def test_inference_defaults_are_shared_by_call_and_predict(monkeypatch):
    calls = []
    monkeypatch.setattr(
        module.BaseModel,
        "__call__",
        lambda self, source, **kwargs: calls.append(kwargs["imgsz"]),
    )
    model = object.__new__(module.LibreMarigoldV2)
    model("image.png")
    model.predict("image.png")
    model.predict("image.png", imgsz=512)
    assert calls == [0, 0, 512]


def test_normal_axes_and_post_resize_renormalization():
    model = object.__new__(module.LibreMarigoldV2)
    model.task = "normal"
    model.model = lambda image: torch.tensor(
        [[[[0.6, 0.0]], [[0.0, 0.6]], [[0.8, 0.8]]]]
    )
    output = model._forward(torch.empty(0))
    actual = model._postprocess(output, 0, 0, (3, 1))["normal"]
    np.testing.assert_allclose(actual[0, 0], [0.6, 0, -0.8], atol=1e-7)
    assert actual[0, -1, 1] < 0
    np.testing.assert_allclose(np.linalg.norm(actual, axis=-1), 1, atol=1e-7)


def test_albedo_clips_after_original_canvas_resizing():
    model = object.__new__(module.LibreMarigoldV2)
    model.task = "albedo"
    values = torch.tensor([[[[-1.0, 1.0]], [[0.2, 0.8]], [[0.3, 0.7]]]])
    actual = model._postprocess({"albedo": values}, 0, 0, (3, 1))["albedo"]
    np.testing.assert_allclose(actual[0, 1], [0, 0.5, 0.5], atol=1e-7)
    assert np.isfinite(actual).all() and actual.min() >= 0 and actual.max() <= 1


def test_validation_batch_preserves_single_image_conditioning():
    model = object.__new__(module.LibreMarigoldV2)
    model.task = "depth"
    seen = []

    def graph(images):
        seen.append(images.shape[0])
        # A multi-context graph would change conditioning with batch position.
        return images + torch.arange(images.shape[0])[:, None, None, None]

    model.model = graph
    images = torch.randn(2, 3, 16, 16)
    together = model._forward(images)["depth"]
    separately = torch.cat([model._forward(image[None])["depth"] for image in images])
    torch.testing.assert_close(together, separately)
    assert seen == [1, 1, 1, 1]
