"""RF-DETR stays traceable by ``torch.compile`` without graph breaks or per-size recompiles.

Eager behavior is unchanged by the compile-specific paths; these tests pin
that the compile paths exist and that eager does not take them.
"""

import subprocess
import sys

import pytest
import torch
from torch._dynamo.utils import counters

pytestmark = pytest.mark.unit

pytest.importorskip("transformers", reason="RF-DETR needs the rfdetr extra")


@pytest.fixture(scope="module")
def nano():
    from libreyolo.models.rfdetr.nn import LibreRFDETRModel

    torch.manual_seed(0)
    return LibreRFDETRModel(config="n", nb_classes=2)


def test_import_leaves_global_dynamo_config_alone():
    """Importing libreyolo must not change torch.compile behavior process-wide."""
    code = (
        "import torch\n"
        "cfg = torch._dynamo.config\n"
        "before = (cfg.automatic_dynamic_shapes, cfg.accumulated_cache_size_limit)\n"
        "import libreyolo\n"
        "import libreyolo.models.deimv2.engine.backbone.dinov3.layers.block\n"
        "after = (cfg.automatic_dynamic_shapes, cfg.accumulated_cache_size_limit)\n"
        "assert before == after, (before, after)\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.mark.parametrize("mode", ["train", "eval"])
def test_native_resolution_traces_as_one_graph(nano, mode):
    getattr(nano, mode)()
    torch._dynamo.reset()
    explanation = torch._dynamo.explain(nano)(torch.randn(1, 3, 384, 384))
    assert explanation.graph_break_count == 0, [
        reason.reason for reason in explanation.break_reasons
    ]
    assert explanation.graph_count == 1


def test_multi_scale_sizes_share_one_dynamic_compile(nano):
    nano.eval()
    torch._dynamo.reset()
    counters.clear()
    compiled = torch.compile(nano, backend="eager", dynamic=True)
    with torch.no_grad():
        compiled(torch.randn(1, 3, 320, 320))
        frames = counters["frames"]["total"]
        for size in (352, 416, 448):
            compiled(torch.randn(1, 3, size, size))
    assert counters["frames"]["total"] == frames


def test_eager_spatial_shapes_keep_the_cache(nano):
    transformer = nano.model.transformer
    shapes = transformer._cached_spatial_shapes([(24, 24)], torch.device("cpu"))
    assert torch.equal(shapes, torch.tensor([[24, 24]]))
    assert transformer._spatial_shapes_cache[1] is shapes


def test_compiled_spatial_shapes_do_not_touch_module_state(nano, monkeypatch):
    transformer = nano.model.transformer
    transformer.__dict__.pop("_spatial_shapes_cache", None)
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    shapes = transformer._cached_spatial_shapes([(14, 16), (7, 8)], torch.device("cpu"))
    assert torch.equal(shapes, torch.tensor([[14, 16], [7, 8]]))
    assert shapes.dtype == torch.long
    assert "_spatial_shapes_cache" not in transformer.__dict__


def test_pos_embed_interpolation_boundary_only_under_compile(nano, monkeypatch):
    embeddings_module = nano.model.backbone[0].encoder.encoder.embeddings
    tokens = torch.zeros(1, 1 + 20 * 20, embeddings_module.position_embeddings.shape[-1])
    eager = embeddings_module.interpolate_pos_encoding(tokens, 320, 320)

    calls = []
    original = type(embeddings_module)._interpolate_pos_encoding_eager

    def spy(self, *args):
        calls.append(args[1:])
        return original(self, *args)

    monkeypatch.setattr(type(embeddings_module), "_interpolate_pos_encoding_eager", spy)
    embeddings_module.interpolate_pos_encoding(tokens, 320, 320)
    assert calls == []  # eager never takes the boundary

    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    boundary = embeddings_module.interpolate_pos_encoding(tokens, 320, 320)
    assert calls == [(320, 320)]
    assert torch.equal(boundary, eager)
