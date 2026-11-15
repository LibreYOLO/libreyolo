"""LibreYOLO org repos that LibreYOLO() does not load fail fast (issue #1027).

Everything here is offline: the Hub client is replaced by fakes that fail the
test if they are called, and the documented calls are checked against the
dispatcher alias tables and constructor signatures instead of being run.
"""

from __future__ import annotations

import ast
import importlib
import inspect

import pytest

from libreyolo.models import LibreYOLO
from libreyolo.utils import hf_hub
from libreyolo.utils.hf_hub import (
    HUB_REPO_USAGES,
    HubRef,
    _select_repo_weight_file,
    hub_repo_usage,
)

pytestmark = pytest.mark.unit


# One repo per family, with a fragment the error must contain.
FAMILY_SAMPLES = [
    ("LibreYOLO/LibreMoondream3", 'LibreVLM("moondream-3")'),
    ("LibreYOLO/LibreMolmo28b", 'LibreVLM("molmo2-8b")'),
    ("LibreYOLO/LibreGemma4e2b", 'LibreVLM("gemma-4-e2b")'),
    ("LibreYOLO/SenseNovaVision7b", 'LibreVLM("sensenova-vision")'),
    ("LibreYOLO/LibreQwen3VLEmbedding2b", 'LibreVLM("qwen3-vl-embedding-2b")'),
    ("LibreYOLO/LibreFalconOCR270m", 'LibreVLM("falcon-ocr")'),
    ("LibreYOLO/LibreDeepSeekOCR2", 'LibreVLM("deepseek-ocr-2")'),
    ("LibreYOLO/LibrePaddleOCRVL0.9b", 'LibreVLM("paddleocr-vl")'),
    ("LibreYOLO/LibreGLMOCR0.9b", 'LibreVLM("glm-ocr")'),
    ("LibreYOLO/LibreShowUI2b", 'LibreGround("showui-2b")'),
    ("LibreYOLO/LibreGroundingDINOt", 'LibreOpenVocab("grounding-dino-tiny")'),
    ("LibreYOLO/LibreOVDEIMm", 'LibreOpenVocab("ov-deim-m")'),
    ("LibreYOLO/LibreSAMhuge", 'LibreSAM("huge")'),
    ("LibreYOLO/LibreSAM2tiny", 'LibreSAM("sam2-tiny")'),
    ("LibreYOLO/LibreSAM31large", 'LibreSAM("sam3.1")'),
    ("LibreYOLO/LibreEdgeTAM", 'LibreSAM("edgetam")'),
    ("LibreYOLO/LibreMobileSAM", 'LibreSAM("mobilesam")'),
    ("LibreYOLO/LibrePicoSAM3", 'LibreSAM("picosam3")'),
    ("LibreYOLO/LibrePRX512-dc-ae-sft", 'LibreGen("prx-512-dc-ae-sft")'),
    ("LibreYOLO/LibrePRXPixel7b", 'LibreGen("prxpixel-7b")'),
    ("LibreYOLO/LibreTRELLISl", 'LibreGen("trellis")'),
    ("LibreYOLO/LibreTRELLIS2-4b", 'LibreGen("trellis2-4b")'),
    ("LibreYOLO/LibreTripoSplat0.6b", 'LibreGen("triposplat")'),
    ("LibreYOLO/LibreLGM0.4b", 'LibreGen("lgm")'),
    ("LibreYOLO/LibreImageDream", 'LibreGen("lgm")'),
    ("LibreYOLO/LibreTripoSR", 'LibreGen("triposr")'),
    ("LibreYOLO/LibreCRM", 'LibreGen("crm")'),
    ("LibreYOLO/LibreDirect3DS2v11", 'LibreGen("direct3d-s2")'),
    ("LibreYOLO/LibreDINOv3vitl16", "not a standalone model"),
    ("LibreYOLO/LibreDetAny3D", "LibreDetAny3D(runtime_path="),
    ("LibreYOLO/LibreFCOS3D", "LibreFCOS3D()"),
    ("LibreYOLO/LibreWildDet3D", "LibreWildDet3D()"),
    ("LibreYOLO/Libre3DMOODb", 'Libre3DMOOD(size="b"'),
    ("LibreYOLO/LibreSAM3DBodyd3-mesh", 'LibreSAM3DBody(size="d3")'),
    ("LibreYOLO/LibreReID-osnet", 'tracker="deepocsort", embedder="osnet_ain_x0_5"'),
    ("LibreYOLO/librefacerec-det", 'LibreYOLO("librefacerec-l")'),
    ("LibreYOLO/libreyolo-web", "browser demo"),
    ("LibreYOLO/face-rfdetr-nano", "not a LibreYOLO checkpoint"),
    ("LibreYOLO/LibreDAMOYOLOs", "DAMO-YOLO was removed"),
    ("LibreYOLO/LibrePAGEsp-gazetarget", "does not include the PAGE family"),
]


@pytest.fixture
def no_hub_calls(monkeypatch):
    """Any Hub API call or download fails the test."""
    hub = pytest.importorskip("huggingface_hub")

    class ForbiddenApi:
        def __init__(self, *args, **kwargs):
            raise AssertionError("HfApi must not be used for a mapped repo")

    def forbidden_download(*args, **kwargs):
        raise AssertionError("hf_hub_download must not run for a mapped repo")

    monkeypatch.setattr(hub, "HfApi", ForbiddenApi)
    monkeypatch.setattr(hub, "hf_hub_download", forbidden_download)
    return hub


@pytest.mark.parametrize("repo_id, fragment", FAMILY_SAMPLES)
def test_mapped_repo_fails_fast_with_the_working_call(repo_id, fragment, no_hub_calls):
    with pytest.raises(ValueError) as excinfo:
        hf_hub.resolve_hub_checkpoint(HubRef(repo_id))
    message = str(excinfo.value)
    assert repo_id in message
    assert "LibreYOLO() does not load" in message
    assert fragment in message


@pytest.mark.parametrize(
    "model_path",
    [
        "LibreYOLO/LibreMoondream3",
        "hf://LibreYOLO/LibreMoondream3",
        "hf://LibreYOLO/LibreSAM2tiny@main/sam2.1_hiera_tiny.pt",
        "libreyolo/libremoondream3",  # repo ids resolve case-insensitively
    ],
)
def test_factory_fails_fast_before_any_download(
    model_path, no_hub_calls, tmp_path, monkeypatch
):
    # A local "libreyolo/" directory (the source tree, on a case-insensitive
    # filesystem) would shadow the bare owner/repo form.
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="does not load"):
        LibreYOLO(model_path)


def test_other_owners_are_not_matched():
    assert hub_repo_usage("someuser/LibreMoondream3") is None
    assert hub_repo_usage("LibreYOLO/LibreYOLO9t") is None
    assert hub_repo_usage("LibreYOLO/LibreFlowSeekt-flow") is None


def test_every_sample_is_mapped():
    names = {repo.split("/")[1].lower() for repo, _ in FAMILY_SAMPLES}
    assert names <= set(HUB_REPO_USAGES)


# ---------------------------------------------------------------------------
# The documented calls are real
# ---------------------------------------------------------------------------


def _usages():
    return sorted(HUB_REPO_USAGES.items())


@pytest.mark.parametrize("repo, usage", _usages(), ids=[r for r, _ in _usages()])
def test_every_imported_name_exists(repo, usage):
    for module, name in usage.imports:
        assert hasattr(importlib.import_module(module), name), (
            f"{repo}: {module}.{name} does not exist"
        )


def _dispatch(dispatcher: str, alias: str):
    """Resolve a dispatcher alias to (family class, size) without loading."""
    key = alias.strip().lower()
    if dispatcher == "LibreVLM":
        from libreyolo.models import vlm

        if key in vlm._ALIASES:
            return vlm._ALIASES[key]
        lazy = [
            (vlm._LAZY_ALIASES, "libreyolo.models.sensenova", "LibreSenseNovaVision"),
            (vlm._DEEPSEEKOCR_ALIASES, "libreyolo.models.deepseekocr", "LibreDeepSeekOCR"),
            (vlm._PADDLEOCRVL_ALIASES, "libreyolo.models.paddleocrvl", "LibrePaddleOCRVL"),
            (vlm._GLMOCR_ALIASES, "libreyolo.models.glmocr", "LibreGLMOCR"),
            (vlm._FALCONOCR_ALIASES, "libreyolo.models.falconocr", "LibreFalconOCR"),
        ]
        for table, module, name in lazy:
            if key in table:
                return getattr(importlib.import_module(module), name), table[key]
        raise AssertionError(f"LibreVLM does not know {alias!r}")
    if dispatcher == "LibreSAM":
        from libreyolo.models.sam import model as sam

        family, size = sam._ALIASES[key]
        if family == sam._MOBILE_SAM:
            from libreyolo.models.mobilesam import LibreMobileSAM as family
        elif family == sam._PICO_SAM3:
            from libreyolo.models.picosam3 import LibrePicoSAM3 as family
        return family, size
    module = {
        "LibreGen": "libreyolo.models.gen",
        "LibreGround": "libreyolo.models.ground",
        "LibreOpenVocab": "libreyolo.models.openvocab",
    }[dispatcher]
    return importlib.import_module(module)._ALIASES[key]


_DISPATCHERS = {"LibreVLM", "LibreSAM", "LibreGen", "LibreGround", "LibreOpenVocab"}


def _model_calls(usage):
    """Yield (callee, args, kwargs) for each ``model = Callee(...)`` line."""
    for line in usage.code:
        node = ast.parse(line).body[0]
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            call = node.value
            if isinstance(call.func, ast.Name):
                args = [ast.literal_eval(a) for a in call.args]
                kwargs = {k.arg: ast.literal_eval(k.value) for k in call.keywords}
                yield call.func.id, args, kwargs


@pytest.mark.parametrize("repo, usage", _usages(), ids=[r for r, _ in _usages()])
def test_documented_call_reaches_this_repo(repo, usage):
    """Dispatcher aliases resolve to the family whose HF_REPOS names this
    repo, and direct constructor calls bind to the real signature."""
    imported = {name: module for module, name in usage.imports}
    for callee, args, kwargs in _model_calls(usage):
        if callee in _DISPATCHERS:
            family, size = _dispatch(callee, args[0])
            hf_repos = getattr(family, "HF_REPOS", None)
            assert hf_repos is not None, f"{family.__name__} has no HF_REPOS"
            if repo == "libreimagedream":
                assert family.__name__ == "LibreLGM"
                continue
            assert hf_repos[size].lower() == f"libreyolo/{repo}", (
                f"{callee}({args[0]!r}) loads {hf_repos[size]}, not {repo}"
            )
        elif callee == "LibreYOLO":
            continue
        else:
            cls = getattr(importlib.import_module(imported[callee]), callee)
            inspect.signature(cls).bind(*args, **kwargs)


def test_sized_3d_classes_point_at_their_repos():
    from libreyolo.models.mood3d.model import WEIGHTS
    from libreyolo.models.sam3dbody import snapshot

    for size in ("t", "b"):
        assert WEIGHTS[size]["filename"][:-3] == f"Libre3DMOOD{size}"
        assert f"libre3dmood{size}" in HUB_REPO_USAGES
    for size, pin in snapshot.SAM_SNAPSHOT_PINS.items():
        assert hub_repo_usage(pin.repo_id).code == (
            f'model = LibreSAM3DBody(size="{size}")',
        )


def test_reid_embedder_variants_exist():
    from libreyolo.tracking.config import DeepOCSortConfig
    from libreyolo.tracking.reid import _OSNET_CHANNELS

    message = hub_repo_usage("LibreYOLO/LibreReID-osnet").message(
        "LibreYOLO/LibreReID-osnet"
    )
    for variant in _OSNET_CHANNELS:
        assert variant.replace("osnet_ain_", "") in message
    config = DeepOCSortConfig.from_kwargs(embedder="osnet_ain_x0_5")
    assert config.embedder in _OSNET_CHANNELS


# ---------------------------------------------------------------------------
# Unmapped repos with several checkpoints
# ---------------------------------------------------------------------------


def test_multiple_checkpoints_message_shows_a_working_example():
    with pytest.raises(ValueError) as excinfo:
        _select_repo_weight_file(["b.pt", "a.pt", "README.md"], "someuser/two")
    message = str(excinfo.value)
    assert "LibreYOLO('hf://someuser/two/<filename>')" in message
    assert "model = LibreYOLO('hf://someuser/two/a.pt')" in message
