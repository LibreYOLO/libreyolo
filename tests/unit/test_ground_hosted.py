"""The three hosted LibreGround families resolve to LibreYOLO snapshot repos."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from libreyolo.models.ground import _ALIASES, HOSTED_SNAPSHOTS, LibreGround
from libreyolo.models.ground.florence import LibreGroundFlorence2
from libreyolo.models.ground.qwen3vl import LibreGroundQwen3VL
from libreyolo.models.ground.showui import LibreShowUI
from libreyolo.models.inventory import collect_model_inventory

pytestmark = pytest.mark.unit

_CLASSES = {
    "LibreGroundFlorence2": LibreGroundFlorence2,
    "LibreShowUI": LibreShowUI,
    "LibreGroundQwen3VL": LibreGroundQwen3VL,
}


def test_three_hosted_snapshots_are_declared():
    assert len(HOSTED_SNAPSHOTS) == 3
    for alias, (class_name, size, repo) in HOSTED_SNAPSHOTS.items():
        assert alias in _ALIASES
        family_cls, alias_size = _ALIASES[alias]
        assert family_cls.__name__ == class_name
        assert alias_size == size
        assert repo.startswith("LibreYOLO/")
        assert family_cls.HF_REPOS[size] == repo


def test_public_factory_resolves_hosted_aliases_without_loading():
    for alias, (class_name, size, _repo) in HOSTED_SNAPSHOTS.items():
        family_cls, resolved_size = _ALIASES[alias]
        assert family_cls is _CLASSES[class_name]
        assert resolved_size == size
    with pytest.raises(ValueError, match="Unknown grounding model"):
        LibreGround("not-a-hosted-grounder")


def test_upload_script_mirrors_match_hosted_factory_repos():
    """The upload helper's new mirrors must match the factory HOSTED_SNAPSHOTS."""
    script = (
        Path(__file__).resolve().parents[2] / "weights" / "upload_ground_snapshot_hf.py"
    )
    tree = ast.parse(script.read_text(encoding="utf-8"))
    mirrors = None
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "MIRRORS":
                    mirrors = ast.literal_eval(node.value)
    assert mirrors is not None
    uploaded = {item["repo"] for item in mirrors}
    hosted = {repo for _alias, (_cls, _size, repo) in HOSTED_SNAPSHOTS.items()}
    assert uploaded <= hosted
    assert {
        "LibreYOLO/LibreGroundFlorence2base",
        "LibreYOLO/LibreShowUI2b",
        "LibreYOLO/LibreGroundQwen3VL2b",
    } <= hosted
    assert uploaded == {
        "LibreYOLO/LibreGroundFlorence2base",
        "LibreYOLO/LibreShowUI2b",
        "LibreYOLO/LibreGroundQwen3VL2b",
    }
    showui = next(item for item in mirrors if item["alias"] == "showui-2b")
    assert showui["license"] == "mit"
    assert "Apache-2.0" in showui["license_note"]
    assert showui["license_source"]["url"].endswith(
        "59e059c4df62db0857cc7a7ef0b15d067a30c274/LICENSE"
    )
    assert showui["license_source"]["sha256"] == (
        "dc3ad771428560c99f7605c777c9c37f5b8ef3cd37158f668985910d6dbf3f47"
    )
    apache = showui["additional_license_sources"]
    assert len(apache) == 1
    assert apache[0]["repo"] == "Qwen/Qwen2-VL-2B-Instruct"
    assert apache[0]["revision"] == "895c3a49bc3fa70a340399125c650a463535e71c"
    assert apache[0]["destination"] == "LICENSE-APACHE-2.0"
    assert "_MIT_LICENSE" not in script.read_text(encoding="utf-8")
    for item in mirrors:
        source = item["license_source"]
        assert len(source["sha256"]) == 64
        if source["kind"] == "hf":
            assert len(source["revision"]) == 40


def test_ground_families_do_not_clobber_vlm_inventory():
    inventory = collect_model_inventory()
    assert inventory["florence2"]["class"].endswith("LibreFlorence2")
    assert inventory["qwen3vl"]["class"].endswith("LibreQwen3VL")
    assert inventory["moondream"]["class"].endswith("LibreMoondream")
    assert inventory["locateanything"]["class"].endswith("LibreLocateAnything")
    assert inventory["ground_florence2"]["class"].endswith("LibreGroundFlorence2")
    assert inventory["ground_qwen3vl"]["class"].endswith("LibreGroundQwen3VL")
    assert "ground_moondream" not in inventory
    assert inventory["showui"]["class"].endswith("LibreShowUI")
    assert inventory["ground_florence2"]["tasks"] == ["point"]
    assert (
        inventory["florence2"]["default_task"] != "point"
        or "detect" in inventory["florence2"]["tasks"]
    )
