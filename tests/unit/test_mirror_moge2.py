from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"


def _load_mirror_module():
    sys.path.insert(0, str(SCRIPTS))
    try:
        spec = importlib.util.spec_from_file_location(
            "mirror_moge2_under_test",
            SCRIPTS / "mirror_moge2.py",
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        try:
            spec.loader.exec_module(module)
        finally:
            sys.modules.pop(spec.name, None)
        return module
    finally:
        sys.path.pop(0)


@pytest.fixture
def mirror_moge2(monkeypatch):
    module = _load_mirror_module()
    support = {
        module.GITATTRIBUTES_URL: b"lfs contract\n",
        module.LICENSE_URL: b"test license\n",
    }

    def fake_fetch(url, _expected_size, _expected_sha256, _label):
        if url.endswith("/README.md"):
            return b"---\r\nlicense: mit\r\n---\r\n"
        if url == module.GITATTRIBUTES_URL:
            return support[url]
        if url == module.LICENSE_URL:
            return support[url]
        pytest.fail(f"unexpected network URL: {url}")

    monkeypatch.setattr(module, "_fetch_verified_bytes", fake_fetch)
    return module


def _checkpoint(**overrides):
    checkpoint = {
        "model": {"normal_head.weight": torch.tensor([1.0])},
        "schema_version": "1.0",
        "libreyolo_version": "test",
        "model_family": "moge2",
        "size": "b",
        "task": "normal",
        "nc": 1,
        "names": {0: "normal"},
        "imgsz": 518,
    }
    checkpoint.update(overrides)
    return checkpoint


def _approved_b_spec(mirror_moge2, source: Path):
    spec = dict(mirror_moge2.SIZES["b"])
    spec["sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    return spec


class _FakeResponse:
    def __init__(self, payload: bytes, *, content_length: str | None = None):
        self.payload = payload
        self.headers = {}
        if content_length is not None:
            self.headers["Content-Length"] = content_length
        self.offset = 0
        self.read_sizes: list[int] = []
        self.closed = False

    def getcode(self):
        return 200

    def read(self, size: int):
        self.read_sizes.append(size)
        chunk = self.payload[self.offset : self.offset + size]
        self.offset += len(chunk)
        return chunk

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc, _traceback):
        self.closed = True


def test_b_source_config_and_license_revision_are_pinned():
    mirror_moge2 = _load_mirror_module()
    assert mirror_moge2.SIZES["b"] == {
        "converted": "LibreMoGe2b-normal-LibreMoGe2b-normal.pt",
        "upstream": "Ruicheng/moge-2-vitb-normal",
        "revision": "54ad3a693e61907ea4633d13dec6ee682fa09419",
        "arch": "MoGe-2 ViT-B/14",
        "sha256": None,
        "card_size": 24,
        "card_sha256": (
            "d8d7a46d41a1a37fe4f0a5f637bf55c649310185329127d8a2204632e480be17"
        ),
    }
    assert mirror_moge2.SIZES["s"]["sha256"] == (
        "0b3c1301ddcae5569234010905f093fa8bec5866c7c06197761ea501651f9d9c"
    )
    assert mirror_moge2.SIZES["l"]["sha256"] == (
        "342c13b7028a2e87d164ee9647ad4f34d822dcb73221004c9f25d0458e17580a"
    )
    assert mirror_moge2.GITATTRIBUTES_REVISION == (
        "1c54f3073f8e03f5818d74ca03e3e2fe5cddfbe0"
    )
    assert mirror_moge2.GITATTRIBUTES_URL == (
        "https://huggingface.co/LibreYOLO/LibreSegformerb5-sem/resolve/"
        "1c54f3073f8e03f5818d74ca03e3e2fe5cddfbe0/.gitattributes"
    )
    assert mirror_moge2.GITATTRIBUTES_SHA256 == (
        "88023d0a029a0c409b30c03b689c68605b559f5cefe06376e4a26b38ed795269"
    )
    assert mirror_moge2.LICENSE_SHA256 == (
        "ad7d951c80c5fc2b2bce035f2041bc0a0dbf9028c8ecc4c9a8e1fba8130b6b59"
    )
    assert mirror_moge2.GITATTRIBUTES_SIZE == 1_554
    assert mirror_moge2.LICENSE_SIZE == 12_500
    assert (
        mirror_moge2.LICENSE_URL == "https://raw.githubusercontent.com/microsoft/MoGe/"
        "925b8ed835a7a9cdb7578ba15c658a0afc969030/LICENSE"
    )


def test_upload_whitelist_contains_all_moge2_normal_artifacts():
    skill = (ROOT / "skills/libreyolo-upload-hf-model/SKILL.md").read_text(
        encoding="utf-8"
    )
    whitelist = skill.split("## Canonical filename whitelist", maxsplit=1)[1]
    whitelist = whitelist.split("```", maxsplit=2)[1]
    assert "LibreMoGe2s-normal.pt" in whitelist
    assert "LibreMoGe2b-normal.pt" in whitelist
    assert "LibreMoGe2l-normal.pt" in whitelist


def test_stage_refuses_unapproved_b_before_loading_or_stamping(
    mirror_moge2, monkeypatch, tmp_path
):
    monkeypatch.setattr(
        mirror_moge2.torch,
        "load",
        lambda *_args, **_kwargs: pytest.fail("unapproved artifact was loaded"),
    )
    staging = tmp_path / "staging"

    with pytest.raises(SystemExit, match="not approved.*reproducible.*parity/load"):
        mirror_moge2.stage("b", mirror_moge2.SIZES["b"], staging)

    assert not staging.exists()


def test_stage_rejects_digest_mismatch_before_loading_or_stamping(
    mirror_moge2, monkeypatch, tmp_path
):
    weights = tmp_path / "weights"
    weights.mkdir()
    source = weights / mirror_moge2.SIZES["b"]["converted"]
    source.write_bytes(b"not the approved artifact")
    spec = dict(mirror_moge2.SIZES["b"])
    spec["sha256"] = "0" * 64
    monkeypatch.setattr(mirror_moge2, "WEIGHTS", weights)
    monkeypatch.setattr(
        mirror_moge2.torch,
        "load",
        lambda *_args, **_kwargs: pytest.fail("digest-mismatched artifact was loaded"),
    )
    staging = tmp_path / "staging"

    with pytest.raises(SystemExit, match="SHA-256 mismatch"):
        mirror_moge2.stage("b", spec, staging)

    assert not (staging / "LibreMoGe2b-normal").exists()
    assert not staging.exists() or not any(staging.iterdir())


def test_source_path_swap_cannot_change_descriptor_or_stamp_provenance(
    mirror_moge2, monkeypatch, tmp_path
):
    weights = tmp_path / "weights"
    weights.mkdir()
    source = weights / mirror_moge2.SIZES["b"]["converted"]
    replacement = weights / "replacement.pt"
    backup = weights / "approved-backup.pt"
    torch.save(_checkpoint(model={"approved.weight": torch.tensor([1.0])}), source)
    torch.save(_checkpoint(model={"forged.weight": torch.tensor([2.0])}), replacement)
    spec = _approved_b_spec(mirror_moge2, source)
    monkeypatch.setattr(mirror_moge2, "WEIGHTS", weights)
    monkeypatch.setattr(
        mirror_moge2.torch,
        "load",
        lambda *_args, **_kwargs: pytest.fail("path-swapped artifact was loaded"),
    )

    real_open = mirror_moge2.Path.open
    swapped = False

    def swapping_open(path, mode="r", *args, **kwargs):
        nonlocal swapped
        if path == source and mode == "rb" and not swapped:
            source.rename(backup)
            replacement.rename(source)
            swapped = True
        return real_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(mirror_moge2.Path, "open", swapping_open)
    staging = tmp_path / "staging"

    with pytest.raises(SystemExit, match="changed while opening"):
        mirror_moge2.stage("b", spec, staging)

    assert swapped
    assert not staging.exists()


def test_source_path_swap_after_hash_cannot_change_staged_bytes(
    mirror_moge2, monkeypatch, tmp_path
):
    weights = tmp_path / "weights"
    weights.mkdir()
    source = weights / mirror_moge2.SIZES["b"]["converted"]
    replacement = weights / "replacement.pt"
    backup = weights / "approved-backup.pt"
    torch.save(_checkpoint(model={"approved.weight": torch.tensor([1.0])}), source)
    torch.save(_checkpoint(model={"forged.weight": torch.tensor([2.0])}), replacement)
    spec = _approved_b_spec(mirror_moge2, source)
    monkeypatch.setattr(mirror_moge2, "WEIGHTS", weights)

    real_copy_and_hash = mirror_moge2._copy_and_hash
    swap_blocked = False

    def swap_after_hash(source_handle, destination_handle):
        nonlocal swap_blocked
        digest = real_copy_and_hash(source_handle, destination_handle)
        try:
            source.rename(backup)
        except PermissionError:
            swap_blocked = True
        else:
            replacement.rename(source)
        return digest

    monkeypatch.setattr(mirror_moge2, "_copy_and_hash", swap_after_hash)

    staged = mirror_moge2.stage("b", spec, tmp_path / "staging")
    staged_weight = staged / "LibreMoGe2b-normal.pt"

    if swap_blocked:
        assert staged_weight.read_bytes() == source.read_bytes()
        assert replacement.exists()
    else:
        assert staged_weight.read_bytes() == backup.read_bytes()
        assert staged_weight.read_bytes() != source.read_bytes()
    assert hashlib.sha256(staged_weight.read_bytes()).hexdigest() == spec["sha256"]


def test_load_uses_verified_private_copy_and_publishes_those_exact_bytes(
    mirror_moge2, monkeypatch, tmp_path
):
    weights = tmp_path / "weights"
    weights.mkdir()
    source = weights / mirror_moge2.SIZES["b"]["converted"]
    forged = weights / "forged.pt"
    torch.save(_checkpoint(model={"approved.weight": torch.tensor([1.0])}), source)
    torch.save(_checkpoint(model={"forged.weight": torch.tensor([2.0])}), forged)
    approved_bytes = source.read_bytes()
    forged_bytes = forged.read_bytes()
    spec = _approved_b_spec(mirror_moge2, source)
    monkeypatch.setattr(mirror_moge2, "WEIGHTS", weights)

    private_path = None
    real_copy = mirror_moge2._copy_exclusive_weight

    def record_private_copy(directory, name, source_handle):
        nonlocal private_path
        record = real_copy(directory, name, source_handle)
        private_path = mirror_moge2._entry_path(directory, name)
        return record

    monkeypatch.setattr(
        mirror_moge2,
        "_copy_exclusive_weight",
        record_private_copy,
    )
    real_load = mirror_moge2.torch.load
    loaded_private = False

    def mutate_original_then_load(handle, *args, **kwargs):
        nonlocal loaded_private
        assert kwargs["weights_only"] is True
        assert private_path is not None
        assert mirror_moge2._identity(os.fstat(handle.fileno())) == (
            mirror_moge2._identity(os.lstat(private_path))
        )
        assert hashlib.sha256(handle.read()).hexdigest() == spec["sha256"]
        handle.seek(0)
        source.write_bytes(forged_bytes)
        checkpoint = real_load(handle, *args, **kwargs)
        assert "approved.weight" in checkpoint["model"]
        assert "forged.weight" not in checkpoint["model"]
        loaded_private = True
        return checkpoint

    monkeypatch.setattr(mirror_moge2.torch, "load", mutate_original_then_load)

    staged = mirror_moge2.stage("b", spec, tmp_path / "staging")
    published = staged / "LibreMoGe2b-normal.pt"

    assert loaded_private
    assert source.read_bytes() == forged_bytes
    assert published.read_bytes() == approved_bytes
    assert hashlib.sha256(published.read_bytes()).hexdigest() == spec["sha256"]


def test_private_checkpoint_is_revalidated_after_load(
    mirror_moge2, monkeypatch, tmp_path
):
    weights = tmp_path / "weights"
    weights.mkdir()
    source = weights / mirror_moge2.SIZES["b"]["converted"]
    torch.save(_checkpoint(), source)
    spec = _approved_b_spec(mirror_moge2, source)
    monkeypatch.setattr(mirror_moge2, "WEIGHTS", weights)

    private_path = None
    real_copy = mirror_moge2._copy_exclusive_weight

    def record_private_copy(directory, name, source_handle):
        nonlocal private_path
        record = real_copy(directory, name, source_handle)
        private_path = mirror_moge2._entry_path(directory, name)
        return record

    monkeypatch.setattr(
        mirror_moge2,
        "_copy_exclusive_weight",
        record_private_copy,
    )
    real_load = mirror_moge2.torch.load

    def mutate_private_after_load(handle, *args, **kwargs):
        checkpoint = real_load(handle, *args, **kwargs)
        assert private_path is not None
        private_path.write_bytes(b"changed after deserialization")
        return checkpoint

    monkeypatch.setattr(mirror_moge2.torch, "load", mutate_private_after_load)
    staging = tmp_path / "staging"

    with pytest.raises(SystemExit, match="private checkpoint changed during load"):
        mirror_moge2.stage("b", spec, staging)

    assert not (staging / "LibreMoGe2b-normal").exists()


def test_hardlinked_source_is_rejected(mirror_moge2, monkeypatch, tmp_path):
    weights = tmp_path / "weights"
    weights.mkdir()
    target = weights / "target.pt"
    source = weights / mirror_moge2.SIZES["b"]["converted"]
    torch.save(_checkpoint(), target)
    try:
        source.hardlink_to(target)
    except OSError as exc:
        pytest.skip(f"hard links unavailable: {exc}")
    spec = dict(mirror_moge2.SIZES["b"])
    spec["sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
    monkeypatch.setattr(mirror_moge2, "WEIGHTS", weights)

    with pytest.raises(SystemExit, match="unlinked regular file"):
        mirror_moge2.stage("b", spec, tmp_path / "staging")

    assert not (tmp_path / "staging").exists()


def test_existing_destination_is_preserved(mirror_moge2, monkeypatch, tmp_path):
    weights = tmp_path / "weights"
    weights.mkdir()
    source = weights / mirror_moge2.SIZES["b"]["converted"]
    torch.save(_checkpoint(), source)
    spec = _approved_b_spec(mirror_moge2, source)
    monkeypatch.setattr(mirror_moge2, "WEIGHTS", weights)

    staging = tmp_path / "staging"
    existing = staging / "LibreMoGe2b-normal"
    existing.mkdir(parents=True)
    sentinel = existing / "keep.txt"
    sentinel.write_bytes(b"preserve me")

    with pytest.raises(FileExistsError, match="destination already exists"):
        mirror_moge2.stage("b", spec, staging)

    assert sentinel.read_bytes() == b"preserve me"
    assert {path.name for path in existing.iterdir()} == {"keep.txt"}


def test_destination_created_at_publish_is_preserved(
    mirror_moge2, monkeypatch, tmp_path
):
    weights = tmp_path / "weights"
    weights.mkdir()
    source = weights / mirror_moge2.SIZES["b"]["converted"]
    torch.save(_checkpoint(), source)
    spec = _approved_b_spec(mirror_moge2, source)
    monkeypatch.setattr(mirror_moge2, "WEIGHTS", weights)

    real_rename = mirror_moge2._rename_create_only

    def race_destination(root, temporary, destination_name, expected):
        destination = root.path / destination_name
        destination.mkdir()
        (destination / "keep.txt").write_bytes(b"preserve race winner")
        real_rename(root, temporary, destination_name, expected)

    monkeypatch.setattr(mirror_moge2, "_rename_create_only", race_destination)
    staging = tmp_path / "staging"

    with pytest.raises(FileExistsError, match="destination already exists"):
        mirror_moge2.stage("b", spec, staging)

    destination = staging / "LibreMoGe2b-normal"
    assert (destination / "keep.txt").read_bytes() == b"preserve race winner"
    retained = [
        path
        for path in staging.iterdir()
        if path.name.startswith(".LibreMoGe2b-normal.")
    ]
    assert len(retained) == 1
    assert {path.name for path in retained[0].iterdir()} == {
        ".gitattributes",
        "LICENSE",
        "NOTICE",
        "README.md",
        "LibreMoGe2b-normal.pt",
    }


def test_repeated_publication_uses_exact_windows_leaf_without_trailing_garbage(
    mirror_moge2, monkeypatch, tmp_path
):
    weights = tmp_path / "weights"
    weights.mkdir()
    source = weights / mirror_moge2.SIZES["b"]["converted"]
    torch.save(_checkpoint(), source)
    spec = _approved_b_spec(mirror_moge2, source)
    monkeypatch.setattr(mirror_moge2, "WEIGHTS", weights)
    repo = "LibreMoGe2b-normal"

    for iteration in range(32):
        pair_root = tmp_path / f"pair-{iteration:02d}"
        for leaf in ("first", "second"):
            staging = pair_root / leaf
            staged = mirror_moge2.stage("b", spec, staging)
            names = [path.name for path in staging.iterdir()]
            assert staged == staging / repo
            assert names == [repo]
            assert not any(name.startswith(repo) and name != repo for name in names)


def test_broken_symlink_destination_is_rejected(mirror_moge2, tmp_path):
    staging = tmp_path / "staging"
    staging.mkdir()
    destination = staging / "LibreMoGe2b-normal"
    try:
        destination.symlink_to(tmp_path / "missing", target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlinks unavailable: {exc}")
    spec = dict(mirror_moge2.SIZES["b"])
    spec["sha256"] = "0" * 64

    with pytest.raises(FileExistsError, match="destination already exists"):
        mirror_moge2.stage("b", spec, staging)

    assert destination.is_symlink()


def test_gitattributes_digest_mismatch_leaves_no_output(
    mirror_moge2, monkeypatch, tmp_path
):
    weights = tmp_path / "weights"
    weights.mkdir()
    source = weights / mirror_moge2.SIZES["b"]["converted"]
    torch.save(_checkpoint(), source)
    spec = _approved_b_spec(mirror_moge2, source)
    monkeypatch.setattr(mirror_moge2, "WEIGHTS", weights)

    def altered_fetch(url, _expected_size, _expected_sha256, _label):
        if url == mirror_moge2.GITATTRIBUTES_URL:
            raise SystemExit(".gitattributes SHA-256 mismatch")
        return b"test support bytes"

    monkeypatch.setattr(mirror_moge2, "_fetch_verified_bytes", altered_fetch)
    staging = tmp_path / "staging"

    with pytest.raises(SystemExit, match=r"\.gitattributes SHA-256 mismatch"):
        mirror_moge2.stage("b", spec, staging)

    assert not staging.exists()


def test_source_card_digest_mismatch_leaves_no_output(
    mirror_moge2, monkeypatch, tmp_path
):
    weights = tmp_path / "weights"
    weights.mkdir()
    source = weights / mirror_moge2.SIZES["b"]["converted"]
    torch.save(_checkpoint(), source)
    spec = _approved_b_spec(mirror_moge2, source)
    monkeypatch.setattr(mirror_moge2, "WEIGHTS", weights)

    def altered_fetch(url, _expected_size, _expected_sha256, _label):
        if url.endswith("/README.md"):
            raise SystemExit("source model card SHA-256 mismatch")
        return b"test support bytes"

    monkeypatch.setattr(mirror_moge2, "_fetch_verified_bytes", altered_fetch)
    staging = tmp_path / "staging"

    with pytest.raises(SystemExit, match="source model card SHA-256 mismatch"):
        mirror_moge2.stage("b", spec, staging)

    assert not staging.exists()


def test_bounded_fetch_rejects_declared_oversize_without_reading(
    monkeypatch,
):
    mirror_moge2 = _load_mirror_module()
    response = _FakeResponse(b"x" * 2_000_000, content_length="2000000")
    observed = {}

    def fake_urlopen(request, *, timeout):
        observed["url"] = request.full_url
        observed["timeout"] = timeout
        return response

    monkeypatch.setattr(mirror_moge2.urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(SystemExit, match="server declared 2000000"):
        mirror_moge2._fetch_verified_bytes(
            "https://example.invalid/LICENSE",
            4,
            hashlib.sha256(b"test").hexdigest(),
            "LICENSE",
        )

    assert response.read_sizes == []
    assert response.closed
    assert observed == {
        "url": "https://example.invalid/LICENSE",
        "timeout": mirror_moge2.SUPPORT_FETCH_TIMEOUT_SECONDS,
    }


def test_bounded_fetch_reads_at_most_expected_plus_one(monkeypatch):
    mirror_moge2 = _load_mirror_module()
    response = _FakeResponse(b"x" * 2_000_000)
    monkeypatch.setattr(
        mirror_moge2.urllib.request,
        "urlopen",
        lambda _request, *, timeout: response,
    )

    with pytest.raises(SystemExit, match="expected 4, got 5"):
        mirror_moge2._fetch_verified_bytes(
            "https://example.invalid/README.md",
            4,
            hashlib.sha256(b"test").hexdigest(),
            "source model card",
        )

    assert sum(response.read_sizes) == 5
    assert response.offset == 5
    assert response.closed


def test_bounded_fetch_accepts_only_exact_bytes(monkeypatch):
    mirror_moge2 = _load_mirror_module()
    content = b"test"
    response = _FakeResponse(content, content_length=str(len(content)))
    monkeypatch.setattr(
        mirror_moge2.urllib.request,
        "urlopen",
        lambda _request, *, timeout: response,
    )

    actual = mirror_moge2._fetch_verified_bytes(
        "https://example.invalid/.gitattributes",
        len(content),
        hashlib.sha256(content).hexdigest(),
        ".gitattributes",
    )

    assert actual == content
    assert max(response.read_sizes) <= len(content) + 1
    assert response.closed


def test_temp_hardlink_injection_cannot_overwrite_external_license(
    mirror_moge2, monkeypatch, tmp_path
):
    weights = tmp_path / "weights"
    weights.mkdir()
    source = weights / mirror_moge2.SIZES["b"]["converted"]
    torch.save(_checkpoint(), source)
    spec = _approved_b_spec(mirror_moge2, source)
    monkeypatch.setattr(mirror_moge2, "WEIGHTS", weights)

    victim = tmp_path / "external-license"
    victim.write_bytes(b"external bytes must survive")
    real_write = mirror_moge2._write_exclusive_bytes
    injected = False

    def inject_hardlink(directory, name, content):
        nonlocal injected
        if name == "LICENSE" and not injected:
            try:
                os.link(victim, mirror_moge2._entry_path(directory, name))
            except OSError as exc:
                pytest.skip(f"hard links unavailable: {exc}")
            injected = True
        return real_write(directory, name, content)

    monkeypatch.setattr(mirror_moge2, "_write_exclusive_bytes", inject_hardlink)
    staging = tmp_path / "staging"

    with pytest.raises(FileExistsError):
        mirror_moge2.stage("b", spec, staging)

    assert injected
    assert victim.read_bytes() == b"external bytes must survive"
    assert not (staging / "LibreMoGe2b-normal").exists()


def test_pre_publish_weight_and_license_swap_is_rejected(
    mirror_moge2, monkeypatch, tmp_path
):
    weights = tmp_path / "weights"
    weights.mkdir()
    source = weights / mirror_moge2.SIZES["b"]["converted"]
    torch.save(_checkpoint(), source)
    spec = _approved_b_spec(mirror_moge2, source)
    monkeypatch.setattr(mirror_moge2, "WEIGHTS", weights)

    real_rename = mirror_moge2._rename_create_only

    def swap_before_validation(root, temporary, destination_name, expected):
        (temporary.path / "LibreMoGe2b-normal.pt").write_bytes(b"forged weight")
        (temporary.path / "LICENSE").write_bytes(b"forged license")
        real_rename(root, temporary, destination_name, expected)

    monkeypatch.setattr(
        mirror_moge2,
        "_rename_create_only",
        swap_before_validation,
    )
    staging = tmp_path / "staging"

    with pytest.raises(SystemExit, match="staged file record mismatch"):
        mirror_moge2.stage("b", spec, staging)

    assert not (staging / "LibreMoGe2b-normal").exists()
    assert (
        len(
            [
                path
                for path in staging.iterdir()
                if path.name.startswith(".LibreMoGe2b-normal.")
            ]
        )
        == 1
    )


def test_post_publish_mutation_is_reported_and_left_for_manual_removal(
    mirror_moge2, monkeypatch, tmp_path, capsys
):
    weights = tmp_path / "weights"
    weights.mkdir()
    source = weights / mirror_moge2.SIZES["b"]["converted"]
    torch.save(_checkpoint(), source)
    spec = _approved_b_spec(mirror_moge2, source)
    monkeypatch.setattr(mirror_moge2, "WEIGHTS", weights)

    real_native_rename = mirror_moge2._native_rename_create_only

    def mutate_after_rename(root, temporary, destination_name):
        real_native_rename(root, temporary, destination_name)
        (root.path / destination_name / "LICENSE").write_bytes(
            b"post-publication mutation"
        )

    monkeypatch.setattr(
        mirror_moge2,
        "_native_rename_create_only",
        mutate_after_rename,
    )
    staging = tmp_path / "staging"

    with pytest.raises(
        SystemExit,
        match="published destination failed final validation.*manual removal",
    ):
        mirror_moge2.stage("b", spec, staging)

    destination = staging / "LibreMoGe2b-normal"
    assert (destination / "LICENSE").read_bytes() == b"post-publication mutation"
    assert "no cleanup attempted" in capsys.readouterr().err


def test_replacement_victim_tree_is_never_deleted(mirror_moge2, monkeypatch, tmp_path):
    weights = tmp_path / "weights"
    weights.mkdir()
    source = weights / mirror_moge2.SIZES["b"]["converted"]
    torch.save(_checkpoint(), source)
    spec = _approved_b_spec(mirror_moge2, source)
    monkeypatch.setattr(mirror_moge2, "WEIGHTS", weights)

    victim = tmp_path / "victim"
    victim.mkdir()
    (victim / "preserve.txt").write_bytes(b"do not delete")
    owned_backup = tmp_path / "owned-backup"
    replacement_path = None
    swap_blocked = False
    real_rename = mirror_moge2._rename_create_only

    def replace_owned_temp(root, temporary, destination_name, expected):
        nonlocal replacement_path, swap_blocked
        replacement_path = temporary.path
        try:
            temporary.path.rename(owned_backup)
        except PermissionError:
            swap_blocked = True
            real_rename(root, temporary, destination_name, expected)
            return
        victim.rename(replacement_path)
        real_rename(root, temporary, destination_name, expected)

    monkeypatch.setattr(mirror_moge2, "_rename_create_only", replace_owned_temp)
    staging = tmp_path / "staging"

    if os.name == "nt":
        staged = mirror_moge2.stage("b", spec, staging)
        assert swap_blocked
        assert replacement_path is not None
        assert (victim / "preserve.txt").read_bytes() == b"do not delete"
        assert not owned_backup.exists()
        assert staged == staging / "LibreMoGe2b-normal"
        assert (staged / "LibreMoGe2b-normal.pt").exists()
    else:
        with pytest.raises(SystemExit, match="staging directory path changed"):
            mirror_moge2.stage("b", spec, staging)

        assert not swap_blocked
        assert replacement_path is not None
        assert (replacement_path / "preserve.txt").read_bytes() == b"do not delete"
        assert (owned_backup / "LibreMoGe2b-normal.pt").exists()
        assert not (staging / "LibreMoGe2b-normal").exists()


@pytest.mark.skipif(os.name != "nt", reason="Windows directory sharing regression")
def test_windows_private_directory_cannot_be_swapped_before_child_write(
    mirror_moge2, monkeypatch, tmp_path
):
    weights = tmp_path / "weights"
    weights.mkdir()
    source = weights / mirror_moge2.SIZES["b"]["converted"]
    torch.save(_checkpoint(), source)
    spec = _approved_b_spec(mirror_moge2, source)
    monkeypatch.setattr(mirror_moge2, "WEIGHTS", weights)

    victim = tmp_path / "victim"
    victim.mkdir()
    sentinel = victim / "preserve.txt"
    sentinel.write_bytes(b"do not write here")
    owned_backup = tmp_path / "owned-backup"
    swap_attempted = False
    real_copy = mirror_moge2._copy_exclusive_weight

    def swap_before_first_child_write(directory, name, source_handle):
        nonlocal swap_attempted
        swap_attempted = True
        with pytest.raises(PermissionError):
            directory.path.rename(owned_backup)
        return real_copy(directory, name, source_handle)

    monkeypatch.setattr(
        mirror_moge2,
        "_copy_exclusive_weight",
        swap_before_first_child_write,
    )

    staging = tmp_path / "staging"
    staged = mirror_moge2.stage("b", spec, staging)

    assert swap_attempted
    assert sentinel.read_bytes() == b"do not write here"
    assert {path.name for path in victim.iterdir()} == {"preserve.txt"}
    assert not owned_backup.exists()
    assert staged == staging / "LibreMoGe2b-normal"
    assert (staged / "LibreMoGe2b-normal.pt").exists()


def test_staging_root_swap_cannot_redirect_successful_publication(
    mirror_moge2, monkeypatch, tmp_path
):
    weights = tmp_path / "weights"
    weights.mkdir()
    source = weights / mirror_moge2.SIZES["b"]["converted"]
    torch.save(_checkpoint(), source)
    spec = _approved_b_spec(mirror_moge2, source)
    monkeypatch.setattr(mirror_moge2, "WEIGHTS", weights)

    staging = tmp_path / "staging"
    original_root = tmp_path / "sealed-root-backup"
    real_native_rename = mirror_moge2._native_rename_create_only
    state = {"blocked": False, "swapped": False}

    def swap_root_before_rename(root, temporary, destination_name):
        try:
            root.path.rename(original_root)
        except OSError:
            state["blocked"] = True
        else:
            state["swapped"] = True
            root.path.mkdir()
        real_native_rename(root, temporary, destination_name)

    monkeypatch.setattr(
        mirror_moge2,
        "_native_rename_create_only",
        swap_root_before_rename,
    )

    if os.name == "nt":
        staged = mirror_moge2.stage("b", spec, staging)
        assert state == {"blocked": True, "swapped": False}
        assert staged == staging / "LibreMoGe2b-normal"
        assert (staged / "LibreMoGe2b-normal.pt").exists()
    else:
        with pytest.raises(SystemExit, match="staging directory path changed"):
            mirror_moge2.stage("b", spec, staging)
        assert state == {"blocked": False, "swapped": True}
        assert not (staging / "LibreMoGe2b-normal").exists()
        assert (original_root / "LibreMoGe2b-normal").exists()


def test_cli_b_refusal_does_not_create_explicit_staging(
    mirror_moge2, monkeypatch, tmp_path
):
    staging = tmp_path / "must-not-exist"
    monkeypatch.setattr(
        mirror_moge2,
        "stage",
        lambda *_args, **_kwargs: pytest.fail("stage was reached before preflight"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["mirror_moge2.py", "b", "--staging", str(staging)],
    )

    with pytest.raises(SystemExit, match="not approved.*reproducible.*parity/load"):
        mirror_moge2.main()

    assert not staging.exists()


def test_cli_default_preflight_refuses_all_before_partial_staging(
    mirror_moge2, monkeypatch, tmp_path
):
    staging = tmp_path / "must-not-exist"
    monkeypatch.setattr(
        mirror_moge2,
        "stage",
        lambda *_args, **_kwargs: pytest.fail("stage was reached before preflight"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["mirror_moge2.py", "--staging", str(staging)],
    )

    with pytest.raises(SystemExit, match="LibreMoGe2b-normal.*not approved"):
        mirror_moge2.main()

    assert not staging.exists()


def test_parse_args_default_keeps_existing_scripts_create_behavior(
    mirror_moge2, monkeypatch, tmp_path
):
    staging = tmp_path / "compat-staging"
    monkeypatch.setattr(
        sys,
        "argv",
        ["mirror_other.py", "--staging", str(staging)],
    )

    args = mirror_moge2.parse_args("other mirror", ["s"])

    assert args.staging == staging
    assert staging.is_dir()


def test_stage_is_deterministic_and_uses_exact_metadata_claims(
    mirror_moge2, monkeypatch, tmp_path
):
    weights = tmp_path / "weights"
    weights.mkdir()
    source = weights / mirror_moge2.SIZES["b"]["converted"]
    torch.save(_checkpoint(), source)

    monkeypatch.setattr(mirror_moge2, "WEIGHTS", weights)
    fetched_urls = []

    def fake_fetch(url, _expected_size, _expected_sha256, _label):
        fetched_urls.append(url)
        if url == mirror_moge2.GITATTRIBUTES_URL:
            return b"lfs contract\n"
        if url == mirror_moge2.LICENSE_URL:
            return b"audited composite license\n"
        if url.endswith("/README.md"):
            return b"---\r\nlicense: mit\r\n---\r\n"
        pytest.fail(f"unexpected network URL: {url}")

    monkeypatch.setattr(mirror_moge2, "_fetch_verified_bytes", fake_fetch)
    approved_spec = _approved_b_spec(mirror_moge2, source)

    first = mirror_moge2.stage("b", approved_spec, tmp_path / "first")
    second = mirror_moge2.stage("b", approved_spec, tmp_path / "second")

    expected_files = {
        ".gitattributes",
        "LICENSE",
        "NOTICE",
        "README.md",
        "LibreMoGe2b-normal.pt",
    }
    assert {path.name for path in first.iterdir()} == expected_files
    assert {path.name for path in second.iterdir()} == expected_files
    for filename in expected_files:
        assert (first / filename).read_bytes() == (second / filename).read_bytes()

    assert (first / "LibreMoGe2b-normal.pt").read_bytes() == source.read_bytes()
    assert (first / ".gitattributes").read_bytes() == b"lfs contract\n"
    assert (first / "LICENSE").read_bytes() == b"audited composite license\n"
    assert fetched_urls == [
        mirror_moge2.GITATTRIBUTES_URL,
        mirror_moge2.LICENSE_URL,
        (
            "https://huggingface.co/Ruicheng/moge-2-vitb-normal/resolve/"
            "54ad3a693e61907ea4633d13dec6ee682fa09419/README.md"
        ),
        mirror_moge2.GITATTRIBUTES_URL,
        mirror_moge2.LICENSE_URL,
        (
            "https://huggingface.co/Ruicheng/moge-2-vitb-normal/resolve/"
            "54ad3a693e61907ea4633d13dec6ee682fa09419/README.md"
        ),
    ]

    readme = (first / "README.md").read_text(encoding="utf-8")
    notice = (first / "NOTICE").read_text(encoding="utf-8")
    exact_claims = (
        "Unused point, mask, and metric-scale head tensors are removed.",
        "Encoder, neck,\nand normal-head tensors are retained unchanged",
        "LibreYOLO checkpoint v1.0\nmetadata is added",
    )
    for claim in exact_claims:
        assert claim in readme
    assert "unused point, mask, and metric-scale head tensors are\n" in notice
    assert "encoder, neck, and normal-head tensors are retained\n" in notice
    assert "LibreYOLO checkpoint v1.0 metadata is added by\n" in notice
    assert "State-dict key remapping only" not in readme
    assert "state-dict key remapping only" not in notice

    source_url = (
        "https://huggingface.co/Ruicheng/moge-2-vitb-normal/blob/"
        "54ad3a693e61907ea4633d13dec6ee682fa09419/model.pt"
    )
    source_card_url = (
        "https://huggingface.co/Ruicheng/moge-2-vitb-normal/blob/"
        "54ad3a693e61907ea4633d13dec6ee682fa09419/README.md"
    )
    for document in (readme, notice):
        assert source_url in document
        assert source_card_url in document
        assert "declares these weights MIT" in " ".join(document.split())
        assert mirror_moge2.MOGE_SOURCE_URL in document
        assert mirror_moge2.MOGE_DINOV2_URL in document
        assert mirror_moge2.MOGE_LICENSE_PAGE_URL in document
        assert "https://github.com/facebookresearch/dinov2" not in document


@pytest.mark.parametrize(
    ("field", "overrides"),
    [
        ("model_family", {"model_family": "other"}),
        ("size", {"size": "s"}),
        ("task", {"task": "depth"}),
        ("nc", {"nc": 2, "names": {0: "normal", 1: "normal"}}),
        ("names", {"names": {0: "not-normal"}}),
        ("imgsz", {"imgsz": 512}),
    ],
)
def test_stage_rejects_unexpected_b_metadata(
    mirror_moge2, monkeypatch, tmp_path, field, overrides
):
    weights = tmp_path / "weights"
    weights.mkdir()
    source = weights / mirror_moge2.SIZES["b"]["converted"]
    torch.save(_checkpoint(**overrides), source)
    monkeypatch.setattr(mirror_moge2, "WEIGHTS", weights)
    approved_spec = _approved_b_spec(mirror_moge2, source)

    with pytest.raises(SystemExit, match=rf"metadata mismatch: .*{field} expected"):
        mirror_moge2.stage("b", approved_spec, tmp_path / "staging")


def test_stage_uses_strict_schema_validation(mirror_moge2, monkeypatch, tmp_path):
    weights = tmp_path / "weights"
    weights.mkdir()
    source = weights / mirror_moge2.SIZES["b"]["converted"]
    checkpoint = _checkpoint()
    del checkpoint["schema_version"]
    torch.save(checkpoint, source)
    monkeypatch.setattr(mirror_moge2, "WEIGHTS", weights)
    approved_spec = _approved_b_spec(mirror_moge2, source)

    with pytest.raises(SystemExit, match="checkpoint metadata invalid"):
        mirror_moge2.stage("b", approved_spec, tmp_path / "staging")
