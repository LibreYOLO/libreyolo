"""Hermetic integrity tests for SAM 3D Body and MHR asset transport."""

from __future__ import annotations

import hashlib
import io
import logging
import os
import subprocess
import sys
import types
import zipfile
from pathlib import Path

import pytest
import torch

from libreyolo.models.sam3dbody import mhr_body
from libreyolo.models.sam3dbody._assets import (
    AssetIntegrityError,
    FileSeal,
    PinnedFile,
    atomic_rename_create_only,
    cleanup_private_file,
    cleanup_private_tree,
    ensure_unlinked_directory,
    inspect_pinned_file,
    make_private_stage,
    require_unlinked_directory,
)
from libreyolo.models.sam3dbody import snapshot


pytestmark = pytest.mark.unit


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


class _Response(io.BytesIO):
    def __init__(self, payload: bytes, declared: str | None = None):
        super().__init__(payload)
        self.read_calls = 0
        self.headers = {}
        if declared is not None:
            self.headers["Content-Length"] = declared

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()

    def read(self, *args, **kwargs):
        self.read_calls += 1
        return super().read(*args, **kwargs)


@pytest.fixture
def tiny_mhr(monkeypatch):
    model = b"reviewed-mhr-model-bytes"
    license_bytes = b"license"
    archive_stream = io.BytesIO()
    with zipfile.ZipFile(
        archive_stream, "w", compression=zipfile.ZIP_DEFLATED
    ) as bundle:
        bundle.writestr(mhr_body.MHR_ARCHIVE_MEMBER, model)
        bundle.writestr(mhr_body.MHR_LICENSE_MEMBER, license_bytes)
    archive = archive_stream.getvalue()
    with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
        entry = bundle.getinfo(mhr_body.MHR_ARCHIVE_MEMBER)

    monkeypatch.setattr(
        mhr_body,
        "MHR_ARCHIVE",
        PinnedFile("assets.zip", len(archive), _sha(archive)),
    )
    monkeypatch.setattr(
        mhr_body,
        "MHR_MODEL_FILE",
        PinnedFile("mhr_model.pt", len(model), _sha(model)),
    )
    monkeypatch.setattr(mhr_body, "MHR_MEMBER_COMPRESSED_SIZE", entry.compress_size)
    monkeypatch.setattr(mhr_body, "MHR_MEMBER_CRC32", entry.CRC)
    monkeypatch.setattr(
        mhr_body,
        "MHR_LICENSE_FILE",
        PinnedFile("LICENSE", len(license_bytes), _sha(license_bytes)),
    )
    return model, archive


def test_mhr_download_extracts_only_the_exact_reviewed_member(
    tmp_path, monkeypatch, tiny_mhr
):
    model, archive = tiny_mhr
    monkeypatch.setattr(
        mhr_body.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: _Response(archive, str(len(archive))),
    )
    target = tmp_path / "cache" / "mhr_model.pt"

    assert mhr_body.ensure_mhr_model(target) == target.absolute()
    assert target.read_bytes() == model
    inspect_pinned_file(target, mhr_body.MHR_MODEL_FILE, label="test MHR")


def test_mhr_invalid_concurrent_winner_reports_owned_stage(
    tmp_path, monkeypatch, caplog, tiny_mhr
):
    _model, archive = tiny_mhr
    monkeypatch.setattr(
        mhr_body.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: _Response(archive, str(len(archive))),
    )
    target = tmp_path / "cache" / "mhr_model.pt"

    def lose_publication(_source, destination, **_kwargs):
        destination.write_bytes(b"invalid concurrent winner")
        raise FileExistsError("concurrent winner")

    monkeypatch.setattr(mhr_body, "atomic_rename_create_only", lose_publication)
    with (
        caplog.at_level(logging.WARNING),
        pytest.raises(AssetIntegrityError, match="pinned|reviewed bytes"),
    ):
        mhr_body.ensure_mhr_model(target)

    leaked = list(target.parent.glob(f".{target.name}.staging-*.tmp"))
    assert len(leaked) == 1
    assert str(leaked[0]) in caplog.text
    assert str(target) in caplog.text
    assert "invalid winner" in caplog.text


def test_mhr_valid_concurrent_winner_cleans_losing_stage(
    tmp_path, monkeypatch, tiny_mhr
):
    model, archive = tiny_mhr
    monkeypatch.setattr(
        mhr_body.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: _Response(archive, str(len(archive))),
    )
    target = tmp_path / "cache" / "mhr_model.pt"

    def lose_publication(_source, destination, **_kwargs):
        destination.write_bytes(model)
        raise FileExistsError("concurrent winner")

    monkeypatch.setattr(mhr_body, "atomic_rename_create_only", lose_publication)
    assert mhr_body.ensure_mhr_model(target) == target.absolute()
    assert not list(target.parent.glob(f".{target.name}.staging-*.tmp"))


def test_mhr_refuses_stale_recovery_before_network(tmp_path, monkeypatch, tiny_mhr):
    target = tmp_path / "cache" / "mhr_model.pt"
    target.parent.mkdir()
    recovery = target.parent / f".{target.name}.staging-recovery.tmp"
    recovery.write_bytes(b"partial")

    def network(*_args, **_kwargs):
        raise AssertionError("network must not run while recovery data exists")

    monkeypatch.setattr(mhr_body.urllib.request, "urlopen", network)
    with pytest.raises(AssetIntegrityError, match="recovery data.*cannot accumulate"):
        mhr_body.ensure_mhr_model(target)


def test_mhr_partial_stage_is_cleaned(tmp_path, tiny_mhr):
    model, _archive = tiny_mhr
    target = tmp_path / "mhr_model.pt"

    class FailingSource(io.BytesIO):
        def __init__(self, payload):
            super().__init__(payload)
            self._reads = 0

        def read(self, size=-1):
            self._reads += 1
            if self._reads > 1:
                raise OSError("forced staging failure")
            return super().read(max(1, min(size, len(model) // 2)))

    with pytest.raises(OSError, match="forced staging failure"):
        mhr_body._stage_model(FailingSource(model), target)
    assert not list(tmp_path.glob(f".{target.name}.staging-*.tmp"))


def test_mhr_post_publication_failure_reports_both_possible_paths(
    tmp_path, monkeypatch, caplog, tiny_mhr
):
    _model, archive = tiny_mhr
    monkeypatch.setattr(
        mhr_body.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: _Response(archive, str(len(archive))),
    )
    target = tmp_path / "cache" / "mhr_model.pt"
    real_inspect = mhr_body.inspect_mhr_model

    def fail_published(path):
        path = Path(path).absolute()
        if path == target.absolute() and os.path.lexists(target):
            raise AssetIntegrityError("forced MHR post-publication failure")
        return real_inspect(path)

    monkeypatch.setattr(mhr_body, "inspect_mhr_model", fail_published)
    with (
        caplog.at_level(logging.WARNING),
        pytest.raises(AssetIntegrityError, match="post-publication"),
    ):
        mhr_body.ensure_mhr_model(target)

    assert target.is_file()
    assert "staging path" in caplog.text
    assert str(target) in caplog.text
    assert "neither path was deleted" in caplog.text


def test_mhr_existing_tampered_file_rejects_before_network(
    tmp_path, monkeypatch, tiny_mhr
):
    target = tmp_path / "mhr_model.pt"
    target.write_bytes(b"tampered")
    called = False

    def network(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("network must not run")

    monkeypatch.setattr(mhr_body.urllib.request, "urlopen", network)
    with pytest.raises(AssetIntegrityError, match="reviewed bytes"):
        mhr_body.ensure_mhr_model(target)
    assert called is False


def test_mhr_rejects_linked_cache_ancestor_before_network(
    tmp_path, monkeypatch, tiny_mhr
):
    real = tmp_path / "real"
    real.mkdir()
    linked = tmp_path / "linked"
    try:
        linked.symlink_to(real, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlink unavailable: {exc}")

    def network(*_args, **_kwargs):
        raise AssertionError("network must not run for a linked cache path")

    monkeypatch.setattr(mhr_body.urllib.request, "urlopen", network)
    with pytest.raises(AssetIntegrityError, match="unlinked directories"):
        mhr_body.ensure_mhr_model(linked / "mhr_model.pt")
    assert not (real / "mhr_model.pt").exists()


def test_mhr_loader_rejects_existing_file_under_linked_ancestor(
    tmp_path, monkeypatch, tiny_mhr
):
    model, _archive = tiny_mhr
    real = tmp_path / "real"
    real.mkdir()
    (real / "mhr_model.pt").write_bytes(model)
    linked = tmp_path / "linked"
    try:
        linked.symlink_to(real, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlink unavailable: {exc}")

    called = False

    def unsafe_load(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("linked parent reached torch.jit.load")

    monkeypatch.setattr(torch.jit, "load", unsafe_load)
    with pytest.raises(AssetIntegrityError, match="unlinked directories"):
        mhr_body.MHRBodyModel.from_file(linked / "mhr_model.pt")
    assert called is False


@pytest.mark.skipif(os.name != "nt", reason="Windows junction regression")
def test_mhr_loader_rejects_existing_file_under_windows_junction(
    tmp_path, monkeypatch, tiny_mhr
):
    model, _archive = tiny_mhr
    real = tmp_path / "real"
    real.mkdir()
    (real / "mhr_model.pt").write_bytes(model)
    junction = tmp_path / "junction"
    result = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(junction), str(real)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        pytest.skip(f"junction creation unavailable: {result.stderr or result.stdout}")

    called = False

    def unsafe_load(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("junction parent reached torch.jit.load")

    monkeypatch.setattr(torch.jit, "load", unsafe_load)
    try:
        with pytest.raises(AssetIntegrityError, match="unlinked directories"):
            mhr_body.MHRBodyModel.from_file(junction / "mhr_model.pt")
        assert called is False
    finally:
        if os.path.lexists(junction):
            junction.rmdir()


def test_mhr_loader_never_deserializes_unreviewed_bytes(
    tmp_path, monkeypatch, tiny_mhr
):
    target = tmp_path / "mhr_model.pt"
    target.write_bytes(b"not-the-reviewed-model")
    called = False

    def unsafe_load(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("unverified TorchScript reached torch.jit.load")

    monkeypatch.setattr(torch.jit, "load", unsafe_load)
    with pytest.raises(AssetIntegrityError, match="reviewed bytes"):
        mhr_body.MHRBodyModel.from_file(target)
    assert called is False


def test_mhr_archive_content_length_mismatch_rejects_before_body_read(
    monkeypatch, tiny_mhr
):
    _model, archive = tiny_mhr
    response = _Response(archive, str(len(archive) + 1))
    monkeypatch.setattr(
        mhr_body.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: response,
    )
    with pytest.raises(RuntimeError, match="Content-Length"):
        mhr_body._download_archive(io.BytesIO())
    assert response.read_calls == 0


def test_mhr_headerless_oversized_archive_is_bounded(monkeypatch, tiny_mhr):
    _model, archive = tiny_mhr
    monkeypatch.setattr(
        mhr_body.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: _Response(archive + b"x"),
    )
    with pytest.raises(RuntimeError, match="exceeded"):
        mhr_body._download_archive(io.BytesIO())


def test_mhr_duplicate_exact_member_is_rejected(monkeypatch, tiny_mhr):
    model, _archive = tiny_mhr
    stream = io.BytesIO()
    with pytest.warns(UserWarning, match="Duplicate name"):
        with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
            bundle.writestr(mhr_body.MHR_LICENSE_MEMBER, b"license")
            bundle.writestr(mhr_body.MHR_ARCHIVE_MEMBER, model)
            bundle.writestr(mhr_body.MHR_ARCHIVE_MEMBER, model)
    stream.seek(0)
    with pytest.raises(RuntimeError, match="exactly one"):
        mhr_body._extract_model(stream, io.BytesIO())


def test_mhr_wrong_member_metadata_is_rejected(monkeypatch, tiny_mhr):
    model, _archive = tiny_mhr
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_STORED) as bundle:
        bundle.writestr(mhr_body.MHR_LICENSE_MEMBER, b"license")
        bundle.writestr(mhr_body.MHR_ARCHIVE_MEMBER, model)
    stream.seek(0)
    with pytest.raises(RuntimeError, match="metadata"):
        mhr_body._extract_model(stream, io.BytesIO())


def test_mhr_wrong_embedded_license_is_rejected(monkeypatch, tiny_mhr):
    model, _archive = tiny_mhr
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr(mhr_body.MHR_LICENSE_MEMBER, b"forged!")
        bundle.writestr(mhr_body.MHR_ARCHIVE_MEMBER, model)
    stream.seek(0)
    with pytest.raises(RuntimeError, match="archive license"):
        mhr_body._extract_model(stream, io.BytesIO())


def test_mhr_hardlink_is_rejected_before_torchscript_load(
    tmp_path, monkeypatch, tiny_mhr
):
    model, _archive = tiny_mhr
    source = tmp_path / "source"
    source.write_bytes(model)
    target = tmp_path / "mhr_model.pt"
    os.link(source, target)
    called = False

    def unsafe_load(*_args, **_kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(torch.jit, "load", unsafe_load)
    with pytest.raises(AssetIntegrityError, match="unlinked regular file"):
        mhr_body.MHRBodyModel.from_file(target)
    assert called is False


def test_descriptor_redirection_is_rejected(tmp_path, monkeypatch):
    from libreyolo.models.sam3dbody import _assets

    expected_bytes = b"expected"
    alternate_bytes = b"attacker"
    source = tmp_path / "source"
    alternate = tmp_path / "alternate"
    source.write_bytes(expected_bytes)
    alternate.write_bytes(alternate_bytes)
    expected = PinnedFile("source", len(expected_bytes), _sha(expected_bytes))
    real_open = _assets.os.open

    def redirected_open(path, flags, *args, **kwargs):
        if os.fspath(path) == os.fspath(source):
            return real_open(alternate, flags, *args, **kwargs)
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(_assets.os, "open", redirected_open)
    with pytest.raises(AssetIntegrityError, match="changed while it was opened"):
        inspect_pinned_file(source, expected, label="redirected source")


def test_private_tree_cleanup_removes_only_owned_tree(tmp_path):
    stage, seal = make_private_stage(tmp_path, prefix=".cleanup-")
    nested = stage / ".cache" / "metadata"
    nested.mkdir(parents=True)
    (stage / "model.ckpt").write_bytes(b"model")
    (nested / "record").write_bytes(b"metadata")

    assert cleanup_private_tree(
        stage,
        expected_object=seal,
        label="test private tree",
    )
    assert not os.path.lexists(stage)


def test_private_tree_cleanup_rejects_replaced_root(tmp_path):
    stage, seal = make_private_stage(tmp_path, prefix=".cleanup-")
    displaced = tmp_path / "displaced"
    stage.rename(displaced)
    stage.mkdir()
    valuable = stage / "valuable"
    valuable.write_bytes(b"user data")

    with pytest.raises(AssetIntegrityError, match="replaced"):
        cleanup_private_tree(
            stage,
            expected_object=seal,
            label="test private tree",
        )
    assert valuable.read_bytes() == b"user data"


def test_private_file_cleanup_rejects_replaced_path(tmp_path):
    stage = tmp_path / ".staging-file"
    stage.write_bytes(b"owned")
    before = os.lstat(stage)
    seal = FileSeal(
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_size,
        before.st_mtime_ns,
        before.st_nlink,
    )
    displaced = tmp_path / "displaced"
    stage.rename(displaced)
    stage.write_bytes(b"user data")

    with pytest.raises(AssetIntegrityError, match="replaced"):
        cleanup_private_file(
            stage,
            expected_object=seal,
            label="test private file",
        )
    assert stage.read_bytes() == b"user data"


def test_create_only_publication_preserves_existing_destination(tmp_path):
    ensure_unlinked_directory(tmp_path, label="test parent")
    source = tmp_path / "staged"
    source.write_bytes(b"new")
    # Build the seal from a stable helper without depending on private stat layout.
    info = os.lstat(source)
    source_seal = FileSeal(
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_size,
        info.st_mtime_ns,
        info.st_nlink,
    )
    destination = tmp_path / "final"
    destination.write_bytes(b"valuable")
    parent = require_unlinked_directory(tmp_path, label="test parent")
    with pytest.raises(FileExistsError):
        atomic_rename_create_only(
            source,
            destination,
            expected_source=source_seal,
            expected_parent=parent,
        )
    assert destination.read_bytes() == b"valuable"
    assert source.read_bytes() == b"new"


@pytest.mark.skipif(os.name != "nt", reason="exercises the Windows rename branch")
def test_publication_detects_source_swap_inside_rename(tmp_path, monkeypatch):
    from libreyolo.models.sam3dbody import _assets

    source = tmp_path / "staged"
    source.write_bytes(b"reviewed")
    before = os.lstat(source)
    seal = FileSeal(
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_size,
        before.st_mtime_ns,
        before.st_nlink,
    )
    parent = require_unlinked_directory(tmp_path, label="test parent")
    destination = tmp_path / "final"
    displaced = tmp_path / "displaced"
    real_rename = _assets.os.rename

    def racing_rename(_source, target):
        real_rename(source, displaced)
        source.write_bytes(b"attacker")
        real_rename(source, target)

    monkeypatch.setattr(_assets.os, "rename", racing_rename)
    with pytest.raises(AssetIntegrityError, match="identity changed"):
        atomic_rename_create_only(
            source,
            destination,
            expected_source=seal,
            expected_parent=parent,
        )
    assert destination.read_bytes() == b"attacker"
    assert displaced.read_bytes() == b"reviewed"


@pytest.fixture
def tiny_sam_pin(monkeypatch):
    payloads = {
        ".gitattributes": b"*.ckpt filter=lfs\n",
        "LICENSE": b"sam license\n",
        "README.md": b"reviewed card\n",
        "model.ckpt": b"reviewed pickle bytes",
        "model_config.yaml": b"MODEL:\n  IMAGE_SIZE: 512\n",
    }
    files = tuple(
        PinnedFile(name, len(payloads[name]), _sha(payloads[name]))
        for name in snapshot._REMOTE_NAMES
    )
    pin = snapshot.SAMSnapshotPin(
        size="d3",
        repo_id="LibreYOLO/TestSAM3DBodyd3-mesh",
        revision="a" * 40,
        files=files,
        legacy_revisions=("b" * 40, "c" * 40),
    )
    monkeypatch.setattr(snapshot, "SAM_SNAPSHOT_PINS", {"d3": pin})
    return pin, payloads


def _write_explicit_snapshot(root: Path, payloads: dict[str, bytes], *, full=False):
    root.mkdir()
    names = snapshot._REMOTE_NAMES if full else snapshot._RUNTIME_NAMES
    for name in names:
        (root / name).write_bytes(payloads[name])


def test_explicit_runtime_snapshot_is_strictly_validated(tmp_path, tiny_sam_pin):
    pin, payloads = tiny_sam_pin
    root = tmp_path / "snapshot"
    _write_explicit_snapshot(root, payloads)
    identity = snapshot.inspect_sam_snapshot(root, "d3", managed=False)
    assert identity.repo_id == pin.repo_id
    assert identity.revision == pin.revision
    assert {file.path for file in identity.files} == set(snapshot._RUNTIME_NAMES)


def test_exact_full_hub_tree_is_accepted_for_explicit_path(tmp_path, tiny_sam_pin):
    _pin, payloads = tiny_sam_pin
    root = tmp_path / "snapshot"
    _write_explicit_snapshot(root, payloads, full=True)
    identity = snapshot.inspect_sam_snapshot(root, "d3", managed=False)
    assert {file.path for file in identity.files} == set(snapshot._REMOTE_NAMES)


def test_snapshot_extra_file_is_rejected(tmp_path, tiny_sam_pin):
    _pin, payloads = tiny_sam_pin
    root = tmp_path / "snapshot"
    _write_explicit_snapshot(root, payloads)
    (root / "unexpected.py").write_text("raise RuntimeError", encoding="utf-8")
    with pytest.raises(AssetIntegrityError, match="unexpected inventory"):
        snapshot.inspect_sam_snapshot(root, "d3", managed=False)


def test_snapshot_tampered_pickle_is_rejected(tmp_path, tiny_sam_pin):
    _pin, payloads = tiny_sam_pin
    root = tmp_path / "snapshot"
    _write_explicit_snapshot(root, payloads)
    (root / "model.ckpt").write_bytes(b"forged pickle bytes")
    with pytest.raises(AssetIntegrityError, match="reviewed bytes"):
        snapshot.inspect_sam_snapshot(root, "d3", managed=False)


def test_snapshot_tampered_marker_is_rejected(tmp_path, tiny_sam_pin):
    pin, payloads = tiny_sam_pin
    root = tmp_path / "snapshot"
    root.mkdir()
    for expected in pin.runtime_files:
        (root / expected.path).write_bytes(payloads[expected.path])
    (root / snapshot.SNAPSHOT_MARKER).write_bytes(
        snapshot._marker_bytes(pin).replace(pin.revision.encode(), b"b" * 40)
    )
    with pytest.raises(AssetIntegrityError, match="reviewed bytes"):
        snapshot.inspect_sam_snapshot(root, "d3", managed=True)


def test_snapshot_hardlinked_file_is_rejected(tmp_path, tiny_sam_pin):
    _pin, payloads = tiny_sam_pin
    root = tmp_path / "snapshot"
    _write_explicit_snapshot(root, payloads)
    external = tmp_path / "external"
    external.write_bytes(payloads["LICENSE"])
    (root / "LICENSE").unlink()
    os.link(external, root / "LICENSE")
    with pytest.raises(AssetIntegrityError, match="unlinked regular file"):
        snapshot.inspect_sam_snapshot(root, "d3", managed=False)


def test_snapshot_symlinked_file_is_rejected(tmp_path, tiny_sam_pin):
    _pin, payloads = tiny_sam_pin
    root = tmp_path / "snapshot"
    _write_explicit_snapshot(root, payloads)
    external = tmp_path / "external"
    external.write_bytes(payloads["LICENSE"])
    (root / "LICENSE").unlink()
    try:
        (root / "LICENSE").symlink_to(external)
    except OSError as exc:
        pytest.skip(f"symlink creation is unavailable: {exc}")
    with pytest.raises(AssetIntegrityError, match="unlinked regular file"):
        snapshot.inspect_sam_snapshot(root, "d3", managed=False)


class _Sibling:
    def __init__(self, file: PinnedFile):
        self.rfilename = file.path
        self.size = file.size
        self.lfs = {"sha256": file.sha256} if file.path == "model.ckpt" else None


def _fake_hub(
    pin,
    sources,
    calls,
    *,
    remote_sha=None,
    extra=False,
    gated="auto",
    dry_run_sizes=None,
    dry_run_commits=None,
):
    siblings = [_Sibling(file) for file in pin.files]
    if extra:
        siblings.append(
            types.SimpleNamespace(rfilename="unexpected.py", size=1, lfs=None)
        )

    class HfApi:
        def model_info(self, **kwargs):
            calls.append(("model_info", kwargs))
            return types.SimpleNamespace(
                sha=pin.revision if remote_sha is None else remote_sha,
                siblings=siblings,
                gated=gated,
            )

    def download(**kwargs):
        if kwargs.get("dry_run"):
            calls.append(("dry_run", kwargs))
            expected = pin.by_name[kwargs["filename"]]
            sizes = dry_run_sizes or {}
            commits = dry_run_commits or {}
            return types.SimpleNamespace(
                file_size=sizes.get(expected.path, expected.size),
                commit_hash=commits.get(
                    expected.path,
                    pin.revision,
                ),
            )
        calls.append(("download", kwargs))
        if kwargs.get("local_dir") is not None:
            destination = Path(kwargs["local_dir"]) / kwargs["filename"]
            destination.write_bytes((sources / kwargs["filename"]).read_bytes())
            metadata = Path(kwargs["local_dir"]) / ".cache" / "huggingface"
            metadata.mkdir(parents=True, exist_ok=True)
            (metadata / f"{kwargs['filename']}.metadata").write_text(
                "test metadata",
                encoding="utf-8",
            )
            return str(destination)
        return str(sources / kwargs["filename"])

    return types.SimpleNamespace(HfApi=HfApi, hf_hub_download=download)


def test_acquire_uses_exact_revision_and_publishes_allowlisted_view(
    tmp_path, monkeypatch, tiny_sam_pin
):
    pin, payloads = tiny_sam_pin
    sources = tmp_path / "hub-cache"
    sources.mkdir()
    for name, payload in payloads.items():
        (sources / name).write_bytes(payload)
    calls = []
    monkeypatch.setitem(sys.modules, "huggingface_hub", _fake_hub(pin, sources, calls))

    identity = snapshot.acquire_sam_snapshot("d3", cache_root=tmp_path / "managed")
    assert identity.root.name == pin.revision
    assert set(path.name for path in identity.root.iterdir()) == {
        *snapshot._RUNTIME_NAMES,
        snapshot.SNAPSHOT_MARKER,
    }
    downloads = [kwargs for kind, kwargs in calls if kind == "download"]
    dry_runs = [kwargs for kind, kwargs in calls if kind == "dry_run"]
    assert {call["filename"] for call in dry_runs} == set(snapshot._REMOTE_NAMES)
    assert {call["filename"] for call in downloads} == set(snapshot._RUNTIME_NAMES)
    assert all(call["revision"] == pin.revision for call in dry_runs)
    assert all(call["local_files_only"] is False for call in dry_runs)
    dry_run_indices = [
        i for i, (kind, _kwargs) in enumerate(calls) if kind == "dry_run"
    ]
    download_indices = [
        i for i, (kind, _kwargs) in enumerate(calls) if kind == "download"
    ]
    assert max(dry_run_indices) < min(download_indices)
    assert all(call["revision"] == pin.revision for call in downloads)
    assert all(call["local_files_only"] is False for call in downloads)
    assert all(
        Path(call["local_dir"]).parent == identity.root.parent for call in downloads
    )
    assert not (identity.root / ".cache").exists()


def test_invalid_concurrent_snapshot_winner_reports_owned_stage(
    tmp_path, monkeypatch, caplog, tiny_sam_pin
):
    pin, payloads = tiny_sam_pin
    sources = tmp_path / "hub-cache"
    sources.mkdir()
    for name, payload in payloads.items():
        (sources / name).write_bytes(payload)
    calls = []
    monkeypatch.setitem(sys.modules, "huggingface_hub", _fake_hub(pin, sources, calls))

    def lose_publication(_source, destination, **_kwargs):
        destination.mkdir()
        (destination / "model.ckpt").write_bytes(b"invalid concurrent winner")
        raise FileExistsError("concurrent winner")

    monkeypatch.setattr(snapshot, "atomic_rename_create_only", lose_publication)
    cache = tmp_path / "managed"
    with (
        caplog.at_level(logging.WARNING),
        pytest.raises(AssetIntegrityError, match="unexpected inventory"),
    ):
        snapshot.acquire_sam_snapshot("d3", cache_root=cache)

    leaked = list(cache.glob(f".{pin.revision}.staging-*"))
    assert len(leaked) == 1
    assert str(leaked[0]) in caplog.text
    assert str(cache / pin.revision) in caplog.text
    assert "invalid winner" in caplog.text


def test_valid_concurrent_snapshot_winner_cleans_losing_stage(
    tmp_path, monkeypatch, tiny_sam_pin
):
    pin, payloads = tiny_sam_pin
    sources = tmp_path / "hub-cache"
    sources.mkdir()
    for name, payload in payloads.items():
        (sources / name).write_bytes(payload)
    calls = []
    monkeypatch.setitem(sys.modules, "huggingface_hub", _fake_hub(pin, sources, calls))

    def lose_publication(source, destination, **_kwargs):
        destination.mkdir()
        for entry in source.iterdir():
            (destination / entry.name).write_bytes(entry.read_bytes())
        raise FileExistsError("concurrent winner")

    monkeypatch.setattr(snapshot, "atomic_rename_create_only", lose_publication)
    cache = tmp_path / "managed"
    identity = snapshot.acquire_sam_snapshot("d3", cache_root=cache)

    assert identity.root == cache / pin.revision
    assert not list(cache.glob(f".{pin.revision}.staging-*"))


def test_snapshot_refuses_stale_recovery_before_hub_import(
    tmp_path, monkeypatch, tiny_sam_pin
):
    pin, _payloads = tiny_sam_pin
    cache = tmp_path / "managed"
    cache.mkdir()
    recovery = cache / f".{pin.revision}.staging-recovery"
    recovery.mkdir()
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)

    with pytest.raises(AssetIntegrityError, match="recovery data.*cannot accumulate"):
        snapshot.acquire_sam_snapshot("d3", cache_root=cache)


def test_snapshot_failed_download_cleans_private_stage(
    tmp_path, monkeypatch, tiny_sam_pin
):
    pin, payloads = tiny_sam_pin
    sources = tmp_path / "hub-cache"
    sources.mkdir()
    for name, payload in payloads.items():
        (sources / name).write_bytes(payload)
    calls = []
    hub = _fake_hub(pin, sources, calls)
    real_download = hub.hf_hub_download

    def fail_model(**kwargs):
        if not kwargs.get("dry_run") and kwargs["filename"] == "model.ckpt":
            raise OSError("forced download failure")
        return real_download(**kwargs)

    hub.hf_hub_download = fail_model
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    cache = tmp_path / "managed"

    with pytest.raises(RuntimeError, match="forced download failure"):
        snapshot.acquire_sam_snapshot("d3", cache_root=cache)
    assert not list(cache.glob(f".{pin.revision}.staging-*"))


def test_snapshot_missing_hub_dependency_has_install_command(
    tmp_path, monkeypatch, tiny_sam_pin
):
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    with pytest.raises(ImportError, match=r'pip install "libreyolo\[hf\]"'):
        snapshot.acquire_sam_snapshot("d3", cache_root=tmp_path / "managed")


def test_post_publication_validation_failure_reports_both_possible_paths(
    tmp_path, monkeypatch, caplog, tiny_sam_pin
):
    pin, payloads = tiny_sam_pin
    sources = tmp_path / "hub-cache"
    sources.mkdir()
    for name, payload in payloads.items():
        (sources / name).write_bytes(payload)
    calls = []
    monkeypatch.setitem(sys.modules, "huggingface_hub", _fake_hub(pin, sources, calls))
    cache = tmp_path / "managed"
    destination = cache / pin.revision
    real_inspect = snapshot.inspect_sam_snapshot

    def fail_published(root, size, *, managed=None):
        root_path = Path(root).absolute()
        if root_path == destination.absolute() and os.path.lexists(destination):
            raise AssetIntegrityError("forced post-publication validation failure")
        return real_inspect(root, size, managed=managed)

    monkeypatch.setattr(snapshot, "inspect_sam_snapshot", fail_published)
    with (
        caplog.at_level(logging.WARNING),
        pytest.raises(AssetIntegrityError, match="post-publication"),
    ):
        snapshot.acquire_sam_snapshot("d3", cache_root=cache)

    assert destination.is_dir()
    assert "staging path" in caplog.text
    assert str(destination) in caplog.text
    assert "verified recovery" in caplog.text


def test_transport_urls_and_revisions_are_immutable():
    assert "/latest/" not in mhr_body.MHR_ASSETS_URL
    assert "/main/" not in mhr_body.MHR_ASSETS_URL
    assert mhr_body.MHR_RELEASE in mhr_body.MHR_ASSETS_URL
    for pin in snapshot.SAM_SNAPSHOT_PINS.values():
        assert len(pin.revision) == 40
        assert all(char in "0123456789abcdef" for char in pin.revision)
        assert pin.gated == "auto"

    assert snapshot.SAM_SNAPSHOT_PINS["d3"].revision == (
        "46e286e25347518d861ab0f21e1b2b5b630dc21f"
    )
    assert snapshot.SAM_SNAPSHOT_PINS["h"].revision == (
        "a745fa6fcd5d71e16c4da921a28a6bb6f1ff9e3e"
    )
    d3_card = snapshot.SAM_SNAPSHOT_PINS["d3"].by_name["README.md"]
    h_card = snapshot.SAM_SNAPSHOT_PINS["h"].by_name["README.md"]
    assert (d3_card.size, d3_card.sha256) == (
        3_982,
        "d1e195edb377518f095717bedf9663cad0286ff37e274fb0940da946e9928d3d",
    )
    assert (h_card.size, h_card.sha256) == (
        3_975,
        "71a09372eacd30fd850647884c9f7f91565e161ee2b2d6cb7c5be1613cc6cd3c",
    )
    assert snapshot.SAM_SNAPSHOT_PINS["d3"].legacy_revisions == (
        "8e822540228d9de9bef1bf26414e27954044c242",
        "4531d41c4b8349d272a9e7efb42b38a1a5f1d737",
    )
    assert snapshot.SAM_SNAPSHOT_PINS["h"].legacy_revisions == (
        "70a2c8ae1f43d6cff94105d83a8dd63d6eeba5ad",
        "b3c59d31106cc69a8ab4cd6510bc289bccf258e9",
    )


def test_acquire_rejects_remote_tree_drift_before_payload_fetch(
    tmp_path, monkeypatch, tiny_sam_pin
):
    pin, _payloads = tiny_sam_pin
    sources = tmp_path / "hub-cache"
    sources.mkdir()
    calls = []
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        _fake_hub(pin, sources, calls, extra=True),
    )
    with pytest.raises(AssetIntegrityError, match="exact tree"):
        snapshot.acquire_sam_snapshot("d3", cache_root=tmp_path / "managed")
    assert not any(kind == "download" for kind, _kwargs in calls)


def test_acquire_rejects_remote_commit_drift_before_payload_fetch(
    tmp_path, monkeypatch, tiny_sam_pin
):
    pin, _payloads = tiny_sam_pin
    sources = tmp_path / "hub-cache"
    sources.mkdir()
    calls = []
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        _fake_hub(pin, sources, calls, remote_sha="b" * 40),
    )
    with pytest.raises(AssetIntegrityError, match="unexpected commit"):
        snapshot.acquire_sam_snapshot("d3", cache_root=tmp_path / "managed")
    assert not any(kind == "download" for kind, _kwargs in calls)


@pytest.mark.parametrize("gated", [False, True, "manual", None])
def test_acquire_rejects_remote_gate_drift_before_payload_fetch(
    tmp_path, monkeypatch, tiny_sam_pin, gated
):
    pin, _payloads = tiny_sam_pin
    sources = tmp_path / "hub-cache"
    sources.mkdir()
    calls = []
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        _fake_hub(pin, sources, calls, gated=gated),
    )
    with pytest.raises(AssetIntegrityError, match="reviewed.*gate"):
        snapshot.acquire_sam_snapshot("d3", cache_root=tmp_path / "managed")
    assert not any(kind in {"dry_run", "download"} for kind, _kwargs in calls)


def test_acquire_rejects_download_leg_size_drift_before_any_payload_fetch(
    tmp_path, monkeypatch, tiny_sam_pin
):
    pin, _payloads = tiny_sam_pin
    sources = tmp_path / "hub-cache"
    sources.mkdir()
    calls = []
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        _fake_hub(
            pin,
            sources,
            calls,
            dry_run_sizes={"README.md": pin.by_name["README.md"].size + 1},
        ),
    )
    with pytest.raises(AssetIntegrityError, match="download metadata size mismatch"):
        snapshot.acquire_sam_snapshot("d3", cache_root=tmp_path / "managed")
    assert any(
        kind == "dry_run" and kwargs["filename"] == "README.md"
        for kind, kwargs in calls
    )
    assert not any(kind == "download" for kind, _kwargs in calls)


def test_acquire_rejects_download_leg_commit_drift_before_any_payload_fetch(
    tmp_path, monkeypatch, tiny_sam_pin
):
    pin, _payloads = tiny_sam_pin
    sources = tmp_path / "hub-cache"
    sources.mkdir()
    calls = []
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        _fake_hub(
            pin,
            sources,
            calls,
            dry_run_commits={"README.md": "d" * 40},
        ),
    )
    with pytest.raises(AssetIntegrityError, match="download metadata commit mismatch"):
        snapshot.acquire_sam_snapshot("d3", cache_root=tmp_path / "managed")
    assert any(
        kind == "dry_run" and kwargs["filename"] == "README.md"
        for kind, kwargs in calls
    )
    assert not any(kind == "download" for kind, _kwargs in calls)


def test_acquire_offline_skips_api_and_requests_only_cached_commit(
    tmp_path, monkeypatch, tiny_sam_pin
):
    pin, payloads = tiny_sam_pin
    sources = tmp_path / "hub-cache"
    sources.mkdir()
    for name, payload in payloads.items():
        (sources / name).write_bytes(payload)
    calls = []
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setitem(sys.modules, "huggingface_hub", _fake_hub(pin, sources, calls))
    snapshot.acquire_sam_snapshot("d3", cache_root=tmp_path / "managed")
    assert not any(kind == "model_info" for kind, _kwargs in calls)
    downloads = [kwargs for kind, kwargs in calls if kind == "download"]
    dry_runs = [kwargs for kind, kwargs in calls if kind == "dry_run"]
    assert {call["filename"] for call in dry_runs} == set(snapshot._RUNTIME_NAMES)
    assert all(call["local_files_only"] is True for call in dry_runs)
    assert downloads and all(call["local_files_only"] is True for call in downloads)


def test_invalid_existing_managed_cache_rejects_before_hub_import(
    tmp_path, monkeypatch, tiny_sam_pin
):
    pin, _payloads = tiny_sam_pin
    root = tmp_path / "managed" / pin.revision
    root.mkdir(parents=True)
    (root / "model.ckpt").write_bytes(b"bad")
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    with pytest.raises(AssetIntegrityError, match="unexpected inventory"):
        snapshot.acquire_sam_snapshot("d3", cache_root=tmp_path / "managed")


def test_exact_legacy_cache_is_adopted_without_network(
    tmp_path, monkeypatch, tiny_sam_pin
):
    pin, payloads = tiny_sam_pin
    legacy = tmp_path / "managed"
    legacy.mkdir()
    for expected in pin.runtime_files:
        (legacy / expected.path).write_bytes(payloads[expected.path])
    (legacy / ".cache").mkdir()
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    identity = snapshot.acquire_sam_snapshot("d3", cache_root=legacy)
    assert identity.root == legacy / pin.revision
    assert (identity.root / snapshot.SNAPSHOT_MARKER).is_file()


@pytest.mark.parametrize("legacy_index", [0, 1])
def test_exact_card_only_revision_cache_is_adopted_without_network(
    tmp_path, monkeypatch, tiny_sam_pin, legacy_index
):
    pin, payloads = tiny_sam_pin
    cache = tmp_path / "managed"
    cache.mkdir()
    previous_revision = pin.legacy_revisions[legacy_index]
    previous = cache / previous_revision
    _write_explicit_snapshot(previous, payloads)
    (previous / snapshot.SNAPSHOT_MARKER).write_bytes(
        snapshot._marker_bytes(pin, revision=previous_revision)
    )
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)

    identity = snapshot.acquire_sam_snapshot("d3", cache_root=cache)

    assert identity.root == cache / pin.revision
    assert identity.revision == pin.revision
    assert previous.is_dir()
    assert (identity.root / snapshot.SNAPSHOT_MARKER).is_file()


def test_partial_legacy_cache_fails_before_network(tmp_path, monkeypatch, tiny_sam_pin):
    _pin, payloads = tiny_sam_pin
    legacy = tmp_path / "managed"
    legacy.mkdir()
    (legacy / "model.ckpt").write_bytes(payloads["model.ckpt"])
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    with pytest.raises(AssetIntegrityError, match="legacy.*incomplete"):
        snapshot.acquire_sam_snapshot("d3", cache_root=legacy)


def test_model_explicit_path_records_strict_identity(tmp_path, tiny_sam_pin):
    _pin, payloads = tiny_sam_pin
    root = tmp_path / "snapshot"
    _write_explicit_snapshot(root, payloads)
    from libreyolo.models.sam3dbody.model import LibreSAM3DBody

    model = LibreSAM3DBody.__new__(LibreSAM3DBody)
    checkpoint = model._resolve_checkpoint(root, "d3")
    assert checkpoint == root.absolute() / "model.ckpt"
    assert model._checkpoint_snapshot_identity.root == root.absolute()


def _bare_sam_model(root: Path, identity, *, mhr_path: Path | None = None):
    from libreyolo.models.sam3dbody.model import LibreSAM3DBody

    model = LibreSAM3DBody.__new__(LibreSAM3DBody)
    model._checkpoint_snapshot_identity = identity
    model._ckpt_path = root / "model.ckpt"
    model._mhr_path = mhr_path or root.parent / "mhr_model.pt"
    model.size = "d3"
    model.device = torch.device("cpu")
    return model


def test_model_rejects_persistent_checkpoint_mutation_before_upstream_load(
    tmp_path, monkeypatch, tiny_sam_pin, tiny_mhr
):
    _pin, payloads = tiny_sam_pin
    mhr_bytes, _archive = tiny_mhr
    root = tmp_path / "snapshot"
    _write_explicit_snapshot(root, payloads)
    (tmp_path / "mhr_model.pt").write_bytes(mhr_bytes)
    identity = snapshot.inspect_sam_snapshot(root, "d3", managed=False)
    model = _bare_sam_model(root, identity)
    (root / "model.ckpt").write_bytes(b"forged")
    called = False

    def load(*_args, **_kwargs):
        nonlocal called
        called = True
        return object(), object()

    upstream = types.ModuleType("sam_3d_body")
    upstream.load_sam_3d_body = load
    monkeypatch.setitem(sys.modules, "sam_3d_body", upstream)
    monkeypatch.setattr(model, "_import_upstream", lambda: upstream)
    with pytest.raises(AssetIntegrityError, match="reviewed bytes"):
        model._init_model()
    assert called is False


def test_model_rejects_checkpoint_mutation_during_upstream_load(
    tmp_path, monkeypatch, tiny_sam_pin, tiny_mhr
):
    _pin, payloads = tiny_sam_pin
    mhr_bytes, _archive = tiny_mhr
    root = tmp_path / "snapshot"
    _write_explicit_snapshot(root, payloads)
    (tmp_path / "mhr_model.pt").write_bytes(mhr_bytes)
    identity = snapshot.inspect_sam_snapshot(root, "d3", managed=False)
    model = _bare_sam_model(root, identity)

    def load(*_args, **_kwargs):
        (root / "model_config.yaml").write_bytes(b"forged")
        return object(), object()

    upstream = types.ModuleType("sam_3d_body")
    upstream.load_sam_3d_body = load
    monkeypatch.setitem(sys.modules, "sam_3d_body", upstream)
    monkeypatch.setattr(model, "_import_upstream", lambda: upstream)
    with pytest.raises(AssetIntegrityError, match="reviewed bytes"):
        model._init_model()


def test_upstream_path_only_trust_boundary_is_explicit():
    from libreyolo.models.sam3dbody import model as model_module

    boundary = model_module.UPSTREAM_PATH_TRUST_BOUNDARY
    assert "trusted, quiescent" in boundary
    assert "same-user local filesystem" in boundary
    assert "path-only" in boundary


@pytest.mark.parametrize("explicit", [True, False])
def test_model_validates_explicit_and_default_mhr_before_upstream_load(
    tmp_path, monkeypatch, tiny_sam_pin, tiny_mhr, explicit
):
    from libreyolo.models.sam3dbody import model as model_module

    _pin, payloads = tiny_sam_pin
    _mhr_bytes, _archive = tiny_mhr
    root = tmp_path / "snapshot"
    _write_explicit_snapshot(root, payloads)
    mhr_path = tmp_path / "mhr_model.pt"
    mhr_path.write_bytes(b"unreviewed MHR")
    identity = snapshot.inspect_sam_snapshot(root, "d3", managed=False)
    model = _bare_sam_model(root, identity, mhr_path=mhr_path)
    if not explicit:
        model._mhr_path = None
        monkeypatch.setattr(model_module, "ensure_mhr_model", lambda: mhr_path)
    called = False

    def load(*_args, **_kwargs):
        nonlocal called
        called = True
        return object(), object()

    upstream = types.ModuleType("sam_3d_body")
    upstream.load_sam_3d_body = load
    monkeypatch.setitem(sys.modules, "sam_3d_body", upstream)
    monkeypatch.setattr(model, "_import_upstream", lambda: upstream)
    with pytest.raises(AssetIntegrityError, match="reviewed bytes"):
        model._init_model()
    assert called is False


@pytest.mark.parametrize("explicit", [True, False])
def test_model_rehashes_explicit_and_default_mhr_after_upstream_load(
    tmp_path, monkeypatch, tiny_sam_pin, tiny_mhr, explicit
):
    from libreyolo.models.sam3dbody import model as model_module

    _pin, payloads = tiny_sam_pin
    mhr_bytes, _archive = tiny_mhr
    root = tmp_path / "snapshot"
    _write_explicit_snapshot(root, payloads)
    mhr_path = tmp_path / "mhr_model.pt"
    mhr_path.write_bytes(mhr_bytes)
    identity = snapshot.inspect_sam_snapshot(root, "d3", managed=False)
    model = _bare_sam_model(root, identity, mhr_path=mhr_path)
    if not explicit:
        model._mhr_path = None
        monkeypatch.setattr(model_module, "ensure_mhr_model", lambda: mhr_path)

    def load(*_args, **_kwargs):
        mhr_path.write_bytes(b"persistent MHR mutation")
        return object(), object()

    upstream = types.ModuleType("sam_3d_body")
    upstream.load_sam_3d_body = load
    monkeypatch.setitem(sys.modules, "sam_3d_body", upstream)
    monkeypatch.setattr(model, "_import_upstream", lambda: upstream)
    with pytest.raises(AssetIntegrityError, match="reviewed bytes"):
        model._init_model()
