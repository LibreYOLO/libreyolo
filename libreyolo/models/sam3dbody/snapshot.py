"""Pinned SAM 3D Body checkpoint transport.

The mirrored checkpoints are SAM-licensed pickle files.  They are downloaded
only from reviewed immutable Hub commits, copied into an exact allow-listed
view, and fully hashed immediately before and after the upstream path-only
loader runs.  That API cannot accept an already-open verified descriptor, so
the handoff assumes a trusted, quiescent same-user local filesystem.
"""

from __future__ import annotations

import hashlib
import logging
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from ._assets import (
    AssetIntegrityError,
    FileIdentity,
    PinnedFile,
    atomic_rename_create_only,
    canonical_json_bytes,
    cleanup_private_tree,
    copy_pinned_source,
    ensure_unlinked_directory,
    inspect_pinned_file,
    make_private_stage,
    require_unlinked_directory,
    write_create_only,
)


logger = logging.getLogger(__name__)

SNAPSHOT_MARKER = ".libreyolo_sam3dbody_snapshot.json"
SNAPSHOT_SCHEMA = "libreyolo.sam3dbody-snapshot.v1"
_RUNTIME_NAMES = ("LICENSE", "model.ckpt", "model_config.yaml")
_REMOTE_NAMES = (
    ".gitattributes",
    "LICENSE",
    "README.md",
    "model.ckpt",
    "model_config.yaml",
)
_REPARSE_POINT = 0x400


@dataclass(frozen=True)
class SAMSnapshotPin:
    """One reviewed gated Hub snapshot."""

    size: str
    repo_id: str
    revision: str
    files: tuple[PinnedFile, ...]
    gated: str = "auto"
    legacy_revisions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.size not in {"d3", "h"}:
            raise ValueError(f"unsupported SAM 3D Body size pin: {self.size!r}")
        if len(self.revision) != 40 or any(
            char not in "0123456789abcdef" for char in self.revision
        ):
            raise ValueError(
                f"SAM 3D Body revision must be a lowercase commit: {self.revision}"
            )
        names = tuple(file.path for file in self.files)
        if names != _REMOTE_NAMES:
            raise ValueError(
                "SAM 3D Body snapshot pin must contain the exact remote tree"
            )
        if self.gated != "auto":
            raise ValueError("SAM 3D Body mirrors must use the reviewed auto gate")
        for revision in self.legacy_revisions:
            if len(revision) != 40 or any(
                char not in "0123456789abcdef" for char in revision
            ):
                raise ValueError(
                    "legacy SAM 3D Body revisions must be lowercase commits"
                )
            if revision == self.revision:
                raise ValueError("a legacy SAM 3D Body revision cannot be current")

    @property
    def by_name(self) -> Mapping[str, PinnedFile]:
        return {file.path: file for file in self.files}

    @property
    def runtime_files(self) -> tuple[PinnedFile, ...]:
        by_name = self.by_name
        return tuple(by_name[name] for name in _RUNTIME_NAMES)


@dataclass(frozen=True)
class SAMSnapshotIdentity:
    """Path-free reviewed identity plus the validated local root."""

    root: Path
    size: str
    repo_id: str
    revision: str
    aggregate_sha256: str
    files: tuple[FileIdentity, ...]


_COMMON_ATTRIBUTES = PinnedFile(
    path=".gitattributes",
    size=1_519,
    sha256="11ad7efa24975ee4b0c3c3a38ed18737f0658a5f75a0a96787b576a78a023361",
)
_COMMON_LICENSE = PinnedFile(
    path="LICENSE",
    size=8_204,
    sha256="b3a5a0e2d973ab80e6610ccf1cffc40756050d0ace3cd4fec879b3ec290b2e9b",
)

SAM_SNAPSHOT_PINS: Mapping[str, SAMSnapshotPin] = {
    "d3": SAMSnapshotPin(
        size="d3",
        repo_id="LibreYOLO/LibreSAM3DBodyd3-mesh",
        revision="46e286e25347518d861ab0f21e1b2b5b630dc21f",
        files=(
            _COMMON_ATTRIBUTES,
            _COMMON_LICENSE,
            PinnedFile(
                path="README.md",
                size=3_982,
                sha256="d1e195edb377518f095717bedf9663cad0286ff37e274fb0940da946e9928d3d",
            ),
            PinnedFile(
                path="model.ckpt",
                size=2_109_129_346,
                sha256="b5a2f9d305dd02626b967aa2e86021fba07065df66ce7a7e00ffb9664f150abf",
            ),
            PinnedFile(
                path="model_config.yaml",
                size=1_488,
                sha256="1012fc3f39cb5e90e3f8fbadf7bded31604bfafdce0321d17a7c1a2d3f08b88d",
            ),
        ),
        legacy_revisions=(
            "8e822540228d9de9bef1bf26414e27954044c242",
            "4531d41c4b8349d272a9e7efb42b38a1a5f1d737",
        ),
    ),
    "h": SAMSnapshotPin(
        size="h",
        repo_id="LibreYOLO/LibreSAM3DBodyh-mesh",
        revision="a745fa6fcd5d71e16c4da921a28a6bb6f1ff9e3e",
        files=(
            _COMMON_ATTRIBUTES,
            _COMMON_LICENSE,
            PinnedFile(
                path="README.md",
                size=3_975,
                sha256="71a09372eacd30fd850647884c9f7f91565e161ee2b2d6cb7c5be1613cc6cd3c",
            ),
            PinnedFile(
                path="model.ckpt",
                size=1_691_205_237,
                sha256="3b1cb897f4bbd977bf81cbb0b30780a9582681ac642ee112865790ceb4d66056",
            ),
            PinnedFile(
                path="model_config.yaml",
                size=1_486,
                sha256="d2e772e108b8727e9367681845fecb32806144acd0debc20868d100689470570",
            ),
        ),
        legacy_revisions=(
            "70a2c8ae1f43d6cff94105d83a8dd63d6eeba5ad",
            "b3c59d31106cc69a8ab4cd6510bc289bccf258e9",
        ),
    ),
}


def _pin(size: str) -> SAMSnapshotPin:
    try:
        return SAM_SNAPSHOT_PINS[size]
    except KeyError as exc:
        raise ValueError(
            f"unsupported SAM 3D Body size {size!r}; expected one of {sorted(SAM_SNAPSHOT_PINS)}"
        ) from exc


def default_sam_snapshot_root(size: str) -> Path:
    """Return the revision-keyed managed snapshot path for ``size``."""

    pin = _pin(size)
    return Path.home() / ".cache" / "libreyolo" / "sam3dbody" / size / pin.revision


def _marker_bytes(pin: SAMSnapshotPin, *, revision: str | None = None) -> bytes:
    marker_revision = pin.revision if revision is None else revision
    return canonical_json_bytes(
        {
            "schema": SNAPSHOT_SCHEMA,
            "repo_id": pin.repo_id,
            "revision": marker_revision,
            "files": [
                {
                    "path": file.path,
                    "size": file.size,
                    "sha256": file.sha256,
                }
                for file in pin.runtime_files
            ],
        }
    )


def _is_link_or_reparse(path: Path, identity: os.stat_result) -> bool:
    return stat.S_ISLNK(identity.st_mode) or bool(
        getattr(identity, "st_file_attributes", 0) & _REPARSE_POINT
    )


def _inventory(root: Path) -> tuple[str, ...]:
    require_unlinked_directory(root, label="SAM 3D Body snapshot root")
    names: list[str] = []
    folded: set[str] = set()
    try:
        entries = list(os.scandir(root))
    except OSError as exc:
        raise AssetIntegrityError(
            f"could not inspect SAM 3D Body snapshot: {root}"
        ) from exc
    for entry in entries:
        name = entry.name
        folded_name = name.casefold()
        if folded_name in folded:
            raise AssetIntegrityError(
                "SAM 3D Body snapshot contains case-insensitive duplicate names"
            )
        folded.add(folded_name)
        path = root / name
        try:
            # Native Windows DirEntry.stat() can report st_nlink=0 for an
            # ordinary file while lstat(path) correctly reports 1.
            identity = os.lstat(path)
        except OSError as exc:
            raise AssetIntegrityError(
                f"could not inspect snapshot entry {name}"
            ) from exc
        if (
            entry.is_symlink()
            or _is_link_or_reparse(path, identity)
            or not stat.S_ISREG(identity.st_mode)
            or getattr(identity, "st_nlink", 1) != 1
        ):
            raise AssetIntegrityError(
                f"SAM 3D Body snapshot entry must be an unlinked regular file: {name}"
            )
        names.append(name)
    return tuple(sorted(names, key=str.casefold))


def _aggregate(pin: SAMSnapshotPin, files: tuple[FileIdentity, ...]) -> str:
    payload = canonical_json_bytes(
        {
            "schema": SNAPSHOT_SCHEMA,
            "size": pin.size,
            "repo_id": pin.repo_id,
            "revision": pin.revision,
            "files": [
                {"path": file.path, "size": file.size, "sha256": file.sha256}
                for file in files
            ],
        }
    )
    return hashlib.sha256(payload).hexdigest()


def inspect_sam_snapshot(
    root: str | Path,
    size: str,
    *,
    managed: bool | None = None,
) -> SAMSnapshotIdentity:
    """Strictly validate a local official checkpoint snapshot.

    ``managed=True`` requires LibreYOLO's exact three-file runtime view and its
    canonical revision marker.  Explicit user paths may contain either the
    exact runtime view or the exact full five-file Hub tree, but no other file.
    """

    pin = _pin(size)
    root_path = Path(root).absolute()
    names = _inventory(root_path)
    runtime = tuple(sorted(_RUNTIME_NAMES, key=str.casefold))
    runtime_with_marker = tuple(
        sorted((*_RUNTIME_NAMES, SNAPSHOT_MARKER), key=str.casefold)
    )
    remote = tuple(sorted(_REMOTE_NAMES, key=str.casefold))
    if managed is True:
        accepted = {runtime_with_marker}
    elif managed is False:
        accepted = {runtime, runtime_with_marker, remote}
    else:
        accepted = {runtime, runtime_with_marker, remote}
    if names not in accepted:
        expected = " or ".join(
            ", ".join(group) for group in sorted(accepted, key=lambda group: len(group))
        )
        raise AssetIntegrityError(
            "SAM 3D Body snapshot has an unexpected inventory; expected exactly "
            f"{expected}, got {', '.join(names) or '<empty>'}"
        )

    by_name = pin.by_name
    observed: list[FileIdentity] = []
    for name in names:
        if name == SNAPSHOT_MARKER:
            payload = _marker_bytes(pin)
            marker = PinnedFile(
                path=SNAPSHOT_MARKER,
                size=len(payload),
                sha256=hashlib.sha256(payload).hexdigest(),
            )
            inspect_pinned_file(root_path / name, marker, label="snapshot marker")
            continue
        expected = by_name[name]
        observed.append(
            inspect_pinned_file(
                root_path / name,
                expected,
                label=f"SAM 3D Body snapshot file {name}",
            )
        )
    files = tuple(sorted(observed, key=lambda file: file.path.casefold()))
    return SAMSnapshotIdentity(
        root=root_path,
        size=pin.size,
        repo_id=pin.repo_id,
        revision=pin.revision,
        aggregate_sha256=_aggregate(pin, files),
        files=files,
    )


def _offline_mode() -> bool:
    truthy = {"1", "on", "true", "yes"}
    return any(
        os.environ.get(name, "").strip().casefold() in truthy
        for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    )


def _validated_legacy_sources(
    parent: Path,
    pin: SAMSnapshotPin,
) -> dict[str, Path] | None:
    """Adopt the former unversioned cache only when every byte is canonical."""

    paths = {expected.path: parent / expected.path for expected in pin.runtime_files}
    present = {name for name, path in paths.items() if os.path.lexists(path)}
    if not present:
        return None
    if present != set(_RUNTIME_NAMES):
        missing = sorted(set(_RUNTIME_NAMES) - present, key=str.casefold)
        raise AssetIntegrityError(
            "legacy SAM 3D Body cache is incomplete; missing " + ", ".join(missing)
        )
    for expected in pin.runtime_files:
        inspect_pinned_file(
            paths[expected.path],
            expected,
            label=f"legacy SAM 3D Body file {expected.path}",
        )
    return paths


def _validated_previous_revision_sources(
    parent: Path,
    pin: SAMSnapshotPin,
) -> dict[str, Path] | None:
    """Adopt exact runtime bytes from a superseded card-only revision."""

    for revision in pin.legacy_revisions:
        root = parent / revision
        if not os.path.lexists(root):
            continue
        names = _inventory(root)
        expected_names = tuple(
            sorted((*_RUNTIME_NAMES, SNAPSHOT_MARKER), key=str.casefold)
        )
        if names != expected_names:
            raise AssetIntegrityError(
                "legacy revision-keyed SAM 3D Body cache has an unexpected inventory"
            )
        marker_payload = _marker_bytes(pin, revision=revision)
        inspect_pinned_file(
            root / SNAPSHOT_MARKER,
            PinnedFile(
                path=SNAPSHOT_MARKER,
                size=len(marker_payload),
                sha256=hashlib.sha256(marker_payload).hexdigest(),
            ),
            label="legacy SAM 3D Body snapshot marker",
        )
        sources: dict[str, Path] = {}
        for expected in pin.runtime_files:
            source = root / expected.path
            inspect_pinned_file(
                source,
                expected,
                label=f"legacy SAM 3D Body file {expected.path}",
            )
            sources[expected.path] = source
        return sources
    return None


def _staging_recoveries(parent: Path, pin: SAMSnapshotPin) -> tuple[Path, ...]:
    prefix = f".{pin.revision}.staging-"
    try:
        entries = list(os.scandir(parent))
    except OSError as exc:
        raise AssetIntegrityError(
            f"could not inspect SAM 3D Body cache directory: {parent}"
        ) from exc
    return tuple(
        parent / entry.name for entry in entries if entry.name.startswith(prefix)
    )


def _require_no_staging_recovery(parent: Path, pin: SAMSnapshotPin) -> None:
    recoveries = _staging_recoveries(parent, pin)
    if recoveries:
        paths = ", ".join(str(path) for path in recoveries)
        raise AssetIntegrityError(
            "A previous SAM 3D Body acquisition left private recovery data. "
            "Inspect and remove it before retrying so multi-gigabyte staging "
            f"copies cannot accumulate: {paths}"
        )


def _remote_preflight(hub, pin: SAMSnapshotPin) -> None:
    api = hub.HfApi()
    info = api.model_info(
        repo_id=pin.repo_id,
        revision=pin.revision,
        files_metadata=True,
    )
    if getattr(info, "sha", None) != pin.revision:
        raise AssetIntegrityError(f"Hub resolved {pin.repo_id} to an unexpected commit")
    if getattr(info, "gated", None) != pin.gated:
        raise AssetIntegrityError(
            f"{pin.repo_id}@{pin.revision} no longer has the reviewed {pin.gated!r} gate"
        )
    siblings = list(getattr(info, "siblings", ()) or ())
    sibling_names = [getattr(item, "rfilename", None) for item in siblings]
    if any(not isinstance(name, str) or not name for name in sibling_names):
        raise AssetIntegrityError(
            f"{pin.repo_id}@{pin.revision} returned malformed tree metadata"
        )
    remote_names = tuple(sorted(sibling_names, key=str.casefold))
    if remote_names != tuple(sorted(_REMOTE_NAMES, key=str.casefold)):
        raise AssetIntegrityError(
            f"{pin.repo_id}@{pin.revision} no longer has the reviewed exact tree"
        )
    expected = pin.by_name
    for item in siblings:
        name = item.rfilename
        size = getattr(item, "size", None)
        if (
            isinstance(size, bool)
            or not isinstance(size, int)
            or size != expected[name].size
        ):
            raise AssetIntegrityError(
                f"Hub metadata size mismatch for {pin.repo_id}/{name}"
            )
        lfs = getattr(item, "lfs", None)
        lfs_sha = (
            lfs.get("sha256")
            if isinstance(lfs, Mapping)
            else getattr(lfs, "sha256", None)
        )
        if lfs_sha is not None and lfs_sha != expected[name].sha256:
            raise AssetIntegrityError(
                f"Hub LFS digest mismatch for {pin.repo_id}/{name}"
            )


def _download_metadata_preflight(hub, pin: SAMSnapshotPin, *, offline: bool) -> None:
    """Validate every download leg's declared size before fetching any payload."""

    expected_files = pin.runtime_files if offline else pin.files
    for expected in expected_files:
        try:
            metadata = hub.hf_hub_download(
                repo_id=pin.repo_id,
                filename=expected.path,
                revision=pin.revision,
                local_files_only=offline,
                dry_run=True,
            )
        except TypeError as exc:
            if "dry_run" in str(exc):
                raise RuntimeError(
                    "SAM 3D Body secure downloads require huggingface_hub>=1.0 "
                    "for dry-run byte metadata"
                ) from exc
            raise
        size = getattr(metadata, "file_size", None)
        if isinstance(size, bool) or not isinstance(size, int) or size != expected.size:
            raise AssetIntegrityError(
                f"Hub download metadata size mismatch for {pin.repo_id}/{expected.path}"
            )
        if getattr(metadata, "commit_hash", None) != pin.revision:
            raise AssetIntegrityError(
                f"Hub download metadata commit mismatch for {pin.repo_id}/{expected.path}"
            )


def acquire_sam_snapshot(
    size: str,
    *,
    cache_root: str | Path | None = None,
) -> SAMSnapshotIdentity:
    """Return an exact local runtime view of a pinned gated Hub snapshot."""

    pin = _pin(size)
    destination = (
        default_sam_snapshot_root(size)
        if cache_root is None
        else Path(cache_root).absolute() / pin.revision
    )
    if os.path.lexists(destination):
        return inspect_sam_snapshot(destination, size, managed=True)

    parent = destination.parent
    ensure_unlinked_directory(parent, label="SAM 3D Body cache directory")
    _require_no_staging_recovery(parent, pin)
    sources = _validated_legacy_sources(parent, pin)
    if sources is None:
        sources = _validated_previous_revision_sources(parent, pin)
    legacy_migration = sources is not None
    legacy_paths = tuple(sources.values()) if sources is not None else ()
    hub = None
    offline = False
    direct_download = False
    if sources is None:
        try:
            import huggingface_hub as hub
        except ImportError as exc:
            raise ImportError(
                "huggingface_hub>=1.0 is required to acquire SAM 3D Body weights. "
                'Install it with `pip install "libreyolo[hf]"`.'
            ) from exc

        offline = _offline_mode()
        if not offline:
            try:
                _remote_preflight(hub, pin)
            except Exception as exc:
                if isinstance(exc, AssetIntegrityError):
                    raise
                raise RuntimeError(
                    f"could not verify the gated snapshot {pin.repo_id}@{pin.revision}: {exc}"
                ) from exc

        try:
            _download_metadata_preflight(hub, pin, offline=offline)
        except Exception as exc:
            if isinstance(exc, AssetIntegrityError):
                raise
            raise RuntimeError(
                f"Could not preflight SAM 3D Body weights from "
                f"{pin.repo_id}@{pin.revision}: {exc}"
            ) from exc

        if offline:
            sources = {}
            try:
                for expected in pin.runtime_files:
                    sources[expected.path] = Path(
                        hub.hf_hub_download(
                            repo_id=pin.repo_id,
                            filename=expected.path,
                            revision=pin.revision,
                            local_files_only=True,
                        )
                    )
            except Exception as exc:
                raise RuntimeError(
                    f"Could not load cached SAM 3D Body weights from "
                    f"{pin.repo_id}@{pin.revision}. Populate the exact pinned Hub "
                    f"cache before enabling offline mode.\n\nUnderlying error: {exc}"
                ) from exc
        else:
            direct_download = True

    stage, stage_object = make_private_stage(
        parent,
        prefix=f".{pin.revision}.staging-",
    )
    staged_identity: SAMSnapshotIdentity | None = None
    try:
        if direct_download:
            assert hub is not None
            try:
                for expected in pin.runtime_files:
                    expected_path = (stage / expected.path).absolute()
                    if os.path.lexists(expected_path):
                        raise AssetIntegrityError(
                            f"private download destination already exists: {expected.path}"
                        )
                    downloaded = Path(
                        hub.hf_hub_download(
                            repo_id=pin.repo_id,
                            filename=expected.path,
                            revision=pin.revision,
                            local_files_only=False,
                            local_dir=stage,
                        )
                    ).absolute()
                    if downloaded != expected_path:
                        raise AssetIntegrityError(
                            "Hugging Face returned an unexpected local download path "
                            f"for {expected.path}: {downloaded}"
                        )
                    inspect_pinned_file(
                        expected_path,
                        expected,
                        label=f"downloaded SAM 3D Body file {expected.path}",
                    )
            except Exception as exc:
                if isinstance(exc, AssetIntegrityError):
                    raise
                raise RuntimeError(
                    f"Could not download SAM 3D Body weights from "
                    f"{pin.repo_id}@{pin.revision}.\n\nThese weights are redistributed "
                    "under Meta's SAM License and the mirror is gated. Accept its "
                    "terms on the model page and authenticate with `hf auth login`.\n\n"
                    f"Underlying error: {exc}"
                ) from exc

            hub_metadata = stage / ".cache"
            if os.path.lexists(hub_metadata):
                metadata_seal = require_unlinked_directory(
                    hub_metadata,
                    label="private Hugging Face download metadata",
                )
                cleanup_private_tree(
                    hub_metadata,
                    expected_object=metadata_seal,
                    label="private Hugging Face download metadata",
                )
        else:
            assert sources is not None
            for expected in pin.runtime_files:
                copy_pinned_source(
                    sources[expected.path],
                    stage / expected.path,
                    expected,
                )
        write_create_only(
            stage / SNAPSHOT_MARKER,
            _marker_bytes(pin),
            label="SAM 3D Body snapshot marker",
        )
        staged_identity = inspect_sam_snapshot(stage, size, managed=True)
        stage_seal = require_unlinked_directory(
            stage,
            label="SAM 3D Body staging directory",
        )
        parent_seal = require_unlinked_directory(
            parent,
            label="SAM 3D Body cache directory",
        )
        atomic_rename_create_only(
            stage,
            destination,
            expected_source=stage_seal,
            expected_parent=parent_seal,
        )
        published = inspect_sam_snapshot(destination, size, managed=True)
        if published != SAMSnapshotIdentity(
            root=destination,
            size=staged_identity.size,
            repo_id=staged_identity.repo_id,
            revision=staged_identity.revision,
            aggregate_sha256=staged_identity.aggregate_sha256,
            files=staged_identity.files,
        ):
            raise AssetIntegrityError(
                "SAM 3D Body snapshot changed during create-only publication"
            )
        if legacy_migration:
            logger.warning(
                "Migrated verified SAM 3D Body runtime files to %s. The source "
                "files remain and consume another %.1f GB; remove them only after "
                "confirming the revision-keyed cache works: %s",
                destination,
                sum(file.size for file in pin.runtime_files) / 1e9,
                ", ".join(str(path) for path in legacy_paths),
            )
        return published
    except FileExistsError:
        try:
            winner = inspect_sam_snapshot(destination, size, managed=True)
        except BaseException:
            logger.warning(
                "A concurrent SAM 3D Body publication produced an invalid winner; "
                "the owned staging directory is recoverable at %s and the invalid "
                "competing destination remains at %s; neither path was deleted",
                stage,
                destination,
            )
            raise
        try:
            cleanup_private_tree(
                stage,
                expected_object=stage_object,
                label="losing SAM 3D Body staging directory",
            )
        except AssetIntegrityError:
            logger.warning(
                "A concurrent SAM 3D Body download won publication, but its losing "
                "staging directory could not be cleaned safely: %s",
                stage,
                exc_info=True,
            )
        return winner
    except BaseException:
        if staged_identity is None and os.path.lexists(stage):
            try:
                cleanup_private_tree(
                    stage,
                    expected_object=stage_object,
                    label="incomplete SAM 3D Body staging directory",
                )
            except AssetIntegrityError:
                logger.warning(
                    "SAM 3D Body staging failed and its incomplete private directory "
                    "could not be cleaned safely: %s",
                    stage,
                    exc_info=True,
                )
        else:
            logger.warning(
                "SAM 3D Body publication or post-publication validation failed; "
                "verified recovery may be at staging path %s or destination path %s. "
                "Inspect it before retrying.",
                stage,
                destination,
            )
        raise


__all__ = [
    "SAMSnapshotIdentity",
    "SAMSnapshotPin",
    "SAM_SNAPSHOT_PINS",
    "SNAPSHOT_MARKER",
    "acquire_sam_snapshot",
    "default_sam_snapshot_root",
    "inspect_sam_snapshot",
]
