"""Strict byte-identity helpers for SAM 3D Body and MHR assets.

This module contains LibreYOLO-authored transport code only.  It does not
inspect, import, or derive from the SAM 3D Body implementation.  The wrapper
uses it to ensure that pickle/TorchScript inputs match reviewed upstream bytes
before handing them to third-party loaders.
"""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import os
import stat
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterator


_CHUNK_BYTES = 1024 * 1024
_REPARSE_POINT = 0x400


class AssetIntegrityError(RuntimeError):
    """A local or downloaded model asset failed its pinned byte contract."""


@dataclass(frozen=True)
class PinnedFile:
    """One reviewed file identity."""

    path: str
    size: int
    sha256: str

    def __post_init__(self) -> None:
        if (
            not self.path
            or Path(self.path).name != self.path
            or "/" in self.path
            or "\\" in self.path
        ):
            raise ValueError(
                f"pinned asset path must be a safe basename: {self.path!r}"
            )
        if (
            isinstance(self.size, bool)
            or not isinstance(self.size, int)
            or self.size <= 0
        ):
            raise ValueError(
                f"pinned asset size must be a positive integer: {self.path}"
            )
        if (
            not isinstance(self.sha256, str)
            or len(self.sha256) != 64
            or any(char not in "0123456789abcdef" for char in self.sha256)
        ):
            raise ValueError(f"pinned asset SHA-256 is malformed: {self.path}")


@dataclass(frozen=True)
class FileIdentity:
    """Observed pinned byte identity for one regular file."""

    path: str
    size: int
    sha256: str


@dataclass(frozen=True)
class FileSeal:
    """Filesystem identity held across a publication operation."""

    device: int
    inode: int
    mode: int
    size: int
    mtime_ns: int
    links: int


def _is_link_or_reparse(path: Path, identity: os.stat_result | None = None) -> bool:
    info = identity if identity is not None else os.lstat(path)
    return stat.S_ISLNK(info.st_mode) or bool(
        getattr(info, "st_file_attributes", 0) & _REPARSE_POINT
    )


def _seal(identity: os.stat_result) -> FileSeal:
    return FileSeal(
        device=identity.st_dev,
        inode=identity.st_ino,
        mode=identity.st_mode,
        size=identity.st_size,
        mtime_ns=identity.st_mtime_ns,
        links=getattr(identity, "st_nlink", 1),
    )


def _same_seal(left: FileSeal, right: os.stat_result) -> bool:
    return left == _seal(right)


def _same_object(left: FileSeal, right: FileSeal | os.stat_result) -> bool:
    observed = right if isinstance(right, FileSeal) else _seal(right)
    return (
        left.device == observed.device
        and left.inode == observed.inode
        and stat.S_IFMT(left.mode) == stat.S_IFMT(observed.mode)
    )


def _directory_chain(path: Path) -> tuple[Path, ...]:
    """Return lexical path components from the filesystem anchor to ``path``."""

    current = path.absolute()
    chain = [current]
    while current.parent != current:
        current = current.parent
        chain.append(current)
    return tuple(reversed(chain))


def _require_existing_directory_chain(path: Path, *, label: str) -> FileSeal:
    """Reject a linked, reparsed, or non-directory component in ``path``."""

    leaf: os.stat_result | None = None
    for component in _directory_chain(path):
        try:
            identity = os.lstat(component)
        except OSError as exc:
            raise AssetIntegrityError(
                f"{label} has a missing or inaccessible path component: {component}"
            ) from exc
        if _is_link_or_reparse(component, identity) or not stat.S_ISDIR(
            identity.st_mode
        ):
            raise AssetIntegrityError(
                f"{label} must contain only unlinked directories: {component}"
            )
        leaf = identity
    if leaf is None:  # pragma: no cover - every absolute path has an anchor
        raise AssetIntegrityError(f"{label} has no filesystem anchor: {path}")
    return _seal(leaf)


def require_unlinked_directory(path: Path, *, label: str) -> FileSeal:
    """Seal a directory whose entire lexical path is link/reparse-free."""

    return _require_existing_directory_chain(path, label=label)


def ensure_unlinked_directory(path: Path, *, label: str) -> FileSeal:
    """Create a cache directory if absent, then seal its lexical location."""

    absolute = path.absolute()
    for component in _directory_chain(absolute):
        if not os.path.lexists(component):
            break
        try:
            identity = os.lstat(component)
        except OSError as exc:
            raise AssetIntegrityError(
                f"could not inspect {label} path component: {component}"
            ) from exc
        if _is_link_or_reparse(component, identity) or not stat.S_ISDIR(
            identity.st_mode
        ):
            raise AssetIntegrityError(
                f"{label} must contain only unlinked directories: {component}"
            )
    try:
        absolute.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise AssetIntegrityError(f"could not create {label}: {absolute}") from exc
    return require_unlinked_directory(absolute, label=label)


def _open_regular_file(
    path: Path,
    *,
    label: str,
    require_single_link: bool,
) -> tuple[int, FileSeal]:
    try:
        before = os.lstat(path)
    except OSError as exc:
        raise AssetIntegrityError(
            f"{label} is missing or inaccessible: {path}"
        ) from exc
    if (
        _is_link_or_reparse(path, before)
        or not stat.S_ISREG(before.st_mode)
        or (require_single_link and getattr(before, "st_nlink", 1) != 1)
    ):
        raise AssetIntegrityError(f"{label} must be an unlinked regular file: {path}")

    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise AssetIntegrityError(f"could not open {label}: {path}") from exc
    try:
        opened = os.fstat(descriptor)
        expected = _seal(before)
        if (
            not stat.S_ISREG(opened.st_mode)
            or (require_single_link and getattr(opened, "st_nlink", 1) != 1)
            or not _same_seal(expected, opened)
        ):
            raise AssetIntegrityError(f"{label} changed while it was opened: {path}")
        return descriptor, expected
    except BaseException:
        os.close(descriptor)
        raise


def _hash_open_file(
    stream: BinaryIO,
    *,
    expected_size: int,
    label: str,
) -> tuple[int, str]:
    stream.seek(0)
    digest = hashlib.sha256()
    total = 0
    while True:
        chunk = stream.read(min(_CHUNK_BYTES, expected_size - total + 1))
        if not chunk:
            break
        total += len(chunk)
        if total > expected_size:
            raise AssetIntegrityError(
                f"{label} exceeds its pinned {expected_size}-byte size"
            )
        digest.update(chunk)
    return total, digest.hexdigest()


@contextmanager
def open_verified_file(
    path: Path,
    expected: PinnedFile,
    *,
    label: str,
) -> Iterator[BinaryIO]:
    """Yield the same descriptor whose complete bytes were verified.

    A second descriptor-bound hash and a pathname identity check run after the
    caller returns.  This closes pathname substitution and persistent in-place
    mutation around deserialization without trusting a prior stat call.
    """

    descriptor, original = _open_regular_file(
        path,
        label=label,
        require_single_link=True,
    )
    stream = os.fdopen(descriptor, "rb")
    try:
        observed_size, observed_sha = _hash_open_file(
            stream,
            expected_size=expected.size,
            label=label,
        )
        if observed_size != expected.size or observed_sha != expected.sha256:
            raise AssetIntegrityError(
                f"{label} does not match the reviewed bytes "
                f"(expected {expected.size} bytes/{expected.sha256}, got "
                f"{observed_size} bytes/{observed_sha})"
            )
        if not _same_seal(original, os.fstat(stream.fileno())):
            raise AssetIntegrityError(f"{label} changed during verification: {path}")
        stream.seek(0)
        yield stream

        final_size, final_sha = _hash_open_file(
            stream,
            expected_size=expected.size,
            label=label,
        )
        if final_size != expected.size or final_sha != expected.sha256:
            raise AssetIntegrityError(f"{label} changed while it was in use: {path}")
        if not _same_seal(original, os.fstat(stream.fileno())):
            raise AssetIntegrityError(f"{label} changed while it was in use: {path}")
        try:
            after = os.lstat(path)
        except OSError as exc:
            raise AssetIntegrityError(
                f"{label} disappeared while it was in use: {path}"
            ) from exc
        if _is_link_or_reparse(path, after) or not _same_seal(original, after):
            raise AssetIntegrityError(
                f"{label} path changed while it was in use: {path}"
            )
    finally:
        stream.close()


def inspect_pinned_file(
    path: Path, expected: PinnedFile, *, label: str
) -> FileIdentity:
    """Hash a local file through a stable descriptor and return its identity."""

    descriptor, original = _open_regular_file(
        path,
        label=label,
        require_single_link=True,
    )
    with os.fdopen(descriptor, "rb") as stream:
        observed_size, observed_sha = _hash_open_file(
            stream,
            expected_size=expected.size,
            label=label,
        )
        if observed_size != expected.size or observed_sha != expected.sha256:
            raise AssetIntegrityError(
                f"{label} does not match the reviewed bytes "
                f"(expected {expected.size} bytes/{expected.sha256}, got "
                f"{observed_size} bytes/{observed_sha})"
            )
        if not _same_seal(original, os.fstat(stream.fileno())):
            raise AssetIntegrityError(f"{label} changed during verification: {path}")
    try:
        after = os.lstat(path)
    except OSError as exc:
        raise AssetIntegrityError(
            f"{label} disappeared during verification: {path}"
        ) from exc
    if _is_link_or_reparse(path, after) or not _same_seal(original, after):
        raise AssetIntegrityError(f"{label} path changed during verification: {path}")
    return FileIdentity(path=expected.path, size=expected.size, sha256=expected.sha256)


def copy_pinned_source(
    source: Path, destination: Path, expected: PinnedFile
) -> FileSeal:
    """Copy exact reviewed bytes from a Hub/cache source into a private file.

    Hub snapshot paths may themselves be symlinks into a content-addressed blob
    store.  The source is therefore allowed to resolve through a link, but the
    opened target descriptor and copied bytes are fully verified.  The private
    destination is always a create-only, single-link regular file.
    """

    try:
        resolved = source.resolve(strict=True)
        source_before = os.stat(resolved, follow_symlinks=False)
    except OSError as exc:
        raise AssetIntegrityError(
            f"downloaded source is inaccessible: {source}"
        ) from exc
    if _is_link_or_reparse(resolved, source_before) or not stat.S_ISREG(
        source_before.st_mode
    ):
        raise AssetIntegrityError(f"downloaded source is not a regular file: {source}")

    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        source_descriptor = os.open(resolved, flags)
        destination_descriptor = os.open(
            destination,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0),
            0o600,
        )
    except OSError as exc:
        if "source_descriptor" in locals():
            os.close(source_descriptor)
        raise AssetIntegrityError(
            f"could not stage pinned asset {expected.path}"
        ) from exc

    digest = hashlib.sha256()
    total = 0
    source_stream = os.fdopen(source_descriptor, "rb")
    try:
        destination_stream = os.fdopen(destination_descriptor, "wb")
    except BaseException:
        source_stream.close()
        os.close(destination_descriptor)
        raise
    try:
        with source_stream as src, destination_stream as dst:
            source_opened = os.fstat(src.fileno())
            if _seal(source_before) != _seal(source_opened):
                raise AssetIntegrityError(
                    f"downloaded source changed while opening: {expected.path}"
                )
            while True:
                chunk = src.read(min(_CHUNK_BYTES, expected.size - total + 1))
                if not chunk:
                    break
                total += len(chunk)
                if total > expected.size:
                    raise AssetIntegrityError(
                        f"downloaded {expected.path} exceeds its pinned size"
                    )
                digest.update(chunk)
                dst.write(chunk)
            dst.flush()
            os.fsync(dst.fileno())
            if _seal(source_opened) != _seal(os.fstat(src.fileno())):
                raise AssetIntegrityError(
                    f"downloaded source changed while copied: {expected.path}"
                )
            destination_seal = _seal(os.fstat(dst.fileno()))
    except BaseException:
        # The caller deliberately owns cleanup.  Never unlink a pathname after
        # a separate identity check: a concurrent replacement could be user data.
        raise

    observed_sha = digest.hexdigest()
    if total != expected.size or observed_sha != expected.sha256:
        raise AssetIntegrityError(
            f"downloaded {expected.path} does not match the reviewed bytes "
            f"(expected {expected.size}/{expected.sha256}, got {total}/{observed_sha})"
        )
    destination_after = os.lstat(destination)
    if (
        _is_link_or_reparse(destination, destination_after)
        or not stat.S_ISREG(destination_after.st_mode)
        or getattr(destination_after, "st_nlink", 1) != 1
        or not _same_seal(destination_seal, destination_after)
    ):
        raise AssetIntegrityError(f"staged asset changed after copy: {expected.path}")
    return destination_seal


def write_create_only(path: Path, payload: bytes, *, label: str) -> FileSeal:
    """Write and fsync one private create-only file."""

    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0),
            0o600,
        )
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
            result = _seal(os.fstat(stream.fileno()))
    except OSError as exc:
        raise AssetIntegrityError(f"could not create {label}: {path}") from exc
    return result


def canonical_json_bytes(value: object) -> bytes:
    """Encode a strict stable JSON file."""

    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("ascii")


def _raise_rename_error(number: int, destination: Path) -> None:
    if number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(
            number,
            "refusing to replace an existing pinned asset",
            os.fspath(destination),
        )
    raise OSError(number, os.strerror(number), os.fspath(destination))


def atomic_rename_create_only(
    source: Path,
    destination: Path,
    *,
    expected_source: FileSeal,
    expected_parent: FileSeal,
) -> None:
    """Publish one file or directory without replacing an existing name.

    POSIX calls are relative to an opened parent descriptor, preventing a
    renamed parent from redirecting publication.  Every platform validates the
    destination inode and lexical parent after the syscall; on uncertainty the
    function fails and intentionally leaves the new entry for manual recovery.
    """

    if source.parent != destination.parent:
        raise ValueError("create-only asset publication requires one shared parent")
    parent = source.parent
    current_parent = require_unlinked_directory(parent, label="asset cache parent")
    if not _same_object(expected_parent, current_parent):
        raise AssetIntegrityError("asset cache parent changed before publication")
    try:
        source_before = os.lstat(source)
    except OSError as exc:
        raise AssetIntegrityError(
            "asset staging entry disappeared before publication"
        ) from exc
    if _is_link_or_reparse(source, source_before) or not _same_seal(
        expected_source, source_before
    ):
        raise AssetIntegrityError("asset staging entry changed before publication")

    if os.name == "nt":
        try:
            os.rename(source, destination)
        except (FileExistsError, PermissionError) as exc:
            if os.path.lexists(destination):
                raise FileExistsError(
                    f"refusing to replace an existing pinned asset: {destination}"
                ) from exc
            raise
    else:
        parent_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        if hasattr(os, "O_NOFOLLOW"):
            parent_flags |= os.O_NOFOLLOW
        parent_descriptor = os.open(parent, parent_flags)
        try:
            if not _same_object(expected_parent, os.fstat(parent_descriptor)):
                raise AssetIntegrityError(
                    "asset cache parent changed while opening for publication"
                )
            libc = ctypes.CDLL(None, use_errno=True)
            if sys.platform.startswith("linux"):
                rename = getattr(libc, "renameat2", None)
                if rename is None:
                    raise AssetIntegrityError(
                        "atomic create-only asset publication requires renameat2"
                    )
                rename.argtypes = [
                    ctypes.c_int,
                    ctypes.c_char_p,
                    ctypes.c_int,
                    ctypes.c_char_p,
                    ctypes.c_uint,
                ]
                rename.restype = ctypes.c_int
                result = rename(
                    parent_descriptor,
                    os.fsencode(source.name),
                    parent_descriptor,
                    os.fsencode(destination.name),
                    1,
                )
            elif sys.platform == "darwin":
                rename = getattr(libc, "renameatx_np", None)
                if rename is None:
                    raise AssetIntegrityError(
                        "atomic create-only asset publication requires renameatx_np"
                    )
                rename.argtypes = [
                    ctypes.c_int,
                    ctypes.c_char_p,
                    ctypes.c_int,
                    ctypes.c_char_p,
                    ctypes.c_uint,
                ]
                rename.restype = ctypes.c_int
                result = rename(
                    parent_descriptor,
                    os.fsencode(source.name),
                    parent_descriptor,
                    os.fsencode(destination.name),
                    0x00000004,
                )
            else:
                raise AssetIntegrityError(
                    f"atomic asset publication is unsupported on {sys.platform!r}"
                )
            if result != 0:
                _raise_rename_error(ctypes.get_errno(), destination)
        finally:
            os.close(parent_descriptor)

    final_parent = require_unlinked_directory(parent, label="asset cache parent")
    if not _same_object(expected_parent, final_parent):
        raise AssetIntegrityError("asset cache parent changed during publication")
    try:
        published = os.lstat(destination)
    except OSError as exc:
        raise AssetIntegrityError("published asset disappeared") from exc
    if _is_link_or_reparse(destination, published) or not _same_seal(
        expected_source, published
    ):
        raise AssetIntegrityError("published asset identity changed during publication")


def make_private_stage(parent: Path, *, prefix: str) -> tuple[Path, FileSeal]:
    """Create and seal a private same-filesystem staging directory."""

    try:
        path = Path(tempfile.mkdtemp(dir=parent, prefix=prefix))
        os.chmod(path, 0o700)
    except OSError as exc:
        raise AssetIntegrityError(
            f"could not create asset staging directory in {parent}"
        ) from exc
    return path, require_unlinked_directory(path, label="asset staging directory")


def cleanup_private_file(
    path: Path,
    *,
    expected_object: FileSeal,
    label: str,
) -> bool:
    """Remove a process-created private file if it is still the same object.

    Returns ``False`` when publication already moved the pathname. Cleanup is
    deliberately limited to unpredictable, mode-0600 staging names created by
    this process; callers must never use it for user-supplied cache entries.
    """

    try:
        observed = os.lstat(path)
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise AssetIntegrityError(f"could not inspect {label}: {path}") from exc
    if (
        _is_link_or_reparse(path, observed)
        or not stat.S_ISREG(observed.st_mode)
        or not _same_object(expected_object, observed)
    ):
        raise AssetIntegrityError(
            f"refusing to clean a replaced or non-regular {label}: {path}"
        )
    try:
        path.unlink()
    except OSError as exc:
        raise AssetIntegrityError(f"could not clean {label}: {path}") from exc
    return True


def _collect_private_tree(
    root: Path,
    *,
    label: str,
) -> tuple[list[tuple[Path, FileSeal]], list[tuple[Path, FileSeal]]]:
    files: list[tuple[Path, FileSeal]] = []
    directories: list[tuple[Path, FileSeal]] = []
    try:
        entries = list(os.scandir(root))
    except OSError as exc:
        raise AssetIntegrityError(f"could not inspect {label}: {root}") from exc
    for entry in entries:
        path = root / entry.name
        try:
            observed = os.lstat(path)
        except OSError as exc:
            raise AssetIntegrityError(f"could not inspect {label}: {path}") from exc
        if _is_link_or_reparse(path, observed):
            raise AssetIntegrityError(
                f"refusing to clean linked or reparsed {label}: {path}"
            )
        if stat.S_ISDIR(observed.st_mode):
            nested_files, nested_directories = _collect_private_tree(
                path,
                label=label,
            )
            files.extend(nested_files)
            directories.extend(nested_directories)
            directories.append((path, _seal(observed)))
        elif stat.S_ISREG(observed.st_mode) and getattr(observed, "st_nlink", 1) == 1:
            files.append((path, _seal(observed)))
        else:
            raise AssetIntegrityError(
                f"refusing to clean non-private {label} entry: {path}"
            )
    return files, directories


def cleanup_private_tree(
    path: Path,
    *,
    expected_object: FileSeal,
    label: str,
) -> bool:
    """Remove a process-created private tree after strict link/object checks.

    The tree is scanned before deletion and every entry is revalidated at the
    point of removal. Returns ``False`` when publication already moved the root.
    As with asset loading, concurrent mutation by another same-user process is
    outside the supported local-filesystem trust boundary.
    """

    try:
        observed_root = os.lstat(path)
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise AssetIntegrityError(f"could not inspect {label}: {path}") from exc
    if (
        _is_link_or_reparse(path, observed_root)
        or not stat.S_ISDIR(observed_root.st_mode)
        or not _same_object(expected_object, observed_root)
    ):
        raise AssetIntegrityError(
            f"refusing to clean a replaced or non-directory {label}: {path}"
        )

    files, directories = _collect_private_tree(path, label=label)
    try:
        for file_path, expected in files:
            current = os.lstat(file_path)
            if (
                _is_link_or_reparse(file_path, current)
                or not stat.S_ISREG(current.st_mode)
                or not _same_seal(expected, current)
            ):
                raise AssetIntegrityError(
                    f"refusing to clean changed {label} entry: {file_path}"
                )
            file_path.unlink()
        for directory, expected in directories:
            current = os.lstat(directory)
            if (
                _is_link_or_reparse(directory, current)
                or not stat.S_ISDIR(current.st_mode)
                or not _same_object(expected, current)
            ):
                raise AssetIntegrityError(
                    f"refusing to clean changed {label} directory: {directory}"
                )
            directory.rmdir()
        final_root = os.lstat(path)
        if (
            _is_link_or_reparse(path, final_root)
            or not stat.S_ISDIR(final_root.st_mode)
            or not _same_object(expected_object, final_root)
        ):
            raise AssetIntegrityError(f"refusing to clean changed {label}: {path}")
        path.rmdir()
    except AssetIntegrityError:
        raise
    except OSError as exc:
        raise AssetIntegrityError(f"could not clean {label}: {path}") from exc
    return True


__all__ = [
    "AssetIntegrityError",
    "FileIdentity",
    "FileSeal",
    "PinnedFile",
    "atomic_rename_create_only",
    "canonical_json_bytes",
    "cleanup_private_file",
    "cleanup_private_tree",
    "copy_pinned_source",
    "ensure_unlinked_directory",
    "inspect_pinned_file",
    "make_private_stage",
    "open_verified_file",
    "require_unlinked_directory",
    "write_create_only",
]
