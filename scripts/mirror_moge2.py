"""Stage the LibreMoGe2 mirror repos for the LibreYOLO Hugging Face org.

    .venv/Scripts/python.exe scripts/mirror_moge2.py <staging-dir>

Why
---
LibreMoGe2 currently fetches from Ruicheng's Hugging Face repos at a pinned
revision, so a first run depends on a third party staying put. The weights are
MIT, which permits redistribution, so there is no reason for that dependency.

This stages one directory per size following the 5-file contract in
skills/libreyolo-upload-hf-model. It does not upload; upload is a separate,
deliberate step.

A size is stageable only after reproducible double conversion, tensor audit,
and parity/load validation produce an approved converted SHA-256 recorded in
``SIZES``.
"""

from __future__ import annotations

import ctypes
import errno
import hashlib
import os
import secrets
import stat
import sys
import tempfile
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterator

import torch

from _mirror_common import parse_args

from libreyolo.utils.serialization import (
    CheckpointMetadataError,
    validate_checkpoint_metadata,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = REPO_ROOT / "weights"
# Upstream repo and the exact revision LibreYOLO pins today, so the mirror
# records which bytes it came from rather than "latest".
SIZES = {
    "l": {
        "converted": "LibreMoGe2l-normal-LibreMoGe2l-normal.pt",
        "upstream": "Ruicheng/moge-2-vitl-normal",
        "revision": "b135031bae30b5ac2ae141a0e68717795ce38340",
        "arch": "MoGe-2 ViT-L/14",
        "sha256": "342c13b7028a2e87d164ee9647ad4f34d822dcb73221004c9f25d0458e17580a",
        "card_size": 24,
        "card_sha256": "d8d7a46d41a1a37fe4f0a5f637bf55c649310185329127d8a2204632e480be17",
    },
    "s": {
        "converted": "LibreMoGe2s-normal-LibreMoGe2s-normal.pt",
        "upstream": "Ruicheng/moge-2-vits-normal",
        "revision": "679230677b4d282c6f304189a93e98e14f085902",
        "arch": "MoGe-2 ViT-S/14",
        "sha256": "0b3c1301ddcae5569234010905f093fa8bec5866c7c06197761ea501651f9d9c",
        "card_size": 24,
        "card_sha256": "d8d7a46d41a1a37fe4f0a5f637bf55c649310185329127d8a2204632e480be17",
    },
    "b": {
        "converted": "LibreMoGe2b-normal-LibreMoGe2b-normal.pt",
        "upstream": "Ruicheng/moge-2-vitb-normal",
        "revision": "54ad3a693e61907ea4633d13dec6ee682fa09419",
        "arch": "MoGe-2 ViT-B/14",
        # Unapproved until reproducible conversion and parity produce a receipt.
        "sha256": None,
        "card_size": 24,
        "card_sha256": "d8d7a46d41a1a37fe4f0a5f637bf55c649310185329127d8a2204632e480be17",
    },
}

MOGE_LICENSE_REVISION = "925b8ed835a7a9cdb7578ba15c658a0afc969030"
MOGE_SOURCE_URL = f"https://github.com/microsoft/MoGe/tree/{MOGE_LICENSE_REVISION}"
MOGE_DINOV2_URL = f"{MOGE_SOURCE_URL}/moge/model/dinov2"
MOGE_LICENSE_PAGE_URL = (
    f"https://github.com/microsoft/MoGe/blob/{MOGE_LICENSE_REVISION}/LICENSE"
)
LICENSE_URL = (
    f"https://raw.githubusercontent.com/microsoft/MoGe/{MOGE_LICENSE_REVISION}/LICENSE"
)
LICENSE_SHA256 = "ad7d951c80c5fc2b2bce035f2041bc0a0dbf9028c8ecc4c9a8e1fba8130b6b59"
LICENSE_SIZE = 12_500
GITATTRIBUTES_REVISION = "1c54f3073f8e03f5818d74ca03e3e2fe5cddfbe0"
GITATTRIBUTES_URL = (
    "https://huggingface.co/LibreYOLO/LibreSegformerb5-sem/resolve/"
    f"{GITATTRIBUTES_REVISION}/.gitattributes"
)
GITATTRIBUTES_SHA256 = (
    "88023d0a029a0c409b30c03b689c68605b559f5cefe06376e4a26b38ed795269"
)
GITATTRIBUTES_SIZE = 1_554
SUPPORT_FETCH_TIMEOUT_SECONDS = 15

README = """---
license: mit
library_name: libreyolo
tags:
  - image-to-image
  - surface-normals
  - moge
---

# {repo}

{arch} surface-normal estimator, repackaged in LibreYOLO checkpoint format.

## Source

Derived from [{upstream} `model.pt`]({source_url})
at revision `{revision}`, the exact commit LibreYOLO pins.
The [revision-pinned source model card]({source_card_url}) declares these
weights MIT.

Conversion follows [MoGe-2 at the audited source commit]({moge_source_url}).
The [vendored DINOv2 encoder snapshot]({moge_dinov2_url}) MoGe-2 builds on is
separately licensed Apache-2.0 by Meta AI; the exact notices are preserved in
MoGe's [commit-pinned composite license]({moge_license_url}).

## Modifications

Unused point, mask, and metric-scale head tensors are removed. Encoder, neck,
and normal-head tensors are retained unchanged, and LibreYOLO checkpoint v1.0
metadata is added.
See `weights/convert_moge2_weights.py` in the
[LibreYOLO source repository](https://github.com/LibreYOLO/libreyolo).

## Usage

```python
from libreyolo import LibreYOLO

model = LibreYOLO("{repo}.pt")
result = model.predict("image.jpg")[0]
normals = result.normals
```

## License

MIT License. See the [`LICENSE`](./LICENSE) and [`NOTICE`](./NOTICE) files in
this repository.
"""

NOTICE = """LibreMoGe2 weights
------------------

This product contains weights derived from MoGe-2
({moge_source_url}).
Copyright (c) Microsoft Corporation.
The revision-pinned source model card declares these weights MIT:
{source_card_url}

Source artifact:  {source_url}
Source revision:  {revision}
Source file:      model.pt
Modification:     unused point, mask, and metric-scale head tensors are
                  removed; encoder, neck, and normal-head tensors are retained
                  unchanged; LibreYOLO checkpoint v1.0 metadata is added by
                  weights/convert_moge2_weights.py in LibreYOLO.

The DINOv2 encoder these weights build on is separately licensed under the
Apache License, Version 2.0, by Meta AI
({moge_dinov2_url}). The exact MIT and Apache notices are preserved in MoGe's
commit-pinned composite license ({moge_license_url}).
"""


@dataclass(frozen=True)
class FileRecord:
    size: int
    sha256: str
    identity: tuple[int, int]


@dataclass
class DirectoryHandle:
    path: Path
    path_identity: tuple[int, int]
    native_identity: tuple[int, int]
    fd: int | None = None
    win_handle: int | None = None
    parent: DirectoryHandle | None = None
    name: str | None = None


def _identity(info: os.stat_result) -> tuple[int, int]:
    return info.st_dev, info.st_ino


def _is_reparse(info: os.stat_result) -> bool:
    attributes = getattr(info, "st_file_attributes", 0)
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return bool(attributes & reparse_flag)


def _is_unlinked_regular(info: os.stat_result) -> bool:
    return stat.S_ISREG(info.st_mode) and not _is_reparse(info) and info.st_nlink == 1


def _open_regular_source(path: Path) -> BinaryIO:
    try:
        entry_info = os.lstat(path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"converted checkpoint missing: {path}") from exc
    if not _is_unlinked_regular(entry_info):
        raise SystemExit(
            f"converted checkpoint must be an unlinked regular file: {path}"
        )

    handle = path.open("rb")
    opened_info = os.fstat(handle.fileno())
    if not _is_unlinked_regular(opened_info) or _identity(opened_info) != _identity(
        entry_info
    ):
        handle.close()
        raise SystemExit(f"converted checkpoint changed while opening: {path}")
    return handle


def _hash_stream(handle: BinaryIO) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)
        size += len(chunk)
    return size, digest.hexdigest()


def _copy_and_hash(source: BinaryIO, destination: BinaryIO) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    for chunk in iter(lambda: source.read(1024 * 1024), b""):
        digest.update(chunk)
        size += len(chunk)
        destination.write(chunk)
    destination.flush()
    os.fsync(destination.fileno())
    return size, digest.hexdigest()


def _fetch_verified_bytes(
    url: str,
    expected_size: int,
    expected_sha256: str,
    label: str,
) -> bytes:
    request = urllib.request.Request(
        url,
        headers={
            "Accept-Encoding": "identity",
            "User-Agent": "LibreYOLO-MoGe-mirror-audit",
        },
    )
    with urllib.request.urlopen(
        request,
        timeout=SUPPORT_FETCH_TIMEOUT_SECONDS,
    ) as response:
        status = response.getcode()
        if status != 200:
            raise SystemExit(f"{label} fetch returned HTTP {status}")
        content_length = response.headers.get("Content-Length")
        if content_length is not None:
            try:
                declared_size = int(content_length)
            except ValueError as exc:
                raise SystemExit(
                    f"{label} returned invalid Content-Length {content_length!r}"
                ) from exc
            if declared_size != expected_size:
                raise SystemExit(
                    f"{label} size mismatch: expected {expected_size}, "
                    f"server declared {declared_size}"
                )

        chunks: list[bytes] = []
        remaining = expected_size + 1
        while remaining:
            chunk = response.read(min(64 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)

    content = b"".join(chunks)
    if len(content) != expected_size:
        raise SystemExit(
            f"{label} size mismatch: expected {expected_size}, got {len(content)}"
        )
    actual_sha256 = hashlib.sha256(content).hexdigest()
    if actual_sha256 != expected_sha256:
        raise SystemExit(
            f"{label} SHA-256 mismatch: expected {expected_sha256}, got {actual_sha256}"
        )
    return content


def _windows_directory_handle(
    path: Path,
    *,
    prevent_rename: bool,
) -> tuple[int, tuple[int, int]]:
    from ctypes import wintypes

    class ByHandleFileInformation(ctypes.Structure):
        _fields_ = [
            ("dwFileAttributes", wintypes.DWORD),
            ("ftCreationTime", wintypes.FILETIME),
            ("ftLastAccessTime", wintypes.FILETIME),
            ("ftLastWriteTime", wintypes.FILETIME),
            ("dwVolumeSerialNumber", wintypes.DWORD),
            ("nFileSizeHigh", wintypes.DWORD),
            ("nFileSizeLow", wintypes.DWORD),
            ("nNumberOfLinks", wintypes.DWORD),
            ("nFileIndexHigh", wintypes.DWORD),
            ("nFileIndexLow", wintypes.DWORD),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create_file = kernel32.CreateFileW
    create_file.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    create_file.restype = wintypes.HANDLE
    share = 0x1 | 0x2
    desired_access = 0x80
    if not prevent_rename:
        # The publishing handle itself needs DELETE access for
        # SetFileInformationByHandle, but sharing DELETE would let another
        # process rename the temporary directory while Windows child opens
        # are still path-based.
        desired_access |= 0x00010000
    handle = create_file(
        str(path),
        desired_access,
        share,
        None,
        3,
        0x02000000 | 0x00200000,
        None,
    )
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())

    get_info = kernel32.GetFileInformationByHandle
    get_info.argtypes = [wintypes.HANDLE, ctypes.POINTER(ByHandleFileInformation)]
    get_info.restype = wintypes.BOOL
    info = ByHandleFileInformation()
    if not get_info(handle, ctypes.byref(info)):
        error = ctypes.get_last_error()
        _close_windows_handle(int(handle))
        raise ctypes.WinError(error)
    if not info.dwFileAttributes & 0x10 or info.dwFileAttributes & 0x400:
        _close_windows_handle(int(handle))
        raise SystemExit(f"staging path is not an unlinked directory: {path}")
    file_index = (info.nFileIndexHigh << 32) | info.nFileIndexLow
    return int(handle), (info.dwVolumeSerialNumber, file_index)


def _close_windows_handle(handle: int) -> None:
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = [wintypes.HANDLE]
    close_handle.restype = wintypes.BOOL
    if not close_handle(handle):
        raise ctypes.WinError(ctypes.get_last_error())


def _directory_native_identity(directory: DirectoryHandle) -> tuple[int, int]:
    if directory.fd is not None:
        return _identity(os.fstat(directory.fd))
    assert directory.win_handle is not None
    from ctypes import wintypes

    class ByHandleFileInformation(ctypes.Structure):
        _fields_ = [
            ("dwFileAttributes", wintypes.DWORD),
            ("ftCreationTime", wintypes.FILETIME),
            ("ftLastAccessTime", wintypes.FILETIME),
            ("ftLastWriteTime", wintypes.FILETIME),
            ("dwVolumeSerialNumber", wintypes.DWORD),
            ("nFileSizeHigh", wintypes.DWORD),
            ("nFileSizeLow", wintypes.DWORD),
            ("nNumberOfLinks", wintypes.DWORD),
            ("nFileIndexHigh", wintypes.DWORD),
            ("nFileIndexLow", wintypes.DWORD),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    get_info = kernel32.GetFileInformationByHandle
    get_info.argtypes = [wintypes.HANDLE, ctypes.POINTER(ByHandleFileInformation)]
    get_info.restype = wintypes.BOOL
    info = ByHandleFileInformation()
    if not get_info(
        directory.win_handle,
        ctypes.byref(info),
    ):
        raise ctypes.WinError(ctypes.get_last_error())
    file_index = (info.nFileIndexHigh << 32) | info.nFileIndexLow
    return info.dwVolumeSerialNumber, file_index


@contextmanager
def _open_directory(
    path: Path,
    *,
    prevent_rename: bool,
    parent: DirectoryHandle | None = None,
    name: str | None = None,
) -> Iterator[DirectoryHandle]:
    entry_info = (
        os.stat(name, dir_fd=parent.fd, follow_symlinks=False)
        if parent is not None and parent.fd is not None and name is not None
        else os.lstat(path)
    )
    if not stat.S_ISDIR(entry_info.st_mode) or _is_reparse(entry_info):
        raise SystemExit(f"staging path is not an unlinked directory: {path}")

    if os.name == "nt":
        win_handle, native_identity = _windows_directory_handle(
            path,
            prevent_rename=prevent_rename,
        )
        if entry_info.st_ino != native_identity[1]:
            _close_windows_handle(win_handle)
            raise SystemExit(f"staging directory changed while opening: {path}")
        directory = DirectoryHandle(
            path=path,
            path_identity=_identity(entry_info),
            native_identity=native_identity,
            win_handle=win_handle,
            parent=parent,
            name=name,
        )
        try:
            yield directory
        finally:
            _close_windows_handle(win_handle)
        return

    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    if parent is not None and parent.fd is not None and name is not None:
        fd = os.open(name, flags, dir_fd=parent.fd)
    else:
        fd = os.open(path, flags)
    opened_info = os.fstat(fd)
    if _identity(opened_info) != _identity(entry_info):
        os.close(fd)
        raise SystemExit(f"staging directory changed while opening: {path}")
    directory = DirectoryHandle(
        path=path,
        path_identity=_identity(entry_info),
        native_identity=_identity(opened_info),
        fd=fd,
        parent=parent,
        name=name,
    )
    try:
        yield directory
    finally:
        os.close(fd)


def _directory_entry_info(directory: DirectoryHandle) -> os.stat_result:
    if (
        directory.parent is not None
        and directory.parent.fd is not None
        and directory.name is not None
    ):
        return os.stat(
            directory.name,
            dir_fd=directory.parent.fd,
            follow_symlinks=False,
        )
    return os.lstat(directory.path)


def _validate_directory_binding(directory: DirectoryHandle) -> None:
    if _directory_native_identity(directory) != directory.native_identity:
        raise SystemExit(f"staging directory handle changed: {directory.path}")
    try:
        entry_info = _directory_entry_info(directory)
    except FileNotFoundError as exc:
        raise SystemExit(f"staging directory disappeared: {directory.path}") from exc
    if (
        not stat.S_ISDIR(entry_info.st_mode)
        or _is_reparse(entry_info)
        or _identity(entry_info) != directory.path_identity
    ):
        raise SystemExit(f"staging directory path changed: {directory.path}")


def _ensure_staging_root(path: Path) -> None:
    if not os.path.lexists(path):
        path.mkdir(parents=True)
    info = os.lstat(path)
    if not stat.S_ISDIR(info.st_mode) or _is_reparse(info):
        raise SystemExit(f"staging root must be an unlinked directory: {path}")


def _create_private_temp(root: DirectoryHandle, repo: str) -> tuple[str, Path]:
    for _ in range(100):
        name = f".{repo}.{secrets.token_hex(12)}"
        path = root.path / name
        try:
            if root.fd is not None:
                os.mkdir(name, mode=0o700, dir_fd=root.fd)
            else:
                os.mkdir(path, mode=0o700)
        except FileExistsError:
            continue
        return name, path
    raise RuntimeError(f"could not allocate a private staging directory for {repo}")


def _entry_path(directory: DirectoryHandle, name: str) -> Path:
    return directory.path / name


def _entry_lstat(directory: DirectoryHandle, name: str) -> os.stat_result:
    if directory.fd is not None:
        return os.stat(name, dir_fd=directory.fd, follow_symlinks=False)
    return os.lstat(_entry_path(directory, name))


def _open_entry(
    directory: DirectoryHandle, name: str, flags: int, mode: int = 0o600
) -> int:
    flags |= getattr(os, "O_BINARY", 0)
    if directory.fd is not None:
        return os.open(name, flags, mode, dir_fd=directory.fd)
    return os.open(_entry_path(directory, name), flags, mode)


def _finish_created_entry(
    directory: DirectoryHandle,
    name: str,
    info: os.stat_result,
    size: int,
    sha256: str,
) -> FileRecord:
    entry_info = _entry_lstat(directory, name)
    if (
        not _is_unlinked_regular(info)
        or not _is_unlinked_regular(entry_info)
        or _identity(info) != _identity(entry_info)
        or info.st_size != size
    ):
        raise SystemExit(f"staged file changed while writing: {name}")
    return FileRecord(size=size, sha256=sha256, identity=_identity(info))


def _write_exclusive_bytes(
    directory: DirectoryHandle,
    name: str,
    content: bytes,
) -> FileRecord:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    fd = _open_entry(directory, name, flags)
    try:
        view = memoryview(content)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise OSError(f"short write while staging {name}")
            view = view[written:]
        os.fsync(fd)
        info = os.fstat(fd)
    finally:
        os.close(fd)
    return _finish_created_entry(
        directory,
        name,
        info,
        len(content),
        hashlib.sha256(content).hexdigest(),
    )


def _copy_exclusive_weight(
    directory: DirectoryHandle,
    name: str,
    source: BinaryIO,
) -> FileRecord:
    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    fd = _open_entry(directory, name, flags)
    try:
        with os.fdopen(fd, "w+b", closefd=False) as destination:
            size, sha256 = _copy_and_hash(source, destination)
        info = os.fstat(fd)
    finally:
        os.close(fd)
    return _finish_created_entry(directory, name, info, size, sha256)


def _read_entry_record(directory: DirectoryHandle, name: str) -> FileRecord:
    before = _entry_lstat(directory, name)
    if not _is_unlinked_regular(before):
        raise SystemExit(f"staged entry is not an unlinked regular file: {name}")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    fd = _open_entry(directory, name, flags)
    try:
        opened = os.fstat(fd)
        if not _is_unlinked_regular(opened) or _identity(opened) != _identity(before):
            raise SystemExit(f"staged entry changed while opening: {name}")
        with os.fdopen(fd, "rb", closefd=False) as handle:
            size, sha256 = _hash_stream(handle)
        after = os.fstat(fd)
    finally:
        os.close(fd)
    final_entry = _entry_lstat(directory, name)
    if (
        not _is_unlinked_regular(after)
        or not _is_unlinked_regular(final_entry)
        or _identity(after) != _identity(before)
        or _identity(final_entry) != _identity(before)
        or after.st_size != size
    ):
        raise SystemExit(f"staged entry changed while hashing: {name}")
    return FileRecord(size=size, sha256=sha256, identity=_identity(after))


@contextmanager
def _open_verified_entry(
    directory: DirectoryHandle,
    name: str,
    expected: FileRecord,
) -> Iterator[BinaryIO]:
    before = _entry_lstat(directory, name)
    if (
        not _is_unlinked_regular(before)
        or _identity(before) != expected.identity
        or before.st_size != expected.size
    ):
        raise SystemExit(f"staged entry changed before opening: {name}")

    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    fd = _open_entry(directory, name, flags)
    try:
        opened = os.fstat(fd)
        if (
            not _is_unlinked_regular(opened)
            or _identity(opened) != expected.identity
            or opened.st_size != expected.size
        ):
            raise SystemExit(f"staged entry changed while opening: {name}")
        with os.fdopen(fd, "rb", buffering=0, closefd=False) as handle:
            yield handle
        after = os.fstat(fd)
    finally:
        os.close(fd)

    final_entry = _entry_lstat(directory, name)
    if (
        not _is_unlinked_regular(after)
        or not _is_unlinked_regular(final_entry)
        or _identity(after) != expected.identity
        or _identity(final_entry) != expected.identity
        or after.st_size != expected.size
    ):
        raise SystemExit(f"staged entry changed while open: {name}")


def _validate_record(
    directory: DirectoryHandle,
    expected: dict[str, FileRecord],
) -> None:
    _validate_directory_binding(directory)
    names = os.listdir(directory.fd if directory.fd is not None else directory.path)
    if sorted(names) != sorted(expected):
        raise SystemExit(
            f"staged inventory mismatch: expected {sorted(expected)}, got {sorted(names)}"
        )
    for name, expected_record in expected.items():
        actual = _read_entry_record(directory, name)
        if actual != expected_record:
            raise SystemExit(
                f"staged file record mismatch for {name}: expected "
                f"{expected_record}, got {actual}"
            )


def _raise_rename_error(error: int, destination: Path) -> None:
    if error in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(
            error,
            "staging destination already exists",
            str(destination),
        )
    raise OSError(error, os.strerror(error), str(destination))


def _native_rename_create_only(
    root: DirectoryHandle,
    temporary: DirectoryHandle,
    destination_name: str,
) -> None:
    destination = root.path / destination_name
    if os.name == "nt":
        from ctypes import wintypes

        assert root.win_handle is not None
        assert temporary.win_handle is not None
        # FileRenameInfo is compatible across supported Windows versions with
        # an absolute target. The source stays bound to temporary.win_handle;
        # root remains open without FILE_SHARE_DELETE and is checked on both
        # sides of the rename.
        destination_path = str(destination.absolute())
        destination_bytes = destination_path.encode("utf-16-le")

        class FileRenameInfo(ctypes.Structure):
            _fields_ = [
                ("ReplaceIfExists", wintypes.BOOLEAN),
                ("RootDirectory", wintypes.HANDLE),
                ("FileNameLength", wintypes.DWORD),
                ("FileName", wintypes.WCHAR * 1),
            ]

        filename_offset = FileRenameInfo.FileName.offset
        buffer_size = (
            filename_offset + len(destination_bytes) + ctypes.sizeof(wintypes.WCHAR)
        )
        rename_buffer = ctypes.create_string_buffer(buffer_size)
        rename_info = FileRenameInfo.from_buffer(rename_buffer)
        rename_info.ReplaceIfExists = False
        rename_info.RootDirectory = None
        rename_info.FileNameLength = len(destination_bytes)
        ctypes.memmove(
            ctypes.addressof(rename_buffer) + filename_offset,
            destination_bytes,
            len(destination_bytes),
        )

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        set_information = kernel32.SetFileInformationByHandle
        set_information.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            wintypes.LPVOID,
            wintypes.DWORD,
        ]
        set_information.restype = wintypes.BOOL
        if not set_information(
            temporary.win_handle,
            3,
            ctypes.byref(rename_buffer),
            buffer_size,
        ):
            error = ctypes.get_last_error()
            if error in {80, 183}:
                raise FileExistsError(
                    errno.EEXIST,
                    "staging destination already exists",
                    str(destination),
                )
            raise ctypes.WinError(error)
        return

    assert temporary.name is not None
    source_name = temporary.name
    assert root.fd is not None
    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform.startswith("linux"):
        renameat2 = getattr(libc, "renameat2", None)
        if renameat2 is None:
            raise RuntimeError("atomic create-only rename requires renameat2")
        renameat2.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameat2.restype = ctypes.c_int
        result = renameat2(
            root.fd,
            os.fsencode(source_name),
            root.fd,
            os.fsencode(destination_name),
            1,
        )
    elif sys.platform == "darwin":
        renameatx_np = getattr(libc, "renameatx_np", None)
        if renameatx_np is None:
            raise RuntimeError("atomic create-only rename requires renameatx_np")
        renameatx_np.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameatx_np.restype = ctypes.c_int
        result = renameatx_np(
            root.fd,
            os.fsencode(source_name),
            root.fd,
            os.fsencode(destination_name),
            0x4,
        )
    else:
        raise RuntimeError(
            f"atomic create-only rename is unsupported on {sys.platform!r}"
        )
    if result != 0:
        _raise_rename_error(ctypes.get_errno(), destination)


def _rename_create_only(
    root: DirectoryHandle,
    temporary: DirectoryHandle,
    destination_name: str,
    expected: dict[str, FileRecord],
) -> None:
    _validate_directory_binding(root)
    _validate_record(temporary, expected)
    _validate_directory_binding(root)
    _validate_directory_binding(temporary)
    assert temporary.name is not None
    _native_rename_create_only(root, temporary, destination_name)
    temporary.name = destination_name
    temporary.path = root.path / destination_name
    try:
        _validate_directory_binding(root)
        _validate_record(temporary, expected)
    except BaseException as exc:
        raise SystemExit(
            f"published destination failed final validation and was left for "
            f"manual removal: {temporary.path}: {exc}"
        ) from exc


def validate_staged_checkpoint(checkpoint: object, *, size: str, repo: str) -> None:
    try:
        validate_checkpoint_metadata(checkpoint, strict=True)
    except CheckpointMetadataError as exc:
        raise SystemExit(f"{repo}: checkpoint metadata invalid: {exc}") from exc

    expected = {
        "model_family": "moge2",
        "size": size,
        "task": "normal",
        "nc": 1,
        "names": {0: "normal"},
        "imgsz": 518,
    }
    if not isinstance(checkpoint, dict):  # Defensive: strict validation rejects this.
        raise SystemExit(f"{repo}: checkpoint metadata invalid: expected a dictionary")
    mismatches = [
        f"{key} expected {value!r}, got {checkpoint.get(key)!r}"
        for key, value in expected.items()
        if checkpoint.get(key) != value
    ]
    if mismatches:
        raise SystemExit(
            f"{repo}: checkpoint metadata mismatch: {'; '.join(mismatches)}"
        )


def _require_approved(size: str, spec: dict) -> str:
    repo = f"LibreMoGe2{size}-normal"
    expected_sha256 = spec.get("sha256")
    if not expected_sha256:
        raise SystemExit(
            f"{repo}: converted artifact is not approved; complete reproducible "
            "double conversion, tensor audit, and parity/load validation, then "
            "record its SHA-256 before staging"
        )
    return expected_sha256


def _destination_exists(root: DirectoryHandle, name: str) -> bool:
    if root.fd is None:
        return os.path.lexists(root.path / name)
    try:
        os.stat(name, dir_fd=root.fd, follow_symlinks=False)
    except FileNotFoundError:
        return False
    return True


def stage(size: str, spec: dict, out_root: Path) -> Path:
    repo = f"LibreMoGe2{size}-normal"
    expected_sha256 = _require_approved(size, spec)

    src = WEIGHTS / spec["converted"]
    out = out_root / repo
    if os.path.lexists(out):
        raise FileExistsError(f"staging destination already exists: {out}")

    with _open_regular_source(src) as source_handle:
        source_size, actual_sha256 = _hash_stream(source_handle)
        if actual_sha256 != expected_sha256:
            raise SystemExit(
                f"{repo}: converted checkpoint SHA-256 mismatch: expected "
                f"{expected_sha256}, got {actual_sha256}"
            )

        source_url = (
            f"https://huggingface.co/{spec['upstream']}/blob/"
            f"{spec['revision']}/model.pt"
        )
        source_card_url = (
            f"https://huggingface.co/{spec['upstream']}/blob/"
            f"{spec['revision']}/README.md"
        )
        source_card_fetch_url = (
            f"https://huggingface.co/{spec['upstream']}/resolve/"
            f"{spec['revision']}/README.md"
        )
        format_args = {
            "repo": repo,
            "arch": spec["arch"],
            "upstream": spec["upstream"],
            "revision": spec["revision"],
            "source_url": source_url,
            "source_card_url": source_card_url,
            "moge_source_url": MOGE_SOURCE_URL,
            "moge_dinov2_url": MOGE_DINOV2_URL,
            "moge_license_url": MOGE_LICENSE_PAGE_URL,
        }
        attributes_bytes = _fetch_verified_bytes(
            GITATTRIBUTES_URL,
            GITATTRIBUTES_SIZE,
            GITATTRIBUTES_SHA256,
            ".gitattributes",
        )
        license_bytes = _fetch_verified_bytes(
            LICENSE_URL,
            LICENSE_SIZE,
            LICENSE_SHA256,
            "LICENSE",
        )
        _fetch_verified_bytes(
            source_card_fetch_url,
            spec["card_size"],
            spec["card_sha256"],
            "source model card",
        )
        notice_bytes = NOTICE.format(**format_args).encode("utf-8")
        readme_bytes = README.format(**format_args).encode("utf-8")

        _ensure_staging_root(out_root)
        temporary_path: Path | None = None
        try:
            with _open_directory(out_root, prevent_rename=True) as root:
                _validate_directory_binding(root)
                if _destination_exists(root, repo):
                    raise FileExistsError(f"staging destination already exists: {out}")
                temporary_name, temporary_path = _create_private_temp(root, repo)
                with _open_directory(
                    temporary_path,
                    prevent_rename=False,
                    parent=root,
                    name=temporary_name,
                ) as temporary:
                    source_handle.seek(0)
                    weight_name = f"{repo}.pt"
                    weight_record = _copy_exclusive_weight(
                        temporary,
                        weight_name,
                        source_handle,
                    )
                    if (
                        weight_record.size != source_size
                        or weight_record.sha256 != expected_sha256
                    ):
                        raise SystemExit(
                            f"{repo}: source changed during descriptor-bound copy"
                        )

                    with _open_verified_entry(
                        temporary,
                        weight_name,
                        weight_record,
                    ) as staged_weight:
                        staged_size, staged_sha256 = _hash_stream(staged_weight)
                        if (
                            staged_size != weight_record.size
                            or staged_sha256 != weight_record.sha256
                        ):
                            raise SystemExit(
                                f"{repo}: private checkpoint changed before load"
                            )
                        staged_weight.seek(0)
                        checkpoint = torch.load(
                            staged_weight,
                            map_location="cpu",
                            weights_only=True,
                        )
                        validate_staged_checkpoint(checkpoint, size=size, repo=repo)
                        staged_weight.seek(0)
                        loaded_size, loaded_sha256 = _hash_stream(staged_weight)
                        if (
                            loaded_size != weight_record.size
                            or loaded_sha256 != weight_record.sha256
                        ):
                            raise SystemExit(
                                f"{repo}: private checkpoint changed during load"
                            )

                    if _read_entry_record(temporary, weight_name) != weight_record:
                        raise SystemExit(
                            f"{repo}: private checkpoint changed after load"
                        )
                    expected = {
                        weight_name: weight_record,
                        ".gitattributes": _write_exclusive_bytes(
                            temporary,
                            ".gitattributes",
                            attributes_bytes,
                        ),
                        "LICENSE": _write_exclusive_bytes(
                            temporary,
                            "LICENSE",
                            license_bytes,
                        ),
                        "NOTICE": _write_exclusive_bytes(
                            temporary,
                            "NOTICE",
                            notice_bytes,
                        ),
                        "README.md": _write_exclusive_bytes(
                            temporary,
                            "README.md",
                            readme_bytes,
                        ),
                    }
                    _rename_create_only(root, temporary, repo, expected)
        except BaseException:
            retained = [
                candidate
                for candidate in (temporary_path, out)
                if candidate is not None and os.path.lexists(candidate)
            ]
            if retained:
                print(
                    f"{repo}: staging failed; no cleanup attempted; retained "
                    f"{', '.join(map(str, retained))} for explicit manual "
                    "inspection/removal",
                    file=sys.stderr,
                    flush=True,
                )
            else:
                print(
                    f"{repo}: staging failed; no cleanup attempted; a sealed "
                    "staging directory may have moved and requires explicit "
                    "manual inspection",
                    file=sys.stderr,
                    flush=True,
                )
            raise

    files = sorted(expected)
    size_mb = source_size / 1e6
    print(f"  {repo}: {files} ({size_mb:.0f} MB)", flush=True)
    return out


def main() -> int:
    args = parse_args(
        __doc__ or "Mirror MoGe-2",
        list(SIZES),
        create_staging=False,
    )
    for size in args.sizes:
        if size not in SIZES:
            raise SystemExit(f"unknown size {size!r}; expected one of {list(SIZES)}")
        _require_approved(size, SIZES[size])
    if args.staging is None:
        args.staging = Path(tempfile.mkdtemp(prefix="libreyolo-mirror-"))
    print(f"staging MoGe-2 mirrors under {args.staging}", flush=True)
    for size in args.sizes:
        stage(size, SIZES[size], args.staging)
    print("staged. upload with huggingface_hub.HfApi().upload_folder(...).", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
