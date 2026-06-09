"""Checks on the dataset YAML itself (``config.*``)."""

from collections import Counter
from collections.abc import Iterator
from pathlib import Path

from ..config import DoctorConfig
from ..report import Finding, Severity
from ..snapshot import DatasetSnapshot
from . import register


@register("config.missing_names")
def check_missing_names(snap: DatasetSnapshot, cfg: DoctorConfig) -> Iterator[Finding]:
    if not snap.names:
        yield Finding(
            "config.missing_names",
            Severity.ERROR,
            "The YAML defines no class names ('names' is missing or empty).",
        )


@register("config.nc_names_mismatch")
def check_nc_names_mismatch(
    snap: DatasetSnapshot, cfg: DoctorConfig
) -> Iterator[Finding]:
    raw_nc = snap.raw_config.get("nc")
    if isinstance(raw_nc, int) and snap.names and raw_nc != len(snap.names):
        yield Finding(
            "config.nc_names_mismatch",
            Severity.ERROR,
            f"nc={raw_nc} but 'names' defines {len(snap.names)} classes.",
            details={"nc": raw_nc, "names": len(snap.names)},
        )


@register("config.missing_split")
def check_missing_split(snap: DatasetSnapshot, cfg: DoctorConfig) -> Iterator[Finding]:
    if not snap.raw_config.get("train"):
        yield Finding(
            "config.missing_split",
            Severity.ERROR,
            "The YAML defines no 'train' split.",
        )
    if not snap.raw_config.get("val"):
        yield Finding(
            "config.missing_split",
            Severity.WARNING,
            "The YAML defines no 'val' split; training cannot be evaluated.",
        )


@register("config.path_not_found")
def check_path_not_found(snap: DatasetSnapshot, cfg: DoctorConfig) -> Iterator[Finding]:
    for split_name in ("train", "val", "test"):
        if not snap.raw_config.get(split_name):
            continue
        split = snap.split(split_name)
        if split is not None and split.records:
            continue
        resolved = snap.config.get(split_name)
        paths = resolved if isinstance(resolved, list) else [resolved]
        missing = [p for p in paths if p and not Path(p).exists()]
        if missing:
            message = f"'{split_name}' path does not exist."
        else:
            message = f"'{split_name}' path exists but contains no images."
        yield Finding(
            "config.path_not_found",
            Severity.ERROR,
            message,
            split=split_name,
            paths=[Path(p) for p in (missing or paths) if p],
        )


@register("config.duplicate_names")
def check_duplicate_names(
    snap: DatasetSnapshot, cfg: DoctorConfig
) -> Iterator[Finding]:
    counts = Counter(snap.names.values())
    duplicated = {name: n for name, n in counts.items() if n > 1}
    if duplicated:
        listing = ", ".join(f"'{name}' x{n}" for name, n in sorted(duplicated.items()))
        yield Finding(
            "config.duplicate_names",
            Severity.WARNING,
            f"Multiple class ids share the same name: {listing}.",
            details={"duplicates": duplicated},
        )
