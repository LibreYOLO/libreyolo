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
    # YAML can deliver nc as a float (e.g. "nc: 5.0"); treat it numerically.
    if isinstance(raw_nc, bool) or not isinstance(raw_nc, (int, float)):
        return
    if snap.names and raw_nc != len(snap.names):
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
    # Validate every configured entry: with list splits, one good directory
    # must not hide a missing or empty sibling.
    for split_name in ("train", "val", "test"):
        if not snap.raw_config.get(split_name):
            continue
        resolved = snap.config.get(split_name)
        entries = resolved if isinstance(resolved, list) else [resolved]
        missing: list[Path] = []
        empty: list[Path] = []
        for entry in entries:
            if not entry:
                continue
            path = Path(entry)
            if not path.exists():
                missing.append(path)
            elif path.is_dir() and not _contains_images(path):
                empty.append(path)
        if missing:
            yield Finding(
                "config.path_not_found",
                Severity.ERROR,
                f"'{split_name}' path(s) do not exist.",
                split=split_name,
                paths=missing[: cfg.max_examples],
                count=len(missing),
            )
        if empty:
            yield Finding(
                "config.path_not_found",
                Severity.ERROR,
                f"'{split_name}' path(s) exist but contain no images.",
                split=split_name,
                paths=empty[: cfg.max_examples],
                count=len(empty),
            )


def _contains_images(directory: Path) -> bool:
    from ...data.utils import IMG_FORMATS

    return any(
        f.suffix.lower() in IMG_FORMATS for f in directory.rglob("*") if f.is_file()
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
