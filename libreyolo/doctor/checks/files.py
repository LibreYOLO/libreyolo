"""File-layout checks (``files.*``): orphans, extensions, stem collisions."""

from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path

from ...data.utils import IMG_FORMATS
from ..config import DoctorConfig
from ..report import Finding, Severity
from ..snapshot import DatasetSnapshot
from . import register

# Files that legitimately live next to images/labels.
_COMPANION_NAMES = {"classes.txt", "labels.txt", "notes.json"}
_COMPANION_SUFFIXES = {".txt", ".cache", ".json", ".xml", ".yaml", ".yml", ".csv"}


@register("files.missing_label")
def check_missing_label(snap: DatasetSnapshot, cfg: DoctorConfig) -> Iterator[Finding]:
    for split in snap.splits:
        missing = [r for r in split.records if not r.label_exists]
        if missing:
            yield Finding(
                "files.missing_label",
                Severity.INFO,
                f"{len(missing)} of {len(split.records)} images have no label "
                "file (treated as background during training).",
                split=split.name,
                paths=[r.path for r in missing[: cfg.max_examples]],
                count=len(missing),
            )


@register("files.orphan_label")
def check_orphan_label(snap: DatasetSnapshot, cfg: DoctorConfig) -> Iterator[Finding]:
    for split in snap.splits:
        expected = {r.label_path.resolve() for r in split.records}
        label_dirs = {r.label_path.parent for r in split.records}
        orphans = []
        for label_dir in label_dirs:
            if not label_dir.is_dir():
                continue
            for txt in label_dir.glob("*.txt"):
                if txt.name in _COMPANION_NAMES:
                    continue
                if txt.resolve() not in expected:
                    orphans.append(txt)
        if orphans:
            yield Finding(
                "files.orphan_label",
                Severity.WARNING,
                f"{len(orphans)} label file(s) have no matching image; "
                "they are silently ignored by training.",
                split=split.name,
                paths=sorted(orphans)[: cfg.max_examples],
                count=len(orphans),
            )


@register("files.unsupported_ext")
def check_unsupported_ext(
    snap: DatasetSnapshot, cfg: DoctorConfig
) -> Iterator[Finding]:
    for split_name in ("train", "val", "test"):
        resolved = snap.config.get(split_name)
        paths = resolved if isinstance(resolved, list) else [resolved]
        offenders: list[Path] = []
        for p in paths:
            if not p:
                continue
            split_dir = Path(p)
            if not split_dir.is_dir():
                continue
            for f in split_dir.rglob("*"):
                if not f.is_file():
                    continue
                suffix = f.suffix.lower()
                if suffix in IMG_FORMATS or suffix in _COMPANION_SUFFIXES:
                    continue
                if f.name.startswith("."):
                    continue
                offenders.append(f)
        if offenders:
            yield Finding(
                "files.unsupported_ext",
                Severity.WARNING,
                f"{len(offenders)} non-image file(s) in the image directory "
                "are ignored by the loader.",
                split=split_name,
                paths=sorted(offenders)[: cfg.max_examples],
                count=len(offenders),
            )


@register("files.case_collision")
def check_case_collision(snap: DatasetSnapshot, cfg: DoctorConfig) -> Iterator[Finding]:
    """Images whose label paths collide (a.jpg vs a.JPG vs a.png -> a.txt)."""
    for split in snap.splits:
        groups: dict[str, list[Path]] = defaultdict(list)
        for r in split.records:
            groups[str(r.label_path).lower()].append(r.path)
        collisions = {k: v for k, v in groups.items() if len(v) > 1}
        if collisions:
            examples = [p for paths in collisions.values() for p in paths]
            yield Finding(
                "files.case_collision",
                Severity.WARNING,
                f"{len(collisions)} group(s) of images map to the same label "
                "file (same stem, different case or extension); their "
                "annotations are ambiguous.",
                split=split.name,
                paths=examples[: cfg.max_examples],
                count=len(collisions),
            )
