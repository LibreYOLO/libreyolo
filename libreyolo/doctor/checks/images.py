"""Image-content checks (``images.*``) and cross-split leakage (``splits.*``).

These require the threaded decode pass (skipped by ``--fast``). Near-duplicate
detection uses 64-bit dHashes with band-bucketing: for a Hamming threshold of
4 bits, splitting the hash into 5 bands guarantees (pigeonhole) that any
qualifying pair shares at least one intact band, so only bucket members are
compared exactly.
"""

from collections import Counter, defaultdict
from collections.abc import Iterator
from itertools import combinations

from ..config import DoctorConfig
from ..report import Finding, Severity
from ..snapshot import DatasetSnapshot, ImageRecord, hamming
from . import register

# Cap pathological buckets (e.g. thousands of identical-looking frames) so the
# exact-comparison stage stays sub-quadratic.
_MAX_BUCKET = 200


def _scanned(records: list[ImageRecord]) -> list[ImageRecord]:
    return [r for r in records if r.scan is not None]


@register("images.corrupt", needs_image_scan=True)
def check_corrupt(snap: DatasetSnapshot, cfg: DoctorConfig) -> Iterator[Finding]:
    for split in snap.splits:
        bad = [r for r in _scanned(split.records) if not r.scan.ok]
        if bad:
            sample = "; ".join(f"{r.path.name} ({r.scan.error})" for r in bad[:3])
            yield Finding(
                "images.corrupt",
                Severity.ERROR,
                f"{len(bad)} image(s) cannot be decoded, e.g. {sample}.",
                split=split.name,
                paths=[r.path for r in bad[: cfg.max_examples]],
                count=len(bad),
            )


@register("images.exif_orientation", needs_image_scan=True)
def check_exif_orientation(
    snap: DatasetSnapshot, cfg: DoctorConfig
) -> Iterator[Finding]:
    for split in snap.splits:
        rotated = [
            r
            for r in _scanned(split.records)
            if r.scan.ok and r.scan.exif_orientation in range(2, 9)
        ]
        if rotated:
            yield Finding(
                "images.exif_orientation",
                Severity.WARNING,
                f"{len(rotated)} image(s) carry an EXIF orientation flag; "
                "annotations drawn on the rotated view will be misaligned in "
                "loaders that ignore EXIF. Bake the rotation into the pixels.",
                split=split.name,
                paths=[r.path for r in rotated[: cfg.max_examples]],
                count=len(rotated),
            )


@register("images.odd_mode", needs_image_scan=True)
def check_odd_mode(snap: DatasetSnapshot, cfg: DoctorConfig) -> Iterator[Finding]:
    records = [r for s in snap.splits for r in _scanned(s.records) if r.scan.ok]
    if not records:
        return
    channel_counts = Counter(
        r.scan.channels for r in records if r.scan.channels is not None
    )
    if len(channel_counts) < 2:
        return
    majority_channels, _ = channel_counts.most_common(1)[0]
    odd = [
        r
        for r in records
        if r.scan.channels is not None and r.scan.channels != majority_channels
    ]
    modes = Counter(r.scan.mode for r in odd)
    listing = ", ".join(f"{m} x{n}" for m, n in modes.most_common())
    yield Finding(
        "images.odd_mode",
        Severity.WARNING,
        f"{len(odd)} image(s) have a different channel layout than the "
        f"rest of the dataset ({listing}); they are converted implicitly "
        "and may train inconsistently.",
        paths=[r.path for r in odd[: cfg.max_examples]],
        count=len(odd),
        details={"modes": dict(modes)},
    )


@register("images.tiny_or_extreme", needs_image_scan=True)
def check_tiny_or_extreme(
    snap: DatasetSnapshot, cfg: DoctorConfig
) -> Iterator[Finding]:
    for split in snap.splits:
        offenders = []
        for r in _scanned(split.records):
            s = r.scan
            if not s.ok or not s.width or not s.height:
                continue
            tiny = min(s.width, s.height) < cfg.min_image_side
            extreme = (
                max(s.width, s.height) / min(s.width, s.height)
                > cfg.extreme_image_aspect
            )
            if tiny or extreme:
                offenders.append(r)
        if offenders:
            yield Finding(
                "images.tiny_or_extreme",
                Severity.WARNING,
                f"{len(offenders)} image(s) are smaller than "
                f"{cfg.min_image_side} px a side or have an aspect ratio "
                f"beyond {cfg.extreme_image_aspect:g}:1.",
                split=split.name,
                paths=[r.path for r in offenders[: cfg.max_examples]],
                count=len(offenders),
            )


@register("images.uniform", needs_image_scan=True)
def check_uniform(snap: DatasetSnapshot, cfg: DoctorConfig) -> Iterator[Finding]:
    for split in snap.splits:
        flat = [r for r in _scanned(split.records) if r.scan.ok and r.scan.uniform]
        if flat:
            yield Finding(
                "images.uniform",
                Severity.WARNING,
                f"{len(flat)} image(s) are a single flat color (often failed "
                "downloads or padding artifacts).",
                split=split.name,
                paths=[r.path for r in flat[: cfg.max_examples]],
                count=len(flat),
            )


@register("images.exact_duplicates", needs_image_scan=True)
def check_exact_duplicates(
    snap: DatasetSnapshot, cfg: DoctorConfig
) -> Iterator[Finding]:
    for split in snap.splits:
        groups: dict[str, list[ImageRecord]] = defaultdict(list)
        for r in _scanned(split.records):
            if r.scan.sha1:
                groups[r.scan.sha1].append(r)
        dup_groups = [g for g in groups.values() if len(g) > 1]
        if dup_groups:
            extras = sum(len(g) - 1 for g in dup_groups)
            examples = [r.path for g in dup_groups for r in g]
            yield Finding(
                "images.exact_duplicates",
                Severity.WARNING,
                f"{extras} duplicate image(s) across {len(dup_groups)} "
                "group(s) of byte-identical files; duplicates skew training "
                "statistics.",
                split=split.name,
                paths=examples[: cfg.max_examples],
                count=extras,
            )


@register("images.near_duplicates", needs_image_scan=True)
def check_near_duplicates(
    snap: DatasetSnapshot, cfg: DoctorConfig
) -> Iterator[Finding]:
    for split in snap.splits:
        records = [
            r for r in _scanned(split.records) if r.scan.ok and r.scan.dhash is not None
        ]
        pairs = _near_pairs(records, cfg.near_duplicate_distance)
        # Exact duplicates are reported separately; keep only visually-near.
        pairs = [(a, b) for a, b in pairs if a.scan.sha1 != b.scan.sha1]
        if pairs:
            examples = [p for a, b in pairs for p in (a.path, b.path)]
            yield Finding(
                "images.near_duplicates",
                Severity.INFO,
                f"{len(pairs)} pair(s) of visually near-identical images "
                "(perceptual hash); common with extracted video frames.",
                split=split.name,
                paths=examples[: cfg.max_examples],
                count=len(pairs),
            )


@register("splits.leakage_exact", needs_image_scan=True)
def check_leakage_exact(snap: DatasetSnapshot, cfg: DoctorConfig) -> Iterator[Finding]:
    by_split: dict[str, dict[str, ImageRecord]] = {}
    for split in snap.splits:
        by_split[split.name] = {
            r.scan.sha1: r for r in _scanned(split.records) if r.scan.sha1
        }
    names = list(by_split)
    for a, b in combinations(names, 2):
        shared = set(by_split[a]) & set(by_split[b])
        if shared:
            examples = []
            for sha in shared:
                examples.extend([by_split[a][sha].path, by_split[b][sha].path])
            yield Finding(
                "splits.leakage_exact",
                Severity.ERROR,
                f"{len(shared)} identical image(s) appear in both {a} and "
                f"{b}; validation metrics are inflated.",
                paths=examples[: cfg.max_examples],
                count=len(shared),
                details={"splits": [a, b]},
            )


@register("splits.leakage_near", needs_image_scan=True)
def check_leakage_near(snap: DatasetSnapshot, cfg: DoctorConfig) -> Iterator[Finding]:
    tagged: list[tuple[str, ImageRecord]] = []
    for split in snap.splits:
        for r in _scanned(split.records):
            if r.scan.ok and r.scan.dhash is not None:
                tagged.append((split.name, r))
    pairs = _near_pairs([r for _, r in tagged], cfg.near_duplicate_distance)
    split_of = {id(r): name for name, r in tagged}
    cross = [
        (a, b)
        for a, b in pairs
        if split_of[id(a)] != split_of[id(b)] and a.scan.sha1 != b.scan.sha1
    ]
    if cross:
        examples = [p for a, b in cross for p in (a.path, b.path)]
        yield Finding(
            "splits.leakage_near",
            Severity.WARNING,
            f"{len(cross)} pair(s) of visually near-identical images span "
            "different splits; validation may be leaking.",
            paths=examples[: cfg.max_examples],
            count=len(cross),
        )


def _near_pairs(
    records: list[ImageRecord], max_dist: int
) -> list[tuple[ImageRecord, ImageRecord]]:
    """All record pairs whose dHash Hamming distance is <= max_dist."""
    n_bands = max_dist + 1
    band_bits = 64 // n_bands
    mask = (1 << band_bits) - 1

    buckets: dict[tuple[int, int], list[ImageRecord]] = defaultdict(list)
    for r in records:
        for band in range(n_bands):
            key = (band, (r.scan.dhash >> (band * band_bits)) & mask)
            buckets[key].append(r)

    seen: set[tuple[int, int]] = set()
    pairs: list[tuple[ImageRecord, ImageRecord]] = []
    for members in buckets.values():
        if len(members) < 2:
            continue
        for a, b in combinations(members[:_MAX_BUCKET], 2):
            key = (min(id(a), id(b)), max(id(a), id(b)))
            if key in seen:
                continue
            seen.add(key)
            if hamming(a.scan.dhash, b.scan.dhash) <= max_dist:
                pairs.append((a, b))
    return pairs
