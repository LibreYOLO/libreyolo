"""Class-balance and split-composition checks (``balance.*``)."""

from collections.abc import Iterator

from ..config import DoctorConfig
from ..report import Finding, Severity
from ..snapshot import DatasetSnapshot
from . import register


def _name(snap: DatasetSnapshot, cid: int) -> str:
    return snap.names.get(cid, f"class {cid}")


@register("balance.class_zero_instances")
def check_class_zero_instances(
    snap: DatasetSnapshot, cfg: DoctorConfig
) -> Iterator[Finding]:
    train = snap.split("train")
    if train is None or not train.records or snap.nc is None:
        return
    counts = train.class_counts()
    missing = [cid for cid in range(snap.nc) if counts.get(cid, 0) == 0]
    if missing:
        listing = ", ".join(_name(snap, c) for c in missing[:10])
        suffix = "..." if len(missing) > 10 else ""
        yield Finding(
            "balance.class_zero_instances",
            Severity.WARNING,
            f"{len(missing)} class(es) have zero training instances: "
            f"{listing}{suffix}. The model can never learn them.",
            split="train",
            count=len(missing),
            details={"classes": [_name(snap, c) for c in missing]},
        )


@register("balance.class_few_instances")
def check_class_few_instances(
    snap: DatasetSnapshot, cfg: DoctorConfig
) -> Iterator[Finding]:
    train = snap.split("train")
    if train is None or not train.records:
        return
    counts = train.class_counts()
    few = {
        cid: n
        for cid, n in counts.items()
        if 0 < n < cfg.few_instances and (snap.nc is None or 0 <= cid < snap.nc)
    }
    if few:
        listing = ", ".join(
            f"{_name(snap, c)} ({n})" for c, n in sorted(few.items())[:10]
        )
        yield Finding(
            "balance.class_few_instances",
            Severity.WARNING,
            f"{len(few)} class(es) have fewer than {cfg.few_instances} "
            f"training instances: {listing}.",
            split="train",
            count=len(few),
            details={"classes": {_name(snap, c): n for c, n in few.items()}},
        )


@register("balance.imbalance")
def check_imbalance(snap: DatasetSnapshot, cfg: DoctorConfig) -> Iterator[Finding]:
    train = snap.split("train")
    if train is None or not train.records:
        return
    counts = {
        cid: n
        for cid, n in train.class_counts().items()
        if n > 0 and (snap.nc is None or 0 <= cid < snap.nc)
    }
    if len(counts) < 2:
        return
    most_id, most = max(counts.items(), key=lambda kv: kv[1])
    least_id, least = min(counts.items(), key=lambda kv: kv[1])
    ratio = most / least
    if ratio < 1.5:  # near-balanced is not worth a report line
        return
    severity = Severity.WARNING if ratio > cfg.imbalance_warn_ratio else Severity.INFO
    yield Finding(
        "balance.imbalance",
        severity,
        f"Class imbalance {ratio:.1f}:1 - most: {_name(snap, most_id)} "
        f"({most}), least: {_name(snap, least_id)} ({least}).",
        split="train",
        details={"ratio": round(ratio, 1)},
    )


@register("balance.split_coverage")
def check_split_coverage(snap: DatasetSnapshot, cfg: DoctorConfig) -> Iterator[Finding]:
    train, val = snap.split("train"), snap.split("val")
    if train is None or val is None or not train.records or not val.records:
        return

    def known(counts):
        # Out-of-range ids are labels.class_out_of_range's job, not coverage's.
        return {
            c
            for c, n in counts.items()
            if n > 0 and (snap.nc is None or 0 <= c < snap.nc)
        }

    train_classes = known(train.class_counts())
    val_classes = known(val.class_counts())
    only_val = sorted(val_classes - train_classes)
    only_train = sorted(train_classes - val_classes)
    if only_val:
        listing = ", ".join(_name(snap, c) for c in only_val[:10])
        yield Finding(
            "balance.split_coverage",
            Severity.WARNING,
            f"{len(only_val)} class(es) appear in val but never in train: "
            f"{listing}. Validation scores them at zero.",
            count=len(only_val),
            details={"classes": [_name(snap, c) for c in only_val]},
        )
    if only_train:
        listing = ", ".join(_name(snap, c) for c in only_train[:10])
        yield Finding(
            "balance.split_coverage",
            Severity.WARNING,
            f"{len(only_train)} class(es) appear in train but never in val: "
            f"{listing}. Their metrics are blind.",
            count=len(only_train),
            details={"classes": [_name(snap, c) for c in only_train]},
        )


@register("balance.background_ratio")
def check_background_ratio(
    snap: DatasetSnapshot, cfg: DoctorConfig
) -> Iterator[Finding]:
    for split in snap.splits:
        if not split.records:
            continue
        background = sum(1 for r in split.records if r.is_background)
        if not background:  # zero background is the unremarkable default
            continue
        ratio = background / len(split.records)
        severity = (
            Severity.WARNING if ratio > cfg.background_warn_ratio else Severity.INFO
        )
        yield Finding(
            "balance.background_ratio",
            severity,
            f"{background} of {len(split.records)} images ({ratio:.1%}) have "
            "no annotations (background).",
            split=split.name,
            count=background,
            details={"ratio": round(ratio, 4)},
        )


@register("balance.split_skew")
def check_split_skew(snap: DatasetSnapshot, cfg: DoctorConfig) -> Iterator[Finding]:
    train, val = snap.split("train"), snap.split("val")
    if train is None or val is None:
        return
    t_counts, v_counts = train.class_counts(), val.class_counts()
    t_total, v_total = sum(t_counts.values()), sum(v_counts.values())
    if not t_total or not v_total:
        return
    skewed = {}
    for cid in set(t_counts) | set(v_counts):
        if snap.nc is not None and not (0 <= cid < snap.nc):
            continue
        diff = abs(t_counts.get(cid, 0) / t_total - v_counts.get(cid, 0) / v_total)
        if diff > cfg.split_skew_points:
            skewed[cid] = diff
    if skewed:
        listing = ", ".join(
            f"{_name(snap, c)} ({d:.0%})"
            for c, d in sorted(skewed.items(), key=lambda kv: -kv[1])[:5]
        )
        yield Finding(
            "balance.split_skew",
            Severity.INFO,
            f"Class distribution differs between train and val by more than "
            f"{cfg.split_skew_points:.0%} for {len(skewed)} class(es): {listing}.",
            count=len(skewed),
            details={
                "classes": {_name(snap, c): round(d, 3) for c, d in skewed.items()}
            },
        )
