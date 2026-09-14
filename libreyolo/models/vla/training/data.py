"""LeRobot dataset access for the LibreVLA trainer and validator.

The LeRobot v3 layout is the interchange format of the field, so ``act`` has
no LibreYOLO dataset YAML (ADR 0028). This module resolves a ``data``
argument (Hub repo id or local directory), splits episodes into train and
validation, and builds the upstream datasets. The split logic is pure and
unit-tested offline; the dataset builders need the ``vla`` extra.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

__all__ = ["DataSource", "resolve_data_source", "split_episodes", "camera_rename_map"]


@dataclass(frozen=True)
class DataSource:
    """A resolved ``data=`` argument."""

    repo_id: str
    root: Optional[Path]

    @property
    def label(self) -> str:
        return str(self.root) if self.root is not None else self.repo_id


def resolve_data_source(data: Any) -> DataSource:
    """Hub repo id, or a local directory in the LeRobot layout."""
    if data is None or not str(data).strip():
        raise ValueError("data must be a LeRobot dataset repo id or directory.")
    text = str(data).strip()
    path = Path(text).expanduser()
    if path.is_dir():
        if not (path / "meta" / "info.json").is_file():
            raise ValueError(
                f"{path} is a directory but not a LeRobot dataset (no meta/info.json)."
            )
        return DataSource(repo_id=path.name, root=path.resolve())
    if "/" not in text:
        raise ValueError(
            f"data={text!r} is neither an existing directory nor a Hub repo id "
            "of the form owner/name."
        )
    return DataSource(repo_id=text, root=None)


def split_episodes(
    total_episodes: int,
    *,
    val_split: float = 0.1,
    val_episodes: Optional[Sequence[int]] = None,
    train_episodes: Optional[Sequence[int]] = None,
    allow_empty_train: bool = False,
) -> Tuple[List[int], List[int]]:
    """Return ``(train, val)`` episode index lists.

    Explicit lists win. Otherwise the last ``val_split`` fraction of episodes
    (at least one when the dataset has two or more) is held out, so
    validation never sees frames adjacent to training frames of the same
    episode. ``allow_empty_train`` lets a validation-only caller hold out
    every episode.
    """
    total = int(total_episodes)
    if total < 1:
        raise ValueError("The dataset has no episodes.")
    every = list(range(total))
    if val_episodes is not None or train_episodes is not None:
        val = sorted({int(e) for e in (val_episodes or [])})
        if train_episodes is not None:
            train = sorted({int(e) for e in train_episodes})
        else:
            train = [e for e in every if e not in set(val)]
        for name, subset in (("train_episodes", train), ("val_episodes", val)):
            bad = [e for e in subset if e < 0 or e >= total]
            if bad:
                raise ValueError(f"{name} has indices outside 0..{total - 1}: {bad}")
        overlap = sorted(set(train) & set(val))
        if overlap:
            raise ValueError(f"train_episodes and val_episodes overlap: {overlap}")
        if not train and not allow_empty_train:
            raise ValueError("train_episodes is empty.")
        return train, val
    if not 0.0 <= float(val_split) < 1.0:
        raise ValueError("val_split must be in [0, 1).")
    n_val = int(round(total * float(val_split)))
    if total >= 2 and float(val_split) > 0:
        n_val = max(1, min(n_val, total - 1))
    else:
        n_val = 0
    train = every[: total - n_val]
    val = every[total - n_val :]
    return train, val


def camera_rename_map(
    dataset_camera_keys: Sequence[str], slots: Sequence[str]
) -> Dict[str, str]:
    """Map dataset camera keys onto the policy's camera slots, in order.

    Returns ``{"observation.images.up": "observation.images.camera1", ...}``.
    Dataset cameras beyond the policy's slot count are dropped with a warning.
    """
    prefix = "observation.images."
    keys = [str(k) for k in dataset_camera_keys]
    if len(keys) > len(slots):
        logger.warning(
            "Dataset has %d cameras but the policy has %d slots; using %s.",
            len(keys),
            len(slots),
            keys[: len(slots)],
        )
        keys = keys[: len(slots)]
    mapping: Dict[str, str] = {}
    for key, slot in zip(keys, slots):
        target = slot if slot.startswith(prefix) else f"{prefix}{slot}"
        if key != target:
            mapping[key] = target
    return mapping


def camera_names(dataset_camera_keys: Sequence[str], slots: Sequence[str]) -> List[str]:
    """Dataset camera names (without the prefix) in slot order."""
    prefix = "observation.images."
    keys = [str(k) for k in dataset_camera_keys][: len(slots)]
    return [k[len(prefix) :] if k.startswith(prefix) else k for k in keys]
