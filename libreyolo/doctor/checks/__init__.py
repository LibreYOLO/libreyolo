"""Check registry for LibreDoctor.

A check is a pure function ``(DatasetSnapshot, DoctorConfig) -> Iterable[Finding]``
registered under a dotted id (``family.name``). ``--skip``/``--only`` select
by full id or by family prefix (``images`` matches ``images.corrupt``).
"""

from collections.abc import Iterable
from typing import Callable

from ..config import DoctorConfig, UnknownCheckError
from ..report import Finding
from ..snapshot import DatasetSnapshot

CheckFn = Callable[[DatasetSnapshot, DoctorConfig], Iterable[Finding]]

_REGISTRY: dict[str, CheckFn] = {}
_NEEDS_IMAGE_SCAN: set[str] = set()


def register(
    check_id: str, needs_image_scan: bool = False
) -> Callable[[CheckFn], CheckFn]:
    def decorator(fn: CheckFn) -> CheckFn:
        _REGISTRY[check_id] = fn
        if needs_image_scan:
            _NEEDS_IMAGE_SCAN.add(check_id)
        return fn

    return decorator


def all_checks() -> dict[str, CheckFn]:
    _load()
    return dict(_REGISTRY)


def needs_image_scan(check_id: str) -> bool:
    return check_id in _NEEDS_IMAGE_SCAN


def select_checks(
    skip: Iterable[str] = (), only: Iterable[str] = ()
) -> tuple[dict[str, CheckFn], list[str]]:
    """Resolve --skip/--only selectors into (selected, skipped_ids).

    Raises UnknownCheckError for selectors that match nothing, so typos fail
    loudly instead of silently passing a dataset.
    """
    _load()
    skip = [s.strip() for s in skip if s.strip()]
    only = [s.strip() for s in only if s.strip()]

    for selector in (*skip, *only):
        if not any(_matches(cid, selector) for cid in _REGISTRY):
            known = sorted({cid.split(".")[0] for cid in _REGISTRY})
            raise UnknownCheckError(
                f"Unknown check '{selector}'. "
                f"Use a check id or family: {', '.join(known)}."
            )

    selected: dict[str, CheckFn] = {}
    skipped: list[str] = []
    for cid, fn in _REGISTRY.items():
        wanted = any(_matches(cid, s) for s in only) if only else True
        excluded = any(_matches(cid, s) for s in skip)
        if wanted and not excluded:
            selected[cid] = fn
        else:
            skipped.append(cid)
    return selected, skipped


def _matches(check_id: str, selector: str) -> bool:
    return check_id == selector or check_id.startswith(selector + ".")


def _load() -> None:
    """Import check modules so their @register calls run."""
    from . import balance, dataset_config, files, images, labels  # noqa: F401
