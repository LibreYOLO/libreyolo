"""Layer-freezing helpers for trainer setup."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from fnmatch import fnmatchcase
from typing import Any, Callable, Iterable, Sequence

import torch.nn as nn


FreezeSelector = int | str
FreezeGroup = tuple[str, nn.Module | Sequence[nn.Module | None]]


@dataclass(frozen=True)
class FreezeSummary:
    """Summary of a freeze operation."""

    selectors: tuple[FreezeSelector, ...]
    frozen_param_names: tuple[str, ...]
    frozen_tensor_count: int
    frozen_param_count: int
    trainable_param_count: int
    total_param_count: int
    frozen_bn_modules: tuple[nn.Module, ...]


def parse_freeze_spec(value: Any) -> Any:
    """Parse CLI/config-friendly freeze values into Python values.

    Numeric values such as ``freeze=10`` expand to ordered family-defined
    groups, while bare names such as ``freeze=backbone`` stay as strings.
    """
    if not isinstance(value, str):
        return value

    raw = value.strip()
    if not raw or raw.lower() in {"none", "null", "false"}:
        return None

    if raw.lower() == "true":
        return True

    try:
        return ast.literal_eval(raw)
    except (SyntaxError, ValueError):
        pass

    if "," in raw:
        return [part.strip() for part in raw.split(",") if part.strip()]

    return raw


def normalize_freeze_selectors(value: Any) -> tuple[FreezeSelector, ...]:
    """Normalize a user freeze value to concrete selectors."""
    value = parse_freeze_spec(value)

    if value is None or value is False:
        return ()
    if value is True:
        raise ValueError("freeze=True is ambiguous; use an int, list, or module name.")

    if isinstance(value, int):
        if value < 0:
            raise ValueError("freeze must be >= 0.")
        return tuple(range(value))

    if isinstance(value, str):
        value = value.strip()
        if not value or value.lower() in {"none", "null", "false"}:
            return ()
        if value.isdecimal():
            return tuple(range(int(value)))
        return (value,)

    if isinstance(value, Iterable):
        selectors: list[FreezeSelector] = []
        for item in value:
            parsed = parse_freeze_spec(item)
            if isinstance(parsed, bool):
                raise ValueError("freeze lists may contain ints or module names, not bools.")
            if isinstance(parsed, int):
                if parsed < 0:
                    raise ValueError("freeze layer indices must be >= 0.")
                selectors.append(parsed)
            elif isinstance(parsed, str):
                parsed = parsed.strip()
                if parsed:
                    selectors.append(int(parsed) if parsed.isdecimal() else parsed)
            else:
                raise TypeError(
                    "freeze lists may contain ints or module-name strings; "
                    f"got {type(parsed).__name__}."
                )
        return tuple(selectors)

    raise TypeError(
        "freeze must be None, an int, a module-name string, or a list of ints/strings."
    )


def default_freeze_groups(model: nn.Module) -> list[FreezeGroup]:
    """Return generic integer-freeze groups for a model.

    Families can override this with semantically stable groups. The fallback is
    intentionally conservative: direct children that own at least one parameter.
    """
    groups: list[FreezeGroup] = []
    for name, child in model.named_children():
        if any(True for _ in child.parameters()):
            groups.append((name, child))
    if not groups and any(True for _ in model.parameters()):
        groups.append(("model", model))
    return groups


def apply_freeze(
    model: nn.Module,
    freeze: Any,
    *,
    freeze_groups: Sequence[FreezeGroup] | None = None,
    freeze_bn_stats: bool = True,
    preserve_trainable_param: Callable[[str, nn.Parameter], bool] | None = None,
) -> FreezeSummary | None:
    """Apply a freeze spec to ``model`` and return a summary.

    Integer selectors address ``freeze_groups`` by index. String selectors match
    group names, module names, and parameter-name prefixes. A leading
    ``model.`` is treated flexibly so both ``backbone`` and ``model.backbone``
    work when a family exposes one style internally.
    """
    selectors = normalize_freeze_selectors(freeze)
    if not selectors:
        return None

    groups = tuple(freeze_groups or default_freeze_groups(model))
    param_by_id = {id(param): (name, param) for name, param in model.named_parameters()}
    if not param_by_id:
        raise ValueError("Cannot freeze layers because the model has no parameters.")

    selected_param_ids: set[int] = set()
    matched_selectors: set[FreezeSelector] = set()

    for selector in selectors:
        if isinstance(selector, int):
            if selector >= len(groups):
                raise ValueError(
                    f"freeze index {selector} is out of range for {len(groups)} "
                    "available freeze groups."
                )
            selected_param_ids.update(_param_ids_for_group(groups[selector], param_by_id))
            matched_selectors.add(selector)
        else:
            matched = _match_string_selector(model, groups, selector, param_by_id)
            if matched:
                selected_param_ids.update(matched)
                matched_selectors.add(selector)

    missing = [selector for selector in selectors if selector not in matched_selectors]
    if missing:
        raise ValueError(
            "freeze selector(s) matched no parameters: "
            + ", ".join(repr(selector) for selector in missing)
        )

    if preserve_trainable_param is not None:
        selected_param_ids = {
            param_id
            for param_id in selected_param_ids
            if not preserve_trainable_param(*param_by_id[param_id])
        }

    trainable_before = {
        param_id
        for param_id, (_name, param) in param_by_id.items()
        if param.requires_grad
    }
    selected_trainable = selected_param_ids & trainable_before
    trainable_after = trainable_before - selected_trainable
    if not trainable_after:
        raise ValueError(
            "freeze would leave no trainable parameters. Use a smaller freeze value "
            "or target a narrower module."
        )

    for param_id in selected_param_ids:
        _name, param = param_by_id[param_id]
        param.requires_grad = False

    frozen_names = tuple(
        name for param_id, (name, _param) in param_by_id.items() if param_id in selected_param_ids
    )
    frozen_bn_modules = (
        tuple(_matched_bn_modules(model, selected_param_ids))
        if freeze_bn_stats
        else ()
    )
    for module in frozen_bn_modules:
        module.eval()

    return FreezeSummary(
        selectors=selectors,
        frozen_param_names=frozen_names,
        frozen_tensor_count=len(selected_param_ids),
        frozen_param_count=sum(param.numel() for param_id, (_name, param) in param_by_id.items() if param_id in selected_param_ids),
        trainable_param_count=sum(param.numel() for param_id, (_name, param) in param_by_id.items() if param_id in trainable_after),
        total_param_count=sum(param.numel() for _name, param in param_by_id.values()),
        frozen_bn_modules=frozen_bn_modules,
    )


def _iter_group_modules(group: FreezeGroup) -> Iterable[nn.Module]:
    modules = group[1]
    if isinstance(modules, nn.Module):
        yield modules
        return
    for module in modules:
        if module is not None:
            yield module


def _param_ids_for_group(
    group: FreezeGroup,
    param_by_id: dict[int, tuple[str, nn.Parameter]],
) -> set[int]:
    selected: set[int] = set()
    known_ids = set(param_by_id)
    for module in _iter_group_modules(group):
        for param in module.parameters():
            param_id = id(param)
            if param_id in known_ids:
                selected.add(param_id)
    return selected


def _selector_variants(selector: str) -> tuple[str, ...]:
    selector = selector.strip().strip("'\"")
    variants = {selector}
    if selector.startswith("model."):
        variants.add(selector[len("model.") :])
    else:
        variants.add(f"model.{selector}")
    return tuple(v for v in variants if v)


def _name_matches(selector: str, name: str) -> bool:
    for variant in _selector_variants(selector):
        if name == variant or name.startswith(f"{variant}."):
            return True
        if any(ch in variant for ch in "*?[]") and fnmatchcase(name, variant):
            return True
    return False


def _match_string_selector(
    model: nn.Module,
    groups: Sequence[FreezeGroup],
    selector: str,
    param_by_id: dict[int, tuple[str, nn.Parameter]],
) -> set[int]:
    selected: set[int] = set()

    if selector.strip().lower() == "all":
        return set(param_by_id)

    for group in groups:
        if _name_matches(selector, group[0]):
            selected.update(_param_ids_for_group(group, param_by_id))

    module_by_name = dict(model.named_modules())
    for name, module in module_by_name.items():
        if name and _name_matches(selector, name):
            for param in module.parameters():
                param_id = id(param)
                if param_id in param_by_id:
                    selected.add(param_id)

    for param_id, (name, _param) in param_by_id.items():
        if _name_matches(selector, name):
            selected.add(param_id)

    return selected


def _matched_bn_modules(
    model: nn.Module,
    selected_param_ids: set[int],
) -> Iterable[nn.Module]:
    bn_types = nn.modules.batchnorm._BatchNorm
    yielded: set[int] = set()
    for module in model.modules():
        if not isinstance(module, bn_types):
            continue
        param_ids = {id(param) for param in module.parameters()}
        if param_ids & selected_param_ids and id(module) not in yielded:
            yielded.add(id(module))
            yield module


__all__ = [
    "FreezeGroup",
    "FreezeSelector",
    "FreezeSummary",
    "apply_freeze",
    "default_freeze_groups",
    "normalize_freeze_selectors",
    "parse_freeze_spec",
]
