"""RF-DETR optimizer grouping and migration of legacy optimizer layouts.

Equivalent-group batching follows RF-DETR's Apache-2.0 optimization, reviewed
at roboflow/rf-detr commit 2d319776673ba840c069b243863a4cbe3a62cb58.
The load hook adapts LibreYOLO's existing group order and checkpoint contract.
"""

from copy import deepcopy

import torch

from ...training.optim import OptimizerStateMigrationError, build_optimizer


def build_rfdetr_optimizer(groups, **kwargs):
    """Merge identical settings and preserve legacy per-parameter resume."""
    original = []
    for group in groups:
        params = group["params"]
        original.append(
            {
                **group,
                "params": [params]
                if isinstance(params, torch.Tensor)
                else list(params),
            }
        )
    merged, buckets = [], []
    by_settings = {}
    for index, group in enumerate(original):
        settings = {**kwargs, **{k: v for k, v in group.items() if k != "params"}}
        key = tuple(sorted(settings.items()))
        if key not in by_settings:
            by_settings[key] = len(merged)
            merged.append({**group, "params": []})
            buckets.append([])
        bucket = by_settings[key]
        merged[bucket]["params"].extend(group["params"])
        buckets[bucket].append(index)
    optimizer = build_optimizer(torch.optim.AdamW, merged, **kwargs)

    def migrate_legacy(_optimizer, state):
        saved = state.get("param_groups", [])
        if len(saved) == len(merged):
            return state
        if len(saved) != len(original) or any(
            len(old["params"]) != len(live["params"])
            for old, live in zip(saved, original)
        ):
            raise OptimizerStateMigrationError(
                "RF-DETR optimizer checkpoint does not match the current parameter layout"
            )
        migrated = deepcopy(state)
        regrouped = []
        for indices in buckets:
            first = migrated["param_groups"][indices[0]]
            settings = {k: v for k, v in first.items() if k != "params"}
            if any(
                {k: v for k, v in saved[index].items() if k != "params"} != settings
                for index in indices[1:]
            ):
                raise OptimizerStateMigrationError(
                    "Cannot merge RF-DETR checkpoint groups with different saved settings"
                )
            regrouped.append(
                {
                    **first,
                    "params": [
                        param for index in indices for param in saved[index]["params"]
                    ],
                }
            )
        migrated["param_groups"] = regrouped
        return migrated

    optimizer.register_load_state_dict_pre_hook(migrate_legacy)
    return optimizer
