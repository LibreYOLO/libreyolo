"""RF-DETR AdamW grouping and checkpoint migration.

Adapted from Roboflow RF-DETR training/param_groups.py at
b49df291cdd6b8d19fb722ab402e9a018d2edcd4 (Apache-2.0).
"""

import torch

from libreyolo.training.optim import OptimizerStateMigrationError

_INDEX_KEY = "_rfdetr_param_indices"
# Resolved run configuration owns these. BaseTrainer reapplies the scheduler
# after restore; weight decay must also honor explicit resume overrides.
_RECIPE_KEYS = ("lr", "lr_mult")
_IMPL_KEYS = ("fused", "foreach", "capturable", "differentiable")


def _settings(group, ignore=()):
    return tuple(
        sorted(
            (key, repr(value))
            for key, value in group.items()
            if key not in ("params", _INDEX_KEY, *ignore)
        )
    )


def merge_adamw_groups(
    optimizer: torch.optim.AdamW, *, override_weight_decay=False
) -> torch.optim.AdamW:
    """Merge fresh equivalent groups and map legacy moments by parameter order.

    AdamW is constructed first to resolve defaults and validate duplicates.
    Stable original indices in new state dictionaries permit regrouping when
    configured learning rates or weight decay change on an explicit resume.
    Adapter groups appended after construction remain separate.
    """
    if optimizer.state:
        raise ValueError(
            "RF-DETR groups must be merged before the first optimizer step"
        )
    original = optimizer.param_groups
    if not original or any(len(group["params"]) != 1 for group in original):
        return optimizer
    buckets = {}
    for index, group in enumerate(original):
        buckets.setdefault(_settings(group), []).append(index)
    indices = [tuple(bucket) for bucket in buckets.values()]
    recipe_keys = (
        (*_RECIPE_KEYS, "weight_decay") if override_weight_decay else _RECIPE_KEYS
    )
    optimizer.param_groups = [
        {
            **original[bucket[0]],
            "params": [original[i]["params"][0] for i in bucket],
            _INDEX_KEY: list(bucket),
        }
        for bucket in indices
    ]
    optimizer._rfdetr_parameter_order = tuple(group["params"][0] for group in original)
    legacy_count, merged_count = len(original), len(indices)

    def restore_groups(opt, state_dict):
        saved = state_dict["param_groups"]
        suffix_count = len(opt.param_groups) - merged_count
        prefix_count = len(saved) - suffix_count
        records = {}
        if prefix_count > 0 and all(_INDEX_KEY in g for g in saved[:prefix_count]):
            for group in saved[:prefix_count]:
                positions = group[_INDEX_KEY]
                if len(positions) != len(group["params"]):
                    raise OptimizerStateMigrationError(
                        "Invalid RF-DETR optimizer parameter index metadata"
                    )
                for position, saved_id in zip(positions, group["params"]):
                    if position in records:
                        raise OptimizerStateMigrationError(
                            "Duplicate RF-DETR optimizer parameter index"
                        )
                    records[position] = (saved_id, group)
        elif prefix_count == legacy_count and all(
            len(g["params"]) == 1 for g in saved[:prefix_count]
        ):
            records = {
                i: (g["params"][0], g) for i, g in enumerate(saved[:prefix_count])
            }
        else:
            raise OptimizerStateMigrationError(
                "Cannot map RF-DETR optimizer state onto this trainable parameter layout"
            )
        if set(records) != set(range(legacy_count)):
            raise OptimizerStateMigrationError(
                "RF-DETR checkpoint and live optimizer have different trainable parameters"
            )
        groups = []
        for fresh, bucket in zip(opt.param_groups, indices):
            first = records[bucket[0]][1]
            if any(
                _settings(records[i][1], (*recipe_keys, *_IMPL_KEYS))
                != _settings(first, (*recipe_keys, *_IMPL_KEYS))
                for i in bucket
            ):
                raise OptimizerStateMigrationError(
                    "Legacy RF-DETR optimizer groups have incompatible saved hyperparameters"
                )
            restored = {
                **first,
                "params": [records[i][0] for i in bucket],
                _INDEX_KEY: list(bucket),
            }
            restored.update(
                {key: fresh[key] for key in (*recipe_keys, *_IMPL_KEYS) if key in fresh}
            )
            groups.append(restored)
        # Saved state IDs remain valid: PyTorch maps each group to live params.
        return {**state_dict, "param_groups": groups + saved[prefix_count:]}

    optimizer.register_load_state_dict_pre_hook(restore_groups)
    return optimizer
