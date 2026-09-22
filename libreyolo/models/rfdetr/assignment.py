"""Optional batched assignment for RF-DETR's compact matching matrices.

Uses the Apache-2.0 torch-hungarian public API. Shape bucketing follows the
RF-DETR optimization at 2d319776673ba840c069b243863a4cbe3a62cb58; this adapter
retains LibreYOLO's per-image/group output ordering and CPU index contract.
"""

from collections import defaultdict

import torch


def assign_compact(matrices, sizes, group_detr):
    """Solve same-shaped layer/image/group problems together.

    Cost matrices are [queries, sum(target_counts)]. Empty images require
    no solver call. Results have the same ascending-row and group order as
    the SciPy loop. The dependency selects its supported CUDA implementation
    or CPU fallback; import it only after the caller selects this backend.
    """
    import torch_linear_assignment

    buckets = defaultdict(list)
    pairs = [[{} for _ in sizes] for _ in matrices]
    for layer_index, matrix in enumerate(matrices):
        if matrix.shape[0] % group_detr:
            raise ValueError("RF-DETR query count must be divisible by group_detr")
        width = matrix.shape[0] // group_detr
        for image_index, image in enumerate(matrix.split(sizes, dim=1)):
            if image.shape[1] == 0:
                continue
            for group, cost in enumerate(image.split(width, dim=0)):
                buckets[(width, image.shape[1])].append(
                    (layer_index, image_index, group, cost)
                )
    for (width, target_count), problems in buckets.items():
        costs = torch.stack([p[3] for p in problems])
        if costs.device.type not in {"cpu", "cuda"}:
            costs = costs.cpu()
        assignment = torch_linear_assignment.batch_linear_assignment(costs)
        matches = min(width, target_count)
        order = torch.argsort(
            (assignment >= 0).to(torch.int8), dim=1, descending=True, stable=True
        )
        rows = order[:, :matches]
        columns = assignment.gather(1, rows)
        host = torch.stack((rows, columns)).cpu()
        for index, (layer, image, group, _) in enumerate(problems):
            pairs[layer][image][group] = (
                host[0, index] + group * width,
                host[1, index],
            )
    empty = torch.empty(0, dtype=torch.int64)
    return [
        [
            (
                torch.cat([groups[group][0] for group in range(group_detr)]),
                torch.cat([groups[group][1] for group in range(group_detr)]),
            )
            if groups
            else (empty.clone(), empty.clone())
            for groups in layer
        ]
        for layer in pairs
    ]
