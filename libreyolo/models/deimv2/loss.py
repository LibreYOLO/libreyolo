"""DEIMv2 criterion for native LibreYOLO training.

DEIMv2 keeps DEIM's MAL + box + optional local FGL/DDF losses, but adds two
training-time controls not present in LibreYOLO's DEIM port:

- epoch-aware matcher switching;
- optional GO-union matching via ``use_uni_set``.

The scalar loss implementations are reused from ``models.deim.loss`` to keep
the two flat ports aligned where the math is identical.
"""

from __future__ import annotations

import copy

import torch

from ..deim.loss import DEIMCriterion


class DEIMv2Criterion(DEIMCriterion):
    """DEIMv2 loss with epoch-aware matching and configurable GO-union use."""

    def __init__(self, *args, use_uni_set=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_uni_set = use_uni_set

    def forward(self, outputs, targets, epoch=0, **kwargs):
        outputs_without_aux = {k: v for k, v in outputs.items() if "aux" not in k}

        self._clear_cache()

        if "aux_outputs" not in outputs:
            raise RuntimeError(
                "DEIMv2Criterion.forward requires 'aux_outputs' in the model's "
                "training output. Got keys: " + str(list(outputs.keys()))
            )

        # Match the output levels as a depth-2 pipeline: the next level's
        # cost matrix is enqueued on the device before the current one's
        # ``.cpu()`` drains, so the transfer overlaps the next level's compute
        # instead of stalling once per level with nothing queued behind.
        # Depth 2 rather than enqueue-everything: a cost matrix is
        # (bs, num_queries, total_targets) fp32, and holding every level at
        # once raises peak VRAM on dense batches.
        aux_outputs_list = outputs["aux_outputs"]
        if "pre_outputs" in outputs:
            aux_outputs_list = outputs["aux_outputs"] + [outputs["pre_outputs"]]
        levels = [outputs_without_aux, *aux_outputs_list, *outputs["enc_aux_outputs"]]
        level_indices = []
        pending_cost = self.matcher.compute_cost_matrix(levels[0], targets, epoch=epoch)
        for next_level in levels[1:]:
            next_cost = self.matcher.compute_cost_matrix(next_level, targets, epoch=epoch)
            level_indices.append(
                self.matcher.solve(pending_cost.cpu(), targets)["indices"]
            )
            pending_cost = next_cost
        level_indices.append(
            self.matcher.solve(pending_cost.cpu(), targets)["indices"]
        )

        indices = level_indices[0]
        cached_indices = level_indices[1 : 1 + len(aux_outputs_list)]
        cached_indices_enc = level_indices[1 + len(aux_outputs_list) :]
        indices_aux_list = cached_indices + cached_indices_enc
        indices_go = self._get_go_indices(indices, indices_aux_list)

        device = next(iter(outputs.values())).device
        num_boxes_go = self._normalizer(sum(len(x[0]) for x in indices_go), device)
        num_boxes = self._normalizer(sum(len(t["labels"]) for t in targets), device)

        losses = {}
        for loss in self.losses:
            use_uni_set = self.use_uni_set and loss in ["boxes", "local"]
            indices_in = indices_go if use_uni_set else indices
            num_boxes_in = num_boxes_go if use_uni_set else num_boxes
            meta = self.get_loss_meta_info(loss, outputs, targets, indices_in)
            l_dict = self.get_loss(
                loss, outputs, targets, indices_in, num_boxes_in, **meta
            )
            l_dict = {
                k: l_dict[k] * self.weight_dict[k]
                for k in l_dict
                if k in self.weight_dict
            }
            losses.update(l_dict)

        for i, aux_outputs in enumerate(outputs["aux_outputs"]):
            if "local" in self.losses:
                aux_outputs["up"], aux_outputs["reg_scale"] = (
                    outputs["up"],
                    outputs["reg_scale"],
                )
            for loss in self.losses:
                use_uni_set = self.use_uni_set and loss in ["boxes", "local"]
                indices_in = indices_go if use_uni_set else cached_indices[i]
                num_boxes_in = num_boxes_go if use_uni_set else num_boxes
                meta = self.get_loss_meta_info(loss, aux_outputs, targets, indices_in)
                l_dict = self.get_loss(
                    loss, aux_outputs, targets, indices_in, num_boxes_in, **meta
                )
                l_dict = {
                    k: l_dict[k] * self.weight_dict[k]
                    for k in l_dict
                    if k in self.weight_dict
                }
                losses.update({k + f"_aux_{i}": v for k, v in l_dict.items()})

        if "pre_outputs" in outputs:
            aux_outputs = outputs["pre_outputs"]
            for loss in self.losses:
                use_uni_set = self.use_uni_set and loss in ["boxes", "local"]
                indices_in = indices_go if use_uni_set else cached_indices[-1]
                num_boxes_in = num_boxes_go if use_uni_set else num_boxes
                meta = self.get_loss_meta_info(loss, aux_outputs, targets, indices_in)
                l_dict = self.get_loss(
                    loss, aux_outputs, targets, indices_in, num_boxes_in, **meta
                )
                l_dict = {
                    k: l_dict[k] * self.weight_dict[k]
                    for k in l_dict
                    if k in self.weight_dict
                }
                losses.update({k + "_pre": v for k, v in l_dict.items()})

        if "enc_aux_outputs" in outputs:
            assert "enc_meta" in outputs, ""
            class_agnostic = outputs["enc_meta"]["class_agnostic"]
            if class_agnostic:
                orig_num_classes = self.num_classes
                self.num_classes = 1
                enc_targets = copy.deepcopy(targets)
                for t in enc_targets:
                    t["labels"] = torch.zeros_like(t["labels"])
            else:
                enc_targets = targets

            for i, aux_outputs in enumerate(outputs["enc_aux_outputs"]):
                for loss in self.losses:
                    use_uni_set = self.use_uni_set and loss == "boxes"
                    indices_in = indices_go if use_uni_set else cached_indices_enc[i]
                    num_boxes_in = num_boxes_go if use_uni_set else num_boxes
                    meta = self.get_loss_meta_info(
                        loss, aux_outputs, enc_targets, indices_in
                    )
                    l_dict = self.get_loss(
                        loss,
                        aux_outputs,
                        enc_targets,
                        indices_in,
                        num_boxes_in,
                        **meta,
                    )
                    l_dict = {
                        k: l_dict[k] * self.weight_dict[k]
                        for k in l_dict
                        if k in self.weight_dict
                    }
                    losses.update({k + f"_enc_{i}": v for k, v in l_dict.items()})

            if class_agnostic:
                self.num_classes = orig_num_classes

        if "dn_outputs" in outputs:
            assert "dn_meta" in outputs, ""
            indices_dn = self.get_cdn_matched_indices(outputs["dn_meta"], targets)
            dn_num_group = outputs["dn_meta"]["dn_num_group"]
            # num_boxes is clamped to >= 1, so the positivity guard reduces to
            # a host-side check on the group count (no device sync).
            dn_num_boxes = num_boxes * dn_num_group if dn_num_group > 0 else 1

            for i, aux_outputs in enumerate(outputs["dn_outputs"]):
                if "local" in self.losses:
                    aux_outputs["is_dn"] = True
                    aux_outputs["up"], aux_outputs["reg_scale"] = (
                        outputs["up"],
                        outputs["reg_scale"],
                    )
                for loss in self.losses:
                    meta = self.get_loss_meta_info(
                        loss, aux_outputs, targets, indices_dn
                    )
                    l_dict = self.get_loss(
                        loss, aux_outputs, targets, indices_dn, dn_num_boxes, **meta
                    )
                    l_dict = {
                        k: l_dict[k] * self.weight_dict[k]
                        for k in l_dict
                        if k in self.weight_dict
                    }
                    losses.update({k + f"_dn_{i}": v for k, v in l_dict.items()})

            if "dn_pre_outputs" in outputs:
                aux_outputs = outputs["dn_pre_outputs"]
                for loss in self.losses:
                    meta = self.get_loss_meta_info(
                        loss, aux_outputs, targets, indices_dn
                    )
                    l_dict = self.get_loss(
                        loss, aux_outputs, targets, indices_dn, dn_num_boxes, **meta
                    )
                    l_dict = {
                        k: l_dict[k] * self.weight_dict[k]
                        for k in l_dict
                        if k in self.weight_dict
                    }
                    losses.update({k + "_dn_pre": v for k, v in l_dict.items()})

        return {k: torch.nan_to_num(v, nan=0.0) for k, v in losses.items()}
