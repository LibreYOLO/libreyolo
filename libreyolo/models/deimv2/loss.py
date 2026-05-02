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

from ..deim.loss import (
    DEIMCriterion,
    _get_world_size,
    _is_dist_available_and_initialized,
)


class DEIMv2Criterion(DEIMCriterion):
    """DEIMv2 loss with epoch-aware matching and configurable GO-union use."""

    def __init__(self, *args, use_uni_set=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_uni_set = use_uni_set

    def forward(self, outputs, targets, epoch=0, **kwargs):
        outputs_without_aux = {k: v for k, v in outputs.items() if "aux" not in k}

        indices = self.matcher(outputs_without_aux, targets, epoch=epoch)["indices"]
        self._clear_cache()

        if "aux_outputs" not in outputs:
            raise RuntimeError(
                "DEIMv2Criterion.forward requires 'aux_outputs' in the model's "
                "training output. Got keys: " + str(list(outputs.keys()))
            )

        indices_aux_list, cached_indices, cached_indices_enc = [], [], []
        aux_outputs_list = outputs["aux_outputs"]
        if "pre_outputs" in outputs:
            aux_outputs_list = outputs["aux_outputs"] + [outputs["pre_outputs"]]
        for aux_outputs in aux_outputs_list:
            indices_aux = self.matcher(aux_outputs, targets, epoch=epoch)["indices"]
            cached_indices.append(indices_aux)
            indices_aux_list.append(indices_aux)
        for aux_outputs in outputs["enc_aux_outputs"]:
            indices_enc = self.matcher(aux_outputs, targets, epoch=epoch)["indices"]
            cached_indices_enc.append(indices_enc)
            indices_aux_list.append(indices_enc)
        indices_go = self._get_go_indices(indices, indices_aux_list)

        num_boxes_go = sum(len(x[0]) for x in indices_go)
        num_boxes_go = torch.as_tensor(
            [num_boxes_go],
            dtype=torch.float,
            device=next(iter(outputs.values())).device,
        )
        if _is_dist_available_and_initialized():
            torch.distributed.all_reduce(num_boxes_go)
        num_boxes_go = torch.clamp(num_boxes_go / _get_world_size(), min=1).item()

        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor(
            [num_boxes],
            dtype=torch.float,
            device=next(iter(outputs.values())).device,
        )
        if _is_dist_available_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / _get_world_size(), min=1).item()

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
            dn_num_boxes = num_boxes * outputs["dn_meta"]["dn_num_group"]
            dn_num_boxes = dn_num_boxes if dn_num_boxes > 0 else 1

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
