"""
Loss functions for RT-DETR.

SetCriterion computes the loss for RT-DETR with support for:
- Varifocal Loss (VFL) — IoU-aware classification (default)
- Sigmoid Focal Loss
- Binary Cross-Entropy Loss
- L1 + GIoU bounding box losses
- Auxiliary losses from intermediate decoder layers
- Denoising auxiliary losses

Reference: https://github.com/lyuwenyu/RT-DETR
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from scipy.optimize import linear_sum_assignment

from .box_ops import box_cxcywh_to_xyxy, box_iou, generalized_box_iou


# =============================================================================
# Hungarian Matcher
# =============================================================================

class HungarianMatcher(nn.Module):
    """Bipartite matching between predictions and ground-truth.

    Supports both focal-loss-based and softmax-based matching costs.
    """
    def __init__(self, weight_dict, use_focal_loss=False, alpha=0.25, gamma=2.0):
        super().__init__()
        self.cost_class = weight_dict['cost_class']
        self.cost_bbox = weight_dict['cost_bbox']
        self.cost_giou = weight_dict['cost_giou']
        self.use_focal_loss = use_focal_loss
        self.alpha = alpha
        self.gamma = gamma
        assert self.cost_class != 0 or self.cost_bbox != 0 or self.cost_giou != 0, \
            "All costs cannot be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """
        Args:
            outputs: dict with 'pred_logits' [B, Q, C] and 'pred_boxes' [B, Q, 4]
            targets: list of dicts with 'labels' and 'boxes'
        
        Returns:
            List of (pred_indices, target_indices) tuples per batch element.
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        if self.use_focal_loss:
            out_prob = F.sigmoid(outputs["pred_logits"].flatten(0, 1))
        else:
            out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)

        out_bbox = outputs["pred_boxes"].flatten(0, 1)

        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        if len(tgt_ids) == 0:
            return [(torch.as_tensor([], dtype=torch.int64),
                     torch.as_tensor([], dtype=torch.int64)) for _ in range(bs)]

        # Classification cost
        if self.use_focal_loss:
            out_prob = out_prob[:, tgt_ids]
            neg_cost_class = (1 - self.alpha) * (out_prob ** self.gamma) * (
                -(1 - out_prob + 1e-8).log())
            pos_cost_class = self.alpha * ((1 - out_prob) ** self.gamma) * (
                -(out_prob + 1e-8).log())
            cost_class = pos_cost_class - neg_cost_class
        else:
            cost_class = -out_prob[:, tgt_ids]

        # L1 cost
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # GIoU cost
        cost_giou = -generalized_box_iou(
            box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

        # Final cost matrix
        C = (self.cost_bbox * cost_bbox +
             self.cost_class * cost_class +
             self.cost_giou * cost_giou)
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i])
                   for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64),
                 torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


# =============================================================================
# Set Criterion (Loss)
# =============================================================================

class SetCriterion(nn.Module):
    """RT-DETR loss computation with Hungarian matching.
    
    Computes classification (VFL/focal/BCE) + box (L1/GIoU) losses
    for the main outputs, auxiliary decoder outputs, and denoising outputs.
    """
    def __init__(self, matcher, weight_dict, losses, alpha=0.2, gamma=2.0,
                 eos_coef=1e-4, num_classes=80):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses

        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = eos_coef
        self.register_buffer('empty_weight', empty_weight)

        self.alpha = alpha
        self.gamma = gamma

    def loss_labels_vfl(self, outputs, targets, indices, num_boxes, log=True):
        """Varifocal Loss — IoU-aware classification."""
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)

        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        if len(target_boxes) > 0:
            ious, _ = box_iou(box_cxcywh_to_xyxy(src_boxes),
                              box_cxcywh_to_xyxy(target_boxes))
            ious = torch.diag(ious).detach()
        else:
            ious = torch.zeros(0, device=src_boxes.device)

        src_logits = outputs['pred_logits']
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]

        target_score_o = torch.zeros_like(target_classes, dtype=src_logits.dtype)
        if len(ious) > 0:
            target_score_o[idx] = ious.to(target_score_o.dtype)
        target_score = target_score_o.unsqueeze(-1) * target

        pred_score = F.sigmoid(src_logits).detach()
        weight = self.alpha * pred_score.pow(self.gamma) * (1 - target) + target_score

        loss = F.binary_cross_entropy_with_logits(
            src_logits, target_score, weight=weight, reduction='none')
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {'loss_vfl': loss}

    def loss_labels_focal(self, outputs, targets, indices, num_boxes, log=True):
        """Sigmoid Focal Loss."""
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]
        loss = torchvision.ops.sigmoid_focal_loss(
            src_logits, target.float(), self.alpha, self.gamma, reduction='none')
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {'loss_focal': loss}

    def loss_labels_bce(self, outputs, targets, indices, num_boxes, log=True):
        """Binary Cross-Entropy Loss."""
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]
        loss = F.binary_cross_entropy_with_logits(
            src_logits, target.float(), reduction='none')
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {'loss_bce': loss}

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """L1 + GIoU box regression losses."""
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        if len(target_boxes) == 0:
            return {'loss_bbox': src_boxes.sum() * 0, 'loss_giou': src_boxes.sum() * 0}

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses = {'loss_bbox': loss_bbox.sum() / num_boxes}

        loss_giou = 1 - torch.diag(generalized_box_iou(
            box_cxcywh_to_xyxy(src_boxes),
            box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'boxes': self.loss_boxes,
            'vfl': self.loss_labels_vfl,
            'focal': self.loss_labels_focal,
            'bce': self.loss_labels_bce,
        }
        assert loss in loss_map, f'Unknown loss: {loss}'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """Compute all losses.
        
        Args:
            outputs: Model output dict with 'pred_logits', 'pred_boxes',
                     and optionally 'aux_outputs', 'dn_aux_outputs', 'dn_meta'.
            targets: List of target dicts with 'labels' and 'boxes'.
        
        Returns:
            Dict of all weighted losses (ready for .backward() on sum).
        """
        # Cast to float32 for numerical stability in AMP
        outputs = self._cast_to_float32(outputs)

        outputs_without_aux = {k: v for k, v in outputs.items() if 'aux' not in k and k != 'dn_meta'}

        # Match last layer outputs to targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute number of target boxes for normalization
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float,
                                    device=next(iter(outputs.values())).device)
        num_boxes = torch.clamp(num_boxes, min=1).item()

        # Compute all requested losses
        losses = {}
        for loss in self.losses:
            l_dict = self.get_loss(loss, outputs, targets, indices, num_boxes)
            l_dict = {k: l_dict[k] * self.weight_dict.get(k, 1.0)
                      for k in l_dict if k in self.weight_dict}
            losses.update(l_dict)

        # Auxiliary losses (intermediate decoder layers)
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    kwargs = {}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k: l_dict[k] * self.weight_dict.get(k, 1.0)
                              for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f'_aux_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # Denoising auxiliary losses
        if 'dn_aux_outputs' in outputs:
            assert 'dn_meta' in outputs
            indices = self.get_cdn_matched_indices(outputs['dn_meta'], targets)
            num_boxes_dn = num_boxes * outputs['dn_meta']['dn_num_group']

            for i, aux_outputs in enumerate(outputs['dn_aux_outputs']):
                for loss in self.losses:
                    kwargs = {}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes_dn, **kwargs)
                    l_dict = {k: l_dict[k] * self.weight_dict.get(k, 1.0)
                              for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f'_dn_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # Compute total loss
        total_loss = sum(v for v in losses.values() if isinstance(v, torch.Tensor))
        losses['total_loss'] = total_loss

        return losses

    def _cast_to_float32(self, outputs):
        """Cast all tensor values to float32 for AMP stability."""
        result = {}
        for k, v in outputs.items():
            if isinstance(v, torch.Tensor):
                result[k] = v.float()
            elif isinstance(v, list):
                result[k] = [{kk: vv.float() if isinstance(vv, torch.Tensor) else vv
                              for kk, vv in item.items()} if isinstance(item, dict) else item
                             for item in v]
            else:
                result[k] = v
        return result

    @staticmethod
    def get_cdn_matched_indices(dn_meta, targets):
        """Get matching indices for contrastive denoising queries."""
        dn_positive_idx = dn_meta["dn_positive_idx"]
        dn_num_group = dn_meta["dn_num_group"]
        num_gts = [len(t['labels']) for t in targets]
        device = targets[0]['labels'].device

        dn_match_indices = []
        for i, num_gt in enumerate(num_gts):
            if num_gt > 0:
                gt_idx = torch.arange(num_gt, dtype=torch.int64, device=device)
                gt_idx = gt_idx.tile(dn_num_group)
                assert len(dn_positive_idx[i]) == len(gt_idx)
                dn_match_indices.append((dn_positive_idx[i], gt_idx))
            else:
                dn_match_indices.append((
                    torch.zeros(0, dtype=torch.int64, device=device),
                    torch.zeros(0, dtype=torch.int64, device=device)))
        return dn_match_indices


# =============================================================================
# Convenience alias
# =============================================================================

def RTDETRLoss(num_classes=80, use_vfl=True):
    """Create an RTDETRLoss (SetCriterion) with standard RT-DETR settings.
    
    Args:
        num_classes: Number of object classes.
        use_vfl: If True, use Varifocal Loss; otherwise use Focal Loss.
    
    Returns:
        SetCriterion instance.
    """
    matcher_weight_dict = {'cost_class': 2, 'cost_bbox': 5, 'cost_giou': 2}
    matcher = HungarianMatcher(
        weight_dict=matcher_weight_dict,
        use_focal_loss=True,
        alpha=0.25,
        gamma=2.0,
    )

    weight_dict = {
        'loss_vfl': 1.0,
        'loss_focal': 1.0,
        'loss_bce': 1.0,
        'loss_bbox': 5.0,
        'loss_giou': 2.0,
    }

    losses = ['vfl', 'boxes'] if use_vfl else ['focal', 'boxes']

    return SetCriterion(
        matcher=matcher,
        weight_dict=weight_dict,
        losses=losses,
        alpha=0.75,
        gamma=2.0,
        num_classes=num_classes,
    )
