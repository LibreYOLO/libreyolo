import torch
import torch.nn as nn
import torch.nn.functional as F

from libreyolo.utils8 import make_anchors

def bbox_iou(box1, box2, eps=1e-7):
    """
    Calculate IoU between box1 (xyxy) and box2 (xyxy).
    """
    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)

    # Intersection area
    inter_x1 = torch.max(b1_x1, b2_x1)
    inter_y1 = torch.max(b1_y1, b2_y1)
    inter_x2 = torch.min(b1_x2, b2_x2)
    inter_y2 = torch.min(b1_y2, b2_y2)
    
    inter_w = (inter_x2 - inter_x1).clamp(0)
    inter_h = (inter_y2 - inter_y1).clamp(0)
    inter_area = inter_w * inter_h

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union_area = w1 * h1 + w2 * h2 - inter_area + eps

    iou = inter_area / union_area
    
    # CIoU
    cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
    ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
    c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
    rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
            (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared
            
    v = (4 / (torch.pi ** 2)) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
    with torch.no_grad():
        alpha = v / (v - iou + (1 + eps))
    return iou - (rho2 / c2 + v * alpha)


class TaskAlignedAssigner(nn.Module):
    """
    Task-aligned Assigner for object detection.
    Assigns ground truth objects to anchors based on alignment metric:
    t = s^alpha * u^beta
    where s is score, u is IoU.
    """
    def __init__(self, topk=10, num_classes=80, alpha=0.5, beta=6.0):
        super().__init__()
        self.topk = topk
        self.num_classes = num_classes
        self.bg_idx = num_classes
        self.alpha = alpha
        self.beta = beta

    @torch.no_grad()
    def forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        """
        Args:
            pd_scores (Tensor): (bs, n_anchors, num_classes)
            pd_bboxes (Tensor): (bs, n_anchors, 4) xyxy
            anc_points (Tensor): (n_anchors, 2)
            gt_labels (Tensor): (bs, n_max_boxes, 1)
            gt_bboxes (Tensor): (bs, n_max_boxes, 4)
            mask_gt (Tensor): (bs, n_max_boxes, 1) - 1 if valid box, 0 if padding
        """
        bs, n_anchors = pd_scores.shape[:2]
        n_max_boxes = gt_bboxes.shape[1]

        if n_max_boxes == 0:
            return (torch.full_like(pd_scores[..., 0], self.bg_idx).to(torch.long),
                    torch.zeros_like(pd_bboxes),
                    torch.zeros_like(pd_scores),
                    torch.zeros_like(pd_scores[..., 0]),
                    torch.zeros_like(pd_scores[..., 0]))

        # Calculate alignment metric for each anchor-GT pair
        # pd_bboxes: (bs, n_anchors, 4)
        # gt_bboxes: (bs, n_max_boxes, 4)
        
        # We need to expand to compare all anchors with all GTs
        # mask_pos: anchors that are within the GT box
        mask_pos, align_metric, overlaps = self.get_pos_mask(
            pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt
        )

        # Select top-k anchors for each GT
        target_gt_idx, fg_mask, mask_pos = self.select_topk_candidates(
            align_metric * mask_pos, topk_mask=mask_gt.repeat([1, 1, self.topk]).bool()
        )

        # Assign targets
        target_labels, target_bboxes, target_scores = self.get_targets(
            gt_labels, gt_bboxes, target_gt_idx, fg_mask, overlaps, align_metric, self.num_classes
        )

        # Normalize target scores (task alignment)
        return target_labels, target_bboxes, target_scores, fg_mask.bool(), target_gt_idx

    def get_pos_mask(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt):
        bs, n_anchors, _ = pd_scores.shape
        n_max_boxes = gt_bboxes.shape[1]
        
        # 1. Check if anchor centers are within GT boxes
        # anc_points: (n_anchors, 2) -> (1, n_anchors, 1, 2)
        # gt_bboxes: (bs, n_max_boxes, 4) -> (bs, 1, n_max_boxes, 4)
        # Returns: (bs, n_anchors, n_max_boxes)
        
        lt = anc_points[None, :, None, :] - gt_bboxes[:, None, :, :2]
        rb = gt_bboxes[:, None, :, 2:] - anc_points[None, :, None, :]
        
        # All differences must be positive
        in_gts = torch.cat([lt, rb], dim=-1).amin(dim=-1) > 0  # (bs, n_anchors, n_max_boxes)
        
        # 2. Calculate alignment metric
        # IoU between all anchors and GTs
        # We cheat a bit: calculate IoU between pred_bboxes and gt_bboxes
        # pd_bboxes: (bs, n_anchors, 4)
        # gt_bboxes: (bs, n_max_boxes, 4)
        
        # Expand for broadcasting
        # bbox_iou needs (..., 4)
        # We want matrix (bs, n_anchors, n_max_boxes)
        # Let's perform IoU calculation manually or broadcast carefully
        
        # To avoid massive memory, let's loop over batch
        overlaps = torch.zeros((bs, n_anchors, n_max_boxes), device=pd_bboxes.device)
        for i in range(bs):
            overlaps[i] = bbox_iou(pd_bboxes[i].unsqueeze(1), gt_bboxes[i].unsqueeze(0)).squeeze(-1)
            
        # Get scores corresponding to GT labels
        # gt_labels: (bs, n_max_boxes, 1)
        # pd_scores: (bs, n_anchors, num_classes)
        
        # We need to gather the score of the correct class for each GT
        # pd_scores_expanded: (bs, n_anchors, n_max_boxes)
        pd_scores_gts = torch.zeros((bs, n_anchors, n_max_boxes), device=pd_scores.device)
        for i in range(bs):
             # For each anchor, get the score of the class of the j-th GT
             # gt_labels[i] shape (n_max_boxes, 1)
             classes = gt_labels[i].long().squeeze(-1) # (n_max_boxes,)
             # pd_scores[i] shape (n_anchors, 80)
             # We want (n_anchors, n_max_boxes)
             pd_scores_gts[i] = pd_scores[i][:, classes]
             
        align_metric = pd_scores_gts.pow(self.alpha) * overlaps.pow(self.beta)
        
        # Mask out invalid GTs (padding)
        # mask_gt: (bs, n_max_boxes, 1)
        mask_gt_expanded = mask_gt[:, None, :, 0].expand(bs, n_anchors, n_max_boxes)
        
        return in_gts * mask_gt_expanded, align_metric, overlaps

    def select_topk_candidates(self, metrics, topk_mask=None):
        # metrics: (bs, n_anchors, n_max_boxes)
        bs, n_anchors, n_max_boxes = metrics.shape
        
        # For each GT, select topk anchors
        topk_metrics, topk_idxs = torch.topk(metrics, self.topk, dim=1, largest=True)
        
        # Create a mask for these topk anchors
        # (bs, n_anchors, n_max_boxes)
        is_in_topk = torch.zeros_like(metrics, dtype=torch.bool)
        for i in range(bs):
            is_in_topk[i].scatter_(0, topk_idxs[i], True)
            
        # Filter invalid candidates (metric=0)
        is_in_topk = is_in_topk & (metrics > 0)
        
        # If an anchor is assigned to multiple GTs, assign to the one with max metric
        # Max over GT dimension
        max_metric_per_anchor, max_gt_idx = metrics.max(dim=2) # (bs, n_anchors)
        
        # Create mask where anchor is assigned to the max GT
        # (bs, n_anchors, n_max_boxes)
        is_best_gt = torch.zeros_like(metrics, dtype=torch.bool)
        for i in range(bs):
            is_best_gt[i].scatter_(1, max_gt_idx[i].unsqueeze(1), True)
            
        mask_pos = is_in_topk & is_best_gt
        
        # Check if anchor is positive (assigned to at least one GT)
        fg_mask = mask_pos.any(dim=2) # (bs, n_anchors)
        
        # Get the GT index for each positive anchor
        target_gt_idx = max_gt_idx # (bs, n_anchors) -> index of GT (0..n_max-1)
        
        return target_gt_idx, fg_mask, mask_pos

    def get_targets(self, gt_labels, gt_bboxes, target_gt_idx, fg_mask, overlaps, align_metric, num_classes):
        bs, n_anchors = fg_mask.shape
        device = gt_labels.device
        
        # Batch indices
        batch_idx = torch.arange(bs, device=device, dtype=torch.long)[:, None].expand(bs, n_anchors)
        
        # Initialize targets
        target_labels = torch.full((bs, n_anchors), self.bg_idx, dtype=torch.long, device=device)
        target_bboxes = torch.zeros((bs, n_anchors, 4), device=device)
        target_scores = torch.zeros((bs, n_anchors, num_classes), device=device)
        
        # Indices of positive anchors
        # We process batch and anchor indices
        # Simplified: iterate batch
        for i in range(bs):
            mask = fg_mask[i]
            if mask.sum() == 0:
                continue
            
            gt_idx = target_gt_idx[i][mask]
            
            # Assign labels
            target_labels[i][mask] = gt_labels[i][gt_idx].long().squeeze(-1)
            
            # Assign bboxes
            target_bboxes[i][mask] = gt_bboxes[i][gt_idx]
            
            # Assign scores
            # alignment metric normalized by max alignment for that GT
            # align_metric: (bs, n_anchors, n_max_boxes)
            # We need max metric for each GT
            metrics = align_metric[i] # (n_anchors, n_max_boxes)
            max_metrics_per_gt = metrics.max(0)[0] # (n_max_boxes,)
            
            # alignment of the chosen pair
            # We need to gather (n_pos,) values
            pos_align_metrics = metrics[mask, gt_idx]
            
            # Normalize
            pos_max_metrics = max_metrics_per_gt[gt_idx]
            norm_align_metric = pos_align_metrics / (pos_max_metrics + 1e-6)
            
            # target score = norm_align_metric * 1.0 (for the class)
            classes = target_labels[i][mask]
            target_scores[i, mask, classes] = norm_align_metric
            
        return target_labels, target_bboxes, target_scores


class ComputeLoss:
    """
    Loss computation for LibreYOLO8.
    """
    def __init__(self, model, reg_max=16):
        super().__init__()
        self.reg_max = reg_max
        self.use_dfl = reg_max > 1
        self.num_classes = model.neck.c2f11.conv1.cnn.in_channels # Hacky way or pass as arg?
        # Actually better to pass config
        self.num_classes = model.head8.conv22.cnn.in_channels # No this is c_cls
        self.num_classes = model.head8.cnn2.out_channels
        
        # Loss weights
        self.box_gain = 7.5
        self.cls_gain = 0.5
        self.dfl_gain = 1.5
        
        self.assigner = TaskAlignedAssigner(topk=10, num_classes=self.num_classes, alpha=0.5, beta=6.0)
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        
        # DFL projection: 0..reg_max
        self.proj = torch.arange(reg_max, dtype=torch.float)

    def preprocess(self, targets, batch_size, scale_tensor):
        """
        Preprocess targets: [idx, cls, x, y, w, h] -> standard format
        targets: (n_targets, 6)
        """
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 5, device=targets.device)
            return out, torch.zeros(batch_size, 0, 1, device=targets.device)
            
        i = targets[:, 0]  # image index
        _, counts = i.unique(return_counts=True)
        counts = counts.int()
        out = torch.zeros(batch_size, counts.max(), 5, device=targets.device)
        mask = torch.zeros(batch_size, counts.max(), 1, device=targets.device)
        
        for j in range(batch_size):
            matches = i == j
            n = matches.sum()
            if n:
                out[j, :n] = targets[matches, 1:]
                mask[j, :n] = 1
                
        # Convert xywh to xyxy
        # targets are normalized (0-1). Scale to image size?
        # Typically YOLOv8 dataloader provides normalized xywh.
        # We need to scale them to feature map space or image space.
        # Loss is usually calculated in image space.
        out[..., 1:5] *= scale_tensor
        
        # xywh to xyxy
        xyxy = out.clone()
        xyxy[..., 1] = out[..., 1] - out[..., 3] / 2
        xyxy[..., 2] = out[..., 2] - out[..., 4] / 2
        xyxy[..., 3] = out[..., 1] + out[..., 3]
        xyxy[..., 4] = out[..., 2] + out[..., 4]
        
        return xyxy, mask

    def __call__(self, preds, batch):
        """
        preds: output of model (dict with x8, x16, x32)
        batch: dict with 'img', 'bboxes', 'cls', 'batch_idx'
        """
        # 1. Unpack predictions
        # We need to concatenate all predictions from all levels
        # preds['x8']['box'] is (B, 4, H, W) (distances) -> need to decode to xyxy
        # Actually, for Assigner we need decoded boxes.
        
        # Extract features
        x8, x16, x32 = preds['x8'], preds['x16'], preds['x32']
        device = x8['box'].device
        batch_size = x8['box'].shape[0]
        
        # Prepare anchors
        # We can reuse make_anchors from utils, but we need features list
        # We only have the dicts.
        # Construct list of features (just for shape)
        feats = [x8['box'], x16['box'], x32['box']]
        strides = [8, 16, 32]
        
        anchor_points, stride_tensor = make_anchors(feats, strides, 0.5)
        
        # Process predictions
        # Concatenate outputs
        # pred_distri: (B, total_anchors, 4*reg_max) - raw distribution
        pred_distri = torch.cat([
            x['raw_box'].flatten(2).permute(0, 2, 1) for x in [x8, x16, x32]
        ], dim=1)
        
        # pred_scores: (B, total_anchors, num_classes)
        pred_scores = torch.cat([
            x['cls'].flatten(2).permute(0, 2, 1) for x in [x8, x16, x32]
        ], dim=1)
        
        # pred_bboxes: (B, total_anchors, 4) - decoded xyxy
        # We need to decode the DISTANCES (decoded_box keys in preds) to XYXY
        # The model returns 'box' as decoded distances (l,t,r,b) * stride?
        # No, 'box' in 'x8' is result of DFL.
        # Let's check model.py again. 
        # DFL returns x.view(b, 4, h, w).
        # In utils.py, decode_boxes assumes these are distances relative to anchor.
        
        # So we concatenate them
        pred_dist_decoded = torch.cat([
            x['box'].flatten(2).permute(0, 2, 1) for x in [x8, x16, x32]
        ], dim=1) # (B, total_anchors, 4)
        
        # Now convert dists (l,t,r,b) to xyxy
        # Formula: x1 = anchor_x - l, etc.
        # But wait, does DFL return value * stride?
        # In DFL module, it does a convolution.
        # It operates on logits. It doesn't know about stride.
        # So the output is in feature map coordinates (relative to stride 1)?
        # Or does the model multiply by stride somewhere?
        # Model does NOT multiply by stride.
        # Utils.decode_boxes multiplies by stride.
        # So pred_dist_decoded are in "grid cell" units (0..reg_max).
        
        # Decode to image space
        # x1 = (anchor - dist) * stride
        # stride_tensor has shape (total_anchors, 1)
        
        # Split pred_dist_decoded
        lt = pred_dist_decoded[..., :2]
        rb = pred_dist_decoded[..., 2:]
        
        x1y1 = anchor_points - lt
        x2y2 = anchor_points + rb
        
        # This is in grid coords? No, anchor_points are in grid coords?
        # utils.make_anchors:
        # sx = torch.arange(end=w) + 0.5
        # So anchors are in feature map grid.
        
        # So x1y1 are in feature map grid.
        # To get image coords, multiply by stride.
        pred_bboxes = torch.cat([x1y1, x2y2], dim=-1) * stride_tensor
        
        # 2. Prepare Targets
        # batch['bboxes'] is (N, 4) xyxy? Or xywh?
        # Standard YOLO datasets are usually normalized xywh.
        # Assume batch comes from a loader that gives (batch_idx, cls, x, y, w, h)
        
        # If we construct the loader ourselves later, we can decide.
        # For now assume targets tensor: (N, 6) -> [img_idx, cls, x, y, w, h] normalized
        
        targets = batch.get('targets') # (N, 6)
        imgsz = torch.tensor([x8['box'].shape[3]*8, x8['box'].shape[2]*8], device=device, dtype=torch.float)
        # Scale tensor (x, y, x, y)
        scale_tensor = torch.tensor([imgsz[0], imgsz[1], imgsz[0], imgsz[1]], device=device)
        
        gt_bboxes, mask_gt = self.preprocess(targets, batch_size, scale_tensor)
        gt_labels = targets[:, 1].long() 
        # Wait, preprocess returns batch-structured GTs. 
        # Need to structure labels too.
        
        # Re-do preprocess to get labels
        # ... (integrated into preprocess or separate)
        
        # Let's fix preprocess to return labels as well
        gt_labels_out = torch.zeros(batch_size, gt_bboxes.shape[1], 1, device=device)
        
        # Simple loop again (inefficient but clear)
        for j in range(batch_size):
            matches = targets[:, 0] == j
            n = matches.sum()
            if n:
                gt_labels_out[j, :n] = targets[matches, 1:2]
                
        gt_labels = gt_labels_out
        
        
        # 3. Assign Targets
        target_labels, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            pred_bboxes.detach(),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt
        )
        
        target_scores_sum = max(target_scores.sum(), 1)
        
        # 4. Calculate Losses
        loss = torch.zeros(3, device=device) # box, cls, dfl
        
        # CLS Loss
        loss[1] = self.bce(pred_scores, target_scores).sum() / target_scores_sum
        
        if fg_mask.sum() > 0:
            # Box Loss (CIoU)
            # Only for positive anchors
            target_bboxes_pos = target_bboxes[fg_mask]
            pred_bboxes_pos = pred_bboxes[fg_mask]
            
            # IoU with weighting
            # Weight box loss by target score (alignment)?
            # Standard YOLOv8: weight = target_scores.sum(-1)
            weight = target_scores[fg_mask].sum(-1).unsqueeze(-1)
            
            iou = bbox_iou(pred_bboxes_pos, target_bboxes_pos)
            loss_box = (1.0 - iou) * weight
            loss[0] = loss_box.sum() / target_scores_sum
            
            # DFL Loss
            if self.use_dfl:
                # pred_distri: (B, A, 4*reg_max)
                pred_dist_pos = pred_distri[fg_mask].view(-1, 4, self.reg_max)
                
                # Target for DFL: dists from anchor
                # We need the target distances (l,t,r,b) in grid units
                # target_bboxes_pos is xyxy in image space
                # anchors_pos
                anchors_pos = (anchor_points * stride_tensor)[fg_mask]
                stride_pos = stride_tensor[fg_mask]
                
                # Convert target xyxy to ltrb relative to anchor
                # l = (anchor_x - x1) / stride
                # t = (anchor_y - y1) / stride
                # r = (x2 - anchor_x) / stride
                # b = (y2 - anchor_y) / stride
                
                t_l = (anchors_pos[:, 0] - target_bboxes_pos[:, 0]) / stride_pos[:, 0]
                t_t = (anchors_pos[:, 1] - target_bboxes_pos[:, 1]) / stride_pos[:, 0]
                t_r = (target_bboxes_pos[:, 2] - anchors_pos[:, 0]) / stride_pos[:, 0]
                t_b = (target_bboxes_pos[:, 3] - anchors_pos[:, 1]) / stride_pos[:, 0]
                
                t_ltrb = torch.stack([t_l, t_t, t_r, t_b], dim=1).clamp(0, self.reg_max - 0.01)
                
                # DFL Loss calculation
                # Cross entropy between predicted distribution and target value
                # Target is float (e.g. 4.5). We want to push prob at 4 and 5.
                
                left = t_ltrb.long()
                right = left + 1
                weight_right = t_ltrb - left.float()
                weight_left = 1 - weight_right
                
                # Cross Entropy
                # pred_dist_pos is logits? Or softmaxed?
                # Usually standard implementation expects logits and uses CrossEntropyLoss
                # But here we have custom weights.
                # F.cross_entropy takes logits.
                
                # loss = - (w_l * log(p_l) + w_r * log(p_r))
                
                loss_dfl = (
                    F.cross_entropy(pred_dist_pos.view(-1, self.reg_max), left.view(-1), reduction='none').view(-1, 4) * weight_left +
                    F.cross_entropy(pred_dist_pos.view(-1, self.reg_max), right.view(-1), reduction='none').view(-1, 4) * weight_right
                )
                
                loss[2] = (loss_dfl.mean(-1, keepdim=True) * weight).sum() / target_scores_sum
                
        # Scale losses
        loss[0] *= self.box_gain
        loss[1] *= self.cls_gain
        loss[2] *= self.dfl_gain
        
        return loss.sum(), loss.detach()


