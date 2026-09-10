# Copyright (c) OpenMMLab. All rights reserved.
"""FCOS3D camera decoding, adapted from OpenMMLab (Apache-2.0; see NOTICE).

The CPU BEV suppression and Results conversion are LibreYOLO additions.
"""

import math

import cv2
import numpy as np
import torch

from ...utils.results import Boxes, Boxes3D

NAMES = dict(
    enumerate(
        (
            "car",
            "truck",
            "trailer",
            "bus",
            "construction_vehicle",
            "bicycle",
            "motorcycle",
            "pedestrian",
            "traffic_cone",
            "barrier",
        )
    )
)
STRIDES = (8, 16, 32, 64, 128)


def calibration(intrinsics):
    k = np.asarray(intrinsics, dtype=np.float32)
    if (
        k.shape != (3, 3)
        or not np.isfinite(k).all()
        or k[0, 0] <= 0
        or k[1, 1] <= 0
        or not np.allclose(k[2], [0, 0, 1])
        or not np.allclose([k[0, 1], k[1, 0]], 0)
    ):
        raise ValueError(
            "intrinsics must be a finite (3, 3) pinhole matrix with positive focal lengths and zero skew."
        )
    return k.copy()


def preprocess(image):
    """Native image size, BGR Caffe mean, right/bottom zero padding."""
    bgr = np.asarray(image, dtype=np.float32)[..., ::-1].copy()
    bgr -= np.array([103.530, 116.280, 123.675], dtype=np.float32)
    h, w = bgr.shape[:2]
    padded = np.zeros(((h + 31) // 32 * 32, (w + 31) // 32 * 32, 3), np.float32)
    padded[:h, :w] = bgr
    return torch.from_numpy(padded.transpose(2, 0, 1).copy()).unsqueeze(0)


def decode(raw, k, conf=0.05, iou=0.8, max_det=200):
    """Return center xyz, dimensions xyz, camera yaw, scores and class ids."""
    boxes, scores = [], []
    kt = raw[0][0].new_tensor(k)
    for stride, (cls, box, direction, _attr, center) in zip(STRIDES, zip(*raw)):
        h, w = cls.shape[-2:]
        cls = cls[0].permute(1, 2, 0).reshape(-1, 10).sigmoid()
        score = cls * center[0].reshape(-1, 1).sigmoid()
        box = box[0].permute(1, 2, 0).reshape(-1, 9).clone()
        direction = direction[0].argmax(0).reshape(-1)
        y, x = torch.meshgrid(
            torch.arange(h, device=box.device),
            torch.arange(w, device=box.device),
            indexing="ij",
        )
        points = torch.stack((x.flatten(), y.flatten()), -1) * stride + stride // 2
        if len(score) > 1000:
            idx = score.max(1).values.topk(1000).indices
            box, score, direction, points = (
                box[idx],
                score[idx],
                direction[idx],
                points[idx],
            )
        uv = points - box[:, :2]
        box[:, 0] = (uv[:, 0] - kt[0, 2]) * box[:, 2] / kt[0, 0]
        box[:, 1] = (uv[:, 1] - kt[1, 2]) * box[:, 2] / kt[1, 1]
        yaw = box[:, 6] - 0.7854
        box[:, 6] = (
            yaw
            - torch.floor(yaw / math.pi) * math.pi
            + 0.7854
            + direction * math.pi
            + torch.atan2(uv[:, 0] - kt[0, 2], kt[0, 0])
        )
        boxes.append(box[:, :7])
        scores.append(score)
    boxes = torch.cat(boxes).float().cpu().numpy()
    scores = torch.cat(scores).float().cpu().numpy()
    valid = np.isfinite(boxes).all(1) & (boxes[:, 3:6] > 0).all(1)
    boxes, scores = boxes[valid], scores[valid]
    selected, selected_scores, labels = [], [], []
    for label in range(10):
        indices = np.flatnonzero(scores[:, label] > conf)
        order = indices[np.argsort(-scores[indices, label], kind="stable")]
        rects = {
            int(i): (
                (float(boxes[i, 0]), float(boxes[i, 2])),
                (float(boxes[i, 3]), float(boxes[i, 5])),
                float(-boxes[i, 6] * 180 / math.pi),
            )
            for i in order
        }
        class_count = 0
        while len(order) and class_count < max_det:
            first = int(order[0])
            selected.append(first)
            class_count += 1
            selected_scores.append(scores[first, label])
            labels.append(label)
            rest = []
            area = float(boxes[first, 3] * boxes[first, 5])
            for other in order[1:]:
                _, polygon = cv2.rotatedRectangleIntersection(
                    rects[first], rects[int(other)]
                )
                intersection = (
                    0.0
                    if polygon is None
                    else abs(cv2.contourArea(cv2.convexHull(polygon)))
                )
                union = area + float(boxes[other, 3] * boxes[other, 5]) - intersection
                if intersection / max(union, 1e-8) <= iou:
                    rest.append(other)
            order = np.asarray(rest, dtype=np.int64)
    selected_scores = np.asarray(selected_scores, np.float32)
    order = np.argsort(-selected_scores, kind="stable")[:max_det]
    return (
        boxes[np.asarray(selected, dtype=np.int64)[order]],
        selected_scores[order],
        np.asarray(labels, dtype=np.int64)[order],
    )


def payloads(boxes, scores, labels, k, shape):
    """Map xyz dimensions/yaw to the existing wlh/quaternion cuboid contract."""
    data = np.zeros((len(boxes), 14), dtype=np.float32)
    data[:, :3] = boxes[:, :3]
    data[:, 3:6] = boxes[:, [5, 3, 4]]
    data[:, 6] = np.cos(boxes[:, 6] / 2)
    data[:, 8] = np.sin(boxes[:, 6] / 2)
    data[:, 10] = scores
    data[:, 11] = labels
    # FCOS3D has one joint score, no separately calibrated 2D/3D scores.
    data[:, 12] = scores
    data[:, 13] = 1.0
    cuboids = Boxes3D(data, shape, k)
    corners = cuboids.corners
    hulls = np.zeros((len(boxes), 6), np.float32)
    for i, points in enumerate(corners):
        near = 1e-5
        visible = [point for point in points if point[2] >= near]
        for a in range(8):
            for bit in (1, 2, 4):
                b = a ^ bit
                if a < b and (points[a, 2] >= near) != (points[b, 2] >= near):
                    t = (near - points[a, 2]) / (points[b, 2] - points[a, 2])
                    visible.append(points[a] + t * (points[b] - points[a]))
        if not visible:
            continue
        projected = np.asarray(visible) @ k.T
        xy = projected[:, :2] / projected[:, 2:3]
        hulls[i, :4] = [xy[:, 0].min(), xy[:, 1].min(), xy[:, 0].max(), xy[:, 1].max()]
    hulls[:, [0, 2]] = hulls[:, [0, 2]].clip(0, shape[1])
    hulls[:, [1, 3]] = hulls[:, [1, 3]].clip(0, shape[0])
    hulls[:, 4] = scores
    hulls[:, 5] = labels
    return Boxes(hulls[:, :4], scores, labels, orig_shape=shape), cuboids
