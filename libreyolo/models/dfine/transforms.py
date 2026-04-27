"""D-FINE training transforms (paper-faithful augmentations + multi-scale collate).

Built on torchvision.transforms.v2 to mirror upstream's augmentation pipeline
exactly. Compared with v1 (hflip + resize only), this adds:

- ``RandomPhotometricDistort`` (p=0.5)
- ``RandomZoomOut``  (fill=0)
- ``RandomIoUCrop``  (p=0.8)
- ``SanitizeBoundingBoxes`` (min_size=1)  — twice in the chain, once after
  the strong ops and once after Resize
- Multi-scale ``DFINEMultiScaleCollate`` collate that randomly resizes each
  batch over a window around 640 (matches upstream's
  ``BatchImageCollateFunction``).

The strong ops (Distort, ZoomOut, IoUCrop) are disabled at ``stop_epoch``
matching upstream's ``policy: name=stop_epoch`` mechanic; HFlip + Resize stay
on.

Output contract is unchanged: ``(image_chw_float01_rgb, padded_labels (max_labels, 5))``
with ``[class, cx, cy, w, h]`` in PIXEL coordinates on the resized image. The
trainer's ``on_forward`` keeps doing the pixel→normalized translation.
"""

from __future__ import annotations

import random
from typing import List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import tv_tensors
from torchvision.transforms import v2 as tv2


def _labels_at_index_2(inputs):
    """Module-level labels_getter for ``SanitizeBoundingBoxes``.

    Defined at module scope (rather than as a lambda) so the transform is
    picklable for ``DataLoader(num_workers > 0)`` under Python 3.14's
    ``forkserver`` multiprocessing default.
    """
    return inputs[2]


def _generate_scales(base_size: int, base_size_repeat: int) -> List[int]:
    """Mirror of ``D-FINE/src/data/dataloader.py::generate_scales``.

    Produces a list like ``[base*0.75, ..., base, base, base, ..., base*1.25]``
    (multiples of 32) where ``base`` is repeated ``base_size_repeat`` times.
    """
    scale_repeat = (base_size - int(base_size * 0.75 / 32) * 32) // 32
    scales = [int(base_size * 0.75 / 32) * 32 + i * 32 for i in range(scale_repeat)]
    scales += [base_size] * base_size_repeat
    scales += [int(base_size * 1.25 / 32) * 32 - i * 32 for i in range(scale_repeat)]
    return scales


class DFINETrainTransform:
    """Per-sample transform with stop_epoch-aware strong-aug toggling.

    Args:
        max_labels: Padding length for the output target tensor.
        flip_prob: Horizontal flip probability.
        imgsz: Final resize target.
        zoomout_fill: Pixel fill for RandomZoomOut.
        photometric_p: Probability for RandomPhotometricDistort.
        iou_crop_p: Probability for RandomIoUCrop.
        strong_augs: Initial state. The trainer flips this off via
            ``DFINEPassThroughDataset.set_epoch`` once epoch >= stop_epoch.
    """

    STRONG_OP_NAMES = ("RandomPhotometricDistort", "RandomZoomOut", "RandomIoUCrop")

    def __init__(
        self,
        max_labels: int = 120,
        flip_prob: float = 0.5,
        imgsz: int = 640,
        zoomout_fill: int = 0,
        photometric_p: float = 0.5,
        iou_crop_p: float = 0.8,
        strong_augs: bool = True,
    ):
        self.max_labels = max_labels
        self.imgsz = imgsz
        self.flip_prob = flip_prob
        self.strong_augs = strong_augs

        # Strong (early-training) ops — disabled at stop_epoch.
        # ``RandomIoUCrop`` has no built-in ``p``; wrap with ``RandomApply`` to
        # match upstream's manual ``if torch.rand(1) >= self.p: return inputs``
        # subclass.
        self._strong = tv2.Compose(
            [
                tv2.RandomPhotometricDistort(p=photometric_p),
                tv2.RandomZoomOut(fill=zoomout_fill),
                tv2.RandomApply([tv2.RandomIoUCrop()], p=iou_crop_p),
                tv2.SanitizeBoundingBoxes(min_size=1, labels_getter=_labels_at_index_2),
            ]
        )
        # Weak (always-on) ops.
        self._weak = tv2.Compose(
            [
                tv2.RandomHorizontalFlip(p=flip_prob),
                tv2.Resize(size=(imgsz, imgsz), antialias=True),
                tv2.SanitizeBoundingBoxes(min_size=1, labels_getter=_labels_at_index_2),
            ]
        )

    def disable_strong_augs(self):
        self.strong_augs = False

    def __call__(self, image: np.ndarray, targets: np.ndarray, input_dim):
        """Args follow LibreYOLO's existing per-sample transform contract.

        Args:
            image: HWC uint8 BGR (cv2 convention).
            targets: ``(N, 5)`` ``[x1, y1, x2, y2, class]`` pixels on the orig image.
            input_dim: ``(H, W)`` target size — kept for API compatibility but
                ``imgsz`` from ``__init__`` controls the actual resize.
        """
        target_h, target_w = input_dim
        orig_h, orig_w = image.shape[:2]

        # BGR → RGB, HWC → CHW uint8 tensor.
        img_rgb = image[:, :, ::-1].copy()
        img_t = tv_tensors.Image(
            torch.from_numpy(np.ascontiguousarray(img_rgb)).permute(2, 0, 1)
        )

        if len(targets):
            boxes_xyxy = targets[:, :4].astype(np.float32, copy=True)
            labels_np = targets[:, 4].astype(np.int64, copy=True)
        else:
            boxes_xyxy = np.zeros((0, 4), dtype=np.float32)
            labels_np = np.zeros((0,), dtype=np.int64)

        boxes = tv_tensors.BoundingBoxes(
            torch.from_numpy(boxes_xyxy),
            format=tv_tensors.BoundingBoxFormat.XYXY,
            canvas_size=(orig_h, orig_w),
        )
        labels = torch.from_numpy(labels_np)

        # Apply ops. v2 ops accept (image, boxes, labels) and update them in lockstep.
        if self.strong_augs and len(boxes) > 0:
            img_t, boxes, labels = self._strong(img_t, boxes, labels)
        img_t, boxes, labels = self._weak(img_t, boxes, labels)

        # Tensor → numpy CHW float32 [0, 1] RGB.
        img_out = img_t.float().div_(255.0).numpy()

        # Boxes back to (N, 4) numpy xyxy in pixel coords on the resized canvas.
        boxes_arr = boxes.detach().numpy().astype(np.float32, copy=True)
        labels_arr = labels.detach().numpy().astype(np.float32, copy=True)

        # xyxy → cxcywh, drop tiny boxes.
        if len(boxes_arr):
            cx = (boxes_arr[:, 0] + boxes_arr[:, 2]) * 0.5
            cy = (boxes_arr[:, 1] + boxes_arr[:, 3]) * 0.5
            w = boxes_arr[:, 2] - boxes_arr[:, 0]
            h = boxes_arr[:, 3] - boxes_arr[:, 1]
            valid = (w > 1) & (h > 1)
            cx, cy, w, h = cx[valid], cy[valid], w[valid], h[valid]
            labels_arr = labels_arr[valid]
            packed = np.stack([labels_arr, cx, cy, w, h], axis=1)
        else:
            packed = np.zeros((0, 5), dtype=np.float32)

        padded = np.zeros((self.max_labels, 5), dtype=np.float32)
        if len(packed):
            n = min(len(packed), self.max_labels)
            padded[:n] = packed[:n]
        return img_out, padded


class DFINEPassThroughDataset:
    """Identity wrapper that runs the train transform per item — no mosaic.

    Constructor signature matches ``BaseTrainer._setup_data``'s
    ``MosaicDatasetClass(...)`` contract. Ignores all mosaic/mixup kwargs.
    """

    def __init__(
        self,
        dataset,
        img_size,
        mosaic=True,
        preproc=None,
        degrees=0.0,
        translate=0.0,
        mosaic_scale=(1.0, 1.0),
        mixup_scale=(1.0, 1.0),
        shear=0.0,
        enable_mixup=False,
        mosaic_prob=0.0,
        mixup_prob=0.0,
    ):
        del mosaic, degrees, translate, mosaic_scale, mixup_scale, shear
        del enable_mixup, mosaic_prob, mixup_prob
        self.dataset = dataset
        self.img_size = img_size
        self.preproc = preproc or DFINETrainTransform()
        self._stop_epoch: Optional[int] = None
        self._epoch = 0

    def __len__(self):
        return len(self.dataset)

    @property
    def input_dim(self):
        return self.img_size

    def set_stop_epoch(self, stop_epoch: int):
        self._stop_epoch = stop_epoch

    def set_epoch(self, epoch: int):
        """Trainer calls this at the start of every epoch.

        When ``epoch >= stop_epoch`` we permanently disable the strong augs
        (matches upstream's ``policy: name=stop_epoch`` semantics).
        """
        self._epoch = epoch
        if self._stop_epoch is not None and epoch >= self._stop_epoch:
            if (
                isinstance(self.preproc, DFINETrainTransform)
                and self.preproc.strong_augs
            ):
                self.preproc.disable_strong_augs()

    def close_mosaic(self):
        # Compatibility shim — mosaic is never enabled here, but BaseTrainer
        # calls this at no_aug_epochs and our hook works fine.
        if isinstance(self.preproc, DFINETrainTransform):
            self.preproc.disable_strong_augs()

    def __getitem__(self, idx):
        img, label, img_info, img_id = self.dataset.pull_item(idx)
        img, label = self.preproc(img, label, self.input_dim)
        return img, label, img_info, img_id


class DFINEMultiScaleCollate:
    """Per-batch random resize, mirroring upstream ``BatchImageCollateFunction``.

    Until ``stop_epoch``, each batch is randomly resized to one of the
    pre-computed scales (multiples of 32 within ±25% of ``base_size``). After
    ``stop_epoch``, batches stay at ``base_size``.
    """

    def __init__(
        self,
        base_size: int = 640,
        base_size_repeat: int = 3,
        stop_epoch: Optional[int] = None,
    ):
        self.base_size = base_size
        self.scales = (
            _generate_scales(base_size, base_size_repeat) if base_size_repeat else None
        )
        self._stop_epoch = stop_epoch if stop_epoch is not None else 10**9
        self._epoch = 0

    def set_epoch(self, epoch: int):
        self._epoch = epoch

    def __call__(self, batch: Sequence):
        # Each item: (img_chw, padded_labels, info, id) — same as YOLOX collate produces.
        imgs = torch.stack(
            [torch.from_numpy(np.ascontiguousarray(item[0])) for item in batch]
        )
        labels = torch.stack(
            [torch.from_numpy(np.ascontiguousarray(item[1])) for item in batch]
        )
        infos = [item[2] for item in batch]
        ids = torch.tensor([item[3] for item in batch])

        if self.scales is not None and self._epoch < self._stop_epoch:
            sz = random.choice(self.scales)
            if sz != imgs.shape[-1]:
                # Rescale targets too — they are pixel cxcywh on the original
                # canvas (which the per-sample transform already resized to base_size).
                ratio = sz / imgs.shape[-1]
                imgs = F.interpolate(
                    imgs, size=sz, mode="bilinear", align_corners=False
                )
                # Scale (cx, cy, w, h) — column 0 is class.
                labels[..., 1:] = labels[..., 1:] * ratio

        return imgs, labels, infos, ids
