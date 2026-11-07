"""Fine-tune InstantHMR on images annotated with 2D person keypoints.

Most people who want to adapt a body-mesh model have 2D keypoints, not 3D
meshes, so this trainer fine-tunes from a pretrained checkpoint on a standard
YOLO-pose dataset (COCO-17 keypoints, class 0 = person, the format of
``coco8-pose.yaml``). Each person becomes one 288 px crop, and three losses
apply:

- 2D: the network's 2D keypoint head against the labelled keypoints.
- Reprojection: the 3D keypoints of the decoded body, placed by the predicted
  camera translation and projected with the crop's focal length, against the
  same labels. This is what moves the mesh.
- Prior: a frozen copy of the starting model sees the same crops; the
  student's root-relative 3D keypoints, identity coefficients and MHR
  parameters are pulled towards it. 2D labels alone do not pin depth or body
  shape, and without this term the mesh drifts into implausible poses.

Only the COCO-17 keypoints are supervised (hands and feet follow the prior).
This is a best-effort recipe: it is not upstream's training procedure, which
learns from SAM 3D Body mesh annotations. Training from scratch is not
supported.
"""

from __future__ import annotations

import copy
import logging
import math
import random
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .body import mesh_checkpoint_metadata
from .utils import default_focal, preprocess_person

logger = logging.getLogger(__name__)

# MHR70 index of each COCO-17 keypoint (nose, eyes, ears, shoulders, elbows,
# wrists, hips, knees, ankles). MHR70 orders wrists after the hands.
COCO17_TO_MHR70 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 62, 41, 9, 10, 11, 12, 13, 14]
COCO17_FLIP = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]


class PersonKeypointCrops(Dataset):
    """One item per labelled person: crop, conditioning and keypoints in crop units."""

    def __init__(
        self,
        image_files,
        label_files,
        input_size: int = 288,
        augment: bool = False,
        min_visible: int = 6,
        min_box: float = 32.0,
    ):
        self.input_size = input_size
        self.augment = augment
        self.items: list[tuple[Path, np.ndarray, np.ndarray]] = []
        for image, label in zip(image_files, label_files):
            label = Path(label)
            if not label.exists():
                continue
            for row in label.read_text().splitlines():
                values = row.split()
                if len(values) < 5 + 17 * 2 or values[0] != "0":
                    continue
                nums = np.asarray(values[1:], dtype=np.float32)
                box = nums[:4]
                kpts = nums[4:]
                kpts = (
                    kpts.reshape(17, 3)
                    if kpts.size >= 51
                    else np.concatenate(
                        [kpts.reshape(17, 2), np.full((17, 1), 2.0, np.float32)], 1
                    )
                )
                if int((kpts[:, 2] > 0).sum()) < min_visible:
                    continue
                self.items.append((Path(image), box, kpts))
        self.min_box = min_box

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict:
        path, box_n, kpts_n = self.items[index]
        bgr = cv2.imread(str(path))
        if bgr is None:
            raise FileNotFoundError(path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        cx, cy, bw, bh = box_n * np.array([w, h, w, h], dtype=np.float32)
        kxy = kpts_n[:, :2] * np.array([w, h], dtype=np.float32)
        vis = (kpts_n[:, 2] > 0).astype(np.float32)

        if self.augment:
            scale = random.uniform(0.9, 1.15)
            cx += random.uniform(-0.05, 0.05) * bw
            cy += random.uniform(-0.05, 0.05) * bh
            bw, bh = bw * scale, bh * scale
            if random.random() < 0.5:
                rgb = np.ascontiguousarray(rgb[:, ::-1])
                cx = w - cx
                kxy = np.stack([w - kxy[:, 0], kxy[:, 1]], 1)[COCO17_FLIP]
                vis = vis[COCO17_FLIP]
        bw, bh = max(bw, self.min_box), max(bh, self.min_box)
        box = (cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2)
        focal = default_focal(h, w)
        crop, cliff, (x0, y0, side) = preprocess_person(
            rgb, box, focal, self.input_size
        )
        if self.augment:
            crop = crop * random.uniform(0.8, 1.2) + random.uniform(-0.2, 0.2)
        kpt = ((kxy - np.array([x0, y0], dtype=np.float32)) / side) * 2.0 - 1.0
        return {
            "image": torch.from_numpy(np.ascontiguousarray(crop, dtype=np.float32)),
            "cliff": torch.from_numpy(cliff),
            "kpts": torch.from_numpy(kpt.astype(np.float32)),
            "vis": torch.from_numpy(vis),
            # Camera of the full frame, and the crop square, for reprojection.
            "camera": torch.tensor(
                [focal, w / 2.0, h / 2.0, x0, y0, side], dtype=torch.float32
            ),
        }


def project_to_crop(joints_cam: torch.Tensor, camera: torch.Tensor) -> torch.Tensor:
    """Pinhole-project camera-space points ``(B, K, 3)`` into crop units in [-1, 1]."""
    f, cx, cy, x0, y0, side = camera.unbind(-1)
    z = joints_cam[..., 2].clamp_min(0.1)
    u = f[:, None] * joints_cam[..., 0] / z + cx[:, None]
    v = f[:, None] * joints_cam[..., 1] / z + cy[:, None]
    u = (u - x0[:, None]) / side[:, None] * 2.0 - 1.0
    v = (v - y0[:, None]) / side[:, None] * 2.0 - 1.0
    return torch.stack([u, v], -1)


def _masked_l1(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    # Capped at one crop half-width per keypoint: a mislabelled or occluded
    # person (or a point projected from behind the camera) cannot dominate.
    err = (pred - target).abs().sum(-1).clamp(max=1.0)
    err = torch.where(mask > 0, err, torch.zeros_like(err))
    return err.sum() / mask.sum().clamp_min(1.0)


class InstantHMRFineTuner:
    """Small self-contained fine-tuning loop for ``LibreInstantHMR``."""

    def __init__(
        self,
        wrapper,
        data: str,
        *,
        epochs: int = 20,
        batch: int = 16,
        lr0: float = 5e-5,
        weight_decay: float = 1e-4,
        backbone_lr_ratio: float = 0.1,
        device: str = "",
        workers: int = 4,
        seed: int = 0,
        project: str = "runs/train",
        name: str = "exp",
        exist_ok: bool = False,
        freeze_backbone: bool = False,
        freeze_bn: bool = True,
        w_2d: float = 1.0,
        w_reproj: float = 1.0,
        w_prior: float = 1.0,
        patience: int = 0,
        validate: bool = True,
    ):
        from ...data.utils import load_data_config
        from ...utils.general import increment_path

        self.wrapper = wrapper
        self.config = load_data_config(data)
        kpt_shape = self.config.get("kpt_shape")
        if kpt_shape is None or int(kpt_shape[0]) != 17:
            raise ValueError(
                "InstantHMR fine-tuning expects a YOLO-pose dataset with COCO-17 "
                f"person keypoints (kpt_shape [17, 3]); {data} has kpt_shape {kpt_shape}."
            )
        self.epochs, self.batch, self.workers = int(epochs), int(batch), int(workers)
        self.lr0, self.weight_decay = float(lr0), float(weight_decay)
        self.backbone_lr_ratio = float(backbone_lr_ratio)
        self.freeze_backbone, self.freeze_bn = bool(freeze_backbone), bool(freeze_bn)
        self.w_2d, self.w_reproj, self.w_prior = (
            float(w_2d),
            float(w_reproj),
            float(w_prior),
        )
        self.patience = int(patience)
        self.validate = bool(validate)
        self.seed = int(seed)
        self.device = (
            wrapper.device if device in (None, "", "auto") else torch.device(
                f"cuda:{device}" if str(device).isdigit() else device
            )
        )
        self.save_dir = increment_path(
            Path(project) / name, exist_ok=exist_ok, mkdir=True
        )
        (self.save_dir / "weights").mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ data
    def _loader(self, split: str, augment: bool) -> DataLoader | None:
        images = self.config.get(f"{split}_img_files")
        labels = self.config.get(f"{split}_label_files")
        if not images:
            return None
        dataset = PersonKeypointCrops(
            images, labels, self.wrapper.model.image_size, augment
        )
        if len(dataset) == 0:
            return None
        return DataLoader(
            dataset,
            batch_size=self.batch,
            shuffle=augment,
            num_workers=self.workers,
            drop_last=augment and len(dataset) > self.batch,
        )

    # ----------------------------------------------------------------- model
    def _set_train_mode(self, net: nn.Module) -> None:
        net.train()
        # Dropout off: the student then starts exactly at the frozen prior,
        # and small fine-tuning sets do not need it.
        for m in net.modules():
            if isinstance(m, nn.Dropout):
                m.eval()
        if self.freeze_bn or self.freeze_backbone:
            for m in net.backbone.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()

    def _optimizer(self, net: nn.Module) -> torch.optim.Optimizer:
        backbone, head = [], []
        for name, p in net.named_parameters():
            if name.startswith("body."):
                p.requires_grad_(False)
                continue
            if name.startswith("backbone."):
                p.requires_grad_(not self.freeze_backbone)
                if not self.freeze_backbone:
                    backbone.append(p)
            else:
                head.append(p)
        groups = [{"params": head, "lr": self.lr0}]
        if backbone:
            groups.append({"params": backbone, "lr": self.lr0 * self.backbone_lr_ratio})
        return torch.optim.AdamW(groups, weight_decay=self.weight_decay)

    def _losses(self, net, teacher, batch: dict) -> dict[str, torch.Tensor]:
        image = batch["image"].to(self.device)
        cliff = batch["cliff"].to(self.device)
        kpts = batch["kpts"].to(self.device)
        vis = batch["vis"].to(self.device)
        camera = batch["camera"].to(self.device)
        out = net(image, cliff)
        joints_3d, _ = net.body(out["mhr_params"], out["shape_params"])
        idx = torch.as_tensor(COCO17_TO_MHR70, device=self.device)

        loss_2d = _masked_l1(out["joints_2d"].index_select(1, idx), kpts, vis)
        projected = project_to_crop(
            joints_3d.index_select(1, idx) + out["cam_trans"][:, None], camera
        )
        loss_reproj = _masked_l1(projected, kpts, vis)

        with torch.no_grad():
            ref = teacher(image, cliff)
            ref_joints, _ = teacher.body(ref["mhr_params"], ref["shape_params"])
        loss_prior = (
            10.0 * (joints_3d - ref_joints).abs().sum(-1).mean()
            + 0.1 * (out["shape_params"] - ref["shape_params"]).pow(2).mean()
            + 0.1 * nn.functional.smooth_l1_loss(out["mhr_params"], ref["mhr_params"])
        )
        total = (
            self.w_2d * loss_2d
            + self.w_reproj * loss_reproj
            + self.w_prior * loss_prior
        )
        return {
            "loss": total,
            "2d": loss_2d,
            "reproj": loss_reproj,
            "prior": loss_prior,
        }

    @torch.no_grad()
    def _validate(self, net, loader) -> dict[str, float]:
        net.eval()
        totals = {"2d": 0.0, "reproj": 0.0}
        count = 0.0
        idx = torch.as_tensor(COCO17_TO_MHR70, device=self.device)
        for batch in loader:
            image = batch["image"].to(self.device)
            out = net(image, batch["cliff"].to(self.device))
            joints_3d, _ = net.body(out["mhr_params"], out["shape_params"])
            kpts, vis = batch["kpts"].to(self.device), batch["vis"].to(self.device)
            projected = project_to_crop(
                joints_3d.index_select(1, idx) + out["cam_trans"][:, None],
                batch["camera"].to(self.device),
            )
            n = vis.sum().item()
            totals["2d"] += (
                _masked_l1(out["joints_2d"].index_select(1, idx), kpts, vis).item() * n
            )
            totals["reproj"] += _masked_l1(projected, kpts, vis).item() * n
            count += n
        # Mean keypoint error as a fraction of the crop side (crop units span 2).
        return {k: v / max(count, 1.0) / 2.0 for k, v in totals.items()}

    def _save(self, net, path: Path, epoch: int, metrics: dict) -> Path:
        from ...utils.serialization import wrap_libreyolo_checkpoint

        checkpoint = wrap_libreyolo_checkpoint(
            {k: v.detach().cpu() for k, v in net.state_dict().items()},
            model_family=self.wrapper.FAMILY,
            size=self.wrapper.size,
            task="mesh",
            nc=1,
            names={0: "person"},
            imgsz=net.image_size,
        )
        checkpoint.update(epoch=epoch, metrics=metrics, cliff_focal=True)
        checkpoint.update(mesh_checkpoint_metadata(net.body))
        torch.save(checkpoint, path)
        return path

    # ------------------------------------------------------------------ loop
    def train(self) -> dict[str, Any]:
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        net = self.wrapper.model.to(self.device)
        if not net.body.is_built:
            raise RuntimeError(
                "Fine-tuning needs a loaded InstantHMR checkpoint (with its MHR body)."
            )
        teacher = copy.deepcopy(net).eval()
        for p in teacher.parameters():
            p.requires_grad_(False)

        train_loader = self._loader("train", augment=True)
        if train_loader is None:
            raise ValueError(
                "No labelled persons with enough visible keypoints in the train split."
            )
        val_loader = self._loader("val", augment=False) if self.validate else None
        optimizer = self._optimizer(net)
        steps = max(1, self.epochs * len(train_loader))
        warmup = min(100, steps // 10)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda s: (
                (s + 1) / warmup
                if s < warmup
                else 0.5
                * (1 + math.cos(math.pi * (s - warmup) / max(1, steps - warmup)))
            ),
        )

        best, best_epoch, history = math.inf, -1, []
        weights = self.save_dir / "weights"
        last_path, best_path = weights / "last.pt", weights / "best.pt"
        for epoch in range(self.epochs):
            self._set_train_mode(net)
            started, sums, n = time.time(), {}, 0
            for batch in train_loader:
                losses = self._losses(net, teacher, batch)
                optimizer.zero_grad(set_to_none=True)
                if not torch.isfinite(losses["loss"]):
                    logger.warning(
                        "InstantHMR: skipping a batch with a non-finite loss"
                    )
                    scheduler.step()
                    continue
                losses["loss"].backward()
                grad_norm = nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                if torch.isfinite(grad_norm):
                    optimizer.step()
                scheduler.step()
                for k, v in losses.items():
                    sums[k] = sums.get(k, 0.0) + float(v.detach())
                n += 1
            record = {f"train/{k}": v / max(n, 1) for k, v in sums.items()}
            if val_loader is not None:
                record.update(
                    {f"val/{k}": v for k, v in self._validate(net, val_loader).items()}
                )
            score = record.get("val/reproj", record["train/reproj"])
            record["epoch"] = epoch + 1
            history.append(record)
            logger.info(
                "InstantHMR epoch %d/%d  %s  (%.0fs)",
                epoch + 1,
                self.epochs,
                "  ".join(f"{k}={v:.4f}" for k, v in record.items() if k != "epoch"),
                time.time() - started,
            )
            self._save(net, last_path, epoch + 1, record)
            if score < best:
                best, best_epoch = score, epoch + 1
                self._save(net, best_path, epoch + 1, record)
            elif self.patience and epoch + 1 - best_epoch >= self.patience:
                logger.info(
                    "InstantHMR: no improvement for %d epochs, stopping.", self.patience
                )
                break

        net.eval()
        return {
            "save_dir": str(self.save_dir),
            "best_checkpoint": str(best_path),
            "last_checkpoint": str(last_path),
            "best_epoch": best_epoch,
            "best_fitness": best,
            "history": history,
        }
