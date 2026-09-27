"""Albedo PSNR/SSIM in linear RGB, averaged per image on the eval canvas."""

import logging

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..data.albedo_dataset import AlbedoDataset, albedo_collate_fn, resolve_albedo_data
from .base import BaseValidator
from .restore_validator import psnr_rgb, ssim_rgb


class AlbedoValidator(BaseValidator):
    task = "albedo"

    def _setup_dataloader(self):
        if not self.config.data:
            raise ValueError("Albedo validation requires data= (a dataset YAML).")
        data = resolve_albedo_data(
            self.config.data,
            allow_scripts=getattr(self.config, "allow_download_scripts", False),
        )
        dataset = AlbedoDataset(data, self.config.split or "val", self.config.imgsz)
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            collate_fn=albedo_collate_fn,
        )

    def _init_metrics(self):
        self._psnr = []
        self._ssim = []

    def _preprocess_batch(self, batch):
        return batch

    def _postprocess_predictions(self, preds, batch):
        value = preds.get("albedo") if isinstance(preds, dict) else preds
        value = torch.as_tensor(value).float()
        if value.ndim != 4 or value.shape[1] != 3:
            raise ValueError("Albedo validation requires [B,3,H,W] linear RGB.")
        if not torch.isfinite(value).all():
            raise ValueError("Albedo prediction contains non-finite values.")
        if value.shape[0] != batch[1].shape[0]:
            raise ValueError("Albedo prediction batch does not match targets.")
        return F.interpolate(
            value, size=batch[1].shape[-2:], mode="bilinear", align_corners=False
        ).clamp(0, 1)

    def _update_metrics(self, preds, targets, img_info, img_ids=None):
        for pred, target in zip(preds.detach().cpu(), targets.detach().cpu()):
            self._psnr.append(psnr_rgb(pred, target))
            self._ssim.append(ssim_rgb(pred, target))

    def _compute_metrics(self):
        if not self._psnr:
            raise ValueError("Albedo validation found no paired images.")
        psnr = sum(self._psnr) / len(self._psnr)
        return {
            "metrics/PSNR": psnr,
            "metrics/SSIM": sum(self._ssim) / len(self._ssim),
            "fitness": psnr,
        }

    def _print_results(self, metrics):
        logging.getLogger(__name__).info("Albedo validation: %s", metrics)
