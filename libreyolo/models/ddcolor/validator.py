"""Checkpoint-faithful paired validator for LibreDDColor.

The shared restore validator forwards prebuilt RGB tensors directly through a
three-channel restoration network. DDColor instead predicts two-channel Lab
chroma and needs the source image's original-resolution OpenCV Lab ``L`` plane
for reconstruction. This validator deliberately calls ``model.predict()`` on
each source path so the same exact preprocessing metadata used in public
inference reaches postprocessing, then applies LibreYOLO's canonical RGB
PSNR/SSIM functions to the restored image and paired target.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from ...validation.restore_validator import RestoreValidator, psnr_rgb, ssim_rgb


class DDColorValidator(RestoreValidator):
    """Evaluate exact public DDColor predictions on standard restore pairs."""

    def _warmup_model(self, n_warmup: int = 3) -> None:
        # A raw tensor warmup cannot exercise DDColor's source-L side input.
        # The first real predict call performs the honest full-pipeline warmup.
        del n_warmup

    def _run_validation(self) -> None:
        self.model.model.eval()
        total_start = time.time()

        for batch in self.dataloader:
            targets = batch[1]
            image_info = batch[2]
            image_ids = batch[3]
            del image_ids

            for index, info in enumerate(image_info):
                inference_start = time.time()
                result = self.model.predict(
                    info["img_path"],
                    imgsz=512,
                    device=str(self.device),
                    save=False,
                )
                self.speed["inference"] += time.time() - inference_start

                postprocess_start = time.time()
                restored = np.asarray(result.restored.array)
                if restored.ndim != 3 or restored.shape[2] != 3:
                    raise ValueError(
                        "DDColor validation expects predict() to return HWC RGB, "
                        f"got {restored.shape}."
                    )
                prediction = (
                    torch.from_numpy(np.ascontiguousarray(restored))
                    .permute(2, 0, 1)
                    .float()
                    .div_(255.0)
                )
                target_h, target_w = info["target_shape"]
                target = targets[index, :, :target_h, :target_w].detach().cpu().float()
                if tuple(prediction.shape[-2:]) != (target_h, target_w):
                    raise ValueError(
                        "DDColor prediction/target shape mismatch: "
                        f"prediction={tuple(prediction.shape[-2:])}, "
                        f"target={(target_h, target_w)}."
                    )
                self._psnr_values.append(psnr_rgb(prediction, target))
                self._ssim_values.append(ssim_rgb(prediction, target))
                self.speed["postprocess"] += time.time() - postprocess_start
                self.seen += 1

        self.speed["total"] = time.time() - total_start


__all__ = ["DDColorValidator"]
