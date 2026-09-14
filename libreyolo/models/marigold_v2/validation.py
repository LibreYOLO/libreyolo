"""Use the family's float Lanczos preprocessing with the shared dense metrics."""

import torch

from ...validation.albedo_validator import AlbedoValidator
from ...validation.depth_validator import DepthValidator
from ...validation.normal_validator import NormalValidator


class _MarigoldInputs:
    def _warmup_model(self, n_warmup=1):
        original_batch = self.config.batch_size
        try:
            self.config.batch_size = 1
            return super()._warmup_model(n_warmup=1)
        finally:
            self.config.batch_size = original_batch

    def _preprocess_batch(self, batch):
        _, targets, infos, ids = batch
        images = torch.cat(
            [
                self.model._preprocess(info["img_path"], input_size=self.config.imgsz)[
                    0
                ]
                for info in infos
            ]
        )
        return images, targets, infos, ids


class MarigoldDepthValidator(_MarigoldInputs, DepthValidator):
    pass


class MarigoldNormalValidator(_MarigoldInputs, NormalValidator):
    pass


class MarigoldAlbedoValidator(_MarigoldInputs, AlbedoValidator):
    pass


VALIDATORS = {
    "depth": MarigoldDepthValidator,
    "normal": MarigoldNormalValidator,
    "albedo": MarigoldAlbedoValidator,
}
