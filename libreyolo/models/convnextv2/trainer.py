"""Shared supervised classification recipe with ConvNeXt V2 metadata."""

from ..convnext.trainer import ConvNeXtTrainer
from .config import ConvNeXtV2Config


class ConvNeXtV2Trainer(ConvNeXtTrainer):
    """Cross-entropy fine-tuning; FCMAE and LoRA are not implemented."""

    supports_lora = False

    @classmethod
    def _config_class(cls):
        return ConvNeXtV2Config

    def get_model_family(self):
        return "convnextv2"

    def get_model_tag(self):
        return f"ConvNeXtV2-{self.config.size}"

    def on_setup(self):
        pass

    def _checkpoint_extra_metadata(self):
        return {
            **super()._checkpoint_extra_metadata(),
            **self.wrapper_model._save_extra_metadata(),
        }
