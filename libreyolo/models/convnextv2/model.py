"""Native ConvNeXt V2 classification and official checkpoint recognition."""

from pathlib import Path
import re

from ..convnext.model import LibreConvNeXt
from .nn import ARCH_DEFS, ConvNeXtV2


class LibreConvNeXtV2(LibreConvNeXt):
    """ConvNeXt V2 atto/femto/pico/n/t/b/l/h classifiers at 224 pixels.

    Official ImageNet-1K weights are CC-BY-NC-4.0; the code is MIT.
    Supervised fine-tuning uses LibreYOLO's shared classification recipe,
    not the upstream FCMAE pretraining or ImageNet reproduction recipe.
    """

    FAMILY = "convnextv2"
    FILENAME_PREFIX = "LibreConvNeXtV2"
    INPUT_SIZES = {size: 224 for size in ARCH_DEFS}
    CROP_PCT = {size: 0.875 for size in ARCH_DEFS}
    SUPPORTED_TASKS = ("classify",)
    DEFAULT_TASK = "classify"
    REQUIRE_TASK_SUFFIX = True
    SUPPORTS_CUDA_GRAPH = False
    TRAIN_CONFIG = None

    def __init__(self, model_path=None, size="atto", nb_classes=1000, device="auto", task=None, **kwargs):
        super().__init__(model_path, size, nb_classes, device, task, **kwargs)

    @classmethod
    def can_load(cls, weights_dict):
        return (
            "downsample_layers.0.0.weight" in weights_dict
            and "head.weight" in weights_dict
            and "stages.0.0.grn.gamma" in weights_dict
            and "stages.0.0.grn.beta" in weights_dict
            and cls.detect_size(weights_dict) is not None
        )

    @classmethod
    def detect_size(cls, weights_dict):
        stem = weights_dict.get("downsample_layers.0.0.weight")
        if stem is None or stem.ndim != 4:
            return None
        indices = [set() for _ in range(4)]
        for key in weights_dict:
            match = re.fullmatch(r"stages\.([0-3])\.(\d+)\.grn\.gamma", key)
            if match:
                indices[int(match[1])].add(int(match[2]))
        for size, (depths, dims) in ARCH_DEFS.items():
            if stem.shape[0] == dims[0] and all(
                blocks == set(range(depth)) for blocks, depth in zip(indices, depths)
            ):
                return size
        return None

    @classmethod
    def detect_nb_classes(cls, weights_dict):
        head = weights_dict.get("head.weight")
        return int(head.shape[0]) if head is not None else None

    @classmethod
    def get_download_notice(cls, filename, url):
        return (
            f"{Path(filename).name}: official ConvNeXt V2 weights are licensed "
            "CC-BY-NC-4.0 (NON-COMMERCIAL). The architecture code is MIT. "
            "See the Hugging Face repository LICENSE and NOTICE."
        )

    def _init_model(self):
        return ConvNeXtV2(size=self.size, num_classes=self.nb_classes)

    def _get_available_layers(self):
        return {"stem": self.model.downsample_layers, "stages": self.model.stages,
                "head_norm": self.model.norm, "classifier": self.model.head}

    def _prepare_model_for_state_dict(self, state_dict):
        pass

    def train(self, *args, **kwargs):
        raise NotImplementedError("ConvNeXt V2 training is being integrated.")
