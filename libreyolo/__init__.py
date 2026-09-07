"""Libre YOLO — open source YOLO library with MIT license."""

from importlib.metadata import version, PackageNotFoundError
from pathlib import Path as _Path

# Core API — resolved lazily through ``__getattr__`` so that ``import
# libreyolo`` does not pull torch. The ONNX inference path (backends/onnx.py +
# BaseBackend._build_result) is numpy-native, which lets a ``pip install
# --no-deps libreyolo`` deployment run inference without the torch wheel.
# See https://github.com/LibreYOLO/libreyolo/discussions/711.
#
# Laziness lives ONLY at this level. ``models/__init__.py`` stays eager and
# atomic: it builds the can_load() registry as an import side effect and its
# import ORDER is load-bearing (first match wins, so e.g. LibreYOLO9P2 must
# register before LibreYOLO9). Touching any model name here imports that whole
# module in one go, so the registry is never observed half-populated. Do not
# make the per-family imports in models/__init__.py lazy.
_MODEL_EXPORTS = (
    "LibreYOLO",
    "LibreYOLOX",
    "LibreYOLO9",
    "LibreYOLO9E2E",
    "LibreYOLO9P2",
    "LibreYOLONAS",
    "LibreDFINE",
    "LibreDOMEDETR",
    "LibreDEIM",
    "LibreDEIMv2",
    "LibreTinyFormer",
    "LibreDETR",
    "LibreDeformableDETR",
    "LibreDINODETR",
    "LibreLWDETR",
    "LibreMaskRCNN",
    "LibreFCOS",
    "LibreFasterRCNN",
    "LibreRetinaNet",
    "LibreSSD",
    "LibreCenterNet",
    "LibreEfficientDet",
    "LibreEC",
    "LibrePICODET",
    "LibrePPYOLOE",
    "LibreRTDETR",
    "LibreRTDETRv2",
    "LibreRTDETRv4",
    "LibreRTMDet",
    "LibreYOLO3",
    "LibreYOLO4",
    "LibreYOLO2",
    "LibreYOLO1",
    "LibreYOLO7",
    "LibreHRNet",
    "LibreDEKR",
    "LibreL2CS",
    "LibreFOMO",
    "LibreMiDaS",
    "LibreDepthAnythingV2",
    "LibreZipDepth",
    "LibreMoGe2",
    "LibreTEED",
    "LibreDexiNed",
    "LibreDepthAnything3",
    "LibreNAFNet",
    "LibreDDColor",
    "LibreHVICIDNet",
    "LibreLaMa",
    "LibreBiRefNet",
    "LibreFeyNobg",
    "LibreBEN2",
    "LibreViTMatte",
    "LibreRealESRGAN",
    "LibreQuickSRNet",
    "LibreSwinIR",
    "LibreFCN",
    "LibreEoMT",
    "LibreDeepLabv3",
    "LibrePIDNet",
    "LibrePPLiteSeg",
    "LibreUNet",
    "LibreSegformer",
    "LibreLingBotVision",
    "LibreViT",
    "LibreMobileNetV4",
    "LibreConvNeXt",
    "LibreDeiT",
    "LibreSwin",
    "LibreEfficientNetV2",
    "LibreVGG",
    "LibreResNet",
    "LibreAlexNet",
    "LibreCLIP",
    "LibreSigLIP2",
    "LibrePE",
    "LibrePPOCR",
    "LibreVJEPA2",
)
_RESULTS_EXPORTS = (
    "Results",
    "Boxes",
    "Masks",
    "Keypoints",
    "Points",
    "Probs",
    "OBB",
    "Gaze",
    "SemanticMask",
    "PanopticSegmentation",
    "DepthMap",
    "EdgeMap",
    "NormalMap",
    "RestoredImage",
    "Matte",
    "Meshes",
    "OCRRegions",
    "Embeddings",
    "Identities",
)

SAMPLE_IMAGE = str(_Path(__file__).parent / "assets" / "parkour.jpg")

try:
    __version__ = version("libreyolo")
except PackageNotFoundError:
    __version__ = "0.0.0.dev0"


# Old class names that were renamed for nomenclature consistency. Resolved
# via __getattr__ with a DeprecationWarning so existing imports keep working.
_DEPRECATED_ALIASES = {
    "LibreYOLORTDETR": "LibreRTDETR",
    "LibreYOLORFDETR": "LibreRFDETR",
}


# Lazy imports for optional/heavy modules
def __getattr__(name):
    if name in _DEPRECATED_ALIASES:
        new_name = _DEPRECATED_ALIASES[name]
        import sys
        import warnings

        warnings.warn(
            f"{name} has been renamed to {new_name}. Update your imports — "
            "the old name will be removed in a future release.",
            DeprecationWarning,
            stacklevel=2,
        )
        # ``getattr`` on the module object resolves both eager imports
        # (``LibreRTDETR`` in globals) and the lazy ``__getattr__`` path
        # (``LibreRFDETR``); recursing into ``__getattr__`` directly would
        # skip the eager case.
        return getattr(sys.modules[__name__], new_name)

    # Core model / results exports. Resolving any of these imports the whole
    # ``models`` package in one atomic step, preserving can_load() registry
    # order (see the _MODEL_EXPORTS comment above).
    if name in _MODEL_EXPORTS or name in _RESULTS_EXPORTS:
        import importlib

        module_path = ".models" if name in _MODEL_EXPORTS else ".utils.results"
        attr = getattr(importlib.import_module(module_path, package=__name__), name)
        # Cache into module globals so repeat access skips __getattr__ entirely.
        globals()[name] = attr
        return attr

    _lazy = {
        "LibreRFDETR": (".models.rfdetr.model", "LibreRFDETR"),
        "LibreDINOv2": (".models.dinov2.model", "LibreDINOv2"),
        "LibreEnsemble": (".ensemble", "LibreEnsemble"),
        "ExternalDetector": (".ensemble", "ExternalDetector"),
        "OnnxBackend": (".backends.onnx", "OnnxBackend"),
        "OpenVINOBackend": (".backends.openvino", "OpenVINOBackend"),
        "PaddleBackend": (".backends.paddle", "PaddleBackend"),
        "TensorRTBackend": (".backends.tensorrt", "TensorRTBackend"),
        "TritonBackend": (".backends.triton", "TritonBackend"),
        "create_triton_config": (".backends.triton", "create_triton_config"),
        "NcnnBackend": (".backends.ncnn", "NcnnBackend"),
        "CoreMLBackend": (".backends.coreml", "CoreMLBackend"),
        "BaseExporter": (".export", "BaseExporter"),
        "DetectionValidator": (".validation", "DetectionValidator"),
        "SegmentationValidator": (".validation", "SegmentationValidator"),
        "PoseValidator": (".validation", "PoseValidator"),
        "SemanticValidator": (".validation", "SemanticValidator"),
        "PanopticValidator": (".validation", "PanopticValidator"),
        "DepthValidator": (".validation", "DepthValidator"),
        "NormalValidator": (".validation", "NormalValidator"),
        "EdgeValidator": (".validation", "EdgeValidator"),
        "ValidationConfig": (".validation", "ValidationConfig"),
        "ByteTracker": (".tracking", "ByteTracker"),
        "BoTSortTracker": (".tracking", "BoTSortTracker"),
        "BoTSortConfig": (".tracking", "BoTSortConfig"),
        "TrackConfig": (".tracking", "TrackConfig"),
        "OCSortTracker": (".tracking", "OCSortTracker"),
        "OCSortConfig": (".tracking", "OCSortConfig"),
        "LibreLLM": (".models.llm", "LibreLLM"),
        "LibreVLM": (".models.vlm", "LibreVLM"),
        "LibreLFM2VL": (".models.vlm", "LibreLFM2VL"),
        "LibreQwen3VL": (".models.vlm", "LibreQwen3VL"),
        "LibreSmolVLM2": (".models.vlm", "LibreSmolVLM2"),
        "LibreInternVL3": (".models.vlm", "LibreInternVL3"),
        "LibreFlorence2": (".models.vlm", "LibreFlorence2"),
        "LibreKosmos2": (".models.vlm", "LibreKosmos2"),
        "LibreLocateAnything": (".models.vlm", "LibreLocateAnything"),
        "LibreGround": (".models.ground", "LibreGround"),
        "LibreShowUI": (".models.ground", "LibreShowUI"),
        "LibreGroundFlorence2": (".models.ground", "LibreGroundFlorence2"),
        "LibreGroundQwen3VL": (".models.ground", "LibreGroundQwen3VL"),
        "LibreGemma4": (".models.vlm", "LibreGemma4"),
        "LibreMoondream": (".models.vlm", "LibreMoondream"),
        "LibreNorthMicroVision": (".models.vlm", "LibreNorthMicroVision"),
        "LibreSenseNovaVision": (".models.sensenova", "LibreSenseNovaVision"),
        "LibreMODUS": (".models.modus", "LibreMODUS"),
        "LibreModus": (".models.modus", "LibreModus"),
        "LibreSAM": (".models.sam", "LibreSAM"),
        "LibreSAM1": (".models.sam", "LibreSAM1"),
        "LibreSAM2": (".models.sam", "LibreSAM2"),
        "LibreEdgeTAM": (".models.sam", "LibreEdgeTAM"),
        "LibreSAM3": (".models.sam", "LibreSAM3"),
        "LibreMobileSAM": (".models.mobilesam", "LibreMobileSAM"),
        "LibrePicoSAM3": (".models.picosam3", "LibrePicoSAM3"),
        "LibreOpenVocab": (".models.openvocab", "LibreOpenVocab"),
        "LibreGroundingDINO": (".models.openvocab", "LibreGroundingDINO"),
        "LibreOWLv2": (".models.openvocab", "LibreOWLv2"),
        "LibreOMDetTurbo": (".models.openvocab", "LibreOMDetTurbo"),
        "DATASETS_DIR": (".data", "DATASETS_DIR"),
        "load_data_config": (".data", "load_data_config"),
        "check_dataset": (".data", "check_dataset"),
        "Distiller": (".distillation", "Distiller"),
        "get_distill_config": (".distillation", "get_distill_config"),
        "LibreFaceEmbedder": (".models.facerec", "LibreFaceEmbedder"),
        "Gallery": (".utils.gallery", "Gallery"),
        "FaceGallery": (".utils.gallery", "FaceGallery"),
    }
    if name in ("LibreRFDETR", "LibreDINOv2"):
        # RF-DETR and DINOv2 share the same transformers dependency check.
        from .models import _ensure_rfdetr

        _ensure_rfdetr()
    if name in _lazy:
        import importlib

        module_path, attr = _lazy[name]
        mod = importlib.import_module(module_path, package=__name__)
        return getattr(mod, attr)
    raise AttributeError(f"module 'libreyolo' has no attribute '{name}'")


def __dir__():
    # Model/results names resolve lazily, so they are absent from globals()
    # until first use. Advertise them anyway to keep tab-completion and
    # introspection behaving as they did when the imports were eager.
    return sorted(set(globals()) | set(__all__))


__all__ = [
    # Main API
    "LibreYOLO",
    "LibreYOLO9",
    "LibreYOLO9E2E",
    "LibreYOLO9P2",
    "LibreYOLONAS",
    "LibreYOLOX",
    "LibreRTDETR",
    "LibreRTDETRv2",
    "LibreRTDETRv4",
    "LibreRFDETR",
    "LibreDFINE",
    "LibreDOMEDETR",
    "LibreDEIM",
    "LibreDEIMv2",
    "LibreTinyFormer",
    "LibreDETR",
    "LibreDeformableDETR",
    "LibreDINODETR",
    "LibreLWDETR",
    "LibreMaskRCNN",
    "LibreFCOS",
    "LibreFasterRCNN",
    "LibreRetinaNet",
    "LibreSSD",
    "LibreCenterNet",
    "LibreEfficientDet",
    "LibreEC",
    "LibrePICODET",
    "LibrePPYOLOE",
    "LibreRTMDet",
    "LibreYOLO3",
    "LibreYOLO4",
    "LibreYOLO2",
    "LibreYOLO1",
    "LibreYOLO7",
    "LibreHRNet",
    "LibreDEKR",
    "LibreL2CS",
    "LibreFOMO",
    "LibreMiDaS",
    "LibreDepthAnythingV2",
    "LibreZipDepth",
    "LibreMoGe2",
    "LibreTEED",
    "LibreDexiNed",
    "LibreDepthAnything3",
    "LibreNAFNet",
    "LibreDDColor",
    "LibreHVICIDNet",
    "LibreLaMa",
    "LibreBiRefNet",
    "LibreFeyNobg",
    "LibreBEN2",
    "LibreViTMatte",
    "LibreRealESRGAN",
    "LibreQuickSRNet",
    "LibreSwinIR",
    "LibreFCN",
    "LibreEoMT",
    "LibreDeepLabv3",
    "LibrePIDNet",
    "LibrePPLiteSeg",
    "LibreUNet",
    "LibreSegformer",
    "LibreLingBotVision",
    "LibreViT",
    "LibreMobileNetV4",
    "LibreConvNeXt",
    "LibreDeiT",
    "LibreSwin",
    "LibreEfficientNetV2",
    "LibreVGG",
    "LibreResNet",
    "LibreAlexNet",
    "LibreCLIP",
    "LibreSigLIP2",
    "LibrePE",
    "LibrePPOCR",
    "LibreVJEPA2",
    "LibreDINOv2",
    # VLM-as-detector tier (optional, requires libreyolo[vlm])
    "LibreVLM",
    "LibreLFM2VL",
    "LibreQwen3VL",
    "LibreSmolVLM2",
    "LibreInternVL3",
    "LibreFlorence2",
    "LibreKosmos2",
    "LibreLocateAnything",
    # Grounding tier (optional, requires libreyolo[vlm] or libreyolo[ground])
    "LibreGround",
    "LibreShowUI",
    "LibreGroundFlorence2",
    "LibreGroundQwen3VL",
    "LibreGemma4",
    "LibreMoondream",
    "LibreNorthMicroVision",
    "LibreMODUS",
    "LibreModus",
    # OpenAI-compatible LLM client (optional, requires libreyolo[llm])
    "LibreLLM",
    # Promptable-segmentation tier (optional, requires libreyolo[sam])
    "LibreSAM",
    "LibreSAM1",
    "LibreSAM2",
    "LibreEdgeTAM",
    "LibreSAM3",
    "LibreMobileSAM",
    "LibrePicoSAM3",
    # Open-vocabulary detector tier (optional, requires libreyolo[openvocab])
    "LibreOpenVocab",
    "LibreGroundingDINO",
    "LibreOWLv2",
    "LibreOMDetTurbo",
    # Results
    "Results",
    "Boxes",
    "Masks",
    "Keypoints",
    "Points",
    "Probs",
    "OBB",
    "Gaze",
    "SemanticMask",
    "PanopticSegmentation",
    "DepthMap",
    "EdgeMap",
    "NormalMap",
    "RestoredImage",
    "Matte",
    "Meshes",
    "OCRRegions",
    "Embeddings",
    "Identities",
    "Gallery",
    "FaceGallery",
    "LibreFaceEmbedder",
    # Assets
    "SAMPLE_IMAGE",
    # Tracking
    "ByteTracker",
    "BoTSortTracker",
    "BoTSortConfig",
    "TrackConfig",
    "OCSortTracker",
    "OCSortConfig",
    # Ensembling
    "LibreEnsemble",
    "ExternalDetector",
    # Lazy-loaded
    "OnnxBackend",
    "OpenVINOBackend",
    "PaddleBackend",
    "TensorRTBackend",
    "TritonBackend",
    "create_triton_config",
    "NcnnBackend",
    "CoreMLBackend",
    "BaseExporter",
    "DetectionValidator",
    "SegmentationValidator",
    "PoseValidator",
    "SemanticValidator",
    "PanopticValidator",
    "DepthValidator",
    "NormalValidator",
    "EdgeValidator",
    "ValidationConfig",
    "DATASETS_DIR",
    "load_data_config",
    "check_dataset",
    # Distillation
    "Distiller",
    "get_distill_config",
]
