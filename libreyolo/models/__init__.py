"""
LibreYOLO model registry and unified factory.

All model families register here via ``__init_subclass__``. Adding a new model means:
1. Create models/<family>/ with model.py defining a class that inherits BaseModel
2. Add classmethods: can_load, detect_size, detect_nb_classes, detect_size_from_filename
3. Import the class so that ``__init_subclass__`` adds it to ``BaseModel._registry``
"""

from __future__ import annotations

import logging
from pathlib import Path

from .base import BaseModel
from ..tasks import resolve_task
from ..utils.download import download_weights
from ..utils.logging import ensure_default_logging
from ..utils.serialization import (
    REQUIRED_CHECKPOINT_METADATA_KEYS,
    validate_checkpoint_metadata,
    load_untrusted_torch_file,
)

logger = logging.getLogger(__name__)

_METADATA_CONVERSION_HELP = (
    "LibreYOLO checkpoints must include metadata keys: "
    f"{', '.join(REQUIRED_CHECKPOINT_METADATA_KEYS)}. "
    "Convert upstream weights with the appropriate weights/convert_*.py script "
    "for that model family, or inspect the file with `libreyolo metadata path=...`."
)

# =============================================================================
# Model registry — auto-populated by BaseModel.__init_subclass__
# Order depends on import order: first match wins in can_load()
# =============================================================================

# Always-available models (importing triggers __init_subclass__ registration)
# Order matters: more-specific can_load() checks must run first. EC's ViT
# backbone keys ("backbone.backbone.register_token") are uniquely identifying,
# so register it before YOLOX which matches the broader "backbone.backbone"
# prefix (skill landmine §9.3).
# NOTE: LibreYOLO9E2E *must* be imported before LibreYOLO9.  E2E checkpoints
# contain all the same backbone/neck key patterns that LibreYOLO9.can_load
# matches, so the E2E discriminator (one2one_cv2 / one2one_cv3) must win first.
from .ec.model import LibreEC  # noqa: E402
from .yolox.model import LibreYOLOX  # noqa: E402
from .yolo9_e2e.model import LibreYOLO9E2E  # noqa: E402
from .yolo9_p2.model import LibreYOLO9P2  # noqa: E402  (must precede LibreYOLO9: P2 checkpoints also match the base backbone/neck patterns)
from .yolo9.model import LibreYOLO9  # noqa: E402
from .yolonas.model import LibreYOLONAS  # noqa: E402
from .tinyformer.model import LibreTinyFormer  # noqa: E402  (DEIMv2 derivative; routing is enforced by can_load on backbone.sda. in both directions, not by this line — importing it pulls in models.deimv2 first, so it registers *after* LibreDEIMv2 regardless of position here)
from .deimv2.model import LibreDEIMv2  # noqa: E402
from .rtdetrv4.model import LibreRTDETRv4  # noqa: E402  (must precede LibreDFINE — sibling arch, more-specific can_load)
from .domedetr.model import LibreDOMEDETR  # noqa: E402  (D-FINE derivative; routing is enforced by can_load on encoder.DeFE. in both directions, not by this line — importing it pulls in models.dfine first, so it registers *after* LibreDFINE regardless of position here)
from .dfine.model import LibreDFINE  # noqa: E402
from .deim.model import LibreDEIM  # noqa: E402

# Vanilla DETR uses a unique top-level query embedding plus packed PyTorch
# cross-attention weights. Register it before descendants with broader DETR
# vocabulary checks (notably the optional RF-DETR family).
from .detr.model import LibreDETR  # noqa: E402

# Original Deformable DETR is a core, dependency-free family. Register its
# precise ResNet/deformable-attention fingerprint before descendants whose lazy
# discriminators intentionally accept broad transformer key patterns.
from .deformable_detr.model import LibreDeformableDETR  # noqa: E402

# DINO-DETR has a strict 900-query + denoising-label signature. Register it
# beside its Deformable DETR ancestor and before broader descendant checks.
from .dinodetr.model import LibreDINODETR  # noqa: E402

# LW-DETR is RF-DETR's ancestor and shares its decoder/projector key names, so
# it registers eagerly and ahead of the lazy RF-DETR import; its plain-ViT
# encoder keys (patch_embed.proj + CAE q_bias) are the discriminator.
from .lwdetr.model import LibreLWDETR  # noqa: E402
# Mask R-CNN shares the Faster R-CNN box graph, so its distinctive mask-head
# discriminator must register first.
from .mask_rcnn.model import LibreMaskRCNN  # noqa: E402
from .fcos.model import LibreFCOS  # noqa: E402  (unique centerness + P6/P7 fingerprint)
from .faster_rcnn.model import LibreFasterRCNN  # noqa: E402
from .retinanet.model import LibreRetinaNet  # noqa: E402
from .ssd.model import LibreSSD  # noqa: E402  (VGG extras + paired MultiBox heads are unique)
from .centernet.model import LibreCenterNet  # noqa: E402
from .efficientdet.model import LibreEfficientDet  # noqa: E402  (BiFPN keys are unique; inference-only)
from .picodet.model import LibrePICODET  # noqa: E402
from .ppyoloe.model import LibrePPYOLOE  # noqa: E402  (head.pred_cls/pred_reg + CSPResNet stem keys are unique)
from .rtdetr.model import LibreRTDETR  # noqa: E402  (registered before LibreRTDETRv2 so metadata-less ckpts default to v1)
from .rtdetrv2.model import LibreRTDETRv2  # noqa: E402
from .rtmdet.model import LibreRTMDet  # noqa: E402

# Darknet-lineage detectors (public domain). Each keys can_load on a unique
# family prefix (yolo1./yolo2./yolo3./yolo4.) so registration order is not sensitive.
from .yolo3.model import LibreYOLO3  # noqa: E402
from .yolo4.model import LibreYOLO4  # noqa: E402
from .yolo2.model import LibreYOLO2  # noqa: E402
from .yolo1.model import LibreYOLO1  # noqa: E402  (VOC museum; can_load keyed on unique yolo1. FC head)
from .yolo7.model import LibreYOLO7  # noqa: E402  (can_load keyed on unique implicit_a.implicit)
from .hrnet.model import LibreHRNet  # noqa: E402,F401  (top-down pose; unique stage-fusion fingerprint)
from .dekr.model import LibreDEKR  # noqa: E402,F401  (bottom-up pose; unique offset_final_layer stack)
from .l2cs.model import LibreL2CS  # noqa: E402,F401  (import registers family)
from .fomo.model import LibreFOMO  # noqa: E402,F401  (import registers family)
from .midas.model import LibreMiDaS  # noqa: E402,F401  (depth-only MiDaS museum family)
from .depth_anything.model import (  # noqa: E402,F401  (import registers family)
    LibreDepthAnythingV2,
)
from .zipdepth.model import LibreZipDepth  # noqa: E402,F401  (depth-only; can_load keyed on encoder.stem_half + decoder.convex_up)
from .moge2.model import LibreMoGe2  # noqa: E402,F401  (normal-only; official Microsoft MIT checkpoint)
from .teed.model import LibreTEED  # noqa: E402,F401  (edge-only; MIT source)
from .dexined.model import LibreDexiNed  # noqa: E402,F401  (edge-only; MIT source)
from .depth_anything3.model import (  # noqa: E402,F401  (import registers family)
    LibreDepthAnything3,
)
from .nafnet.model import LibreNAFNet  # noqa: E402,F401  (restore-only)
from .ddcolor.model import LibreDDColor  # noqa: E402,F401  (restore-only colorization)
from .hvi_cidnet.model import LibreHVICIDNet  # noqa: E402,F401  (restore-only low-light enhancement)
from .lama.model import LibreLaMa  # noqa: E402,F401  (mask-guided restore/inpainting)
from .birefnet.model import LibreBiRefNet  # noqa: E402,F401  (matte-only; can_load keyed on squeeze_module+gdt_convs_attn+ipt_blk)
from .feynobg.model import LibreFeyNobg  # noqa: E402,F401  (matte-only; BiRefNet keys + 24-block stage-3 marker, disjoint from birefnet)
from .ben2.model import LibreBEN2  # noqa: E402,F401  (matte-only; BEN2 cross-attention/refinement fingerprint)
from .vitmatte.model import LibreViTMatte  # noqa: E402,F401  (trimap-guided matte)
from .realesrgan.model import LibreRealESRGAN  # noqa: E402,F401  (restore/super-resolution; RRDBNet+SRVGG keys are unique)
from .quicksrnet.model import LibreQuickSRNet  # noqa: E402,F401  (restore/super-resolution; exact compact CNN fingerprint)
from .swinir.model import LibreSwinIR  # noqa: E402,F401  (restore/super-resolution; RSTB keys are unique)
from .fcn.model import LibreFCN  # noqa: E402,F401  (semantic-only; FCN head + embedded ResNet fingerprint)
from .eomt.model import LibreEoMT  # noqa: E402,F401  (semantic-only; EoMT query/mask keys are unique)
from .deeplabv3.model import LibreDeepLabv3  # noqa: E402,F401  (semantic-only; ASPP branch/project keys are unique)
from .pidnet.model import LibrePIDNet  # noqa: E402,F401  (semantic-only; can_load uses PIDNet fusion keys)
from .ppliteseg.model import LibrePPLiteSeg  # noqa: E402,F401  (semantic-only; can_load needs SPPM branches + UAFM 4->2->1 attention + PP-LiteSeg proj convs together)
from .unet.model import LibreUNet  # noqa: E402,F401  (semantic-only; can_load needs encoder.4.1.convs + decoder interp_upsample + FCN heads together)
from .segformer.model import LibreSegformer  # noqa: E402,F401  (semantic-only; can_load uses decode_head/encoder.stages keys, unique to this family)
from .lingbotvision.model import LibreLingBotVision  # noqa: E402,F401  (semantic-only; can_load keyed on backbone.rope_embed.periods + storage_tokens + predict head)
from .vit.model import LibreViT  # noqa: E402  (classify-only; top-level classic-ViT signature)
from .mobilenetv4.model import LibreMobileNetV4  # noqa: E402  (classify-only; can_load is highly specific)
from .convnext.model import LibreConvNeXt  # noqa: E402  (classify-only; can_load is highly specific)
from .deit.model import LibreDeiT  # noqa: E402  (classify-only museum family; exact ViT geometry)
from .swin.model import LibreSwin  # noqa: E402  (classify-only; V1 window-bias signature rejects SwinV2/backbone-only checkpoints)
from .efficientnetv2.model import LibreEfficientNetV2  # noqa: E402  (classify-only; can_load is highly specific)
from .vgg.model import LibreVGG  # noqa: E402  (classify-only; exact 3x3 stem + FC shape signature)
from .resnet.model import LibreResNet  # noqa: E402  (classify-only; standalone conv1+fc, rejects backbone embeds)
from .alexnet.model import LibreAlexNet  # noqa: E402  (classify-only; unique 11x11 stem + 3-layer classifier)

# Native CLIP zero-shot classifier: pure-torch towers (no open_clip at runtime),
# so it registers eagerly. can_load is uniquely keyed on logit_scale +
# text_projection + visual.conv1, so registration order does not matter.
from .clip.model import LibreCLIP  # noqa: E402,F401  (import registers family)

# Native SigLIP 2 zero-shot classifier: pure-torch towers (no transformers at
# runtime; the SentencePiece tokenizer is imported lazily behind [siglip2]), so
# it registers eagerly. can_load is uniquely keyed on logit_bias +
# vision_model.embeddings.patch_embedding + text_model.head, so order does not
# matter. NB: SigLIP carries logit_bias, which CLIP lacks.
from .siglip2.model import LibreSigLIP2  # noqa: E402,F401  (import registers family)

# Native Perception Encoder Core towers (no timm / open_clip at runtime; the
# OpenAI BPE tokenizer is reused from LibreCLIP), so it registers eagerly.
# can_load is uniquely keyed on visual.trunk.attn_pool.latent +
# visual.trunk.patch_embed.proj.weight + text.text_projection, a signature no
# other family carries, so order does not matter.
from .pe.model import LibrePE  # noqa: E402,F401  (import registers family)

# PP-OCRv5 text detection + recognition pipeline. can_load is uniquely keyed
# on the composite det.*/rec.* checkpoint layout, so order does not matter.
from .ppocr.model import LibrePPOCR  # noqa: E402,F401  (import registers family)

# V-JEPA 2 video encoder + attentive probe: a native pure-torch port (no
# transformers at runtime), so it registers eagerly. can_load is keyed on a 5D
# Conv3d tubelet patch embedding, which no image family can produce, so
# registration order does not matter. OpenCV is imported lazily inside the clip
# sampler, keeping video decoding off the global import path.
from .vjepa2.model import LibreVJEPA2  # noqa: E402,F401  (import registers family)


def _ensure_rfdetr():
    """Lazily register RF-DETR and LibreDINOv2 if their dependencies are installed."""
    if any(c.__name__ == "LibreRFDETR" for c in BaseModel._registry) and any(
        c.__name__ == "LibreDINOv2" for c in BaseModel._registry
    ):
        return
    import importlib.util

    # Native port: no longer depends on the rfdetr PyPI package; transformers
    # is what we need (DINOv2 backbone via AutoBackbone, plus segmentation).
    if importlib.util.find_spec("transformers") is None:
        raise ModuleNotFoundError(
            "RF-DETR support requires extra dependencies.\n"
            "Install with: pip install libreyolo[rfdetr]"
        )
    from .rfdetr.model import LibreRFDETR  # noqa: F401  (import triggers registration)

    # LibreDINOv2 shares the same transformers dependency (DINOv2 backbone).
    from .dinov2.model import LibreDINOv2  # noqa: F401  (import triggers registration)


def try_ensure_rfdetr():
    """Try to register RF-DETR. Returns the model class or ``None`` if unavailable."""
    try:
        _ensure_rfdetr()
    except (ImportError, ModuleNotFoundError):
        return None
    for cls in BaseModel._registry:
        if cls.__name__ == "LibreRFDETR":
            return cls
    return None


# =============================================================================
# Internal helpers
# =============================================================================


def _resolve_weights_path(model_path: str) -> str:
    """Resolve bare filenames to weights/ directory."""
    path = Path(model_path)
    if path.parent == Path(".") and not model_path.startswith(("./", "../")):
        weights_path = Path("weights") / path.name
        if weights_path.exists():
            return str(weights_path)
        if path.exists():
            return str(path)
        return str(weights_path)
    return model_path


def _unwrap_state_dict(state_dict: dict) -> dict:
    """Extract weights from nested checkpoint formats.

    Supports:
    - LibreYOLO trainer checkpoints (``model``)
    - legacy EMA wrappers (``ema``)
    - SuperGradients checkpoints (``ema_net`` / ``net``)
    - generic wrappers (``state_dict``)
    """
    if "ema" in state_dict and isinstance(state_dict.get("ema"), dict):
        ema_data = state_dict["ema"]
        return ema_data.get("module", ema_data)
    if "ema_net" in state_dict and isinstance(state_dict.get("ema_net"), dict):
        return state_dict["ema_net"]
    if "net" in state_dict and isinstance(state_dict.get("net"), dict):
        return state_dict["net"]
    if "model" in state_dict and isinstance(state_dict.get("model"), dict):
        return state_dict["model"]
    if "state_dict" in state_dict and isinstance(state_dict.get("state_dict"), dict):
        return state_dict["state_dict"]
    return state_dict


def _needs_rfdetr_registration(weights_dict: dict) -> bool:
    """Return True when checkpoint keys require lazy RF-DETR registration."""
    if LibreRTDETR.can_load(weights_dict):
        return False

    if LibreDeformableDETR.can_load(weights_dict):
        return False

    # LW-DETR carries enc_out_class_embed / enc_out_bbox_embed (RF-DETR forked
    # them from it), but is a core family with no transformers dependency.
    # Without this guard, loading LW-DETR weights would import RF-DETR and hard
    # fail whenever the optional ``rfdetr`` extra is not installed.
    if LibreDETR.can_load(weights_dict) or LibreLWDETR.can_load(weights_dict):
        return False

    if "linear.weight" in weights_dict and any(
        k.startswith("backbone.") for k in weights_dict
    ):
        return True

    keys_lower = [k.lower() for k in weights_dict]
    return any(
        "dinov2" in k
        or "query_embed" in k
        or "enc_out_class_embed" in k
        or "enc_out_bbox_embed" in k
        for k in keys_lower
    )


def _find_registered_family(family: str):
    for cls in BaseModel._registry:
        if cls.FAMILY == family:
            return cls
    return None


def _matching_model_classes(weights_dict: dict):
    return [cls for cls in BaseModel._registry if cls.can_load(weights_dict)]


def _looks_like_libreyolo_filename(model_path: str) -> bool:
    return Path(model_path).name.lower().startswith("libre")


def _has_any_libreyolo_metadata(loaded: object) -> bool:
    if not isinstance(loaded, dict):
        return False
    metadata_keys = set(REQUIRED_CHECKPOINT_METADATA_KEYS) - {"model"}
    return bool(metadata_keys & set(loaded))


# =============================================================================
# LibreYOLO — unified factory function
# =============================================================================


def LibreYOLO(
    model_path: str,
    size: str | None = None,
    reg_max: int = 16,
    nb_classes: int | None = None,
    device: str = "auto",
    task: str | None = None,
    compute_units: str = "all",
):
    """
    Unified factory that detects model family from weights and returns
    the appropriate model instance.

    Args:
        model_path: Path to weights (.pt), ONNX (.onnx), ExecuTorch (.pte),
                    MNN (.mnn), TensorRT (.engine), OpenVINO/Paddle/ncnn
                    directory, a Triton HTTP(S) model URL, or a Hugging Face
                    Hub reference: "owner/repo" or
                    "hf://owner/repo[@revision][/file.pt]" (requires the
                    optional huggingface_hub package).
        size: Model size variant (auto-detected from weights if omitted).
        reg_max: Regression max for DFL (YOLOv9 only, default: 16).
        nb_classes: Number of classes (auto-detected if omitted).
        device: Device for inference ("auto", "cuda", "cpu", "mps").
        task: Optional canonical task name. See ``libreyolo.tasks.TASKS``.
        compute_units: CoreML-only — Apple silicon routing for .mlpackage loads.
                       One of "all", "cpu_only", "cpu_and_gpu", "cpu_and_ne".
                       Ignored for non-CoreML formats.

    Returns:
        Model instance (LibreYOLOX, LibreYOLO9, LibreRFDETR, or inference backend).
    """
    ensure_default_logging()

    # Remote model references must route before local path normalization. A URL
    # has no meaningful local suffix and must never enter download/checkpoint
    # inspection paths.
    from ..backends.triton import is_triton_model_url

    if is_triton_model_url(model_path):
        from ..backends.triton import TritonBackend

        return TritonBackend(model_path, device=device, task=task)

    # Hugging Face Hub references ("owner/repo" or "hf://owner/repo") resolve
    # to a checkpoint in the shared huggingface_hub cache, then flow through
    # the exact same safe-load + metadata-validation path as a local file.
    from ..utils.hf_hub import maybe_resolve_hub_reference

    hub_source = None
    hub_checkpoint = maybe_resolve_hub_reference(model_path)
    if hub_checkpoint is not None:
        hub_source = model_path
        model_path = hub_checkpoint

    model_path = _resolve_weights_path(model_path)

    # librefacerec-* names route to the face-embedding family regardless of
    # extension: the family is ONNX-only, auto-downloads from the LibreYOLO
    # HF org, and infers task=embed from the name.
    if Path(model_path).name.lower().startswith("librefacerec-"):
        from ..tasks import normalize_task

        if task is not None and normalize_task(task) != "embed":
            raise ValueError(
                f"librefacerec weights only support the 'embed' "
                f"(facial-recognition) task, got task={task!r}."
            )
        from .facerec import LibreFaceEmbedder

        return LibreFaceEmbedder(model_path, device=device)

    # Non-PyTorch formats: delegate to inference backends
    if model_path.endswith(".onnx"):
        # Face embedding (facial-recognition) is an inference-only, two-stage
        # ONNX task with no detection-shaped output, so it routes to its own
        # runner rather than the detection ONNX backend.
        from ..tasks import normalize_task

        if task is not None and normalize_task(task) == "embed":
            from .facerec import LibreFaceEmbedder

            return LibreFaceEmbedder(model_path, device=device)

        from ..backends.onnx import OnnxBackend

        return OnnxBackend(
            model_path, nb_classes=nb_classes or 80, device=device, task=task
        )

    if model_path.endswith(".torchscript"):
        from ..backends.torchscript import TorchScriptBackend

        return TorchScriptBackend(
            model_path, nb_classes=nb_classes, device=device, task=task
        )

    if model_path.endswith(".pte"):
        from ..backends.executorch import ExecuTorchBackend

        return ExecuTorchBackend(
            model_path, nb_classes=nb_classes, device=device, task=task
        )

    if model_path.endswith(".tflite"):
        from ..backends.tflite import TFLiteBackend

        return TFLiteBackend(
            model_path, nb_classes=nb_classes, device=device, task=task
        )

    if model_path.endswith(".mnn"):
        from ..backends.mnn import MNNBackend

        return MNNBackend(
            model_path, nb_classes=nb_classes, device=device, task=task
        )

    if model_path.endswith((".engine", ".tensorrt")):
        from ..backends.tensorrt import TensorRTBackend

        return TensorRTBackend(
            model_path, nb_classes=nb_classes, device=device, task=task
        )

    if Path(model_path).is_dir() and (Path(model_path) / "model.xml").exists():
        from ..backends.openvino import OpenVINOBackend

        return OpenVINOBackend(
            model_path, nb_classes=nb_classes, device=device, task=task
        )

    if Path(model_path).is_dir() and all(
        (Path(model_path) / filename).exists()
        for filename in ("model.pdmodel", "model.pdiparams")
    ):
        from ..backends.paddle import PaddleBackend

        return PaddleBackend(
            model_path, nb_classes=nb_classes, device=device, task=task
        )

    if Path(model_path).is_dir() and Path(model_path).suffix == ".mlpackage":
        from ..backends.coreml import CoreMLBackend

        return CoreMLBackend(
            model_path,
            nb_classes=nb_classes or 80,
            device=device,
            compute_units=compute_units,
            task=task,
        )

    if Path(model_path).is_dir():
        ncnn_param = Path(model_path) / "model.ncnn.param"
        ncnn_bin = Path(model_path) / "model.ncnn.bin"
        if ncnn_param.exists() and ncnn_bin.exists():
            from ..backends.ncnn import NcnnBackend

            return NcnnBackend(
                model_path, nb_classes=nb_classes, device=device, task=task
            )

    if task is not None:
        filename = Path(model_path).name
        for cls in BaseModel._registry:
            if cls.detect_size_from_filename(filename) is not None:
                resolve_task(
                    explicit_task=task,
                    default_task=cls.DEFAULT_TASK,
                    supported_tasks=cls.SUPPORTED_TASKS,
                )
                break

    # Download if missing
    download_error: Exception | None = None
    if not Path(model_path).exists():
        if size is None:
            for cls in BaseModel._registry:
                detected = cls.detect_size_from_filename(Path(model_path).name)
                if detected is not None:
                    size = detected
                    logger.debug("Detected size '%s' from filename", size)
                    break
            # Try RF-DETR (may not be registered yet — cheap check)
            if size is None:
                try:
                    _ensure_rfdetr()
                    for cls in BaseModel._registry:
                        detected = cls.detect_size_from_filename(Path(model_path).name)
                        if detected is not None:
                            size = detected
                            logger.debug("Detected size '%s' from filename", size)
                            break
                except ModuleNotFoundError:
                    pass
            if size is None:
                from ..utils.hf_hub import looks_like_repo_id

                hub_hint = ""
                if looks_like_repo_id(model_path):
                    # A same-named local file or directory shadowed the bare
                    # owner/repo form (local always wins), so point at the
                    # explicit Hub syntax that bypasses that precedence.
                    hub_hint = (
                        f"\nIf you meant a Hugging Face Hub model, use the "
                        f"explicit form: LibreYOLO('hf://{model_path}')."
                    )
                raise ValueError(
                    f"Model weights file not found: {model_path}\n"
                    f"Cannot auto-download: unable to determine size from filename.\n"
                    f"Please specify size explicitly or provide a valid weights "
                    f"file path.{hub_hint}"
                )

        try:
            download_weights(model_path, size)
        except Exception as e:
            download_error = e
            logger.warning("Auto-download failed: %s", e)

    if not Path(model_path).exists():
        if download_error is not None:
            raise FileNotFoundError(
                f"Model weights file not found: {model_path}\n"
                f"Auto-download failed: {download_error}"
            ) from download_error
        raise FileNotFoundError(f"Model weights file not found: {model_path}")

    # Load weights once
    try:
        if Path(model_path).suffix == ".safetensors":
            try:
                from safetensors.torch import load_file as load_safetensors_file
            except ImportError as e:
                raise ImportError(
                    "Loading safetensors weights requires safetensors. "
                    "Install with: pip install safetensors"
                ) from e

            loaded = load_safetensors_file(model_path, device="cpu")
        else:
            loaded = load_untrusted_torch_file(
                model_path,
                map_location="cpu",
                context="model inspection",
            )
    except Exception as e:
        # Upstream flagship checkpoints can fail the safe inspection load — e.g.
        # RF-DETR embeds an argparse.Namespace that weights_only=True rejects.
        # Try auto-converting to a LibreYOLO v1.0 checkpoint and reload it.
        from .autoconvert import autoconvert_upstream_checkpoint

        converted_path = autoconvert_upstream_checkpoint(model_path)
        if converted_path is None:
            raise RuntimeError(
                f"Failed to load model weights from {model_path}: {e}"
            ) from e
        model_path = converted_path
        loaded = load_untrusted_torch_file(
            model_path,
            map_location="cpu",
            context="model inspection",
        )

    metadata_errors = validate_checkpoint_metadata(loaded, strict=False)
    has_v1_metadata = not metadata_errors
    has_partial_metadata = _has_any_libreyolo_metadata(loaded)
    is_legacy_libreyolo = (
        not has_v1_metadata
        and isinstance(loaded, dict)
        and (has_partial_metadata or _looks_like_libreyolo_filename(model_path))
    )
    if not has_v1_metadata:
        # Partial metadata such as ``names`` can appear in upstream fine-tunes.
        # Try recognized flagship conversion before treating the file as an old
        # LibreYOLO checkpoint; otherwise numbered upstream YOLO9 keys never
        # reach the converter.
        from .autoconvert import autoconvert_upstream_checkpoint

        converted_path = autoconvert_upstream_checkpoint(model_path, loaded=loaded)
        if converted_path is not None:
            return LibreYOLO(
                converted_path,
                size=size,
                reg_max=reg_max,
                nb_classes=nb_classes,
                device=device,
                task=task,
                compute_units=compute_units,
            )

        if is_legacy_libreyolo:
            logger.warning(
                "LibreYOLO checkpoint metadata is missing or incomplete for %s: %s. "
                "Loading through the legacy compatibility path. %s",
                model_path,
                "; ".join(metadata_errors),
                _METADATA_CONVERSION_HELP,
            )
        elif hub_source is not None:
            # Guessing a family from raw tensor keys is a coin flip for a file
            # pulled off the open Hub: an unrelated model can satisfy some
            # family's can_load() and then fail with a nonsense error about
            # that family. Metadata is the contract for Hub loads, so refuse
            # here and name the repo instead.
            raise ValueError(
                f"'{hub_source}' does not contain a LibreYOLO checkpoint.\n"
                f"Hugging Face models are loaded by their LibreYOLO metadata "
                f"(schema v1.0), and this file has none, so its architecture "
                f"cannot be identified.\n"
                f"If it is an upstream checkpoint LibreYOLO can convert, "
                f"download it and load it from a local path, which enables "
                f"the legacy architecture-detection path.\n"
                f"{_METADATA_CONVERSION_HELP}"
            )
        else:
            logger.warning(
                "LibreYOLO metadata was not found in %s. Loading through the "
                "legacy architecture-detection path. This appears to be an "
                "upstream or foreign checkpoint, not a LibreYOLO v1.0 checkpoint. %s",
                model_path,
                _METADATA_CONVERSION_HELP,
            )

    weights_dict = _unwrap_state_dict(loaded)

    # Ensure RF-DETR is registered if its keys are present, but avoid
    # treating RT-DETR checkpoints as RF-DETR. D-FINE also has
    # ``encoder``/``decoder``-ish keys, so only RF-DETR-specific markers
    # should trigger the lazy import.
    metadata_family_for_registration = (
        loaded.get("model_family")
        if isinstance(loaded, dict) and isinstance(loaded.get("model_family"), str)
        else None
    )
    if (
        metadata_family_for_registration in ("rfdetr", "dinov2")
        or _needs_rfdetr_registration(weights_dict)
        or (
            "predict.weight" in weights_dict
            and any(k.startswith("backbone.") for k in weights_dict)
        )
    ):
        try:
            _ensure_rfdetr()
        except ModuleNotFoundError:
            raise

    # Find the right model class. Metadata and filename hints come first so
    # DEIM-D-FINE and D-FINE, which intentionally share architecture keys, can
    # coexist without one stealing the other's LibreYOLO-format checkpoints.
    matched_cls = None
    metadata_family = (
        loaded.get("model_family")
        if isinstance(loaded, dict) and isinstance(loaded.get("model_family"), str)
        else None
    )
    if metadata_family:
        cls = _find_registered_family(metadata_family)
        if cls is not None and cls.can_load(weights_dict):
            matched_cls = cls

    if matched_cls is None:
        filename = Path(model_path).name
        for cls in BaseModel._registry:
            if cls.detect_size_from_filename(filename) and cls.can_load(weights_dict):
                matched_cls = cls
                break

    if matched_cls is None:
        matching_classes = _matching_model_classes(weights_dict)
        matching_families = {cls.FAMILY for cls in matching_classes}
        # Only raise on a true D-FINE/DEIM tie. Some optional families can add
        # broader false-positive matches after lazy registration, while EC
        # and DEIMv2 legitimately match D-FINE/DEIM-ish decoder keys and should
        # be allowed to win via their more-specific detectors.
        if {"dfine", "deim"}.issubset(matching_families) and not (
            matching_families & {"ec", "deimv2", "tinyformer"}
        ):
            raise ValueError(
                "Ambiguous D-FINE/DEIM checkpoint: both families share the same "
                "DEIM-D-FINE architecture keys.\n"
                "Use a LibreYOLO checkpoint with model_family metadata, an "
                "upstream-style filename such as dfine_hgnetv2_n_coco.pth or "
                "deim_hgnetv2_n_coco.pth, or instantiate LibreDFINE/LibreDEIM "
                "directly."
            )
        if matching_classes:
            matched_cls = matching_classes[0]

    if matched_cls is None:
        registered = sorted(
            {c.FAMILY for c in BaseModel._registry if getattr(c, "FAMILY", "")}
        )
        raise ValueError(
            "Could not detect model architecture from state dict keys.\n"
            f"Registered model families: {', '.join(registered)}."
        )

    if matched_cls.FAMILY == "pidnet" and not has_v1_metadata:
        raise ValueError(
            "Raw upstream PIDNet checkpoints must be converted before loading. "
            "Use weights/convert_pidnet_weights.py to create a LibreYOLO "
            "checkpoint with Cityscapes semantic metadata."
        )

    # Auto-detect size. Schema v1.0 metadata is authoritative when present;
    # shape sniffing stays as the legacy fallback (and is required for raw
    # upstream state dicts). Finalized quantized checkpoints replace some
    # weight keys with packed payloads, so sniffing alone cannot cover them.
    if size is None:
        meta_size = loaded.get("size") if isinstance(loaded, dict) else None
        if isinstance(meta_size, str) and meta_size:
            size = meta_size
    if size is None:
        if matched_cls.FAMILY == "rfdetr":
            # RF-DETR needs the full checkpoint for args-based detection
            size = matched_cls.detect_size(weights_dict, state_dict=loaded)
        else:
            size = matched_cls.detect_size(weights_dict)

        if size is None:
            # Fallback: try filename
            size = matched_cls.detect_size_from_filename(Path(model_path).name)

        if size is None:
            raise ValueError(
                f"Could not automatically detect {matched_cls.__name__} model size.\n"
                f"Please specify size explicitly: LibreYOLO('{model_path}', size='s')"
            )
        logger.debug("Auto-detected size: %s", size)

    # Determine how to pass weights
    # Checkpoints from our trainers have metadata (nc, names, model_family).
    # For those, pass the file path so _load_weights() handles nc rebuild + names.
    # For old/pretrained checkpoints, pass the extracted state_dict directly.
    has_metadata = has_v1_metadata or has_partial_metadata

    # Auto-detect nb_classes.
    #
    # Metadata checkpoints are reloaded via ``_load_weights()``, which reads the
    # saved ``nc`` and performs any family-specific rebuild logic. Starting from
    # the constructor default (80) avoids baking the fine-tuned class count into
    # the fresh model init too early. This matters for YOLO9-t where the class
    # branch width depends on COCO-vs-custom ``nc`` during construction.
    if nb_classes is None:
        if matched_cls.FAMILY in ("rfdetr", "dinov2", "eomt"):
            # Transformer dense heads build to the checkpoint's class width.
            # The 80 default below is a YOLO9-family convention that would
            # mis-size the head for a metadata-wrapped checkpoint.
            nb_classes = matched_cls.detect_nb_classes(weights_dict)
            if nb_classes is None:
                nb_classes = 80
        elif has_metadata:
            nb_classes = 80
        else:
            nb_classes = matched_cls.detect_nb_classes(weights_dict)
            if nb_classes is None:
                nb_classes = 80

    checkpoint_task = (
        loaded.get("task")
        if isinstance(loaded, dict) and isinstance(loaded.get("task"), str)
        else None
    )
    filename_task = matched_cls.detect_task_from_filename(Path(model_path).name)
    if checkpoint_task is None:
        checkpoint_task = matched_cls.detect_checkpoint_task(weights_dict)
    if checkpoint_task is None and matched_cls.FAMILY == "rfdetr":
        if any(k.startswith("segmentation_head") for k in weights_dict):
            checkpoint_task = "segment"
        elif any(k.startswith("keypoint_head") for k in weights_dict) or any(
            "keypoint" in k for k in weights_dict if k.startswith("transformer.")
        ):
            # Legacy clean-room keypoint_head.* weights or the GroupPose
            # transformer keypoint markers ported from RF-DETR v1.8.0.
            checkpoint_task = "pose"
    if checkpoint_task is None and matched_cls.FAMILY == "yolonas":
        if "heads.head1.pose_pred.weight" in weights_dict:
            checkpoint_task = "pose"
    if (
        checkpoint_task is None
        and filename_task is None
        and matched_cls.FAMILY == "yolo9"
    ):
        checkpoint_task = matched_cls.detect_checkpoint_task(weights_dict)
    if checkpoint_task is None and matched_cls.FAMILY == "ec":
        if "decoder.keypoint_embedding.weight" in weights_dict:
            checkpoint_task = "pose"
        elif any(
            k.startswith("decoder.decoder.segmentation_head") for k in weights_dict
        ):
            checkpoint_task = "segment"

    resolved_task = resolve_task(
        explicit_task=task,
        checkpoint_task=checkpoint_task,
        filename_task=filename_task,
        default_task=matched_cls.DEFAULT_TASK,
        supported_tasks=matched_cls.SUPPORTED_TASKS,
    )
    family_kwargs = (
        {"reg_max": reg_max}
        if matched_cls.FAMILY in ("yolo9", "yolo9_e2e", "yolo9_p2")
        else {}
    )
    if matched_cls.FAMILY in ("rfdetr", "dinov2"):
        # RF-DETR / DINOv2 always need the path (handle their own loading internally)
        model = matched_cls(
            model_path=model_path,
            size=size,
            nb_classes=nb_classes,
            device=device,
            task=resolved_task,
        )
    elif has_metadata:
        # Our trainer checkpoint — pass path for metadata handling
        model = matched_cls(
            model_path=model_path,
            size=size,
            nb_classes=nb_classes,
            device=device,
            task=resolved_task,
            **family_kwargs,
        )
    else:
        # Pretrained checkpoint — pass extracted state dict
        model = matched_cls(
            model_path=weights_dict,
            size=size,
            nb_classes=nb_classes,
            device=device,
            task=resolved_task,
            **family_kwargs,
        )

    model.model_path = model_path
    return model


__all__ = [
    "LibreYOLO",
    "LibreYOLOX",
    "LibreYOLO9",
    "LibreYOLO9E2E",
    "LibreYOLO9P2",
    "LibreYOLONAS",
    "LibreDFINE",
    "LibreDEIM",
    "LibreDETR",
    "LibreDEIMv2",
    "LibreMaskRCNN",
    "LibreFCOS",
    "LibreFasterRCNN",
    "LibreRetinaNet",
    "LibreSSD",
    "LibreCenterNet",
    "LibreEfficientDet",
    "LibreDeformableDETR",
    "LibreDINODETR",
    "LibreEC",
    "LibrePICODET",
    "LibreRTMDet",
    "LibreYOLO3",
    "LibreYOLO4",
    "LibreYOLO2",
    "LibreYOLO1",
    "LibreYOLO7",
    "LibreHRNet",
    "LibreDEKR",
    "LibreRTDETR",
    "LibreRTDETRv2",
    "LibreRTDETRv4",
    "LibreFOMO",
    "LibreMiDaS",
    "LibreDepthAnythingV2",
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
    "LibreUNet",
    "LibreSegformer",
    "LibreLingBotVision",
    "LibreViT",
    "LibreMobileNetV4",
    "LibreConvNeXt",
    "LibreSwin",
    "LibreEfficientNetV2",
    "LibreVGG",
    "LibreResNet",
    "LibreAlexNet",
    "LibreCLIP",
    "LibreSigLIP2",
    "LibrePE",
    "LibrePPOCR",
    "LibreFaceEmbedder",
    "try_ensure_rfdetr",
]


def __getattr__(name):
    # Lazy export so importing the face-embedding family (and its optional
    # onnxruntime dependency) only happens on first use.
    if name == "LibreFaceEmbedder":
        from .facerec import LibreFaceEmbedder

        return LibreFaceEmbedder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
