"""PyTorch-native quantization API.

Grammar:

- ``model.quantize(recipe, calib=...)`` transforms the loaded model in place
  (structure swap + calibration) and returns it. No gradients are involved.
- QAT is plain ``model.train(...)`` on a quantized model: the swapped modules
  carry fp32 master weights and STE fake-quant, so the existing trainers work
  unchanged.
- QAD is the same training step with the existing ``distill_model`` kwargs.

Everything runs in PyTorch (simulation tier: numerics-true on any device).
Checkpoints written by ``model.save()`` and by the trainer carry a ``quant``
manifest so ``LibreYOLO(path)`` can rebuild the quantized structure before
loading weights.
"""

import itertools
import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from .modules import (
    GroupQuantLinear,
    MXFP4Linear,
    NVFP4Linear,
    QUANT_MODULE_TYPES,
    QuantConv2d,
    QuantLinear,
)

logger = logging.getLogger(__name__)

QUANT_SCHEMA_VERSION = "1.0"
RECIPES = (
    "fp16",
    "bf16",
    "fp8",
    "int8",
    "w4a16",
    "w4a8",
    "nvfp4",
    "mxfp4",
    "int2",
)
# Recipe classes: casts touch dtype only; conv recipes quantize Conv2d and
# Linear; linear recipes quantize nn.Linear only (transformer families).
CAST_RECIPES = ("fp16", "bf16")
CONV_RECIPES = ("int8", "fp8")
LINEAR_RECIPES = ("w4a16", "w4a8", "nvfp4", "mxfp4", "int2")
# Recipes whose activations need calibration statistics.
CALIBRATED_RECIPES = ("int8", "fp8", "w4a8", "int2")
RESEARCH_RECIPES = ("int2",)
SUPPORTED_FAMILIES = ("yolo9", "rfdetr", "birefnet", "feynobg")
# Activation-range algorithms. minmax/percentile track running extremes;
# mse/entropy accumulate an absolute-value histogram during the calibration
# pass and sweep clipping thresholds afterwards (see calibrate.py). All of
# them set the per-tensor activation range only; weight scales are always
# per-channel absolute-max regardless of the algorithm.
CALIB_ALGORITHMS = ("auto", "percentile", "minmax", "mse", "entropy")

GROUP_SIZE = 128  # grouped-int recipes: scale group along in_features
INT2_GROUP_SIZE = 64

DEFAULT_CALIB_DATA = "coco128.yaml"

# Per-family keep-high-precision defaults (substring match on qualified module
# names). First layers and heads stay in float: standard practice, and the
# YOLO9 DFL conv is a fixed integral-expectation operator that must never be
# quantized.
_FAMILY_KEEP_HIGH_PRECISION: Dict[str, Tuple[str, ...]] = {
    "yolo9": ("head.", "backbone.conv0."),
    "rfdetr": (
        "class_embed",
        "bbox_embed",
        "angle_embed",
        "keypoint_head",
        "segmentation_head",
        "embeddings",
        "ref_point",
    ),
    # birefnet/feynobg (shared architecture): patch embed is the first layer;
    # conv_out1 is the final matte logit conv; gdt_convs_attn are the tiny
    # bilateral-reference gates whose sigmoids multiply whole feature maps
    # (quantization error there is amplified multiplicatively); gdt_convs_pred
    # and conv_ms_spvn are training-only supervision heads that never run at
    # inference, so quantizing them would only leave permanently uncalibrated
    # activation observers in the manifest.
    "birefnet": (
        "bb.patch_embed.",
        "conv_out1",
        "gdt_convs_attn",
        "gdt_convs_pred",
        "conv_ms_spvn",
        "regular_conv",
    ),
    "feynobg": (
        "bb.patch_embed.",
        "conv_out1",
        "gdt_convs_attn",
        "gdt_convs_pred",
        "conv_ms_spvn",
        "regular_conv",
    ),
}
_ALWAYS_KEEP = ("dfl",)

# Tensorwise weight scaling lets cuBLASLt fuse the complete FP8 Linear
# epilogue. Broad use is not accuracy-neutral on vision transformers, but the
# first Swin stage is both faster and more faithful in the FeyNobg divergence
# sweep. Exact selected names are persisted in each checkpoint. Other families
# stay per-channel until they have their own validation evidence.
_FAMILY_FP8_TENSORWISE_PREFIXES: Dict[str, Tuple[str, ...]] = {
    "feynobg": ("bb.layers.0.",),
}


class QuantizationError(ValueError):
    """Raised for unsupported or invalid quantization requests."""


def default_keep_high_precision(family: str) -> Tuple[str, ...]:
    return _FAMILY_KEEP_HIGH_PRECISION.get(family, ())


def _check_support(family: str, recipe: str):
    if recipe not in RECIPES:
        raise QuantizationError(
            f"Unknown quantization recipe '{recipe}'. Available: {', '.join(RECIPES)}"
        )
    if family not in SUPPORTED_FAMILIES:
        raise QuantizationError(
            f"Quantization is not supported for model family '{family}' yet. "
            f"Supported families: {', '.join(SUPPORTED_FAMILIES)}"
        )
    if family in ("birefnet", "feynobg") and recipe in RESEARCH_RECIPES:
        raise QuantizationError(
            f"'{recipe}' requires QAT/QAD healing, and the {family} family is "
            "inference-only (no trainer), so PTQ-only int2 would be unusable. "
            f"Use 'nvfp4'/'mxfp4'/'w4a16' for sub-8-bit {family}."
        )
    if family == "yolo9" and recipe in LINEAR_RECIPES:
        raise QuantizationError(
            f"'{recipe}' targets nn.Linear layers and is not supported for "
            "the conv-heavy yolo9 family (sub-8-bit acceleration is "
            "GEMM-only, so convolutions stay in higher precision). Use "
            "recipe='int8' or 'fp8' for yolo9, or this recipe on the "
            "transformer-based rfdetr family."
        )


def _is_excluded(name: str, keep: Tuple[str, ...]) -> bool:
    for pattern in (*keep, *_ALWAYS_KEEP):
        if pattern and pattern in name:
            return True
    return False


def _select_modules(
    root: nn.Module,
    recipe: str,
    keep: Tuple[str, ...],
) -> Dict[str, nn.Module]:
    """Deterministically select float modules to swap for a recipe."""
    selected: Dict[str, nn.Module] = {}
    for name, module in root.named_modules():
        if not name or _is_excluded(name, keep):
            continue
        if recipe in CONV_RECIPES:
            if type(module) is nn.Conv2d or type(module) is nn.Linear:
                selected[name] = module
        elif recipe in LINEAR_RECIPES:
            if type(module) is nn.Linear:
                selected[name] = module
    return selected


def _swap_module(root: nn.Module, name: str, new_module: nn.Module):
    parent = root
    *parents, attr = name.split(".")
    for part in parents:
        parent = getattr(parent, part)
    setattr(parent, attr, new_module)


def _swap_selected(root: nn.Module, recipe: str, selected: Dict[str, nn.Module]) -> Dict[str, int]:
    counts: Dict[str, int] = {}

    def _count(kind: str):
        counts[kind] = counts.get(kind, 0) + 1

    for name, module in selected.items():
        if recipe in CONV_RECIPES:
            is_conv = type(module) is nn.Conv2d
            new = (
                QuantConv2d.from_float(module)
                if is_conv
                else QuantLinear.from_float(module)
            )
            new._q_wformat = recipe  # "int8" | "fp8"
            new._q_aformat = recipe
            kind = f"{'conv' if is_conv else 'linear'}_{recipe}"
            new._q_kind = kind
            _swap_module(root, name, new)
            _count(kind)
        elif recipe == "nvfp4":
            new = NVFP4Linear.from_float(module)
            new._q_kind = "linear_nvfp4"
            _swap_module(root, name, new)
            _count("linear_nvfp4")
        elif recipe == "mxfp4":
            new = MXFP4Linear.from_float(module)
            new._q_kind = "linear_mxfp4"
            _swap_module(root, name, new)
            _count("linear_mxfp4")
        elif recipe in ("w4a16", "w4a8", "int2"):
            bits = 2 if recipe == "int2" else 4
            group = INT2_GROUP_SIZE if recipe == "int2" else GROUP_SIZE
            new = GroupQuantLinear.from_float(
                module,
                bits=bits,
                group_size=group,
                act_int8=recipe in ("w4a8", "int2"),
            )
            new._q_kind = f"linear_{recipe}"
            _swap_module(root, name, new)
            _count(f"linear_{recipe}")
    return counts


def _quant_modules(root: nn.Module):
    for name, module in root.named_modules():
        if isinstance(module, QUANT_MODULE_TYPES):
            yield name, module


def _reference_tensor(module: nn.Module) -> torch.Tensor:
    """Return any tensor the module owns, to read its device from.

    Quant modules differ in what they hold depending on their state, so no
    single attribute is always present: a prepared MXFP4Linear owns only the
    weight parameter and registers no buffers, while a finalized one deletes
    that parameter and owns packed buffers instead. Bias is absent on many
    modules. Look at parameters and buffers together.
    """
    for tensor in itertools.chain(
        module.parameters(recurse=False), module.buffers(recurse=False)
    ):
        if tensor is not None:
            return tensor
    raise RuntimeError(
        f"{type(module).__name__} owns no parameters or buffers, so its "
        "device cannot be determined"
    )


def _set_fp8_tensorwise_modules(root: nn.Module, names) -> Tuple[str, ...]:
    """Mark exact QuantLinear names for tensorwise FP8 weight scaling."""
    requested = tuple(str(name) for name in names)
    modules = dict(root.named_modules())
    missing = []
    for name in requested:
        module = modules.get(name)
        if not isinstance(module, QuantLinear):
            missing.append(name)
            continue
        module._q_fp8_weight_scaling = "tensorwise"
    if missing:
        raise QuantizationError(
            "FP8 tensorwise manifest names do not resolve to QuantLinear "
            f"modules: {missing[:8]}"
        )
    return requested


def _default_fp8_tensorwise_modules(family: str, root: nn.Module) -> Tuple[str, ...]:
    prefixes = _FAMILY_FP8_TENSORWISE_PREFIXES.get(family, ())
    if not prefixes:
        return ()
    return tuple(
        name
        for name, module in root.named_modules()
        if isinstance(module, QuantLinear) and name.startswith(prefixes)
    )


def _cast_finalized_remainder(root: nn.Module, dtype: torch.dtype):
    """Cast deployment parameters without changing quantized buffer formats."""
    exact_buffers = {
        id(buffer)
        for _, module in _quant_modules(root)
        for buffer in module.buffers(recurse=False)
    }
    with torch.no_grad():
        for parameter in root.parameters():
            if parameter.is_floating_point() and parameter.dtype != dtype:
                parameter.data = parameter.data.to(dtype)
        for module in root.modules():
            for buffer in module.buffers(recurse=False):
                if (
                    id(buffer) not in exact_buffers
                    and buffer.is_floating_point()
                    and buffer.dtype != dtype
                ):
                    buffer.data = buffer.data.to(dtype)


def _cast_tree(obj, dtype):
    if torch.is_tensor(obj) and obj.is_floating_point():
        return obj.to(dtype)
    if isinstance(obj, dict):
        return {k: _cast_tree(v, dtype) for k, v in obj.items()}
    if isinstance(obj, tuple):
        return tuple(_cast_tree(v, dtype) for v in obj)
    if isinstance(obj, list):
        return [_cast_tree(v, dtype) for v in obj]
    return obj


def _install_io_hooks(root: nn.Module, dtype: torch.dtype):
    """Root-level float32 I/O contract around a half-width interior."""
    if getattr(root, "_q_fp16_hooks", None):
        return

    def _pre(module, args):
        return tuple(
            a.to(dtype) if torch.is_tensor(a) and a.dtype == torch.float32 else a
            for a in args
        )

    def _post(module, args, output):
        return _cast_tree(output, torch.float32)

    root._q_fp16_hooks = [
        root.register_forward_pre_hook(_pre),
        root.register_forward_hook(_post),
    ]


def _install_cast_io_hooks(root: nn.Module, dtype: torch.dtype):
    """Cast the model to a half-width dtype, keeping float32 I/O at the root."""
    root.to(dtype)
    _install_io_hooks(root, dtype)


def _cast_dtype(recipe: str) -> torch.dtype:
    return torch.float16 if recipe == "fp16" else torch.bfloat16


def _set_observing(root: nn.Module, flag: bool):
    for _, module in _quant_modules(root):
        if hasattr(module, "_q_observing"):
            module._q_observing = flag


def _run_calibration(
    wrapper,
    calib: str,
    samples: int,
    batch: int,
    allow_download_scripts: bool,
    algorithm: str = "percentile",
):
    from ..export.calibration import CalibrationDataLoader

    loader = CalibrationDataLoader(
        data=calib,
        imgsz=wrapper._get_input_size(),
        batch=batch,
        fraction=1.0,
        samples=samples,
        preprocess_fn=wrapper._get_preprocess_numpy(),
        allow_download_scripts=allow_download_scripts,
    )
    root = wrapper.model
    was_training = root.training
    root.eval()
    for _, module in _quant_modules(root):
        module._q_obs_method = algorithm
    _set_observing(root, True)
    seen = 0
    try:
        with torch.no_grad():
            for np_batch in loader:
                x = torch.from_numpy(np_batch).to(wrapper.device)
                root(x)
                seen += x.shape[0]
    finally:
        # Restore on every exit path. A batch that raises (bad sample, OOM,
        # interrupt) would otherwise leave every observer live, so each later
        # forward keeps widening the ranges and silently corrupts calibration.
        _set_observing(root, False)
        if was_training:
            root.train()
    for _, module in _quant_modules(root):
        if hasattr(module, "finalize_observation"):
            module.finalize_observation()
        module._q_obs_method = "minmax"
    return seen


def quant_info(wrapper) -> Optional[Dict]:
    """Summary of the model's quantization state, or None if float."""
    manifest = getattr(wrapper, "_quant_manifest", None)
    if not manifest:
        return None
    info = dict(manifest)
    info.setdefault("state", "prepared")
    counts: Dict[str, int] = {}
    calibrated = True
    for _, module in _quant_modules(wrapper.model):
        kind = getattr(module, "_q_kind", type(module).__name__)
        counts[kind] = counts.get(kind, 0) + 1
        needs_act_calib = not (
            isinstance(module, GroupQuantLinear) and not module._q_act_enabled
        )
        if needs_act_calib:
            calibrated = calibrated and module.q_calibrated
    info["module_counts"] = counts
    if manifest.get("recipe") not in CAST_RECIPES:
        info["calibrated"] = calibrated
    return info


def quantize_model(
    wrapper,
    recipe: str,
    calib: Optional[str] = DEFAULT_CALIB_DATA,
    samples: int = 128,
    batch: int = 8,
    algorithm: str = "auto",
    keep_high_precision: Optional[Tuple[str, ...]] = None,
    allow_download_scripts: bool = False,
    verbose: bool = True,
):
    """Quantize a loaded LibreYOLO model in place and return it."""
    if getattr(wrapper, "_quant_manifest", None):
        raise QuantizationError(
            "Model is already quantized "
            f"(recipe='{wrapper._quant_manifest.get('recipe')}'). Load a float "
            "checkpoint to quantize with a different recipe."
        )

    family = wrapper.FAMILY
    recipe = str(recipe).lower()
    _check_support(family, recipe)

    algorithm = str(algorithm).lower()
    if algorithm not in CALIB_ALGORITHMS:
        raise QuantizationError(
            f"Unknown calibration algorithm '{algorithm}'. "
            f"Available: {', '.join(CALIB_ALGORITHMS)}"
        )
    if algorithm == "auto":
        # Measured on coco128: multi-batch min/max beats percentile clipping
        # for every tested model, and percentile collapses transformer
        # activations (their outliers are load-bearing). minmax is the
        # default; percentile remains selectable for workloads where it helps.
        algorithm = "minmax"
    if algorithm == "percentile" and family == "rfdetr":
        logger.warning(
            "percentile calibration clips transformer activation outliers, "
            "which measurably degrades DETR-family accuracy; prefer "
            "algorithm='minmax' for rfdetr."
        )

    keep = (
        tuple(keep_high_precision)
        if keep_high_precision is not None
        else default_keep_high_precision(family)
    )

    manifest = {
        "schema": QUANT_SCHEMA_VERSION,
        "recipe": recipe,
        "keep_high_precision": list(keep),
        "execution": "simulated",
        "calibrated": False,
        "calib_data": None,
        "calib_samples": 0,
        "calib_algorithm": None,
        "module_count": 0,
    }

    if recipe in CAST_RECIPES:
        if wrapper.device.type == "cpu":
            logger.warning(
                "%s on CPU is functional but slow; use a GPU device.", recipe
            )
        _install_cast_io_hooks(wrapper.model, _cast_dtype(recipe))
        manifest["execution"] = "native"
        manifest["calibrated"] = True
    else:
        if recipe in RESEARCH_RECIPES:
            logger.warning(
                "'%s' is a research preview: PTQ at this bit width is not "
                "usable on its own. Run train() on the quantized model "
                "(QAT, or QAD with distill_model=) to recover accuracy.",
                recipe,
            )
        selected = _select_modules(wrapper.model, recipe, keep)
        if not selected:
            raise QuantizationError(
                f"No quantizable modules found for recipe '{recipe}' on family "
                f"'{family}' with keep_high_precision={list(keep)}."
            )
        counts = _swap_selected(wrapper.model, recipe, selected)
        manifest["module_count"] = sum(counts.values())
        if recipe == "fp8":
            tensorwise = _default_fp8_tensorwise_modules(family, wrapper.model)
            if tensorwise:
                _set_fp8_tensorwise_modules(wrapper.model, tensorwise)
                manifest["fp8_tensorwise_weights"] = list(tensorwise)

        if recipe in CALIBRATED_RECIPES:
            if calib is not None:
                try:
                    seen = _run_calibration(
                        wrapper,
                        calib,
                        samples,
                        batch,
                        allow_download_scripts,
                        algorithm=algorithm,
                    )
                except BaseException:
                    # Calibration may already have written partial observer
                    # ranges. Restore the original float modules so the call
                    # is transactional and the caller can safely retry.
                    for name, module in selected.items():
                        _swap_module(wrapper.model, name, module)
                    raise
                manifest["calibrated"] = True
                manifest["calib_data"] = str(calib)
                manifest["calib_samples"] = int(seen)
                manifest["calib_algorithm"] = algorithm
            else:
                logger.warning(
                    "%s quantization without calibration: activations stay "
                    "in float (weight-only simulation). Pass calib= to "
                    "calibrate.",
                    recipe,
                )
        else:
            if calib is not None and calib != DEFAULT_CALIB_DATA:
                logger.info(
                    "'%s' needs no activation calibration (dynamic or "
                    "weight-only scaling); calibration data was ignored.",
                    recipe,
                )
            manifest["calibrated"] = True

    wrapper._quant_manifest = manifest
    wrapper.model.to(wrapper.device)

    if verbose:
        info = quant_info(wrapper)
        counts = info.get("module_counts", {})
        described = ", ".join(f"{v} {k}" for k, v in counts.items() if v) or "cast only"
        logger.info(
            "Quantized %s to %s (%s; execution=%s, calibrated=%s)",
            wrapper._get_model_name(),
            recipe,
            described,
            info.get("execution"),
            info.get("calibrated"),
        )
    return wrapper


def _set_export_mode(root: nn.Module, enabled: bool):
    for _, module in _quant_modules(root):
        if isinstance(module, (QuantConv2d, QuantLinear)):
            if enabled and not module.is_finalized:
                module.freeze_weight_qparams()
            module._q_export_mode = enabled
            module._q_observing = False


def export_finalized_pt(wrapper, out=None, remainder: str = "fp16") -> str:
    """Write the crystallized quantized checkpoint (deployment artifact).

    Packed low-bit weights + scales + manifest, masters stripped. Unpacking
    reproduces the simulation bit for bit, so a finalized checkpoint scores
    exactly what the prepared model scores (with ``remainder="fp32"``;
    ``"fp16"`` additionally halves every non-quantized float tensor, a
    near-lossless size win). Loading one yields an inference-ready model;
    training it re-prepares masters from the packed weights (QAT-from-PTQ).
    """
    import copy
    from pathlib import Path

    from ..utils.serialization import wrap_libreyolo_checkpoint

    manifest = dict(wrapper._quant_manifest)
    recipe = manifest.get("recipe")
    remainder = str(remainder).lower()
    if remainder not in ("fp16", "fp32"):
        raise QuantizationError(
            f"remainder must be 'fp16' or 'fp32', got '{remainder}'."
        )

    # Finalize a deep copy and let the family's own state_dict() produce
    # the keys (some families re-key their state dicts, so name-based key
    # surgery would miss them). The live model stays prepared and trainable.
    # Packing runs on the live device: quantization kernels can round one
    # code differently across devices, and the invariant we promise is
    # "finalized scores exactly what you validated on this device".
    model_copy = copy.deepcopy(wrapper.model)

    # Live LoRA adapters are folded on the copy so the checkpoint carries
    # plain dense weights; the live model keeps its adapters trainable.
    from ..training.lora import merge_lora_adapters, module_has_lora

    if module_has_lora(model_copy):
        merge_lora_adapters(model_copy)

    with torch.no_grad():
        if recipe not in CAST_RECIPES:
            for _, module in _quant_modules(model_copy):
                if hasattr(module, "make_finalized"):
                    module.make_finalized()
        model_copy = model_copy.cpu()
        new_sd = model_copy.state_dict()
        if remainder == "fp16" and recipe not in CAST_RECIPES:
            # Quant buffers keep their contract dtypes (packed payloads,
            # scales, exponents, observer state); only the non-quantized
            # remainder is cast. Collect the buffer names from the
            # finalized modules themselves so recipes that register new
            # buffers stay protected without maintaining a name list here.
            keep_exact = tuple(
                sorted(
                    {
                        "." + name
                        for _, module in _quant_modules(model_copy)
                        for name, _ in module.named_buffers(recurse=False)
                    }
                )
            )
            new_sd = {
                key: (
                    value.half()
                    if torch.is_tensor(value)
                    and value.dtype == torch.float32
                    and not (keep_exact and key.endswith(keep_exact))
                    else value
                )
                for key, value in new_sd.items()
            }
    del model_copy

    manifest["state"] = "finalized"
    manifest["remainder"] = recipe if recipe in CAST_RECIPES else remainder

    # Dual-write rectangular sizes like BaseModel.save() so the finalized
    # checkpoint reloads at the size it was trained at (#899).
    native_imgsz = wrapper._get_input_size()
    if isinstance(native_imgsz, (tuple, list)):
        imgsz_h, imgsz_w = int(native_imgsz[0]), int(native_imgsz[1])
        imgsz_metadata = {
            "imgsz": max(imgsz_h, imgsz_w),
            "imgsz_h": imgsz_h,
            "imgsz_w": imgsz_w,
        }
    else:
        imgsz_metadata = {"imgsz": int(native_imgsz)}
    checkpoint = wrap_libreyolo_checkpoint(
        new_sd,
        model_family=wrapper._get_model_name(),
        size=wrapper.size,
        task=wrapper.task,
        nc=wrapper.nb_classes,
        names=wrapper.names,
        **imgsz_metadata,
    )
    checkpoint["quant"] = manifest

    if out is None:
        if wrapper.model_path:
            src = Path(str(wrapper.model_path))
            out = src.with_name(f"{src.stem}-final{src.suffix or '.pt'}")
        else:
            out = f"{wrapper._get_model_name()}{wrapper.size}-{recipe}-final.pt"
    out = Path(out)
    if out.parent != Path("."):
        out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, out)
    logger.info("Finalized %s checkpoint saved to %s", recipe, out)
    return str(out)


def reprepare_model(wrapper):
    """Reconstruct fp32 masters from a finalized model (QAT-from-PTQ)."""
    manifest = getattr(wrapper, "_quant_manifest", None)
    if not manifest or manifest.get("state") != "finalized":
        return wrapper
    for _, module in _quant_modules(wrapper.model):
        if hasattr(module, "make_prepared"):
            module.make_prepared()
    manifest = dict(manifest)
    manifest["state"] = "prepared"
    wrapper._quant_manifest = manifest
    if manifest.get("recipe") not in CAST_RECIPES:
        wrapper.model.float()
    wrapper.model.to(wrapper.device)
    logger.info(
        "Re-prepared finalized checkpoint: fp32 masters reconstructed from "
        "packed weights (QAT-from-PTQ)."
    )
    return wrapper


def quantized_export(wrapper, format: str = "onnx", **kwargs) -> str:
    """Export a quantized model.

    int8 models export to ONNX with in-graph QuantizeLinear/DequantizeLinear
    pairs carrying the model's own calibrated (or QAT-trained) scales, so
    ONNX Runtime and TensorRT run real INT8 kernels with exactly the
    arithmetic that was validated in PyTorch. Other recipes are rejected
    with instructions.
    """
    manifest = getattr(wrapper, "_quant_manifest", None) or {}
    recipe = manifest.get("recipe")
    fmt = str(format).lower()

    if fmt in ("pt", "pytorch"):
        out = kwargs.pop("out", None)
        remainder = kwargs.pop("remainder", "fp16")
        if kwargs:
            logger.info("Ignoring export kwargs for format='pt': %s", sorted(kwargs))
        # export_finalized_pt works on a deep copy (folding any live LoRA
        # adapters there); the live model is never mutated on this path.
        return export_finalized_pt(wrapper, out=out, remainder=remainder)

    if recipe in CAST_RECIPES:
        raise QuantizationError(
            f"{recipe}-quantized models do not need a quantized exporter: "
            "call model.dequantize() and export with the float exporters "
            "(half=True gives fp16 ONNX), or use format='pt'."
        )
    if recipe != "int8":
        raise QuantizationError(
            f"'{recipe}' has no deployable ONNX form yet; it executes in "
            "PyTorch. Use format='pt' for the crystallized checkpoint, or "
            "dequantize() to use the float exporters."
        )
    if fmt != "onnx":
        raise QuantizationError(
            "int8-quantized export currently supports format='onnx' (QDQ "
            "graph, consumable by ONNX Runtime and TensorRT). Requested: "
            f"'{fmt}'. Export to ONNX and build downstream engines from it, "
            "or dequantize() and use the float exporters."
        )

    if kwargs.pop("int8", False):
        logger.info(
            "Model is already int8-quantized; the int8 export flag is "
            "ignored (Q/DQ scales come from the model itself)."
        )
    if kwargs.pop("half", False):
        logger.warning("half=True is ignored for int8-quantized export.")
    if not manifest.get("calibrated"):
        logger.warning(
            "Exporting an int8 model whose activations were never "
            "calibrated: the graph carries weight Q/DQ only. Pass calib= to "
            "quantize() for full W8A8 deployment."
        )

    from ..export import BaseExporter

    def _pre_trace():
        # Invoked by BaseExporter.__call__ after every request rejection
        # (option preflight, imgsz/path resolution): both steps below mutate
        # the live model, so a rejected export must leave the finalized
        # packed state untouched.
        # QDQ emission traces fake-quant over fp32 masters; reconstruct them
        # from the packed weights (exact by the packing invariant). No-op
        # unless the manifest state is "finalized".
        reprepare_model(wrapper)
        _set_export_mode(wrapper.model, True)

    try:
        return BaseExporter.create("onnx", wrapper)(
            _pre_trace_hook=_pre_trace, **kwargs
        )
    finally:
        _set_export_mode(wrapper.model, False)


def dequantize_model(wrapper):
    """Restore float modules in place, keeping the master weights.

    After QAT/QAD the fp32 masters are quantization-trained, so the restored
    float model is the correct input for deployment exporters (for example
    ``export(format="onnx", int8=True, data=...)``, which re-derives QDQ
    scales through the existing calibrated exporter). For the ``fp16`` recipe
    this restores dtype and removes the I/O cast hooks; the mantissa bits
    lost by the earlier half-cast are not recoverable.
    """
    manifest = getattr(wrapper, "_quant_manifest", None)
    if not manifest:
        return wrapper
    root = wrapper.model

    if manifest.get("recipe") in CAST_RECIPES:
        for handle in getattr(root, "_q_fp16_hooks", ()):
            handle.remove()
        if hasattr(root, "_q_fp16_hooks"):
            del root._q_fp16_hooks
        root.float()
    else:
        for name, module in list(_quant_modules(root)):
            finalized = getattr(module, "is_finalized", False)
            ref = _reference_tensor(module)
            if isinstance(module, QuantConv2d):
                new = nn.Conv2d(
                    module.in_channels,
                    module.out_channels,
                    module.kernel_size,
                    stride=module.stride,
                    padding=module.padding,
                    dilation=module.dilation,
                    groups=module.groups,
                    bias=module.bias is not None,
                    padding_mode=module.padding_mode,
                    device=ref.device,
                    dtype=torch.float32,
                )
            elif isinstance(
                module, (QuantLinear, NVFP4Linear, MXFP4Linear, GroupQuantLinear)
            ):
                new = nn.Linear(
                    module.in_features,
                    module.out_features,
                    bias=module.bias is not None,
                    device=ref.device,
                    dtype=torch.float32,
                )
            else:
                continue
            if finalized:
                with torch.no_grad():
                    new.weight = nn.Parameter(module._effective_weight())
            else:
                new.weight = module.weight
            new.bias = module.bias
            _swap_module(root, name, new)
        root.float()

    wrapper._quant_manifest = None
    logger.info("Restored float modules (recipe '%s' removed).", manifest.get("recipe"))
    return wrapper


def apply_quant_structure(wrapper, manifest: Dict):
    """Re-apply a checkpoint's quantized structure before loading weights.

    Idempotent: already-swapped modules are left alone. Calibration state is
    restored from the checkpoint buffers by the subsequent load_state_dict.
    """
    recipe = manifest.get("recipe")
    if recipe not in RECIPES:
        raise QuantizationError(
            f"Checkpoint has unknown quantization recipe '{recipe}'."
        )
    _check_support(wrapper.FAMILY, recipe)

    if recipe in CAST_RECIPES:
        if not getattr(wrapper, "_quant_manifest", None):
            _install_cast_io_hooks(wrapper.model, _cast_dtype(recipe))
    else:
        keep_raw = manifest.get("keep_high_precision")
        keep = (
            tuple(keep_raw)
            if keep_raw is not None
            else default_keep_high_precision(wrapper.FAMILY)
        )
        selected = _select_modules(wrapper.model, recipe, keep)
        counts = _swap_selected(wrapper.model, recipe, selected)
        tensorwise = manifest.get("fp8_tensorwise_weights", ())
        if tensorwise:
            if recipe != "fp8":
                raise QuantizationError(
                    "fp8_tensorwise_weights is only valid for recipe='fp8'."
                )
            _set_fp8_tensorwise_modules(wrapper.model, tensorwise)
        swapped = sum(counts.values())
        expected = int(manifest.get("module_count") or 0)
        already = sum(1 for _ in _quant_modules(wrapper.model))
        if expected and already != expected:
            logger.warning(
                "Quantized checkpoint expected %d quantized modules but the "
                "rebuilt model has %d; scales may not fully load.",
                expected,
                already,
            )
        if manifest.get("state") == "finalized":
            # Build the packed-buffer structure so the checkpoint's payload
            # and scale tensors resolve; load_state_dict fills them next.
            for _, module in _quant_modules(wrapper.model):
                if hasattr(module, "make_finalized"):
                    module.make_finalized()
            remainder = manifest.get("remainder", "fp32")
            if remainder == "fp16":
                _cast_finalized_remainder(wrapper.model, torch.float16)
                # The fp16 remainder needs the same root-level float32 I/O
                # contract as the cast recipes: without it, fp32 inputs meet
                # fp16 modules (torchvision deform_conv2d and friends reject
                # mixed dtypes outright).
                _install_io_hooks(wrapper.model, torch.float16)
            elif remainder != "fp32":
                raise QuantizationError(
                    "Finalized checkpoint has unsupported remainder dtype "
                    f"'{remainder}'."
                )
        if swapped:
            wrapper.model.to(wrapper.device)

    wrapper._quant_manifest = dict(manifest)
    return wrapper
