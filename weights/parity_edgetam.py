"""Run staged, offline EdgeTAM parity against the pinned upstream checkout.

This development script performs both artifact parity and runtime parity. It
safe-loads the official checkpoint, replays the pinned conversion, then compares
LibreEdgeTAM with the official Apache-2.0 implementation stage by stage:

* exact preprocessing pixels and prompt geometry;
* image encoder/FPN embeddings;
* point, box, and grouped positive/negative prompt embeddings;
* mask-decoder logits and quality scores; and
* final binary masks returned by both public image-prediction APIs.

Every input must already be local. Network access is forcibly disabled and the
upstream runtime source tree, checkpoint, mirror, and optional conversion
reference are verified against immutable digests before execution.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

try:
    from convert_edgetam_weights import (
        OFFICIAL_CHECKPOINT_SHA256,
        OFFICIAL_LICENSE_SHA256,
        OFFICIAL_REVISION,
        OFFICIAL_SOURCE_COMMIT,
        REFERENCE_REVISION,
        compare_state_dicts,
        convert_state_dict,
        load_official_state_dict,
        load_safetensors_state,
        sha256_file,
        strict_load_model,
        validate_payload,
        validate_reference_dir,
    )
except ModuleNotFoundError:  # Imported as weights.parity_edgetam.
    from weights.convert_edgetam_weights import (
        OFFICIAL_CHECKPOINT_SHA256,
        OFFICIAL_LICENSE_SHA256,
        OFFICIAL_REVISION,
        OFFICIAL_SOURCE_COMMIT,
        REFERENCE_REVISION,
        compare_state_dicts,
        convert_state_dict,
        load_official_state_dict,
        load_safetensors_state,
        sha256_file,
        strict_load_model,
        validate_payload,
        validate_reference_dir,
    )

TENSOR_ATOL = 1e-4
MASK_IOU_MIN = 0.99

# SHA-256 over each relative path, a NUL byte, its LF-normalized bytes, and a
# trailing NUL byte, in sorted order. The set is LICENSE plus every *.py under
# sam2/ and every *.yaml under sam2/configs/ at OFFICIAL_SOURCE_COMMIT. This
# permits a GitHub source archive as well as a git checkout while still pinning
# every Python/config file that can participate in the runtime comparison.
UPSTREAM_RUNTIME_TREE_SHA256 = (
    "34747b5dd93ef43aa325a9a1d4c4512cf256c341cbfba43620b86d8c2243f9a1"
)
UPSTREAM_RUNTIME_FILE_COUNT = 36


def _force_offline() -> None:
    """Override false-valued ambient settings so the run cannot reach the Hub.

    Deliberately not done at import: this module is importable (see the
    ``weights.parity_edgetam`` fallback above), and flipping process-wide
    offline flags as an import side effect would break unrelated
    ``from_pretrained`` calls in whatever imported it. ``timm`` and
    ``transformers`` are imported lazily inside the run, after this point, so
    setting the flags here still covers model setup.
    """
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"


def _stage(index: int, message: str) -> None:
    print(f"[{index}/9] {message}", flush=True)


def _upstream_runtime_files(root: Path) -> list[Path]:
    files = {
        *root.joinpath("sam2").rglob("*.py"),
        *root.joinpath("sam2", "configs").rglob("*.yaml"),
        root / "LICENSE",
    }
    return sorted(files, key=lambda path: path.relative_to(root).as_posix())


def validate_upstream_source(path: str | Path) -> dict[str, Any]:
    """Require the exact runtime source tree from the pinned official commit."""
    root = Path(path).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Upstream EdgeTAM directory does not exist: {root}")
    required = (
        root / "LICENSE",
        root / "sam2" / "__init__.py",
        root / "sam2" / "build_sam.py",
        root / "sam2" / "configs" / "edgetam.yaml",
        root / "sam2" / "sam2_image_predictor.py",
    )
    missing = [str(item) for item in required if not item.is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete upstream EdgeTAM source tree: {missing}")
    if sha256_file(root / "LICENSE") != OFFICIAL_LICENSE_SHA256:
        raise ValueError("Upstream checkout LICENSE does not match the pinned source")

    runtime_files = _upstream_runtime_files(root)
    digest = hashlib.sha256()
    for file_path in runtime_files:
        relative = file_path.relative_to(root).as_posix().encode("utf-8")
        contents = file_path.read_bytes().replace(b"\r\n", b"\n")
        digest.update(relative)
        digest.update(b"\0")
        digest.update(contents)
        digest.update(b"\0")
    actual = digest.hexdigest()
    if (
        len(runtime_files) != UPSTREAM_RUNTIME_FILE_COUNT
        or actual != UPSTREAM_RUNTIME_TREE_SHA256
    ):
        raise ValueError(
            "Upstream runtime tree does not match "
            f"facebookresearch/EdgeTAM@{OFFICIAL_SOURCE_COMMIT}: "
            f"files={len(runtime_files)}, sha256={actual}"
        )
    return {
        "directory": str(root),
        "commit": OFFICIAL_SOURCE_COMMIT,
        "runtime_files": len(runtime_files),
        "runtime_tree_sha256": actual,
    }


def _max_abs_diff(actual: torch.Tensor, expected: torch.Tensor) -> float:
    if actual.shape != expected.shape:
        raise AssertionError(
            f"Tensor shapes differ: actual={tuple(actual.shape)}, expected={tuple(expected.shape)}"
        )
    if actual.numel() == 0:
        return 0.0
    return float(
        (actual.detach().cpu().float() - expected.detach().cpu().float()).abs().max()
    )


def _assert_close(
    label: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    tolerance: float = TENSOR_ATOL,
    inclusive: bool = False,
) -> float:
    difference = _max_abs_diff(actual, expected)
    passed = difference <= tolerance if inclusive else difference < tolerance
    print(
        f"  {label:42s} max_abs_diff={difference:.6e} "
        f"limit={'<=' if inclusive else '<'}{tolerance:.1e} "
        f"{'PASS' if passed else 'FAIL'}",
        flush=True,
    )
    if not passed:
        raise AssertionError(
            f"{label} max_abs_diff {difference:.6e} exceeds "
            f"{'<=' if inclusive else '<'} {tolerance:.1e}"
        )
    return difference


def _assert_exact(label: str, actual: torch.Tensor, expected: torch.Tensor) -> float:
    if actual.shape != expected.shape:
        raise AssertionError(
            f"{label} shapes differ: {tuple(actual.shape)} != {tuple(expected.shape)}"
        )
    difference = _max_abs_diff(actual, expected)
    passed = torch.equal(actual.detach().cpu(), expected.detach().cpu())
    print(
        f"  {label:42s} max_abs_diff={difference:.6e} exact "
        f"{'PASS' if passed else 'FAIL'}",
        flush=True,
    )
    if not passed:
        raise AssertionError(
            f"{label} is not bit-exact (max_abs_diff={difference:.6e})"
        )
    return difference


def _binary_mask_iou(actual: torch.Tensor, expected: torch.Tensor) -> float:
    actual_mask = actual.detach().cpu().bool()
    expected_mask = expected.detach().cpu().bool()
    if actual_mask.shape != expected_mask.shape:
        raise AssertionError(
            f"Binary mask shapes differ: {tuple(actual_mask.shape)} != {tuple(expected_mask.shape)}"
        )
    intersection = torch.logical_and(actual_mask, expected_mask).sum().item()
    union = torch.logical_or(actual_mask, expected_mask).sum().item()
    return 1.0 if union == 0 else float(intersection / union)


def _assert_mask_iou(label: str, actual: torch.Tensor, expected: torch.Tensor) -> float:
    iou = _binary_mask_iou(actual, expected)
    passed = iou >= MASK_IOU_MIN
    print(
        f"  {label:42s} binary_iou={iou:.6f} "
        f"limit>={MASK_IOU_MIN:.2f} {'PASS' if passed else 'FAIL'}",
        flush=True,
    )
    if not passed:
        raise AssertionError(
            f"{label} binary IoU {iou:.6f} is below {MASK_IOU_MIN:.2f}"
        )
    return iou


def _load_upstream_model(
    upstream_root: Path, state_dict: Mapping[str, torch.Tensor]
) -> tuple[Any, Any]:
    """Instantiate official code without pretrained downloads, then strict-load."""
    import timm.models

    existing_sam2 = sys.modules.get("sam2")
    if existing_sam2 is not None:
        existing_file = getattr(existing_sam2, "__file__", None)
        if (
            existing_file is None
            or upstream_root not in Path(existing_file).resolve().parents
        ):
            raise RuntimeError(
                "A different sam2 package is already imported; run parity in a fresh process"
            )
    if str(upstream_root) not in sys.path:
        sys.path.insert(0, str(upstream_root))

    real_create_model = timm.models.create_model

    def create_local_backbone(*args, **kwargs):
        kwargs["pretrained"] = False
        return real_create_model(*args, **kwargs)

    try:
        with patch.object(timm.models, "create_model", create_local_backbone):
            build_sam2 = getattr(
                importlib.import_module("sam2.build_sam"), "build_sam2"
            )

            upstream = build_sam2(
                "configs/edgetam.yaml",
                ckpt_path=None,
                device="cpu",
                mode="eval",
                apply_postprocessing=True,
            )
        predictor_cls = getattr(
            importlib.import_module("sam2.sam2_image_predictor"),
            "SAM2ImagePredictor",
        )
    except ImportError as exc:
        raise RuntimeError(
            "Loading the pinned upstream checkout requires its Apache-2.0 runtime "
            "dependencies (hydra-core, omegaconf, and timm)."
        ) from exc

    module_file = getattr(sys.modules["sam2"], "__file__", None)
    if module_file is None:
        raise RuntimeError("The imported sam2 package has no verifiable source path")
    imported_source = Path(module_file).resolve()
    if upstream_root not in imported_source.parents:
        raise RuntimeError(
            f"Imported sam2 from {imported_source}, not the requested pinned tree {upstream_root}"
        )
    for name, module in tuple(sys.modules.items()):
        if name != "sam2" and not name.startswith("sam2."):
            continue
        source = getattr(module, "__file__", None)
        if source is not None and upstream_root not in Path(source).resolve().parents:
            raise RuntimeError(
                f"Imported {name} from {source}, not the requested pinned source tree"
            )
    incompatible = upstream.load_state_dict(dict(state_dict), strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise AssertionError(f"Strict upstream model load failed: {incompatible}")
    upstream.eval()
    return upstream, predictor_cls(upstream)


def _load_libre_model(mirror: Path) -> Any:
    """Construct the actual LibreEdgeTAM adapter entirely from the local mirror."""
    from libreyolo.models.sam.edgetam import LibreEdgeTAM

    libre = object.__new__(LibreEdgeTAM)
    model, processor = libre._load_pretrained(str(mirror))
    libre.model = model.to("cpu").eval()
    libre.processor = processor
    libre.device = torch.device("cpu")
    libre._model_dtype = next(model.parameters()).dtype
    libre._default_multimask = False
    libre.names = {0: "object"}
    libre.size = "edge"
    libre.task = "segment"
    libre._clear_image_state()
    return libre


def _upstream_image_embeddings(
    upstream: Any, pixel_values: torch.Tensor, feature_sizes: list[tuple[int, int]]
) -> list[torch.Tensor]:
    backbone_output = upstream.forward_image(pixel_values)
    _, vision_features, _, _ = upstream._prepare_backbone_features(backbone_output)
    if upstream.directly_add_no_mem_embed:
        vision_features[-1] = vision_features[-1] + upstream.no_mem_embed
    batch_size = pixel_values.shape[0]
    return [
        feature.permute(1, 2, 0).view(batch_size, -1, *size)
        for feature, size in zip(vision_features[::-1], feature_sizes[::-1])
    ][::-1]


def _compare_feature_lists(
    label: str, actual: list[torch.Tensor], expected: list[torch.Tensor]
) -> dict[str, float]:
    if len(actual) != len(expected):
        raise AssertionError(
            f"{label} feature counts differ: {len(actual)} != {len(expected)}"
        )
    return {
        f"level_{index}": _assert_close(
            f"{label} FPN level {index}", actual_tensor, expected_tensor
        )
        for index, (actual_tensor, expected_tensor) in enumerate(zip(actual, expected))
    }


def _preprocess_checks(libre: Any, upstream_predictor: Any) -> dict[str, float]:
    raw = np.arange(31 * 37 * 3, dtype=np.uint8).reshape(31, 37, 3)
    image = Image.fromarray(raw)
    point = [18.0, 15.0]
    box = [4.0, 5.0, 30.0, 25.0]

    point_encoding = libre._encode(image, points=[[point]], labels=[[1]])
    box_encoding = libre._encode(image, boxes=[box])
    upstream_pixels = upstream_predictor._transforms(image).unsqueeze(0)
    pixel_difference = _assert_exact(
        "LibreEdgeTAM official square pixels",
        point_encoding["pixel_values"],
        upstream_pixels,
    )
    _assert_exact(
        "box path uses identical square pixels",
        box_encoding["pixel_values"],
        upstream_pixels,
    )

    original_size = (31, 37)
    upstream_points = upstream_predictor._transforms.transform_coords(
        torch.tensor([[point]], dtype=torch.float32),
        normalize=True,
        orig_hw=original_size,
    )
    point_difference = _assert_exact(
        "point coordinate transform",
        point_encoding["input_points"][0],
        upstream_points,
    )
    upstream_boxes = upstream_predictor._transforms.transform_boxes(
        torch.tensor([box], dtype=torch.float32),
        normalize=True,
        orig_hw=original_size,
    ).reshape(-1, 4)
    box_difference = _assert_exact(
        "box coordinate transform",
        box_encoding["input_boxes"][0],
        upstream_boxes,
    )
    return {
        "pixel_max_abs_diff": pixel_difference,
        "point_coord_max_abs_diff": point_difference,
        "box_coord_max_abs_diff": box_difference,
    }


def _prompt_cases(width: int, height: int) -> dict[str, dict[str, Any]]:
    positive = [0.52 * width, 0.64 * height]
    negative = [0.50 * width, 0.30 * height]
    box = [0.32 * width, 0.50 * height, 0.86 * width, 0.94 * height]
    return {
        "point": {
            "encode": {"points": [[positive]], "labels": [[1]]},
            "libre": {"points": positive, "labels": [1]},
            "upstream": {
                "point_coords": np.asarray([positive], dtype=np.float32),
                "point_labels": np.asarray([1], dtype=np.int32),
            },
        },
        "box": {
            "encode": {"boxes": [box]},
            "libre": {"bboxes": box},
            "upstream": {"box": np.asarray(box, dtype=np.float32)},
        },
        "grouped_positive_negative": {
            "encode": {"points": [[positive, negative]], "labels": [[1, 0]]},
            "libre": {"points": [[positive, negative]], "labels": [1, 0]},
            "upstream": {
                "point_coords": np.asarray([positive, negative], dtype=np.float32),
                "point_labels": np.asarray([1, 0], dtype=np.int32),
            },
        },
    }


def _upstream_prompt_embeddings(upstream: Any, encoding: Mapping[str, torch.Tensor]):
    if "input_boxes" in encoding:
        coordinates = encoding["input_boxes"].squeeze(1).reshape(-1, 2, 2)
        labels = torch.tensor([[2, 3]], dtype=torch.int32)
    else:
        coordinates = encoding["input_points"].squeeze(1)
        labels = encoding["input_labels"].squeeze(1)
    return upstream.sam_prompt_encoder(
        points=(coordinates, labels),
        boxes=None,
        masks=None,
    )


def _hf_prompt_embeddings(hf_model: Any, encoding: Mapping[str, torch.Tensor]):
    return hf_model.get_prompt_embeddings(
        input_points=encoding.get("input_points"),
        input_labels=encoding.get("input_labels"),
        input_boxes=encoding.get("input_boxes"),
    )


def _runtime_checks(
    upstream: Any,
    upstream_predictor: Any,
    libre: Any,
    image_path: Path,
) -> dict[str, Any]:
    hf_model = libre.model
    feature_sizes = list(upstream_predictor._bb_feat_sizes)
    image_metrics: dict[str, Any] = {}

    _stage(5, "Compare image encoder/FPN embeddings on zeros and fixed arange")
    deterministic_inputs = {
        "zeros": torch.zeros((1, 3, 1024, 1024), dtype=torch.float32),
        "arange": (
            torch.arange(3 * 1024 * 1024, dtype=torch.float32)
            .remainder(256)
            .div(255)
            .view(1, 3, 1024, 1024)
        ),
    }
    with torch.inference_mode():
        for input_name, pixels in deterministic_inputs.items():
            upstream_features = _upstream_image_embeddings(
                upstream, pixels, feature_sizes
            )
            hf_features = hf_model.get_image_embeddings(pixels)
            image_metrics[input_name] = _compare_feature_lists(
                input_name, hf_features, upstream_features
            )

    if not image_path.is_file():
        raise FileNotFoundError(f"Parity image does not exist: {image_path}")
    with Image.open(image_path) as opened:
        image = opened.convert("RGB")
    width, height = image.size
    cases = _prompt_cases(width, height)
    base_encoding = libre._encode(image)
    with torch.inference_mode():
        upstream_features = _upstream_image_embeddings(
            upstream, base_encoding["pixel_values"], feature_sizes
        )
        hf_features = hf_model.get_image_embeddings(base_encoding["pixel_values"])
    image_metrics["real_image"] = _compare_feature_lists(
        "real image", hf_features, upstream_features
    )

    _stage(6, "Compare point, box, and grouped prompt embeddings")
    prompt_metrics: dict[str, Any] = {}
    decoder_inputs: dict[str, Any] = {}
    position_difference = _assert_close(
        "dense prompt positional embedding",
        hf_model.get_image_wide_positional_embeddings(),
        upstream.sam_prompt_encoder.get_dense_pe(),
    )
    prompt_metrics["position_max_abs_diff"] = position_difference
    for case_name, case in cases.items():
        encoding = libre._encode(image, **case["encode"])
        upstream_sparse, upstream_dense = _upstream_prompt_embeddings(
            upstream, encoding
        )
        hf_sparse, hf_dense = _hf_prompt_embeddings(hf_model, encoding)
        sparse_difference = _assert_close(
            f"{case_name} sparse prompt embedding",
            hf_sparse.squeeze(1),
            upstream_sparse,
        )
        dense_difference = _assert_close(
            f"{case_name} dense prompt embedding",
            hf_dense,
            upstream_dense,
        )
        prompt_metrics[case_name] = {
            "sparse_max_abs_diff": sparse_difference,
            "dense_max_abs_diff": dense_difference,
        }
        decoder_inputs[case_name] = (
            upstream_sparse,
            upstream_dense,
            hf_sparse,
            hf_dense,
        )

    _stage(7, "Compare mask-decoder logits and IoU/quality scores")
    decoder_metrics: dict[str, Any] = {}
    with torch.inference_mode():
        for case_name, values in decoder_inputs.items():
            upstream_sparse, upstream_dense, hf_sparse, hf_dense = values
            upstream_masks, upstream_iou, _, _ = upstream.sam_mask_decoder(
                image_embeddings=upstream_features[-1],
                image_pe=upstream.sam_prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=upstream_sparse,
                dense_prompt_embeddings=upstream_dense,
                multimask_output=True,
                repeat_image=False,
                high_res_features=upstream_features[:-1],
            )
            hf_masks, hf_iou, _, _ = hf_model.mask_decoder(
                image_embeddings=hf_features[-1],
                image_positional_embeddings=hf_model.get_image_wide_positional_embeddings(),
                sparse_prompt_embeddings=hf_sparse,
                dense_prompt_embeddings=hf_dense,
                multimask_output=True,
                high_resolution_features=hf_features[:-1],
            )
            mask_difference = _assert_close(
                f"{case_name} mask decoder logits",
                hf_masks.squeeze(1),
                upstream_masks,
            )
            iou_difference = _assert_close(
                f"{case_name} decoder quality scores",
                hf_iou.squeeze(1),
                upstream_iou,
            )
            decoder_metrics[case_name] = {
                "mask_logits_max_abs_diff": mask_difference,
                "quality_max_abs_diff": iou_difference,
            }

    _stage(8, "Compare final public-API masks for three real prompt cases")
    upstream_predictor._features = {
        "image_embed": upstream_features[-1],
        "high_res_feats": upstream_features[:-1],
    }
    upstream_predictor._orig_hw = [(height, width)]
    upstream_predictor._is_image_set = True
    libre._image = image
    libre._image_path = image_path
    libre._image_embeddings = hf_features

    mask_ious: dict[str, float] = {}
    for case_name, case in cases.items():
        upstream_masks, _, _ = upstream_predictor.predict(
            **case["upstream"],
            multimask_output=False,
            return_logits=False,
        )
        libre_result = libre.predict(
            **case["libre"],
            conf=0.0,
            multimask=False,
        )
        if libre_result.masks is None or len(libre_result.masks) != 1:
            raise AssertionError(
                f"LibreEdgeTAM returned {0 if libre_result.masks is None else len(libre_result.masks)} "
                f"masks for {case_name}; expected exactly one"
            )
        if upstream_masks.shape[0] != 1:
            raise AssertionError(
                f"Upstream returned {upstream_masks.shape[0]} masks for {case_name}; expected one"
            )
        mask_ious[case_name] = _assert_mask_iou(
            f"{case_name} final mask",
            libre_result.masks.data[0],
            torch.from_numpy(upstream_masks[0]),
        )

    return {
        "image_encoder": image_metrics,
        "prompt_encoder": prompt_metrics,
        "mask_decoder": decoder_metrics,
        "final_mask_iou": mask_ious,
    }


def run_parity(
    checkpoint: str | Path,
    mirror_dir: str | Path,
    upstream_dir: str | Path,
    *,
    reference_dir: str | Path | None = None,
    image: str | Path | None = None,
) -> dict[str, Any]:
    """Execute every artifact and native-runtime comparison locally."""
    _force_offline()
    checkpoint_path = Path(checkpoint).resolve()
    mirror = Path(mirror_dir).resolve()
    upstream_root = Path(upstream_dir).resolve()
    default_image = (
        Path(__file__).resolve().parents[1] / "tests" / "data" / "parkour.jpg"
    )
    image_path = Path(image).resolve() if image is not None else default_image

    _stage(1, "Validate pinned upstream source, checkpoint, and mirror payload")
    upstream_summary = validate_upstream_source(upstream_root)
    mirror_summary = validate_payload(mirror)
    checkpoint_digest = sha256_file(checkpoint_path)
    if checkpoint_digest != OFFICIAL_CHECKPOINT_SHA256:
        raise ValueError(
            "Official checkpoint SHA-256 mismatch: "
            f"expected {OFFICIAL_CHECKPOINT_SHA256}, got {checkpoint_digest}"
        )

    _stage(2, "Safe-load official weights and replay the pinned conversion")
    upstream_state = load_official_state_dict(checkpoint_path)
    converted_state = convert_state_dict(upstream_state)
    mirror_state = load_safetensors_state(mirror / "model.safetensors")
    exact_mirror_tensors = compare_state_dicts(
        converted_state,
        mirror_state,
        actual_label="converted official checkpoint",
        expected_label="LibreEdgeTAM mirror",
    )
    model_state = strict_load_model(mirror / "config.json", mirror_state)
    strict_loaded_tensors = compare_state_dicts(
        model_state,
        mirror_state,
        actual_label="strict-loaded EdgeTamVideoModel",
        expected_label="LibreEdgeTAM mirror",
    )
    del converted_state, mirror_state, model_state

    _stage(3, "Instantiate pinned native EdgeTAM and actual LibreEdgeTAM locally")
    upstream, upstream_predictor = _load_upstream_model(upstream_root, upstream_state)
    del upstream_state
    libre = _load_libre_model(mirror)

    _stage(4, "Compare actual LibreEdgeTAM preprocessing and prompt geometry")
    preprocessing_metrics = _preprocess_checks(libre, upstream_predictor)
    runtime_metrics = _runtime_checks(
        upstream,
        upstream_predictor,
        libre,
        image_path,
    )

    if reference_dir is None:
        _stage(9, "Pinned conversion-reference comparison not requested")
        exact_reference_tensors: int | None = None
    else:
        _stage(9, "Compare every mirror tensor with the pinned local reference")
        reference = validate_reference_dir(reference_dir)
        mirror_state = load_safetensors_state(mirror / "model.safetensors")
        reference_state = load_safetensors_state(reference / "model.safetensors")
        exact_reference_tensors = compare_state_dicts(
            mirror_state,
            reference_state,
            actual_label="LibreEdgeTAM mirror",
            expected_label="pinned reference",
        )

    return {
        "status": "runtime parity PASS",
        "network_access": "disabled",
        "official_revision": OFFICIAL_REVISION,
        "official_source_commit": OFFICIAL_SOURCE_COMMIT,
        "official_checkpoint_sha256": checkpoint_digest,
        "upstream_source": upstream_summary,
        "exact_mirror_tensors": exact_mirror_tensors,
        "strict_loaded_tensors": strict_loaded_tensors,
        "reference_revision": REFERENCE_REVISION if reference_dir is not None else None,
        "exact_reference_tensors": exact_reference_tensors,
        "mirror_model_sha256": mirror_summary["model_sha256"],
        "parity_image": str(image_path),
        "preprocessing": preprocessing_metrics,
        **runtime_metrics,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Local official edgetam.pt (must match the pinned SHA-256).",
    )
    parser.add_argument(
        "--mirror-dir",
        type=Path,
        required=True,
        help="Local generated LibreEdgeTAM mirror payload.",
    )
    parser.add_argument(
        "--upstream-dir",
        type=Path,
        required=True,
        help="Local facebookresearch/EdgeTAM checkout/archive at the pinned commit.",
    )
    parser.add_argument(
        "--reference-dir",
        type=Path,
        help="Optional local pinned EdgeTAM-hf conversion-reference snapshot.",
    )
    parser.add_argument(
        "--image",
        type=Path,
        help="Optional local parity image (default: test fixture tests/data/parkour.jpg).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_parity(
        args.checkpoint,
        args.mirror_dir,
        args.upstream_dir,
        reference_dir=args.reference_dir,
        image=args.image,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
