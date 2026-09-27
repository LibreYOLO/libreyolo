"""Private DetAny3D worker; no UniDepth implementation is bundled here.

The input recipe follows OpenDriveLab/DetAny3D's Apache-2.0 deploy.py.
CPU compatibility uses public xFormers interfaces and LibreYOLO's existing
permissive deformable-attention implementation. It is installed only in this
separate process. The upstream model and checkpoint remain unchanged.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import inspect
import json
import os
import sys
import types
import warnings
from pathlib import Path


def _bootstrap_package():
    """Expose this package without shadowing the external interpreter's deps."""
    package = Path(__file__).resolve().parents[2]
    initial = package / "__init__.py"
    existing = sys.modules.get("libreyolo")
    if existing is not None:
        if Path(existing.__file__).resolve() != initial:
            raise RuntimeError(
                "The external interpreter preloaded a different LibreYOLO package."
            )
        return
    spec = importlib.util.spec_from_file_location(
        "libreyolo", initial, submodule_search_locations=[str(package)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["libreyolo"] = module
    spec.loader.exec_module(module)


def _portable_attention(
    query, key, value, attn_bias=None, p=0.0, scale=None, *, op=None
):
    import torch
    from torch.nn import functional as F

    if op is not None:
        raise NotImplementedError(
            "The DetAny3D CPU worker cannot select a CUDA xFormers operator."
        )
    if attn_bias is not None and not isinstance(attn_bias, torch.Tensor):
        raise NotImplementedError(
            "Structured xFormers attention masks are not supported by this worker."
        )
    three = query.ndim == 3
    if three:
        query, key, value = (x.unsqueeze(2) for x in (query, key, value))
    if query.ndim != 4:
        raise ValueError("Expected xFormers [batch, sequence, heads, channels] inputs.")
    output = F.scaled_dot_product_attention(
        query.transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2),
        attn_mask=attn_bias,
        dropout_p=p,
        scale=scale,
    ).transpose(1, 2)
    return output.squeeze(2) if three else output


def _install_portable_ops():
    import xformers.ops as xops
    from xformers.ops import fmha

    xops.memory_efficient_attention = _portable_attention
    fmha.memory_efficient_attention = _portable_attention
    extension = types.ModuleType("mmcv._ext")
    extension.__spec__ = importlib.machinery.ModuleSpec("mmcv._ext", loader=None)

    def unavailable(*args, **kwargs):
        raise NotImplementedError(
            "This DetAny3D CPU worker supplies only portable multi-scale deformable attention."
        )

    def lookup(name):
        if name.startswith("__"):
            raise AttributeError(name)
        return unavailable

    def deform(
        value,
        spatial_shapes,
        level_start_index,
        sampling_locations,
        attention_weights,
        im2col_step=64,
    ):
        from libreyolo.models.deformable_detr.ms_deform_attn import (
            ms_deform_attn_core_pytorch,
        )

        return ms_deform_attn_core_pytorch(
            value, spatial_shapes, sampling_locations, attention_weights
        )

    extension.__getattr__ = lookup
    extension.ms_deform_attn_forward = deform
    extension.ms_deform_attn_backward = unavailable
    sys.modules["mmcv._ext"] = extension


def _build(config):
    import torch
    import yaml
    from box import Box

    root = Path(config["runtime_path"]).resolve()
    os.chdir(root)
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "GroundingDINO"))
    device = config["device"]
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    if device.type == "cpu":
        _install_portable_ops()
    from wrap_model import WrapModel

    cfg = Box(yaml.safe_load((root / "detect_anything/configs/demo.yaml").read_text()))
    # The full checkpoint contains both backbones; skip redundant initialization.
    cfg.model.checkpoint = None
    cfg.dino_path = None
    cfg.device = str(device)
    model = WrapModel(cfg)
    load_options = {"map_location": "cpu", "weights_only": True}
    if "mmap" in inspect.signature(torch.load).parameters:
        load_options["mmap"] = True
    checkpoint = torch.load(config["checkpoint"], **load_options)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    del checkpoint
    return {
        "model": model.to(device).eval(),
        "cfg": cfg,
        "device": device,
        "config": config,
        "grounding": None,
    }


def _preprocess(image, cfg, device):
    import numpy as np
    import torch
    from detect_anything.utils.transforms import ResizeLongestSide
    from torch.nn import functional as F

    height, width = image.shape[:2]
    if cfg.model.pad <= 0 or cfg.model.pad % 112:
        raise ValueError("DetAny3D canvas size must be a positive multiple of 112.")
    transform = ResizeLongestSide(cfg.model.pad)
    tensor = torch.from_numpy(image.copy()).permute(2, 0, 1).float()[None]
    resized = transform.apply_image_torch(tensor)
    rh, rw = resized.shape[-2:]
    ch, cw = rh // 14 * 14, rw // 14 * 14
    if ch == 0 or cw == 0:
        raise ValueError("This image aspect ratio produces an empty DetAny3D crop.")
    top, left = rh // 2 - ch // 2, rw // 2 - cw // 2
    cropped = resized[:, :, top : top + ch, left : left + cw]
    mean = cropped.new_tensor(cfg.dataset.pixel_mean).view(1, 3, 1, 1)
    std = cropped.new_tensor(cfg.dataset.pixel_std).view(1, 3, 1, 1)
    sam = F.pad((cropped - mean) / std, (0, cfg.model.pad - cw, 0, cfg.model.pad - ch))
    dino = (
        cropped / 255 - cropped.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    ) / cropped.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    patch = cfg.model.image_encoder.patch_size
    vit_hw = (
        (ch // patch, cw // patch)
        if cfg.model.vit_pad_mask
        else (cfg.model.pad // patch,) * 2
    )
    inputs = {
        "images": sam.to(device),
        "image_for_dino": dino.to(device),
        "images_shape": torch.tensor([[ch, cw]], dtype=torch.float32, device=device),
        "vit_pad_size": torch.tensor([vit_hw], device=device),
    }
    inverse = np.array(
        [
            [width / rw, 0, left * width / rw],
            [0, height / rh, top * height / rh],
            [0, 0, 1],
        ],
        np.float32,
    )
    return inputs, transform, inverse


def _ground(state, image, texts):
    import groundingdino
    import numpy as np
    import torch
    from groundingdino.datasets import transforms
    from groundingdino.util.inference import load_model, predict
    from PIL import Image
    from torchvision.ops import box_convert

    config = state["config"]
    root = Path(groundingdino.__file__).resolve().parent.parent
    if state["grounding"] is None:
        weights = Path(
            config["grounding_checkpoint"]
            or root / "weights/groundingdino_swinb_cogcoor.pth"
        )
        architecture = Path(
            config["grounding_config"]
            or root / "groundingdino/config/GroundingDINO_SwinB_cfg.py"
        )
        if not weights.is_file() or not architecture.is_file():
            raise FileNotFoundError(
                "Text prompts need the upstream GroundingDINO Swin-B config and checkpoint; set grounding_config and grounding_checkpoint."
            )
        state["grounding"] = load_model(
            str(architecture), str(weights), device=str(state["device"])
        )
    transform = transforms.Compose(
        [
            transforms.RandomResize([800], max_size=1333),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    transformed, _ = transform(Image.fromarray(image), None)
    caption = " . ".join(texts)
    boxes, scores, phrases = predict(
        model=state["grounding"],
        image=transformed,
        caption=caption,
        box_threshold=config["conf"],
        text_threshold=config["text_threshold"],
        remove_combined=False,
        device=str(state["device"]),
    )
    h, w = image.shape[:2]
    boxes = box_convert(
        boxes * torch.tensor([w, h, w, h]), in_fmt="cxcywh", out_fmt="xyxy"
    )
    return (
        boxes.to(torch.int).cpu().numpy().astype(np.float32),
        scores.cpu().numpy().astype(np.float32),
        phrases,
    )


def _predict(state, image, prompt):
    import numpy as np
    import torch
    from detect_anything.datasets.utils import points_img2cam, rotation_6d_to_matrix

    cfg, device = state["cfg"], state["device"]
    inputs, resize, inverse = _preprocess(image, cfg, device)
    labels, scores = [], []
    with torch.no_grad():
        if "points" in prompt:
            points = torch.tensor(prompt["points"], dtype=torch.int)
            inputs["point_coords"] = resize.apply_coords_torch(
                points, image.shape[:2]
            ).to(device)
            labels = [f"prompt_{i}" for i in range(len(points))]
            scores = [1.0] * len(points)
        else:
            boxes = list(prompt.get("bboxes", []))
            labels = [f"prompt_{i}" for i in range(len(boxes))]
            scores = [1.0] * len(boxes)
            if "text" in prompt:
                detected, confidence, phrases = _ground(state, image, prompt["text"])
                boxes.extend(detected.tolist())
                scores.extend(confidence.tolist())
                labels.extend(phrases)
            if boxes:
                inputs["boxes_coords"] = (
                    resize.apply_boxes_torch(torch.tensor(boxes), image.shape[:2])
                    .to(torch.int)
                    .to(device)
                )
        output = state["model"](inputs)
        k = output["pred_K"][0]
        if labels:
            image_centers = torch.cat(
                (
                    output["pred_center_2d"] * cfg.model.pad,
                    output["pred_bbox_3d_depth"].exp(),
                ),
                -1,
            )
            arrays = {
                "boxes": output["pred_bbox_2d"] * cfg.model.pad,
                "centers": points_img2cam(image_centers, k),
                "dimensions": output["pred_bbox_3d_dims"].exp(),
                "rotations": rotation_6d_to_matrix(output["pred_pose_6d"]),
            }
            arrays = {
                key: value.detach().float().cpu().numpy()
                for key, value in arrays.items()
            }
        else:
            arrays = {
                "boxes": np.empty((0, 4), np.float32),
                "centers": np.empty((0, 3), np.float32),
                "dimensions": np.empty((0, 3), np.float32),
                "rotations": np.empty((0, 3, 3), np.float32),
            }
        arrays.update(
            intrinsics=k.detach().float().cpu().numpy(),
            view_to_original=inverse,
            scores=np.asarray(scores, np.float32),
        )
    return arrays, labels


def main():
    protocol = sys.stdout
    sys.stdout = sys.stderr
    warnings.showwarning = (
        lambda message, category, filename, lineno, file=None, line=None: print(
            f"{category.__name__}: {message}", file=sys.stderr, flush=True
        )
    )
    _bootstrap_package()
    from libreyolo.models.runtime_worker import _PREFIX, decode_array, encode_array

    state = None
    for line in sys.stdin:
        request = json.loads(line)
        try:
            if request["action"] == "load":
                state = _build(request["config"])
                reply = {"device": str(state["device"])}
            elif request["action"] == "predict" and state is not None:
                arrays, labels = _predict(
                    state, decode_array(request["image"]), request["prompt"]
                )
                reply = {
                    "arrays": {
                        key: encode_array(value) for key, value in arrays.items()
                    },
                    "labels": labels,
                }
            else:
                raise ValueError("Invalid DetAny3D worker request.")
        except Exception as exc:  # noqa: BLE001 - external-runtime protocol boundary
            frames = []
            tb = exc.__traceback__
            while tb:
                frames.append(
                    {
                        "module": tb.tb_frame.f_globals.get("__name__"),
                        "function": tb.tb_frame.f_code.co_name,
                        "line": tb.tb_lineno,
                    }
                )
                tb = tb.tb_next
            reply = {"error": {"message": str(exc), "frames": frames}}
        reply["id"] = request["id"]
        protocol.write(_PREFIX + json.dumps(reply, allow_nan=False) + "\n")
        protocol.flush()


if __name__ == "__main__":
    main()
