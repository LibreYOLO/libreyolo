"""Private worker for an independently installed 3D-MOOD checkout.

The optional upstream package is Apache-2.0. Its documented dependency on
``vis4d_cuda_ops`` is not used because that extension has no declared license.
Image inference uses Vis4D's portable PyTorch path on CPU and LibreYOLO's
Apache-2.0 deformable-attention implementation on CUDA.
"""

from __future__ import annotations

import importlib.machinery
import json
import sys
import types
import warnings
from pathlib import Path

# Running this file directly places only its directory on sys.path. Add the
# package root so the permissive portable attention implementation is visible.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

_protocol = sys.stdout
sys.stdout = sys.stderr


def _warning(message, category, filename, lineno, file=None, line=None):
    print(f"{category.__name__}: {message}", file=sys.stderr, flush=True)


warnings.showwarning = _warning

from libreyolo.models.mood3d.runtime import _PREFIX, decode_array, encode_array


def _install_permissive_cuda_ops_interface():
    """Expose only interfaces needed to import Vis4D without its extension."""
    if "vis4d_cuda_ops" in sys.modules:
        return
    module = types.ModuleType("vis4d_cuda_ops")
    module.__spec__ = importlib.machinery.ModuleSpec("vis4d_cuda_ops", loader=None)

    def ms_deform_attn_forward(
        value,
        spatial_shapes,
        level_start_index,
        sampling_locations,
        attention_weights,
        im2col_step,
    ):
        del level_start_index, im2col_step
        from libreyolo.models.deformable_detr.ms_deform_attn import (
            ms_deform_attn_core_pytorch,
        )

        return ms_deform_attn_core_pytorch(
            value, spatial_shapes, sampling_locations, attention_weights
        )

    def unavailable(*args, **kwargs):
        raise NotImplementedError(
            "This 3D-MOOD image-inference worker does not provide training "
            "backward kernels or 3D-IoU CUDA kernels."
        )

    module.ms_deform_attn_forward = ms_deform_attn_forward
    module.ms_deform_attn_backward = unavailable
    module.iou_box3d = unavailable
    sys.modules["vis4d_cuda_ops"] = module


def _build(config):
    import torch
    from opendet3d.zoo.gdino3d.base.model import (
        get_gdino3d_hyperparams_cfg,
        get_gdino3d_swin_base_cfg,
        get_gdino3d_swin_tiny_cfg,
    )
    from vis4d.common.ckpt import load_model_checkpoint
    from vis4d.config import instantiate_classes

    params = get_gdino3d_hyperparams_cfg()
    factory = (
        get_gdino3d_swin_tiny_cfg
        if config["size"] == "t"
        else get_gdino3d_swin_base_cfg
    )
    model_config, _ = factory(
        params,
        pretrained=None,
        use_checkpoint=False,
    )
    # The complete detector checkpoint contains the backbone. Avoid a redundant
    # network download during construction.
    model_config.init_args.basemodel.init_args.pretrained = None
    roi = model_config.init_args.roi2det3d.init_args
    roi.nms = True
    roi.class_agnostic_nms = True
    roi.max_per_img = config["max_det"]
    roi.score_threshold = config["conf"]
    roi.iou_threshold = config["iou"]

    device = torch.device(config["device"])
    model = instantiate_classes(model_config).to(device)
    load_model_checkpoint(
        model,
        weights=config["checkpoint"],
        rev_keys=[(r"^model\.", ""), (r"^module\.", "")],
    )
    return model.eval(), device


def _preprocess(image, intrinsics):
    import numpy as np
    from opendet3d.data.transforms.pad import CenterPadImages, CenterPadIntrinsics
    from opendet3d.data.transforms.resize import GenResizeParameters
    from vis4d.data.transforms.base import compose
    from vis4d.data.transforms.normalize import NormalizeImages
    from vis4d.data.transforms.resize import ResizeImages, ResizeIntrinsics
    from vis4d.data.transforms.to_tensor import ToTensor

    batched = image.astype(np.float32, copy=False)[None]
    data = {
        "images": batched,
        "original_images": batched,
        "input_hw": tuple(image.shape[:2]),
        "original_hw": tuple(image.shape[:2]),
        "intrinsics": intrinsics,
        "original_intrinsics": intrinsics,
    }
    transform = compose(
        [
            GenResizeParameters(shape=(800, 1333)),
            ResizeImages(),
            ResizeIntrinsics(),
            NormalizeImages(),
            CenterPadImages(stride=1, shape=(800, 1333), update_input_hw=True),
            CenterPadIntrinsics(),
        ]
    )
    return ToTensor()(transform([data]))[0]


def _predict(model, device, request):
    import numpy as np
    import torch

    image = decode_array(request["image"])
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("The runtime requires an RGB uint8 image.")
    intrinsics = np.asarray(request["intrinsics"], dtype=np.float32)
    data = _preprocess(image, intrinsics)
    with torch.inference_mode():
        output = model(
            images=data["images"].to(device),
            input_hw=[data["input_hw"]],
            original_hw=[data["original_hw"]],
            intrinsics=data["intrinsics"].to(device)[None],
            padding=[data["padding"]],
            input_texts=[request["text"]],
        )
    fields = [
        output.boxes[0],
        output.boxes3d[0],
        output.scores[0],
        torch.as_tensor(output.class_ids[0]),
        output.depth_maps[0],
    ]
    return [
        encode_array(value.detach().cpu().float().numpy())
        for value in fields
    ]


def main():
    import torch

    _install_permissive_cuda_ops_interface()
    model = device = None
    for line in sys.stdin:
        request = {}
        try:
            request = json.loads(line)
            action = request["action"]
            if action == "load":
                if model is not None:
                    raise ValueError("The worker already has a model.")
                config = request["config"]
                device = torch.device(config["device"])
                if device.type == "cuda" and not torch.cuda.is_available():
                    raise RuntimeError("CUDA is unavailable in the runtime interpreter.")
                if device.type not in {"cpu", "cuda", "mps"}:
                    raise ValueError("Unsupported runtime device.")
                model, device = _build(config)
                reply = {"id": request["id"], "ready": True}
            elif action == "predict" and model is not None:
                reply = {
                    "id": request["id"],
                    "outputs": _predict(model, device, request),
                }
            else:
                raise ValueError("Invalid runtime request or model not loaded.")
        except Exception as exc:  # noqa: BLE001 - serialize worker failures
            frames = []
            traceback = exc.__traceback__
            while traceback:
                frames.append(
                    f"{traceback.tb_frame.f_code.co_filename}:"
                    f"{traceback.tb_lineno} {traceback.tb_frame.f_code.co_name}"
                )
                traceback = traceback.tb_next
            reply = {
                "id": request.get("id"),
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc).splitlines()[0],
                    "frames": frames,
                },
            }
        _protocol.write(_PREFIX + json.dumps(reply, allow_nan=False) + "\n")
        _protocol.flush()


if __name__ == "__main__":
    main()
