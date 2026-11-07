"""Convert released InstantHMR ONNX graphs into LibreYOLO PyTorch checkpoints.

InstantHMR publishes its trained networks as ONNX only. This script rebuilds
the PyTorch state dict from one network graph:

- Initializers that kept their parameter names load directly.
- Linear weights the exporter pre-transposed into ``MatMul`` constants are
  transposed back and named by the bias that follows them.
- Every backbone ``Conv`` carries a BatchNorm folded into it. Convs are matched
  to the backbone's convolutions in execution order, the fused weight becomes
  the conv weight and the BatchNorm is rebuilt so that, in eval mode, it adds
  exactly the fused bias. With ``--calib`` the BatchNorm running statistics are
  set from real person crops (and gamma, beta adjusted so eval output is
  unchanged), which gives train-mode BatchNorm sensible starting statistics
  for fine-tuning.
- Three constants the exporter folded are unfolded: the two tanh range widths
  (``hi = lo + 2 * folded``) and the 2D keypoint bin centres.
- The decode head's index tables are read from the graph's constant ``Gather``
  indices.

The MHR body constants are built from Meta's MHR rig (Apache-2.0, fetched by
LibreYOLO) plus InstantHMR's 70-keypoint landmark readout
(``training/assets/mhr_landmarks70.npz`` in the upstream repository).

The converted model is checked against onnxruntime on the source graph before
anything is written.

    python weights/convert_instanthmr_weights.py \\
        --onnx instanthmr-medium.onnx --size m \\
        --landmarks <InstantHMR>/training/assets/mhr_landmarks70.npz \\
        --calib ~/datasets/coco128 --output weights/LibreInstantHMRm-mesh.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

try:
    from _conversion_utils import (
        add_repo_root_to_path,
        save_checkpoint,
        wrap_libreyolo_checkpoint,
    )
except ModuleNotFoundError:  # imported as weights.convert_instanthmr_weights
    from weights._conversion_utils import (
        add_repo_root_to_path,
        save_checkpoint,
        wrap_libreyolo_checkpoint,
    )

UPSTREAM_REPO = "https://github.com/Naturalpad-studio/InstantHMR"
UPSTREAM_COMMIT = "5437400d4b25c1972821e51916fac34bd96acce6"
WEIGHTS_REPO = "momolesang/InstantHMR"
WEIGHTS_REVISION = "07988f51c8c8908cfc7bb44fc18f645c9551b0bf"

# variant -> (upstream file, licence of the weights)
SOURCES = {
    ("n", None): ("instanthmr-nano.onnx", "CC-BY-NC-4.0"),
    ("t", None): ("instanthmr-tiny.onnx", "CC-BY-NC-4.0"),
    ("s", None): ("instanthmr-small.onnx", "CC-BY-NC-4.0"),
    ("m", None): ("instanthmr-medium.onnx", "CC-BY-NC-4.0"),
    ("m", "commercial"): ("instanthmr-medium-commercial.onnx", "Apache-2.0"),
}


def onnx_state_dict(
    onnx_path: str | Path, net: torch.nn.Module
) -> dict[str, torch.Tensor]:
    """Rebuild ``net``'s state dict (body excluded) from an InstantHMR network graph."""
    import onnx
    from onnx import numpy_helper

    graph = onnx.load(str(onnx_path)).graph
    inits = {
        i.name: torch.from_numpy(numpy_helper.to_array(i).copy())
        for i in graph.initializer
    }
    consumers: dict[str, list] = {}
    for node in graph.node:
        for name in node.input:
            consumers.setdefault(name, []).append(node)
    used: set[str] = set()
    sd: dict[str, torch.Tensor] = {}

    for name, value in inits.items():
        if name.startswith("net."):
            sd[name[len("net.") :]] = value
            used.add(name)

    for node in graph.node:
        if node.op_type == "MatMul" and node.input[1].startswith("onnx::MatMul"):
            adds = [c for c in consumers[node.output[0]] if c.op_type == "Add"]
            biases = [
                i
                for a in adds
                for i in a.input
                if i.startswith("net.") and i.endswith("bias")
            ]
            if len(biases) != 1:
                raise RuntimeError(f"cannot name the weight of {node.name}")
            sd[biases[0][len("net.") : -len("bias")] + "weight"] = inits[
                node.input[1]
            ].T.contiguous()
            used.add(node.input[1])

    convs = [n for n in graph.node if n.op_type == "Conv"]
    order: list[str] = []
    hooks = [
        m.register_forward_hook(lambda _m, _i, _o, name=name: order.append(name))
        for name, m in net.named_modules()
        if isinstance(m, torch.nn.Conv2d)
    ]
    with torch.no_grad():
        net(torch.zeros(1, 3, net.image_size, net.image_size), torch.zeros(1, 3))
    for h in hooks:
        h.remove()
    if len(order) != len(convs):
        raise RuntimeError(
            f"{len(convs)} convs in the graph, {len(order)} in the model"
        )
    modules = dict(net.named_modules())
    for name, node in zip(order, convs):
        weight, bias = inits[node.input[1]], inits[node.input[2]]
        if tuple(weight.shape) != tuple(modules[name].weight.shape):
            raise RuntimeError(f"conv order mismatch at {name}")
        parent = name.rsplit(".", 1)[0]
        bn = modules[parent + ".bn"]
        channels = weight.shape[0]
        sd[name + ".weight"] = weight
        sd[parent + ".bn.weight"] = torch.ones(channels)
        sd[parent + ".bn.bias"] = bias
        sd[parent + ".bn.running_mean"] = torch.zeros(channels)
        sd[parent + ".bn.running_var"] = torch.full((channels,), 1.0 - bn.eps)
        sd[parent + ".bn.num_batches_tracked"] = torch.tensor(0)
        used.update(node.input[1:3])

    for name in [n for n in inits if n not in used]:
        value = inits[name]
        if value.shape == (6,):
            sd["cont_head.flex_hi"] = sd["cont_head.flex_lo"] + 2 * value
        elif value.shape == (28,):
            sd["cont_head.coeff_hi"] = sd["cont_head.coeff_lo"] + 2 * value
        elif value.shape == net.kp2d_bin_centers.shape:
            sd["kp2d_bin_centers"] = value
        else:
            raise RuntimeError(f"unmapped initializer {name} {tuple(value.shape)}")

    constants = {
        n.output[0]: numpy_helper.to_array(n.attribute[0].t)
        for n in graph.node
        if n.op_type == "Constant"
        and n.attribute
        and n.attribute[0].type == onnx.AttributeProto.TENSOR
    }
    tables: dict[int, list[np.ndarray]] = {}
    for node in graph.node:
        if (
            node.op_type == "Gather"
            and "/cont_head/" in node.name
            and node.input[1] in constants
        ):
            idx = constants[node.input[1]]
            if idx.ndim == 1 and idx.size > 6:
                tables.setdefault(idx.size, []).append(idx)
    for size, key in (
        (133, "body_scatter"),
        (54, "hand_cont_gather"),
        (27, "hand_param_scatter"),
    ):
        found = tables.get(size, [])
        if not found or any(not np.array_equal(found[0], f) for f in found):
            raise RuntimeError(f"could not read {key} from the graph")
        sd["cont_head." + key] = torch.from_numpy(found[0].astype(np.int64))
    return sd


def person_crops(dataset: Path, limit: int, input_size: int):
    """ImageNet-normalised person crops from a YOLO-format dataset (class 0 = person)."""
    import cv2

    from libreyolo.models.instanthmr.utils import default_focal, preprocess_person

    crops, cliffs = [], []
    for label in sorted(dataset.rglob("labels/**/*.txt")):
        image = next(
            (
                p
                for ext in (".jpg", ".png", ".jpeg")
                if (
                    p := Path(str(label).replace("/labels/", "/images/")).with_suffix(
                        ext
                    )
                ).exists()
            ),
            None,
        )
        if image is None:
            continue
        rows = [r.split() for r in label.read_text().splitlines() if r.strip()]
        boxes = [list(map(float, r[1:5])) for r in rows if r[0] == "0"]
        if not boxes:
            continue
        rgb = cv2.cvtColor(cv2.imread(str(image)), cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        focal = default_focal(h, w)
        for cx, cy, bw, bh in boxes:
            if bw * w < 32 or bh * h < 64:
                continue
            box = (
                (cx - bw / 2) * w,
                (cy - bh / 2) * h,
                (cx + bw / 2) * w,
                (cy + bh / 2) * h,
            )
            crop, cliff, _ = preprocess_person(rgb, box, focal, input_size)
            crops.append(crop)
            cliffs.append(cliff)
            if len(crops) >= limit:
                return torch.from_numpy(np.stack(crops)), torch.from_numpy(
                    np.stack(cliffs)
                )
    if not crops:
        raise RuntimeError(f"no person boxes found under {dataset}")
    return torch.from_numpy(np.stack(crops)), torch.from_numpy(np.stack(cliffs))


@torch.no_grad()
def calibrate_batchnorm(
    net: torch.nn.Module, images: torch.Tensor, cliffs: torch.Tensor
) -> None:
    """Give each rebuilt BatchNorm real running statistics without changing eval output."""
    stats: dict[str, list[torch.Tensor]] = {}
    hooks = []
    for name, m in net.named_modules():
        if isinstance(m, torch.nn.BatchNorm2d):

            def hook(_m, inputs, _o, name=name):
                x = inputs[0].transpose(0, 1).flatten(1).double()
                s = stats.setdefault(
                    name, [torch.zeros(x.shape[0], dtype=torch.float64)] * 3
                )
                stats[name] = [
                    s[0] + x.shape[1],
                    s[1] + x.sum(1),
                    s[2] + (x * x).sum(1),
                ]

            hooks.append(m.register_forward_hook(hook))
    net.eval()
    for i in range(0, len(images), 16):
        net(images[i : i + 16], cliffs[i : i + 16])
    for h in hooks:
        h.remove()
    modules = dict(net.named_modules())
    for name, (count, total, squares) in stats.items():
        bn = modules[name]
        mean = total / count
        var = (squares / count - mean * mean).clamp_min(1e-6)
        # In eval mode the rebuilt BN adds its bias b; keep that exactly:
        # gamma * (y - mean) / sqrt(var + eps) + beta == y + b.
        bias = bn.bias.double()
        bn.running_mean.copy_(mean)
        bn.running_var.copy_(var)
        bn.weight.copy_(torch.sqrt(var + bn.eps))
        bn.bias.copy_(bias + mean)


@torch.no_grad()
def check_against_onnx(
    net, onnx_path: str | Path, atol: float = 2e-3
) -> dict[str, float]:
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    gen = torch.Generator().manual_seed(0)
    image = torch.randn(4, 3, net.image_size, net.image_size, generator=gen)
    cliff = torch.randn(4, 3, generator=gen) * 0.3
    names = ["mhr_params", "shape_params", "cam_trans", "joints_2d"]
    want = dict(
        zip(
            names,
            session.run(names, {"image": image.numpy(), "cliff_cond": cliff.numpy()}),
        )
    )
    got = net(image, cliff)
    worst = {k: float(np.abs(got[k].numpy() - want[k]).max()) for k in names}
    if max(worst.values()) > atol:
        raise RuntimeError(f"converted model disagrees with onnxruntime: {worst}")
    return worst


def convert(
    onnx_path: str | Path,
    size: str,
    output: str | Path,
    landmarks: str | Path,
    variant: str | None = None,
    calib: str | Path | None = None,
    calib_limit: int = 256,
    mhr_path: str | Path | None = None,
) -> Path:
    add_repo_root_to_path()
    from libreyolo.models.instanthmr.body import MHRBody, mesh_checkpoint_metadata
    from libreyolo.models.instanthmr.nn import InstantHMRNet
    from libreyolo.models.sam3dbody.mhr_body import ensure_mhr_model

    if (size, variant) not in SOURCES:
        raise ValueError(f"unknown InstantHMR release {(size, variant)}")
    source_file, weight_license = SOURCES[(size, variant)]

    net = InstantHMRNet(size).eval()
    sd = onnx_state_dict(onnx_path, net)
    lm = np.load(landmarks)
    body = MHRBody.from_rig(
        ensure_mhr_model(mhr_path), lm["vert_idx"], lm["W_joint"], lm["W_vert"]
    )
    sd.update({"body." + k: v for k, v in body.state_dict().items()})
    net.load_state_dict(sd, strict=True)
    print("parity before calibration (max abs):", check_against_onnx(net, onnx_path))
    if calib is not None:
        images, cliffs = person_crops(
            Path(calib).expanduser(), calib_limit, net.image_size
        )
        calibrate_batchnorm(net, images, cliffs)
        print(f"calibrated BatchNorm on {len(images)} person crops")
        print("parity after calibration (max abs):", check_against_onnx(net, onnx_path))

    wrapped = wrap_libreyolo_checkpoint(
        net.state_dict(),
        model_family="instanthmr",
        size=size,
        task="mesh",
        nc=1,
        names={0: "person"},
        imgsz=net.image_size,
    )
    wrapped.update(
        source_repo=UPSTREAM_REPO,
        source_revision=UPSTREAM_COMMIT,
        weights_repo=WEIGHTS_REPO,
        weights_revision=WEIGHTS_REVISION,
        weights_file=source_file,
        license=weight_license,
        training_labels="SAM 3D Body dataset annotations (SAM License)",
        cliff_focal=True,
        batchnorm_calibrated=calib is not None,
    )
    wrapped.update(mesh_checkpoint_metadata(net.body))
    if variant is not None:
        wrapped["weight_variant"] = variant

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    save_checkpoint(wrapped, temporary)
    temporary.replace(output)
    print(f"Wrote {output}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--onnx", required=True, help="InstantHMR network graph")
    parser.add_argument("--size", required=True, choices=["n", "t", "s", "m"])
    parser.add_argument("--variant", choices=["commercial"], default=None)
    parser.add_argument(
        "--landmarks",
        required=True,
        help="InstantHMR training/assets/mhr_landmarks70.npz",
    )
    parser.add_argument(
        "--calib",
        default=None,
        help="YOLO-format dataset with person boxes, for BatchNorm statistics",
    )
    parser.add_argument("--calib-limit", type=int, default=256)
    parser.add_argument(
        "--mhr", default=None, help="MHR TorchScript rig (default: LibreYOLO cache)"
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    convert(
        args.onnx,
        args.size,
        args.output,
        args.landmarks,
        variant=args.variant,
        calib=args.calib,
        calib_limit=args.calib_limit,
        mhr_path=args.mhr,
    )
