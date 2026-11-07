"""Unit tests for the InstantHMR family.

A tiny synthetic MHR body stands in for the 700 MB rig, so construction,
checkpoint routing, the inference path and ONNX export stay in the fast suite.
Parity against the released InstantHMR graphs lives in
``test_instanthmr_parity.py`` (external_data).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from libreyolo import LibreInstantHMR, LibreYOLO
from libreyolo.models.instanthmr.body import MHRBody, _fk_levels
from libreyolo.models.instanthmr.nn import InstantHMRNet
from libreyolo.utils.serialization import wrap_libreyolo_checkpoint

pytestmark = pytest.mark.unit

N_JOINTS, N_VERTS, N_LANDMARKS = 4, 9, 5


def tiny_body() -> MHRBody:
    """A 4-joint chain with 9 vertices, shaped like the real constants."""
    g = torch.Generator().manual_seed(0)
    parents = np.array([-1, 0, 1, 2])
    partner, active = _fk_levels(parents)
    prerot = torch.zeros(N_JOINTS, 4)
    prerot[:, 3] = 1.0
    inverse_bind = torch.zeros(N_JOINTS, 8)
    inverse_bind[:, 6] = 1.0
    inverse_bind[:, 7] = 1.0
    skin_weight = torch.rand(N_VERTS, 2, generator=g)
    return MHRBody(
        dict(
            parameter_transform=torch.randn(N_JOINTS * 7, 204, generator=g) * 0.01,
            joint_offsets=torch.randn(N_JOINTS, 3, generator=g) * 10,
            joint_prerotations=prerot,
            fk_partner=torch.from_numpy(partner).long(),
            fk_active=torch.from_numpy(active),
            inverse_bind_pose=inverse_bind,
            base_shape=torch.randn(N_VERTS, 3, generator=g) * 10,
            shape_vectors=torch.randn(45, N_VERTS, 3, generator=g) * 0.1,
            skin_joint=torch.randint(0, N_JOINTS, (N_VERTS, 2), generator=g),
            skin_weight=skin_weight / skin_weight.sum(1, keepdim=True),
            landmark_vertices=torch.arange(N_LANDMARKS),
            W_joint=torch.rand(70, N_JOINTS, generator=g) * 0.5,
            W_vert=torch.rand(70, N_LANDMARKS, generator=g) * 0.5,
            faces=torch.randint(0, N_VERTS, (6, 3), generator=g),
        )
    )


@pytest.fixture(scope="module")
def nano_checkpoint(tmp_path_factory):
    torch.manual_seed(0)
    net = InstantHMRNet("n").eval()
    net.body = tiny_body()
    with torch.no_grad():
        # A fresh head decodes zero 6D rotations (no direction, NaN gradients);
        # give it a generic starting point, as released weights have.
        net.cont_head.init_estimate.normal_()
        net.cont_head.hand_mean.normal_()
        net.cont_head.hand_comps.normal_(std=0.1)
        net.cont_head.scale_comps.normal_(std=0.1)
        net.cont_head.coeff_hi.fill_(1.0)
        net.cont_head.flex_hi.fill_(1.0)
    path = tmp_path_factory.mktemp("instanthmr") / "LibreInstantHMRn-mesh.pt"
    torch.save(
        wrap_libreyolo_checkpoint(
            net.state_dict(),
            model_family="instanthmr",
            size="n",
            task="mesh",
            nc=1,
            names={0: "person"},
            imgsz=288,
        ),
        path,
    )
    return path


@pytest.mark.parametrize("size", ["n", "t", "s", "m"])
def test_size_detection_and_can_load(size):
    sd = InstantHMRNet(size).state_dict()
    assert LibreInstantHMR.can_load(sd)
    assert LibreInstantHMR.detect_size(sd) == size


def test_parameter_counts_match_upstream():
    # Upstream reports 20.0 / 20.3 / 24.4 / 32.2 M parameters.
    counts = {
        s: sum(p.numel() for p in InstantHMRNet(s).parameters()) / 1e6 for s in "ntsm"
    }
    for size, expected in {"n": 20.0, "t": 20.3, "s": 24.4, "m": 32.2}.items():
        assert abs(counts[size] - expected) < 0.06, counts


def test_filenames():
    assert LibreInstantHMR.detect_size_from_filename("LibreInstantHMRm-mesh.pt") == "m"
    assert (
        LibreInstantHMR.detect_task_from_filename("LibreInstantHMRm-mesh.pt") == "mesh"
    )
    assert (
        LibreInstantHMR.detect_variant_from_filename(
            "LibreInstantHMRm-mesh-commercial.pt"
        )
        == "commercial"
    )
    assert LibreInstantHMR.detect_size_from_filename("LibreInstantHMRm.pt") is None
    url = LibreInstantHMR.get_download_url("LibreInstantHMRm-mesh-commercial.pt")
    assert url.endswith(
        "LibreInstantHMRm-mesh-commercial/resolve/main/LibreInstantHMRm-mesh-commercial.pt"
    )
    assert "non-commercial" in LibreInstantHMR.get_download_notice(
        "LibreInstantHMRn-mesh.pt", url
    )
    assert "Apache-2.0" in LibreInstantHMR.get_download_notice(
        "LibreInstantHMRm-mesh-commercial.pt", url
    )


def test_body_round_trips_through_state_dict():
    body = tiny_body()
    empty = MHRBody()
    assert not empty.is_built
    empty.load_state_dict(body.state_dict())
    params, shape = torch.randn(2, 204) * 0.3, torch.randn(2, 45)
    a, b = body(params, shape), empty(params, shape)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
    assert a[0].shape == (2, 70, 3) and a[1].shape == (2, N_VERTS, 3)


def test_factory_routes_and_predicts_with_boxes(nano_checkpoint):
    model = LibreYOLO(str(nano_checkpoint), device="cpu")
    assert isinstance(model, LibreInstantHMR)
    assert model.task == "mesh" and model.size == "n"

    image = (np.random.default_rng(0).random((240, 320, 3)) * 255).astype(np.uint8)
    boxes = np.array([[40, 30, 120, 220], [200, 20, 300, 230]], dtype=np.float32)
    result = model(image, person_boxes=boxes)
    meshes = result.meshes
    assert len(result.boxes) == 2 and meshes.body_model == "mhr"
    assert meshes.vertices.shape == (2, N_VERTS, 3)
    assert meshes.joints3d.shape == (2, 70, 3) and meshes.joints2d.shape == (2, 70, 2)
    assert meshes.betas.shape == (2, 45) and meshes.global_orient.shape == (2, 3)
    assert torch.isfinite(meshes.vertices).all()
    # 3D is in the camera frame: root-relative output plus the translation.
    raw = model.estimate(np.ascontiguousarray(image[..., ::-1]), boxes)
    np.testing.assert_allclose(
        meshes.vertices[0].numpy(),
        raw[0]["pred_vertices"] + raw[0]["pred_cam_t"],
        atol=1e-5,
    )


def test_person_detector_callable_and_empty(nano_checkpoint):
    model = LibreInstantHMR(
        str(nano_checkpoint), size="n", device="cpu", person_detector=None
    )
    image = np.zeros((128, 128, 3), dtype=np.uint8)
    result = model(image, person_detector=lambda img: np.zeros((0, 4), np.float32))
    assert len(result.boxes) == 0 and result.meshes is None
    result = model(image, person_detector=lambda img: [[10, 10, 60, 120, 0.9]])
    assert len(result.boxes) == 1


def test_val_and_resume_raise(nano_checkpoint):
    model = LibreInstantHMR(str(nano_checkpoint), size="n", device="cpu")
    with pytest.raises(NotImplementedError):
        model.train(data="x.yaml", resume=True)
    with pytest.raises(NotImplementedError):
        model.val(data="x.yaml")


def test_onnx_export_matches_pytorch(nano_checkpoint, tmp_path):
    ort = pytest.importorskip("onnxruntime")
    pytest.importorskip("onnx")
    model = LibreInstantHMR(str(nano_checkpoint), size="n", device="cpu")
    path = model.export(output_path=tmp_path / "instanthmr.onnx")
    session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    meta = session.get_modelmeta().custom_metadata_map
    assert meta["model_family"] == "instanthmr" and meta["image_size"] == "288"

    g = torch.Generator().manual_seed(1)
    image, cond = (
        torch.randn(2, 3, 288, 288, generator=g),
        torch.randn(2, 3, generator=g) * 0.3,
    )
    got = dict(
        zip(
            [o.name for o in session.get_outputs()],
            session.run(None, {"image": image.numpy(), "cliff_cond": cond.numpy()}),
        )
    )
    with torch.no_grad():
        out = model.model(image, cond)
        joints_3d, vertices = model.model.body(out["mhr_params"], out["shape_params"])
    want = {
        **{k: out[k] for k in ("mhr_params", "shape_params", "cam_trans", "joints_2d")},
        "joints_3d": joints_3d,
        "vertices": vertices,
    }
    for name, value in want.items():
        np.testing.assert_allclose(
            got[name], value.numpy(), atol=1e-3, rtol=1e-3, err_msg=name
        )


def test_other_families_do_not_claim_instanthmr():
    from libreyolo.models import _ensure_rfdetr
    from libreyolo.models.base.model import BaseModel

    try:
        _ensure_rfdetr()  # RF-DETR's discriminator is the broadest; include it
    except ModuleNotFoundError:
        pass
    sd = InstantHMRNet("n").state_dict()
    claimers = [c.FAMILY for c in BaseModel._registry if c.can_load(sd)]
    assert claimers == ["instanthmr"]


def _write_pose_dataset(root) -> str:
    import cv2

    rng = np.random.default_rng(0)
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
        for i in range(2):
            image = (rng.random((160, 200, 3)) * 255).astype(np.uint8)
            cv2.imwrite(str(root / "images" / split / f"{i}.jpg"), image)
            kpts = np.concatenate(
                [rng.uniform(0.35, 0.65, (17, 2)), np.full((17, 1), 2.0)], 1
            ).ravel()
            row = [0, 0.5, 0.5, 0.4, 0.8, *kpts.tolist()]
            (root / "labels" / split / f"{i}.txt").write_text(
                " ".join(map(str, row)) + "\n"
            )
    yaml = root / "pose.yaml"
    yaml.write_text(
        f"path: {root}\ntrain: images/train\nval: images/val\nnames:\n  0: person\nkpt_shape: [17, 3]\n"
    )
    return str(yaml)


def test_fine_tune_runs_and_reloads(nano_checkpoint, tmp_path):
    data = _write_pose_dataset(tmp_path / "data")
    model = LibreInstantHMR(
        str(nano_checkpoint), size="n", device="cpu", person_detector=None
    )
    before = {k: v.clone() for k, v in model.model.state_dict().items()}
    results = model.train(
        data=data,
        epochs=1,
        batch=2,
        workers=0,
        project=str(tmp_path / "runs"),
        name="ft",
    )
    assert results["best_epoch"] == 1 and len(results["history"]) == 1
    assert {"train/2d", "train/reproj", "train/prior", "val/reproj"} <= set(
        results["history"][0]
    )
    saved = torch.load(results["best_checkpoint"], map_location="cpu", weights_only=False)
    assert saved["body_model"] == "mhr" and saved["num_betas"] == 45
    assert saved["num_body_pose"] == 130 and saved["rotation_format"] == "euler_zyx"
    assert saved["num_vertices"] == N_VERTS and saved["num_joints"] == N_JOINTS
    reloaded = LibreYOLO(results["best_checkpoint"], device="cpu")
    assert isinstance(reloaded, LibreInstantHMR)
    after = model.model.state_dict()
    assert not torch.equal(
        before["head_2d_logits.weight"], after["head_2d_logits.weight"]
    )
    assert torch.equal(before["body.base_shape"], after["body.base_shape"])


def test_fine_tune_rejects_non_coco17(nano_checkpoint, tmp_path):
    data = _write_pose_dataset(tmp_path / "data")
    path = Path(data)
    path.write_text(path.read_text().replace("kpt_shape: [17, 3]", "kpt_shape: [5, 3]"))
    model = LibreInstantHMR(
        str(nano_checkpoint), size="n", device="cpu", person_detector=None
    )
    with pytest.raises(ValueError, match="COCO-17"):
        model.train(data=data, epochs=1, project=str(tmp_path / "runs"))


def test_export_accepts_cli_arguments(nano_checkpoint, tmp_path):
    pytest.importorskip("onnx")
    model = LibreInstantHMR(str(nano_checkpoint), size="n", device="cpu", person_detector=None)
    cli_kwargs = {
        "half": False,
        "int8": False,
        "dynamic": True,
        "simplify": False,
        "opset": None,
        "batch": 1,
        "device": "cpu",
        "verbose": False,
    }
    path = model.export(format="onnx", output_path=tmp_path / "m.onnx", **cli_kwargs)
    assert Path(path).exists()
    with pytest.raises(NotImplementedError):
        model.export(format="onnx", output_path=tmp_path / "h.onnx", half=True)
    with pytest.raises(NotImplementedError):
        model.export(format="onnx", output_path=tmp_path / "x.onnx", simplify=True)
    with pytest.raises(ValueError):
        model.export(format="onnx", output_path=tmp_path / "s.onnx", imgsz=640)


def test_cli_passes_only_user_train_options():
    from libreyolo.cli.config import build_family_train_kwargs

    params = {"epochs": 3, "batch": 16, "imgsz": 640, "optimizer": "SGD", "amp": True}
    kwargs = build_family_train_kwargs(params, "instanthmr", user_provided={"epochs"})
    assert kwargs == {"epochs": 3}
