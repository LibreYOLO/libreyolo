"""Unit tests for the SAM 3D Body adapter.

The adapter wraps an optional third-party package and a 2 GB checkpoint, so
these tests stub both: a fake model object stands in for the family and returns
the output dict shape the upstream estimator produces. That keeps the mapping
logic, the person-source resolution and the error paths under test in the fast
suite, independent of whether the upstream package is installed.

Parity against the real model lives in the external_data suite.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from libreyolo.models.sam3dbody.inference import MeshInferenceRunner
from libreyolo.models.sam3dbody.person import (
    CallablePersonDetector,
    LibreYOLOPersonDetector,
    PersonBox,
    normalize_person_boxes,
    resolve_person_detector,
)
from tests.fixture_images import PARKOUR_FIXTURE

pytestmark = pytest.mark.unit

N_VERTS = 12
N_JOINTS = 70


def fake_person_output(seed: int = 0) -> dict:
    """One person's worth of upstream-shaped output."""
    rng = np.random.default_rng(seed)
    return {
        "pred_vertices": rng.normal(size=(N_VERTS, 3)).astype(np.float32),
        "pred_keypoints_3d": rng.normal(size=(N_JOINTS, 3)).astype(np.float32),
        "pred_keypoints_2d": (rng.random((N_JOINTS, 2)) * 100).astype(np.float32),
        "pred_cam_t": np.array([0.1, 0.2, 4.0], dtype=np.float32),
        "global_rot": rng.normal(size=(3,)).astype(np.float32),
        "body_pose_params": rng.normal(size=(133,)).astype(np.float32),
        "shape_params": rng.normal(size=(45,)).astype(np.float32),
        "scale_params": rng.normal(size=(28,)).astype(np.float32),
        "hand_pose_params": rng.normal(size=(108,)).astype(np.float32),
        "focal_length": np.float32(1500.0),
    }


class FakeModel:
    """Stands in for LibreSAM3DBody without the upstream package or weights."""

    names = {0: "person"}
    task = "mesh"

    def __init__(self, person_detector=None, n_out=1):
        self.person_detector = person_detector
        self.faces = torch.randint(0, N_VERTS, (20, 3))
        self.n_out = n_out
        self.last_call = None

    def estimate(self, image_rgb, boxes_xyxy, focal_length=None):
        self.last_call = {"boxes": np.asarray(boxes_xyxy), "focal_length": focal_length}
        return [fake_person_output(i) for i in range(self.n_out)]


def rgb(h=200, w=300):
    return np.zeros((h, w, 3), dtype=np.uint8)


class TestPersonSource:
    def test_normalize_accepts_plain_boxes(self):
        boxes = normalize_person_boxes([[0, 0, 10, 10], [5, 5, 20, 20]])
        assert len(boxes) == 2
        assert boxes[0].score == 1.0

    def test_normalize_accepts_scores_and_filters(self):
        boxes = normalize_person_boxes([[0, 0, 10, 10, 0.9], [0, 0, 5, 5, 0.1]], 0.5)
        assert len(boxes) == 1

    def test_normalize_accepts_arrays(self):
        assert len(normalize_person_boxes(np.zeros((3, 4)))) == 3

    def test_normalize_accepts_single_row_array(self):
        assert len(normalize_person_boxes(np.array([0.0, 0.0, 5.0, 5.0]))) == 1

    def test_normalize_passes_through_personbox(self):
        boxes = normalize_person_boxes([PersonBox((0, 0, 1, 1), 0.7)])
        assert boxes[0].score == 0.7

    def test_normalize_rejects_bad_width(self):
        with pytest.raises(ValueError, match="unsupported person-box length"):
            normalize_person_boxes([[1, 2, 3]])

    def test_normalize_rejects_bad_array_shape(self):
        with pytest.raises(ValueError, match=r"\(N, 4\) or \(N, 5\)"):
            normalize_person_boxes(np.zeros((3, 7)))

    def test_resolve_wraps_callable(self):
        assert isinstance(resolve_person_detector(lambda img: []), CallablePersonDetector)

    def test_resolve_passes_none(self):
        assert resolve_person_detector(None) is None

    def test_resolve_rejects_junk(self):
        with pytest.raises(TypeError, match="Unsupported person_detector"):
            resolve_person_detector(42)

    def test_callable_detector_normalizes(self):
        det = CallablePersonDetector(fn=lambda img: [[0, 0, 9, 9, 0.8]])
        out = det(rgb())
        assert len(out) == 1 and out[0].score == pytest.approx(0.8)


class TestRunnerMapping:
    def _run(self, model=None, **kwargs):
        model = model or FakeModel()
        runner = MeshInferenceRunner(model)
        return runner(rgb(), person_boxes=[[10, 10, 50, 150]], **kwargs)

    def test_returns_boxes_and_meshes_row_aligned(self):
        result = self._run(FakeModel(n_out=2))
        assert len(result.meshes) == 2
        assert result.meshes.body_model == "mhr"

    def test_translation_is_added_into_camera_frame(self):
        """Upstream returns root-relative geometry; the payload must be camera-frame."""
        model = FakeModel()
        result = self._run(model)
        raw = fake_person_output(0)
        expected = raw["pred_vertices"] + raw["pred_cam_t"]
        assert np.allclose(result.meshes.vertices[0].numpy(), expected, atol=1e-5)
        # Depth must be positive: the person is in front of the camera.
        assert float(result.meshes.vertices[..., 2].min()) > 0

    def test_joints3d_also_camera_frame(self):
        result = self._run()
        raw = fake_person_output(0)
        assert np.allclose(
            result.meshes.joints3d[0].numpy(),
            raw["pred_keypoints_3d"] + raw["pred_cam_t"],
            atol=1e-5,
        )

    def test_joints2d_passed_through_unmodified(self):
        result = self._run()
        assert np.allclose(
            result.meshes.joints2d[0].numpy(),
            fake_person_output(0)["pred_keypoints_2d"],
            atol=1e-5,
        )

    def test_projected_vertices_available_for_drawing(self):
        result = self._run()
        v2d = result.meshes.extras["vertices2d"]
        assert v2d.shape == (1, N_VERTS, 2)

    def test_model_specific_params_land_in_extras(self):
        meshes = self._run().meshes
        assert meshes.extras["scale"].shape == (1, 28)
        assert meshes.extras["hand_pose"].shape == (1, 108)

    def test_missing_optional_params_are_omitted_not_fatal(self):
        """Body-only runs omit hand output; that must degrade, not raise."""

        class BodyOnlyModel(FakeModel):
            def estimate(self, image_rgb, boxes_xyxy, focal_length=None):
                out = fake_person_output(0)
                del out["hand_pose_params"]
                return [out]

        meshes = MeshInferenceRunner(BodyOnlyModel())(
            rgb(), person_boxes=[[10, 10, 50, 150]]
        ).meshes
        assert "hand_pose" not in meshes.extras
        assert "scale" in meshes.extras

    def test_topology_is_shared_not_per_person(self):
        meshes = self._run(FakeModel(n_out=2)).meshes
        assert meshes.faces.shape == (20, 3)
        assert meshes[0].faces.shape == (20, 3)

    def test_focal_length_forwarded_to_estimator(self):
        model = FakeModel()
        self._run(model, focal_length=1234.0)
        assert model.last_call["focal_length"] == 1234.0

    def test_boxes_reach_the_estimator(self):
        model = FakeModel()
        self._run(model)
        assert model.last_call["boxes"].shape == (1, 4)

    def test_summary_and_json_round_trip(self):
        import json

        result = self._run()
        assert json.loads(result.to_json())[0]["mesh"]["body_model"] == "mhr"


class TestRunnerGuards:
    def test_no_person_source_is_an_actionable_error(self):
        runner = MeshInferenceRunner(FakeModel())
        with pytest.raises(RuntimeError, match="person_boxes|person_detector"):
            runner(rgb())

    def test_augment_rejected(self):
        runner = MeshInferenceRunner(FakeModel())
        with pytest.raises(ValueError, match="left and right"):
            runner(rgb(), person_boxes=[[0, 0, 9, 9]], augment=True)

    def test_tiling_rejected(self):
        runner = MeshInferenceRunner(FakeModel())
        with pytest.raises(ValueError, match="Tiled inference"):
            runner(rgb(), person_boxes=[[0, 0, 9, 9]], tiling=True)

    def test_person_boxes_with_video_is_refused_upfront(self, tmp_path):
        """Static boxes cannot follow a moving person, so say so clearly.

        Without this guard the boxes are silently dropped and each frame
        raises "pass person_boxes", telling the user to do what they did.
        """
        video = tmp_path / "clip.mp4"
        video.write_bytes(b"\x00" * 16)
        runner = MeshInferenceRunner(FakeModel())
        with pytest.raises(ValueError, match="cannot be reused across video"):
            runner(str(video), person_boxes=[[0, 0, 9, 9]])

    def test_video_with_a_detector_is_not_refused(self, tmp_path):
        """The guard must only reject the boxes case, not video as such."""
        video = tmp_path / "clip.mp4"
        video.write_bytes(b"\x00" * 16)
        model = FakeModel(person_detector=CallablePersonDetector(fn=lambda img: []))
        runner = MeshInferenceRunner(model)
        # Reaches the decode path and fails there on the fake file, rather than
        # being turned away by the person-source guard.
        with pytest.raises(Exception) as exc:
            runner(str(video))
        assert "cannot be reused across video" not in str(exc.value)

    def test_bad_output_format_rejected(self):
        runner = MeshInferenceRunner(FakeModel())
        with pytest.raises(ValueError, match="output_file_format"):
            runner(rgb(), person_boxes=[[0, 0, 9, 9]], output_file_format="gif")

    def test_no_people_returns_empty_result_not_a_crash(self):
        runner = MeshInferenceRunner(FakeModel())
        result = runner(rgb(), person_boxes=[])
        assert len(result.boxes) == 0
        assert result.meshes is None

    def test_detector_below_confidence_yields_empty(self):
        model = FakeModel(person_detector=CallablePersonDetector(
            fn=lambda img: [[0, 0, 9, 9, 0.1]]
        ))
        result = MeshInferenceRunner(model)(rgb(), conf=0.5)
        assert len(result.boxes) == 0


@pytest.mark.external_data
class TestRealModelParity:
    """Runs the real model. Needs the upstream package, weights and CUDA."""

    @pytest.fixture(scope="class")
    def model(self):
        import os

        torch_cuda = torch.cuda.is_available()
        if not torch_cuda:
            pytest.skip("SAM 3D Body inference requires CUDA")
        if not os.environ.get("SAM_3D_BODY_PATH"):
            pytest.skip("SAM_3D_BODY_PATH not set")
        ckpt = os.environ.get("SAM_3D_BODY_CKPT")
        if not ckpt or not os.path.exists(ckpt):
            pytest.skip("SAM_3D_BODY_CKPT not set to an existing checkpoint")
        from libreyolo.models.sam3dbody import LibreSAM3DBody

        return LibreSAM3DBody(model_path=ckpt, size="d3")

    def test_predicts_a_body_in_the_camera_frame(self, model):
        from PIL import Image

        width, height = Image.open(PARKOUR_FIXTURE).size
        result = model(
            PARKOUR_FIXTURE,
            person_boxes=[[width * 0.2, height * 0.05, width * 0.8, height * 0.95]],
        )
        meshes = result.meshes
        assert len(meshes) == 1
        assert meshes.num_vertices == 18439
        assert meshes.num_joints == 70
        assert meshes.betas.shape == (1, 45)
        # Metric depth, in front of the camera, at a plausible distance.
        assert 0.5 < float(meshes.transl[0, 2]) < 20.0
        assert float(meshes.vertices[..., 2].min()) > 0
        # Projected joints should land on the image canvas.
        inside = (
            (meshes.joints2d[..., 0] > 0)
            & (meshes.joints2d[..., 0] < width)
            & (meshes.joints2d[..., 1] > 0)
            & (meshes.joints2d[..., 1] < height)
        )
        assert inside.float().mean() > 0.8

    def test_our_projection_matches_upstreams(self, model):
        """Our camera math must reproduce the upstream 2D joints exactly."""
        from PIL import Image

        from libreyolo.models.sam3dbody.camera import perspective_project

        width, height = Image.open(PARKOUR_FIXTURE).size
        result = model(
            PARKOUR_FIXTURE,
            person_boxes=[[width * 0.2, height * 0.05, width * 0.8, height * 0.95]],
        )
        meshes = result.meshes
        reprojected = perspective_project(
            meshes.joints3d, meshes.focal_length, image_size=(height, width)
        )
        assert torch.allclose(reprojected, meshes.joints2d, atol=1e-2)


class TestDeviceAndTaskGuards:
    """Regression cover for the device and task-dispatch guards."""

    def _bare_model(self, device: str):
        from libreyolo.models.sam3dbody.model import LibreSAM3DBody

        model = LibreSAM3DBody.__new__(LibreSAM3DBody)
        model.device = torch.device(device)
        return model

    def test_estimate_rejects_a_cpu_model_even_when_cuda_exists(self):
        """A global CUDA check would wave this through into a device mismatch."""
        model = self._bare_model("cpu")
        with pytest.raises(RuntimeError, match="requires a CUDA device"):
            model.estimate(rgb(), np.zeros((1, 4), dtype=np.float32))

    def test_estimate_error_names_the_offending_device(self):
        model = self._bare_model("cpu")
        with pytest.raises(RuntimeError, match="is on cpu"):
            model.estimate(rgb(), np.zeros((1, 4), dtype=np.float32))

    def test_faces_degrades_when_upstream_stops_exposing_topology(self):
        model = self._bare_model("cpu")
        model.model = object()  # no head_pose attribute at all
        assert model.faces is None

    def test_track_is_refused_for_mesh_models(self):
        """val() guards mesh; track() must too, or it silently misbehaves."""
        from libreyolo.models.base.model import BaseModel

        class Stub:
            task = "mesh"

        # track() is a generator function, so its guards, like every other
        # task guard there, raise on first iteration rather than on the call.
        with pytest.raises(NotImplementedError, match="body-mesh"):
            next(BaseModel.track(Stub(), source="video.mp4"))


class TestUpstreamDependency:
    def test_missing_package_explains_the_licensing_situation(self, monkeypatch):
        """The error must tell users why it is not bundled and how to install it."""
        import sys

        from libreyolo.models.sam3dbody.model import LibreSAM3DBody

        monkeypatch.setitem(sys.modules, "sam_3d_body", None)
        monkeypatch.delitem(sys.modules, "sam_3d_body")
        real_import = __import__

        def blocked(name, *args, **kwargs):
            if name == "sam_3d_body":
                raise ImportError("not installed")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", blocked)
        model = LibreSAM3DBody.__new__(LibreSAM3DBody)
        model._sam_3d_body_path = None
        with pytest.raises(ImportError) as exc:
            model._import_upstream()
        message = str(exc.value)
        assert "SAM License" in message
        assert "github.com/facebookresearch/sam-3d-body" in message
        assert "sam_3d_body_path" in message
