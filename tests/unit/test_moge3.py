"""Unit tests for the MoGe-3 depth and normal family (no weights, CPU only)."""

from __future__ import annotations

import inspect

import numpy as np
import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.unit

_MIRROR = "https://huggingface.co/LibreYOLO/LibreMoGe3{size}-{task}/resolve/main/LibreMoGe3{size}-{task}.pt"


def _moge2_state(embed_dim: int = 1024) -> dict:
    return {
        "encoder.backbone.cls_token": torch.zeros(1, 1, embed_dim),
        "neck.input_blocks.0.weight": torch.zeros(1),
        "normal_head.output_blocks.4.weight": torch.zeros(3, 32, 1, 1),
        "points_head.output_blocks.4.weight": torch.zeros(3, 32, 1, 1),
        "mask_head.output_blocks.4.weight": torch.zeros(1, 32, 1, 1),
        "scale_head.0.weight": torch.zeros(1, embed_dim),
    }


def _moge3_state(embed_dim: int = 1024) -> dict:
    state = _moge2_state(embed_dim)
    state["refiner.input_proj.weight"] = torch.zeros(32, 3)
    return state


class TestMoGe3Metadata:
    def test_registration_and_class_contract(self):
        from libreyolo.models import LibreMoGe3
        from libreyolo.models.base import BaseModel
        from libreyolo.models.registry import MODEL_GROUPS

        assert LibreMoGe3.FAMILY == "moge3"
        assert LibreMoGe3.FILENAME_PREFIX == "LibreMoGe3"
        assert LibreMoGe3.SUPPORTED_TASKS == ("depth", "normal")
        assert LibreMoGe3.DEFAULT_TASK == "depth"
        assert LibreMoGe3.INPUT_SIZES == {"l": 840, "g": 840}
        assert LibreMoGe3.depth_encoding == "depth"
        assert inspect.signature(LibreMoGe3).parameters["size"].default == "l"
        assert LibreMoGe3 in BaseModel._registry
        assert MODEL_GROUPS["moge3"] == "g3"

    @pytest.mark.parametrize("size", ["l", "g"])
    @pytest.mark.parametrize("task", ["depth", "normal"])
    def test_filenames_resolve_to_pinned_mirror(self, size, task):
        from libreyolo.models.moge3.model import LibreMoGe3

        filename = f"LibreMoGe3{size}-{task}.pt"
        assert LibreMoGe3.detect_size_from_filename(filename) == size
        assert LibreMoGe3.detect_task_from_filename(filename) == task
        assert LibreMoGe3.get_download_url(filename) == _MIRROR.format(
            size=size, task=task
        )
        assert len(LibreMoGe3._MIRROR_SHA256[(size, task)]) == 64
        # The task suffix is required.
        assert LibreMoGe3.get_download_url(f"LibreMoGe3{size}.pt") is None

    def test_download_verification_rejects_wrong_bytes(self, tmp_path):
        from libreyolo.models.moge3.model import LibreMoGe3

        path = tmp_path / "LibreMoGe3l-depth.pt"
        path.write_bytes(b"not a checkpoint")
        with pytest.raises(ValueError, match="SHA-256 mismatch"):
            LibreMoGe3.verify_downloaded_file(
                str(path), _MIRROR.format(size="l", task="depth")
            )
        with pytest.raises(ValueError, match="Unrecognized"):
            LibreMoGe3.verify_downloaded_file(
                str(path), "https://huggingface.co/LibreYOLO/Other/resolve/main/x.pt"
            )

    @pytest.mark.parametrize(("embed_dim", "size"), [(1024, "l"), (1536, "g")])
    def test_size_detection(self, embed_dim, size):
        from libreyolo.models.moge3.model import LibreMoGe3

        state = _moge3_state(embed_dim)
        assert LibreMoGe3.can_load(state)
        assert LibreMoGe3.detect_size(state) == size
        assert LibreMoGe3.detect_nb_classes(state) == 1


class TestMoGeSiblingRouting:
    """MoGe-2 and MoGe-3 share encoder, neck, and head names."""

    def test_moge3_rejects_moge2_checkpoints(self):
        from libreyolo.models.moge3.model import LibreMoGe3

        state = _moge2_state()
        assert not LibreMoGe3.can_load(state)
        assert LibreMoGe3.convert_upstream_state_dict(state) is None

    def test_moge2_rejects_moge3_checkpoints(self):
        from libreyolo.models.moge2.model import LibreMoGe2

        state = _moge3_state()
        assert not LibreMoGe2.can_load(state)
        assert LibreMoGe2.convert_upstream_state_dict(state) is None

    def test_saved_normal_graph_tag_routes_to_moge3(self):
        from libreyolo.models.moge2.model import LibreMoGe2
        from libreyolo.models.moge3.model import LibreMoGe3

        state = _moge2_state()
        state["moge3_tag"] = torch.tensor(3)
        assert LibreMoGe3.can_load(state)
        assert not LibreMoGe2.can_load(state)

    def test_each_family_still_claims_its_own(self):
        from libreyolo.models.moge2.model import LibreMoGe2
        from libreyolo.models.moge3.model import LibreMoGe3

        assert LibreMoGe2.can_load(_moge2_state())
        assert LibreMoGe3.convert_upstream_state_dict(_moge3_state()) is not None

    def test_task_state_selection(self):
        from libreyolo.models.moge3.model import _clean_state_dict

        state = {f"module.{key}": value for key, value in _moge3_state().items()}
        depth = _clean_state_dict(state, "depth")
        normal = _clean_state_dict(state, "normal")
        assert "refiner.input_proj.weight" in depth
        assert "points_head.output_blocks.4.weight" in depth
        assert not any(key.startswith("normal_head.") for key in depth)
        assert set(normal) == {
            "encoder.backbone.cls_token",
            "neck.input_blocks.0.weight",
            "normal_head.output_blocks.4.weight",
            "moge3_tag",
        }


class TestSparseOps:
    @staticmethod
    def _level(coords, size):
        from libreyolo.models.moge3.sparse import SparseLevel

        return SparseLevel(torch.as_tensor(coords, dtype=torch.long), size)

    def test_submanifold_conv_matches_dense_conv_at_active_sites(self):
        from libreyolo.models.moge3.sparse import SubmanifoldConv3d

        torch.manual_seed(0)
        size = (2, 5, 6, 4)
        occupancy = torch.rand(size) < 0.35
        coords = occupancy.nonzero()
        level = self._level(coords, size)
        conv = SubmanifoldConv3d(3, 5)
        feats = torch.randn(coords.shape[0], 3)

        dense = torch.zeros(2, 3, 5, 6, 4)
        dense[coords[:, 0], :, coords[:, 1], coords[:, 2], coords[:, 3]] = feats
        weight = conv.weight.permute(0, 4, 1, 2, 3)
        expected = F.conv3d(dense, weight, conv.bias, padding=1)
        expected = expected[coords[:, 0], :, coords[:, 1], coords[:, 2], coords[:, 3]]

        with torch.no_grad():
            actual = conv(feats, level)
        assert torch.allclose(actual, expected, atol=1e-5)

    def test_neighbor_lookup_does_not_wrap_across_rows(self):
        from libreyolo.models.moge3.sparse import build_neighbor_table

        # (0, 0, 1, 0) would alias (0, 0, 0, -1) under a plain linear key.
        level = self._level([[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], (1, 1, 2, 2))
        table = build_neighbor_table(level)
        z_minus = (1 * 9) + (1 * 3) + 0
        z_plus = (1 * 9) + (1 * 3) + 2
        assert table[z_minus, 1].item() == 3  # absent sentinel, not site 2
        assert table[z_plus, 0].item() == 2

    def test_mean_pool_and_nearest_upsample_round_trip(self):
        from libreyolo.models.moge3.sparse import sparse_mean_pool

        coords = [[0, 0, 0, 0], [0, 1, 1, 1], [0, 2, 0, 0], [1, 0, 0, 3]]
        level = self._level(coords, (2, 3, 2, 4))
        feats = torch.tensor([[1.0], [3.0], [5.0], [7.0]])
        pooled, parent, inverse = sparse_mean_pool(feats, level, 2)

        assert parent.size == (2, 2, 1, 2)
        assert parent.coords.tolist() == [[0, 0, 0, 0], [0, 1, 0, 0], [1, 0, 0, 1]]
        assert pooled.squeeze(1).tolist() == [2.0, 5.0, 7.0]
        assert pooled[inverse].squeeze(1).tolist() == [2.0, 2.0, 5.0, 7.0]

    def test_unet_keeps_one_output_per_input_site(self):
        from libreyolo.models.moge3.sparse import Sparse3DUNet

        torch.manual_seed(0)
        net = Sparse3DUNet(
            in_channels=3,
            out_channels=1,
            encoder_channels=6,
            model_channels=[4, 8],
            downsample_factors=[2],
            encoder_downsample=2,
        ).eval()
        coords = torch.stack(
            torch.meshgrid(
                torch.arange(1), torch.arange(4), torch.arange(6), indexing="ij"
            ),
            dim=-1,
        ).reshape(-1, 3)
        z = torch.randint(0, 3, (coords.shape[0], 1))
        level = self._level(torch.cat((coords, z), dim=1), (1, 4, 6, 3))
        with torch.no_grad():
            out = net(torch.randn(coords.shape[0], 3), level, torch.randn(1, 6, 2, 3))
        assert out.shape == (coords.shape[0], 1)
        assert torch.isfinite(out).all()


class TestGeometry:
    def test_recovers_pinhole_camera_and_shift(self):
        from libreyolo.models.moge3.utils import recover_geometry

        height, width = 48, 64
        aspect = width / height
        fx, fy = 0.9, 0.9 * aspect  # normalized: same pixel focal on both axes
        v, u = torch.meshgrid(
            (torch.arange(height) + 0.5) / height,
            (torch.arange(width) + 0.5) / width,
            indexing="ij",
        )
        depth = 2.0 + 0.5 * u + 0.25 * v
        points = torch.stack(
            ((u - 0.5) / fx * depth, (v - 0.5) / fy * depth, depth), dim=-1
        )
        affine = points.clone()
        affine[..., 2] -= 1.5  # the network predicts depth up to a z shift
        mask = torch.ones(1, height, width)

        out = recover_geometry(affine[None], mask, torch.tensor([2.0]))

        assert torch.allclose(out["intrinsics"][0, 0, 0], torch.tensor(fx), rtol=1e-2)
        assert torch.allclose(out["intrinsics"][0, 1, 1], torch.tensor(fy), rtol=1e-2)
        assert torch.allclose(out["depth"][0], 2.0 * depth, rtol=1e-2)
        assert torch.allclose(out["points"][0], 2.0 * points, rtol=1e-2, atol=1e-3)
        assert out["mask"].all()

    def test_pixel_intrinsics_use_opencv_pixel_centers(self):
        from libreyolo.models.moge3.utils import pixel_intrinsics

        normalized = torch.tensor([[0.8, 0, 0.5], [0, 1.2, 0.5], [0, 0, 1.0]])
        k = pixel_intrinsics(normalized, 640, 480)
        expected = torch.tensor(
            [[512.0, 0.0, 319.5], [0.0, 576.0, 239.5], [0.0, 0.0, 1.0]]
        )
        assert torch.allclose(k, expected)

    def test_token_budget(self):
        from libreyolo.models.moge3.nn import token_grid
        from libreyolo.models.moge3.utils import num_tokens_for

        assert num_tokens_for(840) == 3600
        assert token_grid(3600, 1.0) == (60, 60)
        assert token_grid(3600, 1536 / 1194) == (53, 68)


class TestDepthMapGeometry:
    def test_extras_survive_device_and_numpy_conversion(self):
        from libreyolo.utils.results import DepthMap, Results

        depth = DepthMap(
            torch.ones(4, 5),
            encoding="depth",
            metric=True,
            point_map=torch.zeros(4, 5, 3),
            intrinsics=torch.eye(3),
            valid_mask=torch.ones(4, 5, dtype=torch.bool),
        )
        result = Results(boxes=None, orig_shape=(4, 5), depth_map=depth)
        converted = result.numpy().depth_map
        assert converted.metric and converted.encoding == "depth"
        assert isinstance(converted.point_map, np.ndarray)
        assert converted.point_map.shape == (4, 5, 3)
        assert converted.intrinsics.shape == (3, 3)
        assert converted.valid_mask.dtype == bool
        assert result[0].depth_map.point_map is depth.point_map
        half = depth.to(torch.float16)
        assert half.data.dtype == torch.float16
        assert half.valid_mask.dtype == torch.bool

    def test_rejects_inconsistent_geometry(self):
        from libreyolo.utils.results import DepthMap

        with pytest.raises(ValueError, match="point_map"):
            DepthMap(torch.ones(4, 5), point_map=torch.zeros(5, 4, 3))
        with pytest.raises(ValueError, match="metric"):
            DepthMap(torch.ones(4, 5), metric=True)


@pytest.fixture(scope="module")
def tiny_moge3(request):
    """Random-init MoGe-3 graph on a ViT-S encoder with a narrow refiner."""
    from libreyolo.models.moge3 import model as model_module
    from libreyolo.models.moge3 import nn as nn_module

    patch = pytest.MonkeyPatch()
    patch.setitem(
        nn_module.SIZE_CONFIGS,
        "t",
        {
            "backbone": "dinov2_vits14",
            "intermediate_layers": [5, 11],
            "dim": 384,
            "decoder_dims": [32, 16, 16, 16, 16],
        },
    )
    patch.setitem(nn_module.REFINER_CONFIG, "model_channels", [4, 4, 4, 4, 8])
    patch.setattr(model_module.LibreMoGe3, "INPUT_SIZES", {"t": 28})
    request.addfinalizer(patch.undo)
    torch.manual_seed(0)
    return {
        task: model_module.LibreMoGe3(None, size="t", task=task, device="cpu")
        for task in ("depth", "normal")
    }


class TestMoGe3Forward:
    def test_task_graphs_build_only_their_heads(self, tiny_moge3):
        depth = tiny_moge3["depth"].model
        normal = tiny_moge3["normal"].model
        assert hasattr(depth, "refiner") and not hasattr(depth, "normal_head")
        assert hasattr(normal, "normal_head") and not hasattr(normal, "refiner")

    def test_normal_forward_is_unit_and_dense(self, tiny_moge3):
        with torch.inference_mode():
            output = tiny_moge3["normal"].model(torch.rand(1, 3, 28, 42))
        assert output.shape == (1, 3, 28, 42)
        norms = torch.linalg.vector_norm(output, dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_depth_forward_runs_the_refiner(self, tiny_moge3):
        net = tiny_moge3["depth"].model
        image = torch.rand(1, 3, 28, 42)
        with torch.inference_mode():
            coarse = net(image, refine_steps=0)
            refined = net(image)
        assert refined["points"].shape == (1, 28, 42, 3)
        assert refined["mask"].shape == (1, 28, 42)
        assert refined["metric_scale"].shape == (1,)
        assert torch.isfinite(refined["points"]).all()
        assert not torch.equal(coarse["points"], refined["points"])

    def test_depth_predict_attaches_camera_geometry(self, tiny_moge3):
        image = np.random.default_rng(0).integers(0, 255, (21, 35, 3), dtype=np.uint8)
        result = tiny_moge3["depth"].predict(image, imgsz=28)
        depth = result.depth_map
        assert result.boxes is None
        assert depth.metric and depth.encoding == "depth"
        assert tuple(depth.data.shape) == (21, 35)
        assert tuple(depth.point_map.shape) == (21, 35, 3)
        assert tuple(depth.intrinsics.shape) == (3, 3)
        valid = depth.valid_mask
        assert torch.equal(torch.isfinite(depth.data), valid)
        assert torch.isinf(depth.point_map[~valid]).all()

    def test_normal_predict_returns_original_canvas(self, tiny_moge3):
        image = np.zeros((21, 35, 3), dtype=np.uint8)
        result = tiny_moge3["normal"].predict(image, imgsz=28)
        assert tuple(result.normal_map.data.shape) == (21, 35, 3)
        result.normal_map.assert_normalized(atol=1e-5)

    def test_out_of_scope_paths_raise(self, tiny_moge3):
        with pytest.raises(NotImplementedError, match="not part"):
            tiny_moge3["depth"].train(data="depth.yaml")
        with pytest.raises(NotImplementedError, match="depth export"):
            tiny_moge3["depth"].export(format="onnx")


def test_moge3_export_support_tiers():
    from libreyolo.export.support import EXPORT_FORMATS, get_support

    for format_name in ("onnx", "torchscript"):
        assert get_support("moge3", "normal", format_name).tier == "validated"
    for format_name in EXPORT_FORMATS:
        assert get_support("moge3", "depth", format_name).tier == "blocked"


@pytest.mark.parametrize("format_name", ["onnx", "torchscript"])
def test_exported_moge3_normal_parity(tiny_moge3, tmp_path, format_name):
    if format_name == "onnx":
        pytest.importorskip("onnx")
        pytest.importorskip("onnxruntime")
    from libreyolo import LibreYOLO

    model = tiny_moge3["normal"]
    image = np.zeros((28, 28, 3), dtype=np.uint8)
    image[..., 0] = np.arange(28, dtype=np.uint8)[None] * 9
    image[..., 1] = np.arange(28, dtype=np.uint8)[:, None] * 9
    image[..., 2] = 127
    native = model.predict(image, imgsz=28).normal_map.data.numpy()

    suffix = ".onnx" if format_name == "onnx" else ".torchscript"
    exported_path = model.export(
        format=format_name,
        output_path=str(tmp_path / f"LibreMoGe3t-normal{suffix}"),
        imgsz=28,
        **({"simplify": False} if format_name == "onnx" else {}),
    )
    assert not model.model.onnx_compatible_mode
    exported = LibreYOLO(exported_path, device="cpu")
    actual = exported.predict(image).normal_map.data.numpy()
    dots = np.sum(native.astype(np.float64) * actual.astype(np.float64), axis=-1)
    angular = np.rad2deg(np.arccos(np.clip(dots, -1.0, 1.0)))
    assert actual.shape == native.shape
    assert float(angular.mean()) < 0.1


@pytest.mark.parametrize("task", ["depth", "normal"])
def test_saved_checkpoint_reloads_through_factory(tiny_moge3, tmp_path, task):
    from libreyolo import LibreYOLO

    model = tiny_moge3[task]
    path = model.save(str(tmp_path / f"LibreMoGe3t-{task}.pt"))
    reloaded = LibreYOLO(path, device="cpu")
    assert reloaded.family == "moge3"
    assert reloaded.task == task
    original = model.model.state_dict()
    for key, value in reloaded.model.state_dict().items():
        assert torch.equal(value, original[key]), key
