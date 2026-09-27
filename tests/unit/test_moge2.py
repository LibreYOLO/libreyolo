"""Unit and exported-runtime tests for the MoGe-2 normal family."""

from __future__ import annotations

import inspect

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.unit


class TestMoGe2Metadata:
    def test_task_size_and_factory_registration(self):
        from libreyolo.models import LibreMoGe2
        from libreyolo.models.base import BaseModel

        assert LibreMoGe2.FAMILY == "moge2"
        assert LibreMoGe2.FILENAME_PREFIX == "LibreMoGe2"
        assert LibreMoGe2.SUPPORTED_TASKS == ("normal",)
        assert LibreMoGe2.DEFAULT_TASK == "normal"
        assert LibreMoGe2.INPUT_SIZES == {"s": 518, "b": 518, "l": 518}
        assert inspect.signature(LibreMoGe2).parameters["size"].default == "l"
        assert LibreMoGe2.normal_imgsz_divisor == 14
        assert LibreMoGe2 in BaseModel._registry

    @pytest.mark.parametrize(
        ("size", "repo", "revision"),
        [
            (
                "s",
                "moge-2-vits-normal",
                "679230677b4d282c6f304189a93e98e14f085902",
            ),
            (
                "b",
                "moge-2-vitb-normal",
                "54ad3a693e61907ea4633d13dec6ee682fa09419",
            ),
            (
                "l",
                "moge-2-vitl-normal",
                "b135031bae30b5ac2ae141a0e68717795ce38340",
            ),
        ],
    )
    def test_filename_and_download_are_pinned(self, size, repo, revision):
        from libreyolo.models.moge2.model import LibreMoGe2

        filename = f"LibreMoGe2{size}-normal.pt"
        assert LibreMoGe2.detect_size_from_filename(filename) == size
        assert LibreMoGe2.detect_task_from_filename(filename) == "normal"

        if size in LibreMoGe2._MIRROR_SHA256:
            # Mirrored sizes come from the LibreYOLO org, pinned by the digest
            # of the converted checkpoint the mirror serves.
            assert LibreMoGe2.get_download_url(filename) == (
                f"https://huggingface.co/LibreYOLO/LibreMoGe2{size}-normal"
                f"/resolve/main/LibreMoGe2{size}-normal.pt"
            )
            assert len(LibreMoGe2._MIRROR_SHA256[size]) == 64
        else:
            # Everything else still comes from upstream at a pinned revision.
            assert LibreMoGe2.get_download_url(filename) == (
                f"https://huggingface.co/Ruicheng/{repo}/resolve/{revision}/model.pt"
            )
        assert LibreMoGe2.get_download_url(f"LibreMoGe2{size}.pt") is None

    @pytest.mark.parametrize(
        ("embed_dim", "size"),
        [(384, "s"), (768, "b"), (1024, "l")],
    )
    def test_checkpoint_signature_and_size_detection(self, embed_dim, size):
        from libreyolo.models.moge2.model import LibreMoGe2

        state = {
            "encoder.backbone.cls_token": torch.zeros(1, 1, embed_dim),
            "neck.input_blocks.0.weight": torch.zeros(1),
            "normal_head.output_blocks.4.weight": torch.zeros(3, 32, 1, 1),
        }
        assert LibreMoGe2.can_load(state)
        assert LibreMoGe2.detect_size(state) == size
        assert LibreMoGe2.detect_checkpoint_task(state) == "normal"
        assert LibreMoGe2.detect_nb_classes(state) == 1

    def test_upstream_conversion_keeps_only_normal_graph(self):
        from libreyolo.models.moge2.model import LibreMoGe2

        state = {
            "encoder.backbone.cls_token": torch.zeros(1, 1, 384),
            "neck.input_blocks.0.weight": torch.zeros(1),
            "normal_head.output_blocks.4.weight": torch.zeros(3, 32, 1, 1),
            "points_head.output_blocks.4.weight": torch.zeros(3, 32, 1, 1),
            "mask_head.output_blocks.4.weight": torch.zeros(1, 32, 1, 1),
            "scale_head.weight": torch.zeros(1, 384),
        }
        converted = LibreMoGe2.convert_upstream_state_dict(state)

        assert converted is not None
        assert set(converted) == {
            "encoder.backbone.cls_token",
            "neck.input_blocks.0.weight",
            "normal_head.output_blocks.4.weight",
        }


class TestMoGe2Preprocess:
    def test_keep_aspect_patch_grid_and_rgb_range(self):
        from libreyolo.models.moge2.utils import preprocess_numpy

        image = np.zeros((240, 400, 3), dtype=np.uint8)
        image[..., 0] = 255
        chw, ratio = preprocess_numpy(image, input_size=518)

        assert chw.dtype == np.float32
        assert chw.shape == (3, 518, 868)
        assert chw.flags.c_contiguous
        assert ratio == 1.0
        assert float(chw[0].min()) == 1.0
        assert float(chw[1:].max()) == 0.0

    def test_exported_preprocess_rejects_aspect_ratio_mismatch(self):
        from libreyolo.backends.base import BaseBackend

        image = np.zeros((240, 400, 3), dtype=np.uint8)

        with pytest.raises(
            ValueError,
            match="aspect ratio to match the fixed export canvas",
        ):
            BaseBackend._preprocess_normal(image, 518, "rgb")


@pytest.fixture(scope="module")
def moge_small():
    """A random-init ViT-S wrapper; no network or checkpoint is required."""
    from libreyolo.models.moge2.model import LibreMoGe2

    return LibreMoGe2(None, size="s", task="normal", device="cpu")


class TestMoGe2Forward:
    def test_forward_is_dense_and_unit_normalized(self, moge_small):
        image = torch.rand(1, 3, 28, 42)
        with torch.inference_mode():
            output = moge_small.model(image)

        assert output.shape == (1, 3, 28, 42)
        norms = torch.linalg.vector_norm(output, dim=1)
        assert torch.isfinite(output).all()
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_predict_returns_original_canvas_hwc(self, moge_small):
        image = np.zeros((21, 35, 3), dtype=np.uint8)
        image[..., 0] = np.arange(35, dtype=np.uint8)[None]
        image[..., 1] = np.arange(21, dtype=np.uint8)[:, None]

        result = moge_small.predict(image, imgsz=28)

        assert result.boxes is None
        assert result.normal_map is not None
        assert result.normal_map.data.dtype == torch.float32
        assert tuple(result.normal_map.data.shape) == (21, 35, 3)
        result.normal_map.assert_normalized(atol=1e-5)

    def test_training_is_explicitly_out_of_scope(self, moge_small):
        with pytest.raises(NotImplementedError, match="not part"):
            moge_small.train(data="normals.yaml")


def test_backend_normal_decode_accepts_bchw_and_repairs_invalid_vectors():
    from libreyolo.backends.base import BaseBackend

    output = np.zeros((1, 3, 2, 3), dtype=np.float32)
    output[:, 0] = 4.0
    output[:, :, 0, 0] = np.nan
    output[:, :, 1, 2] = 0.0

    normal = BaseBackend._parse_normal_output([output], (6, 4))

    assert normal.shape == (4, 6, 3)
    assert torch.isfinite(normal).all()
    assert torch.allclose(
        torch.linalg.vector_norm(normal, dim=-1),
        torch.ones((4, 6)),
        atol=1e-5,
    )
    assert normal[0, 0].tolist() == pytest.approx([0.0, 0.0, -1.0])
    assert normal[-1, -1].tolist() == pytest.approx([0.0, 0.0, -1.0])


def test_moge2_export_support_matches_validated_runtimes():
    from libreyolo.export.support import EXPORT_FORMATS, get_support

    validated = {
        "onnx",
        "torchscript",
        "executorch",
        "tensorrt",
        "openvino",
    }
    for format_name in validated:
        assert get_support("moge2", "normal", format_name).tier == "validated"
    assert get_support("moge2", "normal", "ncnn").tier == "available"
    for format_name in set(EXPORT_FORMATS) - validated - {"ncnn"}:
        assert get_support("moge2", "normal", format_name).tier == "blocked"


def test_exported_moge2_normal_parity(moge_small, tmp_path):
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from libreyolo import LibreYOLO

    image = np.zeros((28, 28, 3), dtype=np.uint8)
    image[..., 0] = np.arange(28, dtype=np.uint8)[None]
    image[..., 1] = np.arange(28, dtype=np.uint8)[:, None]
    image[..., 2] = 127
    native = moge_small.predict(image, imgsz=28).normal_map.data.numpy()

    artifact = tmp_path / "LibreMoGe2s-normal.onnx"
    exported_path = moge_small.export(
        format="onnx",
        output_path=str(artifact),
        imgsz=28,
        simplify=False,
    )
    assert not moge_small.model.onnx_compatible_mode
    proto = onnx.load(exported_path, load_external_data=False)
    assert [output.name for output in proto.graph.output] == ["normal"]
    metadata = {item.key: item.value for item in proto.metadata_props}
    assert metadata["task"] == "normal"
    assert metadata["model_family"] == "moge2"
    assert metadata["dynamic"].lower() == "false"

    exported = LibreYOLO(exported_path, device="cpu")
    actual = exported.predict(image).normal_map.data.numpy()
    dots = np.sum(native.astype(np.float64) * actual.astype(np.float64), axis=-1)
    angular = np.rad2deg(np.arccos(np.clip(dots, -1.0, 1.0)))

    assert actual.shape == native.shape
    assert float(angular.mean()) < 0.1
    assert np.linalg.norm(actual, axis=-1) == pytest.approx(1.0, abs=1e-5)
