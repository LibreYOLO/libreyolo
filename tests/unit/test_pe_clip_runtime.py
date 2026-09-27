"""Runtime behavior of the PE whole-clip path, against a real checkpoint.

These need the converted ``LibrePEt16-cls.pt`` artifact, so they are marked
``external_data`` and skip when it is not staged locally. What they cover
cannot be checked with a randomly initialized tower: the wiring between
``predict()``, the source classifier, and ``_predict_video_clip``.
"""

from __future__ import annotations

import types
from pathlib import Path

import numpy as np
import pytest

pytestmark = [pytest.mark.unit, pytest.mark.pe, pytest.mark.external_data]

WEIGHT = Path("weights/LibrePEt16-cls.pt")
pytest.importorskip("cv2")


@pytest.fixture(scope="module")
def embedder():
    if not WEIGHT.exists():
        pytest.skip(f"{WEIGHT} is not staged locally")
    from libreyolo import LibreYOLO

    return LibreYOLO(str(WEIGHT), task="embed")


@pytest.fixture(scope="module")
def clip_path(tmp_path_factory):
    import cv2

    path = tmp_path_factory.mktemp("pe") / "clip.mp4"
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (128, 128)
    )
    rng = np.random.default_rng(0)
    for _ in range(30):
        writer.write((rng.random((128, 128, 3)) * 255).astype("uint8"))
    writer.release()
    return str(path)


def test_finite_video_collapses_to_one_row(embedder, clip_path):
    results = embedder.predict(clip_path)
    assert len(results) == 1, "a whole clip must produce one Results, not one per frame"
    row = np.asarray(results[0].embeddings.data)
    assert row.shape == (1, 512)
    assert np.isclose(np.linalg.norm(row, axis=-1)[0], 1.0, atol=1e-5)
    assert results[0].path.endswith("clip.mp4")


def test_stream_true_returns_an_iterator(embedder, clip_path):
    """The streaming contract holds even though a clip collapses to one row."""
    streamed = embedder.predict(clip_path, stream=True)
    assert isinstance(streamed, types.GeneratorType) or hasattr(streamed, "__next__")
    collected = list(streamed)
    assert len(collected) == 1


def test_clip_frames_changes_the_embedding(embedder, clip_path):
    eight = np.asarray(embedder.predict(clip_path)[0].embeddings.data)
    four = np.asarray(embedder.predict(clip_path, clip_frames=4)[0].embeddings.data)
    assert np.abs(eight - four).max() > 0


def test_clip_embedding_is_deterministic(embedder, clip_path):
    first = np.asarray(embedder.predict(clip_path)[0].embeddings.data)
    second = np.asarray(embedder.predict(clip_path)[0].embeddings.data)
    assert np.abs(first - second).max() == 0.0


def test_non_positive_clip_frames_rejected(embedder, clip_path):
    with pytest.raises(ValueError, match="clip_frames must be positive"):
        embedder.predict(clip_path, clip_frames=0)


def test_save_writes_an_annotated_image(embedder, clip_path, tmp_path):
    """save=True must be honored, not silently dropped."""
    out = tmp_path / "saved"
    embedder.predict(clip_path, save=True, output_path=str(out))
    written = list(out.rglob("*.jpg")) + list(out.rglob("*.png"))
    assert written, f"save=True wrote nothing under {out}"


def test_vid_stride_rejected_as_inapplicable(embedder, clip_path):
    """vid_stride conflicts with uniform whole-clip sampling; say so loudly."""
    with pytest.raises(ValueError, match="clip_frames"):
        embedder.predict(clip_path, vid_stride=2)


def test_show_rejected_as_inapplicable(embedder, clip_path):
    with pytest.raises(NotImplementedError, match="whole-clip"):
        embedder.predict(clip_path, show=True)


def test_live_source_rejected_without_buffering(embedder):
    with pytest.raises(ValueError, match="unbounded"):
        embedder.predict(0, stream=True)


def test_image_list_stays_a_batch(embedder):
    images = [
        (np.random.default_rng(i).random((64, 64, 3)) * 255).astype("uint8")
        for i in range(3)
    ]
    results = embedder.predict(images)
    assert len(results) == 3, "an image list must not be guessed to be a clip"


def test_export_metadata_round_trips(embedder, tmp_path):
    """A reloaded clip graph must know its frame count and layout."""
    import json

    import onnx

    out = str(tmp_path / "vembed.onnx")
    embedder.export(format="onnx", video=True, clip_frames=4, output=out)
    props = {p.key: p.value for p in onnx.load(out).metadata_props}
    assert props["model_family"] == "pe"
    assert props["task"] == "embed"
    assert props["input_kind"] == "video"
    assert json.loads(props["frames"]) == 4
    assert props["input_layout"] == "BFCHW"
    assert props["video_pool"] == "mean_frame_embeddings"
    assert json.loads(props["dynamic_frames"]) is False


def test_exported_backend_uses_pe_normalization(embedder, tmp_path):
    """A reloaded PE graph must preprocess with PE stats, not ImageNet stats.

    Regression guard: without a ``pe`` branch in ``_preprocess_classify`` the
    backend silently falls through to ImageNet normalization, so the exported
    model sees different pixels than native PE and returns wrong embeddings.
    """
    import numpy as np_
    import torch

    from libreyolo.models.pe.nn import PE_MEAN, PE_STD

    out = str(tmp_path / "embed.torchscript")
    embedder.export(format="torchscript", output=out)

    from libreyolo.backends.torchscript import TorchScriptBackend

    backend = TorchScriptBackend(out, task="embed")
    assert backend.model_family == "pe"

    image = (np_.random.default_rng(0).random((240, 320, 3)) * 255).astype("uint8")
    tensor, _, _, _ = backend._preprocess_classify(image, 384, "rgb")

    # Reproduce the native transform and require an exact match.
    native, _ = embedder._get_preprocess_numpy()(image, 384)
    torch.testing.assert_close(
        tensor[0], torch.from_numpy(native), rtol=0, atol=0
    )

    # And sanity-check the stats really are symmetric, not ImageNet.
    assert PE_MEAN == (0.5, 0.5, 0.5) and PE_STD == (0.5, 0.5, 0.5)


def test_video_graph_reload_fails_loudly(embedder, tmp_path):
    """A 5D clip graph must be refused with a reason, not an opaque rank error."""
    from libreyolo.backends.torchscript import TorchScriptBackend

    out = str(tmp_path / "vembed.torchscript")
    embedder.export(format="torchscript", video=True, clip_frames=4, output=out)
    with pytest.raises(NotImplementedError, match="5D"):
        TorchScriptBackend(out, task="embed")


def test_image_graph_reload_still_works(embedder, tmp_path):
    """The guard must not catch ordinary 4D image graphs."""
    from libreyolo.backends.torchscript import TorchScriptBackend

    out = str(tmp_path / "embed.torchscript")
    embedder.export(format="torchscript", output=out)
    backend = TorchScriptBackend(out, task="embed")
    assert backend.model_family == "pe"


def test_torchscript_metadata_round_trips(embedder, tmp_path):
    import json

    import torch

    out = str(tmp_path / "embed.torchscript")
    embedder.export(format="torchscript", output=out)
    extra = {"libreyolo_metadata.json": ""}
    torch.jit.load(out, _extra_files=extra)
    meta = json.loads(extra["libreyolo_metadata.json"])
    assert meta["model_family"] == "pe"
    assert meta["model_size"] == "t16"
    assert meta["task"] == "embed"
    assert meta["imgsz"] == 384
    assert meta["embedding_dim"] == 512
    assert meta["input_kind"] == "image"
