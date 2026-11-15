"""Unit tests for the face-embedding (facial-recognition / ``embed``) task.

Hermetic — no network, no large weights. ONNX-dependent tests build a tiny
synthetic recognition graph; everything else (task aliases, the Embeddings
payload, alignment math) is pure numpy/torch.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from libreyolo.tasks import (
    SUFFIX_TO_TASK,
    normalize_task,
    suffix_to_task,
    task_to_suffix,
)
from libreyolo.models.facerec.align import (
    ARCFACE_DST_112,
    align_face,
    estimate_norm,
)
from libreyolo.utils.results import Boxes, Embeddings, Results

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Task registration / aliases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "alias",
    ["embed", "embedding", "embeddings", "facial-recognition", "facial_recognition",
     "face-recognition", "face_recognition", "recognition", "face", "faceid", "reid"],
)
def test_aliases_resolve_to_embed(alias):
    assert normalize_task(alias) == "embed"


def test_embed_suffix_roundtrip():
    assert task_to_suffix("embed") == "embed"
    assert suffix_to_task("embed") == "embed"
    assert SUFFIX_TO_TASK["embed"] == "embed"


def test_base_embed_verb_rejects_unsupported_family():
    from libreyolo.models.resnet.model import LibreResNet

    model = object.__new__(LibreResNet)
    model.family = "resnet"
    with pytest.raises(NotImplementedError, match="does not support"):
        model.embed("image.jpg")


# ---------------------------------------------------------------------------
# Embeddings payload
# ---------------------------------------------------------------------------


def test_embeddings_basic_and_dim():
    e = Embeddings(torch.randn(3, 8))
    assert e.dim == 8
    assert len(e) == 3
    norms = torch.linalg.vector_norm(e.normalized, dim=-1)
    assert torch.allclose(norms, torch.ones(3), atol=1e-5)


def test_embeddings_promotes_1d():
    e = Embeddings(np.random.rand(16).astype(np.float32))
    assert e.data.shape == (1, 16)


def test_embeddings_rejects_wrong_rank():
    with pytest.raises(ValueError):
        Embeddings(torch.randn(2, 3, 4))


def test_embeddings_similarity_single_and_matrix():
    a = Embeddings(np.eye(4, dtype=np.float32)[:2])  # two orthonormal rows
    # vs a single vector
    sim_vec = a.similarity(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    assert np.asarray(sim_vec).shape == (2,)
    assert sim_vec[0] == pytest.approx(1.0, abs=1e-5)
    assert sim_vec[1] == pytest.approx(0.0, abs=1e-5)
    # vs a gallery matrix
    sim_mat = a.similarity(a)
    assert np.asarray(sim_mat).shape == (2, 2)
    assert np.allclose(np.diag(np.asarray(sim_mat)), 1.0, atol=1e-5)


def test_embeddings_verify():
    data = np.stack([[1, 0, 0], [1, 0, 0], [0, 1, 0]]).astype(np.float32)
    e = Embeddings(data)
    assert e.verify(0, 1, threshold=0.9) is True   # identical rows
    assert e.verify(0, 2, threshold=0.5) is False  # orthogonal rows


def test_embeddings_device_roundtrip_and_index():
    e = Embeddings(torch.randn(2, 8))
    e_np = e.numpy()
    assert isinstance(e_np.data, np.ndarray)
    first = e[0]
    assert first.data.shape == (1, 8)


def test_results_carries_embeddings_slot():
    boxes = Boxes(torch.zeros((2, 4)), torch.tensor([0.9, 0.8]), torch.tensor([0.0, 0.0]))
    e = Embeddings(torch.randn(2, 8))
    r = Results(boxes=boxes, orig_shape=(100, 100), embeddings=e, names={0: "face"})
    assert r.embeddings is e
    assert "embeddings" in r._keys
    # survives device-move / slicing through _apply
    assert r.numpy().embeddings.data.shape == (2, 8)
    assert r[:1].embeddings.data.shape == (1, 8)


def test_summary_omits_vector_by_default():
    boxes = Boxes(torch.zeros((1, 4)), torch.tensor([0.9]), torch.tensor([0.0]))
    e = Embeddings(torch.randn(1, 8))
    r = Results(boxes=boxes, orig_shape=(50, 50), embeddings=e, names={0: "face"})
    row = r.summary()[0]
    assert row["embedding_dim"] == 8
    assert "embedding" not in row          # raw vector omitted by default
    row_full = r.summary(embeddings=True)[0]
    assert len(row_full["embedding"]) == 8  # opt-in includes it


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------


def test_estimate_norm_recovers_template():
    theta = np.deg2rad(23.0)
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    src = 1.37 * (ARCFACE_DST_112 @ R.T) + np.array([40.0, -15.0])
    M = estimate_norm(src, 112)
    recovered = src @ M[:, :2].T + M[:, 2]
    assert np.abs(recovered - ARCFACE_DST_112).max() < 1e-6


def test_align_face_landmarks_and_fallback():
    img = (np.random.rand(200, 200, 3) * 255).astype(np.uint8)
    lm = ARCFACE_DST_112 + np.array([40.0, 30.0])  # template shifted into the image
    aligned = align_face(img, (40, 30, 152, 142), lm, 112)
    assert aligned.shape == (112, 112, 3)
    # no landmarks -> center-crop fallback still yields a 112 crop
    fallback = align_face(img, (10, 10, 110, 110), None, 112)
    assert fallback.shape == (112, 112, 3)


# ---------------------------------------------------------------------------
# End-to-end with a synthetic ONNX recognition head
# ---------------------------------------------------------------------------


def _build_tiny_face_onnx(path: str, dim: int = 8) -> None:
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper, numpy_helper

    inp = helper.make_tensor_value_info("data", TensorProto.FLOAT, [None, 3, 112, 112])
    out = helper.make_tensor_value_info("emb", TensorProto.FLOAT, [None, dim])
    gap = helper.make_node("GlobalAveragePool", ["data"], ["gap"])
    newshape = numpy_helper.from_array(np.array([-1, 3], dtype=np.int64), name="newshape")
    reshape = helper.make_node("Reshape", ["gap", "newshape"], ["flat"])
    rng = np.random.RandomState(0)
    W = numpy_helper.from_array(rng.rand(3, dim).astype(np.float32), name="W")
    matmul = helper.make_node("MatMul", ["flat", "W"], ["emb"])
    graph = helper.make_graph([gap, reshape, matmul], "tiny_face",
                              [inp], [out], initializer=[newshape, W])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    onnx.save(model, path)


@pytest.fixture
def tiny_onnx(tmp_path):
    pytest.importorskip("onnxruntime")
    p = tmp_path / "tiny_face.onnx"
    _build_tiny_face_onnx(str(p), dim=8)
    return str(p)


def test_end_to_end_byo_boxes(tiny_onnx):
    from libreyolo.models.facerec import LibreFaceEmbedder

    model = LibreFaceEmbedder(tiny_onnx, device="cpu")
    assert model.task == "embed"
    assert model.names == {0: "face"}

    img = Image.fromarray((np.random.rand(96, 96, 3) * 255).astype(np.uint8))
    res = model(img, face_boxes=[(0, 0, 96, 96)])
    assert isinstance(res, Results)
    assert res.embeddings is not None
    assert res.embeddings.data.shape == (1, 8)
    assert float(np.linalg.norm(res.embeddings.numpy().data[0])) == pytest.approx(1.0, abs=1e-5)


def test_verify_same_image_is_one(tiny_onnx):
    from libreyolo.models.facerec import LibreFaceEmbedder

    model = LibreFaceEmbedder(tiny_onnx, device="cpu")
    img = Image.fromarray((np.random.rand(96, 96, 3) * 255).astype(np.uint8))
    out = model.verify(img, img, threshold=0.5, face_boxes_a=[(0, 0, 96, 96)],
                       face_boxes_b=[(0, 0, 96, 96)])
    assert out["similarity"] == pytest.approx(1.0, abs=1e-4)
    assert out["same_person"] is True


def test_factory_routes_onnx_embed_task(tiny_onnx):
    from libreyolo import LibreYOLO
    from libreyolo.models.facerec import LibreFaceEmbedder

    model = LibreYOLO(tiny_onnx, task="facial-recognition")
    assert isinstance(model, LibreFaceEmbedder)
    assert model.task == "embed"


def test_inference_only_guards(tiny_onnx, monkeypatch):
    from libreyolo.models.facerec import LibreFaceEmbedder

    model = LibreFaceEmbedder(tiny_onnx, device="cpu")
    for verb in ("train", "val", "export"):
        with pytest.raises(NotImplementedError):
            getattr(model, verb)()

    img = Image.fromarray((np.random.rand(64, 64, 3) * 255).astype(np.uint8))
    with pytest.raises(ValueError):
        model(img, face_boxes=[(0, 0, 64, 64)], augment=True)
    # No detector, no boxes, and the default detector unobtainable (kept
    # hermetic: the real fallback would try an auto-download here).
    monkeypatch.setattr(
        model, "default_face_detector",
        lambda: (_ for _ in ()).throw(FileNotFoundError("offline")),
    )
    with pytest.raises(RuntimeError):
        model(img)


# ---------------------------------------------------------------------------
# Named weights + auto-download wiring
# ---------------------------------------------------------------------------


def test_resolve_facerec_weight_existing_path(tiny_onnx):
    from libreyolo.models.facerec.weights import resolve_facerec_weight

    assert resolve_facerec_weight(tiny_onnx) == tiny_onnx


def test_resolve_facerec_weight_downloads_bare_name(monkeypatch, tmp_path):
    from pathlib import Path as P

    from libreyolo.models.facerec.weights import resolve_facerec_weight

    calls = {}

    def fake_download(url, dest, **kwargs):
        calls["url"] = url
        P(dest).parent.mkdir(parents=True, exist_ok=True)
        P(dest).write_bytes(b"stub")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "libreyolo.utils.download.download_url_to_path", fake_download
    )
    got = resolve_facerec_weight("librefacerec-l")
    assert P(got) == P("weights") / "librefacerec-l.onnx"
    assert P(got).exists()
    assert calls["url"].endswith("librefacerec-l/resolve/main/librefacerec-l.onnx")


def test_resolve_facerec_weight_unknown_name(tmp_path, monkeypatch):
    from libreyolo.models.facerec.weights import resolve_facerec_weight

    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError, match="Known downloadable"):
        resolve_facerec_weight("librefacerec-zz.onnx")


def test_factory_routes_librefacerec_name(monkeypatch, tmp_path):
    import shutil

    from libreyolo import LibreYOLO
    from libreyolo.models.facerec import LibreFaceEmbedder

    pytest.importorskip("onnxruntime")
    monkeypatch.chdir(tmp_path)
    wdir = tmp_path / "weights"
    wdir.mkdir()
    _build_tiny_face_onnx(str(wdir / "librefacerec-l.onnx"), dim=8)

    model = LibreYOLO("librefacerec-l")
    assert isinstance(model, LibreFaceEmbedder)
    assert model.task == "embed"

    with pytest.raises(ValueError, match="embed"):
        LibreYOLO("librefacerec-l", task="detect")


def test_cli_names_resolve_facerec():
    from libreyolo.cli.config import resolve_model_name

    assert resolve_model_name("facerec-l") == "librefacerec-l.onnx"
    assert resolve_model_name("librefacerec-l") == "librefacerec-l.onnx"


# ---------------------------------------------------------------------------
# FaceGallery + identification
# ---------------------------------------------------------------------------


def _vec(dim, hot):
    v = np.zeros(dim, dtype=np.float32)
    v[hot] = 1.0
    return v


def test_gallery_enroll_embedding_and_match():
    from libreyolo.models.facerec import FaceGallery

    g = FaceGallery()
    g.enroll_embedding("alice", _vec(8, 0))
    g.enroll_embedding("alice", _vec(8, 1))  # second reference, different pose
    g.enroll_embedding("bob", _vec(8, 2))
    assert len(g) == 2
    assert g.identities == ["alice", "bob"]
    assert "alice" in g and "carol" not in g

    # max-cosine over references: a query near alice's 2nd ref still hits alice
    matches = g.match(_vec(8, 1), top_k=2, threshold=0.3)
    assert matches[0][0] == ("alice", pytest.approx(1.0))
    # below-threshold queries return no name, never the nearest wrong person
    q = np.full(8, 0.1, dtype=np.float32)
    assert g.match(_vec(8, 5), threshold=0.5) == [[]]

    assert g.remove("bob") == 1
    assert g.identities == ["alice"]


def test_gallery_generic_root_export_and_face_alias():
    from libreyolo import FaceGallery, Gallery
    from libreyolo.models.facerec import FaceGallery as LegacyFaceGallery

    assert FaceGallery is Gallery
    assert LegacyFaceGallery is Gallery
    assert repr(Gallery()).startswith("Gallery(")


class _FakeBoxes:
    def __init__(self, conf):
        self.conf = np.asarray(conf, dtype=np.float32)

    def __len__(self):
        return int(self.conf.shape[0])


class _TwoFaceEmbedder:
    """Reference photo with the subject (conf 0.9) plus a bystander (conf 0.3)."""

    _weights_fingerprint = "region-model"

    def predict(self, _source):
        from types import SimpleNamespace

        return SimpleNamespace(
            embeddings=Embeddings(np.stack([_vec(8, 0), _vec(8, 1)], axis=0)),
            boxes=_FakeBoxes([0.3, 0.9]),
        )


def test_gallery_enroll_defaults_to_most_prominent_row():
    from libreyolo import Gallery

    gallery = Gallery(_TwoFaceEmbedder())
    assert gallery.enroll("person", "image.jpg") == 1
    assert len(gallery._vectors) == 1
    # The stored reference is the conf-0.9 row, not the bystander.
    assert gallery.match(_vec(8, 1), threshold=0.9)[0][0][0] == "person"
    assert gallery.match(_vec(8, 0), threshold=0.9) == [[]]


def test_gallery_enroll_select_all_stores_every_region_row():
    from libreyolo import Gallery

    gallery = Gallery(_TwoFaceEmbedder())
    assert gallery.enroll("person", "image.jpg", select="all") == 2
    assert len(gallery._vectors) == 2
    assert gallery.match(_vec(8, 0), threshold=0.9)[0][0][0] == "person"
    with pytest.raises(ValueError, match="select"):
        gallery.enroll("person", "image.jpg", select="first")


def test_gallery_dim_mismatch_raises():
    from libreyolo.models.facerec import FaceGallery

    g = FaceGallery()
    g.enroll_embedding("alice", _vec(8, 0))
    with pytest.raises(ValueError, match="dim mismatch"):
        g.enroll_embedding("bob", _vec(16, 0))
    with pytest.raises(ValueError, match="dim mismatch"):
        g.match(_vec(16, 0))


def test_gallery_save_load_roundtrip(tmp_path):
    from libreyolo.models.facerec import FaceGallery

    g = FaceGallery()
    g.enroll_embedding("alice", _vec(8, 0))
    g.enroll_embedding("bob", _vec(8, 2))
    p = tmp_path / "team.gallery.npz"
    g.save(p)

    loaded = FaceGallery.load(p)
    assert loaded.identities == ["alice", "bob"]
    assert loaded.dim == 8
    assert loaded.match(_vec(8, 2))[0][0][0] == "bob"

    with pytest.raises(ValueError):
        FaceGallery().save(tmp_path / "empty.npz")


def test_gallery_loads_legacy_face_archive(tmp_path):
    import json

    from libreyolo import Gallery

    path = tmp_path / "legacy.gallery.npz"
    np.savez_compressed(
        path,
        vectors=np.stack([_vec(8, 0), _vec(8, 2)]),
        names=np.asarray(["alice", "bob"]),
        meta=np.asarray(
            json.dumps(
                {
                    "format": "libreyolo-face-gallery-v1",
                    "dim": 8,
                    "model_fingerprint": None,
                }
            )
        ),
    )

    loaded = Gallery.load(path)
    assert loaded.identities == ["alice", "bob"]
    assert loaded.match(_vec(8, 2))[0][0][0] == "bob"


def test_gallery_model_fingerprint_guard(tmp_path, tiny_onnx):
    from libreyolo.models.facerec import FaceGallery, LibreFaceEmbedder

    model = LibreFaceEmbedder(tiny_onnx, device="cpu")
    g = FaceGallery(embedder=model)
    g.enroll_embedding("alice", _vec(8, 0))
    p = tmp_path / "team.gallery.npz"
    g.save(p)

    # same model loads fine
    FaceGallery.load(p, embedder=model)

    # different weights file -> fingerprint mismatch
    other_path = tmp_path / "other.onnx"
    _build_tiny_face_onnx(str(other_path), dim=8)
    other = LibreFaceEmbedder(str(other_path), device="cpu")
    other._weights_fingerprint = "deadbeefdeadbeef"
    with pytest.raises(ValueError, match="different embedding model"):
        FaceGallery.load(p, embedder=other)


def test_gallery_unknown_fingerprint_matches_without_binding():
    from libreyolo import Gallery

    class _NoFingerprint:
        """No weights file, no state dict: fingerprint cannot be computed."""

    bound = Gallery()
    bound._model_fingerprint = "cafecafecafecafe"
    bound.enroll_embedding("alice", _vec(8, 0))
    # A model without a computable fingerprint still matches a bound gallery.
    assert bound.match(_vec(8, 0), model=_NoFingerprint())[0][0][0] == "alice"

    # Matching never binds: an unbound gallery accepts two different models.
    unbound = Gallery()
    unbound.enroll_embedding("alice", _vec(8, 0))
    model_a = type("_A", (), {"_weights_fingerprint": "aaaa"})()
    model_b = type("_B", (), {"_weights_fingerprint": "bbbb"})()
    unbound.match(_vec(8, 0), model=model_a)
    unbound.match(_vec(8, 0), model=model_b)
    assert unbound._model_fingerprint is None


def test_stack_result_embeddings_ignores_zero_row_payloads():
    from libreyolo.utils.results import Results, stack_result_embeddings

    empty = Results(
        boxes=None,
        orig_shape=(4, 4),
        embeddings=Embeddings(np.zeros((0, 0), dtype=np.float32)),
    )
    full = Results(
        boxes=None,
        orig_shape=(4, 4),
        embeddings=Embeddings(np.stack([_vec(8, 0), _vec(8, 1)])),
    )
    assert stack_result_embeddings([empty, full]).shape == (2, 8)
    assert stack_result_embeddings([empty]).shape == (0, 0)


def test_gallery_enroll_from_images_and_identify(tiny_onnx, tmp_path):
    from libreyolo.models.facerec import FaceGallery, LibreFaceEmbedder

    model = LibreFaceEmbedder(tiny_onnx, device="cpu")
    rng = np.random.RandomState(7)
    img_a = Image.fromarray((rng.rand(96, 96, 3) * 255).astype(np.uint8))
    path_a = tmp_path / "alice.jpg"
    img_a.save(path_a)

    g = FaceGallery(embedder=model)
    # enroll needs a face source; the tiny graph has no detector, so BYO boxes
    res = model(img_a, face_boxes=[(0, 0, 96, 96)])
    g.enroll_embedding("alice", res.embeddings.data[0])
    assert len(g) == 1

    # identify the same face -> alice at ~1.0 cosine
    out = model(img_a, face_boxes=[(0, 0, 96, 96)], gallery=g, threshold=0.9)
    assert out.identities is not None
    assert out.identities.name == ["alice"]
    assert float(out.identities.score[0]) == pytest.approx(1.0, abs=1e-5)
    assert out.summary()[0]["identity"] == "alice"

    # impossible threshold -> unknown, but the raw best score stays visible
    out2 = model(img_a, face_boxes=[(0, 0, 96, 96)], gallery=g, threshold=1.1)
    assert out2.identities.name == [None]
    assert float(out2.identities.score[0]) == pytest.approx(1.0, abs=1e-5)


def test_identify_empty_gallery_raises(tiny_onnx):
    from libreyolo.models.facerec import FaceGallery, LibreFaceEmbedder

    model = LibreFaceEmbedder(tiny_onnx, device="cpu")
    img = Image.fromarray((np.random.rand(64, 64, 3) * 255).astype(np.uint8))
    with pytest.raises(ValueError, match="empty FaceGallery"):
        model(img, face_boxes=[(0, 0, 64, 64)], gallery=FaceGallery())


def test_identities_payload_slicing_and_results_plumbing():
    from libreyolo.utils.results import Identities

    ids = Identities(["alice", None, "bob"], np.array([0.9, 0.2, 0.7]))
    assert len(ids) == 3
    assert ids.name[1] is None
    assert ids[0].name == ["alice"]
    assert ids[1:].name == [None, "bob"]
    assert ids[[0, 2]].name == ["alice", "bob"]
    assert ids.numpy() is ids and ids.cpu() is ids

    boxes = Boxes(
        torch.tensor([[0, 0, 10, 10], [5, 5, 20, 20], [8, 8, 30, 30]], dtype=torch.float32),
        torch.tensor([0.9, 0.8, 0.7]),
        torch.zeros(3),
    )
    r = Results(boxes=boxes, orig_shape=(64, 64), identities=ids)
    sliced = r[:1]
    assert sliced.identities.name == ["alice"]
    r2 = r.update(identities=Identities(["carol"] * 3, np.ones(3)))
    assert r2.identities.name == ["carol", "carol", "carol"]


# ---------------------------------------------------------------------------
# Review follow-ups (PR #654)
# ---------------------------------------------------------------------------


def test_enroll_through_a_different_model_raises(tiny_onnx, tmp_path):
    """A gallery holds one embedding space; enrolling via another model fails.

    Without this guard the gallery keeps the first model's fingerprint and
    silently appends a vector that will never compare correctly.
    """
    from libreyolo.models.facerec import FaceGallery, LibreFaceEmbedder

    model_a = LibreFaceEmbedder(tiny_onnx, device="cpu")
    other_path = tmp_path / "other.onnx"
    _build_tiny_face_onnx(str(other_path), dim=8)
    model_b = LibreFaceEmbedder(str(other_path), device="cpu")
    model_b._weights_fingerprint = "deadbeefdeadbeef"

    img = Image.fromarray((np.random.rand(96, 96, 3) * 255).astype(np.uint8))
    path = tmp_path / "face.jpg"
    img.save(path)

    gallery = FaceGallery(embedder=model_a)
    gallery.enroll_embedding("alice", _vec(8, 0))

    with pytest.raises(ValueError, match="different embedding model"):
        gallery.enroll("bob", str(path), embedder=model_b)


def test_video_source_routes_to_video_inference(tiny_onnx, monkeypatch):
    """A video path routes to the video runner AND its frames actually embed.

    The fake driver invokes the frame callback the way the real one does; a
    fake that only records the call would pass even when every frame raises.
    """
    from libreyolo.models.facerec import LibreFaceEmbedder
    from libreyolo.models.facerec import inference as inference_module

    model = LibreFaceEmbedder(tiny_onnx, device="cpu")
    called = {}
    frame = Image.fromarray((np.random.rand(96, 96, 3) * 255).astype(np.uint8))

    def fake_run_video_inference(source, predict_frame, **kwargs):
        called["source"] = str(source)
        called["stride"] = kwargs.get("vid_stride")
        yield predict_frame(frame)

    monkeypatch.setattr(
        inference_module, "run_video_inference", fake_run_video_inference
    )
    monkeypatch.setattr(
        inference_module, "collect_video_results", lambda gen, src, stride: list(gen)
    )

    # Caller-supplied boxes must survive into every frame: with no detector,
    # dropping them left the frame callback with no face source at all.
    results = model("clip.mp4", face_boxes=[(0, 0, 96, 96)], vid_stride=2)
    assert called["source"] == "clip.mp4"
    assert called["stride"] == 2
    assert len(results) == 1
    assert results[0].embeddings.data.shape == (1, 8)


def test_video_with_gallery_identifies_each_frame(tiny_onnx, monkeypatch):
    from libreyolo.models.facerec import FaceGallery, LibreFaceEmbedder
    from libreyolo.models.facerec import inference as inference_module

    model = LibreFaceEmbedder(tiny_onnx, device="cpu")
    frame = Image.fromarray((np.random.rand(96, 96, 3) * 255).astype(np.uint8))

    reference = model(frame, face_boxes=[(0, 0, 96, 96)])
    gallery = FaceGallery(embedder=model)
    gallery.enroll_embedding("alice", reference.embeddings.data[0])

    monkeypatch.setattr(
        inference_module,
        "run_video_inference",
        lambda source, predict_frame, **kw: iter([predict_frame(frame)]),
    )
    monkeypatch.setattr(
        inference_module, "collect_video_results", lambda gen, src, stride: list(gen)
    )

    results = model(
        "clip.mp4", face_boxes=[(0, 0, 96, 96)], gallery=gallery, threshold=0.9
    )
    assert results[0].identities.name == ["alice"]
