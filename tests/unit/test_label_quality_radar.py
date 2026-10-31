"""Unit tests for LibreLabel data-quality features: Radar, geometry lint, dup fix.

All pure-logic / filesystem-only -- no model weights, no network. The model
forward pass is stubbed so the matching/classification logic is exercised in
isolation (and so the trust contract -- "Radar writes nothing" -- can be proven
behaviorally with a session double that refuses writes).
"""

import pytest

from libreyolo.label import radar
from libreyolo.label.quality import lint_annotations

pytestmark = pytest.mark.unit


# --- IoU --------------------------------------------------------------------
def test_iou_identical_disjoint_and_partial():
    assert radar.iou((0.5, 0.5, 0.4, 0.4), (0.5, 0.5, 0.4, 0.4)) == pytest.approx(1.0)
    assert radar.iou((0.1, 0.1, 0.1, 0.1), (0.9, 0.9, 0.1, 0.1)) == 0.0
    # two equal boxes offset by half their width overlap on half their area
    v = radar.iou((0.25, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5))
    assert 0.3 < v < 0.34  # intersection 0.25*0.5 / union (0.5*0.5*2 - inter)


# --- classify ---------------------------------------------------------------
def _pred(cls, cx, cy, w, h, conf, mapped=True, name="x"):
    return {"cls": cls, "cx": cx, "cy": cy, "w": w, "h": h,
            "conf": conf, "mapped": mapped, "name": name}


def test_classify_agreement_yields_nothing():
    labels = [(0, 0.5, 0.5, 0.2, 0.2)]
    preds = [_pred(0, 0.5, 0.5, 0.2, 0.2, 0.9)]
    findings, score = radar.classify(labels, preds)
    assert findings == [] and score == 0.0


def test_classify_class_slip():
    labels = [(0, 0.5, 0.5, 0.2, 0.2)]
    preds = [_pred(1, 0.5, 0.5, 0.2, 0.2, 0.88, name="dog")]
    findings, _ = radar.classify(labels, preds)
    assert len(findings) == 1
    assert findings[0]["type"] == "class"
    assert findings[0]["label_cls"] == 0 and findings[0]["pred_cls"] == 1


def test_classify_phantom_for_unmatched_label():
    findings, score = radar.classify([(0, 0.5, 0.5, 0.2, 0.2)], [])
    assert len(findings) == 1 and findings[0]["type"] == "phantom"
    assert score == pytest.approx(0.5)


def test_classify_miss_for_confident_unmatched_pred():
    labels = [(0, 0.1, 0.1, 0.1, 0.1)]
    preds = [_pred(0, 0.1, 0.1, 0.1, 0.1, 0.9),       # matches the label -> agreement
             _pred(0, 0.7, 0.7, 0.2, 0.2, 0.9)]        # nothing labelled here -> miss
    findings, _ = radar.classify(labels, preds, miss_conf=0.55)
    assert [f["type"] for f in findings] == ["miss"]


def test_classify_low_conf_pred_is_not_a_miss():
    labels = [(0, 0.1, 0.1, 0.1, 0.1)]
    preds = [_pred(0, 0.1, 0.1, 0.1, 0.1, 0.90),      # matches the label -> agreement
             _pred(0, 0.7, 0.7, 0.2, 0.2, 0.30)]       # unmatched but below miss_conf
    findings, _ = radar.classify(labels, preds, miss_conf=0.55)
    assert findings == []  # the low-conf stray pred is ignored -> no noise


def test_classify_ignores_unmapped_predictions():
    # an unmapped detection (a class the dataset doesn't have) must not create a
    # class-slip or a miss; the human's box just looks unmatched -> phantom only.
    labels = [(0, 0.5, 0.5, 0.2, 0.2)]
    preds = [_pred(None, 0.5, 0.5, 0.2, 0.2, 0.95, mapped=False)]
    findings, _ = radar.classify(labels, preds)
    assert [f["type"] for f in findings] == ["phantom"]


# --- geometry linter --------------------------------------------------------
def test_lint_flags_tiny_box():
    issues = lint_annotations([{"type": "box", "cls": 0, "cx": 0.5, "cy": 0.5,
                                "w": 0.002, "h": 0.002}], imgsz=640)
    assert len(issues) == 1 and issues[0]["type"] == "tiny"


def test_lint_flags_sliver_but_not_tiny():
    issues = lint_annotations([{"type": "box", "cls": 0, "cx": 0.5, "cy": 0.5,
                                "w": 0.9, "h": 0.01}], imgsz=640)  # 6.4px tall, ar 90:1
    assert len(issues) == 1 and issues[0]["type"] == "sliver"


def test_lint_flags_full_frame():
    issues = lint_annotations([{"type": "box", "cls": 0, "cx": 0.5, "cy": 0.5,
                                "w": 0.99, "h": 0.99}], imgsz=640)
    assert len(issues) == 1 and issues[0]["type"] == "fullframe"


def test_lint_passes_a_normal_box():
    assert lint_annotations([{"type": "box", "cls": 0, "cx": 0.5, "cy": 0.5,
                              "w": 0.3, "h": 0.4}], imgsz=640) == []


# --- trust contract ---------------------------------------------------------
def test_radar_scan_never_calls_write_label():
    """Behavioral proof of the trust contract: a scan must read, never mutate.

    Stronger than grepping the source -- it runs ``scan_dataset`` end to end over
    a session double whose ``write_label`` blows up if touched, so any future
    write path (not just a literal ``write_label`` string) trips the test.
    """
    from pathlib import Path

    class _NoWriteSession:
        names = ["cat", "dog"]
        _deleted: set = set()

        def __len__(self):
            return 2

        def image_path(self, i):
            return Path(f"img{i}.jpg")

        def read_label(self, i):
            return ([{"type": "box", "cls": 0, "cx": 0.5, "cy": 0.5,
                      "w": 0.2, "h": 0.2}], True)

        def write_label(self, *a, **k):
            raise AssertionError("Radar must never write labels")

    def predict(path, names, model_name, conf):
        # a confident class slip (label says 'cat', model says 'dog') -> a finding
        return [{"cls": 1, "cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2,
                 "conf": 0.9, "name": "dog", "mapped": True}]

    out = radar.scan_dataset(predict, _NoWriteSession())
    assert out["type"] == "done" and out["scanned"] == 2
    assert out["flagged"] == 2   # both images surfaced the slip; nothing was written


# --- dataset integration (filesystem only) ----------------------------------
def _make_split_dataset(root, *, leak=False):
    """Dataset with a train image; optionally a duplicate val image (leakage)."""
    from PIL import Image

    (root / "images" / "train").mkdir(parents=True)
    im = Image.new("RGB", (40, 30), (10, 120, 200))
    im.save(root / "images" / "train" / "a.jpg")
    val_line = "val: images/val\n"
    if leak:
        (root / "images" / "val").mkdir(parents=True)
        im.save(root / "images" / "val" / "a.jpg")  # identical bytes -> same dHash
    yaml_path = root / "data.yaml"
    yaml_path.write_text(
        f"path: {root.as_posix()}\ntrain: images/train\n{val_line}"
        "nc: 2\nnames:\n  0: cat\n  1: dog\n",
        encoding="utf-8",
    )
    return yaml_path


def test_dataset_quality_surfaces_tiny_box(tmp_path):
    from libreyolo.label.dataset import DatasetSession

    ds = DatasetSession(str(_make_split_dataset(tmp_path)))
    ds.write_label(0, [{"type": "box", "cls": 0, "cx": 0.5, "cy": 0.5,
                        "w": 0.002, "h": 0.002}])
    q = ds.quality(imgsz=640)
    assert q["issues"] == 1
    assert q["counts"]["tiny"] == 1
    assert q["flagged"][0]["id"] == 0


def test_resolve_duplicates_keeps_train_deletes_val(tmp_path):
    from libreyolo.label.dataset import DatasetSession

    ds = DatasetSession(str(_make_split_dataset(tmp_path, leak=True)))
    assert len(ds) == 2
    # ids: 0 = train copy, 1 = val copy (same image -> leakage)
    train_path = ds.image_path(0)
    val_path = ds.image_path(1)
    assert train_path.parts[-2] == "train" and val_path.parts[-2] == "val"

    res = ds.resolve_duplicates([0, 1])
    assert res["kept"] == 0
    assert [r["id"] for r in res["removed"]] == [1]
    assert train_path.exists() and not val_path.exists()
    # id stays stable (tombstone), surfaced as "deleted" in the listing
    rows = {r["id"]: r["status"] for r in ds.list_images()}
    assert rows[1] == "deleted"
    # the live count drops the tombstone
    assert ds.stats()["total"] == 1


def test_resolve_duplicates_read_only_raises(tmp_path):
    from libreyolo.label.dataset import DatasetSession

    root = tmp_path / "my" / "images" / "proj"  # ambiguous 'images' -> read-only
    ds = DatasetSession(str(_make_split_dataset(root, leak=True)))
    assert ds.writable is False
    with pytest.raises(RuntimeError):
        ds.resolve_duplicates([0, 1])


def test_boost_agreement_resolves_class_names_to_dataset_space():
    # H1 regression: the base model predicts in its own class space (e.g. COCO);
    # agreement must be measured in the DATASET's space by resolving the predicted
    # class *name*, not the raw index. A detector emitting raw index 5 whose name
    # is "cat" must count as dataset class 0 ("cat"), never as index 5.
    import numpy as np

    from libreyolo.label.boost import BoostEngine

    class _Sess:
        names = ["cat", "dog"]

    class _Boxes:
        def __init__(self, cls, conf):
            self._c = np.array(cls)
            self._k = np.array(conf)

        def __len__(self):
            return len(self._c)

        def numpy(self):
            return self

        @property
        def cls(self):
            return self._c

        @property
        def conf(self):
            return self._k

    class _Res:
        def __init__(self, boxes, names):
            self.boxes = boxes
            self.names = names

    def model_for(name):
        return lambda img: _Res(_Boxes([5], [0.9]), {5: name})  # alien raw idx 5

    be = BoostEngine(_Sess(), model_name="x")
    imgs, want, names = ["a.jpg"], {"a.jpg": 0}, ["cat", "dog"]  # dataset idx 0 == "cat"
    assert be._agreement(model_for("cat"), imgs, want, names) == 1.0
    assert be._agreement(model_for("dog"), imgs, want, names) == 0.0


def test_resolve_duplicates_refuses_list_of_txt_split(tmp_path):
    # M1 regression: a split defined by a LIST of .txt manifests must not be pruned
    # (deleting a file there leaves a dangling manifest row).
    from libreyolo.label.dataset import DatasetSession

    ds = DatasetSession(str(_make_split_dataset(tmp_path, leak=True)))
    ds._split_sources["val"] = ["/data/train2017.txt", "/data/extra.txt"]
    with pytest.raises(RuntimeError):
        ds.resolve_duplicates([0, 1])
    assert ds.image_path(1).exists()  # guard fires before any file is touched


def test_radar_scan_end_to_end_with_stub_model(tmp_path):
    from libreyolo.label.dataset import DatasetSession

    ds = DatasetSession(str(_make_split_dataset(tmp_path)))
    ds.write_label(0, [{"type": "box", "cls": 0, "cx": 0.5, "cy": 0.5,
                        "w": 0.25, "h": 0.5}])  # human labelled it "cat"

    def stub_predict(image_path, names, model_name, conf):
        # the model is sure it's a "dog" in the same place -> a class slip
        return [_pred(1, 0.5, 0.5, 0.25, 0.5, 0.92, name="dog")]

    out = radar.scan_dataset(stub_predict, ds, conf=0.25)
    assert out["scanned"] == 1 and out["flagged"] == 1
    assert out["deck"][0]["id"] == 0
    assert out["deck"][0]["counts"].get("class") == 1
    assert out["findings"][0][0]["type"] == "class"


def test_has_unsupported_rows_flags_keypoints():
    # Codex P1: keypoint/pose rows must be detected so their files stay read-only.
    from libreyolo.label.labelio import has_unsupported_rows

    assert has_unsupported_rows("0 0.5 0.5 0.2 0.2\n") is False            # box
    assert has_unsupported_rows("1 0.1 0.1 0.4 0.1 0.25 0.48\n") is False  # polygon
    assert has_unsupported_rows("0 " + " ".join(["0.5"] * 55) + "\n") is True  # pose (cls+bbox+17*3)


def test_pose_dataset_is_view_only(tmp_path):
    from PIL import Image

    from libreyolo.label.dataset import DatasetSession

    (tmp_path / "images" / "train").mkdir(parents=True)
    Image.new("RGB", (20, 10)).save(tmp_path / "images" / "train" / "a.jpg")
    (tmp_path / "data.yaml").write_text(
        f"path: {tmp_path.as_posix()}\ntrain: images/train\n"
        "kpt_shape: [17, 3]\nnc: 1\nnames:\n  0: person\n", encoding="utf-8")
    ds = DatasetSession(str(tmp_path / "data.yaml"))
    assert ds.writable is False and "keypoint" in ds.reason.lower()


def test_write_label_rejects_unsupported_file(tmp_path):
    # Codex round 2: the write path must re-check the read-only contract.
    from libreyolo.label.dataset import DatasetSession

    ds = DatasetSession(str(_make_split_dataset(tmp_path)))
    lp = tmp_path / "labels" / "train"
    lp.mkdir(parents=True, exist_ok=True)
    (lp / "a.txt").write_text("0 " + " ".join(["0.5"] * 55) + "\n")  # pose row
    _anns, editable = ds.read_label(0)
    assert editable is False
    with pytest.raises(RuntimeError):
        ds.write_label(0, [{"type": "box", "cls": 0, "cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2}])


def test_images_substring_ancestor_is_read_only(tmp_path):
    # Codex round 2: an ancestor like "images_2026" mis-derives the label path.
    from libreyolo.label.dataset import DatasetSession

    ds = DatasetSession(str(_make_split_dataset(tmp_path / "images_2026" / "proj")))
    assert ds.writable is False


def test_mask_dataset_is_view_only(tmp_path):
    from PIL import Image

    from libreyolo.label.dataset import DatasetSession

    (tmp_path / "images" / "train").mkdir(parents=True)
    Image.new("RGB", (20, 10)).save(tmp_path / "images" / "train" / "a.jpg")
    (tmp_path / "data.yaml").write_text(
        f"path: {tmp_path.as_posix()}\ntrain: images/train\n"
        "masks_dir: masks\nnc: 1\nnames:\n  0: road\n", encoding="utf-8")
    ds = DatasetSession(str(tmp_path / "data.yaml"))
    assert ds.writable is False


def test_degenerate_polygon_dropped():
    # Codex P2: a collapsed (collinear) polygon would yield a zero-area box -> drop it.
    from libreyolo.label.labelio import sanitize_annotations

    flat = {"type": "poly", "cls": 0, "points": [0.1, 0.5, 0.5, 0.5, 0.9, 0.5, 0.5, 0.5]}
    assert sanitize_annotations([flat], nc=2) == []
    good = {"type": "poly", "cls": 0, "points": [0.1, 0.1, 0.4, 0.1, 0.4, 0.4, 0.1, 0.4]}
    assert len(sanitize_annotations([good], nc=2)) == 1


def test_write_label_epoch_guard_rejects_stale_save(tmp_path):
    # H1 regression: a save carrying an epoch from a since-switched project is
    # rejected, so an in-flight save can never land in the wrong dataset.
    from libreyolo.label.dataset import DatasetSession
    from libreyolo.label.server import _LabelState

    ds = DatasetSession(str(_make_split_dataset(tmp_path)))
    st = _LabelState(ds, assist=False)
    box = [{"type": "box", "cls": 0, "cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2}]
    count, rev = st.write_label(0, box, epoch=0)   # matches the current epoch -> (count, rev)
    assert count == 1 and rev > 0
    assert st.write_label(0, box)[0] == 1          # no epoch -> always allowed (back-compat)
    with pytest.raises(RuntimeError):
        st.write_label(0, box, epoch=5)            # stale epoch -> rejected


def test_fractional_class_id_is_unsupported():
    # Codex/CodeRabbit: a "0.5" class token must NOT be coerced to 0 and saved;
    # the file is marked read-only instead.
    from libreyolo.label.labelio import has_unsupported_rows

    assert has_unsupported_rows("0.5 0.5 0.5 0.2 0.2\n") is True
    assert has_unsupported_rows("0 0.5 0.5 0.2 0.2\n") is False


def test_my_images_ancestor_stays_writable(tmp_path):
    # Codex round 3: a "my_images" ancestor doesn't start with "images", so
    # img2label_paths leaves it alone -> the dataset must stay writable.
    from libreyolo.label.dataset import DatasetSession

    ds = DatasetSession(str(_make_split_dataset(tmp_path / "my_images" / "proj")))
    assert ds.writable is True


def test_diagonal_collinear_polygon_dropped():
    # Codex round 3: a diagonal collinear polygon has positive bbox extents but
    # zero area -- the area check must still drop it.
    from libreyolo.label.labelio import sanitize_annotations

    diag = {"type": "poly", "cls": 0, "points": [0.0, 0.0, 0.5, 0.5, 1.0, 1.0]}
    assert sanitize_annotations([diag], nc=2) == []


def test_class_slip_requires_confidence():
    # radar.py: a class slip below miss_conf is consumed silently (no finding) so
    # the deck isn't flooded with weak disagreements.
    labels = [(0, 0.5, 0.5, 0.2, 0.2)]
    weak, _ = radar.classify(labels, [_pred(1, 0.5, 0.5, 0.2, 0.2, 0.30, name="dog")],
                             miss_conf=0.55)
    strong, _ = radar.classify(labels, [_pred(1, 0.5, 0.5, 0.2, 0.2, 0.80, name="dog")],
                               miss_conf=0.55)
    assert not any(f["type"] == "class" for f in weak)
    assert any(f["type"] == "class" for f in strong)


def test_write_label_rejects_tombstoned_id(tmp_path):
    # Codex round 3: a stale tab must not recreate a label for a quarantined image.
    from libreyolo.label.dataset import DatasetSession

    ds = DatasetSession(str(_make_split_dataset(tmp_path, leak=True)))
    res = ds.resolve_duplicates([0, 1])              # collapse the leak pair
    removed = res["removed"][0]["id"]
    with pytest.raises(RuntimeError):
        ds.write_label(removed, [{"type": "box", "cls": 0, "cx": 0.5, "cy": 0.5,
                                  "w": 0.2, "h": 0.2}])


def _make_task_dataset(root, task):
    from PIL import Image

    (root / "images" / "train").mkdir(parents=True)
    Image.new("RGB", (20, 10)).save(root / "images" / "train" / "a.jpg")
    (root / "data.yaml").write_text(
        f"path: {root.as_posix()}\ntrain: images/train\ntask: {task}\n"
        "nc: 1\nnames:\n  0: thing\n", encoding="utf-8")
    return root / "data.yaml"


def test_classify_and_depth_datasets_are_view_only(tmp_path):
    # task-only classify/depth yamls must be view-only even without an explicit
    # dense-label dir. OBB is now authorable (4-corner polygons), so it is
    # intentionally writable and excluded here.
    from libreyolo.label.dataset import DatasetSession

    for task in ("classify", "depth"):
        ds = DatasetSession(str(_make_task_dataset(tmp_path / task, task)))
        assert ds.writable is False, f"task={task} should be view-only"

    obb = DatasetSession(str(_make_task_dataset(tmp_path / "obb", "obb")))
    assert obb.writable is True, "OBB is authorable and should be writable"


def test_parse_annotations_skips_fractional_class():
    # Codex round 4: a "0.5" class must NOT be coerced to 0 and counted/trained;
    # the row is skipped entirely (and the file is read-only via has_unsupported_rows).
    from libreyolo.label.labelio import parse_annotations

    anns = parse_annotations("0.5 0.5 0.5 0.2 0.2\n1 0.5 0.5 0.2 0.2\n")
    assert [a["cls"] for a in anns] == [1]   # only the valid integer-class row


def test_locate_optin_rejects_falsey_values(monkeypatch):
    # Codex round 4: LIBRELABEL_ENABLE_LOCATE=0/false must keep LocateAnything off
    # (a bare truthiness test would treat the non-empty "0" string as enabled).
    from libreyolo.label.assist import AssistEngine

    eng = AssistEngine(enabled=True)
    for val in ("0", "false", "no", ""):
        monkeypatch.setenv("LIBRELABEL_ENABLE_LOCATE", val)
        assert eng.locate_available() is False


def test_resolve_duplicates_keeps_labelled_survivor(tmp_path):
    # Codex round 4: when only the non-train copy has labels, Fix must keep the
    # labelled image, not silently turn it unlabelled.
    from libreyolo.label.dataset import DatasetSession

    ds = DatasetSession(str(_make_split_dataset(tmp_path, leak=True)))
    ds.write_label(1, [{"type": "box", "cls": 0, "cx": 0.5, "cy": 0.5,
                        "w": 0.2, "h": 0.2}])   # only the val copy (id 1) is labelled
    res = ds.resolve_duplicates([0, 1])
    assert res["kept"] == 1 and res["removed"][0]["id"] == 0


def test_resolve_duplicates_refuses_inline_image_list(tmp_path):
    # Codex round 4: an inline YAML image-list split still references the file we'd
    # move, so pruning must be refused (same as a .txt manifest).
    from libreyolo.label.dataset import DatasetSession

    ds = DatasetSession(str(_make_split_dataset(tmp_path, leak=True)))
    ds._split_sources["val"] = ["images/val/a.jpg"]
    with pytest.raises(RuntimeError):
        ds.resolve_duplicates([0, 1])


def test_insights_fix_epoch_guard(tmp_path):
    # Codex round 4: a Fix request carrying a stale epoch must be rejected before
    # touching files (same guard as label saves).
    from libreyolo.label.dataset import DatasetSession
    from libreyolo.label.server import _LabelState

    ds = DatasetSession(str(_make_split_dataset(tmp_path, leak=True)))
    st = _LabelState(ds, assist=False)
    with pytest.raises(RuntimeError):
        st.resolve_duplicates([0, 1], epoch=99)   # stale epoch -> refused
    assert ds.image_path(1).exists()              # nothing was moved


def test_box_parser_rejects_fractional_class():
    # Codex round 5: the box-only parser (the format oracle) must use the same
    # integer-class contract -> a "0.5" row is non-box, not silently class 0.
    from libreyolo.label.labelio import parse_label_text

    boxes, has_non_box = parse_label_text("0.5 0.5 0.5 0.2 0.2\n")
    assert boxes == [] and has_non_box is True
    boxes2, nb2 = parse_label_text("0 0.5 0.5 0.2 0.2\n")
    assert len(boxes2) == 1 and nb2 is False


def test_weight_is_local(tmp_path):
    # Codex round 5: the offline weight guard shared by assist/embed/boost.
    from libreyolo.label.assist import weight_is_local

    w = tmp_path / "best.pt"
    w.write_bytes(b"x")
    assert weight_is_local(str(w)) is True
    assert weight_is_local(str(tmp_path / "definitely-absent-zzzz.pt")) is False


def test_store_pending_refuses_after_switch(tmp_path):
    # Codex round 5: the prelabel pending write is atomic with the project switch.
    from libreyolo.label.dataset import DatasetSession
    from libreyolo.label.server import _LabelState

    ds = DatasetSession(str(_make_split_dataset(tmp_path)))
    st = _LabelState(ds, assist=False)
    sugg = [{"cls": 0, "cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2, "mapped": True}]
    assert st.store_pending(0, sugg, ds) is True        # same session -> stored
    assert st.engine.get_pending(0) == sugg
    assert st.store_pending(0, sugg, object()) is False  # switched-away identity -> refused


def test_non_finite_coords_are_unsupported():
    # Codex round 6: nan/inf coords must mark the file read-only and never reach
    # parse output (where they'd become invalid JSON / "nan" labels).
    from libreyolo.label.labelio import has_unsupported_rows, parse_annotations

    assert has_unsupported_rows("0 nan 0.5 0.2 0.2\n") is True
    assert has_unsupported_rows("0 inf 0.5 0.2 0.2\n") is True
    out = parse_annotations("0 nan 0.5 0.2 0.2\n1 0.5 0.5 0.2 0.2\n")
    assert out == [{"type": "box", "cls": 1, "cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2}]


def test_autolabel_aborts_when_weights_missing(tmp_path):
    # Codex round 6: a systemic model-load failure must propagate (so the stream
    # surfaces it) instead of being swallowed per-image into a "0 boxes" finish.
    from libreyolo.label.assist import AssistEngine
    from libreyolo.label.dataset import DatasetSession

    ds = DatasetSession(str(_make_split_dataset(tmp_path)))
    eng = AssistEngine(enabled=True, default_model="definitely-not-a-real-model-zzz")
    with pytest.raises(RuntimeError):
        eng.autolabel_dataset(ds)


def test_sanitize_rejects_non_finite_before_clamp():
    # Codex round 7: inf must be rejected on the RAW value, not clamped to 1.0
    # (a finite edge) and then kept.
    import math

    from libreyolo.label.labelio import sanitize_annotations, sanitize_boxes

    assert sanitize_boxes([{"cls": 0, "cx": 0.5, "cy": 0.5, "w": math.inf, "h": 0.2}], nc=2) == []
    assert sanitize_annotations(
        [{"type": "box", "cls": 0, "cx": -math.inf, "cy": 0.5, "w": 0.2, "h": 0.2}], nc=2) == []
    poly = {"type": "poly", "cls": 0, "points": [0.1, 0.1, 0.4, math.inf, 0.4, 0.4, 0.1, 0.4]}
    assert sanitize_annotations([poly], nc=2) == []
    # a finite, valid box still round-trips
    assert len(sanitize_boxes([{"cls": 0, "cx": 0.5, "cy": 0.5, "w": 0.3, "h": 0.3}], nc=2)) == 1


def test_local_admin_gating_by_bind_and_client():
    # Codex round 8+9: gate host-admin on a *wildcard* (--share -> 0.0.0.0) bind by
    # requiring a loopback client; a loopback bind or an explicit NIC bind allow it
    # (the host has to reach an advertised NIC by its own address).
    from types import SimpleNamespace

    from libreyolo.label.server import _Handler

    assert _Handler._is_loopback("127.0.0.1") and _Handler._is_loopback("::1")
    assert _Handler._is_loopback("localhost") and not _Handler._is_loopback("192.168.1.5")

    h = _Handler.__new__(_Handler)   # bypass the socket-bound __init__

    # wildcard bind (--share): loopback client = host -> admin; LAN client -> denied
    h.state = SimpleNamespace(host="0.0.0.0")
    h.client_address = ("192.168.1.50", 5000)
    assert h._local_admin() is False
    h.client_address = ("127.0.0.1", 5000)
    assert h._local_admin() is True

    # loopback bind: only reachable locally -> always admin
    h.state = SimpleNamespace(host="127.0.0.1")
    h.client_address = ("192.168.1.50", 5000)
    assert h._local_admin() is True

    # explicit NIC bind: the host must connect via that address -> admin allowed
    h.state = SimpleNamespace(host="192.168.1.5")
    h.client_address = ("192.168.1.5", 5000)
    assert h._local_admin() is True


def test_out_of_range_class_is_read_only(tmp_path):
    # Codex round 10: a class id outside [0, nc) must keep the file read-only, or a
    # save would sanitize that annotation away.
    from libreyolo.label.dataset import DatasetSession
    from libreyolo.label.labelio import has_out_of_range_rows

    assert has_out_of_range_rows("5 0.5 0.5 0.2 0.2\n", nc=2) is True
    assert has_out_of_range_rows("1 0.5 0.5 0.2 0.2\n", nc=2) is False

    ds = DatasetSession(str(_make_split_dataset(tmp_path)))   # nc=2
    lp = tmp_path / "labels" / "train"
    lp.mkdir(parents=True, exist_ok=True)
    (lp / "a.txt").write_text("7 0.5 0.5 0.2 0.2\n")          # class 7, out of range
    _anns, editable = ds.read_label(0)
    assert editable is False
    with pytest.raises(RuntimeError):
        ds.write_label(0, [{"type": "box", "cls": 0, "cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2}])


def test_out_of_bounds_coords_read_only(tmp_path):
    # Codex round 11: coordinates outside [0,1] would be clamped & rewritten by a
    # save, so the file must stay read-only.
    from libreyolo.label.dataset import DatasetSession
    from libreyolo.label.labelio import has_out_of_bounds_coords

    assert has_out_of_bounds_coords("0 1.20 0.5 0.2 0.2\n") is True    # cx > 1
    assert has_out_of_bounds_coords("0 0.5 -0.1 0.2 0.2\n") is True    # cy < 0
    assert has_out_of_bounds_coords("0 0.5 0.5 0.2 0.2\n") is False

    ds = DatasetSession(str(_make_split_dataset(tmp_path)))
    lp = tmp_path / "labels" / "train"
    lp.mkdir(parents=True, exist_ok=True)
    (lp / "a.txt").write_text("0 1.20 0.50 0.20 0.20\n")
    _anns, editable = ds.read_label(0)
    assert editable is False
    with pytest.raises(RuntimeError):
        ds.write_label(0, [{"type": "box", "cls": 0, "cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2}])


def test_write_label_revision_conflict(tmp_path):
    # Codex round 11: an optimistic-concurrency guard so two LAN teammates editing
    # the same image don't silently clobber each other.
    from libreyolo.label.dataset import DatasetSession

    ds = DatasetSession(str(_make_split_dataset(tmp_path)))
    box = [{"type": "box", "cls": 0, "cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2}]
    ds.write_label(0, box)
    rev = ds.label_rev(0)
    assert rev > 0
    with pytest.raises(RuntimeError):
        ds.write_label(0, box, expected_rev=rev - 1)   # stale rev -> refused
    ds.write_label(0, box, expected_rev=rev)           # matching rev -> allowed


def test_degenerate_polygon_file_is_read_only(tmp_path):
    # Audit: a zero-area (collinear) polygon already on disk would be silently dropped
    # by a save's sanitizer, so the file must be read-only.
    from libreyolo.label.dataset import DatasetSession
    from libreyolo.label.labelio import has_degenerate_polygon

    assert has_degenerate_polygon("0 0.2 0.2 0.5 0.5 0.8 0.8\n") is True    # collinear 3-pt
    assert has_degenerate_polygon("0 0.1 0.1 0.4 0.1 0.4 0.4\n") is False   # real triangle

    ds = DatasetSession(str(_make_split_dataset(tmp_path)))
    lp = tmp_path / "labels" / "train"
    lp.mkdir(parents=True, exist_ok=True)
    (lp / "a.txt").write_text("0 0.2 0.2 0.5 0.5 0.8 0.8\n")
    _anns, editable = ds.read_label(0)
    assert editable is False


def test_obb_shaped_rows_read_only_unless_segment(tmp_path):
    # Codex round 12: a 4-point (8-coord) row is OBB-or-polygon ambiguous without an
    # explicit task: segment, so it stays read-only to avoid corrupting OBB labels.
    from libreyolo.label.dataset import DatasetSession
    from libreyolo.label.labelio import has_obb_shaped_rows

    quad = "0 0.1 0.1 0.9 0.1 0.9 0.9 0.1 0.9\n"
    assert has_obb_shaped_rows(quad) is True
    assert has_obb_shaped_rows("0 0.5 0.5 0.2 0.2\n") is False   # a box, not a quad

    taskless = DatasetSession(str(_make_split_dataset(tmp_path / "plain")))   # no task key
    lp = tmp_path / "plain" / "labels" / "train"
    lp.mkdir(parents=True, exist_ok=True)
    (lp / "a.txt").write_text(quad)
    assert taskless.read_label(0)[1] is False                   # ambiguous -> read-only

    seg = DatasetSession(str(_make_task_dataset(tmp_path / "seg", "segment")))
    slp = tmp_path / "seg" / "labels" / "train"
    slp.mkdir(parents=True, exist_ok=True)
    (slp / "a.txt").write_text(quad)
    assert seg.read_label(0)[1] is True                         # task: segment -> a real 4-pt polygon, editable


def test_unlabeled_image_inert_in_view_only_session(tmp_path):
    # Codex round 12: a read-only (dense-task) session must report editable=False even
    # for images with no label file, so prelabel/autolabel don't queue unsavable work.
    from libreyolo.label.dataset import DatasetSession

    ds = DatasetSession(str(_make_task_dataset(tmp_path / "pose", "pose")))   # view-only
    _anns, editable = ds.read_label(0)   # the image has no .txt
    assert editable is False


def test_zero_area_box_is_read_only(tmp_path):
    # Codex round 13: a zero-width/height box would be dropped by a save's sanitizer,
    # so the file stays read-only (same as degenerate polygons).
    from libreyolo.label.dataset import DatasetSession
    from libreyolo.label.labelio import has_zero_area_box

    assert has_zero_area_box("0 0.5 0.5 0.0 0.2\n") is True    # w == 0
    assert has_zero_area_box("0 0.5 0.5 0.2 0.0\n") is True    # h == 0
    assert has_zero_area_box("0 0.5 0.5 0.2 0.2\n") is False

    ds = DatasetSession(str(_make_split_dataset(tmp_path)))
    lp = tmp_path / "labels" / "train"
    lp.mkdir(parents=True, exist_ok=True)
    (lp / "a.txt").write_text("0 0.5 0.5 0.0 0.2\n")
    assert ds.read_label(0)[1] is False


def test_resolve_duplicates_refuses_recursive_split(tmp_path):
    # Codex round 13: a recursive split (train: .) would rglob-rediscover quarantined
    # files, so pruning it via quarantine is refused.
    from libreyolo.label.dataset import DatasetSession

    ds = DatasetSession(str(_make_split_dataset(tmp_path, leak=True)))
    ds._split_sources["val"] = [str(tmp_path)]   # val split = the whole dataset root
    with pytest.raises(RuntimeError):
        ds.resolve_duplicates([0, 1])
    assert ds.image_path(1).exists()             # nothing was moved


def test_boost_gather_skips_read_only_files(tmp_path):
    # Codex round 10: a partial-view (keypoint/OBB/out-of-range) file must not be
    # snapshotted/trained as if its unsupported rows didn't exist.
    from libreyolo.label.boost import BoostEngine
    from libreyolo.label.dataset import DatasetSession

    ds = DatasetSession(str(_make_split_dataset(tmp_path)))
    lp = tmp_path / "labels" / "train"
    lp.mkdir(parents=True, exist_ok=True)
    (lp / "a.txt").write_text("0 " + " ".join(["0.5"] * 55) + "\n")   # pose row -> read-only
    be = BoostEngine(ds, model_name="yolo9-t", enabled=False)
    images, _labels, _accepted = be._gather(ds)
    assert images == []


# --- "N: next flagged" walks the deck in worst-first order (#958) -----------
def _radar_next_ids(deck_ids, start, presses):
    """Run the web UI's radarNextId() under node, pressing N `presses` times."""
    import json
    import re
    import shutil
    import subprocess

    from libreyolo.label.web import js_power

    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    fn = re.search(r"^function radarNextId\(.*$", js_power.PART, re.M).group(0)
    script = (fn + f"\nconst deck={json.dumps([{'id': i} for i in deck_ids])}; let cur={start}; const seen=[];"
              f"\nfor(let k=0;k<{presses};k++){{ cur=radarNextId(deck, cur); seen.push(cur); }}"
              "\nconsole.log(JSON.stringify(seen));")
    out = subprocess.run([node, "-e", script], capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def test_next_flagged_visits_every_image_in_deck_order():
    # ids not ascending: the old "first id greater than current" lookup looped 7 -> 25 -> 30 -> 7
    deck = [7, 25, 3, 12, 30, 5]
    assert _radar_next_ids(deck, 7, 6) == [25, 3, 12, 30, 5, 7]


def test_next_flagged_from_unflagged_image_starts_at_top():
    assert _radar_next_ids([7, 25, 3], 99, 2) == [7, 25]
    assert _radar_next_ids([], 4, 1) == [None]
