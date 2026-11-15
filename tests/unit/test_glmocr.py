"""Offline GLM-OCR checks: layout post-processing, region preparation,
content formatting and merges, markdown, and the predict contract."""

import io
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image

from libreyolo import LibreGLMOCR, LibreVLM
from libreyolo.models.glmocr import formatter, layout, regions

pytestmark = pytest.mark.unit

LABEL = {name: i for i, name in enumerate(layout.LAYOUT_LABELS)}


def _row(label, score, box, order):
    return [LABEL[label], score, *box, order]


# =============================================================================
# Layout post-processing
# =============================================================================


def test_label_list_and_routing_match_sdk_config():
    assert len(layout.LAYOUT_LABELS) == 25
    assert layout.LAYOUT_LABELS[5] == "display_formula"
    assert layout.LAYOUT_LABELS[22] == "text"
    assert layout.LAYOUT_THRESHOLD == 0.3
    assert layout.LAYOUT_MERGE_MODES[LABEL["reference"]] == "small"
    assert layout.LAYOUT_MERGE_MODES[LABEL["text"]] == "large"
    assert layout.task_for_label("seal") == "text"
    assert layout.task_for_label("inline_formula") == "formula"
    assert layout.task_for_label("chart") == "skip"
    assert layout.task_for_label("header") == "abandon"
    assert layout.task_for_label("nope") is None


def test_postprocess_nms_is_class_aware_and_sorts_by_order():
    rows = np.array(
        [
            _row("text", 0.9, (10, 10, 100, 50), 2),
            _row("text", 0.8, (12, 10, 100, 50), 3),  # same class, IoU > 0.6
            _row("table", 0.85, (12, 10, 100, 50), 1),  # other class, IoU < 0.98
            _row("doc_title", 0.9, (10, 150, 100, 180), 0),
        ]
    )
    boxes = layout.postprocess_layout(rows, (200, 200), merge_modes={})
    assert [b["label"] for b in boxes] == ["doc_title", "table", "text"]
    assert boxes[0]["coordinate"] == [10, 150, 100, 180]
    assert (
        len(
            layout.postprocess_layout(
                rows, (200, 200), nms_enabled=False, merge_modes={}
            )
        )
        == 4
    )


def test_postprocess_large_merge_and_preserved_pictures():
    rows = np.array(
        [
            _row("paragraph_title", 0.9, (0, 0, 100, 40), 0),
            _row("text", 0.9, (5, 5, 95, 35), 1),  # inside the title box
            _row("image", 0.9, (0, 60, 100, 100), 2),
            _row("seal", 0.9, (10, 65, 90, 95), 3),  # seal is preserved
        ]
    )
    boxes = layout.postprocess_layout(rows, (200, 200), nms_enabled=False)
    assert [b["label"] for b in boxes] == ["paragraph_title", "image", "seal"]


def test_postprocess_small_merge_for_reference():
    rows = np.array(
        [
            _row("text", 0.9, (0, 0, 100, 100), 0),
            _row("reference", 0.9, (5, 5, 95, 50), 1),
        ]
    )
    small_only = {LABEL["reference"]: "small"}
    boxes = layout.postprocess_layout(
        rows, (200, 200), nms_enabled=False, merge_modes=small_only
    )
    # "small" keeps the contained reference and drops the box holding it.
    assert [b["label"] for b in boxes] == ["reference"]
    # With the SDK defaults "large" on text also drops the contained box.
    assert layout.postprocess_layout(rows, (200, 200), nms_enabled=False) == []


def test_postprocess_drops_page_sized_image_and_clamps():
    rows = np.array(
        [
            _row("image", 0.9, (0, 0, 200, 200), 0),
            _row("text", 0.9, (-5, 10, 250, 50), 1),
        ]
    )
    boxes = layout.postprocess_layout(rows, (200, 200), nms_enabled=False)
    assert [b["label"] for b in boxes] == ["text"]
    assert boxes[0]["coordinate"] == [0, 10, 200, 50]
    assert layout.postprocess_layout(np.zeros((0, 7)), (10, 10)) == []


def test_to_regions_drops_abandoned_and_normalizes():
    boxes = [
        {"cls_id": 12, "label": "header", "score": 0.9, "coordinate": [0, 0, 10, 10]},
        {
            "cls_id": 22,
            "label": "text",
            "score": 0.8,
            "coordinate": [50, 100, 150, 300],
        },
        {"cls_id": 14, "label": "image", "score": 0.7, "coordinate": [0, 0, 400, 100]},
    ]
    out = layout.to_regions(boxes, (400, 600))
    assert [r["label"] for r in out] == ["text", "image"]
    assert out[0]["bbox_2d"] == [125, 166, 375, 500]
    assert out[0]["task_type"] == "text" and out[1]["task_type"] == "skip"
    assert [r["index"] for r in out] == [0, 1]
    assert layout.bbox_to_pixels(out[0]["bbox_2d"], (400, 600)) == [50, 99, 150, 300]


def test_raw_rows_from_transformers_result():
    result = {
        "scores": torch.tensor([0.9, 0.5]),
        "labels": torch.tensor([22, 21]),
        "boxes": torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]),
        "order_seq": torch.tensor([1, 2]),
    }
    rows = layout.raw_rows(result)
    assert rows.shape == (2, 7)
    np.testing.assert_allclose(rows[1], [21, 0.5, 5, 6, 7, 8, 2])
    empty = {k: v[:0] for k, v in result.items()}
    assert layout.raw_rows(empty).shape == (0, 7)


# =============================================================================
# Region preparation
# =============================================================================


def test_smart_resize_rounds_to_28_and_respects_budgets():
    assert regions.smart_resize(2, 100, 300, t_factor=2) == (112, 308)
    h, w = regions.smart_resize(2, 10, 20, t_factor=2, min_pixels=112 * 112)
    assert h % 28 == 0 and w % 28 == 0 and 2 * h * w >= 112 * 112
    h, w = regions.smart_resize(2, 4000, 4000, t_factor=2, max_pixels=2 * 1000 * 1000)
    assert 2 * h * w <= 2 * 1000 * 1000
    with pytest.raises(ValueError):
        regions.smart_resize(1, 10, 10, t_factor=2)


def test_prepare_image_matches_the_sdk_jpeg_round_trip():
    rng = np.random.default_rng(0)
    crop = Image.fromarray(rng.integers(0, 255, (70, 130, 3), dtype=np.uint8))
    out = regions.prepare_image(crop)
    assert out.size == (140, 56) and out.mode == "RGB"
    expected = crop.resize((140, 56), Image.Resampling.BICUBIC)
    buffer = io.BytesIO()
    expected.save(buffer, format="JPEG")
    expected = Image.open(io.BytesIO(buffer.getvalue())).convert("RGB")
    assert np.array_equal(np.asarray(out), np.asarray(expected))
    lossless = regions.prepare_image(crop, image_format=None)
    assert np.array_equal(
        np.asarray(lossless),
        np.asarray(crop.resize((140, 56), Image.Resampling.BICUBIC)),
    )


def test_crop_region_uses_the_0_1000_grid():
    img = Image.new("RGB", (333, 777))
    assert regions.crop_region(img, [100, 200, 500, 900]).size == (166 - 33, 699 - 155)


# =============================================================================
# Formatting and merges
# =============================================================================


def test_clean_content_and_inline_formula():
    assert formatter.clean_content("\\t\\tHello.......") == "Hello..."
    assert formatter.clean_content("a ____ b") == "a ___ b"
    assert formatter.normalize_inline_formula("value$ x+1 $is") == "value $x+1$ is"
    assert formatter.clean_content(None) == ""


def test_clean_repeated_content():
    looped = "intro " + "the same words " * 30
    assert formatter.clean_repeated_content(looped) == "intro the same words"
    lines = "\n".join(["header"] + ["dup"] * 12)
    assert formatter.clean_repeated_content(lines) == "header\ndup"


def test_format_content_titles_formulas_and_lists():
    assert formatter.format_content("## Intro", "text", "doc_title") == "# Intro"
    assert (
        formatter.format_content("- Methods", "text", "paragraph_title") == "## Methods"
    )
    assert (
        formatter.format_content("$$x=1$$", "formula", "display_formula")
        == "$$\nx=1\n$$"
    )
    assert (
        formatter.format_content("\\[y\\]", "formula", "display_formula") == "$$\ny\n$$"
    )
    assert formatter.format_content("• item", "text", "text") == "- item"
    assert formatter.format_content("（3）third", "text", "text") == "(3) third"
    assert formatter.format_content("2.next", "text", "text") == "2. next"
    assert formatter.format_content("a\nb", "text", "text") == "a\n\nb"
    # The closing fence is added before newlines are doubled, as upstream.
    assert formatter.format_content("```py\nx", "text", "text") == "```py\n\nx\n\n```"
    table = "<table><tr><td>1</td></tr></table>"
    assert formatter.format_content(table, "table", "table") == table
    assert formatter.format_content(None, "text", "text") is None


def _region(index, label, content, bbox=(0, 0, 10, 10), task="text"):
    return {
        "index": index,
        "label": label,
        "content": content,
        "bbox_2d": list(bbox),
        "task_type": task,
    }


def test_format_page_merges_formula_numbers_and_hyphenation():
    page = formatter.format_page(
        [
            _region(0, "display_formula", "E=mc^2", task="formula"),
            _region(1, "formula_number", "(1)"),
            _region(2, "text", "an exam-"),
            _region(3, "text", "ple of text"),
            _region(4, "image", None, task="skip"),
            _region(5, "text", "   "),  # empty after cleaning: dropped
        ]
    )
    assert [b["native_label"] for b in page] == ["display_formula", "text", "image"]
    assert page[0]["content"] == "$$\nE=mc^2 \\tag{1}\n$$"
    assert page[1]["content"] == "an example of text"
    assert page[2]["is_image"] and page[2]["label"] == "image"
    assert [b["index"] for b in page] == [0, 1, 2]
    assert (
        formatter.page_markdown(page) == "$$\nE=mc^2 \\tag{1}\n$$\n\nan example of text"
    )


def test_hyphen_merge_skips_blocks_that_do_not_start_lowercase():
    page = formatter.format_page(
        [
            _region(0, "text", "an exam-"),
            _region(1, "paragraph_title", "Next"),
            _region(2, "text", "ple here"),
        ]
    )
    assert [b["content"] for b in page] == ["an example here", "## Next"]


def test_format_page_number_before_formula_and_orphan_number():
    page = formatter.format_page(
        [
            _region(0, "formula_number", "（2）"),
            _region(1, "display_formula", "a+b", task="formula"),
            _region(2, "formula_number", "(3)"),
        ]
    )
    assert [b["content"] for b in page] == ["$$\na+b \\tag{2}\n$$"]


def test_format_page_adds_missing_bullets():
    page = formatter.format_page(
        [
            _region(0, "text", "- one", (100, 0, 500, 10)),
            _region(1, "text", "two", (105, 10, 500, 20)),
            _region(2, "text", "- three", (98, 20, 500, 30)),
            _region(3, "text", "far", (300, 30, 500, 40)),
        ]
    )
    assert [b["content"] for b in page] == ["- one", "- two", "- three", "far"]


# =============================================================================
# Predict contract (fake networks)
# =============================================================================


class _FakeVL(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))


LAYOUT = [
    {"cls_id": 6, "label": "doc_title", "score": 0.95, "coordinate": [40, 10, 360, 40]},
    {"cls_id": 12, "label": "header", "score": 0.9, "coordinate": [0, 0, 400, 8]},
    {"cls_id": 14, "label": "image", "score": 0.9, "coordinate": [40, 60, 360, 140]},
    {"cls_id": 22, "label": "text", "score": 0.8, "coordinate": [40, 160, 360, 240]},
    {"cls_id": 21, "label": "table", "score": 0.7, "coordinate": [40, 260, 360, 380]},
]
RESPONSES = {
    "Text Recognition:": "Hello\nworld",
    "Table Recognition:": "<table><tr><td>1</td></tr></table>",
    "Formula Recognition:": "x^2",
}


@pytest.fixture
def harness(monkeypatch, tmp_path):
    state = SimpleNamespace(prompts=[], sizes=[])

    def load_pretrained(self, snapshot_dir):
        return _FakeVL(), SimpleNamespace()

    def recognize(self, image, prompt, **_):
        state.prompts.append(prompt)
        state.sizes.append(image.size)
        text = RESPONSES.get(prompt, "Custom")
        if prompt == "Text Recognition:" and image.size[1] < 60:
            text = "Title"
        return {"text": text, "confidence": 0.75, "input_ids": [], "token_ids": []}

    monkeypatch.setattr(LibreGLMOCR, "_ensure_weights", lambda self: str(tmp_path))
    monkeypatch.setattr(LibreGLMOCR, "_load_pretrained", load_pretrained)
    monkeypatch.setattr(
        LibreGLMOCR, "_detect_layout", lambda self, image: [dict(r) for r in LAYOUT]
    )
    monkeypatch.setattr(LibreGLMOCR, "recognize", recognize)
    return state


@pytest.mark.parametrize("alias", ["glm-ocr", "glmocr", "glm-ocr-0.9b"])
def test_factory(alias, harness):
    model = LibreVLM(alias, device="cpu")
    assert isinstance(model, LibreGLMOCR)
    assert model.size == "0.9b" and model.task == "ocr"
    assert model.names[LABEL["table"]] == "table"


def test_predict_returns_blocks_in_reading_order(harness):
    model = LibreVLM("glm-ocr", device="cpu")
    result = model.predict(Image.new("RGB", (400, 400), "white"))
    ocr = result.ocr
    # The header is abandoned and the picture is not read.
    assert ocr.labels == ["doc_title", "text", "table"]
    assert ocr.texts == [
        "# Title",
        "Hello\n\nworld",
        "<table><tr><td>1</td></tr></table>",
    ]
    np.testing.assert_allclose(np.asarray(ocr.conf), [0.75] * 3)
    np.testing.assert_allclose(np.asarray(ocr.det_conf), [0.95, 0.8, 0.7], atol=1e-6)
    np.testing.assert_allclose(np.asarray(ocr.xyxy)[1], [40, 160, 360, 240], atol=1e-3)
    assert (
        ocr.markdown
        == "# Title\n\nHello\n\nworld\n\n<table><tr><td>1</td></tr></table>"
    )
    assert harness.prompts == [
        "Text Recognition:",
        "Text Recognition:",
        "Table Recognition:",
    ]
    # Every crop reaches the recognizer resized to multiples of 28.
    assert all(w % 28 == 0 and h % 28 == 0 for w, h in harness.sizes)
    assert result.boxes is None


def test_custom_prompt_reaches_the_recognizer(harness):
    model = LibreVLM("glm-ocr", device="cpu", prompt="Read it:")
    result = model.predict(Image.new("RGB", (400, 400), "white"))
    assert set(harness.prompts) == {"Read it:"}
    assert result.ocr.texts[1] == "Custom"


def test_predict_without_layout_reads_the_whole_image(harness):
    model = LibreVLM(
        "glm-ocr", device="cpu", use_layout_detection=False, prompt_label="formula"
    )
    result = model.predict(Image.new("RGB", (120, 80), "white"))
    assert result.ocr.labels == ["display_formula"]
    assert result.ocr.texts == ["$$\nx^2\n$$"]
    assert harness.prompts == ["Formula Recognition:"]
    assert harness.sizes == [(112, 84)]
    np.testing.assert_allclose(np.asarray(result.ocr.xyxy)[0], [0, 0, 120, 80])


def test_predict_saves_annotated_image(harness, tmp_path):
    model = LibreVLM("glm-ocr", device="cpu")
    model.predict(
        Image.new("RGB", (400, 400), "white"),
        save=True,
        output_path=str(tmp_path / "out"),
    )
    assert any((tmp_path / "out").rglob("*.jpg"))


def test_layout_threshold_arguments(harness):
    model = LibreVLM("glm-ocr", device="cpu")
    assert model._threshold_args() == (0.3, None)
    model.layout_threshold = 0.5
    assert model._threshold_args() == (0.5, None)
    model.layout_threshold = {"22": 0.2}
    assert model._threshold_args() == (0.2, {22: 0.2})


def test_contract_errors(harness):
    with pytest.raises(ValueError, match="prompt_label"):
        LibreVLM("glm-ocr", device="cpu", prompt_label="seal")
    model = LibreVLM("glm-ocr", device="cpu")
    with pytest.raises(ValueError, match="no vocabulary"):
        model.set_classes(["cat"])
    with pytest.raises(NotImplementedError, match="inference-only"):
        model.train(data="x.yaml")
    with pytest.raises(NotImplementedError):
        model.export()
    with pytest.raises(ValueError):
        model.set_task("detect")


def test_val_runs_the_ocr_validator(harness, tmp_path):
    root = tmp_path / "ds"
    (root / "images" / "val").mkdir(parents=True)
    (root / "labels").mkdir()
    Image.new("RGB", (400, 400), "white").save(root / "images" / "val" / "page.png")
    labels = [
        {
            "polygon": [[40, 260], [360, 260], [360, 380], [40, 380]],
            "text": "<table><tr><td>1</td></tr></table>",
        },
        {"polygon": [[40, 390], [360, 390], [360, 399], [40, 399]], "text": "Missing"},
    ]
    (root / "labels" / "val.jsonl").write_text(
        json.dumps({"image": "page.png", "regions": labels}) + "\n"
    )
    model = LibreVLM("glm-ocr", device="cpu")
    metrics = model.val(data=str(root), verbose=False)
    assert metrics["metrics/det_recall"] == pytest.approx(0.5)
    assert metrics["metrics/e2e_recall"] == pytest.approx(0.5)
