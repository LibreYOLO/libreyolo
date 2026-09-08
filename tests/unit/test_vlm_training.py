"""Offline unit tests for VLM detection fine-tuning.

Everything here runs without network, GPU, or model weights: target
serialization (including a full round-trip through the inference parser),
dataset reading, collator label masking with a stub processor, the checkpoint
contract, and the train() gating surface.
"""

import json
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.vlm]

torch = pytest.importorskip("torch")
from PIL import Image  # noqa: E402

from libreyolo.models.vlm.parsing import build_detection_dict, extract_detections  # noqa: E402
from libreyolo.models.vlm.training.checkpoint import (  # noqa: E402
    CONTRACT_FILENAME,
    is_vlm_checkpoint,
    read_contract,
    save_vlm_checkpoint,
)
from libreyolo.models.vlm.training.collate import VLMChatCollator  # noqa: E402
from libreyolo.models.vlm.training.data import (  # noqa: E402
    VLMDetectDataset,
    resolve_split_source,
)
from libreyolo.models.vlm.training.recipes import get_recipe  # noqa: E402
from libreyolo.models.vlm.training.targets import (  # noqa: E402
    FamilyFormat,
    serialize_detections,
)

QWEN_FMT = FamilyFormat(
    family="qwen3vl",
    bbox_key="bbox_2d",
    coord_divisor=1000.0,
    box_format="xyxy",
    detection_prompt="Detect all instances of: cat, dog.",
)
UNIT_FMT = FamilyFormat(
    family="unit",
    bbox_key="bbox",
    coord_divisor=1.0,
    box_format="xyxy",
    detection_prompt="Detect.",
)


# ---------------------------------------------------------------------------
# Target serialization
# ---------------------------------------------------------------------------


class TestSerializeDetections:
    def test_qwen_format_scales_to_thousand_ints(self):
        text = serialize_detections([[0.1, 0.2, 0.5, 0.9]], ["cat"], QWEN_FMT)
        assert json.loads(text) == [{"bbox_2d": [100, 200, 500, 900], "label": "cat"}]

    def test_unit_scale_keeps_three_decimal_floats(self):
        text = serialize_detections([[0.1234, 0.2, 0.5, 0.8768]], ["cat"], UNIT_FMT)
        assert json.loads(text) == [{"label": "cat", "bbox": [0.123, 0.2, 0.5, 0.877]}]

    def test_empty_is_empty_array(self):
        assert serialize_detections([], [], QWEN_FMT) == "[]"

    def test_reading_order_is_deterministic(self):
        boxes = [[0.5, 0.8, 0.6, 0.9], [0.1, 0.1, 0.2, 0.2], [0.7, 0.1, 0.8, 0.2]]
        text = serialize_detections(boxes, ["low", "topleft", "topright"], QWEN_FMT)
        labels = [item["label"] for item in json.loads(text)]
        assert labels == ["topleft", "topright", "low"]

    def test_clamps_and_reorders_corners(self):
        text = serialize_detections([[0.9, 1.4, 0.1, -0.2]], ["cat"], QWEN_FMT)
        assert json.loads(text) == [{"bbox_2d": [100, 0, 900, 1000], "label": "cat"}]

    def test_layout_conversion_xywh_and_cxcywh(self):
        xywh = FamilyFormat("f", "bbox", 1.0, "xywh", "p")
        cxcywh = FamilyFormat("f", "bbox", 1.0, "cxcywh", "p")
        box = [[0.2, 0.2, 0.6, 0.8]]
        assert json.loads(serialize_detections(box, ["x"], xywh))[0]["bbox"] == [
            0.2, 0.2, 0.4, 0.6,
        ]
        assert json.loads(serialize_detections(box, ["x"], cxcywh))[0]["bbox"] == [
            0.4, 0.5, 0.4, 0.6,
        ]

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="mismatch"):
            serialize_detections([[0, 0, 1, 1]], [], QWEN_FMT)

    def test_round_trip_through_inference_parser(self):
        """The serializer must be the exact inverse of the predict-time parser."""
        boxes = [[0.10, 0.20, 0.50, 0.90], [0.25, 0.05, 0.75, 0.55]]
        labels = ["cat", "dog"]
        text = serialize_detections(boxes, labels, QWEN_FMT)
        items = extract_detections(text)
        result = build_detection_dict(
            items,
            {"cat": 0, "dog": 1},
            (1000, 1000),  # W, H chosen so pixel coords equal 1000*normalized
            bbox_key=QWEN_FMT.bbox_key,
            coord_divisor=QWEN_FMT.coord_divisor,
            box_format=QWEN_FMT.box_format,
        )
        assert result["num_detections"] == 2
        recovered = sorted(
            (int(c), [round(v) for v in b])
            for c, b in zip(result["classes"], result["boxes"])
        )
        expected = sorted(
            (i, [round(v * 1000) for v in box])
            for i, box in zip([0, 1], boxes)
        )
        assert recovered == expected


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


def _write_dataset(root: Path, rows_by_image: dict) -> Path:
    images = root / "images" / "train"
    labels = root / "labels" / "train"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    for stem, rows in rows_by_image.items():
        Image.new("RGB", (64, 48), (120, 130, 140)).save(images / f"{stem}.png")
        if rows is not None:
            (labels / f"{stem}.txt").write_text(rows, encoding="utf-8")
    return images


class TestVLMDetectDataset:
    def test_renders_prompt_and_target(self, tmp_path):
        images = _write_dataset(tmp_path, {"a": "0 0.5 0.5 0.2 0.4\n"})
        ds = VLMDetectDataset(images, {0: "cat", 1: "dog"}, QWEN_FMT)
        sample = ds[0]
        assert sample["prompt"] == QWEN_FMT.detection_prompt
        parsed = json.loads(sample["target"])
        assert parsed == [{"bbox_2d": [400, 300, 600, 700], "label": "cat"}]
        assert sample["image"].size == (64, 48)

    def test_missing_label_file_teaches_empty_answer(self, tmp_path):
        images = _write_dataset(tmp_path, {"a": None})
        ds = VLMDetectDataset(images, {0: "cat"}, QWEN_FMT)
        assert ds[0]["target"] == "[]"

    def test_malformed_and_unknown_rows_are_skipped(self, tmp_path):
        images = _write_dataset(
            tmp_path, {"a": "not a row\n7 0.5 0.5 0.2 0.2\n0 0.5 0.5 0.2 0.2\n"}
        )
        ds = VLMDetectDataset(images, {0: "cat"}, QWEN_FMT)
        parsed = json.loads(ds[0]["target"])
        assert len(parsed) == 1 and parsed[0]["label"] == "cat"

    def test_hflip_mirrors_boxes(self, tmp_path):
        images = _write_dataset(tmp_path, {"a": "0 0.25 0.5 0.1 0.2\n"})
        ds = VLMDetectDataset(images, {0: "cat"}, QWEN_FMT, augment=True, hflip_p=1.0)
        parsed = json.loads(ds[0]["target"])
        # cx 0.25 -> mirrored cx 0.75; y untouched.
        assert parsed[0]["bbox_2d"] == [700, 400, 800, 600]

    def test_empty_image_dir_raises(self, tmp_path):
        empty = tmp_path / "none"
        empty.mkdir()
        with pytest.raises(FileNotFoundError):
            VLMDetectDataset(empty, {0: "cat"}, QWEN_FMT)

    def test_coco_json_dataset_rejected_clearly(self):
        cfg = {"train": "imgs", "annotations": {"train": "x.json"}}
        with pytest.raises(NotImplementedError, match="COCO JSON"):
            resolve_split_source(cfg, "train")
        cfg2 = {"train": "imgs", "train_annotation_file": "x.json"}
        with pytest.raises(NotImplementedError, match="COCO JSON"):
            resolve_split_source(cfg2, "train")


# ---------------------------------------------------------------------------
# Collator (stub processor: ids are deterministic functions of the text)
# ---------------------------------------------------------------------------


class _StubTokenizer:
    padding_side = "left"  # collator must flip to right and restore


class _StubProcessor:
    """Tokenizes text as its character codes; images add three fixed tokens."""

    PAD = 0
    IMAGE_TOKENS = [900, 901, 902]

    def __init__(self, break_prefix=False):
        self.tokenizer = _StubTokenizer()
        self.break_prefix = break_prefix

    def _encode_conv(self, conversation, add_generation_prompt):
        ids = []
        for turn in conversation:
            for part in turn["content"]:
                if part["type"] == "image":
                    ids.extend(self.IMAGE_TOKENS)
                else:
                    ids.extend((ord(c) % 500) + 1000 for c in part["text"])
        if add_generation_prompt:
            ids.append(3)  # assistant header stub
        elif len(conversation) > 1:
            # Full conversation: header sits between prompt and answer.
            answer = ids[-sum(len(p["text"]) for p in conversation[-1]["content"]):]
            prompt = ids[: len(ids) - len(answer)]
            first = [7] if self.break_prefix else []
            ids = first + prompt + [3] + answer + [4]  # 4 = eos stub
        return ids

    def apply_chat_template(
        self,
        conversations,
        tokenize,
        return_dict,
        return_tensors,
        padding,
        add_generation_prompt=False,
    ):
        assert tokenize and return_dict and padding
        assert self.tokenizer.padding_side == "right"
        encoded = [
            self._encode_conv(conv, add_generation_prompt) for conv in conversations
        ]
        width = max(len(e) for e in encoded)
        input_ids = torch.full((len(encoded), width), self.PAD, dtype=torch.long)
        attention = torch.zeros((len(encoded), width), dtype=torch.long)
        for i, ids in enumerate(encoded):
            input_ids[i, : len(ids)] = torch.tensor(ids)
            attention[i, : len(ids)] = 1
        return {
            "input_ids": input_ids,
            "attention_mask": attention,
            "token_type_ids": torch.zeros_like(input_ids),
        }


def _samples():
    image = Image.new("RGB", (8, 8))
    return [
        {"image": image, "prompt": "find cats", "target": "[]"},
        {"image": image, "prompt": "find cats", "target": '[{"bbox_2d": [1, 2, 3, 4]}]'},
    ]


class TestVLMChatCollator:
    def test_masks_prompt_and_padding_supervises_answer(self):
        collator = VLMChatCollator(_StubProcessor())
        batch = collator(_samples())
        labels, ids = batch["labels"], batch["input_ids"]
        for i, sample in enumerate(_samples()):
            prompt_len = 3 + len(sample["prompt"]) + 1  # image + text + header
            assert (labels[i, :prompt_len] == -100).all()
            answer_len = len(sample["target"]) + 1  # + eos stub
            answer = labels[i, prompt_len : prompt_len + answer_len]
            assert (answer != -100).all()
            assert torch.equal(answer, ids[i, prompt_len : prompt_len + answer_len])
            assert (labels[i, prompt_len + answer_len :] == -100).all()

    def test_supervised_fraction_is_answer_only(self):
        batch = VLMChatCollator(_StubProcessor())(_samples())
        supervised = int((batch["labels"] != -100).sum())
        expected = sum(len(s["target"]) + 1 for s in _samples())
        assert supervised == expected

    def test_prefix_violation_raises(self):
        collator = VLMChatCollator(_StubProcessor(break_prefix=True))
        with pytest.raises(RuntimeError, match="prompt-prefix"):
            collator(_samples())

    def test_padding_side_restored_and_drop_keys(self):
        processor = _StubProcessor()
        batch = VLMChatCollator(processor)(_samples())
        assert processor.tokenizer.padding_side == "left"
        assert "token_type_ids" not in batch


# ---------------------------------------------------------------------------
# Checkpoint contract
# ---------------------------------------------------------------------------


class _StubSaveable:
    def save_pretrained(self, directory):
        Path(directory).mkdir(parents=True, exist_ok=True)
        (Path(directory) / "adapter_config.json").write_text("{}", encoding="utf-8")


class _StubWrapper:
    FAMILY = "qwen3vl"
    size = "2b"
    HF_REPOS = {"2b": "org/base-repo"}
    HF_REVISIONS = {}
    BBOX_KEY = "bbox_2d"
    COORD_DIVISOR = 1000.0
    BOX_FORMAT = "xyxy"
    names = {0: "cat", 1: "dog"}
    processor = _StubSaveable()

    def _detection_prompt(self):
        return "Detect all instances of: cat, dog."


class TestCheckpointContract:
    def test_save_then_read_round_trips(self, tmp_path):
        target = tmp_path / "best"
        save_vlm_checkpoint(
            target,
            peft_model=_StubSaveable(),
            processor=_StubSaveable(),
            wrapper=_StubWrapper(),
            metrics={"train/loss": 0.5},
        )
        assert is_vlm_checkpoint(target)
        contract = read_contract(target)
        assert contract["family"] == "qwen3vl"
        assert contract["size"] == "2b"
        assert contract["base_repo"] == "org/base-repo"
        assert contract["names"] == ["cat", "dog"]
        assert contract["coord_divisor"] == 1000.0
        assert contract["metrics"]["train/loss"] == 0.5

    def test_is_vlm_checkpoint_negative_cases(self, tmp_path):
        assert not is_vlm_checkpoint(tmp_path)  # dir without contract
        assert not is_vlm_checkpoint(tmp_path / "missing")
        assert not is_vlm_checkpoint("qwen3-vl-4b")  # alias string

    def test_wrong_schema_rejected(self, tmp_path):
        (tmp_path / CONTRACT_FILENAME).write_text(
            json.dumps({"schema": 99, "family": "x", "size": "s", "names": []}),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="schema"):
            read_contract(tmp_path)

    def test_missing_field_rejected(self, tmp_path):
        (tmp_path / CONTRACT_FILENAME).write_text(
            json.dumps({"schema": 1, "family": "x"}), encoding="utf-8"
        )
        with pytest.raises(ValueError, match="size"):
            read_contract(tmp_path)


# ---------------------------------------------------------------------------
# Gating and recipes
# ---------------------------------------------------------------------------


class TestTrainGating:
    def _bare(self, cls):
        return object.__new__(cls)

    def test_qwen3vl_is_trainable_and_requires_data(self):
        from libreyolo.models.vlm import LibreQwen3VL

        assert LibreQwen3VL.TRAINABLE is True
        with pytest.raises(ValueError, match="data="):
            LibreQwen3VL.train(self._bare(LibreQwen3VL))

    @pytest.mark.parametrize(
        "module_name, class_name, match",
        [
            ("smolvlm", "LibreSmolVLM2", "grounding"),
            ("kosmos2", "LibreKosmos2", "recipe"),
            ("locateanything", "LibreLocateAnything", "non-commercial"),
        ],
    )
    def test_untrainable_families_explain_why(self, module_name, class_name, match):
        import importlib

        module = importlib.import_module(f"libreyolo.models.vlm.{module_name}")
        cls = getattr(module, class_name)
        assert cls.TRAINABLE is False
        with pytest.raises(NotImplementedError, match=match):
            cls.train(self._bare(cls), data="x.yaml")

    def test_recipe_exists_for_every_trainable_family(self):
        from libreyolo.models.vlm import _ALIASES

        for cls, _size in set(_ALIASES.values()):
            if cls.TRAINABLE:
                recipe = get_recipe(cls.FAMILY)
                assert recipe.target_modules

    def test_unknown_recipe_raises(self):
        with pytest.raises(NotImplementedError):
            get_recipe("not-a-family")


class TestResumePreservesRunDir:
    """resume=True must read weights/last from the exact directory it names,
    not one _resolve_save_dir() incremented away from underneath it."""

    @staticmethod
    def _fake_wrapper():
        return type("FakeWrapper", (), {"FAMILY": "qwen3vl"})()

    def test_resume_true_forces_exist_ok_and_keeps_dir(self, tmp_path):
        from libreyolo.models.vlm.training.trainer import VLMDetectionTrainer

        run_dir = tmp_path / "runs" / "vlm_exp"
        weights_dir = run_dir / "weights" / "last"
        weights_dir.mkdir(parents=True)
        (weights_dir / CONTRACT_FILENAME).write_text("{}")

        trainer = VLMDetectionTrainer(
            self._fake_wrapper(),
            data="data.yaml",
            project=str(tmp_path / "runs"),
            name="vlm_exp",
            resume=True,
        )

        assert trainer.save_dir == run_dir
        assert trainer.config.exist_ok is True
        assert trainer._resolve_resume_dir() == weights_dir

    def test_non_resumed_run_still_increments_by_default(self, tmp_path):
        from libreyolo.models.vlm.training.trainer import VLMDetectionTrainer

        (tmp_path / "runs" / "vlm_exp").mkdir(parents=True)

        trainer = VLMDetectionTrainer(
            self._fake_wrapper(),
            data="data.yaml",
            project=str(tmp_path / "runs"),
            name="vlm_exp",
        )

        assert trainer.save_dir == tmp_path / "runs" / "vlm_exp2"
        assert trainer.config.exist_ok is False
