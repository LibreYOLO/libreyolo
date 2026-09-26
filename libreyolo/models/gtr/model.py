"""GTR wrapper (detect, segment, pose, obb, depth, semantic), strict loading."""

from pathlib import Path
from typing import ClassVar

from libreyolo.training.ddp_spawn import ddp_aware

from ...tasks import normalize_task
from ...training.callbacks import TrainCallbacks
from ...validation.preprocessors import DEIMv2DINOValPreprocessor
from ..dfine.model import LibreDFINE
from ..ec.postprocess import preprocess_image
from . import depth as gtr_depth
from . import sem
from .config import GTRConfig
from .nn import LibreGTRModel
from .obb import OBB_INPUT_SIZES, is_gtr_obb_state_dict
from .pose import LibreGTRPoseModel
from .seg import SEG_MASK_DOWNSAMPLE_RATIO, is_seg_state_dict

_POSE_HEAD_KEY = "decoder.keypoint_embedding.weight"


class LibreGTR(LibreDFINE):
    """GTR detection, sizes S/M/L/X. GPU and convergence evidence is pending."""

    FAMILY = "gtr"
    FILENAME_PREFIX = "LibreGTR"
    INPUT_SIZES: ClassVar[dict[str, int]] = {s: 640 for s in ("s", "m", "l", "x")}
    SUPPORTED_TASKS = ("detect", "segment", "pose", "obb", "depth", "semantic")
    # Semantic runs on the native Cityscapes canvas; the network itself slides
    # 1024px windows over it (sem.LibreGTRSemModel).
    TASK_INPUT_SIZES: ClassVar[dict] = {
        "detect": INPUT_SIZES,
        "segment": INPUT_SIZES,
        "pose": INPUT_SIZES,
        "obb": OBB_INPUT_SIZES,
        "depth": INPUT_SIZES,
        "semantic": {s: (sem.SEM_WINDOW, 2 * sem.SEM_WINDOW) for s in "smlx"},
    }
    DEFAULT_TASK = "detect"
    TRAIN_CONFIG = GTRConfig
    SUPPORTS_CUDA_GRAPH = False
    val_preprocessor_class = DEIMv2DINOValPreprocessor
    POSE_NUM_KEYPOINTS = 17
    KEYPOINT_DIM = 3
    # Depth task (ADR 0006): square stretch resize, as in upstream validation.
    depth_imgsz_divisor = 32
    depth_resize_mode = "stretch"
    # Read by the shared semantic dataset, validator and trainer.
    semantic_resize_mode: ClassVar[str] = "letterbox"
    semantic_imgsz_divisor: ClassVar[int] = 32
    # Upstream large-scale jitter: long side fit to 1024 * [1, 4], then a crop.
    semantic_scale_jitter: ClassVar[tuple[float, float]] = (1.0, 4.0)

    @property
    def semantic_photometric(self):
        from .sem_trainer import _PhotometricDistort

        return _PhotometricDistort(p=0.5)

    @property
    def semantic_val_imgsz(self):
        """Semantic training validates on the canvas, not the square crop."""
        return self.input_size if self.task == "semantic" else None

    @classmethod
    def can_load(cls, sd):
        if "backbone.backbone._model.blocks.0.attn.gk_proj.0.weight" not in sd:
            return False
        if sem.is_semantic_state_dict(sd):
            return "encoder.stages.0.1.weight" in sd
        if _POSE_HEAD_KEY in sd:
            return "encoder.stages.0.1.weight" in sd
        if (
            "backbone.backbone._model.blocks.0.attn.gk_proj.0.weight" in sd
            and "encoder.stages.0.1.weight" in sd
            and gtr_depth.is_depth_state_dict(sd)
        ):
            return True
        return (
            "backbone.backbone._model.blocks.0.attn.gk_proj.0.weight" in sd
            and "decoder.dec_score_head.0.weight" in sd
            and "encoder.stages.0.1.weight" in sd
            and "decoder.pre_bbox_head.layers.2.weight" in sd
            and sd["decoder.pre_bbox_head.layers.2.weight"].shape[0] in (4, 5)
        )

    @classmethod
    def detect_checkpoint_task(cls, sd):
        if sem.is_semantic_state_dict(sd):
            return "semantic"
        if _POSE_HEAD_KEY in sd:
            return "pose"
        if gtr_depth.is_depth_state_dict(sd):
            return "depth"
        if is_gtr_obb_state_dict(sd):
            return "obb"
        if is_seg_state_dict(sd):
            return "segment"
        return super().detect_checkpoint_task(sd)

    @classmethod
    def detect_nb_classes(cls, sd):
        if sem.is_semantic_state_dict(sd):
            return int(sd["head.classifier.weight"].shape[0])
        # Pose is person-only; the head's two logits are not user classes.
        if _POSE_HEAD_KEY in sd:
            return 1
        if gtr_depth.is_depth_state_dict(sd):
            return 1
        return super().detect_nb_classes(sd)

    @classmethod
    def default_checkpoint_names(cls, nc):
        from .obb import DOTA_NAMES

        return dict(DOTA_NAMES) if nc == len(DOTA_NAMES) else None

    def _validate_loaded_state_dict_for_task(self, state_dict, checkpoint=None):
        is_semantic = sem.is_semantic_state_dict(state_dict)
        if is_semantic != (self.task == "semantic"):
            found = "semantic" if is_semantic else "non-semantic"
            raise RuntimeError(
                f"This is a GTR {found} checkpoint but the model was initialized "
                f"for task='{self.task}'. Pass the matching task or filename suffix."
            )
        if is_semantic:
            return super()._validate_loaded_state_dict_for_task(state_dict, checkpoint)
        is_pose = _POSE_HEAD_KEY in state_dict
        if is_pose != (self.task == "pose"):
            raise RuntimeError(
                f"Checkpoint is a GTR-{'pose' if is_pose else 'non-pose'} model but "
                f"this instance was initialized for task='{self.task}'. Pass the "
                "matching task or use a -pose filename suffix."
            )
        if is_pose:
            return
        is_depth = gtr_depth.is_depth_state_dict(state_dict)
        if is_depth != (self.task == "depth"):
            raise RuntimeError(
                "GTR depth checkpoints must be loaded with task='depth' (the "
                "'-depth' filename suffix), and other GTR checkpoints without it."
            )
        if is_depth:
            return
        is_obb = is_gtr_obb_state_dict(state_dict)
        if self.task == "obb" and not is_obb:
            raise ValueError("GTR task='obb' requires a five-coordinate OBB checkpoint")
        if self.task != "obb" and is_obb:
            raise ValueError("GTR OBB checkpoints must be loaded with task='obb'")
        if is_obb:
            return
        super()._validate_loaded_state_dict_for_task(state_dict, checkpoint)
        # Upstream seg init (whole COCO detector, fresh mask head): only as an
        # explicit transfer, and still strict for every non-mask key.
        from .seg import SEG_HEAD_PREFIX

        self._detect_to_segment_load = (
            self.task == "segment" and not is_seg_state_dict(state_dict)
        )
        if self._detect_to_segment_load:
            missing = [
                k
                for k in self.model.state_dict()
                if k not in state_dict and not k.startswith(SEG_HEAD_PREFIX)
            ]
            if missing:
                raise RuntimeError(f"Missing keys in GTR detect weights: {missing[:5]}")

    @staticmethod
    def _weight(sd, key):
        # LoRA checkpoints keep adapted Linear weights under ``.base_layer.``.
        value = sd.get(f"{key}.weight")
        return value if value is not None else sd.get(f"{key}.base_layer.weight")

    @classmethod
    def detect_size(cls, sd):
        weight = cls._weight(sd, "backbone.backbone._model.blocks.0.attn.q_proj")
        if weight is None:
            return None
        width = weight.shape[1]
        if width in (192, 256):
            return {192: "s", 256: "m"}[width]
        if width == 384:
            weight = cls._weight(sd, "decoder.decoder.layers.0.linear1")
            if weight is not None:
                return {1024: "l", 2048: "x"}.get(weight.shape[0])
            # Depth and semantic checkpoints have no transformer decoder; L and
            # X differ in the backbone MLP ratio (4 vs 6).
            weight = sd.get("backbone.backbone._model.blocks.0.mlp.gate_proj.weight")
            if weight is not None:
                return {2048: "l", 3072: "x"}.get(weight.shape[0])
        return None

    @classmethod
    def detect_size_from_filename(cls, filename):
        from ..base import BaseModel

        return BaseModel.detect_size_from_filename.__func__(cls, filename)

    @property
    def _is_segmentation(self):
        return self.task == "segment"

    HF_REVISIONS: ClassVar[dict[str, str]] = {
        "s": "74193dc356e07f51893579211ffdecf0ee2e560a",
        "m": "9b1c76a16dc09bbd0504a01aeadff03d2ecf642d",
        "l": "b828ad6d42dd8dd287244086c2543fb45e60f94d",
        "x": "b029aa3335222ffaba58f8533cc2cd007b79bc3e",
    }

    # Pinned revisions of the LibreYOLO/LibreGTR{size}-{suffix} task repositories.
    HF_TASK_REVISIONS: ClassVar[dict[tuple[str, str], str]] = {
        ("s", "segment"): "d568a30d0141363107d8e61329040aab68b42393",
        ("m", "segment"): "c7653747c2bdc5ea013cbe5eb51ecbd5f32ef417",
        ("l", "segment"): "3955ab3c9f86b5f120c2ec5f504a8c5c0a338f89",
        ("x", "segment"): "704ea55306c31e0c7dd8968a38976cf7d5cdec05",
        ("s", "pose"): "45ceda2e7f7552fe4e9c62bee585ff55831d23b1",
        ("m", "pose"): "e05646914d5d15f8647af02397cf014c1b3e9d0f",
        ("l", "pose"): "441fc9d8c9d75a42a2155e550f799898beb902bc",
        ("x", "pose"): "3ab375d558decf2ac2262671e4c938d6c8028298",
        ("s", "obb"): "c916f8c7f8448df4835d82ca14c62a2d13145200",
        ("x", "obb"): "72e076ee833b9340cfdfbd03c2edb685fdd13126",
        ("s", "depth"): "1cbea39e5b38a2efd83eca5fe8ba0a30a70de5e7",
        ("m", "depth"): "0911829a7ddff9f92a7f835c9107f62f1daf5294",
        ("l", "depth"): "ae8110133430130a4baeaaa1a32a9d5519df16fc",
        ("x", "depth"): "eb655cb7f17e162fdac08fe98b1aa4984f320de6",
        ("s", "semantic"): "37f64a7271571018f511658907eba9e2ccf0f1f0",
        ("m", "semantic"): "16c7e753dae655ea285fe2ef01ca035d961b9e54",
        ("l", "semantic"): "218b1d20f2aa9260c6f86c59d0ed43d87dbeb09b",
        ("x", "semantic"): "bf0cf95b1cf7df47ae0fa0179e54729e89c20e3e",
    }

    @classmethod
    def get_download_url(cls, filename):
        from ...tasks import task_to_suffix

        size = cls.detect_size_from_filename(filename)
        if size is None:
            return None
        task = cls.detect_task_from_filename(filename) or "detect"
        name = f"LibreGTR{size}"
        if task == "detect":
            revision = cls.HF_REVISIONS[size]
        elif (size, task) in cls.HF_TASK_REVISIONS:
            name = f"{name}-{task_to_suffix(task)}"
            revision = cls.HF_TASK_REVISIONS[(size, task)] or "main"
        else:
            return None
        if Path(filename).stem != name:
            return None
        return f"https://huggingface.co/LibreYOLO/{name}/resolve/{revision}/{name}.pt"

    def __init__(
        self, model_path, size, nb_classes=80, device="auto", task=None, **kwargs
    ):
        if task is not None and normalize_task(task) == "pose":
            nb_classes = 1
        self.num_keypoints = self.POSE_NUM_KEYPOINTS
        self.keypoint_dim = self.KEYPOINT_DIM
        super().__init__(model_path, size, nb_classes, device, task, **kwargs)
        if self.task == "semantic" and self.nb_classes == len(sem.CITYSCAPES_NAMES):
            if all(name == f"class_{i}" for i, name in self.names.items()):
                self.names = dict(sem.CITYSCAPES_NAMES)

    def _init_model(self):
        if self.task == "pose":
            if self.names == {0: "class_0"}:
                self.names = {0: "person"}
            return LibreGTRPoseModel(
                self.size, eval_spatial_size=(self.input_size, self.input_size)
            )
        if self.task == "semantic":
            return sem.LibreGTRSemModel(self.size, self.nb_classes)
        if self.task == "depth":
            self.nb_classes = 1
            return gtr_depth.LibreGTRDepthModel(
                self.size, eval_spatial_size=(self.input_size, self.input_size)
            )
        if self.task == "obb":
            from .obb_nn import LibreGTROBBModel

            return LibreGTROBBModel(
                self.size, self.nb_classes, (self.input_size, self.input_size)
            )
        return LibreGTRModel(
            self.size,
            self.nb_classes,
            mask_downsample_ratio=(
                SEG_MASK_DOWNSAMPLE_RATIO if self.task == "segment" else None
            ),
        )

    def _rebuild_for_new_classes(self, new_nb_classes):
        if self.task == "pose":
            if new_nb_classes != 1:
                raise ValueError("GTR pose is single-class (person)")
            return
        if self.task == "depth":
            # Depth has a single schema slot and no class-dependent layers.
            self.nb_classes = 1
            self.names = {0: "depth"}
            return
        super()._rebuild_for_new_classes(new_nb_classes)

    @staticmethod
    def _apply_lora(model):
        from ...training.lora import apply_lora_to_gtr

        apply_lora_to_gtr(model)

    def _strict_loading(self):
        return not getattr(self, "_detect_to_segment_load", False)

    def _load_weights(self, model_path):
        super()._load_weights(model_path)
        # Loading a custom-class checkpoint may rebuild the module after the
        # base constructor has switched the original module to evaluation.
        self.model.eval()
        if self.task == "depth":
            self.nb_classes = 1
            self.names = {0: "depth"}

    def _get_available_layers(self):
        if self.task == "semantic":
            return {
                "backbone": self.model.backbone,
                "encoder": self.model.encoder,
                "head": self.model.head,
            }
        layers = {
            "backbone": self.model.backbone,
            "encoder": self.model.encoder,
            "decoder": self.model.decoder,
        }
        if getattr(self.model, "has_mask_head", False):
            layers["mask_head"] = self.model.decoder.decoder.segmentation_head
        return layers

    def _get_preprocess_numpy(self):
        if getattr(self, "task", "detect") == "depth":
            return gtr_depth.preprocess_numpy
        from ...preprocess.ec import preprocess_numpy

        return preprocess_numpy

    @classmethod
    def _validate_imgsz(cls, imgsz, **kwargs):
        if not isinstance(imgsz, int) or imgsz < 160 or imgsz % 32:
            raise ValueError("GTR imgsz must be a multiple of 32 and at least 160")
        return imgsz

    def _preprocess(self, image, color_format="auto", input_size=None):
        if self.task == "semantic":
            return sem.preprocess_image(
                image, input_size or self.input_size, color_format=color_format
            )
        size = self._validate_imgsz(input_size or self.input_size)
        if self.task == "depth":
            return gtr_depth.preprocess_image(image, size, color_format)
        if self.task == "obb":
            from .obb import preprocess_obb_image

            return preprocess_obb_image(image, size, color_format)
        return preprocess_image(image, input_size=size, color_format=color_format)

    def _postprocess(
        self, output, conf_thres, iou_thres, original_size, max_det=300, **kwargs
    ):
        if self.task == "semantic":
            return sem.postprocess(output, original_size, kwargs.get("ratio", 1.0))
        if self.task == "depth":
            return gtr_depth.postprocess(output, original_size)
        if self.task == "pose":
            from ...postprocess.ec import postprocess_pose

            return postprocess_pose(
                output,
                conf_thres=conf_thres,
                iou_thres=iou_thres,
                original_size=original_size,
                max_det=max_det,
                num_keypoints=self.POSE_NUM_KEYPOINTS,
            )
        if self.task == "segment":
            from ...postprocess.ec import postprocess_seg

            # Upstream thresholds bilinearly resized mask logits at zero.
            return postprocess_seg(
                output,
                conf_thres=conf_thres,
                iou_thres=iou_thres,
                original_size=original_size,
                max_det=max_det,
            )
        if self.task == "obb":
            from ...postprocess.rtdetr import postprocess_obb

            return postprocess_obb(
                output,
                conf_thres,
                iou_thres,
                original_size,
                max_det=max_det,
                input_size=kwargs.get("input_size") or self.input_size,
            )
        return super()._postprocess(
            output, conf_thres, iou_thres, original_size, max_det=max_det, **kwargs
        )

    def _postprocess_semantic_logits(self, output, original_size, ratio=1.0, **kwargs):
        """Pre-argmax logits at ``original_size``; used by flip TTA."""
        return sem.logits_at(output, original_size, ratio)

    def _get_val_preprocessor(self, img_size=None):
        if self.task != "obb":
            return super()._get_val_preprocessor(img_size=img_size)
        from .obb import GTROBBValPreprocessor

        img_size = img_size or self._get_input_size()
        return GTROBBValPreprocessor(img_size=(img_size, img_size))

    @ddp_aware()
    def train(
        self,
        data: str | None = None,
        *,
        epochs: int | None = None,
        batch: int | None = None,
        imgsz: int | None = None,
        lr0: float | None = None,
        device: str = "",
        workers: int | None = None,
        seed: int | None = None,
        project: str | None = None,
        name: str | None = None,
        exist_ok: bool | None = None,
        resume: bool | str = False,
        amp: bool | None = None,
        patience: int | None = None,
        callbacks: TrainCallbacks = None,
        loggers=None,
        **kwargs,
    ) -> dict:
        """Fine-tune GTR; resume restores saved settings before explicit overrides."""
        if self.task == "semantic":
            from .sem_trainer import train_semantic

            return train_semantic(
                self,
                data=data,
                epochs=epochs,
                batch=batch,
                imgsz=imgsz,
                lr0=lr0,
                device=device,
                workers=workers,
                seed=seed,
                project=project,
                name=name,
                exist_ok=exist_ok,
                resume=resume,
                amp=amp,
                patience=patience,
                callbacks=callbacks,
                loggers=loggers,
                **kwargs,
            )
        if self.task == "depth":
            return gtr_depth.train(
                self,
                data=data,
                epochs=epochs,
                batch=batch,
                imgsz=imgsz,
                lr0=lr0,
                device=device,
                workers=workers,
                seed=seed,
                project=project,
                name=name,
                exist_ok=exist_ok,
                resume=resume,
                amp=amp,
                patience=patience,
                callbacks=callbacks,
                loggers=loggers,
                **kwargs,
            )
        if self.task == "obb":
            raise NotImplementedError(
                "GTR OBB is inference-only in LibreYOLO; training is not implemented"
            )
        from dataclasses import fields

        from libreyolo.data import load_data_config

        if self.task == "pose":
            explicit = {
                "epochs": epochs,
                "batch": batch,
                "imgsz": imgsz,
                "lr0": lr0,
                "workers": workers,
                "seed": seed,
                "project": project,
                "name": name,
                "exist_ok": exist_ok,
                "amp": amp,
                "patience": patience,
            }
            kwargs.update({k: v for k, v in explicit.items() if v is not None})
            return self._train_pose(
                data,
                device=device,
                resume=resume,
                callbacks=callbacks,
                loggers=loggers,
                **kwargs,
            )

        if self.task == "segment":
            from .seg_trainer import GTRSegConfig as config_cls
            from .seg_trainer import GTRSegTrainer as trainer_cls
        else:
            from .trainer import GTRTrainer as trainer_cls

            config_cls = GTRConfig

        kwargs.pop("pretrained", None)
        resume_path = None
        settings = {}
        if resume:
            resume_path = (
                str(resume) if isinstance(resume, (str, Path)) else self.model_path
            )
            if not resume_path:
                raise ValueError("resume=True requires a loaded training checkpoint")
            settings = self._checkpoint_train_config(resume_path)
            if not settings:
                raise ValueError(
                    "GTR resume requires a checkpoint with saved training configuration"
                )
            self._load_weights(str(resume_path))

        valid = {field.name for field in fields(config_cls)}
        settings = {
            key: value
            for key, value in settings.items()
            if key in valid and key not in {"size", "num_classes", "resume"}
        }
        explicit = {
            "data": data,
            "epochs": epochs,
            "batch": batch,
            "imgsz": imgsz,
            "lr0": lr0,
            "workers": workers,
            "seed": seed,
            "project": project,
            "name": name,
            "exist_ok": exist_ok,
            "amp": amp,
            "patience": patience,
        }
        settings.update(
            {key: value for key, value in explicit.items() if value is not None}
        )
        settings.update(kwargs)
        if device:
            settings["device"] = device
        settings.setdefault("device", "auto")
        settings.setdefault("seed", 0)
        if not settings.get("data"):
            raise ValueError(
                "GTR training requires data or a resume checkpoint containing data"
            )
        if settings.get("imgsz") is not None:
            settings["imgsz"] = self._validate_imgsz(settings["imgsz"])

        # Seed before replacing the classification heads for a custom dataset.
        self._seed_scratch_initialization(settings["seed"])
        data_config = load_data_config(
            settings["data"],
            autodownload=True,
            single_cls=bool(settings.get("single_cls", False)),
        )
        settings["data"] = data_config.get("yaml_file", settings["data"])
        names = data_config.get("names")
        nc = data_config.get("nc", len(names) if names is not None else self.nb_classes)
        if nc != self.nb_classes:
            if resume_path:
                raise ValueError(
                    "Cannot resume GTR with a different dataset class count"
                )
            self._rebuild_for_new_classes(nc)
        if isinstance(names, list):
            names = dict(enumerate(names))
        if names is not None:
            self.names = self._sanitize_names(names, self.nb_classes)

        trainer = trainer_cls(
            model=self.model,
            wrapper_model=self,
            size=self.size,
            num_classes=self.nb_classes,
            resume=bool(resume_path),
            callbacks=callbacks,
            loggers=loggers,
            **settings,
        )
        if resume_path:
            trainer.setup()
            trainer.resume(str(resume_path))
        results = trainer.train()
        best = results.get("best_checkpoint")
        if best and Path(best).exists():
            self.model_path = best
            self._load_weights(best)
        self.model.to(self.device)
        return results

    def _train_pose(
        self, data, *, device="", resume=False, callbacks=None, loggers=None, **kwargs
    ):
        """Fine-tune GTR pose on a YOLO keypoint dataset (17 COCO keypoints)."""
        from libreyolo.data import load_data_config

        from .pose_trainer import GTRPoseTrainer

        if kwargs.get("lora"):
            raise ValueError("GTR pose training does not support lora=True yet")
        kwargs.pop("pretrained", None)
        imgsz = int(kwargs.setdefault("imgsz", self.input_size))
        if imgsz != int(self.input_size):
            raise ValueError(
                f"GTR pose fine-tuning requires imgsz={self.input_size}; the "
                "decoder anchor grid is built for the native input size."
            )
        resume_path = None
        if resume:
            resume_path = (
                str(resume) if isinstance(resume, (str, Path)) else self.model_path
            )
            if not resume_path:
                raise ValueError("resume=True requires a loaded training checkpoint")
            saved = self._checkpoint_train_config(resume_path) or {}
            data = data or saved.get("data")
        if not data:
            raise ValueError("GTR pose training requires data")

        data_config = load_data_config(data, autodownload=True)
        data = data_config.get("yaml_file", data)
        kpt_shape = data_config.get("kpt_shape")
        if not kpt_shape or int(kpt_shape[0]) != self.POSE_NUM_KEYPOINTS:
            raise ValueError(
                "GTR pose fine-tuning needs 'kpt_shape: [17, 2|3]' in the dataset "
                "yaml; the keypoint head is fixed at 17 COCO keypoints."
            )
        keypoint_dim = int(kpt_shape[1]) if len(kpt_shape) > 1 else 3
        names = data_config.get("names")
        if names is not None and len(names) != 1:
            raise ValueError("GTR pose fine-tuning supports single-class datasets only")
        if isinstance(names, list):
            names = dict(enumerate(names))
        if names:
            self.names = self._sanitize_names(names, 1)

        trainer = GTRPoseTrainer(
            model=self.model,
            wrapper_model=self,
            size=self.size,
            num_classes=1,
            num_keypoints=self.POSE_NUM_KEYPOINTS,
            keypoint_dim=keypoint_dim,
            flip_idx=kwargs.pop("flip_idx", data_config.get("flip_idx")),
            data=data,
            device=device or "auto",
            resume=bool(resume_path),
            callbacks=callbacks,
            loggers=loggers,
            **kwargs,
        )
        if resume_path:
            trainer.setup()
            trainer.resume(resume_path)
        results = trainer.train()
        best = results.get("best_checkpoint")
        if best and Path(best).exists():
            self.model_path = best
            self._load_weights(best)
        self.model.to(self.device)
        return results

    def export(self, format="onnx", **kwargs):
        """Export a fixed-resolution FP32 graph using portable attention."""
        if format == "pt" and getattr(self, "_quant_manifest", None):
            return super().export(format=format, **kwargs)
        if format not in ("onnx", "torchscript"):
            raise NotImplementedError(
                "GTR currently exports to ONNX and TorchScript only"
            )
        if kwargs.get("dynamic", False):
            raise ValueError("GTR export requires dynamic=False")
        if kwargs.get("half", False):
            raise ValueError("GTR portable export currently requires FP32")
        kwargs["dynamic"] = False
        if self.task == "semantic":
            # onnxsim spends ~15 minutes folding the three window Loops on CPU
            # without shrinking the graph; opt in with simplify=True.
            kwargs.setdefault("simplify", False)
        return super().export(format=format, **kwargs)
