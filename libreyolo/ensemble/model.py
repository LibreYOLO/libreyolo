"""Cross-architecture model ensembling behind one ``predict()`` call.

:class:`LibreEnsemble` wraps two or more detection members — model paths,
constructed LibreYOLO models, exported backends, or foreign detectors wrapped
in :class:`ExternalDetector` — runs each member's own
preprocess/forward/postprocess pipeline, and fuses the resulting detections
into one ordinary :class:`~libreyolo.utils.results.Results`.

Fusion happens at the detection level, never at the tensor level: every
member keeps its own input size, normalization, and suppression. That is what
lets heterogeneous families (grid and DETR), different class counts, and
mixed ``.pt``/exported members ensemble freely.

Class spaces are unified by name: identical ``names`` dicts pass through,
otherwise the union is built and member class ids are remapped into it.
Fusion only merges boxes that share a unified class; a class known to only
one member passes through unfused, and consensus voting (``min_votes``) is
automatically capped per class by how many members know that class.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch

from ..ops.fusion import FUSIONS
from ..tasks import normalize_task
from ..utils.general import log_saved_result, resolve_save_path
from ..utils.image_loader import ImageInput, ImageLoader
from ..utils.logging import ensure_default_logging
from ..utils.predict_args import normalize_predict_kwargs
from ..utils.results import Boxes, Results
from ..utils.video import is_video_file

logger = logging.getLogger(__name__)

Detections = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]


def _sanitize_names(names) -> Dict[int, str]:
    if not isinstance(names, dict) or not names:
        raise TypeError("names must be a non-empty dict mapping class id to name")
    return {int(k): str(v) for k, v in names.items()}


class ExternalDetector:
    """Adapt any detection callable into an ensemble member.

    Args:
        fn: Callable taking a PIL image and returning ``(boxes, scores,
            labels)`` — xyxy boxes in original-image pixels, confidences, and
            class ids valid in *names*. Tensors, arrays, or nested lists all
            work.
        names: Mapping of the callable's class ids to class names.

    Example:
        >>> member = ExternalDetector(my_fn, names={0: "person"})
        >>> ens = LibreEnsemble(["LibreYOLO9s.pt", member])
    """

    task = "detect"

    def __init__(self, fn: Callable, names: Dict[int, str]):
        if not callable(fn):
            raise TypeError(f"fn must be callable, got {type(fn).__name__}")
        self.fn = fn
        self.names = _sanitize_names(names)

    def __call__(self, source: ImageInput, *, conf: float = 0.25, **kwargs) -> Results:
        img = ImageLoader.load(source)
        out = self.fn(img)
        if not isinstance(out, (tuple, list)) or len(out) != 3:
            raise TypeError(
                "ExternalDetector callable must return (boxes, scores, labels), "
                f"got {type(out).__name__}"
            )
        boxes = torch.as_tensor(out[0], dtype=torch.float32)
        scores = torch.as_tensor(out[1], dtype=torch.float32).reshape(-1)
        labels = torch.as_tensor(out[2]).reshape(-1)
        if boxes.numel() == 0:
            boxes = boxes.reshape(0, 4)
        if boxes.ndim != 2 or boxes.shape[1] != 4:
            raise ValueError(
                f"ExternalDetector boxes must have shape (N, 4), got {tuple(boxes.shape)}"
            )
        if not (boxes.shape[0] == scores.shape[0] == labels.shape[0]):
            raise ValueError(
                "ExternalDetector boxes, scores, and labels must have equal "
                f"lengths, got {boxes.shape[0]}, {scores.shape[0]}, {labels.shape[0]}"
            )
        unknown = [int(c) for c in labels.unique().tolist() if int(c) not in self.names]
        if unknown:
            raise ValueError(
                f"ExternalDetector returned class ids {unknown} not present in names"
            )

        keep = scores > conf
        w, h = img.size
        return Results(
            boxes=Boxes(boxes[keep], scores[keep], labels[keep].float()),
            orig_shape=(h, w),
            names=self.names,
        )

    predict = __call__

    def __repr__(self) -> str:
        fn_name = getattr(self.fn, "__name__", type(self.fn).__name__)
        return f"ExternalDetector(fn={fn_name}, classes={len(self.names)})"


class LibreEnsemble:
    """Combine several detectors into one model-like prediction surface.

    Members run independently and their detections are fused with a standalone
    op from :mod:`libreyolo.ops`. The result is an ordinary :class:`Results`
    whose ``names`` is the by-name union of the members' class spaces and
    whose ``speed`` breaks the cost down per member.

    Args:
        members: Two or more of — a weights path (resolved through the
            ``LibreYOLO()`` factory), a constructed model or exported backend,
            or an :class:`ExternalDetector`. All members must be detect-task.
        weights: Optional per-member trust factors (convention: proportional
            to validation mAP). Higher weight pulls fused coordinates and
            scores toward that member.
        fusion: ``"wbf"`` (default), ``"wbf_seeded"``, ``"nms"``, or a
            callable with the :mod:`libreyolo.ops.fusion` op signature.
        fusion_iou: IoU threshold for fusion clustering. Distinct from the
            per-call ``iou``, which remains each member's own NMS threshold.
        min_votes: Keep only boxes confirmed by at least this many members
            (``2`` on a two-member ensemble keeps only boxes both found).
            Automatically capped per class by how many members know the class.
            Requires a WBF fusion.

    Example:
        >>> ens = LibreEnsemble(["LibreYOLO9s.pt", "LibreRFDETRs.pt"], min_votes=2)
        >>> results = ens("image.jpg", conf=0.25)
    """

    task = "detect"

    def __init__(
        self,
        members: Sequence,
        *,
        weights: Optional[Sequence[float]] = None,
        fusion: Union[str, Callable] = "wbf",
        fusion_iou: float = 0.55,
        min_votes: int = 1,
    ):
        ensure_default_logging()
        if isinstance(members, (str, Path)) or not isinstance(members, Sequence):
            raise TypeError("members must be a sequence of two or more detectors")
        self.members = [self._resolve_member(m, i) for i, m in enumerate(members)]
        if len(self.members) < 2:
            raise ValueError(
                f"an ensemble needs at least two members, got {len(self.members)}"
            )

        n = len(self.members)
        if weights is not None:
            if len(weights) != n:
                raise ValueError(
                    f"weights has {len(weights)} entries for {n} members"
                )
            # Positivity (not non-negativity) so NaN weights also fail loudly.
            if not all(w > 0 for w in weights):
                raise ValueError("weights must all be positive")
            self.weights = [float(w) for w in weights]
        else:
            self.weights = [1.0] * n

        if callable(fusion):
            self._fusion = fusion
            self.fusion = getattr(fusion, "__name__", "custom")
        elif isinstance(fusion, str) and fusion in FUSIONS:
            self._fusion = FUSIONS[fusion]
            self.fusion = fusion
        else:
            raise ValueError(
                f"unknown fusion {fusion!r}; available: "
                f"{', '.join(sorted(FUSIONS))}, or a callable"
            )

        if isinstance(min_votes, bool) or not isinstance(min_votes, int) or min_votes < 1:
            raise ValueError(f"min_votes must be a positive int, got {min_votes!r}")
        if min_votes > n:
            raise ValueError(
                f"min_votes={min_votes} can never be met by {n} members"
            )
        if self.fusion == "nms" and min_votes > 1:
            raise ValueError(
                "fusion='nms' cannot count votes; use fusion='wbf' or "
                "'wbf_seeded' with min_votes"
            )
        self.min_votes = min_votes
        self.fusion_iou = float(fusion_iou)

        (
            self.names,
            self._luts,
            self._models_per_label,
            self._label_weights,
        ) = self._unify_names()

    # =========================================================================
    # Construction helpers
    # =========================================================================

    @staticmethod
    def _resolve_member(member, index: int):
        if isinstance(member, (str, Path)):
            from ..models import LibreYOLO

            member = LibreYOLO(str(member))
        if not callable(member) or not isinstance(getattr(member, "names", None), dict):
            raise TypeError(
                f"member {index} ({type(member).__name__}) is not a usable "
                "detector: it must be callable and expose a names dict. Wrap "
                "foreign callables in ExternalDetector."
            )
        task = normalize_task(getattr(member, "task", None), default="detect")
        if task != "detect":
            raise ValueError(
                f"ensembling supports detect members only; member {index} "
                f"({type(member).__name__}) has task={task!r}"
            )
        return member

    def _unify_names(self):
        """Union member class spaces by name and build per-member id LUTs."""
        member_names = [_sanitize_names(m.names) for m in self.members]
        union: Dict[int, str] = {}
        by_name: Dict[str, int] = {}
        luts: List[torch.Tensor] = []
        for names in member_names:
            lut = torch.full((max(names) + 1,), -1, dtype=torch.long)
            for cid in sorted(names):
                uid = by_name.setdefault(names[cid], len(by_name))
                union[uid] = names[cid]
                lut[cid] = uid
            luts.append(lut)

        models_per_label = torch.zeros(len(union), dtype=torch.long)
        label_weights = torch.zeros(len(union), dtype=torch.float32)
        for names, weight in zip(member_names, self.weights):
            for name in set(names.values()):
                models_per_label[by_name[name]] += 1
                label_weights[by_name[name]] += weight

        partial = [
            union[uid]
            for uid in sorted(union)
            if int(models_per_label[uid]) < len(self.members)
        ]
        if partial:
            shown = ", ".join(repr(c) for c in partial[:10])
            more = f", … +{len(partial) - 10} more" if len(partial) > 10 else ""
            logger.warning(
                "Ensemble label spaces differ: %d classes in the union, %d not "
                "shared by every member (%s%s). Boxes only fuse within the same "
                "class name — check member names dicts if these should match.",
                len(union), len(partial), shown, more,
            )
        return union, luts, models_per_label, label_weights

    # =========================================================================
    # Prediction
    # =========================================================================

    def __call__(
        self,
        source: ImageInput | None = None,
        *,
        conf: Union[float, Sequence[float]] = 0.25,
        iou: Union[float, Sequence[float]] = 0.45,
        imgsz: Union[int, Tuple[int, int], List, None] = None,
        device: Union[str, Sequence[str], None] = None,
        classes: Optional[List[int]] = None,
        max_det: int = 300,
        augment: bool = False,
        save: bool = False,
        output_path: Optional[str] = None,
        color_format: str = "auto",
        batch: int = 1,
        stream: bool = False,
        **kwargs,
    ) -> Union[Results, List[Results]]:
        """Run every member on *source* and return fused Results.

        ``conf``, ``iou``, ``imgsz``, and ``device`` keep their standard
        per-member meaning and broadcast to all members; ``conf``, ``iou``,
        and ``device`` also accept one value per member (``conf=[0.25, 0.4]``),
        and ``imgsz`` accepts a *list* with one entry per member — an int or
        tuple broadcasts, so ``imgsz=(480, 640)`` is one rectangular size for
        everyone while ``imgsz=[480, 640]`` is 480 for member 0 and 640 for
        member 1. Each entry must be valid for that member's family. ``augment``
        broadcasts to members that support test-time augmentation; exported
        backends ignore it. ``classes`` (union class ids) and ``max_det`` apply
        to the fused result — members run generously and the ensemble trims
        once. ``batch`` is accepted for API parity; images are processed
        sequentially.
        """
        normalize_predict_kwargs(kwargs)
        del batch
        if stream or is_video_file(source):
            raise NotImplementedError(
                "video and stream ensembling are not available yet; run the "
                "members individually for video sources"
            )

        n = len(self.members)
        conf_l = self._per_member(conf, n, "conf")
        iou_l = self._per_member(iou, n, "iou")
        imgsz_l = self._per_member(imgsz, n, "imgsz", list_only=True)
        device_l = self._per_member(device, n, "device")

        if isinstance(source, (str, Path)) and Path(source).is_dir():
            return [
                self._predict_one(
                    p, conf_l, iou_l, imgsz_l, device_l,
                    classes=classes, max_det=max_det, augment=augment,
                    save=save, output_path=output_path, color_format=color_format,
                )
                for p in ImageLoader.collect_images(source)
            ]

        return self._predict_one(
            source, conf_l, iou_l, imgsz_l, device_l,
            classes=classes, max_det=max_det, augment=augment,
            save=save, output_path=output_path, color_format=color_format,
        )

    def predict(self, *args, **kwargs) -> Union[Results, List[Results]]:
        """Alias for ``__call__``."""
        return self(*args, **kwargs)

    @staticmethod
    def _per_member(value, n: int, name: str, *, list_only: bool = False):
        """Broadcast a scalar or map a per-member sequence onto the members."""
        seq_types = (list,) if list_only else (list, tuple)
        if isinstance(value, seq_types):
            if len(value) != n:
                raise ValueError(
                    f"{name} has {len(value)} entries for {n} members"
                )
            return list(value)
        return [value] * n

    def _predict_one(
        self,
        source: ImageInput,
        conf_l: List,
        iou_l: List,
        imgsz_l: List,
        device_l: List,
        *,
        classes: Optional[List[int]],
        max_det: int,
        augment: bool,
        save: bool,
        output_path: Optional[str],
        color_format: str,
    ) -> Results:
        # Decode once; members receive the same PIL image.
        img = ImageLoader.load(source, color_format=color_format)
        w, h = img.size
        image_path = source if isinstance(source, (str, Path)) else None

        speed: Dict[str, float] = {}
        member_results: List[Results] = []
        for i, member in enumerate(self.members):
            start = time.perf_counter()
            member_results.append(
                member(
                    img,
                    conf=conf_l[i],
                    iou=iou_l[i],
                    imgsz=imgsz_l[i],
                    device=device_l[i],
                    # Members run generously; the ensemble trims once at the end.
                    max_det=max(300, max_det),
                    augment=augment,
                )
            )
            speed[f"member_{i}"] = (time.perf_counter() - start) * 1000.0

        boxes, scores, labels, model_ids = self._stack(member_results)

        start = time.perf_counter()
        fused = self._fusion(
            boxes,
            scores,
            labels,
            model_ids,
            weights=torch.tensor(self.weights, dtype=torch.float32, device=boxes.device),
            num_models=len(self.members),
            iou_thr=self.fusion_iou,
            min_votes=self.min_votes,
            models_per_label=self._models_per_label.to(boxes.device),
            label_weights=self._label_weights.to(boxes.device),
        )
        speed["fusion"] = (time.perf_counter() - start) * 1000.0
        f_boxes, f_scores, f_labels = self._validate_fused(fused)

        if classes is not None and f_labels.numel() > 0:
            keep = torch.isin(
                f_labels, torch.as_tensor(list(classes), device=f_labels.device)
            )
            f_boxes, f_scores, f_labels = f_boxes[keep], f_scores[keep], f_labels[keep]
        order = torch.argsort(f_scores, descending=True)[:max_det]
        f_boxes, f_scores, f_labels = f_boxes[order], f_scores[order], f_labels[order]

        result = Results(
            boxes=Boxes(f_boxes, f_scores, f_labels.float()),
            orig_shape=(h, w),
            path=str(image_path) if image_path else None,
            names=dict(self.names),
            speed=speed,
        )

        if save:
            self._save_annotated(result, img, image_path, output_path)
        return result

    def _stack(self, member_results: List[Results]) -> Tuple[torch.Tensor, ...]:
        """Stack member detections into unified-label fusion inputs.

        Fusion runs on the first member's output device; post-suppression row
        counts are small, so cross-device transfers are noise.
        """
        device = None
        parts: List[Detections] = []
        ids: List[torch.Tensor] = []
        for i, res in enumerate(member_results):
            bx = res.boxes
            if bx is None or len(bx) == 0:
                continue
            b = torch.as_tensor(bx.xyxy, dtype=torch.float32)
            s = torch.as_tensor(bx.conf, dtype=torch.float32).reshape(-1)
            c = torch.as_tensor(bx.cls).reshape(-1).long()
            if device is None:
                device = b.device
            b, s, c = b.to(device), s.to(device), c.to(device)

            lut = self._luts[i].to(device)
            if c.numel() > 0 and (
                int(c.min()) < 0
                or int(c.max()) >= lut.numel()
                or bool((lut[c] < 0).any())
            ):
                raise RuntimeError(
                    f"member {i} returned a class id outside its names dict; "
                    "its label space cannot be mapped into the ensemble union"
                )
            c = lut[c]

            finite = torch.isfinite(b).all(dim=1) & torch.isfinite(s)
            b, s, c = b[finite], s[finite], c[finite]
            parts.append((b, s, c))
            ids.append(torch.full((b.shape[0],), i, dtype=torch.long, device=device))

        if not parts:
            device = device or torch.device("cpu")
            return (
                torch.zeros((0, 4), dtype=torch.float32, device=device),
                torch.zeros(0, dtype=torch.float32, device=device),
                torch.zeros(0, dtype=torch.long, device=device),
                torch.zeros(0, dtype=torch.long, device=device),
            )
        return (
            torch.cat([p[0] for p in parts]),
            torch.cat([p[1] for p in parts]),
            torch.cat([p[2] for p in parts]),
            torch.cat(ids),
        )

    @staticmethod
    def _validate_fused(fused) -> Detections:
        if not isinstance(fused, (tuple, list)) or len(fused) != 3:
            raise TypeError(
                "fusion must return (boxes, scores, labels), got "
                f"{type(fused).__name__}"
            )
        boxes = torch.as_tensor(fused[0], dtype=torch.float32)
        scores = torch.as_tensor(fused[1], dtype=torch.float32).reshape(-1)
        labels = torch.as_tensor(fused[2]).reshape(-1).long()
        if boxes.numel() == 0:
            boxes = boxes.reshape(0, 4)
        if boxes.ndim != 2 or boxes.shape[1] != 4 or boxes.shape[0] != scores.shape[0]:
            raise ValueError(
                "fusion returned inconsistent shapes: "
                f"boxes {tuple(boxes.shape)}, scores {tuple(scores.shape)}"
            )
        return boxes, scores, labels

    def _save_annotated(self, result: Results, img, image_path, output_path) -> None:
        from ..utils.drawing import draw_boxes

        if len(result) > 0:
            annotated = draw_boxes(
                img,
                result.boxes.xyxy.tolist(),
                result.boxes.conf.tolist(),
                result.boxes.cls.tolist(),
                class_names=result.names,
            )
        else:
            annotated = img.copy()
        save_path = resolve_save_path(output_path, image_path, ext="jpg")
        annotated.save(save_path)
        log_saved_result(result, save_path)

    # =========================================================================
    # Not yet implemented surfaces
    # =========================================================================

    def val(self, *args, **kwargs):
        raise NotImplementedError(
            "ensemble validation is not available yet; validate members "
            "individually with member.val(...)"
        )

    def export(self, *args, **kwargs):
        raise NotImplementedError(
            "ensemble export is not available yet; export members "
            "individually with member.export(...)"
        )

    def __repr__(self) -> str:
        kinds = ", ".join(type(m).__name__ for m in self.members)
        return (
            f"LibreEnsemble(members=[{kinds}], fusion={self.fusion!r}, "
            f"fusion_iou={self.fusion_iou}, min_votes={self.min_votes}, "
            f"classes={len(self.names)})"
        )
