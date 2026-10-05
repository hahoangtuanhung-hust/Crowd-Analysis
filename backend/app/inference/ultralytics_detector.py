from __future__ import annotations

import time
from contextlib import contextmanager
from collections.abc import Sequence
from types import MethodType
from typing import Any

import numpy as np
from numpy.typing import NDArray

from backend.app.core.config import DetectorConfig
from backend.app.schemas import Detection


DETECTOR_WORKER_MODES = (
    "baseline",
    "no_sync_profile",
)


class _NonSynchronizingProfile:
    """Ultralytics-compatible wall timer without per-stage CUDA barriers."""

    def __init__(self, t: float = 0.0, device: object | None = None) -> None:
        self.t = t
        self.device = device
        self.dt = 0.0

    def __enter__(self) -> _NonSynchronizingProfile:
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type: object, value: object, traceback: object) -> None:
        self.dt = time.perf_counter() - self.start
        self.t += self.dt


@contextmanager
def _ultralytics_profile_mode(*, synchronize_stages: bool):
    """Temporarily select whether Ultralytics inserts six CUDA barriers."""
    if synchronize_stages:
        yield
        return
    from ultralytics.utils import ops

    original = ops.Profile
    ops.Profile = _NonSynchronizingProfile
    try:
        yield
    finally:
        ops.Profile = original


def _tile_ranges(length: int, count: int, overlap: float) -> list[tuple[int, int]]:
    if count == 1:
        return [(0, length)]
    tile_length = min(length, int(np.ceil(length / (count - (count - 1) * overlap))))
    starts = np.linspace(0, length - tile_length, count)
    return [(int(round(start)), int(round(start)) + tile_length) for start in starts]


def _intersection_ratios(first: Detection, second: Detection) -> tuple[float, float]:
    intersection_width = max(0.0, min(first.x2, second.x2) - max(first.x1, second.x1))
    intersection_height = max(0.0, min(first.y2, second.y2) - max(first.y1, second.y1))
    intersection = intersection_width * intersection_height
    first_area = max(0.0, first.x2 - first.x1) * max(0.0, first.y2 - first.y1)
    second_area = max(0.0, second.x2 - second.x1) * max(0.0, second.y2 - second.y1)
    union = first_area + second_area - intersection
    smaller = min(first_area, second_area)
    iou = intersection / union if union > 0.0 else 0.0
    intersection_over_smaller = intersection / smaller if smaller > 0.0 else 0.0
    return iou, intersection_over_smaller


def _box_diagonal(detection: Detection) -> float:
    return float(
        np.hypot(
            max(0.0, detection.x2 - detection.x1),
            max(0.0, detection.y2 - detection.y1),
        )
    )


def _box_center_distance(first: Detection, second: Detection) -> float:
    first_center = ((first.x1 + first.x2) / 2.0, (first.y1 + first.y2) / 2.0)
    second_center = ((second.x1 + second.x2) / 2.0, (second.y1 + second.y2) / 2.0)
    return float(np.hypot(first_center[0] - second_center[0], first_center[1] - second_center[1]))


def _same_detection(first: Detection, second: Detection, iou_threshold: float) -> bool:
    """Recognize a duplicate tile result without deleting a nearby person.

    A containment-only rule is unsafe in crowds: the visible person in front can
    be almost fully inside a larger box for the person behind.  Tile duplicates
    have both a close center and a comparable scale, so require those properties
    in addition to overlap/containment.
    """
    iou, intersection_over_smaller = _intersection_ratios(first, second)
    first_area = max(0.0, first.x2 - first.x1) * max(0.0, first.y2 - first.y1)
    second_area = max(0.0, second.x2 - second.x1) * max(0.0, second.y2 - second.y1)
    if min(first_area, second_area) <= 0.0:
        return False
    scale_ratio = min(first_area, second_area) / max(first_area, second_area)
    center_limit = 0.50 * min(_box_diagonal(first), _box_diagonal(second))
    center_close = _box_center_distance(first, second) <= max(3.0, center_limit)
    if not center_close or scale_ratio < 0.35:
        return False
    return iou >= iou_threshold or intersection_over_smaller >= 0.8


def _merge_detections(
    detections: list[Detection],
    *,
    iou_threshold: float,
    max_detections: int,
    source_ids: Sequence[int] | None = None,
) -> list[Detection]:
    if source_ids is not None and len(source_ids) != len(detections):
        raise ValueError("source_ids must have one entry per detection")
    selected: list[Detection] = []
    selected_boxes = np.empty((max_detections, 4), dtype=np.float32)
    selected_classes = np.empty(max_detections, dtype=np.int32)
    selected_source_values = np.empty(max_detections, dtype=np.int32)
    # Geometry for an accepted box is immutable. Cache it once instead of
    # rebuilding area/center/diagonal arrays for every later candidate.
    selected_areas = np.empty(max_detections, dtype=np.float32)
    selected_centers = np.empty((max_detections, 2), dtype=np.float32)
    selected_diagonals = np.empty(max_detections, dtype=np.float32)
    indexed = sorted(
        enumerate(detections),
        key=lambda item: item[1].confidence,
        reverse=True,
    )
    for original_index, candidate in indexed:
        duplicate = False
        candidate_source = source_ids[original_index] if source_ids is not None else None
        if selected:
            selected_count = len(selected)
            existing = selected_boxes[:selected_count]
            first_area = max(0.0, candidate.x2 - candidate.x1) * max(
                0.0, candidate.y2 - candidate.y1
            )
            second_area = selected_areas[:selected_count]
            intersection = np.maximum(
                0.0,
                np.minimum(candidate.x2, existing[:, 2]) - np.maximum(candidate.x1, existing[:, 0]),
            ) * np.maximum(
                0.0,
                np.minimum(candidate.y2, existing[:, 3]) - np.maximum(candidate.y1, existing[:, 1]),
            )
            union = first_area + second_area - intersection
            iou = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0.0)
            smaller = np.minimum(first_area, second_area)
            intersection_over_smaller = np.divide(
                intersection, smaller, out=np.zeros_like(intersection), where=smaller > 0.0
            )
            first_center = np.asarray(
                ((candidate.x1 + candidate.x2) / 2.0, (candidate.y1 + candidate.y2) / 2.0),
                dtype=np.float32,
            )
            second_centers = selected_centers[:selected_count]
            center_distance = np.linalg.norm(second_centers - first_center[None, :], axis=1)
            first_diagonal = float(
                np.hypot(candidate.x2 - candidate.x1, candidate.y2 - candidate.y1)
            )
            second_diagonal = selected_diagonals[:selected_count]
            center_close = center_distance <= np.maximum(
                3.0, 0.50 * np.minimum(first_diagonal, second_diagonal)
            )
            scale_ratio = np.divide(
                smaller,
                np.maximum(first_area, second_area),
                out=np.zeros_like(smaller),
                where=np.maximum(first_area, second_area) > 0.0,
            )
            same_source = (
                np.zeros(selected_count, dtype=bool)
                if source_ids is None
                else selected_source_values[:selected_count] == int(candidate_source)
            )
            duplicate = bool(
                np.any(
                    (selected_classes[:selected_count] == int(candidate.class_id))
                    & ~same_source
                    & (scale_ratio >= 0.35)
                    & center_close
                    & ((iou >= iou_threshold) | (intersection_over_smaller >= 0.8))
                )
            )
        if not duplicate:
            selected.append(candidate)
            selected_index = len(selected) - 1
            selected_boxes[selected_index] = (
                candidate.x1,
                candidate.y1,
                candidate.x2,
                candidate.y2,
            )
            box = selected_boxes[selected_index]
            box_width = np.maximum(np.float32(0.0), box[2] - box[0])
            box_height = np.maximum(np.float32(0.0), box[3] - box[1])
            selected_areas[selected_index] = box_width * box_height
            selected_centers[selected_index] = (box[:2] + box[2:]) / np.float32(2.0)
            selected_diagonals[selected_index] = np.hypot(box_width, box_height)
            selected_classes[selected_index] = candidate.class_id
            selected_source_values[selected_index] = (
                -1 if candidate_source is None else int(candidate_source)
            )
            if len(selected) >= max_detections:
                break
    return selected


def _in_ignore_region(
    detection: Detection,
    *,
    width: int,
    height: int,
    regions: list[tuple[float, float, float, float]],
) -> bool:
    center_x = (detection.x1 + detection.x2) / (2.0 * width)
    center_y = (detection.y1 + detection.y2) / (2.0 * height)
    return any(x1 <= center_x <= x2 and y1 <= center_y <= y2 for x1, y1, x2, y2 in regions)


class UltralyticsPersonDetector:
    """Person-only detector with a small, backend-neutral output contract."""

    def __init__(
        self,
        config: DetectorConfig,
        *,
        worker_mode: str = "baseline",
    ) -> None:
        from ultralytics import YOLO

        if worker_mode not in DETECTOR_WORKER_MODES:
            raise ValueError(f"Unsupported detector worker mode: {worker_mode}")
        self._config = config
        self._device = None if config.device == "auto" else config.device
        self._model: Any = YOLO(config.model)
        self._worker_mode = worker_mode
        self._phase_predictor_id: int | None = None
        self._phase_stats: dict[str, Any] = {}
        self._last_inference_stats: dict[str, Any] = {}
        self.last_scheduler_decision: Any = None

    def _ensure_worker_state(self) -> None:
        """Supply defaults for lightweight test doubles built via object.__new__."""
        if not hasattr(self, "_worker_mode"):
            self._worker_mode = "baseline"
            self._phase_predictor_id = None
            self._phase_stats = {}

    def _configure_phase_events(self) -> None:
        """Record async phase timing without adding CUDA synchronization."""
        self._ensure_worker_state()
        if self._worker_mode != "no_sync_profile":
            return
        predictor = getattr(self._model, "predictor", None)
        if predictor is None or self._phase_predictor_id == id(predictor):
            return
        import torch

        if not torch.cuda.is_available():
            return
        owner = self

        def wrap(method_name: str) -> None:
            original = getattr(predictor, method_name)

            if method_name == "preprocess":

                def measured_preprocess(predictor_self: Any, im: Any) -> Any:
                    """Mirror BasePredictor.preprocess and isolate its H2D copy.

                    This intentionally preserves Ultralytics' operation order and
                    default blocking ``Tensor.to`` behavior. Both A/B backends use
                    this path, so the profiler does not introduce a backend-only
                    optimization.
                    """
                    started = torch.cuda.Event(enable_timing=True)
                    completed = torch.cuda.Event(enable_timing=True)
                    h2d_started = torch.cuda.Event(enable_timing=True)
                    h2d_completed = torch.cuda.Event(enable_timing=True)
                    conversion_started = torch.cuda.Event(enable_timing=True)
                    conversion_completed = torch.cuda.Event(enable_timing=True)
                    wall_started = time.perf_counter()
                    started.record()
                    not_tensor = not isinstance(im, torch.Tensor)
                    if not_tensor:
                        pre_transform_started = time.perf_counter()
                        transformed = predictor_self.pre_transform(im)
                        owner._phase_stats["pre_transform_cpu_ms"] = (
                            time.perf_counter() - pre_transform_started
                        ) * 1000.0
                        stack_started = time.perf_counter()
                        im = np.stack(transformed)
                        owner._phase_stats["stack_cpu_ms"] = (
                            time.perf_counter() - stack_started
                        ) * 1000.0
                        layout_started = time.perf_counter()
                        if im.shape[-1] == 3:
                            im = im[..., ::-1]
                        im = im.transpose((0, 3, 1, 2))
                        im = np.ascontiguousarray(im)
                        owner._phase_stats["layout_contiguous_cpu_ms"] = (
                            time.perf_counter() - layout_started
                        ) * 1000.0
                        tensor_started = time.perf_counter()
                        im = torch.from_numpy(im)
                        owner._phase_stats["tensor_from_numpy_cpu_ms"] = (
                            time.perf_counter() - tensor_started
                        ) * 1000.0

                    h2d_started.record()
                    im = im.to(predictor_self.device)
                    h2d_completed.record()
                    conversion_started.record()
                    im = im.half() if predictor_self.model.fp16 else im.float()
                    if not_tensor:
                        im /= 255
                    conversion_completed.record()
                    completed.record()
                    owner._phase_stats["h2d_events"] = (h2d_started, h2d_completed)
                    owner._phase_stats["tensor_conversion_events"] = (
                        conversion_started,
                        conversion_completed,
                    )
                    owner._phase_stats["preprocess_events"] = (started, completed)
                    owner._phase_stats["preprocess_submit_ms"] = (
                        time.perf_counter() - wall_started
                    ) * 1000.0
                    return im

                setattr(
                    predictor,
                    method_name,
                    MethodType(measured_preprocess, predictor),
                )
                return

            def measured(predictor_self: Any, *args: Any, **kwargs: Any) -> Any:
                started = torch.cuda.Event(enable_timing=True)
                completed = torch.cuda.Event(enable_timing=True)
                wall_started = time.perf_counter()
                started.record()
                result = original(*args, **kwargs)
                completed.record()
                owner._phase_stats[f"{method_name}_events"] = (started, completed)
                owner._phase_stats[f"{method_name}_submit_ms"] = (
                    time.perf_counter() - wall_started
                ) * 1000.0
                return result

            setattr(predictor, method_name, MethodType(measured, predictor))

        for name in ("preprocess", "inference", "postprocess"):
            wrap(name)
        self._phase_predictor_id = id(predictor)

    def _finalize_worker_stats(self) -> dict[str, Any]:
        self._ensure_worker_state()
        stats: dict[str, Any] = {
            "worker_mode": self._worker_mode,
            "ultralytics_stage_synchronizations_per_batch": (
                0 if self._worker_mode == "no_sync_profile" else 6
            ),
            "result_cpu_transfers_per_image": 3,
        }
        for phase in ("preprocess", "inference", "postprocess"):
            events = self._phase_stats.get(f"{phase}_events")
            stats[f"event_{phase}_total_ms"] = (
                float(events[0].elapsed_time(events[1])) if events else None
            )
            stats[f"{phase}_submit_ms"] = self._phase_stats.get(f"{phase}_submit_ms")
        for phase in ("h2d", "tensor_conversion"):
            events = self._phase_stats.get(f"{phase}_events")
            stats[f"event_{phase}_total_ms"] = (
                float(events[0].elapsed_time(events[1])) if events else None
            )
        for field in (
            "pre_transform_cpu_ms",
            "stack_cpu_ms",
            "layout_contiguous_cpu_ms",
            "tensor_from_numpy_cpu_ms",
        ):
            stats[field] = self._phase_stats.get(field)
        return stats

    def warmup(
        self,
        frame_width: int = 1280,
        frame_height: int = 720,
        warmup_passes: int = 2,
        source_batch_size: int = 1,
    ) -> None:
        """Run the detector with a blank frame to trigger CUDA/TensorRT kernel
        compilation and memory allocation before the processing loop begins.

        Without warm-up, the first inference call incurs a one-time JIT
        compilation spike of ~200-500ms on T4, which distorts FPS measurements
        and can cause false queue overflows.  This eliminates that spike.

        Args:
            frame_width:  Width of the warm-up dummy frame (pixels).
            frame_height: Height of the warm-up dummy frame (pixels).
            warmup_passes: Number of blank inferences to run. ≥2 is recommended
                so that the second pass benchmarks stable steady-state latency.
        """
        dummy = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
        for _ in range(max(1, warmup_passes)):
            if source_batch_size > 1:
                self.detect_batch([dummy] * source_batch_size)
            else:
                self.detect(dummy)

    @property
    def last_inference_stats(self) -> dict[str, Any]:
        return dict(self._last_inference_stats)

    def _predict(
        self, frames: list[NDArray[np.uint8]], imgsz: int | None = None
    ) -> list[list[Detection]]:
        if not frames:
            self._last_inference_stats = {
                "model_invocations": 0,
                "inference_images": 0,
                "batch_size": 0,
                "actual_tensor_shapes": [],
                "sum_tensor_pixels": 0,
                "padding_overhead_pixels": None,
                "workload_source": "measured_input_shapes",
            }
            return []
        self._ensure_worker_state()
        self._configure_phase_events()
        self._phase_stats = {}
        prediction_options: dict[str, Any] = {
            "source": frames,
            "imgsz": imgsz if imgsz is not None else self._config.imgsz,
            "conf": self._config.confidence,
            "iou": self._config.iou,
            "max_det": self._config.max_det,
            "classes": self._config.classes,
            "device": self._device,
            "verbose": False,
        }
        if self._config.precision == "fp16":
            # Ultralytics 8.4.116 maps the legacy half=True alias to this
            # unified precision option and emits one warning per model call.
            prediction_options["quantize"] = 16
        prediction_started = time.perf_counter()
        without_stage_sync = self._worker_mode == "no_sync_profile"
        with _ultralytics_profile_mode(synchronize_stages=not without_stage_sync):
            results = self._model.predict(**prediction_options)
        # Ultralytics creates its predictor lazily during the first warmup.
        self._configure_phase_events()
        prediction_ms = (time.perf_counter() - prediction_started) * 1000
        transfer_started = time.perf_counter()
        self._last_inference_stats = {
            "model_invocations": 1,
            "inference_images": len(frames),
            "batch_size": len(frames),
            "model_predict_ms": prediction_ms,
            # Ultralytics predict() is a fused wrapper covering preprocess,
            # forward and its internal postprocess.  Only result transfer and
            # project-specific merge can be isolated accurately here.
            "ultralytics_predict_ms": prediction_ms,
            "result_transfer_ms": 0.0,
            "postprocess_ms": 0.0,
            "merge_ms": 0.0,
            "actual_tensor_shapes": [list(frame.shape) for frame in frames],
            "shape_source": "source_images_before_ultralytics_letterbox",
            "sum_tensor_pixels": int(sum(int(frame.shape[0] * frame.shape[1]) for frame in frames)),
            "padding_overhead_pixels": None,
            "workload_source": "measured_input_shapes",
        }
        batches: list[list[Detection]] = []
        yolo_preprocess_ms = 0.0
        yolo_inference_ms = 0.0
        yolo_postprocess_ms = 0.0
        for result in results:
            if hasattr(result, "speed") and isinstance(result.speed, dict):
                yolo_preprocess_ms += result.speed.get("preprocess", 0.0)
                yolo_inference_ms += result.speed.get("inference", 0.0)
                yolo_postprocess_ms += result.speed.get("postprocess", 0.0)
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                batches.append([])
                continue
            xyxy = boxes.xyxy.detach().cpu().numpy()
            confidence = boxes.conf.detach().cpu().numpy()
            class_ids = boxes.cls.detach().cpu().numpy().astype(np.int32)
            batches.append(
                [
                    Detection(
                        x1=float(coords[0]),
                        y1=float(coords[1]),
                        x2=float(coords[2]),
                        y2=float(coords[3]),
                        confidence=float(score),
                        class_id=int(class_id),
                    )
                    for coords, score, class_id in zip(xyxy, confidence, class_ids, strict=True)
                ]
            )
        transfer_ms = (time.perf_counter() - transfer_started) * 1000
        self._last_inference_stats["result_transfer_ms"] = transfer_ms
        self._last_inference_stats["postprocess_ms"] = transfer_ms

        # Expose fine-grained YOLO breakdown
        result_count = max(1, len(results))
        self._last_inference_stats["yolo_preprocess_ms"] = (
            yolo_preprocess_ms / result_count if yolo_preprocess_ms > 0 else 0.0
        )
        self._last_inference_stats["yolo_inference_ms"] = (
            yolo_inference_ms / result_count if yolo_inference_ms > 0 else prediction_ms
        )
        self._last_inference_stats["yolo_postprocess_ms"] = (
            yolo_postprocess_ms / result_count if yolo_postprocess_ms > 0 else 0.0
        )
        # Ultralytics reports the three phases per image. Preserve those
        # compatibility fields above, and also expose invocation totals so the
        # offline profiler can compare them with end-to-end frame latency.
        worker_stats = self._finalize_worker_stats()
        if self._worker_mode == "no_sync_profile":
            yolo_preprocess_ms = float(
                worker_stats.get("event_preprocess_total_ms") or yolo_preprocess_ms
            )
            yolo_inference_ms = float(
                worker_stats.get("event_inference_total_ms") or yolo_inference_ms
            )
            yolo_postprocess_ms = float(
                worker_stats.get("event_postprocess_total_ms") or yolo_postprocess_ms
            )
        self._last_inference_stats["yolo_preprocess_total_ms"] = yolo_preprocess_ms
        self._last_inference_stats["yolo_inference_total_ms"] = yolo_inference_ms
        self._last_inference_stats["yolo_postprocess_total_ms"] = yolo_postprocess_ms
        self._last_inference_stats.update(worker_stats)

        return batches

    @staticmethod
    def _touches_internal_edge(
        detection: Detection,
        *,
        tile_width: int,
        tile_height: int,
        left_internal: bool,
        top_internal: bool,
        right_internal: bool,
        bottom_internal: bool,
    ) -> bool:
        margin = 2.0
        return (
            (left_internal and detection.x1 <= margin)
            or (top_internal and detection.y1 <= margin)
            or (right_internal and detection.x2 >= tile_width - margin)
            or (bottom_internal and detection.y2 >= tile_height - margin)
        )

    @classmethod
    def _keep_tile_detection(
        cls,
        detection: Detection,
        *,
        tile_width: int,
        tile_height: int,
        left_internal: bool,
        top_internal: bool,
        right_internal: bool,
        bottom_internal: bool,
        edge_min_confidence: float,
    ) -> bool:
        """Reject only weak internal-edge artifacts before source-aware merge.

        This policy deliberately does not look at person size or overlap.  The
        later merge has full-frame coordinates and can distinguish aligned
        duplicate tile results from two nearby pedestrians.
        """
        return (
            not cls._touches_internal_edge(
                detection,
                tile_width=tile_width,
                tile_height=tile_height,
                left_internal=left_internal,
                top_internal=top_internal,
                right_internal=right_internal,
                bottom_internal=bottom_internal,
            )
            or detection.confidence >= edge_min_confidence
        )

    @property
    def config(self) -> DetectorConfig:
        """Resolved detector settings used by the scheduler/diagnostics."""
        return self._config

    def tile_regions(
        self, frame_width: int, frame_height: int
    ) -> tuple[tuple[int, int, int, int], ...]:
        """Return detector tile geometry in source-image coordinates.

        The motion scheduler is deliberately restricted to this validated
        geometry.  It never invents crops which would change model behavior.
        """
        if not self._config.tiled_inference:
            return ()
        if self._config.tile_regions_normalized:
            return tuple(
                (
                    int(round(x1 * frame_width)),
                    int(round(y1 * frame_height)),
                    int(round(x2 * frame_width)),
                    int(round(y2 * frame_height)),
                )
                for x1, y1, x2, y2 in self._config.tile_regions_normalized
            )
        x_ranges = _tile_ranges(frame_width, self._config.tile_columns, self._config.tile_overlap)
        y_ranges = _tile_ranges(frame_height, self._config.tile_rows, self._config.tile_overlap)
        return tuple((x1, y1, x2, y2) for y1, y2 in y_ranges for x1, x2 in x_ranges)

    def reference_image_count(self, frame_width: int, frame_height: int) -> int:
        if not self._config.tiled_inference:
            return 1
        return len(self.tile_regions(frame_width, frame_height)) + int(
            self._config.tile_include_full_frame
        )

    def detect(
        self,
        frame: NDArray[np.uint8],
        *,
        tile_indices: Sequence[int] | None = None,
    ) -> list[Detection]:
        """Detect people on the reference frame or a validated tile subset.

        ``tile_indices=None`` preserves the historical full reference pass.
        A subset is only honored for tiled inference; non-tiled detectors always
        fall back to the reference pass so enabling the scheduler cannot silently
        reduce coverage for a different backend/profile.
        """
        if self._config.perspective_regions.enabled:
            return self._detect_perspective(frame)

        if not self._config.tiled_inference:
            height, width = frame.shape[:2]
            return [
                detection
                for detection in self._predict([frame])[0]
                if not _in_ignore_region(
                    detection,
                    width=width,
                    height=height,
                    regions=self._config.ignore_regions,
                )
            ]

        if tile_indices is not None:
            return self.detect_regions(frame, tile_indices)

        tile_prepare_started = time.perf_counter()
        height, width = frame.shape[:2]
        inputs: list[NDArray[np.uint8]] = []
        metadata: list[tuple[int, int, int, int] | None] = []
        if self._config.tile_include_full_frame:
            inputs.append(frame)
            metadata.append(None)

        for x1, y1, x2, y2 in self.tile_regions(width, height):
            inputs.append(frame[y1:y2, x1:x2])
            metadata.append((x1, y1, x2, y2))

        tile_prepare_ms = (time.perf_counter() - tile_prepare_started) * 1000
        merged = self._merge_prediction_batches(frame, inputs, metadata)
        self._last_inference_stats["tile_prepare_ms"] = tile_prepare_ms
        return merged

    def _detect_perspective(self, frame: NDArray[np.uint8]) -> list[Detection]:
        import cv2
        import time
        from backend.app.schemas import Detection

        preprocess_started = time.perf_counter()
        height, width = frame.shape[:2]

        groups = {}
        strategy = self._config.perspective_regions.strategy

        counts = {"full": 0, "far": 0, "middle": 0, "near": 0}

        if strategy == "full_frame_plus_far":
            group_key = (self._config.imgsz, False, 1.0, None)
            groups.setdefault(group_key, []).append(frame)
            groups.setdefault(group_key + ("metadata",), []).append(
                (0, 0, width, height, True, "full")
            )
            counts["full"] += 1

            far = self._config.perspective_regions.far
            y1 = int(far.y_min * height)
            y2 = int(far.y_max * height)
            if y1 < y2:
                crop = frame[y1:y2, 0:width]
                tiles = _tile_ranges(width, far.tiles.cols, far.tiles.overlap)
                far_group_key = (
                    far.imgsz,
                    far.upscale.enabled,
                    far.upscale.scale,
                    far.upscale.interpolation,
                )

                for tx1, tx2 in tiles:
                    tile_crop = crop[:, tx1:tx2]
                    if far.upscale.enabled:
                        interp = {
                            "linear": cv2.INTER_LINEAR,
                            "cubic": cv2.INTER_CUBIC,
                            "lanczos": cv2.INTER_LANCZOS4,
                        }[far.upscale.interpolation]
                        new_w = int(tile_crop.shape[1] * far.upscale.scale)
                        new_h = int(tile_crop.shape[0] * far.upscale.scale)
                        tile_crop = cv2.resize(tile_crop, (new_w, new_h), interpolation=interp)
                    groups.setdefault(far_group_key, []).append(tile_crop)
                    groups.setdefault(far_group_key + ("metadata",), []).append(
                        (tx1, y1, tx2 - tx1, y2 - y1, False, "far")
                    )
                    counts["far"] += 1
        elif strategy == "region_only":
            for region_name in ["far", "middle", "near"]:
                region = getattr(self._config.perspective_regions, region_name)
                y1 = int(region.y_min * height)
                y2 = int(region.y_max * height)
                if y1 >= y2:
                    continue
                crop = frame[y1:y2, 0:width]

                y_tiles = _tile_ranges(y2 - y1, region.tiles.rows, region.tiles.overlap)
                x_tiles = _tile_ranges(width, region.tiles.cols, region.tiles.overlap)

                group_key = (
                    region.imgsz,
                    region.upscale.enabled,
                    region.upscale.scale,
                    region.upscale.interpolation,
                )

                for ty1, ty2 in y_tiles:
                    for tx1, tx2 in x_tiles:
                        tile_crop = crop[ty1:ty2, tx1:tx2]
                        if region.upscale.enabled:
                            interp = {
                                "linear": cv2.INTER_LINEAR,
                                "cubic": cv2.INTER_CUBIC,
                                "lanczos": cv2.INTER_LANCZOS4,
                            }[region.upscale.interpolation]
                            new_w = int(tile_crop.shape[1] * region.upscale.scale)
                            new_h = int(tile_crop.shape[0] * region.upscale.scale)
                            tile_crop = cv2.resize(tile_crop, (new_w, new_h), interpolation=interp)
                        groups.setdefault(group_key, []).append(tile_crop)
                        groups.setdefault(group_key + ("metadata",), []).append(
                            (tx1, y1 + ty1, tx2 - tx1, ty2 - ty1, False, region_name)
                        )
                        counts[region_name] += 1

        preprocess_ms = (time.perf_counter() - preprocess_started) * 1000
        detector_started = time.perf_counter()

        all_candidates: list[Detection] = []
        all_candidate_sources: list[int] = []
        source_id_counter = 0

        stats = {
            "model_invocations": 0,
            "inference_images": 0,
            "model_predict_ms": 0.0,
            "result_transfer_ms": 0.0,
        }

        raw_detections = {"full": 0, "far": 0, "middle": 0, "near": 0}

        for key, inputs in groups.items():
            if len(key) > 4:
                continue
            imgsz, upscale_enabled, scale_factor, interpolation = key
            metadata_list = groups[key + ("metadata",)]

            prediction_batches = self._predict(inputs, imgsz=imgsz)
            stats["model_invocations"] += self._last_inference_stats.get("model_invocations", 0)
            stats["inference_images"] += self._last_inference_stats.get("inference_images", 0)
            stats["model_predict_ms"] += self._last_inference_stats.get("model_predict_ms", 0.0)
            stats["result_transfer_ms"] += self._last_inference_stats.get("result_transfer_ms", 0.0)

            for batch, meta in zip(prediction_batches, metadata_list):
                x1_offset, y1_offset, orig_w, orig_h, is_full, region_name = meta

                raw_detections[region_name] += len(batch)

                for det in batch:
                    det_x1, det_y1, det_x2, det_y2 = det.x1, det.y1, det.x2, det.y2
                    if upscale_enabled:
                        det_x1 /= scale_factor
                        det_y1 /= scale_factor
                        det_x2 /= scale_factor
                        det_y2 /= scale_factor

                    if not is_full:
                        if not self._keep_tile_detection(
                            Detection(
                                x1=det_x1,
                                y1=det_y1,
                                x2=det_x2,
                                y2=det_y2,
                                confidence=det.confidence,
                                class_id=det.class_id,
                            ),
                            tile_width=orig_w,
                            tile_height=orig_h,
                            left_internal=x1_offset > 0,
                            top_internal=y1_offset > 0,
                            right_internal=(x1_offset + orig_w) < width,
                            bottom_internal=(y1_offset + orig_h) < height,
                            edge_min_confidence=self._config.tile_edge_min_confidence,
                        ):
                            continue

                    all_candidates.append(
                        Detection(
                            x1=det_x1 + x1_offset,
                            y1=det_y1 + y1_offset,
                            x2=det_x2 + x1_offset,
                            y2=det_y2 + y1_offset,
                            confidence=det.confidence,
                            class_id=det.class_id,
                        )
                    )
                    all_candidate_sources.append(source_id_counter)
                source_id_counter += 1

        detector_ms = (time.perf_counter() - detector_started) * 1000
        merge_started = time.perf_counter()

        filtered: list[Detection] = []
        filtered_sources: list[int] = []
        for det, src_id in zip(all_candidates, all_candidate_sources):
            if _in_ignore_region(
                det, width=width, height=height, regions=self._config.ignore_regions
            ):
                continue
            filtered.append(det)
            filtered_sources.append(src_id)

        merged = _merge_detections(
            filtered,
            iou_threshold=self._config.tile_merge_iou,
            max_detections=self._config.max_det,
            source_ids=filtered_sources,
        )

        merge_ms = (time.perf_counter() - merge_started) * 1000

        stats["merge_ms"] = merge_ms
        stats["postprocess_ms"] = stats["result_transfer_ms"] + merge_ms

        # Additional metrics
        stats["detector_images_total"] = stats["inference_images"]
        stats["detector_images_far"] = counts["far"]
        stats["detector_images_middle"] = counts["middle"]
        stats["detector_images_near"] = counts["near"]
        stats["detector_images_full"] = counts["full"]

        stats["detections_far_raw"] = raw_detections["far"]
        stats["detections_middle_raw"] = raw_detections["middle"]
        stats["detections_near_raw"] = raw_detections["near"]
        stats["detections_after_merge"] = len(merged)

        stats["perspective_preprocess_ms"] = preprocess_ms
        stats["perspective_detector_ms"] = detector_ms
        stats["perspective_merge_ms"] = merge_ms

        # Re-populate self._last_inference_stats with the accumulated values
        self._last_inference_stats = stats

        return merged

    def detect_batch(
        self,
        frames: Sequence[NDArray[np.uint8]],
    ) -> list[list[Detection]]:
        """Detect an ordered batch of independent source frames in one model call.

        Tile predictions are still merged independently per source frame, so a
        box can never suppress a box belonging to another frame.  Callers must
        apply tracking and analytics to the returned lists in source order.
        """
        source_frames = list(frames)
        if not source_frames:
            self._predict([])
            return []
        if not self._config.tiled_inference:
            predictions = self._predict(source_frames)
            outputs: list[list[Detection]] = []
            for frame, detections in zip(source_frames, predictions, strict=True):
                height, width = frame.shape[:2]
                outputs.append(
                    [
                        detection
                        for detection in detections
                        if not _in_ignore_region(
                            detection,
                            width=width,
                            height=height,
                            regions=self._config.ignore_regions,
                        )
                    ]
                )
            return outputs

        tile_prepare_started = time.perf_counter()
        flat_inputs: list[NDArray[np.uint8]] = []
        frame_metadata: list[list[tuple[int, int, int, int] | None]] = []
        for frame in source_frames:
            height, width = frame.shape[:2]
            metadata: list[tuple[int, int, int, int] | None] = []
            if self._config.tile_include_full_frame:
                flat_inputs.append(frame)
                metadata.append(None)
            for x1, y1, x2, y2 in self.tile_regions(width, height):
                flat_inputs.append(frame[y1:y2, x1:x2])
                metadata.append((x1, y1, x2, y2))
            frame_metadata.append(metadata)

        tile_prepare_ms = (time.perf_counter() - tile_prepare_started) * 1000
        flat_predictions = self._predict(flat_inputs)
        outputs = []
        offset = 0
        total_merge_ms = 0.0
        for frame, metadata in zip(source_frames, frame_metadata, strict=True):
            count = len(metadata)
            outputs.append(
                self._merge_prediction_results(
                    frame,
                    flat_predictions[offset : offset + count],
                    metadata,
                )
            )
            total_merge_ms += float(self._last_inference_stats.get("merge_ms") or 0.0)
            offset += count
        self._last_inference_stats["merge_ms"] = total_merge_ms
        self._last_inference_stats["postprocess_ms"] = (
            float(self._last_inference_stats.get("result_transfer_ms") or 0.0) + total_merge_ms
        )
        self._last_inference_stats["source_batch_size"] = len(source_frames)
        self._last_inference_stats["tile_prepare_ms"] = tile_prepare_ms
        return outputs

    def prepare_source_batch(
        self,
        frames: Sequence[NDArray[np.uint8]],
    ) -> tuple[
        list[NDArray[np.uint8]],
        list[list[tuple[int, int, int, int] | None]],
        float,
    ]:
        """Prepare an ordered offline batch without invoking the model."""
        started = time.perf_counter()
        flat_inputs: list[NDArray[np.uint8]] = []
        frame_metadata: list[list[tuple[int, int, int, int] | None]] = []
        for frame in frames:
            height, width = frame.shape[:2]
            metadata: list[tuple[int, int, int, int] | None] = []
            if not self._config.tiled_inference:
                flat_inputs.append(frame)
                metadata.append(None)
            else:
                if self._config.tile_include_full_frame:
                    flat_inputs.append(frame)
                    metadata.append(None)
                for x1, y1, x2, y2 in self.tile_regions(width, height):
                    flat_inputs.append(frame[y1:y2, x1:x2])
                    metadata.append((x1, y1, x2, y2))
            frame_metadata.append(metadata)
        return flat_inputs, frame_metadata, (time.perf_counter() - started) * 1000

    def infer_prepared_batch(
        self,
        inputs: Sequence[NDArray[np.uint8]],
    ) -> tuple[list[list[Detection]], dict[str, Any]]:
        """Infer one prepared batch and snapshot GPU-worker-owned stats."""
        predictions = self._predict(list(inputs))
        return predictions, self.last_inference_stats

    def merge_prepared_frame(
        self,
        frame: NDArray[np.uint8],
        prediction_batches: Sequence[list[Detection]],
        metadata: Sequence[tuple[int, int, int, int] | None],
    ) -> tuple[list[Detection], float]:
        """Merge one frame without mutating stats owned by the GPU worker."""
        return self._merge_prediction_results_timed(frame, prediction_batches, metadata)

    def detect_regions(
        self,
        frame: NDArray[np.uint8],
        tile_indices: Sequence[int],
    ) -> list[Detection]:
        """Run exactly the requested existing detector tiles.

        An empty selection intentionally performs no model call.  Callers use
        this only for a scheduler decision; an empty result is never fed to the
        common-path evidence as an observed measurement.
        """
        if not self._config.tiled_inference:
            return self.detect(frame)
        height, width = frame.shape[:2]
        regions = self.tile_regions(width, height)
        indices = tuple(sorted({int(index) for index in tile_indices}))
        if any(index < 0 or index >= len(regions) for index in indices):
            raise ValueError("tile index is outside the detector reference profile")
        if not indices:
            return []
        metadata = [regions[index] for index in indices]
        inputs = [frame[y1:y2, x1:x2] for x1, y1, x2, y2 in metadata]
        return self._merge_prediction_batches(frame, inputs, metadata)

    def _merge_prediction_batches(
        self,
        frame: NDArray[np.uint8],
        inputs: list[NDArray[np.uint8]],
        metadata: list[tuple[int, int, int, int] | None],
    ) -> list[Detection]:
        prediction_batches = self._predict(inputs)
        return self._merge_prediction_results(frame, prediction_batches, metadata)

    def _merge_prediction_results_timed(
        self,
        frame: NDArray[np.uint8],
        prediction_batches: Sequence[list[Detection]],
        metadata: Sequence[tuple[int, int, int, int] | None],
    ) -> tuple[list[Detection], float]:
        height, width = frame.shape[:2]
        merge_started = time.perf_counter()

        candidates: list[Detection] = []
        candidate_sources: list[int] = []
        for source_id, (tile_detections, tile) in enumerate(
            zip(prediction_batches, metadata, strict=True)
        ):
            if tile is None:
                candidates.extend(tile_detections)
                candidate_sources.extend([source_id] * len(tile_detections))
                continue
            tile_x1, tile_y1, tile_x2, tile_y2 = tile
            tile_width = tile_x2 - tile_x1
            tile_height = tile_y2 - tile_y1
            for detection in tile_detections:
                # Do not discard edge-touching tile results.  A small or
                # occluded pedestrian can genuinely be clipped by a tile. Keep
                # strong edge evidence and let the source-aware merge remove
                # aligned duplicates; weak edge boxes are usually tile-border
                # artifacts and would otherwise multiply assignment cost.
                if not self._keep_tile_detection(
                    detection,
                    tile_width=tile_width,
                    tile_height=tile_height,
                    left_internal=tile_x1 > 0,
                    top_internal=tile_y1 > 0,
                    right_internal=tile_x2 < width,
                    bottom_internal=tile_y2 < height,
                    edge_min_confidence=self._config.tile_edge_min_confidence,
                ):
                    continue
                candidates.append(
                    Detection(
                        x1=detection.x1 + tile_x1,
                        y1=detection.y1 + tile_y1,
                        x2=detection.x2 + tile_x1,
                        y2=detection.y2 + tile_y1,
                        confidence=detection.confidence,
                        class_id=detection.class_id,
                    )
                )
                candidate_sources.append(source_id)

        filtered: list[Detection] = []
        filtered_sources: list[int] = []
        for detection, source_id in zip(candidates, candidate_sources, strict=True):
            if _in_ignore_region(
                detection,
                width=width,
                height=height,
                regions=self._config.ignore_regions,
            ):
                continue
            filtered.append(detection)
            filtered_sources.append(source_id)
        merged = _merge_detections(
            filtered,
            iou_threshold=self._config.tile_merge_iou,
            max_detections=self._config.max_det,
            source_ids=filtered_sources,
        )
        return merged, (time.perf_counter() - merge_started) * 1000

    def _merge_prediction_results(
        self,
        frame: NDArray[np.uint8],
        prediction_batches: Sequence[list[Detection]],
        metadata: Sequence[tuple[int, int, int, int] | None],
    ) -> list[Detection]:
        merged, merge_ms = self._merge_prediction_results_timed(frame, prediction_batches, metadata)
        self._last_inference_stats["merge_ms"] = merge_ms
        self._last_inference_stats["postprocess_ms"] = float(
            self._last_inference_stats.get("postprocess_ms") or 0.0
        ) + float(self._last_inference_stats["merge_ms"])
        return merged
