from __future__ import annotations

import time
import logging
from contextlib import contextmanager
from collections.abc import Sequence
from dataclasses import dataclass
from types import MethodType
from typing import Any

import numpy as np
from numpy.typing import NDArray

from backend.app.core.config import DetectorConfig
from backend.app.schemas import Detection

LOGGER = logging.getLogger(__name__)


DETECTOR_WORKER_MODES = (
    "baseline",
    "no_sync_profile",
)


@dataclass(frozen=True, slots=True)
class _PerspectiveCrop:
    bounds: tuple[int, int, int, int]
    region: str
    group: tuple[int, bool, float, str | None]
    scale_x: float = 1.0
    scale_y: float = 1.0


CropMetadata = tuple[int, int, int, int] | _PerspectiveCrop | None


class _PreparedInputs(list[NDArray[np.uint8]]):
    """Batch-owned resolution groups; no mutable planning state on the worker."""

    def __init__(self) -> None:
        super().__init__()
        self.groups: dict[tuple[int, bool, float, str | None], list[int]] = {}
        self.preprocessed: NDArray[np.uint8] | None = None


def _pack_rgb_chw(images: Sequence[NDArray[np.uint8]]) -> NDArray[np.uint8]:
    """Pack letterboxed BGR images directly into contiguous RGB NCHW.

    Avoid a second full-size NHWC batch allocation and memory copy. Each
    image is copied with the same channel reversal as Ultralytics preprocess.
    """
    height, width, channels = images[0].shape
    packed = np.empty((len(images), channels, height, width), dtype=np.uint8)
    for index, image in enumerate(images):
        source = image[..., ::-1] if channels == 3 else image
        np.copyto(packed[index], source.transpose(2, 0, 1))
    return packed


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


def _merge_detections_reference(
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


def _merge_detections(
    detections: list[Detection],
    *,
    iou_threshold: float,
    max_detections: int,
    source_ids: Sequence[int] | None = None,
) -> list[Detection]:
    """Source-aware greedy suppression with bounded vectorized comparisons.

    Keep confidence order and the reference float32 geometry. Compute overlap
    for blocks rather than allocating half a dozen arrays for each candidate.
    Only previously accepted boxes suppress a candidate (never rejected boxes).
    """
    if source_ids is not None and len(source_ids) != len(detections):
        raise ValueError("source_ids must have one entry per detection")
    if len(detections) < 32:
        return _merge_detections_reference(
            detections, iou_threshold=iou_threshold,
            max_detections=max_detections, source_ids=source_ids,
        )
    order = sorted(range(len(detections)), key=lambda i: detections[i].confidence, reverse=True)
    ordered = [detections[i] for i in order]
    boxes = np.asarray([d.xyxy for d in ordered], dtype=np.float32)
    sizes = np.maximum(0.0, boxes[:, 2:] - boxes[:, :2])
    areas = sizes[:, 0] * sizes[:, 1]
    centers = (boxes[:, :2] + boxes[:, 2:]) / np.float32(2.0)
    diagonals = np.hypot(sizes[:, 0], sizes[:, 1])
    classes = np.asarray([d.class_id for d in ordered])
    sources = np.asarray([source_ids[i] for i in order]) if source_ids is not None else None
    # The candidate side of the reference kernel calculates geometry before
    # casting to float32; preserve that distinction at threshold boundaries.
    candidate_areas = np.asarray([
        max(0.0, d.x2 - d.x1) * max(0.0, d.y2 - d.y1) for d in ordered
    ], dtype=np.float32)
    candidate_centers = np.asarray([
        ((d.x1 + d.x2) / 2.0, (d.y1 + d.y2) / 2.0) for d in ordered
    ], dtype=np.float32)
    candidate_diagonals = np.asarray([
        np.hypot(d.x2 - d.x1, d.y2 - d.y1) for d in ordered
    ], dtype=np.float32)
    selected_indices: list[int] = []
    # Bound scratch memory even for up to 80 detector images. Typical crowded
    # frames use 128 rows; large candidate lists use a smaller block.
    block_size = max(1, min(128, 262144 // len(ordered)))
    for start in range(0, len(ordered), block_size):
        stop = min(len(ordered), start + block_size)
        first = boxes[start:stop]
        # Rejected candidates and later blocks can never suppress this block.
        # Compare only with accepted predecessors and candidates in this block.
        prefix_count = len(selected_indices)
        reference_indices = np.asarray([*selected_indices, *range(start, stop)])
        reference_boxes = boxes[reference_indices]
        selected_columns = list(range(prefix_count))
        intersection = np.maximum(
            0.0, np.minimum(first[:, None, 2], reference_boxes[None, :, 2])
            - np.maximum(first[:, None, 0], reference_boxes[None, :, 0]),
        ) * np.maximum(
            0.0, np.minimum(first[:, None, 3], reference_boxes[None, :, 3])
            - np.maximum(first[:, None, 1], reference_boxes[None, :, 1]),
        )
        first_area = candidate_areas[start:stop, None]
        reference_areas = areas[reference_indices][None, :]
        union = first_area + reference_areas - intersection
        smaller = np.minimum(first_area, reference_areas)
        larger = np.maximum(first_area, reference_areas)
        iou = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
        containment = np.divide(
            intersection, smaller, out=np.zeros_like(intersection), where=smaller > 0,
        )
        scale = np.divide(smaller, larger, out=np.zeros_like(intersection), where=larger > 0)
        distance = np.linalg.norm(
            centers[reference_indices][None, :, :] - candidate_centers[start:stop, None, :], axis=2,
        )
        center_limit = np.maximum(
            3.0, 0.5 * np.minimum(candidate_diagonals[start:stop, None], diagonals[reference_indices][None, :]),
        )
        duplicates = (
            (classes[start:stop, None] == classes[reference_indices][None, :])
            & (scale >= 0.35) & (distance <= center_limit)
            & ((iou >= iou_threshold) | (containment >= 0.8))
        )
        if sources is not None:
            duplicates &= sources[start:stop, None] != sources[reference_indices][None, :]
        for index in range(start, stop):
            if selected_columns and np.any(duplicates[index - start, selected_columns]):
                continue
            selected_indices.append(index)
            selected_columns.append(prefix_count + index - start)
            if len(selected_indices) >= max_detections:
                return [ordered[i] for i in selected_indices]
    return [ordered[i] for i in selected_indices]


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

                    RGB packing preserves the input bytes while avoiding the
                    intermediate NHWC batch. A prepared offline batch can move
                    this CPU work to the bounded producer queue. H2D and numeric
                    conversion retain the original operation order.
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
                        prepared = getattr(owner, "_pending_preprocess", None)
                        if prepared is not None:
                            if prepared.shape[0] != len(im):
                                raise ValueError("Prepared detector batch size mismatch")
                            im = prepared
                            owner._phase_stats.update(pre_transform_cpu_ms=0.0,
                                                      stack_cpu_ms=0.0,
                                                      layout_contiguous_cpu_ms=0.0)
                        else:
                            pre_transform_started = time.perf_counter()
                            transformed = predictor_self.pre_transform(im)
                            owner._phase_stats["pre_transform_cpu_ms"] = (
                                time.perf_counter() - pre_transform_started
                            ) * 1000.0
                            layout_started = time.perf_counter()
                            im = _pack_rgb_chw(transformed)
                            owner._phase_stats["stack_cpu_ms"] = 0.0
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
                    owner._phase_stats["model_tensor_shape"] = list(im.shape)
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
                result = (owner._graph_inference(original, *args, **kwargs)
                          if method_name == "inference" else original(*args, **kwargs))
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

    def _graph_inference(self, original: Any, im: Any, *args: Any, **kwargs: Any) -> Any:
        """Replay the same PyTorch kernels for a warmed, fixed input shape.

        The graph owns its input/output buffers; callers transfer results before
        the next inference. Shape changes use ordinary inference rather than
        recapturing in the timed loop. No numeric precision conversion is added.
        """
        import torch

        model = self._model.predictor.model
        if (not self._config.cuda_graph_inference or getattr(model, "format", None) != "pt"
                or not im.is_cuda or getattr(self, "_graph_disabled_reason", None)):
            return original(im, *args, **kwargs)
        signature = (tuple(im.shape), im.dtype, im.device)
        state = getattr(self, "_cuda_graph_state", None)
        if state is not None and state[0] != signature:
            self._phase_stats["cuda_graph_shape_fallback"] = True
            return original(im, *args, **kwargs)
        if state is None:
            try:
                static_input = torch.empty_like(im)
                static_input.copy_(im)
                stream = torch.cuda.Stream(device=im.device)
                stream.wait_stream(torch.cuda.current_stream(im.device))
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        original(static_input, *args, **kwargs)
                torch.cuda.current_stream(im.device).wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    static_output = original(static_input, *args, **kwargs)
                state = signature, graph, static_input, static_output
                self._cuda_graph_state = state
            except RuntimeError as exc:
                self._graph_disabled_reason = str(exc)
                LOGGER.warning("CUDA graph unavailable; using ordinary GPU inference: %s", exc)
                return original(im, *args, **kwargs)
        state[2].copy_(im)
        state[1].replay()
        self._phase_stats["cuda_graph_replayed"] = True
        return state[3]

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
            "model_tensor_shape",
            "cuda_graph_replayed",
            "cuda_graph_shape_fallback",
        ):
            stats[field] = self._phase_stats.get(field)
        stats["cuda_graph_disabled_reason"] = getattr(self, "_graph_disabled_reason", None)
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
        self, frames: list[NDArray[np.uint8]], imgsz: int | None = None,
        *, preprocessed: NDArray[np.uint8] | None = None,
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
            "imgsz": self._config.input_shape or (imgsz if imgsz is not None else self._config.imgsz),
            "conf": self._config.confidence,
            "iou": self._config.iou,
            "max_det": self._config.max_det,
            "classes": self._config.classes,
            "device": self._device,
            "verbose": False,
        }
        if self._config.input_shape is not None:
            prediction_options["rect"] = False
        if self._config.precision == "fp16":
            # Ultralytics 8.4.116 maps the legacy half=True alias to this
            # unified precision option and emits one warning per model call.
            prediction_options["quantize"] = 16
        prediction_started = time.perf_counter()
        without_stage_sync = self._worker_mode == "no_sync_profile"
        self._pending_preprocess = preprocessed
        try:
            with _ultralytics_profile_mode(synchronize_stages=not without_stage_sync):
                results = self._model.predict(**prediction_options)
        finally:
            self._pending_preprocess = None
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
        if self._config.perspective_regions.enabled:
            return len(self._perspective_layout(frame_width, frame_height))
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

    def _perspective_layout(self, width: int, height: int) -> list[_PerspectiveCrop]:
        policy = self._config.perspective_regions
        layout: list[_PerspectiveCrop] = []
        if policy.strategy == "full_frame_plus_far":
            layout.append(_PerspectiveCrop(
                (0, 0, width, height), "full", (self._config.imgsz, False, 1.0, None),
            ))
        names = ("far",) if policy.strategy == "full_frame_plus_far" else ("far", "middle", "near")
        for name in names:
            region = getattr(policy, name)
            y1, y2 = int(region.y_min * height), int(region.y_max * height)
            if y1 >= y2:
                continue
            group = (region.imgsz, region.upscale.enabled,
                     region.upscale.scale, region.upscale.interpolation)
            for top, bottom in _tile_ranges(y2 - y1, region.tiles.rows, region.tiles.overlap):
                for left, right in _tile_ranges(width, region.tiles.cols, region.tiles.overlap):
                    layout.append(_PerspectiveCrop((left, y1 + top, right, y1 + bottom), name, group))
        return layout

    def _prepare_perspective_frame(self, frame: NDArray[np.uint8]) -> tuple[list, list]:
        import cv2

        height, width = frame.shape[:2]
        inputs, metadata = [], []
        for crop in self._perspective_layout(width, height):
            left, top, right, bottom = crop.bounds
            image = frame[top:bottom, left:right]
            if crop.group[1]:
                scale = crop.group[2]
                interpolation = {"linear": cv2.INTER_LINEAR, "cubic": cv2.INTER_CUBIC,
                                 "lanczos": cv2.INTER_LANCZOS4}[crop.group[3]]
                image = cv2.resize(image, (max(1, int((right - left) * scale)),
                                          max(1, int((bottom - top) * scale))),
                                   interpolation=interpolation)
            # Integer resize dimensions can differ from the requested factor.
            # Map each axis with its actual scale to avoid subpixel box drift.
            crop = _PerspectiveCrop(crop.bounds, crop.region, crop.group,
                                    image.shape[1] / (right - left),
                                    image.shape[0] / (bottom - top))
            inputs.append(image)
            metadata.append(crop)
        return inputs, metadata

    def _detect_perspective(self, frame: NDArray[np.uint8]) -> list[Detection]:
        return self.detect_batch([frame])[0]

    @staticmethod
    def prepared_frame_workload(predictions: Sequence[list[Detection]],
                                metadata: Sequence[CropMetadata]) -> dict[str, int]:
        stats = {"detector_images_total": len(metadata)}
        for name in ("full", "far", "middle", "near"):
            indices = [i for i, crop in enumerate(metadata)
                       if isinstance(crop, _PerspectiveCrop) and crop.region == name]
            stats[f"detector_images_{name}"] = len(indices)
            stats[f"detections_{name}_raw"] = sum(len(predictions[i]) for i in indices)
        return stats

    def detect_batch(
        self,
        frames: Sequence[NDArray[np.uint8]],
    ) -> list[list[Detection]]:
        """Detect an ordered batch, grouping perspective crops by resolution.

        Tile predictions are still merged independently per source frame, so a
        box can never suppress a box belonging to another frame.  Callers must
        apply tracking and analytics to the returned lists in source order.
        """
        source_frames = list(frames)
        if not source_frames:
            self._predict([])
            return []
        if self._config.perspective_regions.enabled:
            inputs, metadata, prepare_ms = self.prepare_source_batch(source_frames)
            predictions, stats = self.infer_prepared_batch(inputs)
            outputs, offset, total_merge_ms = [], 0, 0.0
            workload: dict[str, int] = {}
            for frame, crops in zip(source_frames, metadata, strict=True):
                frame_predictions = predictions[offset:offset + len(crops)]
                merged, merge_ms = self.merge_prepared_frame(frame, frame_predictions, crops)
                for key, value in self.prepared_frame_workload(frame_predictions, crops).items():
                    workload[key] = workload.get(key, 0) + value
                outputs.append(merged)
                total_merge_ms += merge_ms
                offset += len(crops)
            stats.update(workload)
            stats.update(merge_ms=total_merge_ms, tile_prepare_ms=prepare_ms,
                         perspective_preprocess_ms=prepare_ms,
                         perspective_detector_ms=stats.get("model_predict_ms", 0.0),
                         perspective_merge_ms=total_merge_ms,
                         source_batch_size=len(source_frames),
                         detections_after_merge=sum(map(len, outputs)),
                         postprocess_ms=float(stats.get("result_transfer_ms") or 0.0) + total_merge_ms)
            self._last_inference_stats = stats
            return outputs
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
        frame_metadata: list[list[CropMetadata]] = []
        for frame in source_frames:
            height, width = frame.shape[:2]
            metadata: list[CropMetadata] = []
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
        list[list[CropMetadata]],
        float,
    ]:
        """Prepare an ordered offline batch without invoking the model."""
        started = time.perf_counter()
        flat_inputs = _PreparedInputs()
        frame_metadata: list[list[CropMetadata]] = []
        for frame in frames:
            height, width = frame.shape[:2]
            metadata: list[tuple[int, int, int, int] | None] = []
            if self._config.perspective_regions.enabled:
                crops, metadata = self._prepare_perspective_frame(frame)
                for image, crop in zip(crops, metadata, strict=True):
                    flat_inputs.groups.setdefault(crop.group, []).append(len(flat_inputs))
                    flat_inputs.append(image)
            elif not self._config.tiled_inference:
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
        if (self._config.offline_cpu_preprocess and flat_inputs
                and not flat_inputs.groups and self._worker_mode == "no_sync_profile"):
            # The bounded producer queue owns this array. The GPU worker only
            # transfers it; original BGR crops remain available for box scaling.
            from ultralytics.data.augment import LetterBox

            target = self._config.input_shape or (self._config.imgsz, self._config.imgsz)
            target = tuple(int(np.ceil(side / 32)) * 32 for side in target)
            predictor = getattr(getattr(self, "_model", None), "predictor", None)
            model = getattr(predictor, "model", None)
            same_shapes = len({image.shape for image in flat_inputs}) == 1
            auto = (same_shapes and self._config.input_shape is None
                    and (getattr(model, "format", None) == "pt"
                         or getattr(model, "dynamic", False)))
            stride = getattr(model, "stride", 32)
            letterbox = LetterBox(target, auto=auto, stride=stride)
            flat_inputs.preprocessed = _pack_rgb_chw([letterbox(image=image) for image in flat_inputs])
        return flat_inputs, frame_metadata, (time.perf_counter() - started) * 1000

    def infer_prepared_batch(
        self,
        inputs: Sequence[NDArray[np.uint8]],
    ) -> tuple[list[list[Detection]], dict[str, Any]]:
        """Infer one prepared batch and snapshot GPU-worker-owned stats."""
        self._ensure_worker_state()
        groups = getattr(inputs, "groups", {})
        if groups:
            predictions: list[list[Detection]] = [[] for _ in inputs]
            stats: dict[str, Any] = {"model_invocations": 0, "inference_images": 0}
            totals = ("model_predict_ms", "ultralytics_predict_ms", "result_transfer_ms",
                      "yolo_preprocess_total_ms", "yolo_inference_total_ms", "yolo_postprocess_total_ms",
                      "event_h2d_total_ms", "event_tensor_conversion_total_ms",
                      "pre_transform_cpu_ms", "stack_cpu_ms", "layout_contiguous_cpu_ms",
                      "tensor_from_numpy_cpu_ms")
            for group, indices in groups.items():
                batches = self._predict([inputs[i] for i in indices], imgsz=group[0])
                for index, batch in zip(indices, batches, strict=True):
                    predictions[index] = batch
                measured = self.last_inference_stats
                for key in ("model_invocations", "inference_images", *totals):
                    value = measured.get(key)
                    if value is not None:
                        stats[key] = stats.get(key, 0) + value
            stats.update(batch_size=max(map(len, groups.values())),
                         actual_tensor_shapes=[list(image.shape) for image in inputs],
                         sum_tensor_pixels=sum(image.shape[0] * image.shape[1] for image in inputs),
                         worker_mode=self._worker_mode,
                         shape_source="source_images_before_ultralytics_letterbox",
                         workload_source="measured_input_shapes")
            self._last_inference_stats = stats
            return predictions, dict(stats)
        preprocessed = getattr(inputs, "preprocessed", None)
        predictions = (self._predict(list(inputs), preprocessed=preprocessed)
                       if preprocessed is not None else self._predict(list(inputs)))
        return predictions, self.last_inference_stats

    def merge_prepared_frame(
        self,
        frame: NDArray[np.uint8],
        prediction_batches: Sequence[list[Detection]],
        metadata: Sequence[CropMetadata],
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
        metadata: Sequence[CropMetadata],
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
            perspective = tile if isinstance(tile, _PerspectiveCrop) else None
            tile_x1, tile_y1, tile_x2, tile_y2 = perspective.bounds if perspective else tile
            tile_width = tile_x2 - tile_x1
            tile_height = tile_y2 - tile_y1
            for detection in tile_detections:
                if perspective is not None:
                    detection = Detection(
                        detection.x1 / perspective.scale_x, detection.y1 / perspective.scale_y,
                        detection.x2 / perspective.scale_x, detection.y2 / perspective.scale_y,
                        detection.confidence, detection.class_id,
                    )
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
        metadata: Sequence[CropMetadata],
    ) -> list[Detection]:
        merged, merge_ms = self._merge_prediction_results_timed(frame, prediction_batches, metadata)
        self._last_inference_stats["merge_ms"] = merge_ms
        self._last_inference_stats["postprocess_ms"] = float(
            self._last_inference_stats.get("postprocess_ms") or 0.0
        ) + float(self._last_inference_stats["merge_ms"])
        return merged
