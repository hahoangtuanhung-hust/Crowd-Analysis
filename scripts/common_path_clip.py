"""GPU video run or CPU analytics replay. Never infer on the local replay path."""

from __future__ import annotations

import argparse
import cProfile
import csv
import hashlib
import importlib.metadata
import io
import json
import math
import platform
import pstats
import statistics
import sys
import threading
import time
from dataclasses import asdict, replace
from pathlib import Path
from queue import Full, Queue
from types import SimpleNamespace
from typing import Any, Sequence

import cv2
import numpy as np
import psutil
import yaml

from backend.app.analytics.common_path import CommonPathAnalyzer
from backend.app.analytics.directional_grid import DirectionalGridEngine, GridTrackPoint
from backend.app.analytics.tracklet_aggregation import TrackletAggregationEngine, TrackletPoint
from backend.app.analytics.spatial import SpatialTransformer
from backend.app.core.config import load_config
from backend.app.schemas import CommonPathSnapshot, Detection, TrackedObject
from backend.app.video.renderer import FrameRenderer, OverlayOptions
from backend.app.video.motion_roi import MotionROIPlan, MotionROIPlanner


CACHE_SCHEMA = "detections-tracklets-v1"
SCHEDULER_SCHEMA = "motion-roi-scheduler-v1"
PROVENANCE_SCHEMA = "common-path-provenance-v2"
DIAGNOSTIC_REGIONS = ("upper", "middle", "lower")


def _detection_record(item: Detection) -> dict[str, object]:
    """Serialize the flat cache schema without dataclasses.asdict deepcopy."""
    return {
        "x1": item.x1,
        "y1": item.y1,
        "x2": item.x2,
        "y2": item.y2,
        "confidence": item.confidence,
        "class_id": item.class_id,
    }


def _track_record(item: TrackedObject) -> dict[str, object]:
    """Serialize the flat cache schema without recursively copying scalars."""
    return {
        "track_id": item.track_id,
        "x1": item.x1,
        "y1": item.y1,
        "x2": item.x2,
        "y2": item.y2,
        "confidence": item.confidence,
        "class_id": item.class_id,
        "observed": item.observed,
        "observation_coverage": item.observation_coverage,
        "direction_vector": item.direction_vector,
        "direction_state": item.direction_state,
        "direction_quality": item.direction_quality,
        "direction_observed_span_s": item.direction_observed_span_s,
    }


def resolve_source_batch_size(
    *,
    requested: int,
    max_detector_images: int,
    detector_images_per_frame: int,
    motion_enabled: bool,
    batch_supported: bool,
    replay_cache: bool,
) -> tuple[int, str | None]:
    """Resolve an offline source batch without changing observation semantics."""
    if replay_cache:
        return 1, "CACHE_REPLAY"
    maximum = max(1, max_detector_images // max(1, detector_images_per_frame))
    effective = min(requested, maximum)
    if motion_enabled and effective > 1:
        return 1, "MOTION_ROI_TRACK_FEEDBACK"
    if effective > 1 and not batch_supported:
        return 1, "DETECTOR_BATCH_UNSUPPORTED"
    if effective < requested:
        return effective, "MAX_DETECTOR_IMAGES_PER_BATCH"
    return effective, None


def count_inference_batches(timings: Sequence[dict[str, Any]]) -> int:
    """Count source batches independently of their model resolution groups."""
    batch_ids = {row["inference_batch_id"] for row in timings
                 if row.get("inference_batch_id") is not None}
    unbatched = sum(1 for row in timings if row.get("inference_batch_id") is None
                    and int(row.get("model_invocations") or 0) > 0)
    return len(batch_ids) + unbatched


def read_source_batch(
    capture: object,
    *,
    start_frame_id: int,
    end_frame_exclusive: int,
    source_batch_size: int,
    fps: float,
    last_timestamp: float,
) -> tuple[list[np.ndarray], list[dict[str, Any]], float]:
    """Read one bounded, ordered source batch and allow a partial EOF batch."""
    frames: list[np.ndarray] = []
    metadata: list[dict[str, Any]] = []
    batch_last_time = last_timestamp
    for frame_id in range(
        start_frame_id,
        min(end_frame_exclusive, start_frame_id + source_batch_size),
    ):
        decode_started = time.perf_counter()
        ok, frame = capture.read()
        if not ok:
            break
        decode_ms = (time.perf_counter() - decode_started) * 1000
        raw_pts = float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000
        timestamp = raw_pts if raw_pts > batch_last_time else frame_id / fps
        if timestamp < batch_last_time:
            raise ValueError("Media timestamps moved backwards")
        batch_last_time = timestamp
        frames.append(frame)
        metadata.append({
            "frame_id": frame_id,
            "timestamp": timestamp,
            "decode_ms": decode_ms,
            "decode_completed_at": time.perf_counter(),
        })
    return frames, metadata, batch_last_time


class OfflineOverlapPipeline:
    """Bounded ordered CPU preparation -> GPU inference for offline clips."""

    _END = object()

    def __init__(
        self,
        *,
        capture: object,
        detector: object,
        start_frame: int,
        end_frame_exclusive: int,
        source_batch_size: int,
        fps: float,
        last_timestamp: float,
        queue_batches: int,
        inference_interval: int = 1,
        profile_cpu: bool = False,
    ) -> None:
        self.capture = capture
        self.detector = detector
        self.start_frame = start_frame
        self.end_frame_exclusive = end_frame_exclusive
        self.source_batch_size = source_batch_size
        self.inference_interval = max(1, int(inference_interval))
        self.fps = fps
        self.last_timestamp = last_timestamp
        self.input_queue: Queue[object] = Queue(maxsize=queue_batches)
        self.result_queue: Queue[object] = Queue(maxsize=queue_batches)
        self.stop_event = threading.Event()
        self.input_queue_peak = 0
        self.result_queue_peak = 0
        self.producer_put_wait_ms = 0.0
        self.gpu_input_wait_ms = 0.0
        self.result_put_wait_ms = 0.0
        self.detection_scan_deadline_misses = 0
        self.profile_cpu = profile_cpu
        self.producer_profile: cProfile.Profile | None = None
        self.gpu_profile: cProfile.Profile | None = None
        self._producer = threading.Thread(
            target=self._produce, name="offline-frame-producer", daemon=True
        )
        self._gpu = threading.Thread(
            target=self._infer, name="offline-gpu-worker", daemon=True
        )

    def start(self) -> None:
        self._producer.start()
        self._gpu.start()

    def _bounded_put(self, target: Queue[object], item: object) -> float:
        started = time.perf_counter()
        while not self.stop_event.is_set():
            try:
                target.put(item, timeout=0.1)
                return (time.perf_counter() - started) * 1000
            except Full:
                continue
        raise RuntimeError("offline overlap pipeline stopped while queueing")

    def _produce(self) -> None:
        profiler = cProfile.Profile() if self.profile_cpu else None
        if profiler is not None:
            profiler.enable()
        next_frame = self.start_frame
        last_timestamp = self.last_timestamp
        last_detection_frame_id: int | None = None
        pending_frames: list[np.ndarray] = []
        pending_metadata: list[dict[str, Any]] = []
        detection_indices: list[int] = []
        try:
            while next_frame < self.end_frame_exclusive and not self.stop_event.is_set():
                frames, metadata, last_timestamp = read_source_batch(
                    self.capture,
                    start_frame_id=next_frame,
                    end_frame_exclusive=self.end_frame_exclusive,
                    source_batch_size=1,
                    fps=self.fps,
                    last_timestamp=last_timestamp,
                )
                if not frames:
                    break
                frame_id = int(metadata[0]["frame_id"])
                detection_due = (
                    last_detection_frame_id is None
                    or frame_id - last_detection_frame_id >= self.inference_interval
                )
                if detection_due:
                    if (
                        last_detection_frame_id is not None
                        and frame_id - last_detection_frame_id > self.inference_interval
                    ):
                        self.detection_scan_deadline_misses += 1
                    detection_indices.append(len(pending_frames))
                    last_detection_frame_id = frame_id
                pending_frames.extend(frames)
                pending_metadata.extend(metadata)
                next_frame += 1
                if len(detection_indices) >= self.source_batch_size:
                    wait_ms = self._bounded_put(
                        self.input_queue,
                        self._prepare_batch(pending_frames, pending_metadata, detection_indices),
                    )
                    self.producer_put_wait_ms += wait_ms
                    self.input_queue_peak = max(
                        self.input_queue_peak, self.input_queue.qsize()
                    )
                    pending_frames = []
                    pending_metadata = []
                    detection_indices = []
            if pending_frames:
                wait_ms = self._bounded_put(
                    self.input_queue,
                    self._prepare_batch(pending_frames, pending_metadata, detection_indices),
                )
                self.producer_put_wait_ms += wait_ms
                self.input_queue_peak = max(
                    self.input_queue_peak, self.input_queue.qsize()
                )
            self._bounded_put(self.input_queue, self._END)
        except BaseException as exc:
            if not self.stop_event.is_set():
                self._bounded_put(self.input_queue, ("error", exc))
        finally:
            if profiler is not None:
                profiler.disable()
                self.producer_profile = profiler

    def _prepare_batch(self, frames: list[np.ndarray], metadata: list[dict[str, Any]],
                       detection_indices: list[int]) -> dict[str, Any]:
        queued_at = time.perf_counter()
        inference_frames = [frames[index] for index in detection_indices]
        inputs, frame_metadata, prepare_ms = (
            self.detector.prepare_source_batch(inference_frames)
            if inference_frames else ([], [], 0.0)
        )
        return {"frames": frames, "metadata": metadata,
                "detection_indices": detection_indices, "queued_at": queued_at,
                "inputs": inputs, "frame_metadata": frame_metadata,
                "tile_prepare_ms": prepare_ms}

    def _infer(self) -> None:
        profiler = cProfile.Profile() if self.profile_cpu else None
        if profiler is not None:
            profiler.enable()
        try:
            while not self.stop_event.is_set():
                wait_started = time.perf_counter()
                item = self.input_queue.get()
                input_wait_ms = (time.perf_counter() - wait_started) * 1000
                self.gpu_input_wait_ms += input_wait_ms
                if item is self._END:
                    self._bounded_put(self.result_queue, self._END)
                    return
                if isinstance(item, tuple) and item and item[0] == "error":
                    self._bounded_put(self.result_queue, item)
                    return
                batch = dict(item)  # type: ignore[arg-type]
                frames = batch["frames"]
                detection_indices = [int(value) for value in batch["detection_indices"]]
                inference_frames = [frames[index] for index in detection_indices]
                inputs = batch.pop("inputs")
                frame_metadata = batch["frame_metadata"]
                tile_prepare_ms = batch["tile_prepare_ms"]
                if inference_frames:
                    infer = getattr(self.detector, "infer_prepared_batch")
                    inference_started = time.perf_counter()
                    predictions, detector_stats = infer(inputs)
                    inference_ms = (time.perf_counter() - inference_started) * 1000
                else:
                    frame_metadata = []
                    predictions = []
                    detector_stats = {}
                    tile_prepare_ms = 0.0
                    inference_ms = 0.0
                result = {
                    **batch,
                    "frame_metadata": frame_metadata,
                    "predictions": predictions,
                    "detector_stats": detector_stats,
                    "tile_prepare_ms": tile_prepare_ms,
                    "batch_inference_ms": inference_ms,
                    "gpu_input_wait_ms": input_wait_ms,
                    "result_ready_at": time.perf_counter(),
                }
                wait_ms = self._bounded_put(self.result_queue, result)
                self.result_put_wait_ms += wait_ms
                self.result_queue_peak = max(self.result_queue_peak, self.result_queue.qsize())
        except BaseException as exc:
            if not self.stop_event.is_set():
                self._bounded_put(self.result_queue, ("error", exc))
        finally:
            if profiler is not None:
                profiler.disable()
                self.gpu_profile = profiler

    def next_batch(self) -> dict[str, Any] | None:
        wait_started = time.perf_counter()
        item = self.result_queue.get()
        received_at = time.perf_counter()
        if item is self._END:
            return None
        if isinstance(item, tuple) and item and item[0] == "error":
            raise RuntimeError("offline overlap worker failed") from item[1]
        result = dict(item)  # type: ignore[arg-type]
        result["consumer_result_wait_ms"] = (received_at - wait_started) * 1000
        result["result_queue_age_ms"] = (
            received_at - float(result["result_ready_at"])
        ) * 1000
        return result

    def close(self) -> None:
        self.stop_event.set()
        self._producer.join(timeout=5)
        self._gpu.join(timeout=5)


def _write_cpu_profile(
    profiler: cProfile.Profile | None,
    output_dir: Path,
    name: str,
) -> None:
    """Persist a machine-readable profile and a compact cumulative-time view."""
    if profiler is None:
        return
    profiler.dump_stats(str(output_dir / f"cpu_profile_{name}.prof"))
    stream = io.StringIO()
    pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats(
        pstats.SortKey.CUMULATIVE
    ).print_stats(120)
    (output_dir / f"cpu_profile_{name}.txt").write_text(
        stream.getvalue(), encoding="utf-8"
    )


def materialize_overlap_batch(
    detector: object,
    batch: dict[str, Any],
    *,
    detector_images_per_frame: int,
    inference_batch_id: int,
) -> list[dict[str, Any]]:
    """Merge raw predictions in source order on the CPU consumer thread."""
    frames = batch["frames"]
    source_metadata = batch["metadata"]
    frame_metadata = batch["frame_metadata"]
    predictions = batch["predictions"]
    batch_stats = dict(batch["detector_stats"])
    detection_indices = [int(value) for value in batch.get(
        "detection_indices", range(len(frames))
    )]
    detection_count = len(detection_indices)
    detection_rank = {source_index: rank for rank, source_index in enumerate(detection_indices)}
    merge = getattr(detector, "merge_prepared_frame")
    amortized_fields = (
        "model_predict_ms",
        "ultralytics_predict_ms",
        "result_transfer_ms",
        "yolo_preprocess_total_ms",
        "yolo_inference_total_ms",
        "yolo_postprocess_total_ms",
        "event_h2d_total_ms",
        "event_tensor_conversion_total_ms",
        "pre_transform_cpu_ms",
        "stack_cpu_ms",
        "layout_contiguous_cpu_ms",
        "tensor_from_numpy_cpu_ms",
    )
    prepared: list[dict[str, Any]] = []
    offset = 0
    for index, (frame, metadata) in enumerate(zip(frames, source_metadata, strict=True)):
        rank = detection_rank.get(index)
        if rank is None:
            prepared.append({
                **metadata,
                "frame": frame,
                "detections": [],
                "inference_ms": 0.0,
                "detector_stats": {
                    "model_invocations": 0,
                    "inference_images": 0,
                    "batch_size": 0,
                    "source_batch_size": detection_count,
                    "inference_batch_id": None,
                },
                "detector_images": 0,
                "batch_index": index,
                "intentional_skip": True,
                "gpu_input_queue_wait_ms": None,
                "consumer_result_wait_ms": None,
                "result_ready_at": float(batch["result_ready_at"]),
                "pipeline_started_at": (
                    float(metadata["decode_completed_at"])
                    - float(metadata["decode_ms"]) / 1000.0
                ),
            })
            continue
        tile_metadata = frame_metadata[rank]
        count = len(tile_metadata)
        detections, merge_ms = merge(
            frame,
            predictions[offset:offset + count],
            tile_metadata,
        )
        offset += count
        per_frame_stats = dict(batch_stats)
        for field in amortized_fields:
            value = batch_stats.get(field)
            per_frame_stats[field] = (
                float(value) / detection_count if value is not None else None
            )
        per_frame_stats.update({
            "tile_prepare_ms": float(batch["tile_prepare_ms"]) / detection_count,
            "merge_ms": merge_ms,
            "postprocess_ms": (
                float(per_frame_stats.get("result_transfer_ms") or 0.0) + merge_ms
            ),
            "model_invocations": int(batch_stats.get("model_invocations", 1)) if rank == 0 else 0,
            "inference_images": detector_images_per_frame,
            "batch_size": int(batch_stats.get("batch_size") or 0),
            "source_batch_size": detection_count,
            "inference_batch_id": inference_batch_id,
        })
        workload = getattr(detector, "prepared_frame_workload", None)
        if callable(workload) and getattr(detector.config.perspective_regions, "enabled", False):
            per_frame_stats.update(workload(predictions[offset - count:offset], tile_metadata))
            per_frame_stats["detections_after_merge"] = len(detections)
        batch_shapes = batch_stats.get("actual_tensor_shapes")
        if isinstance(batch_shapes, list):
            shape_start = rank * detector_images_per_frame
            frame_shapes = batch_shapes[
                shape_start:shape_start + detector_images_per_frame
            ]
            per_frame_stats["actual_tensor_shapes"] = frame_shapes
            per_frame_stats["sum_tensor_pixels"] = sum(
                int(shape[0]) * int(shape[1])
                for shape in frame_shapes
                if isinstance(shape, (list, tuple)) and len(shape) >= 2
            )
        prepared.append({
            **metadata,
            "frame": frame,
            "detections": detections,
            "inference_ms": (
                float(batch["batch_inference_ms"]) / detection_count
                + float(batch["tile_prepare_ms"]) / detection_count
                + merge_ms
            ),
            "detector_stats": per_frame_stats,
            "detector_images": detector_images_per_frame,
            "batch_index": index,
            "intentional_skip": False,
            "gpu_input_queue_wait_ms": float(batch["gpu_input_wait_ms"]) / detection_count,
            "consumer_result_wait_ms": (
                float(batch["consumer_result_wait_ms"]) / detection_count
            ),
            "result_ready_at": float(batch["result_ready_at"]),
            "pipeline_started_at": (
                float(metadata["decode_completed_at"])
                - float(metadata["decode_ms"]) / 1000.0
            ),
        })
    if offset != len(predictions):
        raise RuntimeError("Prepared prediction count does not match frame metadata")
    return prepared


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def _canonical_digest(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _motion_roi_payload(config: object) -> dict[str, object]:
    motion = getattr(config, "motion_roi", None)
    if motion is None:
        return {"enabled": False, "shadow_mode": False}
    payload = motion.model_dump(mode="json")
    # Logging and overlay choices do not change detector/tracker observations,
    # so they must not invalidate an otherwise reusable upstream cache.
    payload.pop("diagnostics", None)
    return payload


def scheduler_fingerprint(config: object) -> str:
    return _canonical_digest({
        "schema": SCHEDULER_SCHEMA,
        "algorithm": "deadline-cadence-plus-reference-or-existing-tile-mog2-v2",
        "inference_interval": int(config.video.inference_interval),
        "motion_roi": _motion_roi_payload(config),
    })


def _code_fingerprint() -> str:
    root = Path(__file__).resolve().parents[1]
    relative_paths = (
        "scripts/common_path_clip.py",
        "backend/app/core/config.py",
        "backend/app/inference/ultralytics_detector.py",
        "backend/app/tracking/bytetrack_tracker.py",
        "backend/app/analytics/tracklet_aggregation.py",
        "backend/app/video/renderer.py",
        "backend/app/video/motion_roi.py",
    )
    result = hashlib.sha256()
    for relative in relative_paths:
        path = root / relative
        if not path.is_file():
            continue
        result.update(relative.encode("utf-8"))
        result.update(b"\0")
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                result.update(chunk)
    return result.hexdigest()


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {
        "python": platform.python_version(),
        "opencv": cv2.__version__,
        "numpy": np.__version__,
    }
    for package in ("ultralytics", "torch", "pydantic", "psutil"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _diagnostic_region_index(y: float, height: int) -> int:
    if height <= 0:
        return 0
    return min(2, max(0, int(3.0 * float(y) / float(height))))


def _rect_union_area(rectangles: Sequence[Sequence[float]], bounds: Sequence[float]) -> float:
    bx1, by1, bx2, by2 = map(float, bounds)
    clipped: list[tuple[float, float, float, float]] = []
    for rectangle in rectangles:
        x1 = max(bx1, float(rectangle[0]))
        y1 = max(by1, float(rectangle[1]))
        x2 = min(bx2, float(rectangle[2]))
        y2 = min(by2, float(rectangle[3]))
        if x2 > x1 and y2 > y1:
            clipped.append((x1, y1, x2, y2))
    if not clipped:
        return 0.0
    xs = sorted({x for item in clipped for x in (item[0], item[2])})
    area = 0.0
    for left, right in zip(xs, xs[1:]):
        intervals = sorted(
            (top, bottom)
            for x1, top, x2, bottom in clipped
            if x1 < right and x2 > left
        )
        covered = 0.0
        if intervals:
            start, end = intervals[0]
            for top, bottom in intervals[1:]:
                if top <= end:
                    end = max(end, bottom)
                else:
                    covered += end - start
                    start, end = top, bottom
            covered += end - start
        area += (right - left) * covered
    return area


def _box_fully_searched(box: Sequence[float], searched_regions: Sequence[Sequence[float]]) -> bool:
    x1, y1, x2, y2 = map(float, box)
    box_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if box_area <= 0.0:
        return False
    covered = _rect_union_area(searched_regions, (x1, y1, x2, y2))
    return covered >= box_area * (1.0 - 1e-6)


def _plan_to_dict(plan: MotionROIPlan | None, *, width: int, height: int,
                  motion_enabled: bool) -> dict[str, object]:
    return _as_plan_dict(
        plan,
        width=width,
        height=height,
        tiled=True,
        motion_enabled=motion_enabled,
    )


def _as_plan_dict(plan: object | None, *, width: int, height: int,
                  tiled: bool, motion_enabled: bool) -> dict[str, object]:
    """Normalize an optional scheduler plan without coupling the runner to it.

    The reference detector predates MotionROIPlanner.  Emitting an explicit
    reference decision for that path is preferable to silently leaving a hole
    in the diagnostic timeline, while fields that cannot be measured remain
    marked as such instead of being guessed.
    """
    if plan is not None:
        value = plan.to_dict() if callable(getattr(plan, "to_dict", None)) else dict(plan) \
            if isinstance(plan, dict) else {}
        if value:
            value.setdefault("searched_regions", [[0, 0, width, height]])
            value.setdefault("scan_type", "reference")
            value.setdefault("proposed_scan_type", value["scan_type"])
            value.setdefault("state", "FULL_COVERAGE")
            value.setdefault("reasons", [])
            value.setdefault("selected_tiles", [])
            value.setdefault("motion_tiles", [])
            value.setdefault("protected_tiles", [])
            value.setdefault("foreground_ratio", None)
            value.setdefault("roi_union_ratio", None)
            value.setdefault("estimated_cost_ratio", None)
            value.setdefault("motion_ms", None)
            value.setdefault("planning_ms", None)
            value.setdefault("shadow_mode", False)
            return value
    return {
        "state": "REFERENCE",
        "scan_type": "reference",
        "proposed_scan_type": "reference",
        "reasons": ["MOTION_ROI_DISABLED" if not motion_enabled else "PLANNER_NOT_EXPOSED"],
        "selected_tiles": [],
        "searched_regions": [[0, 0, width, height]],
        "motion_tiles": [],
        "protected_tiles": [],
        "foreground_ratio": None,
        "roi_union_ratio": 1.0,
        "estimated_cost_ratio": None,
        "learning_rate": None,
        "motion_ms": None,
        "planning_ms": None,
        "mask_shape": None,
        "shadow_mode": False,
    }


def _plan_regions(plan: dict[str, object], width: int, height: int) -> list[list[float]]:
    regions = plan.get("searched_regions")
    if not isinstance(regions, list):
        return [[0, 0, width, height]]
    # An empty list is an intentional no-search decision.  Do not turn a
    # scheduler skip into full-frame coverage in the diagnostics layer.
    if not regions:
        return []
    result: list[list[float]] = []
    for region in regions:
        if isinstance(region, (list, tuple)) and len(region) == 4:
            try:
                result.append([float(value) for value in region])
            except (TypeError, ValueError):
                continue
    return result or [[0, 0, width, height]]


def _detector_workload(detector: object | None, config: object, frame_shape: Sequence[int],
                       plan: dict[str, object]) -> dict[str, object]:
    """Read optional detector instrumentation, otherwise report unknowns.

    It is intentionally unsafe to infer tensor dimensions from source ROI
    area.  The artifact records that those fields were not instrumented yet so
    a benchmark cannot mistake an estimate for measured GPU work.
    """
    stats = getattr(detector, "last_inference_stats", None)
    if callable(stats):
        stats = stats()
    if not isinstance(stats, dict):
        stats = {}
    scan_type = str(plan.get("scan_type", "reference"))
    detector_config = config.detector
    configured_tile_count = (
        len(detector_config.tile_regions_normalized)
        or detector_config.tile_rows * detector_config.tile_columns
    )
    reference_images = (
        configured_tile_count
        + int(detector_config.tile_include_full_frame)
        if detector_config.tiled_inference else 1
    )
    fallback_images = reference_images if scan_type == "reference" else len(plan.get("selected_tiles", ()))
    return {
        "model_invocations": stats.get("model_invocations", 1 if detector is not None else 0),
        "inference_images": stats.get("inference_images", fallback_images if detector is not None else 0),
        "batch_size": stats.get("batch_size"),
        "actual_tensor_shapes": stats.get("actual_tensor_shapes", []),
        "sum_tensor_pixels": stats.get("sum_tensor_pixels"),
        "padding_overhead_pixels": stats.get("padding_overhead_pixels"),
        "workload_source": stats.get("workload_source", "instrumented" if stats else "not_instrumented"),
        "source_frame_shape": list(frame_shape),
    }


def _coverage_for_tracks(tracks: Sequence[object], plan: dict[str, object], width: int,
                         height: int) -> list[dict[str, object]]:
    searched = _plan_regions(plan, width, height)
    counts = {
        region: {"MEASURED": 0, "SEARCHED_NOT_FOUND": 0, "NOT_SEARCHED_BY_POLICY": 0}
        for region in DIAGNOSTIC_REGIONS
    }
    ages: dict[str, list[float]] = {region: [] for region in DIAGNOSTIC_REGIONS}
    for track in tracks:
        x = float(getattr(track, "bottom_center", (0.0, 0.0))[0])
        y = float(getattr(track, "bottom_center", (0.0, 0.0))[1])
        region = DIAGNOSTIC_REGIONS[_diagnostic_region_index(y, height)]
        box = (
            float(getattr(track, "x1", 0.0)), float(getattr(track, "y1", 0.0)),
            float(getattr(track, "x2", 0.0)), float(getattr(track, "y2", 0.0)),
        )
        declared_state = getattr(track, "observation_coverage", None)
        if declared_state in counts[region]:
            state = declared_state
        elif bool(getattr(track, "observed", True)):
            state = "MEASURED"
        elif _box_fully_searched(box, searched):
            state = "SEARCHED_NOT_FOUND"
        else:
            state = "NOT_SEARCHED_BY_POLICY"
        counts[region][state] += 1
        # Track objects currently do not expose a last-observation age. Keep
        # the column explicit and avoid manufacturing a value from wall time.
        if hasattr(track, "observation_age_s"):
            try:
                ages[region].append(float(track.observation_age_s))
            except (TypeError, ValueError):
                pass
    rows: list[dict[str, object]] = []
    for region in DIAGNOSTIC_REGIONS:
        row = {"region_id": region, **counts[region]}
        values = ages[region]
        row.update({
            "track_count": sum(counts[region].values()),
            "observation_age_p50_s": round(float(np.percentile(values, 50)), 4) if values else None,
            "observation_age_p95_s": round(float(np.percentile(values, 95)), 4) if values else None,
            "observation_age_max_s": round(max(values), 4) if values else None,
            "age_status": "measured" if values else "not_available",
        })
        rows.append(row)
    return rows


def _label_track_coverage(tracks: Sequence[TrackedObject], plan: dict[str, object],
                          width: int, height: int) -> list[TrackedObject]:
    searched = _plan_regions(plan, width, height)
    scan_type = str(plan.get("scan_type", "reference"))
    labeled: list[TrackedObject] = []
    for track in tracks:
        if track.observed:
            state = "MEASURED"
        elif scan_type == "skip":
            state = "NOT_SEARCHED_BY_POLICY"
        elif _box_fully_searched(track.xyxy, searched):
            state = "SEARCHED_NOT_FOUND"
        else:
            state = "NOT_SEARCHED_BY_POLICY"
        labeled.append(replace(track, observation_coverage=state))
    return labeled


def _write_jsonl(path: Path, rows: Sequence[object]) -> None:
    with path.open("w", encoding="utf-8") as sink:
        for row in rows:
            sink.write(json.dumps(row, separators=(",", ":"), ensure_ascii=True) + "\n")


def _write_csv_rows(path: Path, rows: Sequence[dict[str, object]], fieldnames: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as sink:
        writer = csv.DictWriter(sink, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _percentile(values: Sequence[object], percentile: float) -> float | None:
    numeric = [float(value) for value in values if value is not None]
    if not numeric:
        return None
    return round(float(np.percentile(numeric, percentile)), 3)


def _write_stage_reports(output_dir: Path, timings: Sequence[dict[str, object]]) -> None:
    """Write stage percentiles and bounded video-time windows from one run."""
    stages = (
        ("decode", "decode_ms"),
        ("tile_prepare", "tile_prepare_ms"),
        ("yolo_preprocess_including_h2d", "yolo_preprocess_total_ms"),
        ("host_to_device", "h2d_ms"),
        ("tensor_conversion_normalize", "tensor_conversion_ms"),
        ("yolo_model_inference", "yolo_inference_total_ms"),
        ("yolo_postprocess", "yolo_postprocess_total_ms"),
        ("ultralytics_predict_wrapper", "model_predict_ms"),
        ("result_transfer", "result_transfer_ms"),
        ("project_postprocess_including_merge", "postprocess_ms"),
        ("tile_merge", "merge_ms"),
        ("inference_total", "inference_ms"),
        ("tracking", "tracking_ms"),
        ("common_path", "analytics_ms"),
        ("common_path_compute", "common_path_compute_ms"),
        ("render", "render_ms"),
        ("encode", "encode_ms"),
        ("cache_write", "cache_write_ms"),
        ("gpu_input_queue_wait", "gpu_input_queue_wait_ms"),
        ("consumer_result_wait", "consumer_result_wait_ms"),
        ("result_queue_age", "result_queue_age_ms"),
        ("end_to_end_latency", "e2e_latency_ms"),
        ("pipeline_overhead_or_idle", "pipeline_overhead_ms"),
        ("frame_total", "frame_total_ms"),
    )
    stage_rows = []
    for name, field in stages:
        values = [row.get(field) for row in timings]
        numeric = [float(value) for value in values if value is not None]
        if not numeric:
            continue
        stage_rows.append({
            "stage": name,
            "field": field,
            "samples": len(numeric),
            "mean_ms": round(float(np.mean(numeric)), 3),
            "p50_ms": _percentile(numeric, 50),
            "p95_ms": _percentile(numeric, 95),
            "max_ms": round(max(numeric), 3),
            "total_ms": round(sum(numeric), 3),
        })
    _write_csv_rows(
        output_dir / "stage_metrics.csv", stage_rows,
        ("stage", "field", "samples", "mean_ms", "p50_ms", "p95_ms", "max_ms", "total_ms"),
    )

    windows: dict[int, list[dict[str, object]]] = {}
    for row in timings:
        window = int(float(row.get("event_time_s", 0.0)) // 60.0)
        windows.setdefault(window, []).append(row)
    window_rows = []
    previous_window_end = 0.0
    for window, rows in sorted(windows.items()):
        timestamps = [float(row["event_time_s"]) for row in rows]
        span = max(timestamps[-1] - timestamps[0], 1e-9)
        completed = rows[-1].get("pipeline_elapsed_seconds")
        if completed is not None:
            processing_seconds = max(float(completed) - previous_window_end, 1e-9)
            previous_window_end = float(completed)
        else:
            frame_times = [float(row["frame_total_ms"]) for row in rows
                           if row.get("frame_total_ms") is not None]
            processing_seconds = sum(frame_times) / 1000.0 if frame_times else None
        def mean(field: str) -> float | None:
            values = [float(row[field]) for row in rows if row.get(field) is not None]
            return round(float(np.mean(values)), 3) if values else None
        def peak(field: str) -> float | None:
            values = [float(row[field]) for row in rows if row.get(field) is not None]
            return round(max(values), 3) if values else None
        window_rows.append({
            "window_start_s": round(window * 60.0, 3),
            "window_end_s": round(window * 60.0 + 60.0, 3),
            "frames": len(rows),
            "media_span_s": round(span, 3),
            "processing_seconds": round(processing_seconds, 6) if processing_seconds else None,
            "processing_fps": round(len(rows) / processing_seconds, 3) if processing_seconds else None,
            "inference_mean_ms": mean("inference_ms"),
            "tracking_mean_ms": mean("tracking_ms"),
            "common_path_mean_ms": mean("analytics_ms"),
            "render_mean_ms": mean("render_ms"),
            "encode_mean_ms": mean("encode_ms"),
            "frame_total_mean_ms": mean("frame_total_ms"),
            "detections_mean": mean("detections"),
            "tracks_mean": mean("tracks"),
            "tracklet_buffered_peak": peak("tracklet_buffered_segments"),
            "tracklet_candidates_peak": peak("tracklet_candidates"),
            "ram_peak_mb": peak("ram_mb"),
        })
    _write_csv_rows(
        output_dir / "long_run_metrics.csv", window_rows,
        ("window_start_s", "window_end_s", "frames", "media_span_s", "processing_seconds", "processing_fps",
         "inference_mean_ms", "tracking_mean_ms", "common_path_mean_ms", "render_mean_ms",
         "encode_mean_ms", "frame_total_mean_ms", "detections_mean", "tracks_mean",
         "tracklet_buffered_peak", "tracklet_candidates_peak", "ram_peak_mb"),
    )


def cache_key(input_hash: str, model_hash: str, config: object, start: float,
              duration: float | None) -> str:
    data = dict(schema=CACHE_SCHEMA, input_hash=input_hash,
                model_hash=model_hash, start=start, duration=duration,
                detector=config.detector.model_dump(exclude={"device", "model"}
                    | ({"input_shape"} if config.detector.input_shape is None else set())
                    | {"offline_cpu_preprocess", "cuda_graph_inference"}),
                tracker=config.tracker.model_dump(),
                offline_batch={
                    "source_batch_size": config.video.source_batch_size,
                    "inference_interval": config.video.inference_interval,
                    "max_detector_images_per_batch": (
                        config.video.max_detector_images_per_batch
                    ),
                })
    motion = getattr(config, "motion_roi", None)
    if motion is not None and (motion.enabled or motion.shadow_mode):
        # A hybrid run observes a different subset of the source than a
        # reference run.  Never allow a full-coverage cache to masquerade as a
        # scheduler cache.  Disabled production configs retain the historical
        # key shape for backwards-compatible replay.
        data["scheduler_fingerprint"] = scheduler_fingerprint(config)
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def run(input_path: Path, output_dir: Path, *, config_path: Path, model_path: Path | None,
        start_seconds: float, duration_seconds: float | None, run_id: str, engine: str,
        replay_cache: Path | None = None, input_hash: str | None = None,
        model_hash: str | None = None, rebuild_tracks: bool = False,
        allow_cache_key_mismatch: bool = False,
        profile_gpu: bool = False,
        profile_cpu: bool = False,
        detector_worker_mode: str | None = None,
        inference_backend: str = "pytorch_fp32") -> dict:
    """Run video inference or analytics replay.

    Args:
        profile_gpu: When True, calls torch.cuda.synchronize() before/after each
            inference batch so GPU timing is isolated.  This adds ~50-80 ms per
            frame of blocking overhead; never use in production benchmarks.
            Default False preserves GPU pipeline overlap for maximum throughput.
        profile_cpu: When True, writes separate cProfile captures for producer,
            GPU worker, and consumer threads. Use only for diagnostic runs.
        detector_worker_mode: One isolated detector-worker experiment. The
            default preserves the standard Ultralytics path.
    """
    if start_seconds < 0:
        raise ValueError("Start must be >=0 seconds")
    if duration_seconds is not None and duration_seconds <= 0:
        raise ValueError("Duration must be >0 seconds when provided")
    if engine not in ("legacy", "directional_grid", "tracklet_aggregation", "shadow"):
        raise ValueError("Invalid engine")
    if inference_backend not in ("pytorch_fp32", "pytorch_fp16", "tensorrt_fp16"):
        raise ValueError("Invalid inference backend")
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite run: {output_dir}")
    output_dir.mkdir(parents=True)
    config = load_config(config_path)
    detector_worker_mode = (
        detector_worker_mode or config.video.detector_worker_mode
    )
    transformer: SpatialTransformer
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot decode input: {input_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0 or width <= 0 or height <= 0:
        raise ValueError("Source FPS/resolution unavailable")
    transformer = SpatialTransformer.pixel(width, height)
    selected = config.analytics.common_path.shadow_display if engine == "shadow" else engine
    grid = (
        DirectionalGridEngine(config.analytics.directional_grid, transformer,
                              stream_epoch=run_id, zones=config.analytics.zones)
        if engine in ("directional_grid", "shadow") else None
    )
    tracklet = (
        TrackletAggregationEngine(
            config.analytics.common_path.tracklet_aggregation,
            transformer,
            camera_id="cam01",
            stream_epoch=run_id,
            max_paths=config.analytics.common_path.max_paths,
        )
        if engine == "tracklet_aggregation" else None
    )
    legacy = CommonPathAnalyzer(config.analytics, transformer) if engine in ("legacy", "shadow") else None
    renderer = FrameRenderer(config.visualization)
    overlay = OverlayOptions.from_visualization(config.visualization)
    input_hash = input_hash or digest(input_path)
    model_hash = model_hash or (digest(model_path) if model_path else "unknown")
    key = cache_key(input_hash, model_hash, config, start_seconds, duration_seconds)
    config_hash = _canonical_digest(config.model_dump(mode="json"))
    scheduler_hash = scheduler_fingerprint(config)
    code_hash = _code_fingerprint()
    motion_config = getattr(config, "motion_roi", None)
    motion_enabled = bool(
        motion_config is not None and (motion_config.enabled or motion_config.shadow_mode)
    )
    motion_scheduler = MotionROIPlanner(motion_config) if motion_enabled else None
    cache_path = output_dir / "tracking_cache.jsonl" if replay_cache is None else replay_cache
    if rebuild_tracks and replay_cache is None:
        raise ValueError("--rebuild-tracks requires --replay-cache")
    device = "none (CPU analytics replay)"
    cuda_runtime = None
    detector = tracker = None
    detector_init_ms = 0.0
    detector_warmup_ms = 0.0
    requested_source_batch_size = int(config.video.source_batch_size)
    effective_source_batch_size = 1
    source_batch_fallback_reason: str | None = None
    detector_images_per_reference = 1
    if replay_cache is None:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for video inference; no CPU fallback")
        torch.cuda.reset_peak_memory_stats()
        cuda_runtime = torch
        device = torch.cuda.get_device_name(0)
        from backend.app.inference import UltralyticsPersonDetector
        from backend.app.inference.ultralytics_detector import DETECTOR_WORKER_MODES
        from backend.app.tracking import ByteTrackTracker
        if detector_worker_mode not in DETECTOR_WORKER_MODES:
            raise ValueError(f"Unsupported detector worker mode: {detector_worker_mode}")
        detection_config = config.detector.model_copy(
            update={"device": "cuda:0", "model": str(model_path or config.detector.model)}
        )
        detector_init_started = time.perf_counter()
        detector = UltralyticsPersonDetector(
            detection_config,
            worker_mode=detector_worker_mode,
        )
        detector_init_ms = (time.perf_counter() - detector_init_started) * 1000.0
        tracker = ByteTrackTracker(config.tracker)
        image_count_provider = getattr(detector, "reference_image_count", None)
        detector_images_per_reference = (
            int(image_count_provider(width, height))
            if callable(image_count_provider) else 1
        )
        effective_source_batch_size, source_batch_fallback_reason = (
            resolve_source_batch_size(
                requested=requested_source_batch_size,
                max_detector_images=int(config.video.max_detector_images_per_batch),
                detector_images_per_frame=detector_images_per_reference,
                motion_enabled=motion_enabled,
                batch_supported=callable(getattr(detector, "detect_batch", None)),
                replay_cache=False,
            )
        )
        # Warm the exact detector instance and batch shape used by the timed
        # loop. A batch-1 warmup does not allocate the batch-4 working set.
        warmup = getattr(detector, "warmup", None)
        if callable(warmup):
            warmup_started = time.perf_counter()
            warmup(
                frame_width=width,
                frame_height=height,
                warmup_passes=2,
                source_batch_size=effective_source_batch_size,
            )
            detector_warmup_ms = (time.perf_counter() - warmup_started) * 1000.0
    else:
        effective_source_batch_size, source_batch_fallback_reason = (
            resolve_source_batch_size(
                requested=requested_source_batch_size,
                max_detector_images=int(config.video.max_detector_images_per_batch),
                detector_images_per_frame=detector_images_per_reference,
                motion_enabled=motion_enabled,
                batch_supported=False,
                replay_cache=True,
            )
        )
    cache_key_match: bool | None = None
    if replay_cache is not None:
        meta_path = replay_cache.with_suffix(".meta.json")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        cache_key_match = meta.get("cache_key") == key
        if not cache_key_match:
            if not allow_cache_key_mismatch:
                raise ValueError("Tracking cache key mismatch; inference/tracker/input/clip changed")
            if meta.get("source_hash") != input_hash or meta.get("model_hash") != model_hash:
                raise ValueError("Cache key override requires matching source and model hashes")
        if rebuild_tracks:
            from backend.app.tracking import ByteTrackTracker
            tracker = ByteTrackTracker(config.tracker)
    writer = cv2.VideoWriter(str(output_dir / "tracked_points_common_path.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError("Cannot open MP4 writer")
    capture_frames: list[np.ndarray] = []
    timings: list[dict] = []
    ram_samples_mb: list[float] = []
    frame_ids: list[int] = []
    count_tracks: set[int] = set()
    latest_snapshot = CommonPathSnapshot(0.0, ())
    legacy_snapshot = CommonPathSnapshot(0.0, ())
    first_frame: np.ndarray | None = None
    last_frame: np.ndarray | None = None
    last_tracks: tuple[TrackedObject, ...] = ()
    last_time = start_seconds
    start_frame = round(start_seconds * fps)
    source_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 10**9
    end_frame = source_frame_count if duration_seconds is None else min(
        round((start_seconds + duration_seconds) * fps), source_frame_count)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    cache_reader = (
        replay_cache.open("r", encoding="utf-8", buffering=1024 * 1024)
        if replay_cache else None
    )
    cache_writer = (
        cache_path.open("w", encoding="utf-8", buffering=1024 * 1024)
        if replay_cache is None else None
    )
    rebuilt_cache_path = output_dir / "tracking_cache_rebuilt.jsonl" if rebuild_tracks else None
    rebuilt_cache_writer = (rebuilt_cache_path.open("w", encoding="utf-8")
                            if rebuilt_cache_path else None)
    point_type = TrackletPoint if selected == "tracklet_aggregation" else GridTrackPoint
    scheduler_decisions: list[dict[str, object]] = []
    coverage_rows: list[dict[str, object]] = []
    region_funnel_rows: list[dict[str, object]] = []
    candidate_decisions: list[dict[str, object]] = []
    path_support_rows: list[dict[str, object]] = []
    seen_candidate_diagnostics: set[str] = set()
    seen_path_support_diagnostics: set[str] = set()
    diagnostic_compute_count = -1
    candidate_trace: Sequence = ()
    support_trace: Sequence = ()
    seen_track_ids: set[int] = set()
    diagnostic_limit = int(getattr(getattr(motion_config, "diagnostics", None), "max_decisions", 10000))
    diagnostic_stride = max(1, math.ceil(max(1, end_frame - start_frame) / diagnostic_limit))
    overlap_pipeline: OfflineOverlapPipeline | None = None
    overlap_enabled = bool(config.video.offline_overlap_enabled)
    if overlap_enabled:
        if replay_cache is not None:
            raise ValueError("offline overlap is unavailable during cache replay")
        if motion_enabled:
            raise ValueError("offline overlap cannot run with tracker-feedback motion ROI")
        required = ("prepare_source_batch", "infer_prepared_batch", "merge_prepared_frame")
        if detector is None or any(not callable(getattr(detector, name, None)) for name in required):
            raise ValueError("detector does not support offline overlap")
        overlap_pipeline = OfflineOverlapPipeline(
            capture=cap,
            detector=detector,
            start_frame=start_frame,
            end_frame_exclusive=end_frame,
            source_batch_size=effective_source_batch_size,
            fps=fps,
            last_timestamp=last_time,
            queue_batches=int(config.video.offline_overlap_queue_batches),
            inference_interval=int(config.video.inference_interval),
            profile_cpu=profile_cpu,
        )
    began = time.perf_counter()
    consumer_profile = cProfile.Profile() if profile_cpu else None
    prepared_frames: list[dict[str, Any]] = []
    inference_batch_sequence = 0
    if overlap_pipeline is not None:
        overlap_pipeline.start()
    if consumer_profile is not None:
        consumer_profile.enable()
    try:
        for frame_id in range(start_frame, end_frame):
            t0 = time.perf_counter()
            prepared: dict[str, Any] | None = None
            if overlap_pipeline is not None:
                if not prepared_frames:
                    overlap_batch = overlap_pipeline.next_batch()
                    if overlap_batch is None:
                        break
                    inference_batch_sequence += 1
                    prepared_frames.extend(materialize_overlap_batch(
                        detector,
                        overlap_batch,
                        detector_images_per_frame=detector_images_per_reference,
                        inference_batch_id=inference_batch_sequence,
                    ))
                prepared = prepared_frames.pop(0)
                if int(prepared["frame_id"]) != frame_id:
                    raise RuntimeError("Overlap inference frame order mismatch")
                prepared["result_queue_age_ms"] = (
                    time.perf_counter() - float(prepared["result_ready_at"])
                ) * 1000
                frame = prepared["frame"]
                decode_ms = float(prepared["decode_ms"])
                timestamp = float(prepared["timestamp"])
                last_time = timestamp
            elif effective_source_batch_size > 1 and cache_reader is None:
                if not prepared_frames:
                    batch_frames, batch_metadata, _ = read_source_batch(
                        cap,
                        start_frame_id=frame_id,
                        end_frame_exclusive=end_frame,
                        source_batch_size=effective_source_batch_size,
                        fps=fps,
                        last_timestamp=last_time,
                    )
                    if not batch_frames:
                        break
                    import torch
                    batch_inference_started = time.perf_counter()
                    if profile_gpu:
                        torch.cuda.synchronize()
                    batch_detections = detector.detect_batch(batch_frames)
                    if profile_gpu:
                        torch.cuda.synchronize()
                    batch_inference_ms = (
                        time.perf_counter() - batch_inference_started
                    ) * 1000
                    batch_stats = dict(
                        getattr(detector, "last_inference_stats", {}) or {}
                    )
                    source_count = len(batch_frames)
                    inference_batch_sequence += 1
                    amortized_fields = (
                        "model_predict_ms",
                        "ultralytics_predict_ms",
                        "result_transfer_ms",
                        "postprocess_ms",
                        "merge_ms",
                    )
                    for index, (batch_frame, metadata, detections_for_frame) in enumerate(
                        zip(batch_frames, batch_metadata, batch_detections, strict=True)
                    ):
                        per_frame_stats = dict(batch_stats)
                        for field in amortized_fields:
                            value = batch_stats.get(field)
                            per_frame_stats[field] = (
                                float(value) / source_count if value is not None else None
                            )
                        per_frame_stats.update({
                            "model_invocations": int(batch_stats.get("model_invocations", 1)) if index == 0 else 0,
                            "inference_images": detector_images_per_reference,
                            "batch_size": int(batch_stats.get("batch_size") or 0),
                            "source_batch_size": source_count,
                            "inference_batch_id": inference_batch_sequence,
                        })
                        batch_shapes = batch_stats.get("actual_tensor_shapes")
                        if isinstance(batch_shapes, list):
                            shape_start = index * detector_images_per_reference
                            frame_shapes = batch_shapes[
                                shape_start:shape_start + detector_images_per_reference
                            ]
                            per_frame_stats["actual_tensor_shapes"] = frame_shapes
                            per_frame_stats["sum_tensor_pixels"] = sum(
                                int(shape[0]) * int(shape[1])
                                for shape in frame_shapes
                                if isinstance(shape, (list, tuple)) and len(shape) >= 2
                            )
                        prepared_frames.append({
                            **metadata,
                            "frame": batch_frame,
                            "detections": detections_for_frame,
                            "inference_ms": batch_inference_ms / source_count,
                            "detector_stats": per_frame_stats,
                            "detector_images": detector_images_per_reference,
                            "batch_index": index,
                        })
                prepared = prepared_frames.pop(0)
                if int(prepared["frame_id"]) != frame_id:
                    raise RuntimeError("Prepared inference frame order mismatch")
                frame = prepared["frame"]
                decode_ms = float(prepared["decode_ms"])
                timestamp = float(prepared["timestamp"])
                last_time = timestamp
            else:
                ok, frame = cap.read()
                if not ok:
                    break
                decode_ms = (time.perf_counter() - t0) * 1000
                raw_pts = float(cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000
                timestamp = raw_pts if raw_pts > last_time else frame_id / fps
                if timestamp < last_time:
                    raise ValueError("Media timestamps moved backwards")
                last_time = timestamp
            if first_frame is None:
                first_frame = frame.copy()
            entry: dict[str, Any] | None = None
            executed_plan: object | None = None
            detector_images = 0
            detector_stats: dict[str, object] = {}
            if cache_reader:
                entry = json.loads(cache_reader.readline())
                if entry["frame_id"] != frame_id:
                    raise ValueError("Replay cache frame mismatch")
                timestamp = entry["event_time_s"]
                detections = [Detection(**d) for d in entry["detections"]]
                executed_plan = entry.get("scheduler_decision")
                inference_ms = 0.
                if rebuild_tracks:
                    t1 = time.perf_counter()
                    cached_scan_type = (
                        str(executed_plan.get("scan_type", "reference"))
                        if isinstance(executed_plan, dict) else "reference"
                    )
                    if cached_scan_type == "skip":
                        tracks = tracker.coast(
                            frame,
                            frame_id=frame_id,
                            source_timestamp=timestamp,
                        )
                    else:
                        tracks = tracker.update(
                            detections,
                            frame,
                            frame_id=frame_id,
                            source_timestamp=timestamp,
                        )
                    tracking_ms = (time.perf_counter() - t1) * 1000
                else:
                    tracks = [TrackedObject(**t) for t in entry["tracks"]]
                    tracking_ms = 0.
            elif prepared is not None:
                detections = list(prepared["detections"])
                detector_stats = dict(prepared["detector_stats"])
                inference_ms = float(prepared["inference_ms"])
                detector_images = int(prepared["detector_images"])
                t1 = time.perf_counter()
                if bool(prepared.get("intentional_skip")):
                    scan_type = "skip"
                    executed_plan = {
                        "state": "CADENCE_SKIP",
                        "scan_type": "skip",
                        "proposed_scan_type": "skip",
                        "reasons": ["cadence_skip"],
                        "selected_tiles": [],
                        "searched_regions": [],
                    }
                    tracks = tracker.coast(
                        frame,
                        frame_id=frame_id,
                        source_timestamp=timestamp,
                    )
                else:
                    scan_type = "reference"
                    tracks = tracker.update(
                        detections,
                        frame,
                        frame_id=frame_id,
                        source_timestamp=timestamp,
                    )
                tracking_ms = (time.perf_counter() - t1) * 1000
            else:
                import torch
                if motion_scheduler is not None:
                    tile_provider = getattr(detector, "tile_regions", None)
                    tile_regions = (
                        tuple(tile_provider(width, height)) if callable(tile_provider) else ()
                    )
                    image_count_provider = getattr(detector, "reference_image_count", None)
                    reference_image_count = (
                        int(image_count_provider(width, height))
                        if callable(image_count_provider) else 1
                    )
                    executed_plan = motion_scheduler.plan(
                        frame,
                        timestamp,
                        tile_regions=tile_regions,
                        reference_image_count=reference_image_count,
                        protected_boxes=tuple(
                            (int(track.x1), int(track.y1), int(track.x2), int(track.y2))
                            for track in last_tracks
                        ),
                    )
                    if (
                        executed_plan.scan_type == "tiles"
                        and any(
                            not _box_fully_searched(track.xyxy, executed_plan.searched_regions)
                            for track in last_tracks
                        )
                    ):
                        # BYTETracker.update treats an unmatched track as a
                        # searched miss.  A selective pass is only safe when
                        # every existing predicted bbox is fully covered by the
                        # union of selected tiles; partial overlap is not enough.
                        executed_plan = replace(
                            executed_plan,
                            scan_type="reference",
                            searched_regions=((0, 0, width, height),),
                            reasons=executed_plan.reasons
                            + ("TRACK_COVERAGE_REFERENCE_FALLBACK",),
                        )
                scan_type = (
                    str(getattr(executed_plan, "scan_type", "reference"))
                    if executed_plan is not None else "reference"
                )
                # profile_gpu=True inserts synchronize() for accurate per-stage GPU timing.
                # Production runs keep profile_gpu=False (default) to preserve async GPU
                # pipeline overlap.  A synchronize() per frame adds ~50-80ms blocking
                # overhead on T4 and is never acceptable in a throughput benchmark.
                t1 = time.perf_counter()
                if scan_type == "reference":
                    if profile_gpu:
                        torch.cuda.synchronize()
                    detections = detector.detect(frame)
                    if profile_gpu:
                        torch.cuda.synchronize()
                    detector_images = (
                        int(detector.reference_image_count(width, height))
                        if callable(getattr(detector, "reference_image_count", None)) else 1
                    )
                elif scan_type == "tiles":
                    region_detector = getattr(detector, "detect_regions", None)
                    if not callable(region_detector):
                        if profile_gpu:
                            torch.cuda.synchronize()
                        detections = detector.detect(frame)
                        if profile_gpu:
                            torch.cuda.synchronize()
                        detector_images = (
                            int(detector.reference_image_count(width, height))
                            if callable(getattr(detector, "reference_image_count", None)) else 1
                        )
                        executed_plan = replace(
                            executed_plan,
                            scan_type="reference",
                            searched_regions=((0, 0, width, height),),
                            reasons=executed_plan.reasons + ("BACKEND_REFERENCE_FALLBACK",),
                        )
                        scan_type = "reference"
                    else:
                        if profile_gpu:
                            torch.cuda.synchronize()
                        detections = region_detector(frame, executed_plan.selected_tiles)
                        if profile_gpu:
                            torch.cuda.synchronize()
                        detector_images = len(executed_plan.selected_tiles)
                else:
                    detections = []
                detector_stats = dict(getattr(detector, "last_inference_stats", {}) or {})
                inference_ms = (time.perf_counter() - t1) * 1000
                t1 = time.perf_counter()
                if scan_type == "skip":
                    coast = getattr(tracker, "coast", None)
                    if not callable(coast):
                        raise RuntimeError(
                            "motion ROI skip requires tracker.coast; refusing update([]) semantics"
                        )
                    tracks = coast(
                        frame,
                        frame_id=frame_id,
                        source_timestamp=timestamp,
                    )
                else:
                    tracks = tracker.update(
                        detections,
                        frame,
                        frame_id=frame_id,
                        source_timestamp=timestamp,
                    )
                tracking_ms = (time.perf_counter() - t1) * 1000

            exposed_plan = executed_plan
            if exposed_plan is None and entry is not None:
                exposed_plan = entry.get("scheduler_decision")
            if exposed_plan is None and detector is not None:
                exposed_plan = getattr(detector, "last_scheduler_decision", None)
            plan = _as_plan_dict(
                exposed_plan,
                width=width,
                height=height,
                tiled=config.detector.tiled_inference,
                motion_enabled=motion_enabled,
            )
            tracks = _label_track_coverage(tracks, plan, width, height)
            workload = _detector_workload(detector, config, frame.shape, plan)
            if prepared is not None:
                workload.update({
                    "model_invocations": int(detector_stats.get("model_invocations") or 0),
                    "inference_images": detector_images,
                    "batch_size": int(detector_stats.get("batch_size") or 0),
                    "source_batch_size": int(detector_stats.get("source_batch_size") or 1),
                    "inference_batch_id": detector_stats.get("inference_batch_id"),
                    "actual_tensor_shapes": detector_stats.get("actual_tensor_shapes", []),
                    "sum_tensor_pixels": detector_stats.get("sum_tensor_pixels"),
                })
            elif replay_cache is None:
                skipped = str(plan.get("scan_type")) == "skip"
                workload["model_invocations"] = 0 if skipped else int(
                    workload.get("model_invocations") or 1
                )
                workload["inference_images"] = 0 if skipped else detector_images
                if skipped:
                    workload["batch_size"] = 0
                    workload["actual_tensor_shapes"] = []
                    workload["sum_tensor_pixels"] = 0
                    workload["padding_overhead_pixels"] = 0
                    workload["workload_source"] = "intentional_skip"
            plan_record = {
                "schema": SCHEDULER_SCHEMA,
                "frame_id": frame_id,
                "event_time_s": round(timestamp, 6),
                "scheduler_fingerprint": scheduler_hash,
                **plan,
                **workload,
            }
            coverage = _coverage_for_tracks(tracks, plan, width, height)
            coverage_by_track: list[dict[str, object]] = []
            searched_regions = _plan_regions(plan, width, height)
            for track in tracks:
                if track.observed:
                    coverage_state = "MEASURED"
                elif _box_fully_searched(track.xyxy, searched_regions):
                    coverage_state = "SEARCHED_NOT_FOUND"
                else:
                    coverage_state = "NOT_SEARCHED_BY_POLICY"
                coverage_by_track.append({"track_id": track.track_id, "state": coverage_state})

            cache_row = dict(
                frame_id=frame_id,
                event_time_s=timestamp,
                detections=[_detection_record(d) for d in detections],
                tracks=[_track_record(t) for t in tracks],
                scheduler_decision=plan_record,
                observation_coverage=coverage_by_track,
            )
            cache_write_started = time.perf_counter()
            if cache_writer is not None:
                cache_writer.write(json.dumps(cache_row, separators=(",", ":")) + "\n")
            if rebuilt_cache_writer is not None:
                rebuilt_cache_writer.write(json.dumps(cache_row, separators=(",", ":")) + "\n")
            cache_write_ms = (time.perf_counter() - cache_write_started) * 1000

            should_log_diagnostics = (frame_id - start_frame) % diagnostic_stride == 0
            if should_log_diagnostics:
                scheduler_decisions.append(plan_record)
                measured_by_region = {
                    row["region_id"]: row for row in coverage
                }
                detection_counts = {region: 0 for region in DIAGNOSTIC_REGIONS}
                new_track_counts = {region: 0 for region in DIAGNOSTIC_REGIONS}
                for detection in detections:
                    region = DIAGNOSTIC_REGIONS[_diagnostic_region_index(detection.y2, height)]
                    detection_counts[region] += 1
                for track in tracks:
                    if track.track_id not in seen_track_ids:
                        region = DIAGNOSTIC_REGIONS[
                            _diagnostic_region_index(track.bottom_center[1], height)
                        ]
                        new_track_counts[region] += 1
                region_height = height / 3.0
                for region_index, region in enumerate(DIAGNOSTIC_REGIONS):
                    bounds = (0.0, region_index * region_height, float(width),
                              min(float(height), (region_index + 1) * region_height))
                    region_area = max(1.0, (bounds[2] - bounds[0]) * (bounds[3] - bounds[1]))
                    coverage_ratio = _rect_union_area(searched_regions, bounds) / region_area
                    coverage_row = {
                        "frame_id": frame_id,
                        "event_time_s": round(timestamp, 6),
                        "region_id": region,
                        "region_definition": "equal_height_diagnostic_band",
                        "sample_stride_frames": diagnostic_stride,
                        "coverage_ratio": round(min(1.0, coverage_ratio), 6),
                        "detections": detection_counts[region],
                        "new_track_ids": new_track_counts[region],
                        **measured_by_region[region],
                        "deadline_violations": None,
                        "grace_expirations": None,
                        "limitations": "age/deadline unavailable from current tracker contract",
                    }
                    coverage_rows.append(coverage_row)
                    region_funnel_rows.append({
                        "frame_id": frame_id,
                        "event_time_s": round(timestamp, 6),
                        "region_id": region,
                        "region_definition": "equal_height_diagnostic_band",
                        "detections": detection_counts[region],
                        "observed_tracks": measured_by_region[region]["MEASURED"],
                        "predicted_only_tracks": (
                            measured_by_region[region]["SEARCHED_NOT_FOUND"]
                            + measured_by_region[region]["NOT_SEARCHED_BY_POLICY"]
                        ),
                        "new_track_ids": new_track_counts[region],
                        "tracklet_deltas_in": None,
                        "tracklet_deltas_accepted": None,
                        "rejected_by_reason": None,
                        "accepted_motion_segments": None,
                        "candidate_count_before_top_k": None,
                        "selected_path_count": None,
                        "supported_path_length_px": None,
                        "diagnostic_status": "UPSTREAM_ONLY_DOWNSTREAM_NOT_INSTRUMENTED",
                    })
            seen_track_ids.update(track.track_id for track in tracks)
            count_tracks.update(t.track_id for t in tracks)
            last_tracks = tuple(tracks)
            points = [point_type("cam01", run_id, t.track_id, 0, frame_id,
                                 timestamp, *t.bottom_center, observed=t.observed)
                      for t in tracks]
            t2 = time.perf_counter()
            compute_count_before = len(tracklet.compute_ms) if tracklet is not None else 0
            if engine in ("directional_grid", "shadow"):
                assert grid is not None
                latest_snapshot = grid.update(points, timestamp)
            elif engine == "tracklet_aggregation":
                assert tracklet is not None
                latest_snapshot = tracklet.update(points, timestamp)
            directional_ms = (time.perf_counter() - t2) * 1000
            legacy_ms = 0.
            if engine in ("legacy", "shadow"):
                assert legacy is not None
                legacy_started = time.perf_counter()
                legacy_snapshot = legacy.process_points(
                    [SimpleNamespace(camera_id="cam01", track_id=t.track_id,
                                     frame_id=frame_id, timestamp=timestamp,
                                     x=t.bottom_center[0], y=t.bottom_center[1],
                                     confidence=t.confidence, zone_id=None)
                    for t in tracks if t.observed],
                    active_track_ids=(t.track_id for t in tracks), timestamp=timestamp)
                legacy_ms = (time.perf_counter() - legacy_started) * 1000
            analytics_ms = (time.perf_counter() - t2) * 1000
            common_path_compute_ms = None
            if tracklet is not None and len(tracklet.compute_ms) > compute_count_before:
                common_path_compute_ms = float(tracklet.compute_ms[-1])
            snapshot = legacy_snapshot if selected == "legacy" else latest_snapshot
            if should_log_diagnostics:
                rejection_counts = (
                    dict(getattr(tracklet, "rejections", {}))
                    if tracklet is not None else {}
                )
                # Engine traces change only during compute. Avoid copying and
                # hashing the same bounded history again on every video frame.
                compute_count = len(tracklet.compute_ms) if tracklet is not None else 0
                diagnostics_changed = compute_count != diagnostic_compute_count
                if diagnostics_changed:
                    candidate_trace = getattr(tracklet, "candidate_decisions", ())
                    support_trace = getattr(tracklet, "path_support_diagnostics", ())
                    diagnostic_compute_count = compute_count
                if isinstance(candidate_trace, Sequence) and candidate_trace:
                    for decision in candidate_trace if diagnostics_changed else ():
                        if isinstance(decision, dict):
                            trace_key = _canonical_digest(decision)
                            if trace_key in seen_candidate_diagnostics:
                                continue
                            seen_candidate_diagnostics.add(trace_key)
                            candidate_decisions.append({
                                "frame_id": frame_id,
                                "event_time_s": round(timestamp, 6),
                                "trace_scope": "engine",
                                **decision,
                            })
                else:
                    candidate_decisions.append({
                        "frame_id": frame_id,
                        "event_time_s": round(timestamp, 6),
                        "trace_scope": "snapshot_only",
                        "candidate_count_before_top_k": getattr(tracklet, "candidate_count", None),
                        "selected_path_count": len(snapshot.paths),
                        "selected_path_ids": [path.path_id for path in snapshot.paths],
                        "rejection_counts": rejection_counts,
                        "decision_status": "ENGINE_TRACE_NOT_EXPOSED",
                    })
                if isinstance(support_trace, Sequence) and support_trace:
                    for support in support_trace if diagnostics_changed else ():
                        if isinstance(support, dict):
                            trace_key = _canonical_digest(support)
                            if trace_key in seen_path_support_diagnostics:
                                continue
                            seen_path_support_diagnostics.add(trace_key)
                            path_support_rows.append({
                                "frame_id": frame_id,
                                "event_time_s": round(timestamp, 6),
                                **support,
                                "trace_scope": "engine",
                            })
                else:
                    for path in snapshot.paths:
                        polyline = np.asarray(path.polyline, dtype=np.float64)
                        length_px = float(np.linalg.norm(np.diff(polyline, axis=0), axis=1).sum()) \
                            if len(polyline) > 1 else 0.0
                        path_support_rows.append({
                            "frame_id": frame_id,
                            "event_time_s": round(timestamp, 6),
                            "path_id": path.path_id,
                            "revision": getattr(path, "revision", None),
                            "rank": getattr(path, "rank", None),
                            "sample_index": None,
                            "s_norm": None,
                            "x": None,
                            "y": None,
                            "raw_track_ids": None,
                            "raw_segments": None,
                            "observed_passage_support": path.support_tracks,
                            "deduplicated_support": path.support_tracks,
                            "weighted_support": path.score,
                            "direction_agreement": path.confidence,
                            "supported_length_px": round(length_px, 3),
                            "evidence_coverage": None,
                            "trace_scope": "whole_path_summary",
                            "limitations": "local support/segment provenance unavailable from current CommonPath schema",
                        })
                if region_funnel_rows:
                    candidate_count = getattr(tracklet, "candidate_count", None) if tracklet is not None else None
                    selected_count = len(snapshot.paths)
                    rejected_json = json.dumps(rejection_counts, sort_keys=True) if rejection_counts else None
                    supported_length = sum(
                        float(np.linalg.norm(np.diff(np.asarray(path.polyline, dtype=np.float64), axis=0), axis=1).sum())
                        if len(path.polyline) > 1 else 0.0
                        for path in snapshot.paths
                    )
                    for row in region_funnel_rows[-len(DIAGNOSTIC_REGIONS):]:
                        row["candidate_count_before_top_k"] = candidate_count
                        row["selected_path_count"] = selected_count
                        row["rejected_by_reason"] = rejected_json
                        row["supported_path_length_px"] = round(supported_length, 3)
                        row["diagnostic_status"] = (
                            "ENGINE_COUNTERS_ONLY" if tracklet is not None
                            else "NO_COMMON_PATH_ENGINE"
                        )
            t3 = time.perf_counter()
            preview_fps = (len(frame_ids) + 1) / max(time.perf_counter() - began, 1e-6)
            preview_latency_ms = (
                (time.perf_counter() - float(prepared["pipeline_started_at"])) * 1000
                if prepared is not None and prepared.get("pipeline_started_at") is not None
                else (time.perf_counter() - t0) * 1000
            )
            rendered = renderer.render_point_only_frame(
                frame, [SimpleNamespace(track_id=t.track_id, x=t.bottom_center[0],
                                        y=t.bottom_center[1])
                        for t in tracks if t.observed],
                snapshot, tuple(config.analytics.zones), transformer, overlay,
                people_count=len(tracks), processing_fps=preview_fps, latency_ms=preview_latency_ms,
                timestamp=timestamp,
                tracked_objects=tuple(tracks))
            render_ms = (time.perf_counter() - t3) * 1000
            t4 = time.perf_counter()
            writer.write(rendered)
            encode_ms = (time.perf_counter() - t4) * 1000
            if len(frame_ids) % max(1, math.ceil((end_frame - start_frame) / 8)) == 0:
                cv2.putText(rendered, f"frame={frame_id} media={timestamp:.2f}s", (15, height - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 2)
                capture_frames.append(cv2.resize(rendered, (width // 3, height // 3)))
            frame_ids.append(frame_id)
            frame_total_ms = (time.perf_counter() - t0) * 1000
            e2e_latency_ms = (
                (time.perf_counter() - float(prepared["pipeline_started_at"])) * 1000
                if prepared is not None and prepared.get("pipeline_started_at") is not None
                else frame_total_ms
            )
            pipeline_overhead_ms = max(
                0.0,
                frame_total_ms
                - decode_ms
                - inference_ms
                - tracking_ms
                - analytics_ms
                - render_ms
                - encode_ms
                - cache_write_ms,
            )
            intentional_skip = str(plan.get("scan_type")) == "skip"
            timings.append(dict(frame_id=frame_id, event_time_s=round(timestamp, 5),
                                detections=len(detections), tracks=len(tracks), decode_ms=decode_ms,
                                inference_ms=(None if intentional_skip else inference_ms),
                                tracking_ms=tracking_ms,
                                tile_prepare_ms=detector_stats.get("tile_prepare_ms"),
                                model_tensor_shape=detector_stats.get("model_tensor_shape"),
                                cuda_graph_replayed=detector_stats.get("cuda_graph_replayed"),
                                cuda_graph_disabled_reason=detector_stats.get("cuda_graph_disabled_reason"),
                                yolo_preprocess_total_ms=detector_stats.get("yolo_preprocess_total_ms"),
                                h2d_ms=detector_stats.get("event_h2d_total_ms"),
                                tensor_conversion_ms=detector_stats.get(
                                    "event_tensor_conversion_total_ms"
                                ),
                                pre_transform_cpu_ms=detector_stats.get("pre_transform_cpu_ms"),
                                stack_cpu_ms=detector_stats.get("stack_cpu_ms"),
                                layout_contiguous_cpu_ms=detector_stats.get(
                                    "layout_contiguous_cpu_ms"
                                ),
                                tensor_from_numpy_cpu_ms=detector_stats.get(
                                    "tensor_from_numpy_cpu_ms"
                                ),
                                yolo_inference_total_ms=detector_stats.get("yolo_inference_total_ms"),
                                yolo_postprocess_total_ms=detector_stats.get("yolo_postprocess_total_ms"),
                                detector_worker_mode=detector_stats.get(
                                    "worker_mode", detector_worker_mode
                                ),
                                ultralytics_stage_synchronizations_per_batch=detector_stats.get(
                                    "ultralytics_stage_synchronizations_per_batch"
                                ),
                                result_cpu_transfers_per_image=detector_stats.get(
                                    "result_cpu_transfers_per_image"
                                ),
                                model_predict_ms=detector_stats.get("model_predict_ms"),
                                result_transfer_ms=detector_stats.get("result_transfer_ms"),
                                postprocess_ms=detector_stats.get("postprocess_ms"),
                                merge_ms=detector_stats.get("merge_ms"),
                                motion_ms=plan.get("motion_ms"), planning_ms=plan.get("planning_ms"),
                                roi_union_ratio=plan.get("roi_union_ratio"),
                                scan_type=plan.get("scan_type"),
                                model_invocations=workload.get("model_invocations"),
                                inference_images=workload.get("inference_images"),
                                source_batch_size=workload.get("source_batch_size", 1),
                                detector_batch_size=workload.get("batch_size"),
                                inference_batch_id=workload.get("inference_batch_id"),
                                sum_tensor_pixels=workload.get("sum_tensor_pixels"),
                                directional_analytics_ms=directional_ms,
                                legacy_analytics_ms=legacy_ms, analytics_ms=analytics_ms,
                                common_path_compute_ms=common_path_compute_ms,
                                render_ms=render_ms, encode_ms=encode_ms,
                                cache_write_ms=cache_write_ms,
                                gpu_input_queue_wait_ms=(
                                    prepared.get("gpu_input_queue_wait_ms")
                                    if prepared is not None else None
                                ),
                                consumer_result_wait_ms=(
                                    prepared.get("consumer_result_wait_ms")
                                    if prepared is not None else None
                                ),
                                result_queue_age_ms=(
                                    prepared.get("result_queue_age_ms")
                                    if prepared is not None else None
                                ),
                                e2e_latency_ms=e2e_latency_ms,
                                pipeline_overhead_ms=pipeline_overhead_ms,
                                frame_total_ms=frame_total_ms,
                                pipeline_elapsed_seconds=time.perf_counter() - began,
                                tracklet_buffered_segments=(
                                    tracklet.buffered_segment_count if tracklet is not None else None
                                ),
                                tracklet_valid_tracklets=(
                                    tracklet.rejections.get("VALID_TRACKLETS", 0)
                                    if tracklet is not None else None
                                ),
                                tracklet_candidates=(
                                    tracklet.candidate_count if tracklet is not None else None
                                ),
                                tracklet_link_cache_hits=(
                                    tracklet.link_score_cache_hits if tracklet is not None else None
                                ),
                                ram_mb=psutil.Process().memory_info().rss / 1024**2))
            ram_samples_mb.append(float(timings[-1]["ram_mb"]))
            last_frame = frame.copy()
    finally:
        if consumer_profile is not None:
            consumer_profile.disable()
        if overlap_pipeline is not None:
            overlap_pipeline.close()
        cap.release()
        writer.release()
        if cache_reader:
            cache_reader.close()
        if cache_writer:
            cache_writer.close()
        if rebuilt_cache_writer:
            rebuilt_cache_writer.close()
        if profile_cpu:
            _write_cpu_profile(consumer_profile, output_dir, "consumer")
            if overlap_pipeline is not None:
                _write_cpu_profile(
                    overlap_pipeline.producer_profile, output_dir, "producer"
                )
                _write_cpu_profile(overlap_pipeline.gpu_profile, output_dir, "gpu_worker")
    frame_loop_elapsed = time.perf_counter() - began
    if not frame_ids:
        raise RuntimeError("Clip has no decoded frames")
    if tracklet is not None:
        latest_snapshot = tracklet.finalize(last_time)
        # Finalization may publish one last candidate/support revision after
        # the final sampled frame. Export the engine-owned bounded traces below
        # as an authoritative tail without duplicating earlier rows.
        for decision in tracklet.candidate_decisions:
            trace_key = _canonical_digest(decision)
            if trace_key not in seen_candidate_diagnostics:
                seen_candidate_diagnostics.add(trace_key)
                candidate_decisions.append({"trace_scope": "engine_final", **decision})
        for support in tracklet.path_support_diagnostics:
            trace_key = _canonical_digest(support)
            if trace_key not in seen_path_support_diagnostics:
                seen_path_support_diagnostics.add(trace_key)
                path_support_rows.append({"trace_scope": "engine_final", **support})
    _write_jsonl(output_dir / "scheduler_decisions.jsonl", scheduler_decisions)
    _write_jsonl(output_dir / "candidate_decisions.jsonl", candidate_decisions)
    _write_jsonl(output_dir / "path_support.jsonl", path_support_rows)
    scheduler_trace_hash = digest(output_dir / "scheduler_decisions.jsonl")
    if cache_writer is not None:
        cache_path.with_suffix(".meta.json").write_text(
            json.dumps(dict(cache_key=key, source_hash=input_hash, model_hash=model_hash,
                            start_seconds=start_seconds, duration_seconds=duration_seconds,
                            frame_count=len(frame_ids), cache_schema=CACHE_SCHEMA,
                            provenance_schema=PROVENANCE_SCHEMA,
                            config_hash=config_hash, scheduler_fingerprint=scheduler_hash,
                            scheduler_decision_trace_hash=scheduler_trace_hash,
                            code_fingerprint=code_hash,
                            motion_roi_enabled=motion_enabled,
                            first_frame_id=frame_ids[0], last_frame_id=frame_ids[-1],
                            first_event_time_s=timings[0]["event_time_s"],
                            last_event_time_s=timings[-1]["event_time_s"]), indent=2), encoding="utf-8")
    if rebuilt_cache_path is not None:
        rebuilt_key = hashlib.sha256(f"{key}:tracklets-observed-v2".encode()).hexdigest()
        rebuilt_cache_path.with_suffix(".meta.json").write_text(
            json.dumps(dict(cache_key=rebuilt_key, source_cache_key=key,
                            source_hash=input_hash, model_hash=model_hash,
                            tracklet_schema="tracklets-observed-v2",
                            start_seconds=start_seconds, duration_seconds=duration_seconds,
                            frame_count=len(frame_ids), provenance_schema=PROVENANCE_SCHEMA,
                            config_hash=config_hash, scheduler_fingerprint=scheduler_hash,
                            scheduler_decision_trace_hash=scheduler_trace_hash,
                            code_fingerprint=code_hash), indent=2), encoding="utf-8")
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as sink:
        metrics_writer = csv.DictWriter(sink, fieldnames=list(timings[0]))
        metrics_writer.writeheader()
        metrics_writer.writerows(timings)
    _write_stage_reports(output_dir, timings)
    timing_by_frame = {int(item["frame_id"]): item for item in timings}
    _write_csv_rows(
        output_dir / "motion_roi_metrics.csv",
        [
            {
                "frame_id": row["frame_id"],
                "event_time_s": row["event_time_s"],
                "state": row.get("state"),
                "scan_type": row.get("scan_type"),
                "proposed_scan_type": row.get("proposed_scan_type"),
                "motion_ms": row.get("motion_ms"),
                "planning_ms": row.get("planning_ms"),
                "inference_ms": timing_by_frame.get(int(row["frame_id"]), {}).get("inference_ms"),
                "tracking_ms": timing_by_frame.get(int(row["frame_id"]), {}).get("tracking_ms"),
                "model_invocations": row.get("model_invocations"),
                "inference_images": row.get("inference_images"),
                "batch_size": row.get("batch_size"),
                "actual_tensor_shapes": json.dumps(row.get("actual_tensor_shapes", [])),
                "sum_tensor_pixels": row.get("sum_tensor_pixels"),
                "padding_overhead_pixels": row.get("padding_overhead_pixels"),
                "foreground_ratio": row.get("foreground_ratio"),
                "roi_union_ratio": row.get("roi_union_ratio"),
                "estimated_cost_ratio": row.get("estimated_cost_ratio"),
                "workload_source": row.get("workload_source"),
            }
            for row in scheduler_decisions
        ],
        (
            "frame_id", "event_time_s", "state", "scan_type", "proposed_scan_type",
            "motion_ms", "planning_ms", "inference_ms", "tracking_ms",
            "model_invocations", "inference_images", "batch_size", "actual_tensor_shapes",
            "sum_tensor_pixels", "padding_overhead_pixels", "foreground_ratio",
            "roi_union_ratio", "estimated_cost_ratio", "workload_source",
        ),
    )
    _write_csv_rows(
        output_dir / "observation_coverage_by_region.csv",
        coverage_rows,
        (
            "frame_id", "event_time_s", "region_id", "region_definition",
            "sample_stride_frames", "coverage_ratio", "detections", "new_track_ids",
            "track_count", "MEASURED", "SEARCHED_NOT_FOUND", "NOT_SEARCHED_BY_POLICY",
            "observation_age_p50_s", "observation_age_p95_s", "observation_age_max_s",
            "age_status", "deadline_violations", "grace_expirations", "limitations",
        ),
    )
    _write_csv_rows(
        output_dir / "region_funnel.csv",
        region_funnel_rows,
        (
            "frame_id", "event_time_s", "region_id", "region_definition", "detections",
            "observed_tracks", "predicted_only_tracks", "new_track_ids", "tracklet_deltas_in",
            "tracklet_deltas_accepted", "rejected_by_reason", "accepted_motion_segments",
            "candidate_count_before_top_k", "selected_path_count", "supported_path_length_px",
            "diagnostic_status",
        ),
    )
    if capture_frames:
        sheet = np.vstack([np.hstack(capture_frames[i:i+3]) for i in range(0, len(capture_frames)-2, 3)])
        cv2.imwrite(str(output_dir / "preview_contact_sheet.jpg"), sheet)
    assert first_frame is not None and last_frame is not None
    cv2.imwrite(str(output_dir / "common_path_map.png"), renderer.render_point_only_frame(
        last_frame, (), latest_snapshot if selected in ("directional_grid", "tracklet_aggregation") else legacy_snapshot,
        tuple(config.analytics.zones), transformer, overlay,
        people_count=0, processing_fps=0, latency_ms=0, timestamp=last_time))
    if grid is not None:
        for image_name in ("directional_field_debug.png", "edge_flow_debug.png"):
            debug = first_frame.copy()
            if image_name.startswith("directional"):
                for cell, bins in grid.histogram().items():
                    x = int((cell[1]+.5)*width/config.analytics.directional_grid.columns)
                    y = int((cell[0]+.5)*height/config.analytics.directional_grid.rows)
                    strong = sorted(((direction, support[1]) for direction, support in bins.items()
                                     if support[1] >= config.analytics.directional_grid.min_edge_unique_tracks),
                                    key=lambda item: -item[1])[:2]
                    for direction, total in strong:
                        angle = direction*2*math.pi/config.analytics.directional_grid.direction_bins
                        cv2.arrowedLine(debug, (x,y), (x+int(17*math.cos(angle)),y+int(17*math.sin(angle))),
                                        (0,255,255), max(1,min(3,total)), tipLength=.35)
            else:
                for edge in grid.flow_snapshot().edges:
                    if edge.unique_tracks_long < config.analytics.directional_grid.min_edge_unique_tracks:
                        continue
                    a,b = edge.from_cell,edge.to_cell
                    scale = lambda cell: (int((cell[1]+.5)*width/config.analytics.directional_grid.columns),
                                          int((cell[0]+.5)*height/config.analytics.directional_grid.rows))
                    cv2.arrowedLine(debug,scale(a),scale(b),(40,220,40),
                                    max(1,min(4,edge.unique_tracks_long)),tipLength=.4)
            cv2.imwrite(str(output_dir / image_name), debug)
    path_events = list(tracklet.events) if tracklet is not None else list(grid.events) if grid is not None else []
    if not path_events:
        path_events.append(dict(
            event="insufficient_data",
            reason="no complete boundary-to-boundary route met configured support",
            path_id="", event_time_s=last_time, evidence_until_s=last_time,
        ))
    with (output_dir / "path_events.jsonl").open("w", encoding="utf-8") as events:
        for event in path_events:
            events.write(json.dumps(event) + "\n")
    (output_dir / "config_resolved.yaml").write_text(
        yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False), encoding="utf-8")
    overlap_metrics = {
        "enabled": overlap_pipeline is not None,
        "queue_capacity_batches": int(config.video.offline_overlap_queue_batches),
        "input_queue_peak_batches": (
            overlap_pipeline.input_queue_peak if overlap_pipeline is not None else 0
        ),
        "result_queue_peak_batches": (
            overlap_pipeline.result_queue_peak if overlap_pipeline is not None else 0
        ),
        "producer_put_wait_total_ms": round(
            overlap_pipeline.producer_put_wait_ms, 3
        ) if overlap_pipeline is not None else 0.0,
        "gpu_input_wait_total_ms": round(
            overlap_pipeline.gpu_input_wait_ms, 3
        ) if overlap_pipeline is not None else 0.0,
        "result_put_wait_total_ms": round(
            overlap_pipeline.result_put_wait_ms, 3
        ) if overlap_pipeline is not None else 0.0,
    }
    provenance = {
        "schema": PROVENANCE_SCHEMA,
        "run_id": run_id,
        "input": {
            "path": str(input_path),
            "sha256": input_hash,
            "fps": fps,
            "width": width,
            "height": height,
            "source_frame_count": source_frame_count,
        },
        "clip": {
            "start_seconds": start_seconds,
            "duration_seconds": duration_seconds,
            "start_frame": start_frame,
            "end_frame_exclusive": end_frame,
            "first_processed_frame": frame_ids[0],
            "last_processed_frame": frame_ids[-1],
            "first_event_time_s": timings[0]["event_time_s"],
            "last_event_time_s": timings[-1]["event_time_s"],
            "timestamp_policy": "source PTS, fallback frame_id/fps",
        },
        "model": {"sha256": model_hash, "path": str(model_path) if model_path else None},
        "engine": engine,
        "replay_cache": str(replay_cache) if replay_cache else None,
        "cache_key": key,
        "cache_key_match": cache_key_match,
        "cache_key_override": bool(allow_cache_key_mismatch) if replay_cache is not None else False,
        "config_hash": config_hash,
        "scheduler_fingerprint": scheduler_hash,
        "scheduler_decision_trace_hash": scheduler_trace_hash,
        "code_fingerprint": code_hash,
        "motion_roi_enabled": motion_enabled,
        "motion_roi_shadow_mode": bool(getattr(motion_config, "shadow_mode", False)),
        "profile_gpu": profile_gpu,
        "profile_cpu": profile_cpu,
        "detector_worker_mode": detector_worker_mode,
        "inference_backend": inference_backend,
        "phases": {
            "detector_init_ms": round(detector_init_ms, 3),
            "detector_warmup_ms": round(detector_warmup_ms, 3),
            "frame_loop_seconds": round(frame_loop_elapsed, 3),
        },
        "offline_batch": {
            "source_batch_size_requested": requested_source_batch_size,
            "source_batch_size_effective": effective_source_batch_size,
            "max_detector_images_per_batch": int(
                config.video.max_detector_images_per_batch
            ),
            "detector_images_per_reference_frame": detector_images_per_reference,
            "fallback_reason": source_batch_fallback_reason,
            "overlap": overlap_metrics,
        },
        "versions": _package_versions(),
        "runtime": {"python_executable": sys.executable, "platform": platform.platform()},
        "diagnostic_sampling": {
            "max_decisions": diagnostic_limit,
            "stride_frames": diagnostic_stride,
            "region_definition": "equal_height_diagnostic_band",
            "regions": list(DIAGNOSTIC_REGIONS),
        },
        "limitations": [
            "Tensor shapes and padding are unavailable unless detector instrumentation is enabled",
            "Track observation age/deadline is unavailable from the current batch tracker contract",
            "Common Path trace is bounded by engine diagnostics and may expose aggregate rather than per-segment support",
        ],
    }
    (output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    elapsed = time.perf_counter() - began
    artifact_finalize_seconds = max(0.0, elapsed - frame_loop_elapsed)
    def p50(field: str) -> float:
        values = [t[field] for t in timings if t.get(field) is not None]
        return round(float(np.percentile(values, 50)), 3) if values else 0.0
    def p95(field: str) -> float:
        values = [t[field] for t in timings if t.get(field) is not None]
        return round(float(np.percentile(values, 95)), 3) if values else 0.0
    switch_count = sum(event["event"] == "switch" for event in path_events)
    media_duration = max(last_time - start_seconds, 1e-9)
    policy_skipped_frames = sum(
        str(row.get("scan_type")) == "skip" for row in timings
    )
    association_updates = len(timings) - policy_skipped_frames
    detector_images_total = sum(
        int(row.get("inference_images") or 0) for row in timings
    )
    final_snapshot = legacy_snapshot if selected == "legacy" else latest_snapshot
    tracking_cache_sha256 = (
        digest(cache_path) if replay_cache is None and cache_path.is_file() else None
    )
    summary = dict(run_type="analytics_replay" if replay_cache else "gpu_pipeline",
                   engine=engine, selected_display=selected, frames=len(frame_ids),
                   first_frame=frame_ids[0], last_frame=frame_ids[-1],
                   media_start_s=timings[0]["event_time_s"], media_end_s=last_time,
                   device=device,
                   tracks_source=("rebuilt_from_cached_detections" if rebuild_tracks else
                                  "cached_tracks" if replay_cache else "live_gpu_pipeline"),
                   unique_track_ids=len(count_tracks),
                   result="active" if final_snapshot.paths else "insufficient_data",
                   active_paths=[asdict(p) for p in final_snapshot.paths],
                   legacy_paths=[asdict(p) for p in legacy_snapshot.paths] if legacy is not None else [],
                   directional_edges=len(grid.flow_snapshot().edges) if grid is not None else 0,
                   support_cap_drops=grid.overflow if grid is not None else 0,
                   rejected_jump_segments=grid.rejected_jumps if grid is not None else 0,
                   analytics_p50_ms=p50("analytics_ms"), analytics_p95_ms=p95("analytics_ms"),
                   common_path_compute_p50_ms=p50("common_path_compute_ms"),
                   common_path_compute_p95_ms=p95("common_path_compute_ms"),
                   decode_p50_ms=p50("decode_ms"), decode_p95_ms=p95("decode_ms"),
                   detector_model_p50_ms=p50("model_predict_ms"),
                   detector_model_p95_ms=p95("model_predict_ms"),
                   yolo_preprocess_p50_ms=p50("yolo_preprocess_total_ms"),
                   yolo_preprocess_p95_ms=p95("yolo_preprocess_total_ms"),
                   h2d_p50_ms=p50("h2d_ms"), h2d_p95_ms=p95("h2d_ms"),
                   tensor_conversion_p50_ms=p50("tensor_conversion_ms"),
                   tensor_conversion_p95_ms=p95("tensor_conversion_ms"),
                   yolo_forward_p50_ms=p50("yolo_inference_total_ms"),
                   yolo_forward_p95_ms=p95("yolo_inference_total_ms"),
                   yolo_postprocess_p50_ms=p50("yolo_postprocess_total_ms"),
                   yolo_postprocess_p95_ms=p95("yolo_postprocess_total_ms"),
                    detector_result_transfer_p50_ms=p50("result_transfer_ms"),
                    detector_result_transfer_p95_ms=p95("result_transfer_ms"),
                   detector_postprocess_p50_ms=p50("postprocess_ms"),
                   detector_postprocess_p95_ms=p95("postprocess_ms"),
                   detector_merge_p50_ms=p50("merge_ms"),
                   detector_merge_p95_ms=p95("merge_ms"),
                   inference_p50_ms=p50("inference_ms"), inference_p95_ms=p95("inference_ms"),
                   tracking_p50_ms=p50("tracking_ms"), tracking_p95_ms=p95("tracking_ms"),
                   render_p50_ms=p50("render_ms"), render_p95_ms=p95("render_ms"),
                   encode_p50_ms=p50("encode_ms"), encode_p95_ms=p95("encode_ms"),
                   cache_write_p50_ms=p50("cache_write_ms"),
                   cache_write_p95_ms=p95("cache_write_ms"),
                   gpu_input_queue_wait_p50_ms=p50("gpu_input_queue_wait_ms"),
                   gpu_input_queue_wait_p95_ms=p95("gpu_input_queue_wait_ms"),
                   consumer_result_wait_p50_ms=p50("consumer_result_wait_ms"),
                   consumer_result_wait_p95_ms=p95("consumer_result_wait_ms"),
                   result_queue_age_p50_ms=p50("result_queue_age_ms"),
                   result_queue_age_p95_ms=p95("result_queue_age_ms"),
                   e2e_latency_p50_ms=p50("e2e_latency_ms"),
                   e2e_latency_p95_ms=p95("e2e_latency_ms"),
                   offline_overlap=overlap_metrics,
                   directional_analytics_p95_ms=p95("directional_analytics_ms"),
                   legacy_analytics_p95_ms=p95("legacy_analytics_ms"),
                   path_switches=switch_count,
                   switches_per_minute=round(switch_count / (media_duration / 60), 3),
                   validated_route_support=[path.validated_complete_tracks
                                            for path in final_snapshot.paths],
                   polyline_jitter_px=None,
                   change_detection_delay_s=None,
                   processing_fps=round(len(frame_ids)/frame_loop_elapsed, 3),
                   processing_seconds=round(frame_loop_elapsed, 3),
                   artifact_finalize_seconds=round(artifact_finalize_seconds, 3),
                   detector_init_ms=round(detector_init_ms, 3),
                   detector_warmup_ms=round(detector_warmup_ms, 3),
                   source_batch_size_requested=requested_source_batch_size,
                   source_batch_size_effective=effective_source_batch_size,
                   inference_interval=int(config.video.inference_interval),
                   max_detector_images_per_batch=int(
                       config.video.max_detector_images_per_batch
                   ),
                   source_batch_fallback_reason=source_batch_fallback_reason,
                   detector_worker_mode=detector_worker_mode,
                   inference_backend=inference_backend,
                   tracking_cache_sha256=tracking_cache_sha256,
                   inference_batches_total=count_inference_batches(timings),
                    detector_invocations_total=sum(
                        int(row.get("model_invocations") or 0) for row in timings
                    ),
                    detector_images_total=detector_images_total,
                    detector_images_per_source_frame=round(
                        detector_images_total / max(1, len(frame_ids)), 3
                    ),
                   detector_scans_total=association_updates,
                   detector_scans_per_source_second=round(
                       association_updates / media_duration, 3
                   ),
                   detector_scans_per_wall_second=round(
                       association_updates / frame_loop_elapsed, 3
                   ),
                   detector_images_per_source_second=round(
                       detector_images_total / media_duration, 3
                   ),
                   detector_images_per_wall_second=round(
                       detector_images_total / frame_loop_elapsed, 3
                   ),
                   policy_skipped_frames=policy_skipped_frames,
                   prediction_steps=policy_skipped_frames,
                   association_updates=association_updates,
                   detection_scan_deadline_misses=(
                       overlap_pipeline.detection_scan_deadline_misses
                       if overlap_pipeline is not None else 0
                   ),
                   provenance_schema=PROVENANCE_SCHEMA,
                   config_hash=config_hash,
                   scheduler_fingerprint=scheduler_hash,
                   scheduler_decision_trace_hash=scheduler_trace_hash,
                   cache_key_match=cache_key_match,
                   cache_key_override=(bool(allow_cache_key_mismatch) if replay_cache is not None else False),
                   diagnostics={
                       "scheduler_decisions": len(scheduler_decisions),
                       "candidate_decisions": len(candidate_decisions),
                       "path_support_rows": len(path_support_rows),
                       "coverage_rows": len(coverage_rows),
                       "region_funnel_rows": len(region_funnel_rows),
                       "diagnostic_stride_frames": diagnostic_stride,
                   },
                   ram_peak_mb=round(max(ram_samples_mb), 2),
                   vram_allocated_peak_mb=(round(cuda_runtime.cuda.max_memory_allocated()/1024**2, 2)
                                           if cuda_runtime else None),
                   vram_reserved_peak_mb=(round(cuda_runtime.cuda.max_memory_reserved()/1024**2, 2)
                                          if cuda_runtime else None),
                   limitations=["Short clip does not fill 180-second window",
                                "Pixel-space grid has uncalibrated perspective",
                                "Tracking ID fragmentation can undercount complete routes",
                                "Path jitter and change delay are N/A without an Active Path"])
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "inspection.md").write_text(
        "# Artifact inspection\n\nStatus: NOT_REVIEWED\n\n"
        "Review the MP4, contact sheet, both debug fields, path events, and summary.\n",
        encoding="utf-8",
    )
    manifest = dict(run_id=run_id, status="success", **{k: summary[k] for k in
                     ("run_type", "engine", "device", "frames", "processing_fps",
                      "processing_seconds", "artifact_finalize_seconds",
                      "detector_init_ms", "detector_warmup_ms",
                      "source_batch_size_requested", "source_batch_size_effective",
                      "inference_interval",
                      "max_detector_images_per_batch", "source_batch_fallback_reason",
                      "detector_worker_mode", "inference_backend", "tracking_cache_sha256",
                      "inference_batches_total",
                      "detector_invocations_total", "detector_images_total",
                      "detector_images_per_source_frame",
                      "detector_scans_total", "detector_scans_per_source_second",
                      "detector_scans_per_wall_second",
                      "detector_images_per_source_second",
                      "detector_images_per_wall_second", "policy_skipped_frames",
                      "prediction_steps", "association_updates",
                      "detection_scan_deadline_misses",
                      "inference_p50_ms", "inference_p95_ms", "tracking_p50_ms",
                      "tracking_p95_ms", "analytics_p50_ms", "analytics_p95_ms",
                      "render_p50_ms", "render_p95_ms", "encode_p50_ms",
                      "encode_p95_ms", "detector_result_transfer_p50_ms",
                      "detector_result_transfer_p95_ms",
                      "yolo_preprocess_p50_ms", "yolo_preprocess_p95_ms",
                      "h2d_p50_ms", "h2d_p95_ms",
                      "yolo_forward_p50_ms", "yolo_forward_p95_ms",
                      "yolo_postprocess_p50_ms", "yolo_postprocess_p95_ms",
                      )},
                     input_hash=input_hash, model_hash=model_hash, cache_key=key,
                     cache_schema=CACHE_SCHEMA,
                     provenance_schema=PROVENANCE_SCHEMA,
                     config_hash=config_hash,
                     scheduler_fingerprint=scheduler_hash,
                     scheduler_decision_trace_hash=scheduler_trace_hash,
                     code_fingerprint=code_hash,
                     motion_roi_enabled=motion_enabled,
                     motion_roi_shadow_mode=bool(getattr(motion_config, "shadow_mode", False)),
                     cache_file=cache_path.name if replay_cache is None else str(replay_cache),
                    rebuilt_cache_file=(rebuilt_cache_path.name if rebuilt_cache_path else None),
                    clip=dict(start_seconds=start_seconds, duration_seconds=duration_seconds),
                    artifacts=sorted(p.name for p in output_dir.iterdir()))
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--config", default="configs/default.yaml", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--start-seconds", type=float, default=0.)
    parser.add_argument("--duration-seconds", type=float,
                        help="Optional processing limit; omit to process through end of input")
    parser.add_argument("--engine", choices=("legacy", "directional_grid", "tracklet_aggregation", "shadow"), default="tracklet_aggregation")
    parser.add_argument("--replay-cache", type=Path)
    parser.add_argument("--source-hash")
    parser.add_argument("--model-hash")
    parser.add_argument("--rebuild-tracks", action="store_true",
                        help="Re-run ByteTrack from cached detections without inference")
    parser.add_argument("--allow-cache-key-mismatch", action="store_true",
                        help="Audit-only: allow an old cache key when source/model hashes match")
    parser.add_argument("--profile-gpu", action="store_true",
                        help="Insert cuda.synchronize() around each inference for accurate per-stage "
                             "GPU timing. Adds ~50-80ms/frame overhead; never use in throughput benchmarks.")
    parser.add_argument("--profile-cpu", action="store_true",
                        help="Write per-thread cProfile artifacts; diagnostic runs only.")
    parser.add_argument(
        "--detector-worker-mode",
        choices=("baseline", "no_sync_profile"),
        default=None,
        help="Run one isolated detector-worker optimization experiment.",
    )
    parser.add_argument(
        "--inference-backend",
        choices=("pytorch_fp32", "pytorch_fp16", "tensorrt_fp16"),
        default="pytorch_fp32",
    )
    args = parser.parse_args()
    print(json.dumps(run(args.input, args.output_dir, config_path=args.config, model_path=args.model,
                         start_seconds=args.start_seconds, duration_seconds=args.duration_seconds,
                         run_id=args.run_id, engine=args.engine, replay_cache=args.replay_cache,
                         input_hash=args.source_hash, model_hash=args.model_hash,
                         rebuild_tracks=args.rebuild_tracks,
                         allow_cache_key_mismatch=args.allow_cache_key_mismatch,
                         profile_gpu=args.profile_gpu,
                         profile_cpu=args.profile_cpu,
                         detector_worker_mode=args.detector_worker_mode,
                         inference_backend=args.inference_backend)))


if __name__ == "__main__":
    main()
