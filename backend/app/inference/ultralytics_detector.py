from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from backend.app.core.config import DetectorConfig
from backend.app.schemas import Detection


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
    selected_sources: list[int | None] = []
    selected_boxes = np.empty((max_detections, 4), dtype=np.float32)
    selected_classes = np.empty(max_detections, dtype=np.int32)
    selected_source_values = np.empty(max_detections, dtype=np.int32)
    indexed = sorted(
        enumerate(detections),
        key=lambda item: item[1].confidence,
        reverse=True,
    )
    for original_index, candidate in indexed:
        duplicate = False
        candidate_source = source_ids[original_index] if source_ids is not None else None
        if selected:
            existing = selected_boxes[:len(selected)]
            first_area = max(0.0, candidate.x2 - candidate.x1) * max(0.0, candidate.y2 - candidate.y1)
            second_area = np.maximum(0.0, existing[:, 2] - existing[:, 0]) * np.maximum(
                0.0, existing[:, 3] - existing[:, 1]
            )
            intersection = (
                np.maximum(0.0, np.minimum(candidate.x2, existing[:, 2]) - np.maximum(candidate.x1, existing[:, 0]))
                * np.maximum(0.0, np.minimum(candidate.y2, existing[:, 3]) - np.maximum(candidate.y1, existing[:, 1]))
            )
            union = first_area + second_area - intersection
            iou = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0.0)
            smaller = np.minimum(first_area, second_area)
            intersection_over_smaller = np.divide(
                intersection, smaller, out=np.zeros_like(intersection), where=smaller > 0.0
            )
            first_center = np.asarray(((candidate.x1 + candidate.x2) / 2.0,
                                       (candidate.y1 + candidate.y2) / 2.0), dtype=np.float32)
            second_centers = (existing[:, :2] + existing[:, 2:]) / 2.0
            center_distance = np.linalg.norm(second_centers - first_center[None, :], axis=1)
            first_diagonal = float(np.hypot(candidate.x2 - candidate.x1, candidate.y2 - candidate.y1))
            second_diagonal = np.hypot(existing[:, 2] - existing[:, 0], existing[:, 3] - existing[:, 1])
            center_close = center_distance <= np.maximum(3.0, 0.50 * np.minimum(first_diagonal, second_diagonal))
            scale_ratio = np.divide(
                smaller, np.maximum(first_area, second_area),
                out=np.zeros_like(smaller), where=np.maximum(first_area, second_area) > 0.0,
            )
            same_source = (
                np.zeros(len(selected), dtype=bool)
                if source_ids is None
                else selected_source_values[:len(selected)] == int(candidate_source)
            )
            duplicate = bool(np.any(
                (selected_classes[:len(selected)] == int(candidate.class_id))
                & ~same_source
                & (scale_ratio >= 0.35)
                & center_close
                & ((iou >= iou_threshold) | (intersection_over_smaller >= 0.8))
            ))
        if not duplicate:
            selected.append(candidate)
            selected_sources.append(candidate_source)
            selected_boxes[len(selected) - 1] = (candidate.x1, candidate.y1, candidate.x2, candidate.y2)
            selected_classes[len(selected) - 1] = candidate.class_id
            selected_source_values[len(selected) - 1] = -1 if candidate_source is None else int(candidate_source)
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

    def __init__(self, config: DetectorConfig) -> None:
        from ultralytics import YOLO

        self._config = config
        self._device = None if config.device == "auto" else config.device
        self._model: Any = YOLO(config.model)
        self._last_inference_stats: dict[str, Any] = {}

    @property
    def last_inference_stats(self) -> dict[str, Any]:
        return dict(self._last_inference_stats)

    def _predict(self, frames: list[NDArray[np.uint8]]) -> list[list[Detection]]:
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
        prediction_options: dict[str, Any] = {
            "source": frames,
            "imgsz": self._config.imgsz,
            "conf": self._config.confidence,
            "iou": self._config.iou,
            "max_det": self._config.max_det,
            "classes": self._config.classes,
            "device": self._device,
            "verbose": False,
        }
        if self._config.precision == "fp16":
            # Avoid passing the deprecated flag at all for FP32.  Older
            # Ultralytics builds still require ``half=True`` for CUDA FP16.
            prediction_options["half"] = True
        prediction_started = time.perf_counter()
        results = self._model.predict(**prediction_options)
        prediction_ms = (time.perf_counter() - prediction_started) * 1000
        decode_started = time.perf_counter()
        self._last_inference_stats = {
            "model_invocations": 1,
            "inference_images": len(frames),
            "batch_size": len(frames),
            "model_predict_ms": prediction_ms,
            "postprocess_ms": (time.perf_counter() - decode_started) * 1000,
            "merge_ms": 0.0,
            "actual_tensor_shapes": [list(frame.shape) for frame in frames],
            "sum_tensor_pixels": int(sum(int(frame.shape[0] * frame.shape[1]) for frame in frames)),
            "padding_overhead_pixels": None,
            "workload_source": "measured_input_shapes",
        }
        batches: list[list[Detection]] = []
        for result in results:
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
                    for coords, score, class_id in zip(
                        xyxy, confidence, class_ids, strict=True
                    )
                ]
            )
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

    @property
    def config(self) -> DetectorConfig:
        """Resolved detector settings used by the scheduler/diagnostics."""
        return self._config

    def tile_regions(self, frame_width: int, frame_height: int) -> tuple[tuple[int, int, int, int], ...]:
        """Return detector tile geometry in source-image coordinates.

        The motion scheduler is deliberately restricted to this validated
        geometry.  It never invents crops which would change model behavior.
        """
        if not self._config.tiled_inference:
            return ()
        x_ranges = _tile_ranges(
            frame_width, self._config.tile_columns, self._config.tile_overlap
        )
        y_ranges = _tile_ranges(
            frame_height, self._config.tile_rows, self._config.tile_overlap
        )
        return tuple(
            (x1, y1, x2, y2)
            for y1, y2 in y_ranges
            for x1, x2 in x_ranges
        )

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

        height, width = frame.shape[:2]
        inputs: list[NDArray[np.uint8]] = []
        metadata: list[tuple[int, int, int, int] | None] = []
        if self._config.tile_include_full_frame:
            inputs.append(frame)
            metadata.append(None)

        for x1, y1, x2, y2 in self.tile_regions(width, height):
            inputs.append(frame[y1:y2, x1:x2])
            metadata.append((x1, y1, x2, y2))

        return self._merge_prediction_batches(
            frame, inputs, metadata
        )

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
        height, width = frame.shape[:2]
        prediction_batches = self._predict(inputs)
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
                if self._touches_internal_edge(
                    detection,
                    tile_width=tile_width,
                    tile_height=tile_height,
                    left_internal=tile_x1 > 0,
                    top_internal=tile_y1 > 0,
                    right_internal=tile_x2 < width,
                    bottom_internal=tile_y2 < height,
                ) and detection.confidence < max(self._config.confidence, 0.12):
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

        filtered = [
            detection
            for detection in candidates
            if not _in_ignore_region(
                detection,
                width=width,
                height=height,
                regions=self._config.ignore_regions,
            )
        ]
        merged = _merge_detections(
            filtered,
            iou_threshold=self._config.tile_merge_iou,
            max_detections=self._config.max_det,
            source_ids=[
                source_id
                for detection, source_id in zip(
                    candidates, candidate_sources, strict=True
                )
                if not _in_ignore_region(
                    detection,
                    width=width,
                    height=height,
                    regions=self._config.ignore_regions,
                )
            ],
        )
        self._last_inference_stats["merge_ms"] = (time.perf_counter() - merge_started) * 1000
        self._last_inference_stats["postprocess_ms"] = (
            float(self._last_inference_stats.get("postprocess_ms") or 0.0)
            + float(self._last_inference_stats["merge_ms"])
        )
        return merged
