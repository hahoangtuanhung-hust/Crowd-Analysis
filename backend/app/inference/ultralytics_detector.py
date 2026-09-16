from __future__ import annotations

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


def _merge_detections(
    detections: list[Detection],
    *,
    iou_threshold: float,
    max_detections: int,
) -> list[Detection]:
    selected: list[Detection] = []
    for candidate in sorted(detections, key=lambda item: item.confidence, reverse=True):
        duplicate = False
        for existing in selected:
            if candidate.class_id != existing.class_id:
                continue
            iou, intersection_over_smaller = _intersection_ratios(candidate, existing)
            if iou >= iou_threshold or intersection_over_smaller >= 0.8:
                duplicate = True
                break
        if not duplicate:
            selected.append(candidate)
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

    def _predict(self, frames: list[NDArray[np.uint8]]) -> list[list[Detection]]:
        results = self._model.predict(
            source=frames,
            imgsz=self._config.imgsz,
            conf=self._config.confidence,
            iou=self._config.iou,
            max_det=self._config.max_det,
            classes=self._config.classes,
            device=self._device,
            quantize=16 if self._config.precision == "fp16" else 32,
            verbose=False,
        )
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

    def detect(self, frame: NDArray[np.uint8]) -> list[Detection]:
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

        height, width = frame.shape[:2]
        inputs: list[NDArray[np.uint8]] = []
        metadata: list[tuple[int, int, int, int] | None] = []
        if self._config.tile_include_full_frame:
            inputs.append(frame)
            metadata.append(None)

        x_ranges = _tile_ranges(
            width, self._config.tile_columns, self._config.tile_overlap
        )
        y_ranges = _tile_ranges(height, self._config.tile_rows, self._config.tile_overlap)
        for y1, y2 in y_ranges:
            for x1, x2 in x_ranges:
                inputs.append(frame[y1:y2, x1:x2])
                metadata.append((x1, y1, x2, y2))

        candidates: list[Detection] = []
        for tile_detections, tile in zip(self._predict(inputs), metadata, strict=True):
            if tile is None:
                candidates.extend(tile_detections)
                continue
            tile_x1, tile_y1, tile_x2, tile_y2 = tile
            tile_width = tile_x2 - tile_x1
            tile_height = tile_y2 - tile_y1
            for detection in tile_detections:
                if self._touches_internal_edge(
                    detection,
                    tile_width=tile_width,
                    tile_height=tile_height,
                    left_internal=tile_x1 > 0,
                    top_internal=tile_y1 > 0,
                    right_internal=tile_x2 < width,
                    bottom_internal=tile_y2 < height,
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
        return _merge_detections(
            filtered,
            iou_threshold=self._config.tile_merge_iou,
            max_detections=self._config.max_det,
        )
