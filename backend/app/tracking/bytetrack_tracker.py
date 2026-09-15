from __future__ import annotations

from collections.abc import Sequence
from types import SimpleNamespace

import numpy as np
from numpy.typing import NDArray

from backend.app.core.config import TrackerConfig
from backend.app.schemas import Detection, TrackedObject


class _DetectionResults:
    """Minimal Results-like object expected by Ultralytics tracker classes."""

    def __init__(self, xyxy: NDArray, confidence: NDArray, class_ids: NDArray) -> None:
        self.xyxy = np.asarray(xyxy, dtype=np.float32).reshape(-1, 4)
        self.conf = np.asarray(confidence, dtype=np.float32).reshape(-1)
        self.cls = np.asarray(class_ids, dtype=np.float32).reshape(-1)

    @property
    def xywh(self) -> NDArray:
        result = self.xyxy.copy()
        result[:, 0] = (self.xyxy[:, 0] + self.xyxy[:, 2]) / 2.0
        result[:, 1] = (self.xyxy[:, 1] + self.xyxy[:, 3]) / 2.0
        result[:, 2] = self.xyxy[:, 2] - self.xyxy[:, 0]
        result[:, 3] = self.xyxy[:, 3] - self.xyxy[:, 1]
        return result

    def __getitem__(self, index: object) -> _DetectionResults:
        return _DetectionResults(self.xyxy[index], self.conf[index], self.cls[index])

    def __len__(self) -> int:
        return len(self.conf)


class ByteTrackTracker:
    def __init__(self, config: TrackerConfig) -> None:
        self._config = config
        self._tracker = self._create_tracker()

    def _create_tracker(self):
        from ultralytics.trackers.byte_tracker import BYTETracker

        return BYTETracker(SimpleNamespace(**self._config.model_dump()))

    def reset(self) -> None:
        self._tracker = self._create_tracker()

    def update(
        self,
        detections: Sequence[Detection],
        frame: NDArray[np.uint8],
    ) -> list[TrackedObject]:
        results = _DetectionResults(
            xyxy=np.asarray([item.xyxy for item in detections], dtype=np.float32),
            confidence=np.asarray([item.confidence for item in detections], dtype=np.float32),
            class_ids=np.asarray([item.class_id for item in detections], dtype=np.float32),
        )
        tracked = self._tracker.update(results, img=frame)
        if tracked.size == 0:
            return []

        return [
            TrackedObject(
                x1=float(row[0]),
                y1=float(row[1]),
                x2=float(row[2]),
                y2=float(row[3]),
                track_id=int(row[4]),
                confidence=float(row[5]),
                class_id=int(row[6]),
            )
            for row in tracked
        ]
