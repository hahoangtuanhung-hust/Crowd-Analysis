from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from backend.app.core.config import DetectorConfig
from backend.app.schemas import Detection


class UltralyticsPersonDetector:
    """Person-only detector with a small, backend-neutral output contract."""

    def __init__(self, config: DetectorConfig) -> None:
        from ultralytics import YOLO

        self._config = config
        self._device = None if config.device == "auto" else config.device
        self._model: Any = YOLO(config.model)

    def detect(self, frame: NDArray[np.uint8]) -> list[Detection]:
        result = self._model.predict(
            source=frame,
            imgsz=self._config.imgsz,
            conf=self._config.confidence,
            iou=self._config.iou,
            max_det=self._config.max_det,
            classes=self._config.classes,
            device=self._device,
            quantize=16 if self._config.precision == "fp16" else 32,
            verbose=False,
        )[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.detach().cpu().numpy()
        confidence = boxes.conf.detach().cpu().numpy()
        class_ids = boxes.cls.detach().cpu().numpy().astype(np.int32)
        return [
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
