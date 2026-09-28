import numpy as np

from backend.app.core.config import DetectorConfig
from backend.app.inference.ultralytics_detector import (
    UltralyticsPersonDetector,
    _in_ignore_region,
    _merge_detections,
    _tile_ranges,
)
from backend.app.schemas import Detection


def test_tile_ranges_cover_axis_with_overlap() -> None:
    ranges = _tile_ranges(1280, 2, 0.2)

    assert ranges[0][0] == 0
    assert ranges[-1][1] == 1280
    assert ranges[0][1] > ranges[1][0]


def test_merge_detections_suppresses_tile_duplicates() -> None:
    detections = [
        Detection(0, 0, 20, 40, 0.9),
        Detection(4, 5, 16, 35, 0.8),
        Detection(50, 10, 70, 45, 0.7),
    ]

    merged = _merge_detections(detections, iou_threshold=0.45, max_detections=100)

    assert merged == [detections[0], detections[2]]


def test_merge_detections_keeps_overlapping_people_from_same_tile() -> None:
    detections = [
        Detection(100, 40, 140, 150, 0.92),
        Detection(106, 45, 135, 145, 0.88),
    ]

    merged = _merge_detections(
        detections,
        iou_threshold=0.45,
        max_detections=100,
        source_ids=[2, 2],
    )

    assert merged == detections


def test_merge_detections_still_suppresses_aligned_cross_tile_duplicate() -> None:
    detections = [
        Detection(100, 40, 140, 150, 0.92),
        Detection(102, 42, 138, 148, 0.88),
    ]

    merged = _merge_detections(
        detections,
        iou_threshold=0.45,
        max_detections=100,
        source_ids=[1, 2],
    )

    assert merged == [detections[0]]


def test_normalized_ignore_region_filters_screen_detection() -> None:
    detection = Detection(550, 50, 650, 150, 0.8)

    assert _in_ignore_region(
        detection,
        width=1280,
        height=720,
        regions=[(0.4, 0.05, 0.6, 0.25)],
    )


def test_fp32_prediction_does_not_pass_deprecated_half_flag() -> None:
    class FakeModel:
        def __init__(self) -> None:
            self.options: dict = {}

        def predict(self, **options):
            self.options = options
            return []

    detector = object.__new__(UltralyticsPersonDetector)
    detector._config = DetectorConfig(precision="fp32")
    detector._device = None
    detector._model = FakeModel()

    assert detector._predict([np.zeros((8, 8, 3), dtype=np.uint8)]) == []
    assert "half" not in detector._model.options


def test_fp16_prediction_passes_half_true() -> None:
    class FakeModel:
        def __init__(self) -> None:
            self.options: dict = {}

        def predict(self, **options):
            self.options = options
            return []

    detector = object.__new__(UltralyticsPersonDetector)
    detector._config = DetectorConfig(precision="fp16")
    detector._device = "cuda"
    detector._model = FakeModel()

    assert detector._predict([np.zeros((8, 8, 3), dtype=np.uint8)]) == []
    assert detector._model.options["half"] is True
