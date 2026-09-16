from backend.app.inference.ultralytics_detector import (
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


def test_normalized_ignore_region_filters_screen_detection() -> None:
    detection = Detection(550, 50, 650, 150, 0.8)

    assert _in_ignore_region(
        detection,
        width=1280,
        height=720,
        regions=[(0.4, 0.05, 0.6, 0.25)],
    )
