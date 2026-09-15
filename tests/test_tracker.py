import numpy as np

from backend.app.core.config import TrackerConfig
from backend.app.schemas import Detection
from backend.app.tracking import ByteTrackTracker


def test_bytetrack_assigns_stable_id_to_moving_detection() -> None:
    tracker = ByteTrackTracker(TrackerConfig())
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    ids: list[int] = []

    for step in range(5):
        detections = [Detection(30 + step * 3, 40, 80 + step * 3, 180, 0.9)]
        tracks = tracker.update(detections, frame)
        ids.extend(track.track_id for track in tracks)

    assert len(ids) >= 3
    assert len(set(ids)) == 1


def test_bytetrack_handles_empty_detections() -> None:
    tracker = ByteTrackTracker(TrackerConfig())
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    assert tracker.update([], frame) == []
