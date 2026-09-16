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
    tracker = ByteTrackTracker(TrackerConfig(lost_track_grace_frames=0))
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    assert tracker.update([], frame) == []


def test_hybrid_association_keeps_id_when_small_person_moves_without_iou() -> None:
    tracker = ByteTrackTracker(
        TrackerConfig(
            association_mode="hybrid",
            max_center_distance_ratio=0.1,
            fuse_score=False,
        )
    )
    frame = np.zeros((240, 320, 3), dtype=np.uint8)

    first = tracker.update([Detection(20, 40, 28, 64, 0.9)], frame)
    second = tracker.update([Detection(36, 40, 44, 64, 0.9)], frame)

    assert len(first) == len(second) == 1
    assert first[0].track_id == second[0].track_id


def test_low_confidence_detection_recovers_existing_moving_track() -> None:
    tracker = ByteTrackTracker(
        TrackerConfig(
            association_mode="hybrid",
            max_center_distance_ratio=0.1,
            second_match_thresh=0.8,
            fuse_score=False,
        )
    )
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    first = tracker.update([Detection(20, 40, 28, 64, 0.9)], frame)
    second = tracker.update([Detection(36, 40, 44, 64, 0.08)], frame)

    assert len(first) == len(second) == 1
    assert first[0].track_id == second[0].track_id


def test_confirmed_stationary_track_survives_short_detection_gap() -> None:
    tracker = ByteTrackTracker(TrackerConfig(lost_track_grace_frames=2))
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    detection = Detection(30, 40, 80, 180, 0.9)
    observed = tracker.update([detection], frame)
    observed = tracker.update([detection], frame)

    predicted_once = tracker.update([], frame)
    predicted_twice = tracker.update([], frame)
    expired = tracker.update([], frame)

    assert len(observed) == len(predicted_once) == len(predicted_twice) == 1
    assert predicted_once[0].track_id == observed[0].track_id
    assert predicted_twice[0].track_id == observed[0].track_id
    assert expired == []


def test_single_frame_detection_is_not_carried_as_a_lost_track() -> None:
    tracker = ByteTrackTracker(TrackerConfig(lost_track_grace_frames=2))
    frame = np.zeros((240, 320, 3), dtype=np.uint8)

    assert tracker.update([Detection(30, 40, 80, 180, 0.9)], frame)
    assert tracker.update([], frame) == []
