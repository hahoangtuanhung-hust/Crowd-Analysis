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


def test_stationary_track_uses_longer_grace_period() -> None:
    tracker = ByteTrackTracker(
        TrackerConfig(
            lost_track_grace_frames=1,
            stationary_lost_track_grace_frames=3,
            stationary_speed_threshold=2.0,
        )
    )
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    detection = Detection(30, 40, 80, 180, 0.9)
    tracker.update([detection], frame)
    observed = tracker.update([detection], frame)

    predicted = [tracker.update([], frame) for _ in range(3)]
    expired = tracker.update([], frame)

    assert all(len(items) == 1 for items in predicted)
    assert all(items[0].track_id == observed[0].track_id for items in predicted)
    assert expired == []


def test_tracker_preserves_id_across_crossing_with_short_occlusion() -> None:
    tracker = ByteTrackTracker(
        TrackerConfig(
            association_mode="hybrid",
            max_center_distance_ratio=0.10,
            motion_cost_weight=0.55,
            motion_direction_penalty=0.35,
            fuse_score=False,
            lost_track_grace_frames=6,
        )
    )
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    track_ids: dict[int, list[int]] = {0: [], 1: []}

    for step in range(11):
        left = 24 + step * 11
        right = 166 - step * 11
        detections = []
        if step not in {5, 6}:
            detections.append(Detection(left, 80, left + 24, 170, 0.9))
        detections.append(Detection(right, 80, right + 24, 170, 0.9))
        tracks = tracker.update(detections, frame, frame_id=step)
        observed = [item for item in tracks if item.observed]
        if step == 0:
            assert len(observed) == 2
            by_x = sorted(observed, key=lambda item: item.x1)
            track_ids[0].append(by_x[0].track_id)
            track_ids[1].append(by_x[1].track_id)
            continue
        # The left-to-right and right-to-left tracks retain their side/order
        # through the occlusion; predicted (non-observed) output is allowed.
        if step not in {5, 6}:
            assert len(observed) == 2
            expected_left = left
            expected_right = right
            left_track = min(observed, key=lambda item: abs(item.x1 - expected_left))
            right_track = min(observed, key=lambda item: abs(item.x1 - expected_right))
            assert left_track.track_id != right_track.track_id
            track_ids[0].append(left_track.track_id)
            track_ids[1].append(right_track.track_id)

    assert len(set(track_ids[0])) == 1
    assert len(set(track_ids[1])) == 1
    assert track_ids[0][0] != track_ids[1][0]


def test_tracker_uses_source_frame_id_when_realtime_drops_frames() -> None:
    tracker = ByteTrackTracker(
        TrackerConfig(
            association_mode="hybrid",
            max_center_distance_ratio=0.10,
            motion_cost_weight=0.45,
            fuse_score=False,
        )
    )
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    first = tracker.update([Detection(20, 25, 32, 70, 0.9)], frame, frame_id=0)
    second = tracker.update([Detection(44, 25, 56, 70, 0.9)], frame, frame_id=3)

    assert len(first) == len(second) == 1
    assert first[0].track_id == second[0].track_id
