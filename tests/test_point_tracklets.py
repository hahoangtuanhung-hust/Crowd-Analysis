from backend.app.analytics import PointTrackletManager
from backend.app.core.config import AnalyticsConfig, TrackerConfig, ZoneConfig
from backend.app.schemas import TrackedObject


def tracked(track_id: int, x: float, y: float, size: float = 4.0) -> TrackedObject:
    return TrackedObject(track_id, x - size / 2, y - size, x + size / 2, y, 0.9)


def manager(
    *,
    history: int = 4,
    minimum: int = 3,
    debounce: int = 2,
    track_buffer: int = 2,
) -> PointTrackletManager:
    analytics = AnalyticsConfig(
        trajectory_history_points=history,
        trajectory_smoothing_alpha=1.0,
        min_confirmed_points=minimum,
        movement_threshold_pixels=1.0,
        zone_debounce_points=debounce,
        max_active_tracks=4,
    )
    tracker = TrackerConfig(track_buffer=track_buffer)
    zones = (
        ZoneConfig(zone_id="a", name="Entrance", points=[(0, 0), (45, 0), (45, 100), (0, 100)]),
        ZoneConfig(zone_id="b", name="Exit", points=[(55, 0), (100, 0), (100, 100), (55, 100)]),
    )
    return PointTrackletManager(analytics, tracker, zones)


def test_small_far_person_uses_exact_bottom_center_point() -> None:
    tracklets = manager(minimum=1)
    update = tracklets.update([tracked(9, 1000.0, 220.0, size=2.0)], frame_id=0, timestamp=0.0)

    assert len(update.persist_points) == 1
    assert (update.persist_points[0].x, update.persist_points[0].y) == (1000.0, 220.0)


def test_multiple_crossing_tracks_remain_separate_and_memory_is_bounded() -> None:
    tracklets = manager(history=3, minimum=2)
    persisted = []
    for frame_id in range(6):
        update = tracklets.update(
            [tracked(1, 10 + frame_id * 4, 50), tracked(2, 34 - frame_id * 4, 50)],
            frame_id=frame_id,
            timestamp=frame_id / 10,
        )
        persisted.extend(update.persist_points)

    histories = tracklets.histories()
    assert set(histories) == {1, 2}
    assert all(len(points) == 3 for points in histories.values())
    assert tracklets.buffered_point_count == 6
    assert {point.track_id for point in persisted} == {1, 2}


def test_occluded_track_reappears_before_buffer_then_finishes_after_buffer() -> None:
    tracklets = manager(minimum=2, track_buffer=2)
    tracklets.update([tracked(5, 10, 50)], frame_id=0, timestamp=0.0)
    tracklets.update([], frame_id=1, timestamp=0.1)
    update = tracklets.update([tracked(5, 15, 50)], frame_id=2, timestamp=0.2)

    assert len(update.persist_points) == 2
    assert tracklets.active_tracklet_count == 1
    tracklets.update([], frame_id=3, timestamp=0.3)
    tracklets.update([], frame_id=4, timestamp=0.4)
    completed = tracklets.update([], frame_id=5, timestamp=0.5).completed
    assert len(completed) == 1
    assert completed[0].confirmed


def test_boundary_jitter_does_not_duplicate_first_to_last_zone_flow() -> None:
    tracklets = manager(minimum=2, debounce=2, track_buffer=1)
    positions = [10, 12, 60, 14, 62, 64]
    for frame_id, x in enumerate(positions):
        tracklets.update([tracked(3, x, 50)], frame_id=frame_id, timestamp=frame_id / 10)
    completed = tracklets.update([], frame_id=7, timestamp=0.7).completed

    assert completed[0].transition == ("a", "b")
    assert tracklets.zone_flows()[0]["unique_track_ids"] == 1

    for frame_id, x in enumerate((10, 12, 62, 64), start=9):
        tracklets.update([tracked(3, x, 50)], frame_id=frame_id, timestamp=frame_id / 10)
    tracklets.finalize_all()
    assert tracklets.zone_flows()[0]["unique_track_ids"] == 1


def test_short_track_is_discarded_and_normal_end_flushes_confirmed_track() -> None:
    tracklets = manager(minimum=3)
    first = tracklets.update([tracked(1, 10, 50)], frame_id=0, timestamp=0.0)
    assert first.persist_points == ()
    short = tracklets.finalize_all()
    assert not short[0].confirmed
    assert tracklets.discarded_short_tracklets == 1

    tracklets = manager(minimum=2)
    tracklets.update([tracked(2, 10, 50)], frame_id=0, timestamp=0.0)
    tracklets.update([tracked(2, 15, 50)], frame_id=1, timestamp=0.1)
    completed = tracklets.finalize_all()
    assert completed[0].confirmed
    assert tracklets.active_tracklet_count == 0
