from backend.app.analytics import TrajectoryManager
from backend.app.core.config import AnalyticsConfig
from backend.app.schemas import TrackedObject


def tracked(track_id: int, x1: float, y1: float, x2: float, y2: float) -> TrackedObject:
    return TrackedObject(track_id, x1, y1, x2, y2, 0.9)


def test_trajectory_uses_bottom_center_and_ema() -> None:
    manager = TrajectoryManager(
        AnalyticsConfig(trajectory_smoothing_alpha=0.5, min_confirmed_points=2)
    )
    first = manager.update([tracked(7, 10, 20, 30, 60)], frame_id=1, timestamp=0.0)[0]
    second = manager.update([tracked(7, 20, 20, 40, 80)], frame_id=2, timestamp=0.1)[0]

    assert (first.points[-1].x, first.points[-1].y) == (20.0, 60)
    assert (second.points[-1].raw_x, second.points[-1].raw_y) == (30.0, 80)
    assert (second.points[-1].x, second.points[-1].y) == (25.0, 70.0)
    assert second.confirmed


def test_missing_detection_does_not_create_phantom_point() -> None:
    manager = TrajectoryManager(AnalyticsConfig(inactive_track_ttl_seconds=2.0))
    manager.update([tracked(1, 0, 0, 10, 20)], frame_id=0, timestamp=0.0)
    observed = manager.update([], frame_id=1, timestamp=1.0)

    assert observed == ()
    trajectory = manager.get(1)
    assert trajectory is not None
    assert len(trajectory.points) == 1


def test_history_and_track_cardinality_are_bounded() -> None:
    manager = TrajectoryManager(
        AnalyticsConfig(
            trajectory_history_points=3,
            max_active_tracks=2,
            trajectory_smoothing_alpha=1.0,
        )
    )
    for frame_id in range(6):
        manager.update(
            [tracked(1, frame_id, 0, frame_id + 10, 20)],
            frame_id=frame_id,
            timestamp=float(frame_id),
        )
    assert manager.get(1) is not None
    assert len(manager.get(1).points) == 3  # type: ignore[union-attr]

    manager.update([tracked(2, 0, 0, 10, 20)], frame_id=7, timestamp=7.0)
    manager.update([tracked(3, 0, 0, 10, 20)], frame_id=8, timestamp=8.0)
    assert manager.track_count == 2
    assert manager.get(1) is None


def test_stale_tracks_are_evicted_by_source_time() -> None:
    manager = TrajectoryManager(AnalyticsConfig(inactive_track_ttl_seconds=1.0))
    manager.update([tracked(9, 0, 0, 10, 20)], frame_id=0, timestamp=5.0)

    assert manager.evict_stale(6.0) == ()
    assert manager.evict_stale(6.01) == (9,)
    assert manager.track_count == 0
