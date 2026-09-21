"""Synthetic labeled tests for current-track dominant live flow."""

from __future__ import annotations

from backend.app.analytics.dominant_live_flow import DominantLiveFlowEngine
from backend.app.analytics.directional_grid import GridTrackPoint
from backend.app.analytics.spatial import SpatialTransformer
from backend.app.core.config import DominantLiveFlowConfig


def make_engine(**updates) -> DominantLiveFlowEngine:
    values = dict(
        mode="dominant_live_flow", columns=10, rows=6,
        direction_min_seconds=0.3, direction_max_seconds=0.8,
        direction_smoothing_alpha=1.0,
        min_displacement_cell_fraction=0.05,
        observation_timeout_seconds=0.6, history_seconds=3.0,
        evaluation_interval_seconds=0.5, min_active_tracks=3,
        confirmation_seconds=2.0, challenger_margin_tracks=1,
        stale_seconds=1.0,
    )
    values.update(updates)
    config = DominantLiveFlowConfig(**values)
    return DominantLiveFlowEngine(
        config, SpatialTransformer.pixel(1000, 600),
        camera_id="cam", stream_epoch="epoch-a",
    )


def frame_points(
    frame: int,
    time_s: float,
    east: int = 0,
    west: int = 0,
    stationary: int = 0,
    *,
    separated_east: int = 0,
) -> list[GridTrackPoint]:
    points = []
    for index in range(east):
        points.append(GridTrackPoint(
            "cam", "epoch-a", 100 + index, 0, frame, time_s,
            200 + time_s * 45 + index * 4, 250 + index * 3,
        ))
    for index in range(west):
        points.append(GridTrackPoint(
            "cam", "epoch-a", 200 + index, 0, frame, time_s,
            800 - time_s * 45 - index * 4, 260 + index * 3,
        ))
    for index in range(stationary):
        points.append(GridTrackPoint(
            "cam", "epoch-a", 300 + index, 0, frame, time_s,
            500 + index * 3, 120,
        ))
    for index in range(separated_east):
        points.append(GridTrackPoint(
            "cam", "epoch-a", 400 + index, 0, frame, time_s,
            700 + time_s * 35 + index * 3, 500 + index * 2,
        ))
    return points


def advance(
    target: DominantLiveFlowEngine,
    start: float,
    end: float,
    *,
    east: int = 0,
    west: int = 0,
    stationary: int = 0,
    separated_east: int = 0,
) -> None:
    frame = round(start * 10)
    time_s = start
    while time_s <= end + 1e-9:
        target.update(
            frame_points(
                frame, time_s, east, west, stationary,
                separated_east=separated_east,
            ),
            time_s,
        )
        frame += 1
        time_s = round(time_s + 0.1, 6)


def test_first_dominant_flow_requires_current_tracks_and_confirmation() -> None:
    target = make_engine()
    advance(target, 0.0, 1.9, east=4, stationary=5)
    assert not any(path.state == "active" for path in target.snapshot().paths)
    assert target.status.state == "confirming"
    advance(target, 2.0, 2.6, east=4, stationary=5)
    active = next(path for path in target.snapshot().paths if path.state == "active")
    assert active.direction == "east"
    assert active.unique_tracks_short == 4
    assert target.status.moving_track_count == 4


def test_opposing_flows_do_not_cancel_and_tie_keeps_active() -> None:
    target = make_engine()
    advance(target, 0.0, 2.6, east=5, west=3)
    active = next(path for path in target.snapshot().paths if path.state == "active")
    assert active.direction == "east"
    advance(target, 2.7, 5.0, east=5, west=5)
    active = next(path for path in target.snapshot().paths if path.state == "active")
    assert active.direction == "east"
    assert not any(event["event"] == "switch" for event in target.events)


def test_challenger_must_lead_for_two_seconds_before_switch() -> None:
    target = make_engine()
    advance(target, 0.0, 2.6, east=5, west=2)
    advance(target, 2.7, 4.4, east=3, west=6)
    assert next(path for path in target.snapshot().paths if path.state == "active").direction == "east"
    assert target.status.challenger_direction == "west"
    advance(target, 4.5, 5.7, east=3, west=6)
    assert next(path for path in target.snapshot().paths if path.state == "active").direction == "west"
    assert sum(event["event"] == "switch" for event in target.events) == 1


def test_short_challenger_stationary_and_lost_tracks_do_not_inflate_count() -> None:
    target = make_engine()
    advance(target, 0.0, 2.6, east=4, stationary=8)
    advance(target, 2.7, 3.8, east=3, west=6, stationary=8)
    advance(target, 3.9, 5.0, east=4, stationary=8)
    active = next(path for path in target.snapshot().paths if path.state == "active")
    assert active.direction == "east"
    assert target.status.moving_track_count == 4
    target.update([], 5.5)
    target.update([], 6.1)
    target.update([], 6.6)
    assert not target.snapshot().paths


def test_same_direction_in_disconnected_regions_stays_separate() -> None:
    target = make_engine(confirmation_seconds=0.0)
    advance(target, 0.0, 1.0, east=3, separated_east=4)
    active = next(path for path in target.snapshot().paths if path.state == "active")
    assert active.unique_tracks_short == 4
    x_values = [point[0] for point in active.polyline]
    assert min(x_values) > 600


def test_repeated_snapshot_and_epoch_reset_cannot_confirm() -> None:
    target = make_engine()
    advance(target, 0.0, 0.8, east=4)
    for time_s in (1.3, 1.8, 2.3, 2.8):
        target.tick(time_s)
    assert not any(path.state == "active" for path in target.snapshot().paths)
    target.reset("epoch-b")
    assert target.snapshot().paths == ()
    assert target.retained_tracks == 0


def test_initial_candidate_survives_a_brief_change_in_largest_cluster() -> None:
    target = make_engine()
    for frame in range(31):
        time_s = frame / 10
        if 1.0 <= time_s < 1.5:
            east, west = 4, 5
        else:
            east, west = 5, 4
        target.update(frame_points(frame, time_s, east=east, west=west), time_s)

    active = next(path for path in target.snapshot().paths if path.state == "active")
    assert active.direction == "east"
