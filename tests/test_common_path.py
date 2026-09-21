from __future__ import annotations

from dataclasses import replace

from backend.app.analytics import CommonPathAnalyzer, SpatialTransformer
from backend.app.core.config import AnalyticsConfig, CommonPathConfig, ZoneConfig
from backend.app.schemas import CommonPath, TrackPoint


def analyzer(
    *,
    min_points: int = 2,
    min_distance: float = 0.0,
    min_duration: float = 0.0,
    min_unique: int = 1,
    confirmation: float = 1.0,
    cooling: float = 2.0,
    short_window: int = 3,
    long_window: int = 10,
    hysteresis: float = 0.0,
    confirmation_points: int = 1,
    zones: bool = False,
) -> CommonPathAnalyzer:
    common_path = CommonPathConfig(
        grid_columns=4,
        grid_rows=2,
        cell_hysteresis_ratio=hysteresis,
        cell_confirmation_points=confirmation_points,
        bucket_seconds=1,
        short_window_seconds=short_window,
        long_window_seconds=long_window,
        short_weight=0.7,
        long_weight=0.3,
        update_interval_seconds=1.0,
        flow_update_interval_seconds=1.0,
        min_unique_tracks=min_unique,
        switch_margin=0.2,
        confirmation_seconds=confirmation,
        cooling_seconds=cooling,
        min_path_edge_support=1,
        path_similarity_threshold=0.7,
        min_track_duration_seconds=min_duration,
        min_track_distance_pixels=min_distance,
        min_track_points=min_points,
        max_stationary_speed_pixels_second=0.0,
        max_step_pixels=200.0,
        track_lost_timeout_seconds=0.5,
        zone_min_inside_frames=1,
        zone_debounce_seconds=0.0,
        max_path_cells=32,
    )
    zone_configs = []
    if zones:
        zone_configs = [
            ZoneConfig(
                zone_id="a",
                name="A",
                zone_type="entry",
                points=[(0, 0), (24, 0), (24, 100), (0, 100)],
            ),
            ZoneConfig(
                zone_id="b",
                name="B",
                zone_type="exit",
                points=[(76, 0), (100, 0), (100, 100), (76, 100)],
            ),
        ]
    config = AnalyticsConfig(
        trajectory_min_point_distance_pixels=0.0,
        max_active_tracks=32,
        common_path=common_path,
        zones=zone_configs,
    )
    return CommonPathAnalyzer(config, SpatialTransformer.pixel(100, 100))


def point(track_id: int, frame_id: int, timestamp: float, x: float, y: float) -> TrackPoint:
    return TrackPoint(track_id, timestamp, frame_id, x, y, x, y, 0.9)


def feed_route(
    target: CommonPathAnalyzer,
    track_id: int,
    coordinates: list[tuple[float, float]],
    *,
    start: float = 0.0,
) -> None:
    for offset, (x, y) in enumerate(coordinates):
        timestamp = start + offset
        target.process_points(
            [point(track_id, offset, timestamp, x, y)],
            active_track_ids=[track_id],
            timestamp=timestamp,
        )


def feed_parallel_routes(
    target: CommonPathAnalyzer,
    track_ids: list[int],
    coordinates: list[tuple[float, float]],
    *,
    start: float,
) -> None:
    for offset, (x, y) in enumerate(coordinates):
        timestamp = start + offset
        target.process_points(
            [point(track_id, offset, timestamp, x, y) for track_id in track_ids],
            active_track_ids=track_ids,
            timestamp=timestamp,
        )


def activate_route(target: CommonPathAnalyzer) -> CommonPath:
    feed_route(target, 1, [(10, 25), (30, 25), (50, 25), (70, 25), (90, 25)])
    snapshot = target.process_points([], active_track_ids=[], timestamp=5.0)
    return next(path for path in snapshot.paths if path.state == "active")


def test_two_points_in_same_cell_do_not_create_edge() -> None:
    target = analyzer()
    feed_route(target, 1, [(5, 25), (20, 25)])

    assert target.flow_snapshot().edges == ()


def test_opposite_directions_are_distinct() -> None:
    target = analyzer()
    feed_route(target, 1, [(10, 25), (35, 25)])
    feed_route(target, 2, [(35, 75), (10, 75)], start=2.0)

    edges = {(edge.from_cell, edge.to_cell) for edge in target.flow_snapshot().edges}
    assert ((0, 0), (0, 1)) in edges
    assert ((1, 1), (1, 0)) in edges


def test_track_is_counted_once_per_directed_edge() -> None:
    target = analyzer()
    feed_route(target, 1, [(10, 25), (35, 25), (10, 25), (35, 25)])

    edge = next(
        item
        for item in target.flow_snapshot().edges
        if item.from_cell == (0, 0) and item.to_cell == (0, 1)
    )
    assert edge.unique_tracks_long == 1


def test_short_track_does_not_affect_flow() -> None:
    target = analyzer(min_points=3)
    feed_route(target, 1, [(10, 25), (35, 25)])
    target.finalize_all()

    assert target.flow_snapshot().edges == ()
    assert target.discarded_tracks == 1


def test_stationary_track_does_not_increase_popularity() -> None:
    target = analyzer(min_points=3)
    feed_route(target, 1, [(10, 25), (10.2, 25), (10.4, 25), (10.6, 25)])

    assert target.flow_snapshot().edges == ()


def test_boundary_jitter_does_not_create_false_transitions() -> None:
    target = analyzer(hysteresis=0.2, confirmation_points=2)
    feed_route(target, 1, [(49, 25), (51, 25), (49, 25), (51, 25), (49, 25)])

    assert target.flow_snapshot().edges == ()


def test_candidate_does_not_replace_active_immediately() -> None:
    target = analyzer(zones=True, confirmation=2.0)
    feed_route(target, 1, [(10, 25), (30, 25), (50, 25), (70, 25), (90, 25)])

    assert [path.state for path in target.snapshot().paths] == ["candidate"]


def test_strong_candidate_becomes_active_after_confirmation() -> None:
    target = analyzer(zones=True, confirmation=2.0)
    feed_route(target, 1, [(10, 25), (30, 25), (50, 25), (70, 25), (90, 25)])
    target.process_points([], active_track_ids=[], timestamp=5.0)
    snapshot = target.process_points([], active_track_ids=[], timestamp=6.0)

    assert any(path.state == "active" for path in snapshot.paths)


def test_sustained_stronger_route_switches_active_path() -> None:
    target = analyzer(
        zones=True,
        confirmation=1.0,
        short_window=3,
        long_window=8,
    )
    first = activate_route(target)
    feed_parallel_routes(
        target,
        [10, 11, 12],
        [(10, 75), (30, 75), (50, 75), (70, 75), (90, 75)],
        start=9.0,
    )
    snapshot = target.process_points([], active_track_ids=[], timestamp=14.0)
    active = next(path for path in snapshot.paths if path.state == "active")

    assert active.path_id != first.path_id
    assert target.common_path_switches == 1
    assert any(path.state == "cooling" for path in snapshot.paths)


def test_active_path_cools_before_retirement() -> None:
    target = analyzer(
        zones=True,
        confirmation=1.0,
        cooling=2.0,
        short_window=2,
        long_window=6,
    )
    activate_route(target)

    cooling = target.process_points([], active_track_ids=[], timestamp=11.0)
    retired = target.process_points([], active_track_ids=[], timestamp=13.0)

    assert any(path.state == "cooling" for path in cooling.paths)
    assert any(path.state == "retired" for path in retired.paths)


def test_similar_paths_merge_centerlines() -> None:
    target = analyzer()
    active = CommonPath(
        path_id="a__b__01",
        origin_zone="a",
        destination_zone="b",
        state="active",
        unique_tracks_short=2,
        unique_tracks_long=4,
        score=0.7,
        confidence=0.8,
        direction="A_TO_B",
        polyline=((0.0, 0.0), (10.0, 10.0)),
        updated_at=1.0,
    )
    blended = target._blend_polylines(
        active.polyline,
        ((0.0, 0.0), (12.0, 10.0)),
        0.2,
    )
    similarity = target.directed_edge_similarity(
        {((0, 0), (0, 1)), ((0, 1), (0, 2)), ((0, 2), (0, 3))},
        {((0, 0), (0, 1)), ((0, 1), (0, 2)), ((0, 2), (0, 3))},
    )

    assert similarity == 1.0
    assert blended[1][0] == 10.4
    assert replace(active, polyline=blended).path_id == active.path_id


def test_expired_window_data_is_removed() -> None:
    target = analyzer(short_window=2, long_window=4)
    feed_route(target, 1, [(10, 25), (35, 25)])
    target.process_points([], active_track_ids=[], timestamp=6.0)

    assert target.flow_snapshot().edges == ()


def test_bucket_ring_and_track_state_are_bounded() -> None:
    target = analyzer(short_window=3, long_window=10)
    for track_id in range(30):
        start = float(track_id)
        feed_route(target, track_id, [(10, 25), (35, 25)], start=start)

    assert target.bucket_count <= 12
    assert target.tracked_state_count <= 2


def test_empty_stream_and_shutdown_flush_do_not_error() -> None:
    target = analyzer()
    assert target.process_points([], active_track_ids=[], timestamp=0.0).paths == ()
    feed_route(target, 1, [(10, 25), (35, 25)])

    target.finalize_all()

    assert target.tracked_state_count == 0
    assert target.completed_tracks == 1
