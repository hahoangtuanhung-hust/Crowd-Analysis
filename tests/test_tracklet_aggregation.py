from __future__ import annotations

import numpy as np

from backend.app.analytics.directional_grid import GridTrackPoint
from backend.app.analytics.spatial import SpatialTransformer
from backend.app.analytics.tracklet_aggregation import TrackletAggregationEngine
from backend.app.core.config import TrackletAggregationConfig
from backend.app.schemas.analytics import CommonPath


def engine(**kwargs):
    cfg = TrackletAggregationConfig(confirmation_seconds=0, update_interval_seconds=0.1,
                                    min_step_fraction=0.0001, **kwargs)
    return TrackletAggregationEngine(cfg, SpatialTransformer.pixel(1000, 1000),
                                      camera_id="cam", stream_epoch="epoch")


def feed(analyzer, routes, start=0, count=10):
    for frame in range(start, start + count):
        points = [GridTrackPoint("cam", "epoch", key, 0, frame, frame * 0.2,
                                 *route[frame % len(route)])
                  for key, route in routes.items()]
        analyzer.update(points, frame * 0.2)
    return analyzer.snapshot().paths


def test_curved_overlap_deduplicates_tracks_and_keeps_id_on_k_change():
    a = [(100 + i * 65, 190 + i * i * 5) for i in range(10)]
    routes = {i: [(x + i, y + i) for x, y in a] for i in (1, 2, 3)}
    analyzer = engine()
    paths = feed(analyzer, routes)
    assert len(paths) == 1
    path = paths[0]
    assert path.support_tracks == 3
    assert len(path.polyline) > 3
    assert path.polyline[len(path.polyline) // 2][1] < (path.polyline[0][1] + path.polyline[-1][1]) / 2
    assert analyzer.set_max_paths(5) == 5
    assert analyzer.snapshot().paths[0].path_id == path.path_id
    assert analyzer.snapshot().paths[0].color == path.color


def test_common_path_is_a_dense_piecewise_linear_polyline():
    analyzer = engine(segment_length_fraction=0.02)
    route = [(80 + i * 70, 260 + i * i * 3) for i in range(12)]
    paths = feed(analyzer, {1: route, 2: route, 3: route})

    assert len(paths) == 1
    polyline = np.asarray(paths[0].polyline, dtype=np.float64) / 1000.0
    lengths = np.linalg.norm(np.diff(polyline, axis=0), axis=1)
    assert len(polyline) > 20
    assert np.all(lengths > 0)
    assert float(lengths.max()) <= 0.025
    # A turn must remain represented by multiple local vectors, not an
    # endpoint-only line.
    local_cosines = np.sum(
        (polyline[1:-1] - polyline[:-2]) * (polyline[2:] - polyline[1:-1]), axis=1
    )
    assert np.any(local_cosines < 0.001)


def test_scale_aware_displacement_keeps_small_far_field_motion():
    analyzer = engine(
        min_support_tracks=2,
        min_displacement_fraction=0.018,
        min_track_duration_seconds=0.3,
    )
    # The total displacement is below the global 1.8% floor but is valid for
    # a person far from the camera (top of the normalized image).
    route = [(100 + i * 4, 20 + i) for i in range(4)]
    paths = feed(analyzer, {1: route, 2: route}, count=4)

    assert len(paths) == 1
    assert paths[0].support_tracks == 2
    assert analyzer.rejections["VALID_TRACKLETS"] >= 2


def test_crossing_and_reverse_not_joined():
    across = [(100 + i * 65, 500) for i in range(10)]
    down = [(500, 100 + i * 65) for i in range(10)]
    routes = {1: across, 2: across, 3: across, 4: down, 5: down, 6: down,
              7: across[::-1], 8: across[::-1], 9: across[::-1]}
    paths = feed(engine(), routes)
    assert len(paths) == 3
    assert all(path.support_tracks == 3 for path in paths)
    assert all(abs(path.polyline[0][0] - path.polyline[-1][0]) < 20 or
               abs(path.polyline[0][1] - path.polyline[-1][1]) < 20 for path in paths)


def test_expiry_and_prediction_do_not_create_support():
    analyzer = engine(evidence_window_seconds=2, cooling_seconds=0.5)
    route = [(100 + i * 60, 200) for i in range(10)]
    feed(analyzer, {1: route, 2: route, 3: route})
    assert analyzer.snapshot().paths[0].support_tracks == 3
    for frame in range(10, 27):
        analyzer.update([GridTrackPoint("cam", "epoch", 4, 0, frame, frame * .2,
                                        300, 200, observed=False)], frame * .2)
    assert analyzer.snapshot().paths == ()


def test_cut_segments_remain_evidence_and_active_path_cools_before_retire():
    analyzer = engine(evidence_window_seconds=2, cooling_seconds=1)
    route = [(100 + i * 60, 200) for i in range(10)]
    feed(analyzer, {1: route, 2: route, 3: route})
    original = analyzer.snapshot().paths[0]
    analyzer.update([
        GridTrackPoint("cam", "epoch", key, 0, 10, 3.0, 900, 900)
        for key in (1, 2, 3)
    ], 3.0)
    assert len(analyzer._closed) == 3
    assert analyzer.snapshot().paths[0].path_id == original.path_id
    analyzer.update([], 4.0)
    assert analyzer.snapshot().paths[0].state == "cooling"
    analyzer.update([], 6.1)
    assert analyzer.snapshot().paths == ()


def test_sequential_short_tracklets_are_connected_into_one_route():
    cfg = TrackletAggregationConfig(
        confirmation_seconds=0,
        update_interval_seconds=0.1,
        min_step_fraction=0.0001,
        min_track_duration_seconds=0.3,
        max_link_gap_seconds=0.6,
    )
    analyzer = TrackletAggregationEngine(
        cfg, SpatialTransformer.pixel(1000, 1000), camera_id="cam", stream_epoch="epoch"
    )
    route = [(100 + i * 55, 220 + i * i * 2) for i in range(12)]
    for track_id, start in ((1, 0), (2, 4), (3, 8)):
        for offset in range(4):
            frame = start + offset
            x, y = route[frame]
            analyzer.update(
                [GridTrackPoint("cam", "epoch", track_id, 0, frame, frame * 0.2, x, y)],
                frame * 0.2,
            )

    paths = analyzer.snapshot().paths
    assert len(paths) == 1
    assert paths[0].support_tracks == 3
    assert paths[0].polyline[0][0] < 150
    assert paths[0].polyline[-1][0] > 650


def test_short_inconsistent_zigzag_tracklets_are_filtered_before_clustering():
    cfg = TrackletAggregationConfig(
        confirmation_seconds=0,
        update_interval_seconds=0.1,
        min_step_fraction=0.0001,
        min_support_tracks=2,
        min_track_duration_seconds=0.3,
    )
    analyzer = TrackletAggregationEngine(
        cfg, SpatialTransformer.pixel(1000, 1000), camera_id="cam", stream_epoch="epoch"
    )
    noisy = [(100, 100), (200, 100), (100, 100), (200, 100), (100, 100), (200, 100)]
    for frame in range(len(noisy)):
        points = [
            GridTrackPoint("cam", "epoch", track_id, 0, frame, frame * 0.2,
                           *noisy[frame])
            for track_id in (1, 2)
        ]
        analyzer.update(points, frame * 0.2)

    assert analyzer.snapshot().paths == ()
    assert analyzer.rejections["WRONG_DIRECTION"] + analyzer.rejections["ZIGZAG_STRETCH"] > 0


def test_active_path_is_kept_through_a_short_observation_gap():
    analyzer = engine(evidence_window_seconds=2.0, cooling_seconds=2.0)
    route = [(100 + i * 60, 200) for i in range(10)]
    feed(analyzer, {1: route, 2: route, 3: route})
    original = analyzer.snapshot().paths[0]

    short_gap = analyzer.update([], 2.2).paths
    assert short_gap[0].path_id == original.path_id
    assert short_gap[0].state == "active"

    long_gap = analyzer.update([], 4.3).paths
    assert long_gap[0].path_id == original.path_id
    assert long_gap[0].state == "cooling"


def test_route_memory_keeps_long_common_path_after_recent_evidence_expires():
    analyzer = engine(evidence_window_seconds=1.0, cooling_seconds=1.0, route_memory_seconds=10.0)
    route = [(100 + i * 60, 200 + i * 2) for i in range(10)]
    feed(analyzer, {1: route, 2: route, 3: route})
    original = analyzer.snapshot().paths[0]

    remembered = analyzer.update([], 4.0).paths
    assert remembered[0].path_id == original.path_id
    assert remembered[0].state == "cooling"
    assert remembered[0].polyline[0] == original.polyline[0]
    assert remembered[0].polyline[-1] == original.polyline[-1]

    assert analyzer.update([], 12.1).paths == ()


def test_identity_merge_extends_camera_entry_route_to_new_exit_segment():
    analyzer = engine()
    previous = CommonPath(
        "path-001", "tracklet", "common_path", "active", 3, 3, 1.0, 1.0,
        "FORWARD", tuple((float(x), 300.0) for x in range(100, 601, 100)), 1.0,
        revision=2,
    )
    extension = CommonPath(
        "", "tracklet", "common_path", "candidate", 3, 3, 1.0, 1.0,
        "FORWARD", tuple((float(x), 300.0) for x in range(620, 901, 70)), 2.0,
    )

    merged = analyzer._merge_identity_paths(previous, extension)

    assert merged[0][0] < 130.0
    assert merged[-1][0] > 850.0


def test_different_people_on_same_route_link_across_long_time_gap():
    analyzer = engine(
        min_support_tracks=2,
        min_track_duration_seconds=0.3,
        max_link_gap_seconds=0.5,
        evidence_window_seconds=20.0,
    )
    route = [(100 + index * 80, 260) for index in range(5)]

    for offset, point in enumerate(route):
        analyzer.update([
            GridTrackPoint("cam", "epoch", 1, 0, offset, offset * 0.2, *point)
        ], offset * 0.2)
    for offset, point in enumerate(route, start=25):
        timestamp = offset * 0.2
        analyzer.update([
            GridTrackPoint("cam", "epoch", 2, 0, offset, timestamp, *point)
        ], timestamp)

    paths = analyzer.finalize().paths

    assert len(paths) == 1
    assert paths[0].support_tracks == 2


def test_same_track_fragments_remain_time_gated_and_diagnostics_explain_support():
    analyzer = engine(
        min_support_tracks=2,
        min_track_duration_seconds=0.3,
        max_observation_gap_seconds=0.5,
        max_link_gap_seconds=0.5,
        evidence_window_seconds=20.0,
    )
    route = [(100 + index * 80, 260) for index in range(5)]

    for offset, point in enumerate(route):
        analyzer.update([
            GridTrackPoint("cam", "epoch", 1, 0, offset, offset * 0.2, *point)
        ], offset * 0.2)
    for offset, point in enumerate(route, start=25):
        timestamp = offset * 0.2
        analyzer.update([
            GridTrackPoint("cam", "epoch", 1, 1, offset, timestamp, *point)
        ], timestamp)

    assert analyzer.finalize().paths == ()
    decisions = analyzer.diagnostics()["candidate_decisions"]
    assert decisions
    assert all(item["decision_reason"] == "INSUFFICIENT_SUPPORT_IDS" for item in decisions[-2:])


def test_candidate_and_selected_path_support_diagnostics_are_bounded_and_explicit():
    analyzer = engine(min_support_tracks=2)
    route = [(100 + index * 60, 220) for index in range(10)]

    paths = feed(analyzer, {1: route, 2: route})

    assert len(paths) == 1
    diagnostics = analyzer.diagnostics()
    selected = [
        item for item in diagnostics["candidate_decisions"]
        if item["decision_reason"] == "SELECTED_CANDIDATE_POOL"
    ]
    assert selected
    assert selected[-1]["score_components"]["support_tracks"] == 2
    assert 0.0 <= selected[-1]["evidence_coverage"] <= 1.0
    assert diagnostics["path_support"][-1]["path_id"] == paths[0].path_id
    assert diagnostics["path_support"][-1]["support_unit"] == "temporary_track_id"


def test_duplicate_points_do_not_inflate_segment_or_support():
    analyzer = engine()
    point = GridTrackPoint("cam", "epoch", 1, 7, 0, 0.0, 100, 200)
    analyzer.update([point], 0.0)
    analyzer.update([point], 0.1)
    assert analyzer.rejections["DUPLICATE_OR_LATE_POINT"] == 1
    assert analyzer.buffered_segment_count == 1
    assert len(analyzer._segments[('cam', 'epoch', 1)].points) == 1


def test_segment_id_change_closes_generation_without_reusing_old_points():
    analyzer = engine()
    for frame, segment_id in ((0, 1), (1, 1), (2, 2), (3, 2)):
        analyzer.update([
            GridTrackPoint("cam", "epoch", 1, segment_id, frame, frame * 0.2,
                           100 + frame * 30, 200)
        ], frame * 0.2)
    assert len(analyzer._closed) == 1
    assert analyzer._segments[('cam', 'epoch', 1)].segment_id == 2
    assert analyzer._segments[('cam', 'epoch', 1)].number == 2


def test_second_compute_reuses_unchanged_segment_signature():
    analyzer = engine()
    route = [(100 + index * 70, 220) for index in range(8)]
    for frame, point in enumerate(route[:5]):
        analyzer.update([
            GridTrackPoint("cam", "epoch", 1, 0, frame, frame * 0.2, *point)
        ], frame * 0.2)
    # The update interval already computed the active signature. A later
    # compute without a new point should reuse it.
    analyzer.finalize(2.0)
    first_count = analyzer.last_compute_new_segments
    analyzer.finalize(2.1)
    assert first_count == 0
    assert analyzer.last_compute_new_segments == 0
