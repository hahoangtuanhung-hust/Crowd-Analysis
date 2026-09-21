"""Synthetic event-time fixtures, not a video benchmark."""

from __future__ import annotations

import pytest

from backend.app.analytics.directional_grid import DirectionalGridEngine, GridTrackPoint
from backend.app.analytics.spatial import SpatialTransformer
from backend.app.core.config import DirectionalGridConfig, LocalCorridorConfig


def engine(**kwargs) -> DirectionalGridEngine:
    config = DirectionalGridConfig(columns=4, rows=2, direction_sample_seconds=.2,
        min_track_age_seconds=0, min_displacement_cell_fraction=.02,
        max_observation_gap_seconds=2., short_window_seconds=3,
        long_window_seconds=12, min_edge_unique_tracks=1,
        min_support_tracks=2, min_complete_tracks=2, cell_tolerance=0,
        update_interval_seconds=.5, confirmation_seconds=.5, min_path_cells=3,
        **kwargs)
    return DirectionalGridEngine(config, SpatialTransformer.pixel(400, 200))


def point(track: int, frame: int, time_s: float, x: float, y: float = 50.,
          epoch: str = "initial") -> GridTrackPoint:
    return GridTrackPoint("cam01", epoch, track, 0, frame, time_s, x, y)


def trace(target: DirectionalGridEngine, track: int, y: float = 50.,
          backwards: bool = False, offset: float = 0.) -> None:
    cells = [350., 250., 150., 50.] if backwards else [50., 150., 250., 350.]
    frame = 0
    for x in cells:
        for _ in range(6):
            target.update([point(track, frame, offset + frame * .1, x, y)],
                          offset + frame * .1)
            frame += 1


def test_east_south_west_north_diagonal_and_wraparound() -> None:
    target = engine()
    assert [target.direction_bin(*vector) for vector in
            ((1,0),(1,1),(0,1),(-1,1),(-1,0),(-1,-1),(0,-1),(1,-1))] == list(range(8))
    assert target.direction_bin(1,-.0001) == 0


def test_distinct_support_windows_and_opposing_modes() -> None:
    target = engine()
    trace(target, 1)
    trace(target, 2, backwards=True, offset=3.)
    edges = target.flow_snapshot().edges
    assert any(e.from_cell == (0,0) and e.to_cell == (0,1) for e in edges)
    assert any(e.from_cell == (0,1) and e.to_cell == (0,0) for e in edges)
    assert all(e.unique_tracks_long == 1 for e in edges)
    assert any(len(modes) > 1 for modes in target.histogram().values())
    target.tick(20.)
    assert not target.flow_snapshot().edges
    assert not target.histogram()


def test_stationary_jitter_and_gap_do_not_create_edges() -> None:
    target = engine()
    for frame in range(12):
        x = 50 + (frame % 2) * .2
        target.update([point(7, frame, frame*.1, x)], frame*.1)
    assert not target.flow_snapshot().edges
    target.update([point(7, 50, 5., 350.)], 5.)
    assert not target.flow_snapshot().edges
    assert len(target._tracks) == 2
    assert all(len(track.cells) == 1 for track in target._tracks.values())


def test_complete_track_required_and_no_confirmation_on_repeated_tick() -> None:
    target = engine()
    trace(target, 1)
    trace(target, 2, offset=3.)
    target.tick(5.5)
    assert not any(p.state == "active" for p in target.snapshot().paths)
    # Repeating the same evidence cannot advance confirmation.
    target.tick(5.5)
    assert not any(p.state == "active" for p in target.snapshot().paths)
    trace(target, 3, offset=5.6)
    target.tick(8.)
    active = [p for p in target.snapshot().paths if p.state == "active"]
    assert active and active[0].validated_complete_tracks >= 2
    path_id = active[0].path_id
    target.tick(8.7)
    assert target.snapshot().paths[0].path_id == path_id
    target.tick(30.)
    target.tick(31.)
    assert target.snapshot().paths[0].state == "cooling"
    target.tick(52.)
    assert target.snapshot().paths[0].state == "retired"
    target.tick(53.)
    assert not target.snapshot().paths


def test_dominant_direction_activates_top_supported_route_without_complete_track() -> None:
    target = engine(
        display_policy="dominant_direction",
        dominant_min_support_tracks=1,
        diagnostics_enabled=True,
    )
    trace(target, 1)
    active = [path for path in target.snapshot().paths if path.state == "active"]
    assert len(active) == 1
    assert active[0].unique_tracks_short == 1
    assert active[0].support_tracks == 1
    assert active[0].validated_complete_tracks < target.config.min_complete_tracks
    assert active[0].direction == "east"
    assert any(
        event["reason"] == "dominant_direction_support"
        for event in target.events
    )


def test_ordered_validation_rejects_disjoint_support_and_reverse() -> None:
    target = engine()
    trace(target, 1)
    trace(target, 2, backwards=True, offset=3.)
    assert target._validate(((0,0),(0,1),(0,2),(0,3)), 5.5)[1] == 1
    assert target._validate(((0,3),(0,2),(0,1),(0,0)), 5.5)[1] == 1


def test_challenger_must_be_stronger_and_sustained_before_switch() -> None:
    target = engine()
    trace(target, 1)
    trace(target, 2, offset=3.)
    trace(target, 3, offset=5.6)
    target.tick(8.)
    original = next(path for path in target.snapshot().paths if path.state == "active")
    trace(target, 10, y=150., offset=9.)
    assert next(path for path in target.snapshot().paths if path.state == "active").path_id == original.path_id
    for index, track_id in enumerate(range(11, 16)):
        trace(target, track_id, y=150., offset=12. + index * 3.)
    target.tick(28.)
    switched = next(path for path in target.snapshot().paths if path.state == "active")
    assert switched.path_id != original.path_id
    assert sum(event["event"] == "switch" for event in target.events) == 1


def test_clock_epoch_and_bounded_history() -> None:
    target = engine()
    trace(target, 1)
    with pytest.raises(ValueError, match="monotonic"):
        target.tick(-1.)
    with pytest.raises(ValueError, match="epoch"):
        target.update([point(3, 90, 3., 50., epoch="wrong")], 3.)
    target.reset("next")
    assert target.snapshot().paths == ()
    assert target.retained_points == 0
    with pytest.raises(ValueError, match="order"):
        target.update([point(2, 3, 1., 50., epoch="next"),
                       point(2, 2, 1., 50., epoch="next")], 1.)


def test_track_and_point_caps_are_enforced() -> None:
    target = engine(max_tracks=2, max_points_per_track=3)
    for track_id in range(3):
        for frame in range(6):
            time_s = track_id * 2 + frame * .1
            target.update([point(track_id, frame, time_s, 40. + frame)], time_s)
    assert len(target._tracks) <= 2
    assert all(len(track.points) <= 3 for track in target._tracks.values())
    assert target.overflow > 0


def test_interpolation_does_not_cross_invalid_cell() -> None:
    config = DirectionalGridConfig(
        columns=4, rows=2, min_track_age_seconds=0,
        min_displacement_cell_fraction=.01, max_observation_gap_seconds=2,
        min_edge_unique_tracks=1,
    )
    target = DirectionalGridEngine(
        config, SpatialTransformer.pixel(400, 200),
        valid_cells={(0, 0), (0, 2)},
    )
    for frame, x in enumerate((50., 50., 250., 250., 250.)):
        target.update([point(1, frame, frame * .1, x)], frame * .1)
    assert target.flow_snapshot().edges == ()


def test_replay_speed_does_not_change_event_time_result() -> None:
    first, second = engine(), engine()
    schedule = [(1, 0.), (2, 3.), (3, 5.6)]
    for target in (first, second):
        for track_id, offset in schedule:
            trace(target, track_id, offset=offset)
        target.tick(8.)
    assert first.snapshot() == second.snapshot()
    assert first.events == second.events


def test_local_complete_uses_ordered_gates_without_weakening_global_od() -> None:
    corridor = LocalCorridorConfig(
        roi=[(0., 0.), (1., 0.), (1., 1.), (0., 1.)],
        source_gate=[(.25, 0.), (.5, 0.), (.5, 1.), (.25, 1.)],
        target_gate=[(.5, 0.), (.75, 0.), (.75, 1.), (.5, 1.)],
    )
    target = engine(route_scope="local_corridor", local_corridor=corridor)
    trace(target, 1)
    route = ((0, 1), (0, 2))
    local = target._validate_detailed(route, 2.3, {(0, 1)}, {(0, 2)},
                                      scope="local_corridor")
    global_od = target._validate_detailed(route, 2.3, {(0, 1)}, {(0, 2)},
                                          scope="global_od")
    assert local["complete_tracks"] == 1
    assert global_od["complete_tracks"] == 0


def test_diagnostics_distinguish_pending_and_use_stable_candidate_identity() -> None:
    target = engine(diagnostics_enabled=True)
    trace(target, 1)
    trace(target, 2, offset=3.)
    trace(target, 3, offset=5.6)
    pending = [item for item in target.candidate_evaluations
               if item["decision"] == "pending"]
    assert pending
    assert all(item["primary_reason"] == "HYSTERESIS_CONFIRMING" for item in pending)
    assert not any(item["decision"] == "rejected" and
                   item["primary_reason"] == "HYSTERESIS_CONFIRMING"
                   for item in target.candidate_evaluations)
    same_route = [item for item in target.candidate_evaluations
                  if item["cell_sequence"] == pending[0]["cell_sequence"]]
    assert len({item["candidate_id"] for item in same_route}) == 1
    assert len({item["evaluation_id"] for item in same_route}) == len(same_route)
    assert sum(target.candidate_decision_counts.values()) == len(target.candidate_evaluations)
    assert any(item["warmup_state"] == "short_ready" for item in target.graph_diagnostics)


def test_disconnected_graph_reason_precedes_search_budget() -> None:
    target = engine(diagnostics_enabled=True, beam_width=1, max_path_cells=3)
    target._edges[((0, 0), (0, 1))][("cam01", "initial", 1, 0)] = 0.
    target._endpoint_pairs = lambda: [(
        "forward", "source", "target", {(0, 0)}, {(0, 3)}
    )]
    target.tick(0.)
    disconnected = [item for item in target.graph_diagnostics
                    if item["start_seeds"] and not item["connected_pairs"]]
    assert disconnected
    assert all(item["primary_reason"] == "NO_CONNECTED_PATH" for item in disconnected)
