from backend.app.analytics import SpatialTransformer, ZoneAnalyzer
from backend.app.core.config import AnalyticsConfig, ZoneConfig
from backend.app.schemas import TrackPoint, Trajectory


def make_trajectory(track_id: int, coordinates: list[tuple[float, float]]) -> Trajectory:
    points = tuple(
        TrackPoint(track_id, float(i), i, x, y, x, y, 0.9) for i, (x, y) in enumerate(coordinates)
    )
    return Trajectory(track_id, points, len(points), True, points[-1].timestamp)


def config(debounce: int = 1) -> AnalyticsConfig:
    return AnalyticsConfig(
        zone_debounce_points=debounce,
        zones=[
            ZoneConfig(zone_id="a", name="Entrance", points=[(0, 0), (45, 0), (45, 100), (0, 100)]),
            ZoneConfig(
                zone_id="b", name="Lobby", points=[(55, 0), (100, 0), (100, 100), (55, 100)]
            ),
        ],
    )


def test_zone_transition_counts_entries_exits_dwell_and_flow() -> None:
    analyzer = ZoneAnalyzer(config(), SpatialTransformer.pixel(100, 100), retain_entire=True)
    analyzer.process([make_trajectory(1, [(10, 50), (20, 50), (70, 50)])])

    snapshot = analyzer.snapshot()
    metrics = {item.zone_id: item for item in snapshot.zones}
    assert metrics["a"].entry_count == 1
    assert metrics["a"].exit_count == 1
    assert metrics["a"].average_dwell_seconds == 2.0
    assert metrics["b"].entry_count == 1
    assert metrics["b"].current_people == 1
    assert snapshot.flows[0].count == 1
    assert analyzer.top_paths()[0].label == "Entrance -> Lobby"


def test_zone_debounce_rejects_single_frame_boundary_jitter() -> None:
    analyzer = ZoneAnalyzer(
        config(debounce=2), SpatialTransformer.pixel(100, 100), retain_entire=True
    )
    analyzer.process([make_trajectory(1, [(10, 50), (20, 50), (70, 50), (20, 50)])])

    metrics = {item.zone_id: item for item in analyzer.snapshot().zones}
    assert metrics["a"].entry_count == 1
    assert metrics["a"].exit_count == 0
    assert metrics["a"].current_people == 1
    assert metrics["b"].entry_count == 0
    assert analyzer.snapshot().flows == ()


def test_same_track_transition_is_counted_only_once() -> None:
    analyzer = ZoneAnalyzer(config(), SpatialTransformer.pixel(100, 100), retain_entire=True)
    analyzer.process(
        [make_trajectory(1, [(10, 50), (70, 50), (10, 50), (70, 50)])]
    )

    flows = {(item.from_zone, item.to_zone): item.count for item in analyzer.snapshot().flows}
    assert flows[("a", "b")] == 1
    assert flows[("b", "a")] == 1
