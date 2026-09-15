from backend.app.analytics import FlowAnalyzer, SpatialTransformer
from backend.app.core.config import AnalyticsConfig
from backend.app.schemas import TrackPoint, Trajectory


def trajectory(track_id: int, coordinates: list[tuple[float, float]]) -> Trajectory:
    points = tuple(
        TrackPoint(track_id, float(i), i, x, y, x, y, 0.9) for i, (x, y) in enumerate(coordinates)
    )
    return Trajectory(track_id, points, len(points), True, points[-1].timestamp)


def test_bidirectional_routes_remain_distinct_and_ranked() -> None:
    config = AnalyticsConfig(
        grid_width=10,
        grid_height=10,
        path_grid_width=4,
        path_grid_height=2,
        movement_threshold_pixels=1,
    )
    analyzer = FlowAnalyzer(config, SpatialTransformer.pixel(100, 100), retain_entire=True)
    analyzer.process(
        [
            trajectory(1, [(5, 25), (35, 25), (65, 25), (95, 25)]),
            trajectory(2, [(5, 25), (35, 25), (65, 25), (95, 25)]),
            trajectory(3, [(95, 75), (65, 75), (35, 75), (5, 75)]),
        ]
    )
    analyzer.finalize_all()

    paths = analyzer.top_paths()
    snapshot = analyzer.snapshot("entire")
    assert [path.count for path in paths] == [2, 1]
    assert paths[0].label == "R1C1 -> R1C2 -> R1C3 -> R1C4"
    assert paths[1].label == "R2C4 -> R2C3 -> R2C2 -> R2C1"
    assert paths[0].percentage == 66.67
    assert snapshot.dominant_direction == "east"


def test_jitter_does_not_create_direction_samples_or_path() -> None:
    config = AnalyticsConfig(movement_threshold_pixels=5)
    analyzer = FlowAnalyzer(config, SpatialTransformer.pixel(100, 100), retain_entire=True)
    analyzer.process([trajectory(1, [(50, 50), (51, 50), (49, 51), (50, 50)])])
    analyzer.finalize_all()

    snapshot = analyzer.snapshot("entire")
    assert int(snapshot.samples.sum()) == 0
    assert analyzer.top_paths() == ()


def test_abnormal_jump_does_not_create_flow() -> None:
    config = AnalyticsConfig(
        movement_threshold_pixels=1,
        max_movement_step_pixels=20,
    )
    analyzer = FlowAnalyzer(config, SpatialTransformer.pixel(100, 100), retain_entire=True)
    analyzer.process([trajectory(1, [(10, 10), (90, 90)])])
    analyzer.finalize_all()

    assert int(analyzer.snapshot("entire").samples.sum()) == 0
    assert analyzer.top_paths() == ()


def test_active_route_memory_is_bounded() -> None:
    config = AnalyticsConfig(max_active_tracks=2, movement_threshold_pixels=1)
    analyzer = FlowAnalyzer(config, SpatialTransformer.pixel(100, 100), retain_entire=False)
    for track_id in range(5):
        analyzer.observe(TrackPoint(track_id, track_id * 2.0, 0, 10, 10, 10, 10, 0.9))
        analyzer.observe(TrackPoint(track_id, track_id * 2.0 + 1, 1, 90, 10, 90, 10, 0.9))
    assert analyzer.active_route_count <= 2
