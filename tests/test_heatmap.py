import numpy as np

from backend.app.analytics import HeatmapAnalyzer, SpatialTransformer
from backend.app.core.config import AnalyticsConfig
from backend.app.schemas import TrackPoint


def point(track_id: int, frame_id: int, timestamp: float, x: float, y: float) -> TrackPoint:
    return TrackPoint(track_id, timestamp, frame_id, x, y, x, y, 0.9)


def analyzer(*, retain_entire: bool = True, **overrides: object) -> HeatmapAnalyzer:
    config = AnalyticsConfig(
        grid_width=10,
        grid_height=10,
        movement_threshold_pixels=2.0,
        gaussian_sigma=0,
        **overrides,
    )
    return HeatmapAnalyzer(config, SpatialTransformer.pixel(100, 100), retain_entire=retain_entire)


def test_stationary_track_affects_occupancy_but_not_movement() -> None:
    heatmaps = analyzer()
    heatmaps.observe(point(1, 0, 0.0, 50, 50))
    heatmaps.observe(point(1, 1, 1.0, 50, 50))

    snapshot = heatmaps.snapshot("entire")
    assert float(snapshot.occupancy.sum()) == 1.0
    assert float(snapshot.movement.sum()) == 0.0


def test_moving_track_affects_both_maps_and_rasterizes_segment() -> None:
    heatmaps = analyzer()
    heatmaps.observe(point(1, 0, 0.0, 10, 50))
    heatmaps.observe(point(1, 1, 1.0, 90, 50))

    snapshot = heatmaps.snapshot("entire")
    assert float(snapshot.occupancy.sum()) == 1.0
    assert np.isclose(float(snapshot.movement.sum()), 80.0)
    assert np.count_nonzero(snapshot.movement) >= 8


def test_live_entire_is_bounded_to_retention() -> None:
    heatmaps = analyzer(retain_entire=False, live_retention_seconds=10)
    heatmaps.observe(point(1, 0, 0.0, 10, 10))
    heatmaps.observe(point(1, 1, 1.0, 20, 10))
    heatmaps.observe(point(2, 2, 20.0, 10, 20))
    heatmaps.observe(point(2, 3, 21.0, 20, 20))

    snapshot = heatmaps.snapshot("entire")
    assert heatmaps.bucket_count == 1
    assert float(snapshot.occupancy.sum()) == 1.0
    assert np.isclose(float(snapshot.movement.sum()), 10.0)


def test_large_gap_does_not_create_movement_teleport() -> None:
    heatmaps = analyzer(max_point_gap_seconds=2.0)
    heatmaps.observe(point(1, 0, 0.0, 10, 10))
    heatmaps.observe(point(1, 1, 5.0, 90, 90))

    snapshot = heatmaps.snapshot("entire")
    assert float(snapshot.occupancy.sum()) == 2.0
    assert float(snapshot.movement.sum()) == 0.0


def test_ground_plane_homography_maps_corners() -> None:
    transformer = SpatialTransformer.ground_plane(
        [(10, 10), (90, 10), (90, 90), (10, 90)],
        width=8,
        height=4,
    )
    x, y = transformer.transform(90, 90)

    assert transformer.mode == "ground"
    assert np.allclose([x, y], [8, 4])
    assert transformer.grid_cell(x, y, grid_width=8, grid_height=4) == (3, 7)
