import pytest

from backend.app.analytics.ddcrp import DDCRPClustering
from backend.app.core.config import AnalyticsConfig
from backend.app.schemas import TrackPoint, Trajectory


def _make_trajectory(track_id: int, points: list[tuple[float, float]], start_time: float = 0.0) -> Trajectory:
    track_points = tuple(
        TrackPoint(
            track_id=track_id,
            timestamp=start_time + i * 0.1,
            frame_id=i,
            x=pt[0],
            y=pt[1],
            raw_x=pt[0],
            raw_y=pt[1],
            confidence=0.9,
        )
        for i, pt in enumerate(points)
    )
    return Trajectory(
        track_id=track_id,
        points=track_points,
        age=len(points),
        confirmed=True,
        last_seen_timestamp=start_time + (len(points) - 1) * 0.1,
    )


def test_ddcrp_clusters_moving_pathways() -> None:
    config = AnalyticsConfig(min_confirmed_points=3)
    ddcrp = DDCRPClustering(config, alpha=0.4, spatial_scale=80.0)

    # Cluster 1: Moving Eastward along y=100
    t1 = _make_trajectory(1, [(10.0, 100.0), (30.0, 100.0), (50.0, 100.0), (70.0, 100.0)])
    t2 = _make_trajectory(2, [(12.0, 105.0), (32.0, 105.0), (52.0, 105.0), (72.0, 105.0)])

    # Cluster 2: Moving Southward along x=300
    t3 = _make_trajectory(3, [(300.0, 20.0), (300.0, 50.0), (300.0, 80.0), (300.0, 110.0)])
    t4 = _make_trajectory(4, [(305.0, 22.0), (305.0, 52.0), (305.0, 82.0), (305.0, 112.0)])

    ddcrp.observe_all([t1, t2, t3, t4])
    paths = ddcrp.top_paths(limit=5)

    assert len(paths) >= 2
    assert any("ddcrp" in p.kind for p in paths)
    total_count = sum(p.count for p in paths)
    assert total_count == 4


def test_ddcrp_identifies_stationary_ticket_buyers() -> None:
    config = AnalyticsConfig(min_confirmed_points=3)
    ddcrp = DDCRPClustering(config, alpha=0.4, spatial_scale=60.0, stationary_threshold=15.0)

    # 3 people standing still buying tickets near (150, 400)
    t1 = _make_trajectory(10, [(150.0, 400.0), (150.5, 400.2), (150.2, 400.1), (150.4, 400.3)])
    t2 = _make_trajectory(11, [(152.0, 402.0), (152.1, 401.9), (152.3, 402.1), (152.0, 402.0)])
    t3 = _make_trajectory(12, [(148.0, 398.0), (148.2, 398.1), (148.1, 398.0), (148.3, 398.2)])

    ddcrp.observe_all([t1, t2, t3])
    paths = ddcrp.top_paths(limit=5)

    assert len(paths) >= 1
    stationary_path = paths[0]
    assert stationary_path.kind == "ddcrp_stationary"
    assert "Quầy vé" in stationary_path.label or "Xếp hàng" in stationary_path.label
    assert stationary_path.count == 3
