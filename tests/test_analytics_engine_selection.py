import pytest

from backend.app.analytics.engine import AnalyticsEngine
from backend.app.analytics.spatial import SpatialTransformer
from backend.app.core.config import AnalyticsConfig, CommonPathConfig


@pytest.mark.parametrize(
    ("mode", "shadow_display", "ddcrp_enabled", "selected"),
    [
        ("directional_grid", "legacy", True, "directional"),
        ("shadow", "directional_grid", True, "directional"),
        ("legacy", "legacy", True, "ddcrp"),
        ("shadow", "legacy", True, "ddcrp"),
        ("legacy", "legacy", False, "legacy"),
    ],
)
def test_common_path_snapshot_respects_selected_engine_with_ddcrp(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    shadow_display: str,
    ddcrp_enabled: bool,
    selected: str,
) -> None:
    analytics = AnalyticsEngine(
        AnalyticsConfig(
            ddcrp_enabled=ddcrp_enabled,
            common_path=CommonPathConfig(engine=mode, shadow_display=shadow_display),
        ),
        SpatialTransformer.pixel(1280, 720),
        retain_entire=False,
    )
    snapshots = {name: object() for name in ("directional", "ddcrp", "legacy")}
    monkeypatch.setattr(analytics.directional_path, "snapshot", lambda: snapshots["directional"])
    monkeypatch.setattr(analytics.ddcrp, "snapshot", lambda: snapshots["ddcrp"])
    monkeypatch.setattr(analytics.common_path, "snapshot", lambda: snapshots["legacy"])

    assert analytics.common_path_snapshot() is snapshots[selected]
