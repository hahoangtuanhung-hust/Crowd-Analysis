from pathlib import Path

import pytest

from backend.app.artifacts import (
    POINT_TRACKING_ARTIFACTS,
    validate_point_tracking_artifacts,
)


def write_artifacts(directory: Path) -> None:
    for name in POINT_TRACKING_ARTIFACTS:
        (directory / name).write_bytes(b"artifact")


def test_point_tracking_artifact_contract_has_exact_six_names() -> None:
    assert POINT_TRACKING_ARTIFACTS == (
        "frame_metrics.csv",
        "heatmap.png",
        "path_map.png",
        "tracked_points.mp4",
        "trajectories.csv",
        "zone_flows.json",
    )


def test_artifact_validation_returns_contract_order(tmp_path: Path) -> None:
    write_artifacts(tmp_path)

    artifacts = validate_point_tracking_artifacts(tmp_path, reject_unexpected=True)

    assert tuple(artifacts) == POINT_TRACKING_ARTIFACTS


def test_artifact_validation_rejects_missing_empty_and_unexpected_files(
    tmp_path: Path,
) -> None:
    write_artifacts(tmp_path)
    (tmp_path / "heatmap.png").write_bytes(b"")
    (tmp_path / "trajectories.csv").unlink()
    (tmp_path / "extra.txt").write_text("extra", encoding="utf-8")

    with pytest.raises(RuntimeError) as error:
        validate_point_tracking_artifacts(tmp_path, reject_unexpected=True)

    message = str(error.value)
    assert "missing: trajectories.csv" in message
    assert "empty: heatmap.png" in message
    assert "unexpected: extra.txt" in message
