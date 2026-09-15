from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from backend.app.core.config import AnalyticsConfig
from backend.app.datasets import (
    DatasetAnalyticsRunner,
    GrandCentralDataset,
    GrandCentralPaths,
    NormalizedTrajectoryPoint,
)
from backend.app.datasets.evaluation import evaluate_predictions

FIXTURE = Path(__file__).parent / "fixtures" / "grand_central"


def paths(root: Path, annotations: Path | None = None) -> GrandCentralPaths:
    return GrandCentralPaths(
        root=root,
        video=root / "raw" / "video" / "grand_central.mp4",
        annotations=annotations or FIXTURE / "Annotation",
        homography=FIXTURE,
        processed=root / "processed",
        output=root / "outputs",
    )


def make_video(path: Path, *, frame_count: int = 4) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (100, 100)
    )
    assert writer.isOpened()
    for _ in range(frame_count):
        writer.write(np.zeros((100, 100, 3), dtype=np.uint8))
    writer.release()


def point(track_id: int, frame_id: int, x: float, y: float) -> NormalizedTrajectoryPoint:
    return NormalizedTrajectoryPoint(
        camera_id="grand-central-01",
        source="ground_truth",
        frame_id=frame_id,
        timestamp=frame_id / 25.0,
        track_id=track_id,
        x=x,
        y=y,
        x1=None,
        y1=None,
        x2=None,
        y2=None,
        foot_x=x,
        foot_y=y,
        world_x=None,
        world_y=None,
        confidence=None,
        zone_id=None,
    )


def test_real_format_parser_timestamp_schema_and_gap_grouping(tmp_path: Path) -> None:
    dataset = GrandCentralDataset(paths(tmp_path))
    raw = list(dataset.iter_annotations())
    normalized = list(dataset.load_trajectories())

    assert [(item.x, item.y, item.frame_id) for item in raw[:2]] == [
        (10.0, 20.0, 0),
        (20.0, 20.0, 20),
    ]
    assert normalized[1].timestamp == 0.8
    assert normalized[0].source == "ground_truth"
    assert normalized[0].foot_x == normalized[0].x
    assert normalized[0].x1 is None
    assert normalized[0].confidence is None
    assert normalized[2].track_id != normalized[1].track_id


def test_homography_load_transform_and_invalid_matrix(tmp_path: Path) -> None:
    dataset = GrandCentralDataset(paths(tmp_path))
    matrix = dataset.load_homography(required=True)
    assert matrix is not None
    assert dataset.transform_point(10, 20, matrix) == (12.0, 23.0)

    with pytest.raises(ValueError, match="shape"):
        dataset.validate_homography(np.eye(2))
    with pytest.raises(ValueError, match="singular"):
        dataset.validate_homography(np.zeros((3, 3)))


def test_empty_missing_corrupt_and_invalid_annotation_inputs(tmp_path: Path) -> None:
    missing = GrandCentralDataset(paths(tmp_path, tmp_path / "missing"))
    with pytest.raises(FileNotFoundError):
        list(missing.iter_annotations())

    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    assert list(GrandCentralDataset(paths(tmp_path, empty_dir)).iter_annotations()) == []

    corrupt_dir = tmp_path / "corrupt"
    corrupt_dir.mkdir()
    (corrupt_dir / "000001.txt").write_text("1 2", encoding="utf-8")
    with pytest.raises(ValueError, match="triples"):
        list(GrandCentralDataset(paths(tmp_path, corrupt_dir)).iter_annotations())

    (corrupt_dir / "000001.txt").unlink()
    (corrupt_dir / "track-a.txt").write_text("1 2 0", encoding="utf-8")
    with pytest.raises(ValueError, match="numeric"):
        list(GrandCentralDataset(paths(tmp_path, corrupt_dir)).iter_annotations())


def test_ground_truth_analytics_integration_and_empty_scene(tmp_path: Path) -> None:
    dataset_paths = paths(tmp_path)
    dataset_paths.create()
    make_video(dataset_paths.video)
    (dataset_paths.processed / "zones.json").write_text(
        json.dumps(
            {
                "coordinate_space": "pixel_space",
                "zones": [
                    {
                        "zone_id": "a",
                        "name": "A",
                        "points": [[0, 0], [45, 0], [45, 100], [0, 100]],
                    },
                    {
                        "zone_id": "b",
                        "name": "B",
                        "points": [[55, 0], [100, 0], [100, 100], [55, 100]],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    runner = DatasetAnalyticsRunner(
        GrandCentralDataset(dataset_paths),
        AnalyticsConfig(
            min_confirmed_points=3,
            movement_threshold_pixels=1,
            zone_debounce_points=1,
            grid_width=10,
            grid_height=10,
            path_grid_width=4,
            path_grid_height=2,
        ),
        coordinate_mode="pixel_space",
    )
    points = [point(1, 0, 10, 50), point(1, 20, 40, 50), point(1, 40, 80, 50)]
    result = runner.run(points, source="ground_truth", duration_label="1m", persist=False)

    assert result["source"] == "ground_truth"
    assert result["heatmap"]["occupancy_max"] > 0
    assert result["flow"]["sample_count"] > 0
    assert result["paths"][0]["source"] == "ground_truth"
    assert result["zone_flows"][0] == {
        "source": "ground_truth",
        "from_zone": "a",
        "to_zone": "b",
        "count": 1,
    }
    with pytest.raises(ValueError, match="empty"):
        runner.run([], source="ground_truth", duration_label="1m", persist=False)


def test_evaluation_handles_missing_prediction_frames(tmp_path: Path) -> None:
    dataset_paths = paths(tmp_path)
    dataset_paths.create()
    make_video(dataset_paths.video, frame_count=30)
    dataset = GrandCentralDataset(dataset_paths)
    result = evaluate_predictions(
        dataset,
        [point(1, 0, 10, 10), point(1, 20, 20, 10)],
        [],
        duration_label="1m",
        inference_interval=1,
        persist=False,
    )

    assert result["metrics"]["person_count_mae"] == 1.0
    assert result["metrics"]["detection_point_precision"] == 0.0
    assert result["metrics"]["hota"] is None
