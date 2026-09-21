from __future__ import annotations

from pathlib import Path

POINT_TRACKING_ARTIFACTS = (
    "common_path_map.png",
    "common_path_timeline.csv",
    "common_paths.json",
    "edge_flows.json",
    "frame_metrics.csv",
    "heatmap.png",
    "path_map.png",
    "realtime_point_common_path.mp4",
    "tracked_points.mp4",
    "trajectories.csv",
    "realtime_benchmark.csv",
    "zone_flows.json",
)


def point_tracking_artifact_paths(output_dir: str | Path) -> dict[str, Path]:
    directory = Path(output_dir)
    return {name: directory / name for name in POINT_TRACKING_ARTIFACTS}


def validate_point_tracking_artifacts(
    output_dir: str | Path,
    *,
    reject_unexpected: bool = False,
) -> dict[str, Path]:
    directory = Path(output_dir)
    artifacts = point_tracking_artifact_paths(directory)
    missing = [name for name, path in artifacts.items() if not path.is_file()]
    empty = [name for name, path in artifacts.items() if path.is_file() and path.stat().st_size == 0]
    unexpected = []
    if reject_unexpected and directory.is_dir():
        expected = set(artifacts)
        unexpected = sorted(
            path.name for path in directory.iterdir() if path.is_file() and path.name not in expected
        )

    problems = []
    if missing:
        problems.append(f"missing: {', '.join(missing)}")
    if empty:
        problems.append(f"empty: {', '.join(empty)}")
    if unexpected:
        problems.append(f"unexpected: {', '.join(unexpected)}")
    if problems:
        raise RuntimeError("Invalid point-tracking output (" + "; ".join(problems) + ")")
    return artifacts
