from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.core.config import load_config
from backend.app.datasets import (
    DatasetAnalyticsRunner,
    GrandCentralDataset,
    GrandCentralPaths,
    NormalizedTrajectoryPoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Grand Central offline analytics")
    parser.add_argument("--root", type=Path, help="Override GC_DATASET_ROOT")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--source", choices=("ground_truth", "prediction"), default="ground_truth")
    parser.add_argument(
        "--coordinate-mode",
        choices=("pixel_space", "ground_plane"),
        default="pixel_space",
    )
    parser.add_argument("--duration", choices=("1m", "5m", "all"), default="all")
    return parser.parse_args()


def load_points(
    path: Path, *, source: str, duration_seconds: float | None
) -> list[NormalizedTrajectoryPoint]:
    filters: list[tuple[str, str, object]] = [("source", "=", source)]
    if duration_seconds is not None:
        filters.append(("timestamp", "<=", duration_seconds))
    table = pq.read_table(path, filters=filters)
    if table.num_rows:
        table = pc.take(table, pc.sort_indices(table, sort_keys=[("timestamp", "ascending"), ("track_id", "ascending")]))
    return [NormalizedTrajectoryPoint(**row) for row in table.to_pylist()]


def main() -> int:
    args = parse_args()
    paths = GrandCentralPaths.from_env(args.root)
    dataset = GrandCentralDataset(paths)
    if args.source == "ground_truth":
        trajectory_path = paths.processed / "trajectories.parquet"
    else:
        trajectory_path = paths.output / "tracking" / "prediction_trajectories.parquet"
    if not trajectory_path.is_file():
        raise FileNotFoundError(f"Normalized trajectories do not exist: {trajectory_path}")
    video_duration = float(dataset.video_metadata()["duration_seconds"])
    requested = {"1m": 60.0, "5m": 300.0, "all": video_duration}[args.duration]
    duration_seconds = min(requested, video_duration)
    points = load_points(trajectory_path, source=args.source, duration_seconds=duration_seconds)
    config = load_config(args.config)
    runner = DatasetAnalyticsRunner(
        dataset,
        config.analytics,
        coordinate_mode=args.coordinate_mode,
    )
    result = runner.run(
        points,
        source=args.source,
        duration_label=args.duration,
        persist=True,
    )
    print(json.dumps({
        "dataset": result["dataset"],
        "source": result["source"],
        "coordinate_mode": result["coordinate_mode"],
        "duration": result["duration"],
        "summary": result["summary"],
        "metrics": result["metrics"],
        "top_paths": result["paths"],
        "zone_flow_count": len(result["zone_flows"]),
        "artifacts": result["artifacts"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
