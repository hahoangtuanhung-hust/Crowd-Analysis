from __future__ import annotations

import argparse
import csv
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from backend.app.analytics import CommonPathAnalyzer, SpatialTransformer
from backend.app.analytics.point_tracklets import TrajectoryPoint
from backend.app.core.config import load_config
from scripts.process_point_tracks import load_zones

FIELDS = (
    "profile",
    "grid",
    "short_window_seconds",
    "long_window_seconds",
    "switch_margin",
    "confirmation_seconds",
    "active_paths_at_end",
    "peak_active_paths",
    "first_active_seconds",
    "common_path_switches",
    "switches_per_minute",
    "candidate_rejections",
    "valid_tracks",
    "peak_path_support",
    "compute_median_ms",
    "compute_p95_ms",
    "replay_seconds",
)


@dataclass(frozen=True, slots=True)
class Profile:
    name: str
    changes: dict[str, int | float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay real trajectory artifacts through CommonPathAnalyzer"
    )
    parser.add_argument(
        "--trajectories",
        default="outputs/common-path-realtime/trajectories.csv",
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--zones", default="configs/zones.json")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument(
        "--output",
        default="outputs/common-path-realtime/parameter_benchmark.csv",
    )
    return parser.parse_args()


def load_points(
    path: Path,
) -> tuple[list[tuple[float, tuple[TrajectoryPoint, ...]]], dict[int, tuple[float, float]]]:
    grouped: dict[float, list[TrajectoryPoint]] = defaultdict(list)
    lifetimes: dict[int, tuple[float, float]] = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            timestamp = float(row["timestamp"])
            track_id = int(row["track_id"])
            grouped[timestamp].append(
                TrajectoryPoint(
                    camera_id=row["camera_id"],
                    track_id=track_id,
                    frame_id=int(row["frame_id"]),
                    timestamp=timestamp,
                    x=float(row["x"]),
                    y=float(row["y"]),
                    confidence=float(row["confidence"]),
                    zone_id=row["zone_id"] or None,
                )
            )
            first, last = lifetimes.get(track_id, (timestamp, timestamp))
            lifetimes[track_id] = min(first, timestamp), max(last, timestamp)
    ordered = [(timestamp, tuple(grouped[timestamp])) for timestamp in sorted(grouped)]
    return ordered, lifetimes


def profiles() -> tuple[Profile, ...]:
    return (
        Profile("selected", {}),
        Profile("short_15", {"short_window_seconds": 15}),
        Profile("short_60", {"short_window_seconds": 60}),
        Profile("long_120", {"long_window_seconds": 120}),
        Profile("long_300", {"long_window_seconds": 300}),
        Profile("margin_0.10", {"switch_margin": 0.10}),
        Profile("margin_0.30", {"switch_margin": 0.30}),
        Profile("confirm_3", {"confirmation_seconds": 3.0}),
        Profile("confirm_15", {"confirmation_seconds": 15.0}),
        Profile("grid_16x9", {"grid_columns": 16, "grid_rows": 9}),
        Profile("grid_32x18", {"grid_columns": 32, "grid_rows": 18}),
        Profile("grid_64x36", {"grid_columns": 64, "grid_rows": 36}),
    )


def run_profile(
    profile: Profile,
    *,
    base_config,
    zones,
    groups: list[tuple[float, tuple[TrajectoryPoint, ...]]],
    lifetimes: dict[int, tuple[float, float]],
    width: int,
    height: int,
) -> dict[str, object]:
    settings = base_config.analytics.common_path.model_copy(update=profile.changes)
    analytics = base_config.analytics.model_copy(
        update={"common_path": settings, "zones": list(zones)}
    )
    analyzer = CommonPathAnalyzer(analytics, SpatialTransformer.pixel(width, height))
    first_timestamp = groups[0][0]
    last_timestamp = groups[-1][0]
    first_active: float | None = None
    peak_active = 0
    peak_support = 0
    started_at = time.perf_counter()

    for timestamp, points in groups:
        active_ids = {
            track_id
            for track_id, (first, last) in lifetimes.items()
            if first <= timestamp <= last
        }
        snapshot = analyzer.process_points(
            points,
            active_track_ids=active_ids,
            timestamp=timestamp,
        )
        active_paths = [path for path in snapshot.paths if path.state == "active"]
        peak_active = max(peak_active, len(active_paths))
        peak_support = max(
            peak_support,
            max((path.unique_tracks_short for path in snapshot.paths), default=0),
        )
        if active_paths and first_active is None:
            first_active = timestamp - first_timestamp

    snapshot = analyzer.finalize_all(last_timestamp)
    active_at_end = sum(path.state == "active" for path in snapshot.paths)
    peak_support = max(
        peak_support,
        max((path.unique_tracks_short for path in snapshot.paths), default=0),
    )
    elapsed = time.perf_counter() - started_at
    duration_minutes = max((last_timestamp - first_timestamp) / 60.0, 1 / 60)
    metrics = analyzer.metrics()
    return {
        "profile": profile.name,
        "grid": f"{settings.grid_columns}x{settings.grid_rows}",
        "short_window_seconds": settings.short_window_seconds,
        "long_window_seconds": settings.long_window_seconds,
        "switch_margin": settings.switch_margin,
        "confirmation_seconds": settings.confirmation_seconds,
        "active_paths_at_end": active_at_end,
        "peak_active_paths": peak_active,
        "first_active_seconds": "" if first_active is None else round(first_active, 3),
        "common_path_switches": analyzer.common_path_switches,
        "switches_per_minute": round(analyzer.common_path_switches / duration_minutes, 3),
        "candidate_rejections": analyzer.candidate_rejections,
        "valid_tracks": analyzer.valid_tracks,
        "peak_path_support": peak_support,
        "compute_median_ms": metrics["common_path_compute_ms"],
        "compute_p95_ms": metrics["common_path_compute_ms_p95"],
        "replay_seconds": round(elapsed, 4),
    }


def main() -> None:
    args = parse_args()
    trajectory_path = Path(args.trajectories)
    output_path = Path(args.output)
    groups, lifetimes = load_points(trajectory_path)
    if not groups:
        raise ValueError(f"No trajectory rows found in {trajectory_path}")

    base_config = load_config(args.config)
    zones = load_zones(Path(args.zones), args.width, args.height)
    replay_arguments = {
        "base_config": base_config,
        "zones": zones,
        "groups": groups,
        "lifetimes": lifetimes,
        "width": args.width,
        "height": args.height,
    }
    run_profile(Profile("warmup", {}), **replay_arguments)
    results = [
        run_profile(profile, **replay_arguments)
        for profile in profiles()
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(results)
    print(f"Wrote {len(results)} real-trajectory profiles to {output_path}")


if __name__ == "__main__":
    main()
