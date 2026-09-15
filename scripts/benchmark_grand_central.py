from __future__ import annotations

import argparse
import csv
import json
import platform
import shutil
import sys
from pathlib import Path
from typing import Any

import psutil

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.core.config import load_config
from backend.app.datasets import DatasetAnalyticsRunner, GrandCentralDataset, GrandCentralPaths
from backend.app.datasets.storage import read_trajectory_points

FIELDS = (
    "mode",
    "detector",
    "tracker",
    "imgsz",
    "inference_interval",
    "processing_fps",
    "inference_ms",
    "tracking_ms",
    "analytics_ms",
    "e2e_ms",
    "cpu_percent",
    "ram_mb",
    "gpu_memory_mb",
    "dropped_frames",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate measured Grand Central benchmarks")
    parser.add_argument("--root", type=Path, help="Override GC_DATASET_ROOT")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--duration", choices=("1m",), default="1m")
    parser.add_argument(
        "--output", type=Path, default=Path("benchmarks/grand_central_results.csv")
    )
    return parser.parse_args()


def analytics_measurement(
    dataset: GrandCentralDataset,
    trajectory_path: Path,
    *,
    source: str,
    config_path: Path,
    duration_seconds: float,
) -> tuple[dict[str, Any], float, float]:
    points = read_trajectory_points(
        trajectory_path, source=source, duration_seconds=duration_seconds
    )
    config = load_config(config_path)
    process = psutil.Process()
    process.cpu_percent(None)
    runner = DatasetAnalyticsRunner(
        dataset, config.analytics, coordinate_mode="pixel_space"
    )
    result = runner.run(
        points,
        source=source,  # type: ignore[arg-type]
        duration_label="1m",
        persist=False,
    )
    cpu_percent = process.cpu_percent(None)
    ram_mb = round(process.memory_info().rss / 1024**2, 3)
    return result, cpu_percent, ram_mb


def main() -> int:
    args = parse_args()
    dataset = GrandCentralDataset(GrandCentralPaths.from_env(args.root))
    paths = dataset.paths
    duration_seconds = 60.0
    rows = []

    gt_result, gt_cpu, gt_ram = analytics_measurement(
        dataset,
        paths.processed / "trajectories.parquet",
        source="ground_truth",
        config_path=args.config,
        duration_seconds=duration_seconds,
    )
    rows.append(
        {
            "mode": "ground_truth_analytics",
            "detector": "none",
            "tracker": "none",
            "imgsz": None,
            "inference_interval": 0,
            "processing_fps": gt_result["metrics"]["processing_fps"],
            "inference_ms": None,
            "tracking_ms": None,
            "analytics_ms": gt_result["metrics"]["analytics_ms"],
            "e2e_ms": gt_result["metrics"]["e2e_ms"],
            "cpu_percent": gt_cpu,
            "ram_mb": gt_ram,
            "gpu_memory_mb": None,
            "dropped_frames": 0,
        }
    )

    for interval in (1, 2):
        measured_path = (
            paths.output / "benchmarks" / f"prediction_interval_{interval}_1m.json"
        )
        trajectory_path = (
            paths.output / "tracking" / f"prediction_interval_{interval}_1m.parquet"
        )
        if not measured_path.is_file() or not trajectory_path.is_file():
            raise FileNotFoundError(
                f"Run scripts/run_grand_central_inference.py --duration 1m "
                f"--inference-interval {interval} first"
            )
        measured = json.loads(measured_path.read_text(encoding="utf-8"))
        analytics, _, _ = analytics_measurement(
            dataset,
            trajectory_path,
            source="prediction",
            config_path=args.config,
            duration_seconds=duration_seconds,
        )
        rows.append(
            {
                "mode": "yolo_bytetrack",
                "detector": measured["detector"],
                "tracker": measured["tracker"],
                "imgsz": measured["imgsz"],
                "inference_interval": interval,
                "processing_fps": measured["processing_fps"],
                "inference_ms": measured["inference_ms"],
                "tracking_ms": measured["tracking_ms"],
                "analytics_ms": analytics["metrics"]["analytics_ms"],
                "e2e_ms": measured["e2e_ms"],
                "cpu_percent": measured["cpu_percent"],
                "ram_mb": measured["ram_mb"],
                "gpu_memory_mb": measured["gpu_memory_mb"],
                "dropped_frames": measured["dropped_frames"],
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    dataset_copy = paths.output / "benchmarks" / "grand_central_results.csv"
    shutil.copy2(args.output, dataset_copy)
    metadata = {
        "dataset": "grand-central",
        "duration": args.duration,
        "duration_seconds": duration_seconds,
        "coordinate_mode": "pixel_space",
        "hardware": {
            "processor": platform.processor() or platform.machine(),
            "physical_cores": psutil.cpu_count(logical=False),
            "logical_cores": psutil.cpu_count(),
            "ram_gb": round(psutil.virtual_memory().total / 1024**3, 1),
            "gpu": None,
        },
        "notes": [
            "All rows were measured on the supplied data.mp4; no values were estimated.",
            "Ground-truth processing_fps counts sparse annotated frames per second of wall time.",
            "YOLO processing_fps counts detector frames per second of wall time.",
            "Blank GPU fields indicate a CPU-only run.",
        ],
        "csv": str(args.output.resolve()),
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps({"rows": rows, "output": str(args.output.resolve())}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
