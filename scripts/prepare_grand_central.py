from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.datasets import GrandCentralDataset, GrandCentralPaths
from backend.app.datasets.grand_central import NormalizedTrajectoryPoint

FIELDS = (
    "camera_id",
    "source",
    "frame_id",
    "timestamp",
    "track_id",
    "x",
    "y",
    "x1",
    "y1",
    "x2",
    "y2",
    "foot_x",
    "foot_y",
    "world_x",
    "world_y",
    "confidence",
    "zone_id",
)
PARQUET_SCHEMA = pa.schema(
    [
        pa.field("camera_id", pa.string(), nullable=False),
        pa.field("source", pa.string(), nullable=False),
        pa.field("frame_id", pa.int64(), nullable=False),
        pa.field("timestamp", pa.float64(), nullable=False),
        pa.field("track_id", pa.int64(), nullable=False),
        pa.field("x", pa.float64()),
        pa.field("y", pa.float64()),
        pa.field("x1", pa.float64()),
        pa.field("y1", pa.float64()),
        pa.field("x2", pa.float64()),
        pa.field("y2", pa.float64()),
        pa.field("foot_x", pa.float64(), nullable=False),
        pa.field("foot_y", pa.float64(), nullable=False),
        pa.field("world_x", pa.float64()),
        pa.field("world_y", pa.float64()),
        pa.field("confidence", pa.float64()),
        pa.field("zone_id", pa.string()),
    ]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize Grand Central annotations to Parquet and optional CSV"
    )
    parser.add_argument("--root", type=Path, help="Override GC_DATASET_ROOT")
    parser.add_argument(
        "--coordinate-mode",
        choices=("pixel_space", "ground_plane"),
        default="ground_plane",
    )
    parser.add_argument(
        "--duration",
        choices=("1m", "5m", "all"),
        default="all",
        help="Restrict annotation export by source timestamp",
    )
    parser.add_argument("--csv", action="store_true", help="Also write trajectories.csv")
    parser.add_argument("--batch-size", type=int, default=10_000)
    return parser.parse_args()


def load_zones(paths: GrandCentralPaths) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source = PROJECT_ROOT / "configs" / "grand_central_zones.json"
    if not source.is_file():
        raise FileNotFoundError(f"Missing zone definition: {source}")
    document = json.loads(source.read_text(encoding="utf-8"))
    destination = paths.processed / "zones.json"
    destination.write_text(json.dumps(document, indent=2), encoding="utf-8")
    return list(document["zones"]), document


def write_homography(dataset: GrandCentralDataset) -> Path | None:
    matrix = dataset.load_homography(required=False)
    if matrix is None:
        return None
    destination = dataset.paths.processed / "homography.json"
    payload = {
        "matrix": matrix.tolist(),
        "source": "OpenTraj datasets/GC/H.json",
        "calibration_method": "manually calculated by OpenTraj",
        "world_unit": "unverified",
        "warning": "Do not label these coordinates as metres without independent calibration.",
    }
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return destination


def batched(
    rows: Iterable[NormalizedTrajectoryPoint], batch_size: int
) -> Iterable[list[dict[str, Any]]]:
    batch: list[dict[str, Any]] = []
    for row in rows:
        batch.append(row.as_dict())
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def export(
    dataset: GrandCentralDataset,
    *,
    coordinate_mode: str,
    duration_seconds: float | None,
    include_csv: bool,
    batch_size: int,
    zones: list[dict[str, Any]],
) -> dict[str, Any]:
    if batch_size < 1:
        raise ValueError("batch-size must be positive")
    parquet_path = dataset.paths.processed / "trajectories.parquet"
    csv_path = dataset.paths.processed / "trajectories.csv"
    writer = pq.ParquetWriter(parquet_path, PARQUET_SCHEMA, compression="zstd")
    csv_stream = csv_path.open("w", newline="", encoding="utf-8") if include_csv else None
    csv_writer = csv.DictWriter(csv_stream, fieldnames=FIELDS) if csv_stream else None
    if csv_writer:
        csv_writer.writeheader()
    count = 0
    track_ids: set[int] = set()
    frame_ids: set[int] = set()
    started = time.perf_counter()
    try:
        rows = dataset.load_trajectories(
            duration_seconds=duration_seconds,
            coordinate_mode=coordinate_mode,  # type: ignore[arg-type]
            zones=zones,
        )
        for batch in batched(rows, batch_size):
            writer.write_table(pa.Table.from_pylist(batch, schema=PARQUET_SCHEMA))
            if csv_writer:
                csv_writer.writerows(batch)
            count += len(batch)
            track_ids.update(int(row["track_id"]) for row in batch)
            frame_ids.update(int(row["frame_id"]) for row in batch)
    finally:
        writer.close()
        if csv_stream:
            csv_stream.close()
    if count == 0:
        parquet_path.unlink(missing_ok=True)
        if include_csv:
            csv_path.unlink(missing_ok=True)
        raise ValueError("No normalized trajectory rows were produced")
    return {
        "source": "ground_truth",
        "coordinate_mode": coordinate_mode,
        "duration_seconds": duration_seconds,
        "rows": count,
        "tracks": len(track_ids),
        "annotated_frames": len(frame_ids),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "parquet": str(parquet_path.resolve()),
        "parquet_bytes": parquet_path.stat().st_size,
        "csv": str(csv_path.resolve()) if include_csv else None,
        "csv_bytes": csv_path.stat().st_size if include_csv else None,
        "schema": list(FIELDS),
    }


def main() -> int:
    args = parse_args()
    duration_seconds = {"1m": 60.0, "5m": 300.0, "all": None}[args.duration]
    paths = GrandCentralPaths.from_env(args.root)
    dataset = GrandCentralDataset(paths)
    validation = dataset.validate()
    zones, zone_document = load_zones(paths)
    homography_path = write_homography(dataset)
    result = export(
        dataset,
        coordinate_mode=args.coordinate_mode,
        duration_seconds=duration_seconds,
        include_csv=args.csv,
        batch_size=args.batch_size,
        zones=zones,
    )
    result["validation_report"] = str((paths.processed / "validation_report.json").resolve())
    result["zones"] = len(zone_document["zones"])
    result["homography"] = str(homography_path.resolve()) if homography_path else None
    report_path = paths.processed / "normalization_report.json"
    report_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    metadata_path = paths.processed / "metadata.json"
    metadata = dataset.load_metadata()
    metadata["validation"] = validation
    metadata["normalization"] = result
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
