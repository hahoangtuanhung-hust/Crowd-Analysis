from __future__ import annotations

import csv
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Self

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from backend.app.datasets.grand_central import NormalizedTrajectoryPoint

TRAJECTORY_FIELDS = (
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
TRAJECTORY_SCHEMA = pa.schema(
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


class TrajectoryBatchWriter:
    def __init__(
        self,
        parquet_path: Path,
        *,
        csv_path: Path | None = None,
        batch_size: int = 10_000,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        self.parquet_path = parquet_path
        self.csv_path = csv_path
        self.batch_size = batch_size
        self.rows = 0
        self._batch: list[dict[str, Any]] = []
        self._parquet = pq.ParquetWriter(parquet_path, TRAJECTORY_SCHEMA, compression="zstd")
        self._csv_stream = None
        self._csv_writer = None
        if csv_path is not None:
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            self._csv_stream = csv_path.open("w", newline="", encoding="utf-8")
            self._csv_writer = csv.DictWriter(self._csv_stream, fieldnames=TRAJECTORY_FIELDS)
            self._csv_writer.writeheader()

    def write(self, points: Iterable[NormalizedTrajectoryPoint]) -> None:
        for point in points:
            self._batch.append(point.as_dict())
            if len(self._batch) >= self.batch_size:
                self.flush()

    def flush(self) -> None:
        if not self._batch:
            return
        self._parquet.write_table(pa.Table.from_pylist(self._batch, schema=TRAJECTORY_SCHEMA))
        if self._csv_writer is not None:
            self._csv_writer.writerows(self._batch)
        self.rows += len(self._batch)
        self._batch.clear()

    def close(self) -> None:
        self.flush()
        self._parquet.close()
        if self._csv_stream is not None:
            self._csv_stream.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def read_trajectory_points(
    path: Path,
    *,
    source: str,
    duration_seconds: float | None = None,
) -> list[NormalizedTrajectoryPoint]:
    if not path.is_file():
        raise FileNotFoundError(f"Normalized trajectories do not exist: {path}")
    filters: list[tuple[str, str, object]] = [("source", "=", source)]
    if duration_seconds is not None:
        filters.append(("timestamp", "<=", duration_seconds))
    table = pq.read_table(path, filters=filters)
    if table.num_rows:
        order = pc.sort_indices(
            table,
            sort_keys=[("timestamp", "ascending"), ("track_id", "ascending")],
        )
        table = pc.take(table, order)
    return [NormalizedTrajectoryPoint(**row) for row in table.to_pylist()]
