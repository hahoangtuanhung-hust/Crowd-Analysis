from __future__ import annotations

import math
from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np
from numpy.typing import NDArray

from backend.app.analytics.spatial import SpatialTransformer
from backend.app.core.config import AnalyticsConfig
from backend.app.schemas import HeatmapSnapshot, TrackPoint, Trajectory

HeatmapWindow = Literal["current", "1m", "5m", "entire"]


@dataclass(slots=True)
class _GridBucket:
    timestamp: int
    occupancy: NDArray[np.float32]
    movement: NDArray[np.float32]


class HeatmapAnalyzer:
    """Incremental occupancy and movement grids with bounded live windows."""

    def __init__(
        self,
        config: AnalyticsConfig,
        transformer: SpatialTransformer,
        *,
        retain_entire: bool,
    ) -> None:
        self._config = config
        self.transformer = transformer
        self.retain_entire = retain_entire
        self._buckets: OrderedDict[int, _GridBucket] = OrderedDict()
        self._cursors: OrderedDict[int, TrackPoint] = OrderedDict()
        self._shape = (config.grid_height, config.grid_width)
        self._entire_occupancy = np.zeros(self._shape, dtype=np.float32)
        self._entire_movement = np.zeros(self._shape, dtype=np.float32)
        self._latest_timestamp = 0.0

    @property
    def bucket_count(self) -> int:
        return len(self._buckets)

    def reset(self) -> None:
        self._buckets.clear()
        self._cursors.clear()
        self._entire_occupancy.fill(0.0)
        self._entire_movement.fill(0.0)
        self._latest_timestamp = 0.0

    def process(self, trajectories: Iterable[Trajectory]) -> None:
        pending: list[TrackPoint] = []
        for trajectory in trajectories:
            if not trajectory.confirmed:
                continue
            for point in trajectory.points:
                cursor = self._cursors.get(point.track_id)
                if cursor is not None and point.frame_id <= cursor.frame_id:
                    continue
                pending.append(point)
        for point in sorted(
            pending, key=lambda item: (item.timestamp, item.frame_id, item.track_id)
        ):
            self.observe(point)

    def observe(self, point: TrackPoint) -> None:
        if self._latest_timestamp and point.timestamp < self._latest_timestamp:
            self.reset()
        self._latest_timestamp = max(self._latest_timestamp, point.timestamp)
        self._prune(point.timestamp)
        self._prune_cursors(point.timestamp)

        previous = self._cursors.get(point.track_id)
        self._cursors[point.track_id] = point
        self._cursors.move_to_end(point.track_id)
        while len(self._cursors) > self._config.max_active_tracks:
            self._cursors.popitem(last=False)
        if previous is None:
            return

        elapsed = point.timestamp - previous.timestamp
        if elapsed <= 0:
            return
        occupancy_weight = min(elapsed, self._config.max_point_gap_seconds)
        x, y = self.transformer.transform(point.x, point.y)
        cell = self.transformer.grid_cell(
            x,
            y,
            grid_width=self._config.grid_width,
            grid_height=self._config.grid_height,
        )
        if cell is not None:
            bucket = self._bucket(point.timestamp)
            bucket.occupancy[cell] += occupancy_weight
            if self.retain_entire:
                self._entire_occupancy[cell] += occupancy_weight

        if elapsed > self._config.max_point_gap_seconds:
            return
        previous_x, previous_y = self.transformer.transform(previous.x, previous.y)
        distance = math.hypot(x - previous_x, y - previous_y)
        if (
            distance < self._config.movement_threshold_pixels
            or distance > self._config.max_movement_step_pixels
        ):
            return
        self._add_movement_segment(previous_x, previous_y, x, y, distance, point.timestamp)

    def snapshot(self, window: HeatmapWindow = "current") -> HeatmapSnapshot:
        if window == "entire" and self.retain_entire:
            occupancy = self._entire_occupancy.copy()
            movement = self._entire_movement.copy()
            from_timestamp = 0.0
        else:
            seconds = self._window_seconds(window)
            from_timestamp = max(0.0, self._latest_timestamp - seconds)
            selected = [
                bucket
                for timestamp, bucket in self._buckets.items()
                if timestamp >= math.floor(from_timestamp)
            ]
            occupancy = self._sum(selected, "occupancy")
            movement = self._sum(selected, "movement")
        return HeatmapSnapshot(
            window=window,
            spatial_mode=self.transformer.mode,
            from_timestamp=from_timestamp,
            to_timestamp=self._latest_timestamp,
            occupancy=occupancy,
            movement=movement,
        )

    def render(self, values: NDArray[np.float32]) -> NDArray[np.uint8]:
        smoothed = values.astype(np.float32, copy=True)
        if self._config.gaussian_sigma > 0:
            smoothed = cv2.GaussianBlur(
                smoothed,
                ksize=(0, 0),
                sigmaX=self._config.gaussian_sigma,
                sigmaY=self._config.gaussian_sigma,
            )
        maximum = float(smoothed.max(initial=0.0))
        if maximum <= 0:
            return np.zeros((*self._shape, 3), dtype=np.uint8)
        normalized = np.clip(smoothed / maximum * 255.0, 0, 255).astype(np.uint8)
        return cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)

    def _bucket(self, timestamp: float) -> _GridBucket:
        key = math.floor(timestamp)
        bucket = self._buckets.get(key)
        if bucket is None:
            bucket = _GridBucket(
                timestamp=key,
                occupancy=np.zeros(self._shape, dtype=np.float32),
                movement=np.zeros(self._shape, dtype=np.float32),
            )
            self._buckets[key] = bucket
        return bucket

    def _add_movement_segment(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        distance: float,
        timestamp: float,
    ) -> None:
        start = self.transformer.grid_cell(
            x1,
            y1,
            grid_width=self._config.grid_width,
            grid_height=self._config.grid_height,
        )
        end = self.transformer.grid_cell(
            x2,
            y2,
            grid_width=self._config.grid_width,
            grid_height=self._config.grid_height,
        )
        if start is None or end is None:
            return
        steps = max(abs(end[0] - start[0]), abs(end[1] - start[1]), 1) + 1
        weight = distance / steps
        bucket = self._bucket(timestamp)
        for row_value, column_value in zip(
            np.linspace(start[0], end[0], steps),
            np.linspace(start[1], end[1], steps),
            strict=True,
        ):
            cell = round(row_value), round(column_value)
            bucket.movement[cell] += weight
            if self.retain_entire:
                self._entire_movement[cell] += weight

    def _prune(self, timestamp: float) -> None:
        cutoff = math.floor(timestamp - self._config.live_retention_seconds)
        while self._buckets:
            first = next(iter(self._buckets))
            if first >= cutoff:
                break
            self._buckets.popitem(last=False)

    def _prune_cursors(self, timestamp: float) -> None:
        stale = [
            track_id
            for track_id, point in self._cursors.items()
            if timestamp - point.timestamp > self._config.inactive_track_ttl_seconds
        ]
        for track_id in stale:
            del self._cursors[track_id]

    def _window_seconds(self, window: HeatmapWindow) -> int:
        if window == "current":
            return self._config.current_window_seconds
        if window == "1m":
            return 60
        return self._config.live_retention_seconds if window in {"5m", "entire"} else 300

    def _sum(self, buckets: list[_GridBucket], name: str) -> NDArray[np.float32]:
        result = np.zeros(self._shape, dtype=np.float32)
        for bucket in buckets:
            result += getattr(bucket, name)
        return result
