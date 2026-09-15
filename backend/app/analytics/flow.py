from __future__ import annotations

import math
from collections import Counter, OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from backend.app.analytics.heatmap import HeatmapWindow
from backend.app.analytics.spatial import SpatialTransformer
from backend.app.core.config import AnalyticsConfig
from backend.app.schemas import FlowSnapshot, PopularPath, TrackPoint, Trajectory

Cell = tuple[int, int]
Route = tuple[Cell, ...]


@dataclass(slots=True)
class _FlowBucket:
    timestamp: int
    vx: NDArray[np.float32]
    vy: NDArray[np.float32]
    samples: NDArray[np.uint32]
    edges: Counter[tuple[Cell, Cell]]


class FlowAnalyzer:
    """Bounded direction field plus completed macro-grid route heavy hitters."""

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
        self._shape = (config.grid_height, config.grid_width)
        self._buckets: OrderedDict[int, _FlowBucket] = OrderedDict()
        self._cursors: OrderedDict[int, TrackPoint] = OrderedDict()
        self._routes: dict[int, list[Cell]] = {}
        self._completed_paths: Counter[Route] = Counter()
        self._other_path_count = 0
        self._entire_vx = np.zeros(self._shape, dtype=np.float32)
        self._entire_vy = np.zeros(self._shape, dtype=np.float32)
        self._entire_samples = np.zeros(self._shape, dtype=np.uint32)
        self._latest_timestamp = 0.0

    @property
    def active_route_count(self) -> int:
        return len(self._routes)

    @property
    def completed_trajectory_count(self) -> int:
        return sum(self._completed_paths.values()) + self._other_path_count

    def reset(self) -> None:
        self._buckets.clear()
        self._cursors.clear()
        self._routes.clear()
        self._completed_paths.clear()
        self._other_path_count = 0
        self._entire_vx.fill(0.0)
        self._entire_vy.fill(0.0)
        self._entire_samples.fill(0)
        self._latest_timestamp = 0.0

    def process(self, trajectories: Iterable[Trajectory]) -> None:
        latest = self._latest_timestamp
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
            latest = max(latest, point.timestamp)
        self.finalize_stale(latest)

    def observe(self, point: TrackPoint) -> None:
        if self._latest_timestamp and point.timestamp < self._latest_timestamp:
            self.reset()
        self._latest_timestamp = max(self._latest_timestamp, point.timestamp)
        self._prune_buckets(point.timestamp)

        previous = self._cursors.get(point.track_id)
        self._cursors[point.track_id] = point
        self._cursors.move_to_end(point.track_id)
        route = self._routes.setdefault(point.track_id, [])
        current_x, current_y = self.transformer.transform(point.x, point.y)
        current_macro = self.transformer.grid_cell(
            current_x,
            current_y,
            grid_width=self._config.path_grid_width,
            grid_height=self._config.path_grid_height,
        )
        if not route and current_macro is not None:
            route.append(current_macro)

        while len(self._cursors) > self._config.max_active_tracks:
            evicted_id, _ = self._cursors.popitem(last=False)
            self._finalize_route(evicted_id)

        if previous is None or point.frame_id <= previous.frame_id:
            return
        elapsed = point.timestamp - previous.timestamp
        if elapsed <= 0 or elapsed > self._config.max_point_gap_seconds:
            return
        previous_x, previous_y = self.transformer.transform(previous.x, previous.y)
        dx = current_x - previous_x
        dy = current_y - previous_y
        distance = math.hypot(dx, dy)
        if (
            distance < self._config.movement_threshold_pixels
            or distance > self._config.max_movement_step_pixels
        ):
            return

        midpoint = ((previous_x + current_x) / 2.0, (previous_y + current_y) / 2.0)
        fine_cell = self.transformer.grid_cell(
            *midpoint,
            grid_width=self._config.grid_width,
            grid_height=self._config.grid_height,
        )
        if fine_cell is not None:
            bucket = self._bucket(point.timestamp)
            bucket.vx[fine_cell] += dx
            bucket.vy[fine_cell] += dy
            bucket.samples[fine_cell] += 1
            if self.retain_entire:
                self._entire_vx[fine_cell] += dx
                self._entire_vy[fine_cell] += dy
                self._entire_samples[fine_cell] += 1

        previous_macro = self.transformer.grid_cell(
            previous_x,
            previous_y,
            grid_width=self._config.path_grid_width,
            grid_height=self._config.path_grid_height,
        )
        if (
            previous_macro is not None
            and current_macro is not None
            and previous_macro != current_macro
        ):
            bucket = self._bucket(point.timestamp)
            bucket.edges[(previous_macro, current_macro)] += 1
            if not route:
                route.append(previous_macro)
            if route[-1] != current_macro:
                if len(route) < self._config.max_route_cells:
                    route.append(current_macro)
                else:
                    route[-1] = current_macro

    def finalize_stale(self, timestamp: float) -> None:
        stale = [
            track_id
            for track_id, point in self._cursors.items()
            if timestamp - point.timestamp > self._config.inactive_track_ttl_seconds
        ]
        for track_id in stale:
            self._cursors.pop(track_id, None)
            self._finalize_route(track_id)

    def finalize_all(self) -> None:
        for track_id in tuple(self._routes):
            self._finalize_route(track_id)
        self._cursors.clear()

    def top_paths(self, limit: int = 5) -> tuple[PopularPath, ...]:
        total = self.completed_trajectory_count
        ranked = sorted(self._completed_paths.items(), key=lambda item: (-item[1], item[0]))[:limit]
        return tuple(
            PopularPath(
                path_id=f"grid-{index}",
                label=" -> ".join(self._cell_label(cell) for cell in route),
                count=count,
                percentage=round(count / total * 100.0, 2) if total else 0.0,
                regions=tuple(self._cell_label(cell) for cell in route),
                kind="grid",
            )
            for index, (route, count) in enumerate(ranked, start=1)
        )

    def snapshot(self, window: HeatmapWindow = "current") -> FlowSnapshot:
        if window == "entire" and self.retain_entire:
            vx = self._entire_vx.copy()
            vy = self._entire_vy.copy()
            samples = self._entire_samples.copy()
            from_timestamp = 0.0
        else:
            seconds = self._window_seconds(window)
            from_timestamp = max(0.0, self._latest_timestamp - seconds)
            buckets = [
                bucket
                for timestamp, bucket in self._buckets.items()
                if timestamp >= math.floor(from_timestamp)
            ]
            vx = self._sum(buckets, "vx", np.float32)
            vy = self._sum(buckets, "vy", np.float32)
            samples = self._sum(buckets, "samples", np.uint32)
        return FlowSnapshot(
            window=window,
            spatial_mode=self.transformer.mode,
            from_timestamp=from_timestamp,
            to_timestamp=self._latest_timestamp,
            dominant_direction=self._direction(float(vx.sum()), float(vy.sum())),
            vx=vx,
            vy=vy,
            samples=samples,
        )

    def render(
        self, snapshot: FlowSnapshot, *, width: int = 768, height: int = 432
    ) -> NDArray[np.uint8]:
        canvas = np.full((height, width, 3), (23, 28, 34), dtype=np.uint8)
        cell_width = width / self._config.grid_width
        cell_height = height / self._config.grid_height
        active = np.argwhere(snapshot.samples > 0)
        for row, column in active:
            count = float(snapshot.samples[row, column])
            dx = float(snapshot.vx[row, column]) / count
            dy = float(snapshot.vy[row, column]) / count
            magnitude = math.hypot(dx, dy)
            if magnitude <= 0:
                continue
            scale = min(cell_width, cell_height) * 0.42 / magnitude
            center = (
                int((column + 0.5) * cell_width),
                int((row + 0.5) * cell_height),
            )
            end = (int(center[0] + dx * scale), int(center[1] + dy * scale))
            cv2.arrowedLine(canvas, center, end, (65, 213, 168), 2, cv2.LINE_AA, tipLength=0.35)
        return canvas

    def _finalize_route(self, track_id: int) -> None:
        route = tuple(self._routes.pop(track_id, []))
        if len(route) < 2:
            return
        if (
            route in self._completed_paths
            or len(self._completed_paths) < self._config.max_completed_paths
        ):
            self._completed_paths[route] += 1
        else:
            self._other_path_count += 1

    def _bucket(self, timestamp: float) -> _FlowBucket:
        key = math.floor(timestamp)
        bucket = self._buckets.get(key)
        if bucket is None:
            bucket = _FlowBucket(
                timestamp=key,
                vx=np.zeros(self._shape, dtype=np.float32),
                vy=np.zeros(self._shape, dtype=np.float32),
                samples=np.zeros(self._shape, dtype=np.uint32),
                edges=Counter(),
            )
            self._buckets[key] = bucket
        return bucket

    def _prune_buckets(self, timestamp: float) -> None:
        cutoff = math.floor(timestamp - self._config.live_retention_seconds)
        while self._buckets:
            first = next(iter(self._buckets))
            if first >= cutoff:
                break
            self._buckets.popitem(last=False)

    def _window_seconds(self, window: HeatmapWindow) -> int:
        if window == "current":
            return self._config.current_window_seconds
        if window == "1m":
            return 60
        return self._config.live_retention_seconds

    def _sum(self, buckets: list[_FlowBucket], name: str, dtype: type) -> NDArray:
        result = np.zeros(self._shape, dtype=dtype)
        for bucket in buckets:
            result += getattr(bucket, name)
        return result

    @staticmethod
    def _direction(dx: float, dy: float) -> str:
        if math.hypot(dx, dy) < 1e-6:
            return "stationary"
        names = (
            "east",
            "south-east",
            "south",
            "south-west",
            "west",
            "north-west",
            "north",
            "north-east",
        )
        index = round(math.atan2(dy, dx) / (math.pi / 4)) % 8
        return names[index]

    @staticmethod
    def _cell_label(cell: Cell) -> str:
        return f"R{cell[0] + 1}C{cell[1] + 1}"
