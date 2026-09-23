from __future__ import annotations

import math
import threading
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np

from backend.app.analytics.common_path import CommonPathAnalyzer
from backend.app.analytics.ddcrp import DDCRPClustering
from backend.app.analytics.directional_grid import DirectionalGridEngine, GridTrackPoint
from backend.app.analytics.flow import FlowAnalyzer
from backend.app.analytics.heatmap import HeatmapAnalyzer, HeatmapWindow
from backend.app.analytics.spatial import SpatialTransformer
from backend.app.analytics.tracklet_aggregation import TrackletAggregationEngine, TrackletPoint
from backend.app.analytics.trajectory import TrajectoryManager
from backend.app.analytics.zones import ZoneAnalyzer
from backend.app.core.config import AnalyticsConfig
from backend.app.schemas import (
    CommonPathSnapshot,
    CrowdSummary,
    DirectedFlowSnapshot,
    FlowSnapshot,
    FrameResult,
    HeatmapSnapshot,
    PopularPath,
    TimelinePoint,
    Trajectory,
    ZoneSnapshot,
)


@dataclass(slots=True)
class _TimelineBucket:
    total: int = 0
    samples: int = 0
    peak: int = 0


class AnalyticsEngine:
    """Thread-safe owner of all mutable analytics state for one camera session."""

    def __init__(
        self,
        config: AnalyticsConfig,
        transformer: SpatialTransformer,
        *,
        retain_entire: bool,
        camera_id: str = "cam01",
        stream_epoch: str = "initial",
    ) -> None:
        self.config = config
        self.camera_id = camera_id
        self.transformer = transformer
        self.retain_entire = retain_entire
        self.trajectories = TrajectoryManager(config)
        self.auxiliary_analytics_enabled = config.auxiliary_analytics_enabled
        if self.auxiliary_analytics_enabled:
            self.heatmaps: HeatmapAnalyzer | None = HeatmapAnalyzer(
                config, transformer, retain_entire=retain_entire
            )
            self.flows: FlowAnalyzer | None = FlowAnalyzer(
                config, transformer, retain_entire=retain_entire
            )
            self.zones: ZoneAnalyzer | None = ZoneAnalyzer(
                config, transformer, retain_entire=retain_entire
            )
        else:
            self.heatmaps = None
            self.flows = None
            self.zones = None

        # The production profile owns one Common Path engine. Legacy/grid
        # engines are constructed only for an explicitly selected compatibility
        # mode or for callers that opt into auxiliary analytics.
        self.common_path = (
            CommonPathAnalyzer(config, transformer)
            if self.auxiliary_analytics_enabled or config.common_path.engine in ("legacy", "shadow")
            else None
        )
        self.directional_path = (
            DirectionalGridEngine(
                config.directional_grid, transformer, camera_id=camera_id,
                stream_epoch=stream_epoch, zones=config.zones
            )
            if self.auxiliary_analytics_enabled or config.common_path.engine in ("directional_grid", "shadow")
            else None
        )
        self.tracklet_path = TrackletAggregationEngine(
            config.common_path.tracklet_aggregation, transformer, camera_id=camera_id,
            stream_epoch=stream_epoch, max_paths=config.common_path.max_paths,
        )
        self.ddcrp = (
            DDCRPClustering(config)
            if self.auxiliary_analytics_enabled and config.common_path.engine != "tracklet_aggregation"
            else None
        )
        self._lock = threading.RLock()
        self._current_count = 0
        self._peak_count = 0
        self._processed_frames = 0
        self._weighted_count = 0.0
        self._weighted_duration = 0.0
        self._previous_count = 0
        self._previous_timestamp: float | None = None
        self._unique_last_seen: OrderedDict[int, float] = OrderedDict()
        self._timeline: OrderedDict[int, _TimelineBucket] = OrderedDict()
        self._latest_trajectories: tuple[Trajectory, ...] = ()

    def process_frame(self, result: FrameResult) -> tuple[Trajectory, ...]:
        with self._lock:
            timestamp = result.packet.source_timestamp
            trajectories = self.trajectories.update(
                result.tracks,
                frame_id=result.packet.frame_id,
                timestamp=timestamp,
            )
            if self.heatmaps is not None:
                self.heatmaps.process(trajectories)
            if self.flows is not None:
                self.flows.process(trajectories)
            if self.zones is not None:
                self.zones.process(trajectories)
            latest_points = tuple(
                trajectory.points[-1]
                for trajectory in trajectories
                if trajectory.points
                and trajectory.points[-1].frame_id == result.packet.frame_id
            )
            engine_mode = self.config.common_path.engine
            confirmed_ids = {trajectory.track_id for trajectory in trajectories
                             if trajectory.confirmed}
            if self.common_path is not None and engine_mode in ("legacy", "shadow"):
                self.common_path.process_points(
                    latest_points,
                    active_track_ids=(track.track_id for track in result.tracks),
                    timestamp=timestamp,
                )
            if self.directional_path is not None and engine_mode in ("directional_grid", "shadow"):
                self.directional_path.update(
                    (GridTrackPoint(self.camera_id, self.directional_path.stream_epoch,
                                    track.track_id, 0, result.packet.frame_id,
                                    timestamp, *track.bottom_center)
                     for track in result.tracks
                     if track.observed and track.track_id in confirmed_ids),
                    timestamp,
                )
            if engine_mode == "tracklet_aggregation" and self.config.common_path.enabled:
                self.tracklet_path.update(
                    (TrackletPoint(self.camera_id, self.tracklet_path.stream_epoch,
                                   track.track_id, 0, result.packet.frame_id,
                                   timestamp, *track.bottom_center,
                                   confirmed=True, observed=track.observed)
                     for track in result.tracks
                     if track.track_id in confirmed_ids), timestamp,
                )
            if self.ddcrp is not None and self.config.ddcrp_enabled:
                self.ddcrp.observe_all(trajectories, timestamp=timestamp)
            self._latest_trajectories = trajectories

            self._current_count = len(result.tracks)
            self._peak_count = max(self._peak_count, self._current_count)
            self._processed_frames += 1
            if self._previous_timestamp is not None:
                elapsed = max(0.0, timestamp - self._previous_timestamp)
                elapsed = min(elapsed, self.config.max_point_gap_seconds)
                self._weighted_count += self._previous_count * elapsed
                self._weighted_duration += elapsed
            self._previous_timestamp = timestamp
            self._previous_count = self._current_count

            for trajectory in trajectories:
                if trajectory.confirmed:
                    self._unique_last_seen[trajectory.track_id] = timestamp
                    self._unique_last_seen.move_to_end(trajectory.track_id)
            self._prune_unique(timestamp)
            self._update_timeline(timestamp, self._current_count)
            return trajectories

    def finalize(self) -> None:
        with self._lock:
            if self.flows is not None:
                self.flows.finalize_all()
            if self.common_path is not None and self.config.common_path.engine in ("legacy", "shadow"):
                self.common_path.finalize_all()
            if self.config.common_path.engine == "tracklet_aggregation":
                self.tracklet_path.finalize()

    def summary(self) -> CrowdSummary:
        with self._lock:
            average = (
                self._weighted_count / self._weighted_duration
                if self._weighted_duration > 0
                else float(self._current_count)
            )
            return CrowdSummary(
                current_crowd_count=self._current_count,
                average_crowd_count=round(average, 3),
                peak_crowd_count=self._peak_count,
                unique_track_count=len(self._unique_last_seen),
                processed_frames=self._processed_frames,
                spatial_mode=self.transformer.mode,
                calibration_required=self.transformer.mode == "pixel",
            )

    def heatmap(self, window: HeatmapWindow = "current") -> HeatmapSnapshot:
        with self._lock:
            if self.heatmaps is not None:
                return self.heatmaps.snapshot(window)
            shape = (self.config.grid_height, self.config.grid_width)
            values = np.zeros(shape, dtype=np.float32)
            return HeatmapSnapshot(
                window, self.transformer.mode, 0.0, 0.0, values.copy(), values.copy()
            )

    def flow(self, window: HeatmapWindow = "current") -> FlowSnapshot:
        with self._lock:
            if self.flows is not None:
                return self.flows.snapshot(window)
            shape = (self.config.grid_height, self.config.grid_width)
            return FlowSnapshot(
                window, self.transformer.mode, 0.0, 0.0, np.zeros(shape, dtype=np.float32),
                np.zeros(shape, dtype=np.float32), np.zeros(shape, dtype=np.uint32),
            )

    def zone_snapshot(self) -> ZoneSnapshot:
        with self._lock:
            return self.zones.snapshot() if self.zones is not None else ZoneSnapshot(0.0, (), ())

    def top_paths(self, limit: int = 5) -> tuple[PopularPath, ...]:
        with self._lock:
            if self.zones is None:
                return ()
            ddcrp_paths = list(self.ddcrp.top_paths(limit)) if self.ddcrp is not None else []
            zone_paths = list(self.zones.top_paths(limit))
            if ddcrp_paths:
                # Combine DD-CRP pathways (including stationary queues) and zone transitions
                combined = ddcrp_paths + [p for p in zone_paths if not any(p.label == d.label for d in ddcrp_paths)]
                return tuple(sorted(combined, key=lambda p: -p.count)[:limit])
            return self.zones.top_paths(limit) or (self.flows.top_paths(limit) if self.flows else ())

    def common_path_snapshot(self) -> CommonPathSnapshot:
        with self._lock:
            if self.config.common_path.engine == "tracklet_aggregation":
                return self.tracklet_path.snapshot()
            if self.config.common_path.engine == "directional_grid" or (
                self.config.common_path.engine == "shadow" and
                self.config.common_path.shadow_display == "directional_grid"
            ):
                return self.directional_path.snapshot() if self.directional_path is not None else CommonPathSnapshot(0, ())
            if self.ddcrp is not None and self.config.ddcrp_enabled:
                return self.ddcrp.snapshot()
            return self.common_path.snapshot() if self.common_path is not None else CommonPathSnapshot(0, ())

    def common_path_flows(self) -> DirectedFlowSnapshot:
        with self._lock:
            if self.config.common_path.engine == "tracklet_aggregation":
                # Tracklet aggregation deliberately has no grid/edge output.
                return DirectedFlowSnapshot(
                    0.0, 0.0, self.config.common_path.grid_columns,
                    self.config.common_path.grid_rows, (),
                )
            if self.config.common_path.engine in ("directional_grid", "shadow"):
                return self.directional_path.flow_snapshot() if self.directional_path is not None else DirectedFlowSnapshot(0.0, 0.0, 0, 0, ())
            return self.common_path.flow_snapshot() if self.common_path is not None else DirectedFlowSnapshot(0.0, 0.0, 0, 0, ())

    def common_path_metrics(self) -> dict[str, int | float | None]:
        with self._lock:
            if self.config.common_path.engine == "tracklet_aggregation":
                compute_ms = self.tracklet_path.compute_ms
                return {
                    "tracklet_candidates": self.tracklet_path.candidate_count,
                    "tracklet_active": len(self.tracklet_path.snapshot().paths),
                    "common_path_compute_ms": round(float(np.percentile(compute_ms, 50)), 3) if compute_ms else None,
                    "common_path_compute_ms_p95": round(float(np.percentile(compute_ms, 95)), 3) if compute_ms else None,
                    "tracklet_segments": self.tracklet_path.buffered_segment_count,
                    "tracklet_rejections": sum(self.tracklet_path.rejections.values()),
                }
            if self.config.common_path.engine == "directional_grid":
                return {"directional_tracks": len(self.directional_path._tracks),
                        "directional_retained_points": self.directional_path.retained_points,
                        "directional_cap_drops": self.directional_path.overflow}
            metrics = self.common_path.metrics() if self.common_path is not None else {}
            if self.config.common_path.engine == "shadow" and self.directional_path is not None:
                metrics.update({"directional_tracks": len(self.directional_path._tracks),
                                "directional_cap_drops": self.directional_path.overflow})
            return metrics


    def timeline(self) -> tuple[TimelinePoint, ...]:
        with self._lock:
            return tuple(
                TimelinePoint(
                    timestamp=float(timestamp),
                    average_count=round(bucket.total / bucket.samples, 3),
                    peak_count=bucket.peak,
                )
                for timestamp, bucket in self._timeline.items()
                if bucket.samples
            )

    def latest_trajectories(self) -> tuple[Trajectory, ...]:
        with self._lock:
            return self._latest_trajectories

    def _prune_unique(self, timestamp: float) -> None:
        if self.retain_entire:
            return
        cutoff = timestamp - self.config.live_retention_seconds
        stale = [
            track_id for track_id, last_seen in self._unique_last_seen.items() if last_seen < cutoff
        ]
        for track_id in stale:
            del self._unique_last_seen[track_id]
        while len(self._unique_last_seen) > self.config.max_active_tracks:
            self._unique_last_seen.popitem(last=False)

    def _update_timeline(self, timestamp: float, count: int) -> None:
        key = math.floor(timestamp)
        bucket = self._timeline.setdefault(key, _TimelineBucket())
        bucket.total += count
        bucket.samples += 1
        bucket.peak = max(bucket.peak, count)
        if self.retain_entire:
            return
        cutoff = math.floor(timestamp - self.config.live_retention_seconds)
        while self._timeline and next(iter(self._timeline)) < cutoff:
            self._timeline.popitem(last=False)
