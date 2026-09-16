from __future__ import annotations

import math
import threading
from collections import OrderedDict
from dataclasses import dataclass

from backend.app.analytics.ddcrp import DDCRPClustering
from backend.app.analytics.flow import FlowAnalyzer
from backend.app.analytics.heatmap import HeatmapAnalyzer, HeatmapWindow
from backend.app.analytics.spatial import SpatialTransformer
from backend.app.analytics.trajectory import TrajectoryManager
from backend.app.analytics.zones import ZoneAnalyzer
from backend.app.core.config import AnalyticsConfig
from backend.app.schemas import (
    CrowdSummary,
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
    ) -> None:
        self.config = config
        self.transformer = transformer
        self.retain_entire = retain_entire
        self.trajectories = TrajectoryManager(config)
        self.heatmaps = HeatmapAnalyzer(config, transformer, retain_entire=retain_entire)
        self.flows = FlowAnalyzer(config, transformer, retain_entire=retain_entire)
        self.zones = ZoneAnalyzer(config, transformer, retain_entire=retain_entire)
        self.ddcrp = DDCRPClustering(
            config,
            alpha=config.ddcrp_alpha,
            spatial_scale=config.ddcrp_spatial_scale,
            direction_weight=config.ddcrp_direction_weight,
            stationary_threshold=config.ddcrp_stationary_threshold,
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
            self.heatmaps.process(trajectories)
            self.flows.process(trajectories)
            self.zones.process(trajectories)
            if self.config.ddcrp_enabled:
                self.ddcrp.observe_all(trajectories)
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
            self.flows.finalize_all()

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
            return self.heatmaps.snapshot(window)

    def flow(self, window: HeatmapWindow = "current") -> FlowSnapshot:
        with self._lock:
            return self.flows.snapshot(window)

    def zone_snapshot(self) -> ZoneSnapshot:
        with self._lock:
            return self.zones.snapshot()

    def top_paths(self, limit: int = 5) -> tuple[PopularPath, ...]:
        with self._lock:
            ddcrp_paths = list(self.ddcrp.top_paths(limit)) if self.config.ddcrp_enabled else []
            zone_paths = list(self.zones.top_paths(limit))
            if ddcrp_paths:
                # Combine DD-CRP pathways (including stationary queues) and zone transitions
                combined = ddcrp_paths + [p for p in zone_paths if not any(p.label == d.label for d in ddcrp_paths)]
                return tuple(sorted(combined, key=lambda p: -p.count)[:limit])
            return self.zones.top_paths(limit) or self.flows.top_paths(limit)


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
