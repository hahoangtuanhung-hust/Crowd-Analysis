from __future__ import annotations

from collections import Counter, OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass

import cv2
import numpy as np

from backend.app.analytics.spatial import SpatialTransformer
from backend.app.core.config import AnalyticsConfig, ZoneConfig
from backend.app.schemas import (
    PopularPath,
    TrackPoint,
    Trajectory,
    ZoneFlow,
    ZoneMetric,
    ZoneSnapshot,
)


@dataclass(slots=True)
class _TrackZoneState:
    stable_zone: str | None = None
    candidate_zone: str | None = None
    candidate_count: int = 0
    last_nonempty_zone: str | None = None
    entered_at: float | None = None
    last_seen: float = 0.0
    last_frame_id: int = -1


class ZoneAnalyzer:
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
        self._zones = tuple(config.zones)
        self._polygons = {
            zone.zone_id: np.asarray(zone.points, dtype=np.float32) for zone in self._zones
        }
        self._states: dict[int, _TrackZoneState] = {}
        self._entries: Counter[str] = Counter()
        self._exits: Counter[str] = Counter()
        self._peak: Counter[str] = Counter()
        self._dwell_total: Counter[str] = Counter()
        self._completed_visits: Counter[str] = Counter()
        self._flows: Counter[tuple[str, str]] = Counter()
        self._counted_flow_keys: OrderedDict[tuple[int, str, str], None] = OrderedDict()
        self._unique_last_seen: dict[str, dict[int, float]] = {
            zone.zone_id: {} for zone in self._zones
        }
        self._current: Counter[str] = Counter()
        self._latest_timestamp = 0.0

    def reset(self) -> None:
        self.__init__(self._config, self.transformer, retain_entire=self.retain_entire)

    def process(self, trajectories: Iterable[Trajectory]) -> None:
        observed_ids: set[int] = set()
        latest = self._latest_timestamp
        for trajectory in trajectories:
            if not trajectory.confirmed:
                continue
            observed_ids.add(trajectory.track_id)
            state = self._states.setdefault(trajectory.track_id, _TrackZoneState())
            for point in trajectory.points:
                if point.frame_id <= state.last_frame_id:
                    continue
                self._observe(point, state)
                latest = max(latest, point.timestamp)

        self._latest_timestamp = latest
        self._evict_stale(latest)
        self._prune_unique(latest)
        self._current = Counter(
            state.stable_zone
            for track_id, state in self._states.items()
            if track_id in observed_ids and state.stable_zone is not None
        )
        for zone_id, count in self._current.items():
            self._peak[zone_id] = max(self._peak[zone_id], count)

    def snapshot(self) -> ZoneSnapshot:
        metrics = tuple(self._zone_metric(zone) for zone in self._zones)
        flows = tuple(
            ZoneFlow(from_zone=source, to_zone=target, count=count)
            for (source, target), count in sorted(
                self._flows.items(), key=lambda item: (-item[1], item[0])
            )
        )
        return ZoneSnapshot(timestamp=self._latest_timestamp, zones=metrics, flows=flows)

    def top_paths(self, limit: int = 5) -> tuple[PopularPath, ...]:
        total = sum(self._flows.values())
        names = {zone.zone_id: zone.name for zone in self._zones}
        ranked = sorted(self._flows.items(), key=lambda item: (-item[1], item[0]))[:limit]
        return tuple(
            PopularPath(
                path_id=f"zone-{index}",
                label=f"{names[source]} -> {names[target]}",
                count=count,
                percentage=round(count / total * 100.0, 2) if total else 0.0,
                regions=(source, target),
                kind="zone",
            )
            for index, ((source, target), count) in enumerate(ranked, start=1)
        )

    def _observe(self, point: TrackPoint, state: _TrackZoneState) -> None:
        x, y = self.transformer.transform(point.x, point.y)
        detected = self._zone_at(x, y)
        state.last_seen = point.timestamp
        state.last_frame_id = point.frame_id

        if detected == state.stable_zone:
            state.candidate_zone = None
            state.candidate_count = 0
        else:
            if detected == state.candidate_zone:
                state.candidate_count += 1
            else:
                state.candidate_zone = detected
                state.candidate_count = 1
            if state.candidate_count >= self._config.zone_debounce_points:
                self._transition(state, detected, point.track_id, point.timestamp)
                state.candidate_zone = None
                state.candidate_count = 0

        if state.stable_zone is not None:
            self._unique_last_seen[state.stable_zone][point.track_id] = point.timestamp

    def _transition(
        self,
        state: _TrackZoneState,
        new_zone: str | None,
        track_id: int,
        timestamp: float,
    ) -> None:
        old_zone = state.stable_zone
        if old_zone is not None:
            self._exits[old_zone] += 1
            if state.entered_at is not None:
                self._dwell_total[old_zone] += max(0.0, timestamp - state.entered_at)
                self._completed_visits[old_zone] += 1

        if new_zone is not None:
            self._entries[new_zone] += 1
            self._unique_last_seen[new_zone][track_id] = timestamp
            if state.last_nonempty_zone is not None and state.last_nonempty_zone != new_zone:
                key = (track_id, state.last_nonempty_zone, new_zone)
                if key not in self._counted_flow_keys:
                    self._flows[(state.last_nonempty_zone, new_zone)] += 1
                    self._counted_flow_keys[key] = None
                    max_keys = self._config.max_active_tracks * 4
                    while len(self._counted_flow_keys) > max_keys:
                        self._counted_flow_keys.popitem(last=False)
            state.last_nonempty_zone = new_zone
            state.entered_at = timestamp
        else:
            state.entered_at = None
        state.stable_zone = new_zone

    def _evict_stale(self, timestamp: float) -> None:
        stale = [
            track_id
            for track_id, state in self._states.items()
            if timestamp - state.last_seen > self._config.inactive_track_ttl_seconds
        ]
        for track_id in stale:
            state = self._states.pop(track_id)
            if state.stable_zone is not None:
                self._transition(state, None, track_id, state.last_seen)

    def _prune_unique(self, timestamp: float) -> None:
        if self.retain_entire:
            return
        cutoff = timestamp - self._config.live_retention_seconds
        for seen in self._unique_last_seen.values():
            stale = [track_id for track_id, last_seen in seen.items() if last_seen < cutoff]
            for track_id in stale:
                del seen[track_id]

    def _zone_metric(self, zone: ZoneConfig) -> ZoneMetric:
        completed = self._completed_visits[zone.zone_id]
        average = self._dwell_total[zone.zone_id] / completed if completed else 0.0
        return ZoneMetric(
            zone_id=zone.zone_id,
            name=zone.name,
            current_people=self._current[zone.zone_id],
            unique_tracks=len(self._unique_last_seen[zone.zone_id]),
            entry_count=self._entries[zone.zone_id],
            exit_count=self._exits[zone.zone_id],
            average_dwell_seconds=round(average, 3),
            peak_occupancy=self._peak[zone.zone_id],
        )

    def _zone_at(self, x: float, y: float) -> str | None:
        for zone in self._zones:
            if cv2.pointPolygonTest(self._polygons[zone.zone_id], (x, y), False) >= 0:
                return zone.zone_id
        return None
