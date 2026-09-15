from __future__ import annotations

import math
from collections import Counter, OrderedDict, deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import cv2
import numpy as np

from backend.app.core.config import AnalyticsConfig, TrackerConfig, ZoneConfig
from backend.app.schemas import TrackedObject


@dataclass(frozen=True, slots=True)
class TrajectoryPoint:
    camera_id: str
    track_id: int
    frame_id: int
    timestamp: float
    x: float
    y: float
    zone_id: str | None

    def as_csv_row(self) -> dict[str, str | int | float]:
        return {
            "camera_id": self.camera_id,
            "track_id": self.track_id,
            "frame_id": self.frame_id,
            "timestamp": round(self.timestamp, 6),
            "x": round(self.x, 3),
            "y": round(self.y, 3),
            "zone_id": self.zone_id or "",
        }


@dataclass(frozen=True, slots=True)
class CompletedTracklet:
    track_id: int
    points: tuple[TrajectoryPoint, ...]
    confirmed: bool
    first_zone: str | None
    last_zone: str | None
    transition: tuple[str, str] | None


@dataclass(frozen=True, slots=True)
class PointTrackletUpdate:
    accepted_points: tuple[TrajectoryPoint, ...]
    persist_points: tuple[TrajectoryPoint, ...]
    completed: tuple[CompletedTracklet, ...]


@dataclass(slots=True)
class _TrackletState:
    last_seen_frame: int
    accepted_count: int = 0
    confirmed: bool = False
    current_x: float | None = None
    current_y: float | None = None
    stable_zone: str | None = None
    candidate_zone: str | None = None
    candidate_count: int = 0
    first_zone: str | None = None
    last_zone: str | None = None


class PointTrackletManager:
    """Bounded point histories, delayed persistence, and unique entry/exit flows."""

    def __init__(
        self,
        analytics: AnalyticsConfig,
        tracker: TrackerConfig,
        zones: Sequence[ZoneConfig],
        *,
        camera_id: str = "cam01",
    ) -> None:
        if analytics.trajectory_history_points < analytics.min_confirmed_points:
            raise ValueError("trajectory history must hold all confirmation points")
        self.analytics = analytics
        self.track_buffer = tracker.track_buffer
        self.camera_id = camera_id
        self.zones = tuple(zones)
        self._polygons = {
            zone.zone_id: np.asarray(zone.points, dtype=np.float32) for zone in self.zones
        }
        self._zone_names = {zone.zone_id: zone.name for zone in self.zones}
        self._histories: dict[int, deque[TrajectoryPoint]] = {}
        self._states: dict[int, _TrackletState] = {}
        self._flows: Counter[tuple[str, str]] = Counter()
        self._finalized_ids: OrderedDict[int, None] = OrderedDict()
        self.completed_tracklets = 0
        self.discarded_short_tracklets = 0

    @property
    def active_tracklet_count(self) -> int:
        return len(self._histories)

    @property
    def buffered_point_count(self) -> int:
        return sum(len(history) for history in self._histories.values())

    def histories(
        self, track_ids: Iterable[int] | None = None
    ) -> dict[int, tuple[TrajectoryPoint, ...]]:
        selected = set(track_ids) if track_ids is not None else set(self._histories)
        return {
            track_id: tuple(self._histories[track_id])
            for track_id in sorted(selected)
            if track_id in self._histories
        }

    def update(
        self,
        tracks: Iterable[TrackedObject],
        *,
        frame_id: int,
        timestamp: float,
    ) -> PointTrackletUpdate:
        completed = list(self._finalize_stale(frame_id))
        accepted: list[TrajectoryPoint] = []
        persist: list[TrajectoryPoint] = []

        for track in tracks:
            state = self._states.get(track.track_id)
            if state is None:
                overflow = self._reserve_track_slot()
                if overflow is not None:
                    completed.append(overflow)
                state = _TrackletState(last_seen_frame=frame_id)
                self._states[track.track_id] = state
                self._histories[track.track_id] = deque(
                    maxlen=self.analytics.trajectory_history_points
                )

            state.last_seen_frame = frame_id
            raw_x, raw_y = track.bottom_center
            if state.current_x is None or state.current_y is None:
                x, y = raw_x, raw_y
            else:
                alpha = self.analytics.trajectory_smoothing_alpha
                x = alpha * raw_x + (1.0 - alpha) * state.current_x
                y = alpha * raw_y + (1.0 - alpha) * state.current_y
            state.current_x, state.current_y = x, y
            self._observe_zone(state, x, y)

            history = self._histories[track.track_id]
            if history:
                distance = math.hypot(x - history[-1].x, y - history[-1].y)
                if distance < self.analytics.movement_threshold_pixels:
                    continue
                if distance > self.analytics.max_movement_step_pixels:
                    state.current_x, state.current_y = history[-1].x, history[-1].y
                    continue

            point = TrajectoryPoint(
                camera_id=self.camera_id,
                track_id=track.track_id,
                frame_id=frame_id,
                timestamp=timestamp,
                x=x,
                y=y,
                zone_id=state.stable_zone,
            )
            history.append(point)
            state.accepted_count += 1
            accepted.append(point)
            if state.confirmed:
                persist.append(point)
            elif state.accepted_count >= self.analytics.min_confirmed_points:
                state.confirmed = True
                persist.extend(history)

        return PointTrackletUpdate(tuple(accepted), tuple(persist), tuple(completed))

    def finalize_all(self) -> tuple[CompletedTracklet, ...]:
        return tuple(self._finalize(track_id) for track_id in sorted(self._states))

    def zone_flows(self) -> tuple[dict[str, str | int], ...]:
        return tuple(
            {
                "from_zone": source,
                "from_name": self._zone_names.get(source, source),
                "to_zone": target,
                "to_name": self._zone_names.get(target, target),
                "unique_track_ids": count,
            }
            for (source, target), count in sorted(
                self._flows.items(), key=lambda item: (-item[1], item[0])
            )
        )

    def _finalize_stale(self, frame_id: int) -> tuple[CompletedTracklet, ...]:
        stale = [
            track_id
            for track_id, state in self._states.items()
            if frame_id - state.last_seen_frame > self.track_buffer
        ]
        return tuple(self._finalize(track_id) for track_id in stale)

    def _reserve_track_slot(self) -> CompletedTracklet | None:
        if len(self._states) < self.analytics.max_active_tracks:
            return None
        oldest_id = min(
            self._states,
            key=lambda track_id: self._states[track_id].last_seen_frame,
        )
        return self._finalize(oldest_id)

    def _finalize(self, track_id: int) -> CompletedTracklet:
        state = self._states.pop(track_id)
        points = tuple(self._histories.pop(track_id))
        transition: tuple[str, str] | None = None
        if state.confirmed:
            self.completed_tracklets += 1
            if (
                track_id not in self._finalized_ids
                and state.first_zone is not None
                and state.last_zone is not None
                and state.first_zone != state.last_zone
            ):
                transition = (state.first_zone, state.last_zone)
                self._flows[transition] += 1
        else:
            self.discarded_short_tracklets += 1

        self._finalized_ids[track_id] = None
        self._finalized_ids.move_to_end(track_id)
        while len(self._finalized_ids) > self.analytics.max_active_tracks * 4:
            self._finalized_ids.popitem(last=False)
        return CompletedTracklet(
            track_id=track_id,
            points=points,
            confirmed=state.confirmed,
            first_zone=state.first_zone,
            last_zone=state.last_zone,
            transition=transition,
        )

    def _observe_zone(self, state: _TrackletState, x: float, y: float) -> None:
        detected = self._zone_at(x, y)
        if detected == state.stable_zone:
            state.candidate_zone = None
            state.candidate_count = 0
            return
        if detected == state.candidate_zone:
            state.candidate_count += 1
        else:
            state.candidate_zone = detected
            state.candidate_count = 1
        if state.candidate_count < self.analytics.zone_debounce_points:
            return
        state.stable_zone = detected
        state.candidate_zone = None
        state.candidate_count = 0
        if detected is not None:
            if state.first_zone is None:
                state.first_zone = detected
            state.last_zone = detected

    def _zone_at(self, x: float, y: float) -> str | None:
        for zone in self.zones:
            polygon = self._polygons[zone.zone_id]
            if cv2.pointPolygonTest(polygon, (x, y), False) >= 0:
                return zone.zone_id
        return None
