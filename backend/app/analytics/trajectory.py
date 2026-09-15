from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass

from backend.app.core.config import AnalyticsConfig
from backend.app.schemas import TrackedObject, TrackPoint, Trajectory


@dataclass(slots=True)
class _TrackState:
    points: deque[TrackPoint]
    age: int
    last_seen_timestamp: float


class TrajectoryManager:
    """Build bounded, smoothed trajectories from confirmed tracker observations."""

    def __init__(self, config: AnalyticsConfig) -> None:
        self._config = config
        self._tracks: dict[int, _TrackState] = {}

    @property
    def track_count(self) -> int:
        return len(self._tracks)

    def reset(self) -> None:
        self._tracks.clear()

    def update(
        self,
        tracks: Iterable[TrackedObject],
        *,
        frame_id: int,
        timestamp: float,
    ) -> tuple[Trajectory, ...]:
        self.evict_stale(timestamp)
        observed_ids: list[int] = []

        for track in tracks:
            state = self._tracks.get(track.track_id)
            if state is None:
                self._reserve_track_slot()
                state = _TrackState(
                    points=deque(maxlen=self._config.trajectory_history_points),
                    age=0,
                    last_seen_timestamp=timestamp,
                )
                self._tracks[track.track_id] = state

            raw_x, raw_y = track.bottom_center
            if state.points:
                previous = state.points[-1]
                alpha = self._config.trajectory_smoothing_alpha
                x = alpha * raw_x + (1.0 - alpha) * previous.x
                y = alpha * raw_y + (1.0 - alpha) * previous.y
            else:
                x, y = raw_x, raw_y

            state.age += 1
            state.last_seen_timestamp = timestamp
            state.points.append(
                TrackPoint(
                    track_id=track.track_id,
                    timestamp=timestamp,
                    frame_id=frame_id,
                    x=x,
                    y=y,
                    raw_x=raw_x,
                    raw_y=raw_y,
                    confidence=track.confidence,
                )
            )
            observed_ids.append(track.track_id)

        return tuple(self._snapshot(track_id) for track_id in sorted(set(observed_ids)))

    def get(self, track_id: int) -> Trajectory | None:
        if track_id not in self._tracks:
            return None
        return self._snapshot(track_id)

    def snapshots(self, *, confirmed_only: bool = False) -> tuple[Trajectory, ...]:
        trajectories = tuple(self._snapshot(track_id) for track_id in sorted(self._tracks))
        if confirmed_only:
            return tuple(item for item in trajectories if item.confirmed)
        return trajectories

    def evict_stale(self, timestamp: float) -> tuple[int, ...]:
        stale = tuple(
            track_id
            for track_id, state in self._tracks.items()
            if timestamp - state.last_seen_timestamp > self._config.inactive_track_ttl_seconds
        )
        for track_id in stale:
            del self._tracks[track_id]
        return stale

    def _reserve_track_slot(self) -> None:
        if len(self._tracks) < self._config.max_active_tracks:
            return
        oldest_id = min(
            self._tracks,
            key=lambda track_id: self._tracks[track_id].last_seen_timestamp,
        )
        del self._tracks[oldest_id]

    def _snapshot(self, track_id: int) -> Trajectory:
        state = self._tracks[track_id]
        return Trajectory(
            track_id=track_id,
            points=tuple(state.points),
            age=state.age,
            confirmed=state.age >= self._config.min_confirmed_points,
            last_seen_timestamp=state.last_seen_timestamp,
        )
