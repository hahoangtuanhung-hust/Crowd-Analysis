"""Causal dominant local-flow engine based on currently observed moving tracks."""

from __future__ import annotations

import math
import zlib
from collections import defaultdict, deque
from dataclasses import dataclass, field, replace
from typing import Iterable

from backend.app.analytics.directional_grid import GridTrackPoint, TrackKey
from backend.app.analytics.spatial import SpatialTransformer
from backend.app.core.config import DominantLiveFlowConfig
from backend.app.schemas.analytics import Cell, CommonPath, CommonPathSnapshot


@dataclass(slots=True)
class _LiveTrack:
    points: deque[tuple[float, float, float]]
    smoothed_dx: float = 0.0
    smoothed_dy: float = 0.0
    last_frame: int = -1
    last_observed_s: float = -math.inf


@dataclass(frozen=True, slots=True)
class _Vote:
    key: tuple[str, str, int]
    cell: Cell
    direction_bin: int
    x: float
    y: float


@dataclass(frozen=True, slots=True)
class _Cluster:
    cluster_id: str
    direction_bin: int
    cells: frozenset[Cell]
    votes: tuple[_Vote, ...]
    polyline: tuple[tuple[float, float], ...]

    @property
    def count(self) -> int:
        return len(self.votes)


@dataclass(frozen=True, slots=True)
class DominantLiveFlowStatus:
    state: str = "learning"
    reason: str = "INSUFFICIENT_CURRENT_MOVING_TRACKS"
    active_count: int = 0
    active_direction: str | None = None
    challenger_count: int = 0
    challenger_direction: str | None = None
    confirmation_elapsed_s: float = 0.0
    confirmation_required_s: float = 0.0
    moving_track_count: int = 0
    evaluated_at_s: float = 0.0


class DominantLiveFlowEngine:
    """Select the largest connected current-motion cluster with causal hysteresis."""

    def __init__(
        self,
        config: DominantLiveFlowConfig,
        transformer: SpatialTransformer,
        *,
        camera_id: str = "cam01",
        stream_epoch: str = "initial",
    ) -> None:
        self.config = config
        self.transformer = transformer
        self.camera_id = camera_id
        self.reset(stream_epoch)

    def reset(self, stream_epoch: str) -> None:
        self.stream_epoch = stream_epoch
        self._tracks: dict[tuple[str, str, int], _LiveTrack] = {}
        self._clock: float | None = None
        self._last_evaluation_s: float | None = None
        self._observation_version = 0
        self._last_confirm_version = -1
        self._active: _Cluster | None = None
        self._active_path: CommonPath | None = None
        self._active_missing_since: float | None = None
        self._pending: _Cluster | None = None
        self._pending_since: float | None = None
        self._pending_last_version = -1
        self._cluster_identities: list[tuple[int, frozenset[Cell], str]] = []
        self._identity_counter = 0
        self._version = 0
        self._events: list[dict[str, object]] = []
        self._status = DominantLiveFlowStatus(
            confirmation_required_s=self.config.confirmation_seconds
        )
        self._snapshot = CommonPathSnapshot(0.0, ())

    @property
    def events(self) -> tuple[dict[str, object], ...]:
        return tuple(self._events)

    @property
    def status(self) -> DominantLiveFlowStatus:
        return self._status

    @property
    def retained_tracks(self) -> int:
        return len(self._tracks)

    @property
    def retained_points(self) -> int:
        return sum(len(track.points) for track in self._tracks.values())

    def update(
        self, points: Iterable[GridTrackPoint], event_time_s: float
    ) -> CommonPathSnapshot:
        if not math.isfinite(event_time_s) or (
            self._clock is not None and event_time_s < self._clock
        ):
            raise ValueError("Event time must be finite and monotonic")
        self._clock = event_time_s
        observed_keys: set[tuple[str, str, int]] = set()
        for point in points:
            if point.camera_id != self.camera_id or point.stream_epoch != self.stream_epoch:
                raise ValueError("TrackPoint belongs to another camera/epoch")
            if point.event_time_s != event_time_s:
                raise ValueError("TrackPoint time differs from frame event time")
            if not point.confirmed or not point.observed:
                continue
            key = point.key[:3]
            observed_keys.add(key)
            track = self._tracks.get(key)
            if track is None:
                if len(self._tracks) >= self.config.max_tracks:
                    oldest = min(self._tracks, key=lambda item: self._tracks[item].last_observed_s)
                    del self._tracks[oldest]
                track = _LiveTrack(deque(maxlen=self.config.max_points_per_track))
                self._tracks[key] = track
            if point.frame_id <= track.last_frame or event_time_s <= track.last_observed_s:
                raise ValueError("Track frame/time is out of order")
            track.points.append((event_time_s, point.x, point.y))
            track.last_frame = point.frame_id
            track.last_observed_s = event_time_s
            self._observation_version += 1

        cutoff = event_time_s - self.config.history_seconds
        for key, track in list(self._tracks.items()):
            while track.points and track.points[0][0] < cutoff:
                track.points.popleft()
            if not track.points:
                del self._tracks[key]

        if (
            self._last_evaluation_s is None
            or event_time_s - self._last_evaluation_s
            >= self.config.evaluation_interval_seconds
        ):
            self._last_evaluation_s = event_time_s
            self._evaluate(event_time_s, observed_keys)
        return self._snapshot

    def tick(self, event_time_s: float) -> CommonPathSnapshot:
        return self.update((), event_time_s)

    def snapshot(self) -> CommonPathSnapshot:
        return self._snapshot

    def _direction_bin(self, dx: float, dy: float) -> int:
        angle = math.atan2(dy, dx) % (2 * math.pi)
        return int(
            (angle + math.pi / self.config.direction_bins)
            / (2 * math.pi / self.config.direction_bins)
        ) % self.config.direction_bins

    def _vote(
        self, key: tuple[str, str, int], track: _LiveTrack, now: float,
    ) -> _Vote | None:
        if now - track.last_observed_s > self.config.observation_timeout_seconds:
            return None
        current = track.points[-1]
        candidates = [
            sample for sample in track.points
            if self.config.direction_min_seconds
            <= current[0] - sample[0]
            <= self.config.direction_max_seconds
        ]
        if not candidates:
            return None
        previous = candidates[0]
        dx, dy = current[1] - previous[1], current[2] - previous[2]
        cell_width = self.transformer.width / self.config.columns
        cell_height = self.transformer.height / self.config.rows
        displacement = math.hypot(dx / cell_width, dy / cell_height)
        if displacement < self.config.min_displacement_cell_fraction:
            return None
        alpha = self.config.direction_smoothing_alpha
        track.smoothed_dx = alpha * dx + (1 - alpha) * track.smoothed_dx
        track.smoothed_dy = alpha * dy + (1 - alpha) * track.smoothed_dy
        if math.hypot(track.smoothed_dx / cell_width, track.smoothed_dy / cell_height) < (
            self.config.min_displacement_cell_fraction
        ):
            return None
        cell = self.transformer.grid_cell(
            current[1], current[2],
            grid_width=self.config.columns,
            grid_height=self.config.rows,
        )
        if cell is None:
            return None
        return _Vote(
            key, cell, self._direction_bin(track.smoothed_dx, track.smoothed_dy),
            current[1], current[2],
        )

    def _cluster_identity(self, direction: int, cells: frozenset[Cell]) -> str:
        best: tuple[float, str] | None = None
        for known_direction, known_cells, cluster_id in self._cluster_identities:
            if known_direction != direction:
                continue
            union = len(cells | known_cells)
            overlap = len(cells & known_cells) / union if union else 0.0
            distance = min(
                max(abs(left[0] - right[0]), abs(left[1] - right[1]))
                for left in cells for right in known_cells
            )
            identity_score = overlap if overlap > 0 else (0.01 if distance <= 2 else 0.0)
            if identity_score > 0 and (best is None or identity_score > best[0]):
                best = identity_score, cluster_id
        if best is not None:
            cluster_id = best[1]
            for index, item in enumerate(self._cluster_identities):
                if item[2] == cluster_id:
                    self._cluster_identities[index] = (direction, cells, cluster_id)
                    break
            return cluster_id
        self._identity_counter += 1
        signature = zlib.crc32(repr((direction, sorted(cells))).encode("ascii"))
        cluster_id = f"flow-{direction}-{self._identity_counter}-{signature:08x}"
        self._cluster_identities.append((direction, cells, cluster_id))
        if len(self._cluster_identities) > 128:
            del self._cluster_identities[:-128]
        return cluster_id

    def _polyline(self, votes: list[_Vote], direction: int) -> tuple[tuple[float, float], ...]:
        angle = direction * 2 * math.pi / self.config.direction_bins
        ux, uy = math.cos(angle), math.sin(angle)
        vx, vy = -uy, ux
        projections = [vote.x * ux + vote.y * uy for vote in votes]
        perpendicular = sum(vote.x * vx + vote.y * vy for vote in votes) / len(votes)
        low, high = min(projections), max(projections)
        minimum = self.config.min_geometry_cells * max(
            self.transformer.width / self.config.columns,
            self.transformer.height / self.config.rows,
        )
        if high - low < minimum:
            center = (low + high) / 2
            low, high = center - minimum / 2, center + minimum / 2
        result = []
        for projection in (low, high):
            x = projection * ux + perpendicular * vx
            y = projection * uy + perpendicular * vy
            result.append((
                min(self.transformer.width - 1.0, max(0.0, x)),
                min(self.transformer.height - 1.0, max(0.0, y)),
            ))
        return tuple(result)

    def _clusters(self, votes: list[_Vote]) -> list[_Cluster]:
        by_direction_cell: dict[int, dict[Cell, list[_Vote]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for vote in votes:
            by_direction_cell[vote.direction_bin][vote.cell].append(vote)
        clusters: list[_Cluster] = []
        for direction, by_cell in by_direction_cell.items():
            remaining = set(by_cell)
            while remaining:
                start = remaining.pop()
                component = {start}
                pending = [start]
                while pending:
                    row, column = pending.pop()
                    for dr in (-1, 0, 1):
                        for dc in (-1, 0, 1):
                            neighbor = (row + dr, column + dc)
                            if neighbor in remaining:
                                remaining.remove(neighbor)
                                component.add(neighbor)
                                pending.append(neighbor)
                component_votes = [vote for cell in component for vote in by_cell[cell]]
                cells = frozenset(component)
                clusters.append(_Cluster(
                    self._cluster_identity(direction, cells), direction, cells,
                    tuple(component_votes), self._polyline(component_votes, direction),
                ))
        return sorted(clusters, key=lambda cluster: (-cluster.count, cluster.cluster_id))

    def _same_cluster(self, left: _Cluster | None, right: _Cluster | None) -> bool:
        return left is not None and right is not None and left.cluster_id == right.cluster_id

    def _path(self, cluster: _Cluster, state: str, revision: int, now: float) -> CommonPath:
        names = (
            "east", "southeast", "south", "southwest",
            "west", "northwest", "north", "northeast",
        )
        name = names[round(cluster.direction_bin * 8 / self.config.direction_bins) % 8]
        return CommonPath(
            path_id=f"{self.stream_epoch}-{cluster.cluster_id}",
            origin_zone="dominant_live_flow",
            destination_zone=name,
            state=state,
            unique_tracks_short=cluster.count,
            unique_tracks_long=cluster.count,
            score=float(cluster.count),
            confidence=1.0,
            direction=name,
            polyline=cluster.polyline,
            updated_at=now,
            revision=revision,
            coordinate_space="image_pixels",
            support_tracks=cluster.count,
            validated_complete_tracks=cluster.count,
            evidence_until_s=now,
            stale=state == "cooling",
        )

    def _event(self, event: str, reason: str, cluster: _Cluster | None, now: float) -> None:
        self._events.append({
            "event": event,
            "reason": reason,
            "path_id": f"{self.stream_epoch}-{cluster.cluster_id}" if cluster else "",
            "direction": self._path(cluster, "active", 1, now).direction if cluster else None,
            "count": cluster.count if cluster else 0,
            "event_time_s": now,
            "evidence_until_s": now,
        })

    def _set_pending(self, cluster: _Cluster | None, now: float) -> None:
        if cluster is None:
            self._pending = None
            self._pending_since = None
            self._pending_last_version = -1
        elif not self._same_cluster(self._pending, cluster):
            self._pending = cluster
            self._pending_since = now
            self._pending_last_version = self._observation_version
        elif self._observation_version > self._pending_last_version:
            self._pending = cluster
            self._pending_last_version = self._observation_version

    def _evaluate(self, now: float, observed_keys: set[tuple[str, str, int]]) -> None:
        votes = [
            vote for key, track in self._tracks.items()
            if key in observed_keys and (vote := self._vote(key, track, now)) is not None
        ]
        clusters = self._clusters(votes)
        eligible = [cluster for cluster in clusters if cluster.count >= self.config.min_active_tracks]
        best = eligible[0] if eligible else None
        active_match = next(
            (cluster for cluster in eligible if self._same_cluster(cluster, self._active)), None
        )

        if self._active is None:
            pending_match = next(
                (cluster for cluster in eligible if self._same_cluster(cluster, self._pending)),
                None,
            )
            self._set_pending(pending_match or best, now)
            if (
                self._pending is not None
                and self._pending_since is not None
                and self._observation_version > self._last_confirm_version
                and now - self._pending_since >= self.config.confirmation_seconds
            ):
                self._active = self._pending
                self._active_path = self._path(self._active, "active", 1, now)
                self._event("activate", "initial_confirmation_complete", self._active, now)
                self._last_confirm_version = self._observation_version
                self._set_pending(None, now)
        else:
            if active_match is not None:
                self._active = active_match
                revision = (self._active_path.revision + 1) if self._active_path else 1
                self._active_path = self._path(active_match, "active", revision, now)
                self._active_missing_since = None
            elif self._active_missing_since is None:
                self._active_missing_since = now
            challengers = [
                cluster for cluster in eligible
                if not self._same_cluster(cluster, self._active)
                and cluster.count >= (
                    (active_match.count if active_match else 0)
                    + self.config.challenger_margin_tracks
                )
            ]
            challenger = next(
                (cluster for cluster in challengers if self._same_cluster(cluster, self._pending)),
                challengers[0] if challengers else None,
            )
            self._set_pending(challenger, now)
            if (
                self._pending is not None
                and self._pending_since is not None
                and self._observation_version > self._last_confirm_version
                and now - self._pending_since >= self.config.confirmation_seconds
            ):
                self._active = self._pending
                self._active_path = self._path(self._active, "active", 1, now)
                self._event("switch", "challenger_confirmation_complete", self._active, now)
                self._last_confirm_version = self._observation_version
                self._active_missing_since = None
                self._set_pending(None, now)
            elif (
                active_match is None
                and self._active_missing_since is not None
                and now - self._active_missing_since >= self.config.stale_seconds
            ):
                self._event("retire", "active_flow_stale", self._active, now)
                self._active = None
                self._active_path = None
                self._active_missing_since = None

        paths: tuple[CommonPath, ...] = ()
        state, reason = "learning", "INSUFFICIENT_CURRENT_MOVING_TRACKS"
        active_count = 0
        active_direction = None
        if self._active_path is not None:
            if self._active_missing_since is not None:
                self._active_path = replace(
                    self._active_path, state="cooling", stale=True, updated_at=now
                )
                state, reason = "cooling", "ACTIVE_FLOW_TEMPORARILY_STALE"
            else:
                state, reason = "active", "DOMINANT_CURRENT_FLOW"
            paths += (self._active_path,)
            active_count = self._active_path.unique_tracks_short
            active_direction = self._active_path.direction
        confirmation_elapsed = 0.0
        challenger_count = 0
        challenger_direction = None
        if self._pending is not None and self._pending_since is not None:
            candidate = self._path(self._pending, "candidate", 0, now)
            paths += (candidate,)
            confirmation_elapsed = max(0.0, now - self._pending_since)
            challenger_count = self._pending.count
            challenger_direction = candidate.direction
            if self._active_path is None:
                state, reason = "confirming", "INITIAL_FLOW_CONFIRMING"
            else:
                reason = "CHALLENGER_CONFIRMING"
        self._version += 1
        self._snapshot = CommonPathSnapshot(
            now, paths, self._version,
            max((path.evidence_until_s for path in paths), default=0.0),
        )
        self._status = DominantLiveFlowStatus(
            state=state,
            reason=reason,
            active_count=active_count,
            active_direction=active_direction,
            challenger_count=challenger_count,
            challenger_direction=challenger_direction,
            confirmation_elapsed_s=confirmation_elapsed,
            confirmation_required_s=self.config.confirmation_seconds,
            moving_track_count=len(votes),
            evaluated_at_s=now,
        )
