from __future__ import annotations

import heapq
import math
import statistics
import time
from collections import Counter, OrderedDict, deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from itertools import pairwise
from typing import Literal, Protocol

import cv2
import numpy as np

from backend.app.analytics.spatial import SpatialTransformer
from backend.app.core.config import AnalyticsConfig, CommonPathConfig, ZoneConfig
from backend.app.schemas import (
    CommonPath,
    CommonPathSnapshot,
    CommonPathTimelinePoint,
    DirectedFlowEdge,
    DirectedFlowSnapshot,
)

Cell = tuple[int, int]
Edge = tuple[Cell, Cell]
Route = tuple[Cell, ...]
OriginDestination = tuple[str, str]
TrackLifecycle = Literal["tentative", "active", "lost", "completed", "discarded"]


class PointLike(Protocol):
    track_id: int
    frame_id: int
    timestamp: float
    x: float
    y: float
    confidence: float


@dataclass(slots=True)
class FlowBucket:
    start_time: float
    end_time: float
    edge_unique_tracks: dict[Edge, set[int]]
    origin_destination_tracks: dict[OriginDestination, dict[Edge, set[int]]]
    routes_by_origin_destination: dict[OriginDestination, dict[int, Route]]


@dataclass(slots=True)
class _TrackState:
    track_id: int
    lifecycle: TrackLifecycle
    first_seen_at: float
    last_seen_at: float
    last_frame_id: int
    last_x: float
    last_y: float
    point_count: int = 1
    total_distance: float = 0.0
    confidence_total: float = 0.0
    stable_cell: Cell | None = None
    candidate_cell: Cell | None = None
    candidate_cell_count: int = 0
    visited_edges: set[Edge] | None = None
    pending_edges: list[tuple[float, Edge]] | None = None
    journey_cells: list[Cell] | None = None
    stable_zone: str | None = None
    candidate_zone: str | None = None
    candidate_zone_count: int = 0
    last_nonempty_zone: str | None = None
    origin_zone: str | None = None
    destination_zone: str | None = None
    last_zone_transition_at: float | None = None
    validated: bool = False

    def __post_init__(self) -> None:
        self.visited_edges = set() if self.visited_edges is None else self.visited_edges
        self.pending_edges = [] if self.pending_edges is None else self.pending_edges
        self.journey_cells = [] if self.journey_cells is None else self.journey_cells


@dataclass(frozen=True, slots=True)
class _Candidate:
    origin_destination: OriginDestination
    route: Route
    edges: frozenset[Edge]
    polyline: tuple[tuple[float, float], ...]
    score: float
    confidence: float
    unique_tracks_short: int
    unique_tracks_long: int


@dataclass(slots=True)
class _PathState:
    active: CommonPath | None = None
    active_edges: frozenset[Edge] = frozenset()
    phase: Literal["active", "cooling"] = "active"
    cooling_since: float | None = None
    pending: _Candidate | None = None
    pending_since: float | None = None
    sequence: int = 0


@dataclass(slots=True)
class _CoolingPath:
    path: CommonPath
    edges: frozenset[Edge]
    started_at: float


class CommonPathAnalyzer:
    """Bounded unique-track directed flow and stable common-path snapshots."""

    def __init__(self, config: AnalyticsConfig, transformer: SpatialTransformer) -> None:
        self.config = config
        self.settings: CommonPathConfig = config.common_path
        self.transformer = transformer
        self._zones: tuple[ZoneConfig, ...] = tuple(config.zones)
        self._polygons = {
            zone.zone_id: np.asarray(zone.points, dtype=np.float32) for zone in self._zones
        }
        self._buckets: OrderedDict[int, FlowBucket] = OrderedDict()
        self._tracks: dict[int, _TrackState] = {}
        self._finalized_track_ids: OrderedDict[int, None] = OrderedDict()
        self._path_states: dict[OriginDestination, _PathState] = {}
        self._cooling_paths: list[_CoolingPath] = []
        self._retired_once: list[CommonPath] = []
        self._snapshot = CommonPathSnapshot(timestamp=0.0, paths=())
        self._flow_snapshot = DirectedFlowSnapshot(
            from_timestamp=0.0,
            to_timestamp=0.0,
            grid_columns=self.settings.grid_columns,
            grid_rows=self.settings.grid_rows,
            edges=(),
        )
        self._last_flow_snapshot_at: float | None = None
        self._timeline: deque[CommonPathTimelinePoint] = deque(
            maxlen=max(
                128,
                math.ceil(
                    self.settings.long_window_seconds
                    / self.settings.update_interval_seconds
                )
                * self.settings.top_k
                * 2,
            )
        )
        self._latest_timestamp = 0.0
        self._last_compute_at: float | None = None
        self._compute_ms: deque[float] = deque(maxlen=256)
        self.completed_tracks = 0
        self.valid_tracks = 0
        self.discarded_tracks = 0
        self.common_path_switches = 0
        self.candidate_rejections = 0

    @property
    def bucket_count(self) -> int:
        return len(self._buckets)

    @property
    def tracked_state_count(self) -> int:
        return len(self._tracks)

    def process_points(
        self,
        points: Iterable[PointLike],
        *,
        active_track_ids: Iterable[int],
        timestamp: float,
    ) -> CommonPathSnapshot:
        if timestamp < self._latest_timestamp:
            self.reset()
        self._latest_timestamp = max(self._latest_timestamp, timestamp)
        self._prune_buckets(timestamp)

        active_ids = set(active_track_ids)
        for state in self._tracks.values():
            if state.track_id not in active_ids and state.lifecycle == "active":
                state.lifecycle = "lost"

        ordered = sorted(points, key=lambda item: (item.timestamp, item.frame_id, item.track_id))
        for point in ordered:
            self._observe(point)
        for track_id in active_ids:
            state = self._tracks.get(track_id)
            if state is None:
                continue
            state.last_seen_at = max(state.last_seen_at, timestamp)
            self._update_eligibility(state)
        self._finalize_stale(timestamp)
        return self._maybe_update(timestamp)

    def finalize_all(self, timestamp: float | None = None) -> CommonPathSnapshot:
        final_timestamp = self._latest_timestamp if timestamp is None else timestamp
        for track_id in tuple(self._tracks):
            self._finalize_track(track_id)
        return self._maybe_update(final_timestamp, force=True)

    def reset(self) -> None:
        self._buckets.clear()
        self._tracks.clear()
        self._finalized_track_ids.clear()
        self._path_states.clear()
        self._cooling_paths.clear()
        self._retired_once.clear()
        self._snapshot = CommonPathSnapshot(timestamp=0.0, paths=())
        self._flow_snapshot = DirectedFlowSnapshot(
            from_timestamp=0.0,
            to_timestamp=0.0,
            grid_columns=self.settings.grid_columns,
            grid_rows=self.settings.grid_rows,
            edges=(),
        )
        self._last_flow_snapshot_at = None
        self._timeline.clear()
        self._latest_timestamp = 0.0
        self._last_compute_at = None
        self._compute_ms.clear()
        self.completed_tracks = 0
        self.valid_tracks = 0
        self.discarded_tracks = 0
        self.common_path_switches = 0
        self.candidate_rejections = 0

    def snapshot(self) -> CommonPathSnapshot:
        return self._snapshot

    def timeline(self) -> tuple[CommonPathTimelinePoint, ...]:
        return tuple(self._timeline)

    def flow_snapshot(
        self, timestamp: float | None = None, *, force: bool = False
    ) -> DirectedFlowSnapshot:
        now = self._latest_timestamp if timestamp is None else timestamp
        if (
            not force
            and self._last_flow_snapshot_at is not None
            and now - self._last_flow_snapshot_at
            < self.settings.flow_update_interval_seconds
        ):
            return self._flow_snapshot
        stats = self._edge_statistics(now, origin_destination=None)
        edges = tuple(
            DirectedFlowEdge(
                from_cell=edge[0],
                to_cell=edge[1],
                unique_tracks_short=len(values[1]),
                unique_tracks_long=len(values[2]),
                score=round(values[0], 6),
            )
            for edge, values in sorted(
                stats.items(), key=lambda item: (-item[1][0], item[0])
            )
        )
        self._flow_snapshot = DirectedFlowSnapshot(
            from_timestamp=max(0.0, now - self.settings.long_window_seconds),
            to_timestamp=now,
            grid_columns=self.settings.grid_columns,
            grid_rows=self.settings.grid_rows,
            edges=edges,
        )
        self._last_flow_snapshot_at = now
        return self._flow_snapshot

    def metrics(self) -> dict[str, int | float | None]:
        active = sum(state.lifecycle == "active" for state in self._tracks.values())
        lost = sum(state.lifecycle == "lost" for state in self._tracks.values())
        return {
            "active_tracks": active,
            "lost_tracks": lost,
            "completed_tracks": self.completed_tracks,
            "valid_tracks": self.valid_tracks,
            "discarded_tracks": self.discarded_tracks,
            "flow_bucket_count": self.bucket_count,
            "common_path_switches": self.common_path_switches,
            "candidate_rejections": self.candidate_rejections,
            "common_path_compute_ms": self._percentile(self._compute_ms, 50),
            "common_path_compute_ms_p95": self._percentile(self._compute_ms, 95),
        }

    @staticmethod
    def directed_edge_similarity(edges_a: Iterable[Edge], edges_b: Iterable[Edge]) -> float:
        first = set(edges_a)
        second = set(edges_b)
        return len(first & second) / len(second) if second else 1.0

    def _observe(self, point: PointLike) -> None:
        if point.track_id in self._finalized_track_ids:
            return
        x, y = self.transformer.transform(point.x, point.y)
        state = self._tracks.get(point.track_id)
        if state is None:
            self._reserve_track_slot()
            cell = self._grid_cell(x, y)
            state = _TrackState(
                track_id=point.track_id,
                lifecycle="tentative",
                first_seen_at=point.timestamp,
                last_seen_at=point.timestamp,
                last_frame_id=point.frame_id,
                last_x=x,
                last_y=y,
                confidence_total=point.confidence,
                stable_cell=cell,
            )
            if cell is not None:
                state.journey_cells.append(cell)
            self._tracks[point.track_id] = state
            self._observe_zone(state, x, y, point.timestamp)
            self._update_eligibility(state)
            return
        if point.frame_id <= state.last_frame_id:
            return

        state.last_seen_at = point.timestamp
        state.last_frame_id = point.frame_id
        state.lifecycle = "active" if self._eligible(state) else "tentative"
        distance = math.hypot(x - state.last_x, y - state.last_y)
        state.last_x, state.last_y = x, y
        if distance < self.config.trajectory_min_point_distance_pixels:
            self._observe_zone(state, x, y, point.timestamp)
            return
        if distance > self.settings.max_step_pixels:
            state.candidate_cell = None
            state.candidate_cell_count = 0
            return

        state.point_count += 1
        state.total_distance += distance
        state.confidence_total += point.confidence
        self._observe_cell(state, x, y, point.timestamp)
        self._observe_zone(state, x, y, point.timestamp)
        self._update_eligibility(state)

    def _update_eligibility(self, state: _TrackState) -> None:
        was_active = state.lifecycle == "active"
        if not self._eligible(state):
            if state.lifecycle != "lost":
                state.lifecycle = "tentative"
            return
        state.lifecycle = "active"
        if was_active:
            return
        if not state.validated:
            self.valid_tracks += 1
            state.validated = True
            
            if state.origin_zone is not None and state.destination_zone is not None:
                self._record_origin_destination_route(
                    state.last_seen_at,
                    (state.origin_zone, state.destination_zone),
                    state.track_id,
                    tuple(state.journey_cells),
                )
                
        for timestamp, edge in state.pending_edges:
            self._record_global_edge(timestamp, edge, state.track_id)
        state.pending_edges.clear()

    def _eligible(self, state: _TrackState) -> bool:
        duration = max(0.0, state.last_seen_at - state.first_seen_at)
        speed = state.total_distance / duration if duration > 0 else 0.0
        return (
            duration >= self.settings.min_track_duration_seconds
            and state.total_distance >= self.settings.min_track_distance_pixels
            and state.point_count >= self.settings.min_track_points
            and speed > self.settings.max_stationary_speed_pixels_second
        )

    def _observe_cell(
        self, state: _TrackState, x: float, y: float, timestamp: float
    ) -> None:
        current = self._grid_cell(x, y)
        if current is None:
            return
        if state.stable_cell is None:
            state.stable_cell = current
            state.journey_cells.append(current)
            return
        if current == state.stable_cell:
            state.candidate_cell = None
            state.candidate_cell_count = 0
            return
        if not self._deep_inside_cell(x, y, current):
            return
        if current == state.candidate_cell:
            state.candidate_cell_count += 1
        else:
            state.candidate_cell = current
            state.candidate_cell_count = 1
        if state.candidate_cell_count < self.settings.cell_confirmation_points:
            return

        cells = self._interpolate_cells(state.stable_cell, current)
        for previous, following in pairwise(cells):
            edge = (previous, following)
            if edge not in state.visited_edges:
                state.visited_edges.add(edge)
                if state.lifecycle == "active" or self._eligible(state):
                    self._record_global_edge(timestamp, edge, state.track_id)
                else:
                    state.pending_edges.append((timestamp, edge))
            if not state.journey_cells or state.journey_cells[-1] != following:
                if len(state.journey_cells) < self.settings.max_path_cells:
                    state.journey_cells.append(following)
                else:
                    state.journey_cells[-1] = following
        state.stable_cell = current
        state.candidate_cell = None
        state.candidate_cell_count = 0
        # Override for global dominant path
        state.origin_zone = "scene"
        state.destination_zone = "scene"
        
        if state.lifecycle == "active" or self._eligible(state):
            self._record_origin_destination_route(
                timestamp,
                ("scene", "scene"),
                state.track_id,
                tuple(state.journey_cells),
            )

    def _observe_zone(
        self, state: _TrackState, x: float, y: float, timestamp: float
    ) -> None:
        detected = self._zone_at(x, y)
        if detected == state.stable_zone:
            state.candidate_zone = None
            state.candidate_zone_count = 0
            return
        if detected == state.candidate_zone:
            state.candidate_zone_count += 1
        else:
            state.candidate_zone = detected
            state.candidate_zone_count = 1
        if state.candidate_zone_count < self.settings.zone_min_inside_frames:
            return
        if (
            state.last_zone_transition_at is not None
            and timestamp - state.last_zone_transition_at
            < self.settings.zone_debounce_seconds
        ):
            return
        self._transition_zone(state, detected, timestamp)
        state.candidate_zone = None
        state.candidate_zone_count = 0

    def _transition_zone(
        self, state: _TrackState, new_zone: str | None, timestamp: float
    ) -> None:
        state.stable_zone = new_zone
        state.last_zone_transition_at = timestamp
        
        # Override for global dominant path
        state.origin_zone = "scene"
        state.destination_zone = "scene"
        state.last_nonempty_zone = "scene"

        if self._eligible(state):
            self._record_origin_destination_route(
                timestamp,
                ("scene", "scene"),
                state.track_id,
                tuple(state.journey_cells),
            )

    def _record_global_edge(self, timestamp: float, edge: Edge, track_id: int) -> None:
        bucket = self._bucket(timestamp)
        bucket.edge_unique_tracks.setdefault(edge, set()).add(track_id)

    def _remove_origin_destination_route(
        self, origin_destination: OriginDestination, track_id: int
    ) -> None:
        for bucket in self._buckets.values():
            routes = bucket.routes_by_origin_destination.get(origin_destination)
            if routes and track_id in routes:
                route = routes.pop(track_id)
                edge_tracks = bucket.origin_destination_tracks.get(origin_destination)
                if edge_tracks:
                    for edge in pairwise(route):
                        tracks = edge_tracks.get(edge)
                        if tracks and track_id in tracks:
                            tracks.remove(track_id)
                            if not tracks:
                                del edge_tracks[edge]

    def _record_origin_destination_route(
        self,
        timestamp: float,
        origin_destination: OriginDestination,
        track_id: int,
        route: Route,
    ) -> None:
        route = self._deduplicate_route(route)
        if len(route) < 2:
            return
        bucket = self._bucket(timestamp)
        
        old_route = bucket.routes_by_origin_destination.get(origin_destination, {}).get(track_id)
        if old_route is not None:
            edge_tracks = bucket.origin_destination_tracks.get(origin_destination, {})
            for edge in pairwise(old_route):
                tracks = edge_tracks.get(edge)
                if tracks and track_id in tracks:
                    tracks.remove(track_id)
                    if not tracks:
                        del edge_tracks[edge]
                        
        edge_tracks = bucket.origin_destination_tracks.setdefault(origin_destination, {})
        for edge in pairwise(route):
            edge_tracks.setdefault(edge, set()).add(track_id)
        bucket.routes_by_origin_destination.setdefault(origin_destination, {})[track_id] = route

    def _finalize_stale(self, timestamp: float) -> None:
        stale = [
            track_id
            for track_id, state in self._tracks.items()
            if timestamp - state.last_seen_at > self.settings.track_lost_timeout_seconds
        ]
        for track_id in stale:
            self._finalize_track(track_id)

    def _finalize_track(self, track_id: int) -> None:
        state = self._tracks.pop(track_id)
        if self._eligible(state):
            state.lifecycle = "completed"
            self.completed_tracks += 1
        else:
            state.lifecycle = "discarded"
            self.discarded_tracks += 1
        self._finalized_track_ids[track_id] = None
        self._finalized_track_ids.move_to_end(track_id)
        maximum = self.config.max_active_tracks * 4
        while len(self._finalized_track_ids) > maximum:
            self._finalized_track_ids.popitem(last=False)

    def _reserve_track_slot(self) -> None:
        if len(self._tracks) < self.config.max_active_tracks:
            return
        oldest = min(self._tracks, key=lambda track_id: self._tracks[track_id].last_seen_at)
        self._finalize_track(oldest)

    def _maybe_update(self, timestamp: float, *, force: bool = False) -> CommonPathSnapshot:
        if not self.settings.enabled:
            return self._snapshot
        if (
            not force
            and self._last_compute_at is not None
            and timestamp - self._last_compute_at < self.settings.update_interval_seconds
        ):
            return self._snapshot
        started = time.perf_counter()
        candidates = self._extract_candidates(timestamp)
        paths = self._update_states(candidates, timestamp)
        self._snapshot = CommonPathSnapshot(timestamp=timestamp, paths=paths)
        self._last_compute_at = timestamp
        self._compute_ms.append((time.perf_counter() - started) * 1000.0)
        for path in paths:
            self._timeline.append(
                CommonPathTimelinePoint(
                    timestamp=timestamp,
                    path_id=path.path_id,
                    state=path.state,
                    score=path.score,
                    confidence=path.confidence,
                    unique_tracks_short=path.unique_tracks_short,
                    unique_tracks_long=path.unique_tracks_long,
                )
            )
        return self._snapshot

    def _extract_candidates(self, timestamp: float) -> list[_Candidate]:
        origin_destination_tracks: dict[OriginDestination, set[int]] = {}
        for bucket in self._window_buckets(timestamp, self.settings.long_window_seconds):
            for origin_destination, routes in (
                bucket.routes_by_origin_destination.items()
            ):
                origin_destination_tracks.setdefault(origin_destination, set()).update(routes)
        origin_destinations = [
            origin_destination
            for origin_destination, track_ids in sorted(
                origin_destination_tracks.items(),
                key=lambda item: (-len(item[1]), item[0]),
            )
            if len(track_ids) >= self.settings.min_unique_tracks
        ][: self.settings.top_k]

        candidates: list[_Candidate] = []
        for origin_destination in sorted(origin_destinations):
            stats = self._edge_statistics(timestamp, origin_destination=origin_destination)
            supported = {
                edge: values
                for edge, values in stats.items()
                if len(values[2]) >= self.settings.min_path_edge_support
            }
            if not supported:
                continue
            connected: list[_Candidate] = []
            for start, end in self._route_endpoint_pairs(
                timestamp, origin_destination
            )[:4]:
                route = self._dijkstra(start, end, supported)
                if len(route) < 2:
                    continue
                edges = frozenset(pairwise(route))
                short_tracks, long_tracks = self._supporting_tracks(
                    timestamp, origin_destination, edges
                )
                if not long_tracks:
                    continue
                edge_scores = [supported[edge][0] for edge in pairwise(route)]
                score = statistics.fmean(edge_scores)
                continuity = (
                    min(edge_scores) / max(edge_scores) if max(edge_scores) > 0 else 0.0
                )
                support_factor = 1.0 - math.exp(
                    -len(long_tracks) / max(1.0, float(self.settings.min_unique_tracks))
                )
                confidence = min(1.0, 0.7 * support_factor + 0.3 * continuity)
                connected.append(
                    _Candidate(
                        origin_destination=origin_destination,
                        route=route,
                        edges=edges,
                        polyline=self._route_polyline(route),
                        score=round(score, 6),
                        confidence=round(confidence, 6),
                        unique_tracks_short=len(short_tracks),
                        unique_tracks_long=len(long_tracks),
                    )
                )
            if connected:
                candidates.append(
                    max(
                        connected,
                        key=lambda item: (
                            item.unique_tracks_long,
                            item.score,
                            -len(item.route),
                        ),
                    )
                )
        return sorted(candidates, key=lambda item: (-item.score, item.origin_destination))

    def _update_states(
        self, candidates: Sequence[_Candidate], timestamp: float
    ) -> tuple[CommonPath, ...]:
        candidate_by_od = {candidate.origin_destination: candidate for candidate in candidates}
        all_origin_destinations = set(self._path_states) | set(candidate_by_od)
        visible_candidates: list[CommonPath] = []
        self._retired_once.clear()

        for origin_destination in sorted(all_origin_destinations):
            state = self._path_states.setdefault(origin_destination, _PathState())
            candidate = candidate_by_od.get(origin_destination)
            qualified = (
                candidate is not None
                and candidate.unique_tracks_long >= self.settings.min_unique_tracks
            )
            if state.active is None:
                if qualified and candidate is not None:
                    self._advance_pending(state, candidate, timestamp)
                    if self._pending_confirmed(state, timestamp):
                        self._promote(state, candidate, timestamp)
                elif candidate is not None:
                    self._clear_pending(state, rejected=state.pending is not None)
                if state.pending is not None:
                    visible_candidates.append(self._candidate_path(state.pending, timestamp))
                continue

            state.active = self._refresh_active(
                state.active,
                state.active_edges,
                origin_destination,
                timestamp,
            )
            if candidate is None or not qualified:
                self._clear_pending(state, rejected=state.pending is not None)
                if state.active.unique_tracks_long >= self.settings.min_unique_tracks:
                    state.phase = "active"
                    state.cooling_since = None
                else:
                    self._start_or_advance_cooling(state, timestamp)
                continue

            similarity = self.directed_edge_similarity(state.active_edges, candidate.edges)
            if similarity >= self.settings.path_similarity_threshold:
                state.active = self._merge_active(state.active, candidate, timestamp)
                state.active_edges = candidate.edges
                state.phase = "active"
                state.cooling_since = None
                self._clear_pending(state, rejected=False)
                continue

            threshold = state.active.score * (1.0 + self.settings.switch_margin)
            if candidate.score < threshold:
                self.candidate_rejections += 1
                self._clear_pending(state, rejected=False)
                continue
            self._advance_pending(state, candidate, timestamp)
            if self._pending_confirmed(state, timestamp):
                self._cooling_paths.append(
                    _CoolingPath(
                        path=replace(state.active, state="cooling", updated_at=timestamp),
                        edges=state.active_edges,
                        started_at=timestamp,
                    )
                )
                self._promote(state, candidate, timestamp)
                self.common_path_switches += 1
            elif state.pending is not None:
                visible_candidates.append(self._candidate_path(state.pending, timestamp))

        retained_cooling: list[_CoolingPath] = []
        for cooling in self._cooling_paths:
            if timestamp - cooling.started_at > self.settings.cooling_seconds:
                self._retired_once.append(
                    replace(cooling.path, state="retired", updated_at=timestamp)
                )
            else:
                retained_cooling.append(cooling)
        self._cooling_paths = retained_cooling

        paths = [
            state.active
            for state in self._path_states.values()
            if state.active is not None
        ]
        paths.extend(visible_candidates)
        paths.extend(item.path for item in self._cooling_paths)
        paths.extend(self._retired_once)
        priority = {"active": 0, "candidate": 1, "cooling": 2, "retired": 3}
        paths.sort(key=lambda path: (priority[path.state], -path.score, path.path_id))
        return tuple(paths[: self.settings.top_k])

    def _advance_pending(
        self, state: _PathState, candidate: _Candidate, timestamp: float
    ) -> None:
        if state.pending is None:
            state.pending = candidate
            state.pending_since = timestamp
            return
        similarity = self.directed_edge_similarity(state.pending.edges, candidate.edges)
        if similarity < self.settings.path_similarity_threshold:
            self.candidate_rejections += 1
            state.pending_since = timestamp
        state.pending = candidate

    def _pending_confirmed(self, state: _PathState, timestamp: float) -> bool:
        return (
            state.pending is not None
            and state.pending_since is not None
            and timestamp - state.pending_since >= self.settings.confirmation_seconds
        )

    def _promote(
        self, state: _PathState, candidate: _Candidate, timestamp: float
    ) -> None:
        state.sequence += 1
        state.active = self._candidate_path(
            candidate,
            timestamp,
            state="active",
            path_id=self._path_id(candidate.origin_destination, state.sequence),
        )
        state.active_edges = candidate.edges
        state.phase = "active"
        state.cooling_since = None
        state.pending = None
        state.pending_since = None

    def _start_or_advance_cooling(self, state: _PathState, timestamp: float) -> None:
        if state.active is None:
            return
        if state.phase == "active":
            state.phase = "cooling"
            state.cooling_since = timestamp
            state.active = replace(state.active, state="cooling", updated_at=timestamp)
            return
        if (
            state.cooling_since is not None
            and timestamp - state.cooling_since >= self.settings.cooling_seconds
        ):
            self._retired_once.append(
                replace(state.active, state="retired", updated_at=timestamp)
            )
            state.active = None
            state.active_edges = frozenset()
            state.cooling_since = None
            state.phase = "active"

    def _merge_active(
        self, active: CommonPath, candidate: _Candidate, timestamp: float
    ) -> CommonPath:
        polyline = self._blend_polylines(
            active.polyline,
            candidate.polyline,
            self.settings.centerline_ema_alpha,
        )
        return replace(
            active,
            state="active",
            unique_tracks_short=candidate.unique_tracks_short,
            unique_tracks_long=candidate.unique_tracks_long,
            score=candidate.score,
            confidence=candidate.confidence,
            polyline=polyline,
            updated_at=timestamp,
        )

    def _refresh_active(
        self,
        active: CommonPath,
        edges: frozenset[Edge],
        origin_destination: OriginDestination,
        timestamp: float,
    ) -> CommonPath:
        statistics_by_edge = self._edge_statistics(
            timestamp, origin_destination=origin_destination
        )
        scores = [statistics_by_edge.get(edge, (0.0, set(), set()))[0] for edge in edges]
        short_tracks, long_tracks = self._supporting_tracks(
            timestamp, origin_destination, edges
        )
        score = statistics.fmean(scores) if scores else 0.0
        nonzero = [value for value in scores if value > 0]
        continuity = min(nonzero) / max(nonzero) if nonzero else 0.0
        support_factor = 1.0 - math.exp(
            -len(long_tracks) / max(1.0, float(self.settings.min_unique_tracks))
        )
        confidence = min(1.0, 0.7 * support_factor + 0.3 * continuity)
        return replace(
            active,
            unique_tracks_short=len(short_tracks),
            unique_tracks_long=len(long_tracks),
            score=round(score, 6),
            confidence=round(confidence, 6),
            updated_at=timestamp,
        )

    def _clear_pending(self, state: _PathState, *, rejected: bool) -> None:
        if rejected:
            self.candidate_rejections += 1
        state.pending = None
        state.pending_since = None

    def _candidate_path(
        self,
        candidate: _Candidate,
        timestamp: float,
        *,
        state: Literal["candidate", "active"] = "candidate",
        path_id: str | None = None,
    ) -> CommonPath:
        origin, destination = candidate.origin_destination
        return CommonPath(
            path_id=path_id or f"{origin}__{destination}__candidate",
            origin_zone=origin,
            destination_zone=destination,
            state=state,
            unique_tracks_short=candidate.unique_tracks_short,
            unique_tracks_long=candidate.unique_tracks_long,
            score=candidate.score,
            confidence=candidate.confidence,
            direction=f"{origin.upper()}_TO_{destination.upper()}",
            polyline=candidate.polyline,
            updated_at=timestamp,
        )

    def _edge_statistics(
        self,
        timestamp: float,
        *,
        origin_destination: OriginDestination | None,
    ) -> dict[Edge, tuple[float, set[int], set[int]]]:
        short_tracks: dict[Edge, set[int]] = {}
        long_tracks: dict[Edge, set[int]] = {}
        short_weighted: Counter[Edge] = Counter()
        long_weighted: Counter[Edge] = Counter()
        for bucket in self._window_buckets(timestamp, self.settings.long_window_seconds):
            if origin_destination is None:
                edge_tracks = bucket.edge_unique_tracks
            else:
                edge_tracks = bucket.origin_destination_tracks.get(origin_destination, {})
            age = max(0.0, timestamp - bucket.end_time)
            long_decay = math.exp(-age / self.settings.long_window_seconds)
            in_short = bucket.end_time > timestamp - self.settings.short_window_seconds
            short_decay = (
                math.exp(-age / self.settings.short_window_seconds) if in_short else 0.0
            )
            for edge, track_ids in edge_tracks.items():
                long_tracks.setdefault(edge, set()).update(track_ids)
                long_weighted[edge] += len(track_ids) * long_decay
                if in_short:
                    short_tracks.setdefault(edge, set()).update(track_ids)
                    short_weighted[edge] += len(track_ids) * short_decay

        maximum_short = max(short_weighted.values(), default=0.0)
        maximum_long = max(long_weighted.values(), default=0.0)
        weight_total = self.settings.short_weight + self.settings.long_weight
        result: dict[Edge, tuple[float, set[int], set[int]]] = {}
        for edge in set(long_tracks) | set(short_tracks):
            normalized_short = (
                short_weighted[edge] / maximum_short if maximum_short > 0 else 0.0
            )
            normalized_long = (
                long_weighted[edge] / maximum_long if maximum_long > 0 else 0.0
            )
            score = (
                self.settings.short_weight * normalized_short
                + self.settings.long_weight * normalized_long
            ) / weight_total
            result[edge] = (
                score,
                short_tracks.get(edge, set()),
                long_tracks.get(edge, set()),
            )
        return result

    def _route_endpoint_pairs(
        self, timestamp: float, origin_destination: OriginDestination
    ) -> list[tuple[Cell, Cell]]:
        pairs: dict[tuple[Cell, Cell], set[int]] = {}
        for bucket in self._window_buckets(timestamp, self.settings.long_window_seconds):
            for track_id, route in bucket.routes_by_origin_destination.get(
                origin_destination, {}
            ).items():
                if len(route) < 2:
                    continue
                pairs.setdefault((route[0], route[-1]), set()).add(track_id)
        return [
            pair
            for pair, _ in sorted(
                pairs.items(),
                key=lambda item: (-len(item[1]), item[0]),
            )
        ]

    def _supporting_tracks(
        self,
        timestamp: float,
        origin_destination: OriginDestination,
        candidate_edges: frozenset[Edge],
    ) -> tuple[set[int], set[int]]:
        short_routes: dict[int, Route] = {}
        long_routes: dict[int, Route] = {}
        short_cutoff = timestamp - self.settings.short_window_seconds
        for bucket in self._window_buckets(timestamp, self.settings.long_window_seconds):
            routes = bucket.routes_by_origin_destination.get(origin_destination, {})
            long_routes.update(routes)
            if bucket.end_time > short_cutoff:
                short_routes.update(routes)

        def supported(routes: dict[int, Route]) -> set[int]:
            result: set[int] = set()
            for track_id, route in routes.items():
                edges = set(pairwise(route))
                if (
                    self.directed_edge_similarity(candidate_edges, edges)
                    >= self.settings.path_similarity_threshold
                ):
                    result.add(track_id)
            return result

        return supported(short_routes), supported(long_routes)

    def _dijkstra(
        self,
        start: Cell,
        end: Cell,
        edge_stats: dict[Edge, tuple[float, set[int], set[int]]],
    ) -> Route:
        if start == end:
            return ()
        adjacency: dict[Cell, list[tuple[Cell, float]]] = {}
        for (source, target), values in edge_stats.items():
            length = math.hypot(target[0] - source[0], target[1] - source[1])
            cost = length / (values[0] + 1e-6)
            adjacency.setdefault(source, []).append((target, cost))
        queue: list[tuple[float, int, Cell, Route]] = [(0.0, 0, start, (start,))]
        best: dict[Cell, float] = {start: 0.0}
        while queue:
            cost, steps, node, route = heapq.heappop(queue)
            if node == end:
                return route
            if cost > best.get(node, math.inf) or steps >= self.settings.max_path_cells:
                continue
            for following, edge_cost in adjacency.get(node, ()):
                if following in route:
                    continue
                new_cost = cost + edge_cost
                if new_cost >= best.get(following, math.inf):
                    continue
                best[following] = new_cost
                heapq.heappush(
                    queue,
                    (new_cost, steps + 1, following, route + (following,)),
                )
        return ()

    def _route_polyline(self, route: Route) -> tuple[tuple[float, float], ...]:
        cell_width = self.transformer.width / self.settings.grid_columns
        cell_height = self.transformer.height / self.settings.grid_rows
        points = [
            self.transformer.inverse_transform(
                (column + 0.5) * cell_width,
                (row + 0.5) * cell_height,
            )
            for row, column in route
        ]
        if len(points) <= 2:
            return tuple(points)
        smoothed = [points[0]]
        for before, current, after in zip(points, points[1:], points[2:]):
            smoothed.append(
                (
                    0.25 * before[0] + 0.5 * current[0] + 0.25 * after[0],
                    0.25 * before[1] + 0.5 * current[1] + 0.25 * after[1],
                )
            )
        smoothed.append(points[-1])
        epsilon = min(cell_width, cell_height) * 0.15
        return tuple(self._rdp(smoothed, epsilon))

    def _bucket(self, timestamp: float) -> FlowBucket:
        key = math.floor(timestamp / self.settings.bucket_seconds)
        bucket = self._buckets.get(key)
        if bucket is None:
            start = key * self.settings.bucket_seconds
            bucket = FlowBucket(
                start_time=float(start),
                end_time=float(start + self.settings.bucket_seconds),
                edge_unique_tracks={},
                origin_destination_tracks={},
                routes_by_origin_destination={},
            )
            self._buckets[key] = bucket
        return bucket

    def _window_buckets(self, timestamp: float, seconds: float) -> list[FlowBucket]:
        cutoff = timestamp - seconds
        return [bucket for bucket in self._buckets.values() if bucket.end_time > cutoff]

    def _prune_buckets(self, timestamp: float) -> None:
        cutoff = timestamp - self.settings.long_window_seconds
        while self._buckets:
            first = next(iter(self._buckets.values()))
            if first.end_time > cutoff:
                break
            self._buckets.popitem(last=False)
        maximum = math.ceil(
            self.settings.long_window_seconds / self.settings.bucket_seconds
        ) + 2
        while len(self._buckets) > maximum:
            self._buckets.popitem(last=False)

    def _grid_cell(self, x: float, y: float) -> Cell | None:
        return self.transformer.grid_cell(
            x,
            y,
            grid_width=self.settings.grid_columns,
            grid_height=self.settings.grid_rows,
        )

    def _deep_inside_cell(self, x: float, y: float, cell: Cell) -> bool:
        row, column = cell
        width = self.transformer.width / self.settings.grid_columns
        height = self.transformer.height / self.settings.grid_rows
        local_x = (x - column * width) / width
        local_y = (y - row * height) / height
        margin = self.settings.cell_hysteresis_ratio
        return margin <= local_x <= 1.0 - margin and margin <= local_y <= 1.0 - margin

    def _zone_at(self, x: float, y: float) -> str | None:
        for zone in self._zones:
            if cv2.pointPolygonTest(self._polygons[zone.zone_id], (x, y), False) >= 0:
                return zone.zone_id
        return None

    def _path_id(self, origin_destination: OriginDestination, sequence: int) -> str:
        return f"{origin_destination[0]}__{origin_destination[1]}__{sequence:02d}"

    @staticmethod
    def _deduplicate_route(route: Route) -> Route:
        result: list[Cell] = []
        for cell in route:
            if not result or result[-1] != cell:
                result.append(cell)
        return tuple(result)

    @staticmethod
    def _interpolate_cells(start: Cell, end: Cell) -> Route:
        row1, column1 = start
        row2, column2 = end
        steps = max(abs(row2 - row1), abs(column2 - column1), 1)
        cells = [
            (round(row), round(column))
            for row, column in zip(
                np.linspace(row1, row2, steps + 1),
                np.linspace(column1, column2, steps + 1),
                strict=True,
            )
        ]
        return CommonPathAnalyzer._deduplicate_route(tuple(cells))

    @staticmethod
    def _blend_polylines(
        first: Sequence[tuple[float, float]],
        second: Sequence[tuple[float, float]],
        alpha: float,
    ) -> tuple[tuple[float, float], ...]:
        size = max(len(first), len(second))
        if size == 0:
            return ()
        first_points = CommonPathAnalyzer._resample(first, size)
        second_points = CommonPathAnalyzer._resample(second, size)
        return tuple(
            (
                (1.0 - alpha) * old[0] + alpha * new[0],
                (1.0 - alpha) * old[1] + alpha * new[1],
            )
            for old, new in zip(first_points, second_points, strict=True)
        )

    @staticmethod
    def _resample(
        points: Sequence[tuple[float, float]], size: int
    ) -> tuple[tuple[float, float], ...]:
        if not points:
            return tuple((0.0, 0.0) for _ in range(size))
        if len(points) == 1:
            return tuple(points[0] for _ in range(size))
        positions = np.linspace(0.0, len(points) - 1, size)
        result = []
        for position in positions:
            lower = math.floor(position)
            upper = min(len(points) - 1, lower + 1)
            fraction = position - lower
            result.append(
                (
                    points[lower][0] * (1.0 - fraction) + points[upper][0] * fraction,
                    points[lower][1] * (1.0 - fraction) + points[upper][1] * fraction,
                )
            )
        return tuple(result)

    @staticmethod
    def _rdp(
        points: Sequence[tuple[float, float]], epsilon: float
    ) -> list[tuple[float, float]]:
        if len(points) < 3:
            return list(points)
        start = np.asarray(points[0], dtype=np.float64)
        end = np.asarray(points[-1], dtype=np.float64)
        line = end - start
        length = float(np.linalg.norm(line))
        distances = []
        for point in points[1:-1]:
            vector = np.asarray(point, dtype=np.float64) - start
            distance = (
                float(np.linalg.norm(vector))
                if length == 0
                else abs(float(line[0] * vector[1] - line[1] * vector[0])) / length
            )
            distances.append(distance)
        maximum = max(distances, default=0.0)
        if maximum <= epsilon:
            return [points[0], points[-1]]
        index = distances.index(maximum) + 1
        left = CommonPathAnalyzer._rdp(points[: index + 1], epsilon)
        right = CommonPathAnalyzer._rdp(points[index:], epsilon)
        return left[:-1] + right

    @staticmethod
    def _percentile(values: deque[float], percentile: int) -> float | None:
        if not values:
            return None
        return round(float(np.percentile(np.asarray(values), percentile)), 3)
