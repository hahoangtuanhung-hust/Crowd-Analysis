"""Event-time, bounded directional-grid Common Path engine (no detector dependencies)."""

from __future__ import annotations

import math
import zlib
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field, replace
from typing import Any, Iterable

import cv2
import numpy as np

from backend.app.analytics.spatial import SpatialTransformer
from backend.app.core.config import DirectionalGridConfig, ZoneConfig
from backend.app.schemas.analytics import (
    Cell, CommonPath, CommonPathSnapshot, DirectedFlowEdge, DirectedFlowSnapshot,
)

TrackKey = tuple[str, str, int, int]
Edge = tuple[Cell, Cell]


@dataclass(frozen=True, slots=True)
class GridTrackPoint:
    camera_id: str
    stream_epoch: str
    track_id: int
    segment_id: int
    frame_id: int
    event_time_s: float
    x: float
    y: float
    confirmed: bool = True
    observed: bool = True

    @property
    def key(self) -> TrackKey:
        return self.camera_id, self.stream_epoch, self.track_id, self.segment_id


@dataclass(slots=True)
class _Track:
    points: deque[tuple[float, float, float]]
    first_s: float
    last_s: float
    last_frame: int
    cell: Cell | None
    cells: list[Cell] = field(default_factory=list)
    pending: Cell | None = None
    pending_count: int = 0


@dataclass(frozen=True, slots=True)
class _Route:
    cells: tuple[Cell, ...]
    support: int
    complete: int
    score: float
    evidence_s: float
    scope: str = "global_od"
    direction: str = "forward"
    source_gate: str = "scene_boundary"
    target_gate: str = "scene_boundary"
    candidate_id: str = ""
    evaluation_id: str = ""
    diagnostics: dict[str, Any] = field(default_factory=dict, compare=False)


class DirectionalGridEngine:
    """One instance belongs to one camera epoch; all times are media/event time."""

    def __init__(
        self, config: DirectionalGridConfig, transformer: SpatialTransformer,
        *, camera_id: str = "cam01", stream_epoch: str = "initial",
        zones: Iterable[ZoneConfig] = (), valid_cells: set[Cell] | None = None,
        run_id: str = "run", variant: str = "default",
    ) -> None:
        self.config = config
        self.transformer = transformer
        self.camera_id = camera_id
        self.zones = tuple(zones)
        self.run_id = run_id
        self.variant = variant
        self._explicit_valid_cells = valid_cells
        self._source_cells: set[Cell] = set()
        self._target_cells: set[Cell] = set()
        self.valid_cells = valid_cells
        self._configure_scope()
        self.reset(stream_epoch)

    def reset(self, stream_epoch: str) -> None:
        self.stream_epoch = stream_epoch
        self._tracks: dict[TrackKey, _Track] = {}
        self._segments: dict[tuple[str, str, int], int] = {}
        self._bins: dict[tuple[Cell, int], dict[TrackKey, float]] = defaultdict(dict)
        self._edges: dict[Edge, dict[TrackKey, float]] = defaultdict(dict)
        self._occupancy: dict[Cell, dict[TrackKey, float]] = defaultdict(dict)
        self._clock: float | None = None
        self._start: float | None = None
        self._last_extract: float | None = None
        self._version = 0
        self._active: CommonPath | None = None
        self._pending: _Route | None = None
        self._pending_since: float | None = None
        self._pending_last_evidence: float | None = None
        self._cooling_since: float | None = None
        self._retired_once: CommonPath | None = None
        self._events: list[dict[str, object]] = []
        self._candidate_evaluations: list[dict[str, Any]] = []
        self._graph_diagnostics: list[dict[str, Any]] = []
        self._tracklet_rejections: Counter[str] = Counter()
        self._candidate_decisions: Counter[str] = Counter()
        self._latest_routes: tuple[_Route, ...] = ()
        self._route_identities: list[tuple[tuple[Cell, ...], str]] = []
        self._evaluation_count = 0
        self._diagnostic_dropped = 0
        self._overflow = 0
        self._rejected_jumps = 0
        self._snapshot = CommonPathSnapshot(0.0, ())

    @property
    def events(self) -> tuple[dict[str, object], ...]:
        return tuple(self._events)

    @property
    def overflow(self) -> int:
        return self._overflow

    @property
    def retained_points(self) -> int:
        return sum(len(track.points) for track in self._tracks.values())

    @property
    def rejected_jumps(self) -> int:
        return self._rejected_jumps

    @property
    def candidate_evaluations(self) -> tuple[dict[str, Any], ...]:
        return tuple(self._candidate_evaluations)

    @property
    def graph_diagnostics(self) -> tuple[dict[str, Any], ...]:
        return tuple(self._graph_diagnostics)

    @property
    def tracklet_rejection_counts(self) -> dict[str, int]:
        return dict(self._tracklet_rejections)

    @property
    def candidate_decision_counts(self) -> dict[str, int]:
        return dict(self._candidate_decisions)

    @property
    def latest_routes(self) -> tuple[_Route, ...]:
        return self._latest_routes

    @property
    def diagnostic_dropped(self) -> int:
        return self._diagnostic_dropped

    def _configure_scope(self) -> None:
        corridor = self.config.local_corridor
        if self.config.route_scope != "local_corridor" or corridor is None:
            return
        roi_cells = self._polygon_cells(corridor.roi, corridor.coordinate_space)
        self._source_cells = self._polygon_cells(
            corridor.source_gate, corridor.coordinate_space
        ) & roi_cells
        self._target_cells = self._polygon_cells(
            corridor.target_gate, corridor.coordinate_space
        ) & roi_cells
        if not roi_cells or not self._source_cells or not self._target_cells:
            raise ValueError("Local corridor ROI and gates must cover grid cell centers")
        self.valid_cells = (roi_cells if self._explicit_valid_cells is None else
                            roi_cells & self._explicit_valid_cells)

    def _polygon_cells(
        self, points: Iterable[tuple[float, float]], coordinate_space: str
    ) -> set[Cell]:
        polygon_points = []
        for x, y in points:
            if coordinate_space == "image_normalized":
                x, y = x * self.transformer.width, y * self.transformer.height
            polygon_points.append(self.transformer.transform(x, y))
        polygon = np.asarray(polygon_points, dtype=np.float32)
        result = set()
        for row in range(self.config.rows):
            for col in range(self.config.columns):
                center = ((col + .5) * self.transformer.width / self.config.columns,
                          (row + .5) * self.transformer.height / self.config.rows)
                if cv2.pointPolygonTest(polygon, center, False) >= 0:
                    result.add((row, col))
        return result

    def cell(self, x: float, y: float) -> Cell | None:
        tx, ty = self.transformer.transform(x, y)
        cell = self.transformer.grid_cell(
            tx, ty, grid_width=self.config.columns, grid_height=self.config.rows
        )
        return cell if self.valid_cells is None or cell in self.valid_cells else None

    def direction_bin(self, dx: float, dy: float) -> int:
        angle = math.atan2(dy, dx) % (2 * math.pi)
        return int((angle + math.pi / self.config.direction_bins) /
                   (2 * math.pi / self.config.direction_bins)) % self.config.direction_bins

    def update(self, points: Iterable[GridTrackPoint], event_time_s: float) -> CommonPathSnapshot:
        self._advance(event_time_s)
        for point in points:
            if point.camera_id != self.camera_id or point.stream_epoch != self.stream_epoch:
                raise ValueError("TrackPoint belongs to another camera/epoch")
            if point.event_time_s != event_time_s:
                raise ValueError("TrackPoint time differs from frame event time")
            if not point.confirmed or not point.observed:
                continue
            cell = self.cell(point.x, point.y)
            if cell is None:
                continue
            base = point.key[:3]
            key = (*base, max(point.segment_id, self._segments.get(base, point.segment_id)))
            track = self._tracks.get(key)
            if track and (point.frame_id <= track.last_frame or event_time_s <= track.last_s):
                raise ValueError("Track frame/time is out of order")
            if track and event_time_s - track.last_s > self.config.max_observation_gap_seconds:
                # Do not bridge lost observations. The next segment needs a new key.
                key = (*base, key[3] + 1)
                track = None
            self._segments[base] = key[3]
            tx, ty = self.transformer.transform(point.x, point.y)
            if track is None:
                if len(self._tracks) >= self.config.max_tracks:
                    oldest = min(self._tracks, key=lambda k: self._tracks[k].last_s)
                    del self._tracks[oldest]
                    self._overflow += 1
                track = _Track(deque(maxlen=self.config.max_points_per_track),
                               event_time_s, event_time_s, point.frame_id, cell, [cell])
                self._tracks[key] = track
            else:
                previous_time, px, py = track.points[-1]
                dt = event_time_s - previous_time
                step = math.hypot((tx - px) / (self.transformer.width / self.config.columns),
                                  (ty - py) / (self.transformer.height / self.config.rows))
                if step > self.config.max_step_cells:
                    key = (*base, key[3] + 1)
                    self._segments[base] = key[3]
                    if len(self._tracks) >= self.config.max_tracks:
                        oldest = min(self._tracks, key=lambda k: self._tracks[k].last_s)
                        del self._tracks[oldest]
                        self._overflow += 1
                    track = _Track(
                        deque(maxlen=self.config.max_points_per_track), event_time_s,
                        event_time_s, point.frame_id, cell, [cell]
                    )
                    self._tracks[key] = track
                    self._rejected_jumps += 1
                else:
                    alpha = 1 - math.pow(0.5, dt / max(0.01, self.config.direction_sample_seconds))
                    tx, ty = px + alpha * (tx - px), py + alpha * (ty - py)
                    filtered_cell = self.transformer.grid_cell(
                        tx, ty, grid_width=self.config.columns, grid_height=self.config.rows
                    )
                    if filtered_cell is not None:
                        self._movement(key, track, filtered_cell, tx, ty, event_time_s)
                track.last_s, track.last_frame = event_time_s, point.frame_id
            track.points.append((event_time_s, tx, ty))
            self._occupancy[cell][key] = event_time_s
        self._prune(event_time_s)
        self._maybe_extract(event_time_s)
        return self.snapshot()

    def tick(self, event_time_s: float) -> CommonPathSnapshot:
        self._advance(event_time_s)
        self._prune(event_time_s)
        self._maybe_extract(event_time_s)
        return self.snapshot()

    def _advance(self, timestamp: float) -> None:
        if not math.isfinite(timestamp) or (self._clock is not None and timestamp < self._clock):
            raise ValueError("Event time must be finite and monotonic; reset on seek/reconnect")
        self._clock = timestamp
        if self._start is None:
            self._start = timestamp

    def _movement(self, key: TrackKey, track: _Track, cell: Cell,
                  x: float, y: float, now: float) -> None:
        history = track.points
        sample = next((p for p in history if p[0] >= now - self.config.direction_sample_seconds),
                      history[-1])
        dx, dy = x - sample[1], y - sample[2]
        cell_width = self.transformer.width / self.config.columns
        cell_height = self.transformer.height / self.config.rows
        moving = (now - track.first_s >= self.config.min_track_age_seconds and
                  math.hypot(dx / cell_width, dy / cell_height) >=
                  self.config.min_displacement_cell_fraction)
        if moving:
            self._bins[(track.cell, self.direction_bin(dx, dy))][key] = now
        if cell == track.cell:
            track.pending, track.pending_count = None, 0
            return
        # A stable observation inside the new cell, repeated twice, prevents boundary jitter.
        cx = (cell[1] + 0.5) * cell_width
        cy = (cell[0] + 0.5) * cell_height
        if abs(x - cx) > 0.43 * cell_width or abs(y - cy) > 0.43 * cell_height:
            return
        track.pending_count = track.pending_count + 1 if track.pending == cell else 1
        track.pending = cell
        if track.pending_count < 2 or not moving:
            return
        origin = track.cell
        assert origin is not None
        steps = max(abs(cell[0] - origin[0]), abs(cell[1] - origin[1]))
        if steps > 2:
            return
        sequence = [origin]
        for index in range(1, steps + 1):
            fraction = index / steps
            next_cell = (round(origin[0] + fraction * (cell[0] - origin[0])),
                         round(origin[1] + fraction * (cell[1] - origin[1])))
            if self.valid_cells is not None and next_cell not in self.valid_cells:
                return
            if next_cell != sequence[-1]:
                sequence.append(next_cell)
        for src, dst in zip(sequence, sequence[1:]):
            self._edges[(src, dst)][key] = now
            if track.cells[-1] != dst:
                track.cells.append(dst)
        track.cell, track.pending, track.pending_count = cell, None, 0
        if len(track.cells) > self.config.max_path_cells * 2:
            del track.cells[:len(track.cells) - self.config.max_path_cells * 2]
            self._overflow += 1

    def _prune(self, now: float) -> None:
        cutoff = now - self.config.long_window_seconds
        for collection in (self._bins, self._edges, self._occupancy):
            for node, entries in list(collection.items()):
                for key, time_s in list(entries.items()):
                    if time_s < cutoff:
                        del entries[key]
                if not entries:
                    del collection[node]
        for key, track in list(self._tracks.items()):
            if track.last_s < cutoff:
                del self._tracks[key]
        for base in list(self._segments):
            key = (*base, self._segments[base])
            if key not in self._tracks:
                del self._segments[base]
        evidence_count = sum(map(len, self._bins.values())) + sum(map(len, self._edges.values()))
        if evidence_count > self.config.max_evidence_keys:
            # Discard oldest whole edges first; do not report capped data as complete.
            for edge in sorted(self._edges, key=lambda e: max(self._edges[e].values())):
                evidence_count -= len(self._edges.pop(edge))
                self._overflow += 1
                if evidence_count <= self.config.max_evidence_keys:
                    break

    def histogram(self, now: float | None = None) -> dict[Cell, dict[int, tuple[int, int]]]:
        now = self._clock if now is None else now
        if now is None:
            return {}
        result: dict[Cell, dict[int, tuple[int, int]]] = defaultdict(dict)
        for (cell, direction), evidence in self._bins.items():
            result[cell][direction] = (
                sum(t >= now - self.config.short_window_seconds for t in evidence.values()),
                len(evidence),
            )
        return dict(result)

    def flow_snapshot(self) -> DirectedFlowSnapshot:
        now = self._clock or 0.0
        age = max(1.0, now - (self._start if self._start is not None else now))
        short_age = min(self.config.short_window_seconds, age)
        long_age = min(self.config.long_window_seconds, age)
        edges = []
        for (src, dst), support in self._edges.items():
            short = sum(t >= now - self.config.short_window_seconds for t in support.values())
            long = len(support)
            score = 0.65 * short / short_age + 0.35 * long / long_age
            edges.append(DirectedFlowEdge(src, dst, short, long, score))
        return DirectedFlowSnapshot(max(0., now - self.config.long_window_seconds), now,
                                    self.config.columns, self.config.rows,
                                    tuple(sorted(edges, key=lambda e: -e.score)))

    def _endpoints(self) -> tuple[set[Cell], set[Cell]]:
        if self.config.route_scope == "local_corridor":
            return set(self._source_cells), set(self._target_cells)
        entrance, exits = set(), set()
        for zone in self.zones:
            if zone.zone_type not in ("entry", "exit"):
                continue
            polygon = np.asarray([self.transformer.transform(*p) for p in zone.points],
                                 dtype=np.float32)
            for row in range(self.config.rows):
                for col in range(self.config.columns):
                    center = ((col + .5) * self.transformer.width / self.config.columns,
                              (row + .5) * self.transformer.height / self.config.rows)
                    if cv2.pointPolygonTest(polygon, center, False) >= 0:
                        (entrance if zone.zone_type == "entry" else exits).add((row, col))
        if entrance and exits:
            return entrance, exits
        # Without configured zones, only tracks observed starting/ending at the ROI boundary.
        def boundary(c: Cell) -> bool:
            return c[0] in (0, self.config.rows - 1) or c[1] in (0, self.config.columns - 1)
        return ({t.cells[0] for t in self._tracks.values() if t.cells and boundary(t.cells[0])},
                {t.cells[-1] for t in self._tracks.values() if t.cells and boundary(t.cells[-1])})

    def _endpoint_pairs(self) -> list[tuple[str, str, str, set[Cell], set[Cell]]]:
        origins, destinations = self._endpoints()
        if self.config.route_scope != "local_corridor":
            return [("forward", "scene_entry", "scene_exit", origins, destinations)]
        corridor = self.config.local_corridor
        assert corridor is not None
        pairs = [("S_TO_T", "source_gate", "target_gate", origins, destinations)]
        if corridor.allow_reverse:
            pairs.append(("T_TO_S", "target_gate", "source_gate", destinations, origins))
        return pairs

    @staticmethod
    def _reachable(adjacency: dict[Cell, list[tuple[Cell, float]]], start: Cell,
                   destinations: set[Cell]) -> bool:
        pending, visited = [start], {start}
        while pending:
            node = pending.pop()
            if node in destinations:
                return True
            for child, _ in adjacency.get(node, ()):
                if child not in visited:
                    visited.add(child)
                    pending.append(child)
        return False

    def _candidate_id(self, cells: tuple[Cell, ...]) -> str:
        for known_cells, candidate_id in self._route_identities:
            if self._same_route(cells, known_cells):
                return candidate_id
        signature = zlib.crc32(
            repr((self.config.route_scope, cells[0], cells[-1], cells)).encode("ascii")
        )
        candidate_id = f"{self.config.route_scope}-{signature:08x}"
        self._route_identities.append((cells, candidate_id))
        if len(self._route_identities) > 512:
            del self._route_identities[:len(self._route_identities) - 512]
        return candidate_id

    def _candidates(self, now: float) -> list[_Route]:
        flows = list(self.flow_snapshot().edges)
        adjacency: dict[Cell, list[tuple[Cell, float]]] = defaultdict(list)
        edge_support_histogram: Counter[int] = Counter()
        for item in flows:
            edge_support_histogram[item.unique_tracks_long] += 1
            if item.unique_tracks_long >= self.config.min_edge_unique_tracks:
                adjacency[item.from_cell].append((item.to_cell, item.score))
        endpoint_pairs = self._endpoint_pairs()
        seed_count = sum(sum(cell in adjacency for cell in origins)
                         for _, _, _, origins, _ in endpoint_pairs)
        possible_pairs = sum(len(origins) * len(destinations)
                             for _, _, _, origins, destinations in endpoint_pairs)
        connected_pairs = sum(
            self._reachable(adjacency, origin, destinations)
            for _, _, _, origins, destinations in endpoint_pairs
            for origin in origins
        )
        connectivity_by_threshold = {}
        for threshold in range(1, self.config.min_edge_unique_tracks + 1):
            threshold_adjacency: dict[Cell, list[tuple[Cell, float]]] = defaultdict(list)
            for item in flows:
                if item.unique_tracks_long >= threshold:
                    threshold_adjacency[item.from_cell].append((item.to_cell, item.score))
            connectivity_by_threshold[str(threshold)] = sum(
                self._reachable(threshold_adjacency, origin, destinations)
                for _, _, _, origins, destinations in endpoint_pairs
                for origin in origins
            )
        results: list[_Route] = []
        loop_pruned = max_length_pruned = 0
        search_budget_reached = False
        expansion_budget = self.config.beam_width * self.config.max_path_cells
        for direction, source_gate, target_gate, origins, destinations in endpoint_pairs:
            for origin in sorted(origins):
                if origin not in adjacency:
                    continue
                queue: deque[tuple[tuple[Cell, ...], float]] = deque([((origin,), 0.)])
                visited = {origin}
                expansions = 0
                found_paths: list[tuple[tuple[Cell, ...], float]] = []
                while queue and expansions < expansion_budget:
                    cells, rank = queue.popleft()
                    expansions += 1
                    if len(cells) >= self.config.max_path_cells:
                        max_length_pruned += len(adjacency.get(cells[-1], ()))
                        continue
                    for dst, score in sorted(adjacency.get(cells[-1], ()),
                                             key=lambda item: -item[1]):
                        if dst in visited:
                            loop_pruned += 1
                            continue
                        new_cells = (*cells, dst)
                        new_rank = rank + math.log1p(score)
                        if dst in destinations and len(new_cells) >= self.config.min_path_cells:
                            found_paths.append((new_cells, new_rank))
                        visited.add(dst)
                        queue.append((new_cells, new_rank))
                if queue and expansions >= expansion_budget:
                    search_budget_reached = True
                if not found_paths:
                    continue
                for new_cells, new_rank in found_paths:
                    details = self._validate_detailed(
                        new_cells, now, origins, destinations,
                        scope=self.config.route_scope,
                    )
                    candidate_id = self._candidate_id(new_cells)
                    self._evaluation_count += 1
                    evaluation_id = f"{self.run_id}:{self.variant}:{self._evaluation_count}"
                    results.append(_Route(
                        new_cells, details["support_tracks"], details["complete_tracks"],
                        new_rank / len(new_cells) * (1 + details["complete_tracks"]),
                        details["evidence_until_s"], self.config.route_scope, direction,
                        source_gate, target_gate, candidate_id, evaluation_id, details,
                    ))
        if self.config.display_policy == "dominant_direction":
            ranked_results = sorted(
                results,
                key=lambda route: (
                    -int(route.diagnostics.get("short_support_tracks", 0)),
                    -route.support,
                    -route.score,
                ),
            )
        else:
            ranked_results = sorted(results, key=lambda r: (-r.complete, -r.support, -r.score))
        top_k_pruned = max(0, len(ranked_results) - self.config.max_candidates)
        selected = ranked_results[:self.config.max_candidates]
        evidence_times = [time_s for evidence in self._edges.values()
                          for time_s in evidence.values()]
        observed = max(0., now - (self._start if self._start is not None else now))
        if not any(origins and destinations for _, _, _, origins, destinations in endpoint_pairs):
            reason = "NO_SEEDS"
        elif flows and not adjacency:
            reason = "INSUFFICIENT_EDGE_SUPPORT"
        elif seed_count == 0:
            reason = "NO_SEEDS"
        elif connected_pairs == 0:
            reason = "NO_CONNECTED_PATH"
        elif not selected and search_budget_reached:
            reason = "SEARCH_BUDGET_REACHED"
        elif not selected:
            reason = "NO_CONNECTED_PATH"
        else:
            reason = None
        graph = dict(
            run_id=self.run_id, variant=self.variant,
            evaluation_batch=f"{self.run_id}:{self.variant}:batch-{len(self._graph_diagnostics)+1}",
            event_time_s=now, route_scope=self.config.route_scope,
            evidence_cells=len(self._occupancy), directed_edges_before_filter=len(flows),
            directed_edges_after_filter=sum(len(items) for items in adjacency.values()),
            edge_support_histogram={str(k): v for k, v in sorted(edge_support_histogram.items())},
            edge_support_threshold=self.config.min_edge_unique_tracks,
            start_seeds=seed_count,
            end_seeds=sum(len(destinations) for _, _, _, _, destinations in endpoint_pairs),
            possible_pairs=possible_pairs, connected_pairs=connected_pairs,
            connected_origin_seeds_by_edge_threshold=connectivity_by_threshold,
            candidate_routes_found=len(results), generated_evaluations=len(selected),
            pruned=dict(insufficient_edge_support=len(flows)-sum(len(v) for v in adjacency.values()),
                        roi=0, direction=0, loop=loop_pruned, max_length=max_length_pruned,
                        beam=0, top_k=top_k_pruned),
            search_budget_reached=search_budget_reached, primary_reason=reason,
            observed_window_seconds=observed,
            warmup_state=("long_ready" if observed >= self.config.long_window_seconds else
                          "short_ready" if observed >= self.config.short_window_seconds else
                          "warming_short"),
            earliest_evidence_s=min(evidence_times) if evidence_times else None,
            latest_evidence_s=max(evidence_times) if evidence_times else None,
        )
        if self.config.diagnostics_enabled:
            self._graph_diagnostics.append(graph)
        return selected

    def _validate(self, cells: tuple[Cell, ...], now: float) -> tuple[int, int, float]:
        details = self._validate_detailed(
            cells, now, {cells[0]}, {cells[-1]}, scope="global_od"
        )
        return (details["support_tracks"], details["complete_tracks"],
                details["evidence_until_s"])

    def _ordered_match_count(self, seen_cells: Iterable[Cell], route: tuple[Cell, ...]) -> int:
        cursor = 0
        for seen in seen_cells:
            if cursor >= len(route):
                break
            target = route[cursor]
            if max(abs(seen[0] - target[0]), abs(seen[1] - target[1])) <= self.config.cell_tolerance:
                cursor += 1
        return cursor

    @staticmethod
    def _gate_order(cells: Iterable[Cell], origins: set[Cell], destinations: set[Cell]) -> bool:
        origin_seen = False
        for cell in cells:
            if cell in origins:
                origin_seen = True
            elif origin_seen and cell in destinations:
                return True
        return False

    def _validate_detailed(
        self, cells: tuple[Cell, ...], now: float, origins: set[Cell],
        destinations: set[Cell], *, scope: str,
    ) -> dict[str, Any]:
        support = short_support = complete = order_pass_count = 0
        evidence = 0.
        evidence_keys: list[str] = []
        coverages: list[float] = []
        rejection_counts: Counter[str] = Counter()
        for key, track in self._tracks.items():
            if track.last_s < now - self.config.long_window_seconds:
                rejection_counts["STALE_EVIDENCE"] += 1
                continue
            if not track.cells:
                rejection_counts["INSUFFICIENT_COVERAGE"] += 1
                continue
            matched = self._ordered_match_count(track.cells, cells)
            reverse_matched = self._ordered_match_count(track.cells, tuple(reversed(cells)))
            coverage = matched / len(cells)
            reverse_coverage = reverse_matched / len(cells)
            coverages.append(coverage)
            unordered = sum(any(max(abs(seen[0]-target[0]), abs(seen[1]-target[1])) <=
                                self.config.cell_tolerance for seen in track.cells)
                            for target in cells) / len(cells)
            if coverage >= self.config.min_ordered_coverage:
                support += 1
                if track.last_s >= now - self.config.short_window_seconds:
                    short_support += 1
                order_pass_count += 1
                evidence = max(evidence, track.last_s)
                is_complete = (self._gate_order(track.cells, origins, destinations)
                               if scope == "local_corridor" else
                               matched == len(cells) and track.cells[0] == cells[0]
                               and track.cells[-1] == cells[-1])
                if is_complete:
                    complete += 1
                    if len(evidence_keys) < self.config.max_diagnostic_evidence_samples:
                        evidence_keys.append(":".join(map(str, key)))
                else:
                    rejection_counts["INSUFFICIENT_COMPLETE_TRACKS"] += 1
            elif reverse_coverage >= self.config.min_ordered_coverage:
                rejection_counts["DIRECTION_MISMATCH"] += 1
            elif unordered >= self.config.min_ordered_coverage:
                rejection_counts["ORDER_MISMATCH"] += 1
            else:
                rejection_counts["INSUFFICIENT_COVERAGE"] += 1
        for reason, count in rejection_counts.items():
            self._tracklet_rejections[reason] += count
        if coverages:
            distribution = dict(
                minimum=round(min(coverages), 4),
                median=round(float(np.percentile(coverages, 50)), 4),
                p95=round(float(np.percentile(coverages, 95)), 4),
                maximum=round(max(coverages), 4),
                samples=len(coverages),
            )
        else:
            distribution = dict(minimum=None, median=None, p95=None, maximum=None, samples=0)
        return dict(
            support_tracks=support, short_support_tracks=short_support,
            complete_tracks=complete,
            evidence_until_s=evidence, coverage_distribution=distribution,
            order_pass_count=order_pass_count, evidence_track_keys=evidence_keys,
            tracklet_rejection_counts=dict(rejection_counts),
        )

    def _same_route(self, a: tuple[Cell, ...], b: tuple[Cell, ...]) -> bool:
        if a[0] != b[0] or a[-1] != b[-1]:
            return False
        edges_a, edges_b = set(zip(a, a[1:])), set(zip(b, b[1:]))
        return len(edges_a & edges_b) / max(len(edges_a), len(edges_b)) >= 0.65

    def _path(self, route: _Route, path_id: str, state: str, revision: int) -> CommonPath:
        polyline = tuple(self.transformer.inverse_transform(
            (cell[1] + .5) * self.transformer.width / self.config.columns,
            (cell[0] + .5) * self.transformer.height / self.config.rows,
        ) for cell in route.cells)
        short_support = int(route.diagnostics.get("short_support_tracks", route.support))
        return CommonPath(path_id, route.source_gate, route.target_gate, state,
                          short_support, route.support, route.score, 1.0,
                          route.direction, polyline, self._clock or 0., revision,
                          "ground_plane" if self.transformer.mode == "ground" else "image_pixels",
                          route.support, route.complete, route.evidence_s, state == "cooling")

    def _event(self, kind: str, reason: str, path_id: str, evidence: float) -> None:
        self._events.append(dict(event=kind, reason=reason, path_id=path_id,
                                 event_time_s=self._clock, evidence_until_s=evidence))
        if len(self._events) > 2048:
            del self._events[:len(self._events) - 2048]

    def _record_candidate_evaluations(
        self, candidates: Iterable[_Route], decisions: dict[str, tuple[str, str]], now: float,
    ) -> None:
        edge_support = {edge: len(evidence) for edge, evidence in self._edges.items()}
        for route in candidates:
            secondary = []
            short_support = int(route.diagnostics.get("short_support_tracks", 0))
            required_support = (
                self.config.dominant_min_support_tracks
                if self.config.display_policy == "dominant_direction"
                else self.config.min_support_tracks
            )
            evaluated_support = (
                short_support
                if self.config.display_policy == "dominant_direction"
                else route.support
            )
            if evaluated_support < required_support:
                secondary.append("INSUFFICIENT_UNIQUE_SUPPORT")
            if (self.config.display_policy == "validated_route"
                    and route.complete < self.config.min_complete_tracks):
                secondary.append("INSUFFICIENT_COMPLETE_TRACKS")
            decision, primary = decisions.get(
                route.evaluation_id, ("eligible_not_selected", "NOT_TOP_RANKED")
            )
            self._candidate_decisions[decision] += 1
            confirmation_elapsed = (
                max(0., now - self._pending_since)
                if self._pending_since is not None and self._pending is not None
                and self._same_route(route.cells, self._pending.cells) else 0.
            )
            record = dict(
                run_id=self.run_id, variant=self.variant,
                evaluation_id=route.evaluation_id, event_time_s=now,
                candidate_id=route.candidate_id, route_scope=route.scope,
                source_gate=route.source_gate, target_gate=route.target_gate,
                cell_sequence=[list(cell) for cell in route.cells],
                direction=route.direction, path_length=len(route.cells),
                min_edge_support=min(
                    (edge_support.get(edge, 0) for edge in zip(route.cells, route.cells[1:])),
                    default=0,
                ),
                support_tracks=route.support, short_support_tracks=short_support,
                complete_tracks=route.complete,
                coverage_distribution=route.diagnostics["coverage_distribution"],
                order_pass_count=route.diagnostics["order_pass_count"],
                score=round(route.score, 8),
                active_score=(self._active.score if self._active is not None else None),
                confirmation_elapsed_s=round(confirmation_elapsed, 4),
                decision=decision, primary_reason=primary,
                evaluated_secondary_reasons=secondary,
                checks_not_evaluated=[],
                thresholds=dict(
                    display_policy=self.config.display_policy,
                    dominant_min_support_tracks=self.config.dominant_min_support_tracks,
                    min_edge_unique_tracks=self.config.min_edge_unique_tracks,
                    min_support_tracks=self.config.min_support_tracks,
                    min_complete_tracks=self.config.min_complete_tracks,
                    min_ordered_coverage=self.config.min_ordered_coverage,
                    confirmation_seconds=self.config.confirmation_seconds,
                    switch_relative_margin=self.config.switch_relative_margin,
                    switch_absolute_margin=self.config.switch_absolute_margin,
                ),
                evidence_track_keys=route.diagnostics["evidence_track_keys"],
                tracklet_rejection_counts=route.diagnostics["tracklet_rejection_counts"],
            )
            if not self.config.diagnostics_enabled:
                continue
            if len(self._candidate_evaluations) < self.config.max_diagnostic_evaluations:
                self._candidate_evaluations.append(record)
            else:
                self._diagnostic_dropped += 1

    def _select_dominant_direction(self, now: float) -> None:
        cutoff = now - self.config.short_window_seconds
        keys_by_direction: dict[int, set[tuple[str, str, int]]] = defaultdict(set)
        keys_by_cell: dict[tuple[Cell, int], set[tuple[str, str, int]]] = {}
        for (cell, direction), evidence in self._bins.items():
            recent = {key[:3] for key, time_s in evidence.items() if time_s >= cutoff}
            if recent:
                keys_by_direction[direction].update(recent)
                keys_by_cell[(cell, direction)] = recent

        eligible_directions = [
            direction for direction, keys in keys_by_direction.items()
            if len(keys) >= self.config.dominant_min_support_tracks
        ]
        direction = max(
            eligible_directions,
            key=lambda item: (len(keys_by_direction[item]), -item),
            default=None,
        )
        previous = self._active
        self._pending = None
        self._pending_since = self._pending_last_evidence = None
        self._cooling_since = None
        self._active = None

        if direction is not None:
            cells = {cell for cell, item_direction in keys_by_cell if item_direction == direction}
            components: list[set[Cell]] = []
            remaining = set(cells)
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
                components.append(component)

            def component_rank(component: set[Cell]) -> tuple[int, int, int]:
                keys = set().union(*(
                    keys_by_cell[(cell, direction)] for cell in component
                ))
                weight = sum(len(keys_by_cell[(cell, direction)]) for cell in component)
                return len(keys), weight, len(component)

            component = max(components, key=component_rank, default=set())
            component_keys = set().union(*(
                keys_by_cell[(cell, direction)] for cell in component
            )) if component else set()
            if len(component_keys) >= self.config.dominant_min_support_tracks:
                cell_width = self.transformer.width / self.config.columns
                cell_height = self.transformer.height / self.config.rows
                angle = direction * 2 * math.pi / self.config.direction_bins
                ux, uy = math.cos(angle), math.sin(angle)
                vx, vy = -uy, ux
                samples = []
                for row, column in component:
                    weight = len(keys_by_cell[((row, column), direction)])
                    x = (column + .5) * cell_width
                    y = (row + .5) * cell_height
                    samples.append((x * ux + y * uy, x * vx + y * vy, weight))
                weight_sum = sum(sample[2] for sample in samples)
                perpendicular = sum(q * weight for _, q, weight in samples) / weight_sum
                projections = [projection for projection, _, _ in samples]
                low, high = min(projections), max(projections)
                minimum_length = 2.5 * max(cell_width, cell_height)
                if high - low < minimum_length:
                    middle = (low + high) / 2
                    low, high = middle - minimum_length / 2, middle + minimum_length / 2
                endpoints = []
                for projection in (low, high):
                    x = projection * ux + perpendicular * vx
                    y = projection * uy + perpendicular * vy
                    endpoints.append((
                        min(self.transformer.width - 1., max(0., x)),
                        min(self.transformer.height - 1., max(0., y)),
                    ))
                direction_names = (
                    "east", "southeast", "south", "southwest",
                    "west", "northwest", "north", "northeast",
                )
                direction_name = direction_names[
                    round(direction * 8 / self.config.direction_bins) % 8
                ]
                path_id = f"{self.stream_epoch}-dominant-direction-{direction}"
                revision = (
                    previous.revision + 1
                    if previous is not None and previous.path_id == path_id else 1
                )
                long_keys = set().union(*(
                    {key[:3] for key in self._bins[(cell, direction)]}
                    for cell in component
                ))
                evidence_until = max(
                    time_s
                    for cell in component
                    for key, time_s in self._bins[(cell, direction)].items()
                    if key[:3] in keys_by_cell[(cell, direction)] and time_s >= cutoff
                )
                direction_total = sum(len(keys) for keys in keys_by_direction.values())
                confidence = len(component_keys) / max(1, direction_total)
                self._active = CommonPath(
                    path_id=path_id,
                    origin_zone="dominant_direction",
                    destination_zone=direction_name,
                    state="active",
                    unique_tracks_short=len(component_keys),
                    unique_tracks_long=len(long_keys),
                    score=confidence,
                    confidence=confidence,
                    direction=direction_name,
                    polyline=tuple(endpoints),
                    updated_at=now,
                    revision=revision,
                    coordinate_space="image_pixels",
                    support_tracks=len(component_keys),
                    validated_complete_tracks=0,
                    evidence_until_s=evidence_until,
                    stale=False,
                )
                if previous is None:
                    self._event("activate", "dominant_direction_support", path_id, evidence_until)
                elif previous.path_id != path_id:
                    self._event("switch", "dominant_direction_changed", path_id, evidence_until)

        if self._active is None and previous is not None:
            self._event(
                "retired", "no_recent_dominant_direction",
                previous.path_id, previous.evidence_until_s,
            )
        paths = (self._active,) if self._active is not None else ()
        evidence_until = max((path.evidence_until_s for path in paths), default=0.0)
        self._version += 1
        self._snapshot = CommonPathSnapshot(now, paths, self._version, evidence_until)

    def _maybe_extract(self, now: float) -> None:
        if self._last_extract is not None and now - self._last_extract < self.config.update_interval_seconds:
            return
        self._last_extract = now
        self._retired_once = None
        if self.config.display_policy == "dominant_direction":
            self._latest_routes = ()
            self._select_dominant_direction(now)
            return
        candidates = self._candidates(now)
        self._latest_routes = tuple(candidates)
        decisions: dict[str, tuple[str, str]] = {}
        for route in candidates:
            if route.support < self.config.min_support_tracks:
                decisions[route.evaluation_id] = ("rejected", "INSUFFICIENT_UNIQUE_SUPPORT")
            elif route.complete < self.config.min_complete_tracks:
                decisions[route.evaluation_id] = ("rejected", "INSUFFICIENT_COMPLETE_TRACKS")
        eligible = [r for r in candidates if r.support >= self.config.min_support_tracks and
                    r.complete >= self.config.min_complete_tracks and r.evidence_s > 0]
        best = eligible[0] if eligible else None
        for route in eligible[1:]:
            decisions[route.evaluation_id] = ("eligible_not_selected", "NOT_TOP_RANKED")
        active = self._active
        if active is not None:
            matching = next((r for r in eligible if self._same_route(r.cells, self._route_cells(active))), None)
            if matching is not None:
                updated = self._path(matching, active.path_id, "active", active.revision + 1)
                self._active = updated
                self._cooling_since = None
                active = updated
                decisions[matching.evaluation_id] = ("selected", "KEPT_ACTIVE")
            elif self._cooling_since is None:
                self._cooling_since = now
                self._active = replace(active, state="cooling", stale=True, updated_at=now)
                self._event("cooling", "insufficient_fresh_complete_support", active.path_id,
                            active.evidence_until_s)
            if self._cooling_since is not None and now - self._cooling_since >= self.config.cooling_seconds:
                self._event("retired", "cooling_timeout", active.path_id, active.evidence_until_s)
                self._retired_once = replace(
                    active, state="retired", stale=True, updated_at=now
                )
                self._active = None
        if best is not None and self._active is not None and self._active.state == "active":
            if self._same_route(best.cells, self._route_cells(self._active)):
                best = None
            elif best.score <= self._active.score * (1 + self.config.switch_relative_margin) + self.config.switch_absolute_margin:
                decisions[best.evaluation_id] = (
                    "eligible_not_selected", "HYSTERESIS_MARGIN_NOT_MET"
                )
                best = None
        if best is None:
            self._pending = None
            self._pending_since = self._pending_last_evidence = None
        elif self._pending is None or not self._same_route(best.cells, self._pending.cells):
            self._pending, self._pending_since = best, now
            self._pending_last_evidence = best.evidence_s
            self._event("candidate", "complete_evidence", "pending", best.evidence_s)
            decisions[best.evaluation_id] = ("pending", "HYSTERESIS_CONFIRMING")
        elif best.evidence_s > (self._pending_last_evidence or -1):
            self._pending = best
            self._pending_last_evidence = best.evidence_s
            if now - (self._pending_since or now) >= self.config.confirmation_seconds:
                old = self._active
                path_id = f"{self.stream_epoch}-{self._version + 1}"
                self._active = self._path(best, path_id, "active", 1)
                self._event("switch" if old else "activate", "confirmed_complete_tracks",
                            path_id, best.evidence_s)
                self._cooling_since = None
                self._pending = None
                self._pending_since = self._pending_last_evidence = None
                decisions[best.evaluation_id] = ("selected", "ACTIVATED")
            else:
                decisions[best.evaluation_id] = ("pending", "HYSTERESIS_CONFIRMING")
        elif best is not None:
            decisions[best.evaluation_id] = ("pending", "HYSTERESIS_CONFIRMING")
        paths: tuple[CommonPath, ...] = ()
        if self._active is not None:
            paths += (self._active,)
        if self._pending is not None:
            paths += (self._path(
                self._pending, self._pending.candidate_id, "candidate", 0
            ),)
        if self._retired_once is not None:
            paths += (self._retired_once,)
        evidence_until = max((path.evidence_until_s for path in paths), default=0.0)
        self._version += 1
        self._snapshot = CommonPathSnapshot(now, paths, self._version, evidence_until)
        self._record_candidate_evaluations(candidates, decisions, now)

    def _route_cells(self, path: CommonPath) -> tuple[Cell, ...]:
        return tuple(self.cell(*xy) for xy in path.polyline)  # type: ignore[return-value]

    def snapshot(self) -> CommonPathSnapshot:
        return self._snapshot
