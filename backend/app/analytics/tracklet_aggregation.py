"""Tracklet-based Common Path aggregation without a spatial grid.

The engine keeps short observed tracklets, rejects weak or inconsistent motion,
links compatible tracklets with a directed graph, and merges each component into
one smoothed representative polyline.  A grid is deliberately not used for the
Common Path decision; ``GridTrackPoint`` is only the existing point transport
schema shared by the analytics pipeline.
"""

from __future__ import annotations

import math
import statistics
import time
from collections import Counter, deque
from dataclasses import dataclass, replace
from itertools import chain
from typing import Iterable

import numpy as np

from backend.app.analytics.directional_grid import GridTrackPoint
from backend.app.analytics.spatial import SpatialTransformer
from backend.app.core.config import TrackletAggregationConfig
from backend.app.schemas import CommonPath, CommonPathSnapshot

TrackKey = tuple[str, str, int]
COLORS = ("#ffca42", "#37d5c4", "#ff718d", "#80aaff", "#d3a1f5", "#9ce56a", "#ff9e68")


@dataclass
class _Segment:
    key: TrackKey
    number: int
    points: deque[tuple[float, float, float]]
    last_frame: int


@dataclass(frozen=True)
class _Tracklet:
    key: TrackKey
    number: int
    start_time: float
    end_time: float
    points: np.ndarray
    direction: np.ndarray
    arc_length: float
    displacement: float
    quality: float
    direction_consistency: float


@dataclass
class _Identity:
    path: CommonPath
    first_seen: float
    last_seen: float
    selected_at: float | None = None
    missing_since: float | None = None


class TrackletAggregationEngine:
    """Bounded evidence store for direction-compatible short tracklets."""

    def __init__(self, config: TrackletAggregationConfig, transformer: SpatialTransformer,
                 *, camera_id: str = "cam01", stream_epoch: str = "initial", max_paths: int = 3):
        self.config = config
        self.width, self.height = transformer.width, transformer.height
        self.camera_id, self.stream_epoch = camera_id, stream_epoch
        self.max_paths = max_paths
        self._segments: dict[TrackKey, _Segment] = {}
        self._closed: deque[_Segment] = deque()
        self._number: Counter[TrackKey] = Counter()
        self._identities: dict[str, _Identity] = {}
        self._sequence = 0
        self._last_time = -math.inf
        self._last_compute = -math.inf
        self._snapshot = CommonPathSnapshot(0, ())
        self.events: list[dict] = []
        self.rejections: Counter[str] = Counter()
        self.compute_ms: deque[float] = deque(maxlen=256)
        self.candidate_count = 0

    def reset(self) -> None:
        self._segments.clear()
        self._closed.clear()
        self._number.clear()
        self._identities.clear()
        self._snapshot = CommonPathSnapshot(0, ())
        self._last_compute = -math.inf
        self._last_time = -math.inf
        self.events.clear()
        self.rejections.clear()

    def set_max_paths(self, value: int) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 5:
            raise ValueError("max_paths must be an integer in [1, 5]")
        self.max_paths = value
        self._publish(self._last_time if math.isfinite(self._last_time) else 0)
        return value

    def snapshot(self) -> CommonPathSnapshot:
        return self._snapshot

    def update(self, points: Iterable[GridTrackPoint], timestamp: float) -> CommonPathSnapshot:
        points = tuple(points)
        if timestamp < self._last_time or any(
            p.camera_id != self.camera_id or p.stream_epoch != self.stream_epoch for p in points
        ):
            self.reset()
            raise ValueError("camera/epoch mismatch or backwards source timestamp")
        self._last_time = timestamp
        cfg = self.config
        for point in points:
            if not point.observed or not point.confirmed or point.event_time_s > timestamp:
                continue
            key = point.camera_id, point.stream_epoch, point.track_id
            x, y = point.x / self.width, point.y / self.height
            if not (math.isfinite(x) and math.isfinite(y) and -0.01 <= x <= 1.01 and -0.01 <= y <= 1.01):
                self.rejections["INVALID_POINT"] += 1
                continue
            segment = self._segments.get(key)
            if segment and segment.points:
                last_s, last_x, last_y = segment.points[-1]
                if point.frame_id <= segment.last_frame or point.event_time_s <= last_s:
                    continue
                distance = math.hypot(x - last_x, y - last_y)
                if point.event_time_s - last_s > cfg.max_observation_gap_seconds or distance > cfg.max_step_fraction:
                    self._closed.append(segment)
                    segment = None
                elif distance < cfg.min_step_fraction:
                    segment.last_frame = point.frame_id
                    continue
            if segment is None:
                self._number[key] += 1
                segment = _Segment(
                    key, self._number[key],
                    deque(maxlen=cfg.max_points_per_track), point.frame_id,
                )
                self._segments[key] = segment
            segment.points.append((point.event_time_s, x, y))
            segment.last_frame = point.frame_id

        cutoff = timestamp - cfg.evidence_window_seconds
        for key, segment in list(self._segments.items()):
            while segment.points and segment.points[0][0] < cutoff:
                segment.points.popleft()
            if not segment.points:
                del self._segments[key]
        retained: deque[_Segment] = deque()
        for segment in self._closed:
            while segment.points and segment.points[0][0] < cutoff:
                segment.points.popleft()
            if segment.points:
                retained.append(segment)
        self._closed = retained
        while len(self._closed) + len(self._segments) > cfg.max_tracks and self._closed:
            self._closed.popleft()
        if len(self._segments) > cfg.max_tracks:
            stale = sorted(self._segments, key=lambda k: self._segments[k].points[-1][0])
            for key in stale[:len(self._segments) - cfg.max_tracks]:
                del self._segments[key]

        if timestamp - self._last_compute >= cfg.update_interval_seconds:
            started = time.perf_counter()
            self._compute(timestamp)
            self.compute_ms.append((time.perf_counter() - started) * 1000)
            self._last_compute = timestamp
        return self._snapshot

    @staticmethod
    def _smooth(points: np.ndarray, passes: int = 2) -> np.ndarray:
        if len(points) < 3:
            return points.copy()
        result = points.astype(np.float64, copy=True)
        first, last = result[0].copy(), result[-1].copy()
        for _ in range(passes):
            padded = np.vstack((result[0], result, result[-1]))
            result = (padded[:-2] + 2.0 * padded[1:-1] + padded[2:]) / 4.0
            result[0], result[-1] = first, last
        return result

    @staticmethod
    def _deduplicate(points: np.ndarray) -> np.ndarray:
        if len(points) < 2:
            return points
        keep = np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-7]
        return points[keep]

    @staticmethod
    def _resample(points: np.ndarray, count: int = 48) -> np.ndarray:
        if len(points) == 0:
            return np.zeros((count, 2), dtype=np.float64)
        if len(points) == 1:
            return np.repeat(points.astype(np.float64), count, axis=0)
        distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
        cumulative = np.r_[0.0, np.cumsum(distances)]
        if cumulative[-1] <= 1e-9:
            return np.repeat(points[:1].astype(np.float64), count, axis=0)
        samples = np.linspace(0.0, cumulative[-1], count)
        return np.column_stack((
            np.interp(samples, cumulative, points[:, 0]),
            np.interp(samples, cumulative, points[:, 1]),
        ))

    def _sample(self, coordinates: np.ndarray) -> np.ndarray:
        """Compatibility helper used by older diagnostics and notebooks."""
        return self._resample(coordinates)

    def _tracklet_from_segment(self, segment: _Segment) -> _Tracklet | None:
        cfg = self.config
        if len(segment.points) < cfg.min_track_points:
            self.rejections["TOO_SHORT"] += 1
            return None
        times = np.asarray([item[0] for item in segment.points], dtype=np.float64)
        raw = self._deduplicate(np.asarray([(item[1], item[2]) for item in segment.points], dtype=np.float64))
        if len(raw) < cfg.min_track_points:
            self.rejections["TOO_SHORT"] += 1
            return None
        duration = float(times[-1] - times[0])
        if duration < cfg.min_track_duration_seconds:
            self.rejections["SHORT_DURATION"] += 1
            return None
        # Validate against the observed geometry before smoothing; otherwise a
        # back-and-forth jitter can be averaged into a convincing straight line.
        deltas = np.diff(raw, axis=0)
        lengths = np.linalg.norm(deltas, axis=1)
        arc_length = float(lengths.sum())
        displacement = float(np.linalg.norm(raw[-1] - raw[0]))
        if displacement < cfg.min_displacement_fraction or arc_length <= 1e-9:
            self.rejections["LOW_DISPLACEMENT"] += 1
            return None
        if arc_length / displacement > cfg.max_path_stretch:
            self.rejections["ZIGZAG_STRETCH"] += 1
            return None
        net = (raw[-1] - raw[0]) / displacement
        valid = lengths > 1e-8
        step_directions = deltas[valid] / lengths[valid, None]
        consistency = float(np.mean(step_directions @ net)) if len(step_directions) else 0.0
        if consistency < cfg.min_direction_consistency:
            self.rejections["WRONG_DIRECTION"] += 1
            return None
        points = self._smooth(raw)
        quality = max(0.0, min(1.0, consistency * min(1.0, displacement / max(cfg.min_displacement_fraction * 3.0, 1e-9))))
        return _Tracklet(
            segment.key,
            segment.number,
            float(times[0]),
            float(times[-1]),
            points,
            net,
            arc_length,
            displacement,
            quality,
            consistency,
        )

    @staticmethod
    def _direction_cos(first: np.ndarray, second: np.ndarray) -> float:
        denominator = max(float(np.linalg.norm(first) * np.linalg.norm(second)), 1e-9)
        return float(np.dot(first, second) / denominator)

    @staticmethod
    def _project(samples: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if len(target) < 2:
            return np.zeros(len(samples)), np.full(len(samples), math.inf)
        starts = target[:-1]
        vectors = target[1:] - starts
        lengths = np.sum(vectors * vectors, axis=1)
        delta = samples[:, None, :] - starts[None, :, :]
        fractions = np.clip(np.einsum("ijk,jk->ij", delta, vectors) /
                            np.maximum(lengths, 1e-10), 0.0, 1.0)
        projections = starts[None, :, :] + fractions[:, :, None] * vectors[None, :, :]
        squared = np.sum((projections - samples[:, None, :]) ** 2, axis=2)
        nearest = np.argmin(squared, axis=1)
        rows = np.arange(len(samples))
        return nearest + fractions[rows, nearest], np.sqrt(squared[rows, nearest])

    def _overlap_stats(self, first: np.ndarray, second: np.ndarray) -> tuple[float, float]:
        if len(first) < 2 or len(second) < 2:
            return 0.0, math.inf
        # Pair matching runs for every compatible tracklet pair. A compact
        # arc-length sample preserves overlap/direction evidence while keeping
        # live analytics bounded when a scene contains many IDs.
        if len(first) > 12:
            first = self._resample(first, 12)
        if len(second) > 12:
            second = self._resample(second, 12)
        positions, distances = self._project(first, second)
        segments = np.diff(second, axis=0)
        segment_index = np.minimum(len(segments) - 1, np.floor(positions).astype(int))
        local = segments[segment_index]
        tangents = np.diff(first, axis=0)
        tangents = np.vstack((tangents, tangents[-1]))
        cosines = np.array([
            self._direction_cos(tangent, vector)
            for tangent, vector in zip(tangents, local)
        ])
        valid = (distances <= self.config.match_distance_fraction) & (
            cosines >= math.cos(math.radians(self.config.match_angle_degrees))
        )
        if not np.any(valid):
            return 0.0, math.inf
        return float(np.mean(valid)), float(np.mean(distances[valid]))

    def _overlap(self, seed: np.ndarray, other: np.ndarray) -> dict[int, tuple[float, float, float]]:
        """Return ordered projections retained by the legacy diagnostic API."""
        if len(seed) < 2 or len(other) < 2:
            return {}
        positions, distances = self._project(seed, other)
        segments = np.diff(other, axis=0)
        segment_index = np.minimum(len(segments) - 1, np.floor(positions).astype(int))
        tangents = np.diff(seed, axis=0)
        tangents = np.vstack((tangents, tangents[-1]))
        valid = []
        for index, (position, distance, tangent, segment) in enumerate(
            zip(positions, distances, tangents, segments[segment_index])
        ):
            if distance <= self.config.match_distance_fraction and self._direction_cos(tangent, segment) >= math.cos(math.radians(self.config.match_angle_degrees)):
                valid.append((index, float(position), float(distance)))
        result: dict[int, tuple[float, float, float]] = {}
        previous = -1.0
        for index, position, distance in valid:
            if position + 0.05 < previous:
                continue
            previous = position
            segment_index_value = min(len(other) - 2, max(0, int(math.floor(position))))
            fraction = position - segment_index_value
            point = other[segment_index_value] + fraction * (other[segment_index_value + 1] - other[segment_index_value])
            result[index] = (float(point[0]), float(point[1]), distance)
        return result

    def _link_score(self, first: _Tracklet, second: _Tracklet) -> float:
        cfg = self.config
        direction = self._direction_cos(first.direction, second.direction)
        if direction < math.cos(math.radians(cfg.match_angle_degrees)):
            return 0.0
        overlap_a, distance_a = self._overlap_stats(first.points, second.points)
        overlap_b, distance_b = self._overlap_stats(second.points, first.points)
        overlap = max(overlap_a, overlap_b)
        mean_distance = min(distance_a, distance_b)
        overlap_time_gap = max(first.start_time, second.start_time) - min(first.end_time, second.end_time)
        if overlap >= cfg.min_overlap_fraction and overlap_time_gap <= cfg.max_link_gap_seconds:
            return 1.0 + overlap * 0.7 + direction * 0.25 - mean_distance

        before, after = (first, second) if first.end_time <= second.start_time else (second, first)
        gap = after.start_time - before.end_time
        if gap < 0.0 or gap > cfg.max_link_gap_seconds:
            return 0.0
        predicted = before.points[-1] + before.direction * min(
            before.displacement / max(before.end_time - before.start_time, 1e-6) * gap,
            cfg.max_step_fraction * 2.0,
        )
        endpoint_distance = float(np.linalg.norm(after.points[0] - predicted))
        if endpoint_distance > cfg.match_distance_fraction * 3.0:
            return 0.0
        return 0.65 + direction * 0.25 - endpoint_distance / max(cfg.match_distance_fraction * 3.0, 1e-9)

    def _merge_overlap(self, first: np.ndarray, second: np.ndarray) -> np.ndarray:
        """Median-blend compatible samples while retaining the longer route extent."""
        if len(second) > len(first) or np.linalg.norm(second[-1] - second[0]) > np.linalg.norm(first[-1] - first[0]):
            base, other = second, first
        else:
            base, other = first, second
        route = self._resample(base)
        positions, distances = self._project(other, route)
        bins: list[list[np.ndarray]] = [[] for _ in route]
        for point, position, distance in zip(other, positions, distances):
            if distance <= self.config.match_distance_fraction * 1.5:
                bins[min(len(route) - 1, max(0, round(float(position))))].append(point)
        merged = route.copy()
        for index, values in enumerate(bins):
            if values:
                merged[index] = np.median(np.vstack([merged[index], *values]), axis=0)
        return self._smooth(merged)

    def _attach(self, route: np.ndarray, tracklet: _Tracklet) -> tuple[float, np.ndarray]:
        cfg = self.config
        candidate = tracklet.points
        direction = self._direction_cos(route[-1] - route[0], tracklet.direction)
        if direction < math.cos(math.radians(cfg.match_angle_degrees)):
            return 0.0, route
        overlap_a, _ = self._overlap_stats(route, candidate)
        overlap_b, _ = self._overlap_stats(candidate, route)
        overlap = max(overlap_a, overlap_b)
        if overlap >= cfg.min_overlap_fraction:
            return 2.0 + overlap, self._merge_overlap(route, candidate)
        threshold = cfg.match_distance_fraction * 3.0
        append_distance = float(np.linalg.norm(route[-1] - candidate[0]))
        prepend_distance = float(np.linalg.norm(candidate[-1] - route[0]))
        if append_distance <= threshold:
            bridge = (route[-1] + candidate[0]) / 2.0
            merged = np.vstack((route[:-1], bridge, candidate))
            return 1.0 - append_distance / threshold, self._smooth(merged)
        if prepend_distance <= threshold:
            bridge = (candidate[-1] + route[0]) / 2.0
            merged = np.vstack((candidate[:-1], bridge, route))
            return 1.0 - prepend_distance / threshold, self._smooth(merged)
        return 0.0, route

    def _merge_cluster(self, tracklets: list[_Tracklet]) -> np.ndarray:
        seed = max(tracklets, key=lambda item: (item.arc_length, len(item.points)))
        route = seed.points.copy()
        remaining = [item for item in tracklets if item is not seed]
        while remaining:
            best_index, best_score, best_route = -1, 0.0, route
            for index, tracklet in enumerate(remaining):
                score, merged = self._attach(route, tracklet)
                if score > best_score:
                    best_index, best_score, best_route = index, score, merged
            if best_index < 0:
                break
            route = best_route
            remaining.pop(best_index)
        route = self._smooth(self._deduplicate(route), passes=2)
        return self._resample(route, min(48, max(8, len(route))))

    def _compute(self, timestamp: float) -> None:
        cfg = self.config
        tracklets: list[_Tracklet] = []
        segments = sorted(
            chain(self._closed, self._segments.values()),
            key=lambda item: item.points[-1][0] if item.points else -math.inf,
            reverse=True,
        )[:max(cfg.max_matching_tracklets * 2, cfg.max_matching_tracklets + 256)]
        for segment in segments:
            tracklet = self._tracklet_from_segment(segment)
            if tracklet is not None:
                tracklets.append(tracklet)
        if len(tracklets) > cfg.max_matching_tracklets:
            tracklets.sort(key=lambda item: (item.end_time, item.quality, item.arc_length), reverse=True)
            dropped = len(tracklets) - cfg.max_matching_tracklets
            tracklets = tracklets[:cfg.max_matching_tracklets]
            self.rejections["MATCHING_TRACKLET_CAP"] += dropped
        self.rejections["VALID_TRACKLETS"] = len(tracklets)
        if not tracklets:
            self.candidate_count = 0
            self._update_identities((), timestamp)
            return

        parent = list(range(len(tracklets)))

        def find(value: int) -> int:
            while parent[value] != value:
                parent[value] = parent[parent[value]]
                value = parent[value]
            return value

        def union(left: int, right: int) -> None:
            root_left, root_right = find(left), find(right)
            if root_left != root_right:
                parent[root_right] = root_left

        # Build a nearest-neighbour graph using cheap endpoint/centroid
        # distances, then run the more expensive polyline projection only on
        # those pairs. This is spatially continuous matching, not grid logic.
        centers = np.asarray([item.points.mean(axis=0) for item in tracklets])
        starts = np.asarray([item.points[0] for item in tracklets])
        ends = np.asarray([item.points[-1] for item in tracklets])
        neighbor_count = min(24, max(1, len(tracklets) - 1))
        pair_candidates: set[tuple[int, int]] = set()
        for index in range(len(tracklets)):
            center_distance = np.linalg.norm(centers - centers[index], axis=1)
            endpoint_distance = np.minimum(
                np.linalg.norm(starts - ends[index], axis=1),
                np.linalg.norm(ends - starts[index], axis=1),
            )
            nearest = np.argsort(np.minimum(center_distance, endpoint_distance))[:neighbor_count + 1]
            for other_index in nearest:
                other_index = int(other_index)
                if other_index != index:
                    pair_candidates.add((min(index, other_index), max(index, other_index)))
        for index, other_index in pair_candidates:
            first, second = tracklets[index], tracklets[other_index]
            if self._direction_cos(first.direction, second.direction) < math.cos(math.radians(cfg.match_angle_degrees)):
                continue
            if self._link_score(first, second) > 0.0:
                union(index, other_index)

        grouped: dict[int, list[_Tracklet]] = {}
        for index, tracklet in enumerate(tracklets):
            grouped.setdefault(find(index), []).append(tracklet)

        candidates: list[CommonPath] = []
        for cluster in grouped.values():
            support_ids = {item.key[2] for item in cluster}
            if len(support_ids) < cfg.min_support_tracks:
                self.rejections["INSUFFICIENT_SUPPORT_IDS"] += 1
                continue
            polyline = self._merge_cluster(cluster)
            if len(polyline) < 2:
                self.rejections["EMPTY_MERGE"] += 1
                continue
            quality = float(statistics.mean(item.quality for item in cluster))
            temporal_coverage = min(1.0, len(cluster) / max(cfg.min_support_tracks, 1))
            score = len(support_ids) * (0.55 + 0.45 * temporal_coverage) * (0.5 + 0.5 * quality)
            evidence = max(item.end_time for item in cluster)
            candidates.append(CommonPath(
                "", "tracklet", "common_path", "candidate", len(support_ids), len(support_ids),
                score, quality, "FORWARD", tuple(
                    (float(x * self.width), float(y * self.height)) for x, y in polyline
                ), timestamp, coordinate_space="image_pixels", support_tracks=len(support_ids),
                evidence_until_s=evidence,
            ))

        candidates.sort(key=lambda item: (-item.score, -item.support_tracks, -item.confidence))
        distinct: list[CommonPath] = []
        for candidate in candidates:
            if any(self._path_similarity(candidate, other) >= 0.84 for other in distinct):
                self.rejections["DUPLICATE_GEOMETRY"] += 1
                continue
            distinct.append(candidate)
            if len(distinct) >= cfg.max_candidates:
                break
        self.candidate_count = len(distinct)
        self._update_identities(tuple(distinct), timestamp)

    def _path_similarity(self, first: CommonPath, second: CommonPath) -> float:
        a = np.asarray(first.polyline, dtype=np.float64) / (self.width, self.height)
        b = np.asarray(second.polyline, dtype=np.float64) / (self.width, self.height)
        if len(a) < 2 or len(b) < 2:
            return 0.0
        direction = self._direction_cos(a[-1] - a[0], b[-1] - b[0])
        if direction < math.cos(math.radians(self.config.match_angle_degrees)):
            return 0.0
        overlap_a, distance_a = self._overlap_stats(a, b)
        overlap_b, distance_b = self._overlap_stats(b, a)
        overlap = max(overlap_a, overlap_b)
        endpoint_distances = (
            float(np.linalg.norm(a[0] - b[0])),
            float(np.linalg.norm(a[-1] - b[-1])),
            # A new candidate may be a forward extension of the remembered
            # route, so compare directed end-to-start joins as well.
            float(np.linalg.norm(a[-1] - b[0])),
            float(np.linalg.norm(b[-1] - a[0])),
        )
        endpoint = min(endpoint_distances)
        endpoint_score = max(0.0, 1.0 - endpoint / max(self.config.match_distance_fraction * 4.0, 1e-9))
        distance_score = max(0.0, 1.0 - min(distance_a, distance_b) /
                             max(self.config.match_distance_fraction, 1e-9))
        return max(overlap, endpoint_score * 0.65 + distance_score * 0.35)

    def _merge_identity_paths(self, previous: CommonPath, candidate: CommonPath) -> tuple[tuple[float, float], ...]:
        """Keep the remembered route extent while attaching a new route section."""
        first = np.asarray(previous.polyline, dtype=np.float64)
        second = np.asarray(candidate.polyline, dtype=np.float64)
        if len(first) < 2:
            return tuple((float(x), float(y)) for x, y in second)
        if len(second) < 2:
            return tuple((float(x), float(y)) for x, y in first)
        scale = np.asarray((self.width, self.height), dtype=np.float64)
        first_normalized, second_normalized = first / scale, second / scale
        overlap_first, _ = self._overlap_stats(first_normalized, second_normalized)
        overlap_second, _ = self._overlap_stats(second_normalized, first_normalized)
        if max(overlap_first, overlap_second) >= self.config.min_overlap_fraction:
            merged = self._merge_overlap(first_normalized, second_normalized) * scale
            return tuple((float(x), float(y)) for x, y in merged)

        threshold = self.config.match_distance_fraction * 4.0
        if np.linalg.norm(first_normalized[-1] - second_normalized[0]) <= threshold:
            bridge = (first[-1] + second[0]) / 2.0
            merged = np.vstack((first[:-1], bridge, second))
        elif np.linalg.norm(second_normalized[-1] - first_normalized[0]) <= threshold:
            bridge = (second[-1] + first[0]) / 2.0
            merged = np.vstack((second[:-1], bridge, first))
        else:
            # A weak match is not evidence that the remembered route should
            # be shortened. Keep the complete camera-entry-to-exit route and
            # wait for an overlap or directed endpoint join before extending.
            merged = first.copy()
        merged = self._smooth(self._deduplicate(merged), passes=2)
        return tuple(
            (float(x), float(y)) for x, y in self._resample(merged, min(48, max(8, len(merged))))
        )

    @staticmethod
    def _blend_paths(first: CommonPath, second: CommonPath, alpha: float = 0.30) -> tuple[tuple[float, float], ...]:
        a = TrackletAggregationEngine._resample(np.asarray(first.polyline, dtype=np.float64))
        b = TrackletAggregationEngine._resample(np.asarray(second.polyline, dtype=np.float64))
        blended = a * (1.0 - alpha) + b * alpha
        blended = TrackletAggregationEngine._smooth(blended, passes=1)
        return tuple((float(x), float(y)) for x, y in blended)

    def _update_identities(self, candidates: tuple[CommonPath, ...], timestamp: float) -> None:
        unmatched = set(self._identities)
        for candidate in candidates:
            choices = sorted(
                ((self._path_similarity(candidate, self._identities[item].path), item)
                 for item in unmatched), reverse=True,
            )
            similarity, matched = choices[0] if choices else (0.0, None)
            if matched is None or similarity < 0.48:
                self._sequence += 1
                matched = f"path-{self._sequence:03d}"
                self._identities[matched] = _Identity(candidate, timestamp, timestamp)
                self.events.append({
                    "event": "path_created", "path_id": matched,
                    "event_time_s": timestamp, "support_tracks": candidate.support_tracks,
                })
            else:
                unmatched.remove(matched)
            identity = self._identities[matched]
            if identity.last_seen <= timestamp and identity.path.path_id == matched and identity.path.revision > 0:
                alpha = self.config.path_ema_alpha
                previous = identity.path
                candidate = replace(
                    candidate,
                    polyline=self._merge_identity_paths(previous, candidate),
                    score=previous.score * (1.0 - alpha) + candidate.score * alpha,
                    confidence=previous.confidence * (1.0 - alpha) + candidate.confidence * alpha,
                    support_tracks=max(1, round(previous.support_tracks * (1.0 - alpha) +
                                                candidate.support_tracks * alpha)),
                    unique_tracks_short=max(1, round(previous.unique_tracks_short * (1.0 - alpha) +
                                                     candidate.unique_tracks_short * alpha)),
                    unique_tracks_long=max(1, round(previous.unique_tracks_long * (1.0 - alpha) +
                                                    candidate.unique_tracks_long * alpha)),
                )
            state = "active" if timestamp - identity.first_seen >= self.config.confirmation_seconds else "candidate"
            identity.path = replace(
                candidate,
                path_id=matched,
                color=COLORS[(int(matched[5:]) - 1) % len(COLORS)],
                state=state,
                revision=identity.path.revision + 1,
            )
            identity.last_seen = timestamp
            identity.missing_since = None
        for path_id in unmatched:
            identity = self._identities[path_id]
            age = timestamp - identity.last_seen
            if identity.missing_since is None:
                identity.missing_since = timestamp
            route_memory = self.config.route_memory_seconds or self.config.cooling_seconds * 2.0
            if identity.path.state == "candidate" and age > self.config.cooling_seconds:
                del self._identities[path_id]
            elif identity.path.state in {"active", "cooling"} and age > route_memory:
                self.events.append({
                    "event": "path_retired", "path_id": path_id,
                    "event_time_s": timestamp, "evidence_until_s": identity.path.evidence_until_s,
                })
                del self._identities[path_id]
            elif identity.path.state in {"active", "cooling"} and age >= self.config.cooling_seconds:
                if identity.path.state == "active":
                    self.events.append({
                        "event": "path_cooling", "path_id": path_id,
                        "event_time_s": timestamp, "evidence_until_s": identity.path.evidence_until_s,
                    })
                identity.path = replace(identity.path, state="cooling")
        if len(self.events) > 512:
            del self.events[:-512]
        if len(self._identities) > self.config.max_candidates * 3:
            stale = sorted(self._identities, key=lambda item: self._identities[item].last_seen)
            for path_id in stale[:len(self._identities) - self.config.max_candidates * 3]:
                del self._identities[path_id]
        self._publish(timestamp)

    def _publish(self, timestamp: float) -> None:
        cfg = self.config
        eligible = sorted(
            (state for state in self._identities.values() if state.path.state == "active"),
            key=lambda state: (-state.path.score, state.path.path_id),
        )
        incumbents = sorted(
            (state for state in self._identities.values()
             if state.path.state in {"active", "cooling"} and state.selected_at is not None),
            key=lambda state: state.selected_at or 0,
        )
        selected = incumbents[:self.max_paths]
        for challenger in eligible:
            if challenger in selected:
                continue
            if len(selected) < self.max_paths:
                selected.append(challenger)
            else:
                weakest = min(selected, key=lambda state: state.path.score)
                if (challenger.path.support_tracks >= weakest.path.support_tracks + cfg.switch_margin_tracks
                        and timestamp - challenger.first_seen >= max(
                            cfg.confirmation_seconds, cfg.switch_hold_seconds
                        )):
                    selected.remove(weakest)
                    selected.append(challenger)
        for identity in self._identities.values():
            if identity not in selected:
                identity.selected_at = None
        for identity in selected:
            if identity.selected_at is None:
                identity.selected_at = timestamp
        paths = tuple(
            replace(state.path, rank=index + 1)
            for index, state in enumerate(sorted(selected, key=lambda item: (-item.path.score, item.path.path_id)))
        )
        self._snapshot = CommonPathSnapshot(
            timestamp, paths, self._snapshot.version + 1,
            max((path.evidence_until_s for path in paths), default=timestamp),
        )
