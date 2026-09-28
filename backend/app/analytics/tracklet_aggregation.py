"""Tracklet-based Common Path aggregation without a spatial grid.

The engine keeps short observed tracklets, rejects weak or inconsistent motion,
links compatible tracklets with a directed graph, and merges each component into
one smoothed representative polyline. A grid is deliberately not used for the
Common Path decision; the engine consumes only camera/session, track, frame,
timestamp, and coordinate fields.
"""

from __future__ import annotations

import math
import statistics
import time
from collections import Counter, deque
from collections.abc import Iterable
from copy import deepcopy
from dataclasses import dataclass, replace
from itertools import chain

import numpy as np

from backend.app.analytics.spatial import SpatialTransformer
from backend.app.core.config import TrackletAggregationConfig
from backend.app.schemas import CommonPath, CommonPathSnapshot

TrackKey = tuple[str, str, int]
COLORS = ("#ffca42", "#37d5c4", "#ff718d", "#80aaff", "#d3a1f5", "#9ce56a", "#ff9e68")


@dataclass(frozen=True, slots=True)
class TrackletPoint:
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


@dataclass
class _Segment:
    key: TrackKey
    number: int
    segment_id: int
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
        self._last_ingested: dict[tuple[TrackKey, int], tuple[int, float]] = {}
        self._tracklet_cache: dict[tuple[object, ...], _Tracklet | None] = {}
        self._link_score_cache: dict[tuple[object, ...], float] = {}
        self.link_score_cache_hits = 0
        self._last_compute_new_segments = 0
        self._identities: dict[str, _Identity] = {}
        self._sequence = 0
        self._last_time = -math.inf
        self._last_compute = -math.inf
        self._snapshot = CommonPathSnapshot(0, ())
        self.events: list[dict] = []
        # Bounded evidence buffers for offline diagnostics/export.  Keep them
        # separate from the compact product snapshot and return copies below.
        self._candidate_diagnostics: deque[dict[str, object]] = deque(maxlen=512)
        self._path_support_diagnostics: deque[dict[str, object]] = deque(maxlen=512)
        self.rejections: Counter[str] = Counter()
        self.compute_ms: deque[float] = deque(maxlen=256)
        self.candidate_count = 0
        self.candidate_pool_count = 0

    def reset(self) -> None:
        self._segments.clear()
        self._closed.clear()
        self._number.clear()
        self._last_ingested.clear()
        self._tracklet_cache.clear()
        self._link_score_cache.clear()
        self.link_score_cache_hits = 0
        self._last_compute_new_segments = 0
        self._identities.clear()
        self._snapshot = CommonPathSnapshot(0, ())
        self._last_compute = -math.inf
        self._last_time = -math.inf
        self.events.clear()
        self._candidate_diagnostics.clear()
        self._path_support_diagnostics.clear()
        self.rejections.clear()
        self.compute_ms.clear()
        self.candidate_count = 0
        self.candidate_pool_count = 0

    def set_max_paths(self, value: int) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 5:
            raise ValueError("max_paths must be an integer in [1, 5]")
        self.max_paths = value
        self._publish(self._last_time if math.isfinite(self._last_time) else 0)
        return value

    def snapshot(self) -> CommonPathSnapshot:
        return self._snapshot

    @property
    def candidate_decisions(self) -> tuple[dict[str, object], ...]:
        """Bounded candidate funnel for diagnostics/export."""
        return tuple(deepcopy(item) for item in self._candidate_diagnostics)

    @property
    def path_support_diagnostics(self) -> tuple[dict[str, object], ...]:
        """Bounded selected-path support history for diagnostics/export."""
        return tuple(deepcopy(item) for item in self._path_support_diagnostics)

    def diagnostics(self) -> dict[str, object]:
        """Return Common Path evidence without exposing mutable internals."""
        return {
            "candidate_pool_count": self.candidate_pool_count,
            "candidate_count": self.candidate_count,
            "candidate_decisions": self.candidate_decisions,
            "path_support": self.path_support_diagnostics,
            "rejections": dict(self.rejections),
        }

    def finalize(self, timestamp: float | None = None) -> CommonPathSnapshot:
        """Evaluate the final active segments without waiting for the interval."""
        if self._last_time == -math.inf and timestamp is None:
            return self._snapshot
        event_time = self._last_time if timestamp is None else timestamp
        if event_time < self._last_time:
            raise ValueError("finalization timestamp moved backwards")
        self._last_time = event_time
        self._compute(event_time)
        self._last_compute = event_time
        return self._snapshot

    @property
    def buffered_segment_count(self) -> int:
        """Number of active and closed segments retained by the bounded buffer."""
        return len(self._segments) + len(self._closed)

    @property
    def last_compute_new_segments(self) -> int:
        """How many segment signatures required fresh validation last compute."""
        return self._last_compute_new_segments

    def update(self, points: Iterable[TrackletPoint], timestamp: float) -> CommonPathSnapshot:
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
            watermark_key = (key, int(point.segment_id))
            previous_ingest = self._last_ingested.get(watermark_key)
            if previous_ingest is not None and (
                point.frame_id <= previous_ingest[0]
                or point.event_time_s <= previous_ingest[1]
            ):
                self.rejections["DUPLICATE_OR_LATE_POINT"] += 1
                continue
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
                if (
                    point.segment_id != segment.segment_id
                    or point.event_time_s - last_s > cfg.max_observation_gap_seconds
                    or distance > cfg.max_step_fraction
                ):
                    self._closed.append(segment)
                    segment = None
                elif distance < cfg.min_step_fraction:
                    segment.last_frame = point.frame_id
                    continue
            if segment is None:
                self._number[key] += 1
                segment = _Segment(
                    key, self._number[key], int(point.segment_id),
                    deque(maxlen=cfg.max_points_per_track), point.frame_id,
                )
                self._segments[key] = segment
            segment.points.append((point.event_time_s, x, y))
            segment.last_frame = point.frame_id
            self._last_ingested[watermark_key] = (point.frame_id, point.event_time_s)

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

        # Watermarks are bounded by the same evidence horizon as segments.
        self._last_ingested = {
            key: value for key, value in self._last_ingested.items()
            if value[1] >= cutoff
        }

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

    def _resample_by_spacing(self, points: np.ndarray) -> np.ndarray:
        """Resample an observed route into short straight segments.

        The interpolation is only along the observed piecewise-linear route;
        no spline or endpoint-only shortcut is introduced.  A bounded point
        count keeps rendering and identity matching predictable on long clips.
        """
        points = self._deduplicate(np.asarray(points, dtype=np.float64))
        if len(points) < 2:
            return points.copy()
        spacing = max(float(self.config.segment_length_fraction), 1e-6)
        length = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
        if length <= 1e-9:
            return points[:1].copy()
        count = int(math.ceil(length / spacing)) + 1
        count = max(2, min(128, count))
        return self._resample(points, count)

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
        # Bottom-centre coordinates carry a perspective scale signal: at the
        # top of the frame the same world motion occupies fewer pixels.  Use a
        # continuous image-derived factor instead of hard-coding zones so small
        # distant people are not discarded by one global displacement floor.
        mean_y = float(np.mean(raw[:, 1])) if len(raw) else 0.5
        perspective_scale = max(0.35, min(1.0, 0.35 + 0.65 * mean_y))
        min_displacement = cfg.min_displacement_fraction * perspective_scale
        if displacement < min_displacement or arc_length <= 1e-9:
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
        quality = max(0.0, min(1.0, consistency * min(1.0, displacement / max(min_displacement * 3.0, 1e-9))))
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
        first_fingerprint = (
            first.key, first.number, round(first.start_time, 6), round(first.end_time, 6),
            len(first.points), tuple(np.round(first.points[-1], 6)),
        )
        second_fingerprint = (
            second.key, second.number, round(second.start_time, 6), round(second.end_time, 6),
            len(second.points), tuple(np.round(second.points[-1], 6)),
        )
        cache_key = (first_fingerprint, second_fingerprint)
        cached = self._link_score_cache.get(cache_key)
        if cached is not None:
            self.link_score_cache_hits += 1
            return cached
        direction = self._direction_cos(first.direction, second.direction)
        if direction < math.cos(math.radians(cfg.match_angle_degrees)):
            score = 0.0
            self._link_score_cache[cache_key] = score
            return score
        overlap_a, distance_a = self._overlap_stats(first.points, second.points)
        overlap_b, distance_b = self._overlap_stats(second.points, first.points)
        overlap = max(overlap_a, overlap_b)
        mean_distance = min(distance_a, distance_b)
        overlap_time_gap = max(first.start_time, second.start_time) - min(first.end_time, second.end_time)
        if overlap >= cfg.min_overlap_fraction:
            # Different temporary IDs are commonly produced when people enter
            # the same corridor at different times.  A strong directed
            # geometric overlap is evidence for one common path even when the
            # time gap exceeds the short same-track observation gap.  Keep the
            # strict temporal gate for fragments of the same track ID, where a
            # long gap is more likely disappearance/ID reuse.
            same_track = first.key == second.key
            if not same_track or overlap_time_gap <= cfg.max_link_gap_seconds:
                temporal_penalty = 0.0
                if overlap_time_gap > 0.0 and cfg.max_link_gap_seconds > 0.0:
                    temporal_penalty = min(
                        0.15,
                        overlap_time_gap / cfg.max_link_gap_seconds * 0.15,
                    )
                score = 1.0 + overlap * 0.7 + direction * 0.25 - mean_distance - temporal_penalty
                self._link_score_cache[cache_key] = score
                if len(self._link_score_cache) > self.config.max_matching_tracklets * 256:
                    self._link_score_cache.pop(next(iter(self._link_score_cache)))
                return score

        before, after = (first, second) if first.end_time <= second.start_time else (second, first)
        gap = after.start_time - before.end_time
        if gap < 0.0 or gap > cfg.max_link_gap_seconds:
            score = 0.0
            self._link_score_cache[cache_key] = score
            return score
        predicted = before.points[-1] + before.direction * min(
            before.displacement / max(before.end_time - before.start_time, 1e-6) * gap,
            cfg.max_step_fraction * 2.0,
        )
        endpoint_distance = float(np.linalg.norm(after.points[0] - predicted))
        if endpoint_distance > cfg.match_distance_fraction * 3.0:
            score = 0.0
        else:
            score = 0.65 + direction * 0.25 - endpoint_distance / max(cfg.match_distance_fraction * 3.0, 1e-9)
        self._link_score_cache[cache_key] = score
        if len(self._link_score_cache) > self.config.max_matching_tracklets * 256:
            self._link_score_cache.pop(next(iter(self._link_score_cache)))
        return score

    def _merge_overlap(self, first: np.ndarray, second: np.ndarray) -> np.ndarray:
        """Median-blend compatible samples while retaining the longer route extent."""
        if len(second) > len(first) or np.linalg.norm(second[-1] - second[0]) > np.linalg.norm(first[-1] - first[0]):
            base, other = second, first
        else:
            base, other = first, second
        route = self._resample_by_spacing(base)
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
        if len(route) < 2 or len(candidate) < 2:
            return 0.0, route
        # A curved common path may change heading substantially from its
        # origin to its destination. Compare a candidate with the local
        # tangent at the endpoint being joined, rather than the route's
        # end-to-end vector. This preserves real turns while still rejecting
        # opposite-direction segments at intersections.
        angle_gate = math.cos(math.radians(cfg.match_angle_degrees))
        append_direction = self._direction_cos(route[-1] - route[-2], tracklet.direction)
        prepend_direction = self._direction_cos(route[1] - route[0], tracklet.direction)
        if max(append_direction, prepend_direction) < angle_gate:
            return 0.0, route
        overlap_a, _ = self._overlap_stats(route, candidate)
        overlap_b, _ = self._overlap_stats(candidate, route)
        overlap = max(overlap_a, overlap_b)
        if overlap >= cfg.min_overlap_fraction:
            return 2.0 + overlap, self._merge_overlap(route, candidate)
        threshold = cfg.match_distance_fraction * 3.0
        append_distance = float(np.linalg.norm(route[-1] - candidate[0]))
        prepend_distance = float(np.linalg.norm(candidate[-1] - route[0]))
        if append_distance <= threshold and append_direction >= angle_gate:
            bridge = (route[-1] + candidate[0]) / 2.0
            merged = np.vstack((route[:-1], bridge, candidate))
            return 1.0 + append_direction - append_distance / threshold, self._smooth(merged)
        if prepend_distance <= threshold and prepend_direction >= angle_gate:
            bridge = (candidate[-1] + route[0]) / 2.0
            merged = np.vstack((candidate[:-1], bridge, route))
            return 1.0 + prepend_direction - prepend_distance / threshold, self._smooth(merged)
        return 0.0, route

    def _merge_cluster(self, tracklets: list[_Tracklet]) -> np.ndarray:
        # Pick a representative-quality seed instead of giving the longest or
        # densest track disproportionate influence over the common route.
        median_length = statistics.median(item.arc_length for item in tracklets)
        seed = min(
            tracklets,
            key=lambda item: (-item.quality, abs(item.arc_length - median_length), item.start_time, item.key),
        )
        route = seed.points.copy()
        remaining = [item for item in tracklets if item is not seed]
        attached = [seed]
        while remaining:
            best_index, best_score, best_route = -1, 0.0, route
            for index, tracklet in enumerate(remaining):
                # Geometry alone is insufficient at a crossing.  Require a
                # direct directed overlap or temporally plausible endpoint
                # transition with at least one already attached observation.
                evidence = max((self._link_score(tracklet, item) for item in attached), default=0.0)
                if evidence <= 0.0:
                    continue
                score, merged = self._attach(route, tracklet)
                if score > best_score:
                    best_index, best_score, best_route = index, score, merged
            if best_index < 0:
                break
            route = best_route
            attached.append(remaining.pop(best_index))
        route = self._smooth(self._deduplicate(route), passes=1)
        return self._resample_by_spacing(route)

    def _evidence_coverage(self, tracklets: list[_Tracklet], route: np.ndarray) -> float:
        """Estimate route-bin coverage from observed tracklet geometry.

        This is a diagnostic only.  Bins are marked once across the candidate,
        so a high-FPS or unusually long track cannot inflate coverage by adding
        more samples.
        """
        if len(route) < 2 or not tracklets:
            return 0.0
        bins = np.zeros(32, dtype=bool)
        tolerance = self.config.match_distance_fraction * 1.5
        for tracklet in tracklets:
            positions, distances = self._project(tracklet.points, route)
            valid = distances <= tolerance
            if not np.any(valid):
                continue
            normalized = np.clip(
                positions[valid] / max(len(route) - 1, 1), 0.0, 1.0
            )
            indices = np.rint(normalized * (len(bins) - 1)).astype(int)
            bins[indices] = True
        return float(np.mean(bins))

    def _record_candidate_diagnostic(self, payload: dict[str, object]) -> dict[str, object]:
        """Append JSON-friendly bounded candidate evidence."""
        normalized: dict[str, object] = {}
        for key, value in payload.items():
            if isinstance(value, np.integer):
                normalized[key] = int(value)
            elif isinstance(value, np.floating):
                normalized[key] = float(value)
            else:
                normalized[key] = value
        self._candidate_diagnostics.append(normalized)
        return normalized

    def _compute(self, timestamp: float) -> None:
        cfg = self.config
        tracklets: list[_Tracklet] = []
        segments = sorted(
            chain(self._closed, self._segments.values()),
            key=lambda item: item.points[-1][0] if item.points else -math.inf,
            reverse=True,
        )[:max(cfg.max_matching_tracklets * 2, cfg.max_matching_tracklets + 256)]
        new_segments = 0
        active_signatures: set[tuple[object, ...]] = set()
        for segment in segments:
            if not segment.points:
                continue
            first_time, first_x, first_y = segment.points[0]
            last_time, last_x, last_y = segment.points[-1]
            signature = (
                segment.key,
                segment.number,
                segment.segment_id,
                segment.last_frame,
                len(segment.points),
                float(first_time),
                float(first_x),
                float(first_y),
                float(last_time),
                float(last_x),
                float(last_y),
            )
            active_signatures.add(signature)
            if signature not in self._tracklet_cache:
                self._tracklet_cache[signature] = self._tracklet_from_segment(segment)
                new_segments += 1
            tracklet = self._tracklet_cache[signature]
            if tracklet is not None:
                tracklets.append(tracklet)
        self._tracklet_cache = {
            signature: value for signature, value in self._tracklet_cache.items()
            if signature in active_signatures
        }
        self._last_compute_new_segments = new_segments
        if len(tracklets) > cfg.max_matching_tracklets:
            tracklets.sort(key=lambda item: (item.end_time, item.quality, item.arc_length), reverse=True)
            dropped = len(tracklets) - cfg.max_matching_tracklets
            tracklets = tracklets[:cfg.max_matching_tracklets]
            self.rejections["MATCHING_TRACKLET_CAP"] += dropped
        self.rejections["VALID_TRACKLETS"] = len(tracklets)
        self.candidate_pool_count = 0
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
        candidate_diagnostic_by_object: dict[int, dict[str, object]] = {}
        for cluster_index, cluster in enumerate(grouped.values()):
            support_ids = {item.key[2] for item in cluster}
            candidate_id = f"candidate-{timestamp:.3f}-{cluster_index:03d}"
            if len(support_ids) < cfg.min_support_tracks:
                self.rejections["INSUFFICIENT_SUPPORT_IDS"] += 1
                self._record_candidate_diagnostic({
                    "candidate_id": candidate_id,
                    "timestamp": float(timestamp),
                    "support_tracks": len(support_ids),
                    "raw_track_ids": sorted(int(track_id) for track_id in support_ids),
                    "segment_count": len(cluster),
                    "decision_reason": "INSUFFICIENT_SUPPORT_IDS",
                })
                continue
            polyline = self._merge_cluster(cluster)
            if len(polyline) < 2:
                self.rejections["EMPTY_MERGE"] += 1
                self._record_candidate_diagnostic({
                    "candidate_id": candidate_id,
                    "timestamp": float(timestamp),
                    "support_tracks": len(support_ids),
                    "raw_track_ids": sorted(int(track_id) for track_id in support_ids),
                    "segment_count": len(cluster),
                    "decision_reason": "EMPTY_MERGE",
                })
                continue
            quality = float(statistics.mean(item.quality for item in cluster))
            temporal_coverage = min(1.0, len(cluster) / max(cfg.min_support_tracks, 1))
            evidence_coverage = self._evidence_coverage(cluster, polyline)
            # Support is the number of distinct observed track IDs.  Route
            # length, bbox size and sample count are deliberately absent from
            # the ranking; coverage is capped so repeated samples cannot win.
            score = len(support_ids) * (0.65 + 0.35 * quality) * (0.75 + 0.25 * evidence_coverage)
            evidence = max(item.end_time for item in cluster)
            route_vector = polyline[-1] - polyline[0]
            direction_label = "FORWARD" if (
                route_vector[0] > 1e-6
                or (abs(route_vector[0]) <= 1e-6 and route_vector[1] >= 0.0)
            ) else "REVERSE"
            path = CommonPath(
                "", "tracklet", "common_path", "candidate", len(support_ids), len(support_ids),
                score, quality, direction_label, tuple(
                    (float(x * self.width), float(y * self.height)) for x, y in polyline
                ), timestamp, coordinate_space="image_pixels", support_tracks=len(support_ids),
                evidence_until_s=evidence,
            )
            candidates.append(path)
            diagnostic = {
                "candidate_id": candidate_id,
                "timestamp": float(timestamp),
                "raw_track_ids": sorted(int(track_id) for track_id in support_ids),
                "support_tracks": len(support_ids),
                "segment_count": len(cluster),
                "score": float(score),
                "score_components": {
                    "support_tracks": len(support_ids),
                    "temporal_coverage_proxy": float(temporal_coverage),
                    "mean_quality": float(quality),
                },
                "evidence_coverage": evidence_coverage,
                "direction_consistency": float(
                    statistics.mean(item.direction_consistency for item in cluster)
                ),
                "direction_vector": [float(route_vector[0]), float(route_vector[1])],
                "direction_label": direction_label,
                "decision_reason": "POOL_CANDIDATE",
            }
            candidate_diagnostic_by_object[id(path)] = self._record_candidate_diagnostic(diagnostic)

        candidates.sort(key=lambda item: (-item.score, -item.support_tracks, -item.confidence))
        self.candidate_pool_count = len(candidates)
        distinct: list[CommonPath] = []
        for candidate in candidates:
            if any(self._path_similarity(candidate, other) >= 0.84 for other in distinct):
                self.rejections["DUPLICATE_GEOMETRY"] += 1
                diagnostic = candidate_diagnostic_by_object.get(id(candidate))
                if diagnostic is not None:
                    diagnostic["decision_reason"] = "DUPLICATE_GEOMETRY"
                continue
            if len(distinct) >= cfg.max_candidates:
                self.rejections["CANDIDATE_CAP"] += 1
                diagnostic = candidate_diagnostic_by_object.get(id(candidate))
                if diagnostic is not None:
                    diagnostic["decision_reason"] = "MAX_CANDIDATE_CAP"
                continue
            distinct.append(candidate)
            diagnostic = candidate_diagnostic_by_object.get(id(candidate))
            if diagnostic is not None:
                diagnostic["decision_reason"] = "SELECTED_CANDIDATE_POOL"
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
            (float(x), float(y)) for x, y in self._resample_by_spacing(merged)
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
        for path in paths:
            self._path_support_diagnostics.append({
                "timestamp": float(timestamp),
                "path_id": path.path_id,
                "rank": int(path.rank),
                "state": path.state,
                "support_tracks": int(path.support_tracks),
                # Until passage stitching/ground truth is available, expose
                # the actual unit instead of presenting temporary IDs as
                # unique people.
                "support_unit": "temporary_track_id",
                "score": float(path.score),
                "confidence": float(path.confidence),
                "evidence_until_s": float(path.evidence_until_s),
                "polyline_points": len(path.polyline),
            })
        self._snapshot = CommonPathSnapshot(
            timestamp, paths, self._snapshot.version + 1,
            max((path.evidence_until_s for path in paths), default=timestamp),
        )
