from __future__ import annotations

import math
import logging
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Any

import numpy as np
from numpy.typing import NDArray

from backend.app.core.config import TrackerConfig
from backend.app.schemas import Detection, TrackedObject

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class _MotionState:
    """Observed bottom-center motion in source-frame units."""

    center: NDArray[np.float32]
    velocity: NDArray[np.float32]
    frame_id: int
    quality: float


@dataclass(slots=True)
class _ObservedPoint:
    frame_id: int
    timestamp_s: float
    center: NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class _DirectionEstimate:
    vector: tuple[float, float] | None
    state: str
    quality: float
    observed_span_s: float


class _DetectionResults:
    """Minimal Results-like object expected by Ultralytics tracker classes."""

    def __init__(self, xyxy: NDArray, confidence: NDArray, class_ids: NDArray) -> None:
        self.xyxy = np.asarray(xyxy, dtype=np.float32).reshape(-1, 4)
        self.conf = np.asarray(confidence, dtype=np.float32).reshape(-1)
        self.cls = np.asarray(class_ids, dtype=np.float32).reshape(-1)

    @property
    def xywh(self) -> NDArray:
        result = self.xyxy.copy()
        result[:, 0] = (self.xyxy[:, 0] + self.xyxy[:, 2]) / 2.0
        result[:, 1] = (self.xyxy[:, 1] + self.xyxy[:, 3]) / 2.0
        result[:, 2] = self.xyxy[:, 2] - self.xyxy[:, 0]
        result[:, 3] = self.xyxy[:, 3] - self.xyxy[:, 1]
        return result

    def __getitem__(self, index: object) -> _DetectionResults:
        return _DetectionResults(self.xyxy[index], self.conf[index], self.cls[index])

    def __len__(self) -> int:
        return len(self.conf)


class ByteTrackTracker:
    def __init__(self, config: TrackerConfig) -> None:
        self._config = config
        self._frame_diagonal = 1.0
        self._motion_by_track: dict[int, _MotionState] = {}
        self._observation_history: dict[int, list[_ObservedPoint]] = {}
        self._direction_by_track: dict[int, _DirectionEstimate] = {}
        self._coast_misses: dict[int, int] = {}
        self._last_source_frame_id: int | None = None
        self._current_source_timestamp: float | None = None
        self._association_diagnostics: deque[dict[str, Any]] = deque(
            maxlen=config.association_debug_max_events
        )
        self._tracker = self._create_tracker()

    def _create_tracker(self):
        from ultralytics.trackers.basetrack import TrackState
        from ultralytics.trackers.byte_tracker import BYTETracker
        from ultralytics.trackers.utils import matching

        owner = self

        class CrowdBYTETracker(BYTETracker):
            """ByteTrack with point-distance association for distant pedestrians."""

            def get_dists(self, tracks: list[Any], detections: list[Any]) -> NDArray:
                return owner._association_cost(
                    tracks,
                    detections,
                    fuse_score=self.args.fuse_score,
                    stage="high",
                )

            def _second_association(
                self,
                strack_pool: list[Any],
                u_track: list[int],
                detections_second: list[Any],
                activated: list[Any],
                refind: list[Any],
                lost: list[Any],
            ) -> None:
                remaining = [
                    strack_pool[index]
                    for index in u_track
                    if strack_pool[index].state == TrackState.Tracked
                ]
                if remaining and detections_second:
                    dists = owner._association_cost(
                        remaining,
                        detections_second,
                        fuse_score=False,
                        stage="low",
                    )
                    matches, unmatched, _ = matching.linear_assignment(
                        dists,
                        thresh=owner._config.second_match_thresh,
                    )
                    self._apply_matches(
                        matches,
                        remaining,
                        detections_second,
                        activated,
                        refind,
                    )
                else:
                    unmatched = list(range(len(remaining)))

                for index in unmatched:
                    track = remaining[index]
                    if track.state != TrackState.Lost:
                        track.mark_lost()
                        lost.append(track)

        return CrowdBYTETracker(SimpleNamespace(**self._config.model_dump()))

    def _association_cost(
        self,
        tracks: list[Any],
        detections: list[Any],
        *,
        fuse_score: bool,
        stage: str,
    ) -> NDArray[np.float32]:
        from ultralytics.trackers.utils import matching

        hybrid = self._config.association_mode == "hybrid" and tracks and detections
        if hybrid:
            # Share one Kalman-box snapshot between IoU and motion geometry.
            track_boxes = [item.xyxy for item in tracks]
            detection_boxes = [item.xyxy for item in detections]
            iou_cost = np.asarray(matching.iou_distance(track_boxes, detection_boxes), dtype=np.float32)
        else:
            iou_cost = np.asarray(matching.iou_distance(tracks, detections), dtype=np.float32)
        cost = iou_cost.copy()
        if hybrid:
            # Ultralytics computes xyxy from Kalman state on every property
            # access. Snapshot each box once for all project-specific geometry.
            track_points = np.asarray(
                [self._bottom_center(box) for box in track_boxes],
                dtype=np.float32,
            )
            detection_points = np.asarray(
                [self._bottom_center(box) for box in detection_boxes],
                dtype=np.float32,
            )
            distances = np.linalg.norm(
                track_points[:, None, :] - detection_points[None, :, :],
                axis=2,
            )
            track_heights_arr = np.asarray(
                [max(1.0, float(box[3] - box[1])) for box in track_boxes],
                dtype=np.float32,
            )
            det_heights_arr = np.asarray(
                [max(1.0, float(box[3] - box[1])) for box in detection_boxes],
                dtype=np.float32,
            )
            current_frame = int(self._tracker.frame_id)
            elapsed_frames = np.ones(len(tracks), dtype=np.float32)
            velocities = np.zeros((len(tracks), 2), dtype=np.float32)
            last_centers = np.zeros((len(tracks), 2), dtype=np.float32)
            predicted_points = track_points.copy()
            motion_quality = np.zeros(len(tracks), dtype=np.float32)
            uncertainty = np.zeros(len(tracks), dtype=np.float32)
            has_history = np.zeros(len(tracks), dtype=bool)
            for row, track in enumerate(tracks):
                history = self._motion_by_track.get(int(track.track_id))
                uncertainty[row] = self._track_uncertainty_pixels(track)
                if history is None:
                    continue
                elapsed = max(1, current_frame - history.frame_id)
                elapsed_frames[row] = float(elapsed)
                velocities[row] = history.velocity
                last_centers[row] = history.center
                predicted_points[row] = history.center + history.velocity * float(elapsed)
                motion_quality[row] = history.quality
                has_history[row] = True

            if self._config.association_size_adaptive:
                geometry_gate = np.maximum(
                    self._config.association_distance_floor_pixels,
                    self._config.association_distance_height_ratio
                    * np.maximum(track_heights_arr[:, None], det_heights_arr[None, :]),
                )
                gap_multiplier = 1.0 + self._config.association_time_gap_growth * np.log1p(
                    np.maximum(0.0, elapsed_frames - 1.0)
                )
                speed = np.linalg.norm(velocities, axis=1)
                motion_allowance = speed * np.maximum(0.0, elapsed_frames - 1.0)
                geometry_gate = np.maximum(
                    geometry_gate * gap_multiplier[:, None],
                    geometry_gate + uncertainty[:, None] + motion_allowance[:, None],
                )
                geometry_gate = np.minimum(
                    geometry_gate,
                    self._config.association_max_distance_pixels,
                ).astype(np.float32)
            else:
                geometry_gate = np.full(
                    (len(tracks), len(detections)),
                    max(1.0, self._frame_diagonal * self._config.max_center_distance_ratio),
                    dtype=np.float32,
                )

            center_cost = np.clip(distances / geometry_gate, 0.0, 1.0).astype(np.float32)
            # When boxes do not overlap, retain a small penalty so an adjacent
            # pedestrian cannot win merely because their bottom-center happens
            # to be close.  The optional hard gate is applied after all soft
            # motion costs so it cannot be bypassed by a low score.
            cost = np.where(
                cost < 1.0,
                np.minimum(cost, center_cost),
                np.clip(center_cost + 0.15, 0.0, 1.0),
            ).astype(np.float32)

            motion_distance = np.zeros_like(distances)
            motion_cost = np.full_like(distances, np.nan)
            motion_weight = self._config.motion_cost_weight
            if motion_weight > 0.0 and np.any(has_history):
                motion_distance = np.linalg.norm(
                    predicted_points[:, None, :] - detection_points[None, :, :],
                    axis=2,
                )
                speed = np.linalg.norm(velocities, axis=1)
                if self._config.association_size_adaptive:
                    motion_gate = np.maximum(
                        geometry_gate,
                        speed[:, None] * elapsed_frames[:, None]
                        + self._config.association_distance_floor_pixels,
                    )
                    motion_gate = np.minimum(
                        motion_gate,
                        self._config.association_max_distance_pixels,
                    )
                else:
                    median_detection_height = max(
                        1.0, float(np.median(det_heights_arr))
                    )
                    motion_scale = np.maximum.reduce(
                        [
                            np.full(len(tracks), 6.0, dtype=np.float32),
                            np.full(
                                len(tracks),
                                self._frame_diagonal
                                * self._config.max_center_distance_ratio,
                                dtype=np.float32,
                            ),
                            1.5 * speed * elapsed_frames,
                            0.5
                            * np.maximum(track_heights_arr, median_detection_height)
                            * elapsed_frames,
                        ]
                    )
                    motion_gate = np.broadcast_to(
                        motion_scale[:, None], distances.shape
                    )
                motion_cost = np.clip(motion_distance / motion_gate, 0.0, 1.0)
                displacement = detection_points[None, :, :] - last_centers[:, None, :]
                directional_progress = np.sum(displacement * velocities[:, None, :], axis=2)
                reverse = directional_progress < -0.25 * speed[:, None] * np.maximum(
                    np.linalg.norm(displacement, axis=2), 1.0
                )
                # Direction is intentionally only a soft tie-breaker after
                # multiple recent observations.  New, stationary and long-gap
                # tracks may turn, so their geometry remains decisive.
                reliable_motion = (
                    has_history
                    & (motion_quality >= 0.5)
                    & (speed >= self._config.stationary_speed_threshold)
                    & (elapsed_frames <= max(2.0, float(self._config.track_buffer)))
                )
                motion_cost = np.where(
                    reverse & reliable_motion[:, None],
                    np.minimum(1.0, motion_cost + self._config.motion_direction_penalty),
                    motion_cost,
                )
                cost[has_history] = (
                    (1.0 - motion_weight) * cost[has_history]
                    + motion_weight * motion_cost[has_history]
                )

            accepted_by_gate = distances <= geometry_gate
            if self._config.association_hard_gate:
                cost[~accepted_by_gate] = 1.0

        if fuse_score:
            cost = np.asarray(matching.fuse_score(cost, detections), dtype=np.float32)
        if self._config.association_mode == "hybrid" and tracks and detections:
            if self._config.association_hard_gate:
                cost[~accepted_by_gate] = 1.0
            self._record_association_diagnostics(
                tracks=tracks,
                detections=detections,
                stage=stage,
                iou_cost=iou_cost,
                final_cost=cost,
                distances=distances,
                geometry_gate=geometry_gate,
                motion_distance=motion_distance,
                motion_cost=motion_cost,
                elapsed_frames=elapsed_frames,
                accepted_by_gate=accepted_by_gate,
            )
        return cost

    def _track_uncertainty_pixels(self, track: Any) -> float:
        """Return a bounded positional standard deviation from ByteTrack state.

        This is not Mahalanobis matching: the project uses the tracker's
        published state covariance only to avoid shrinking a geometry gate
        below its own uncertainty.  Missing or malformed covariance means no
        uncertainty expansion.
        """
        if not self._config.association_size_adaptive:
            return 0.0
        covariance = getattr(track, "covariance", None)
        if covariance is None:
            return 0.0
        matrix = np.asarray(covariance, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[1] < 2:
            return 0.0
        variance = max(0.0, float(matrix[0, 0]) + float(matrix[1, 1]))
        return min(
            self._config.association_max_uncertainty_pixels,
            self._config.association_uncertainty_weight * math.sqrt(variance),
        )

    def _record_association_diagnostics(
        self,
        *,
        tracks: list[Any],
        detections: list[Any],
        stage: str,
        iou_cost: NDArray[np.float32],
        final_cost: NDArray[np.float32],
        distances: NDArray[np.float32],
        geometry_gate: NDArray[np.float32],
        motion_distance: NDArray[np.float32],
        motion_cost: NDArray[np.float32],
        elapsed_frames: NDArray[np.float32],
        accepted_by_gate: NDArray[np.bool_],
    ) -> None:
        if not self._config.association_debug:
            return
        frame_id = self._last_source_frame_id
        frame_filter = set(self._config.association_debug_frame_ids)
        if frame_filter and frame_id not in frame_filter:
            return
        track_filter = set(self._config.association_debug_track_ids)
        for row, track in enumerate(tracks):
            track_id = int(track.track_id)
            if track_filter and track_id not in track_filter:
                continue
            # With no explicit target, retain only the two strongest choices
            # per track.  This keeps the default diagnostic bounded even in a
            # dense frame; a focused investigation may supply track IDs.
            candidate_indices = np.argsort(final_cost[row])
            if not track_filter:
                candidate_indices = candidate_indices[:2]
            for column in candidate_indices:
                if len(self._association_diagnostics) >= self._config.association_debug_max_events:
                    return
                detection = detections[int(column)]
                track_box = [round(float(value), 3) for value in track.xyxy]
                detection_box = [round(float(value), 3) for value in detection.xyxy]
                event = {
                    "frame_id": frame_id,
                    "source_timestamp_s": self._current_source_timestamp,
                    "stage": stage,
                    "track_id": track_id,
                    "detection_index": int(column),
                    "detection_confidence": round(float(getattr(detection, "score", 0.0)), 6),
                    "track_bbox": track_box,
                    "detection_bbox": detection_box,
                    "track_height": round(track_box[3] - track_box[1], 3),
                    "detection_height": round(detection_box[3] - detection_box[1], 3),
                    "iou": round(1.0 - float(iou_cost[row, column]), 6),
                    "bottom_center_distance_px": round(float(distances[row, column]), 4),
                    "geometry_gate_px": round(float(geometry_gate[row, column]), 4),
                    "motion_distance_px": round(float(motion_distance[row, column]), 4),
                    "motion_score": (
                        None
                        if np.isnan(motion_cost[row, column])
                        else round(float(motion_cost[row, column]), 6)
                    ),
                    "time_gap_frames": round(float(elapsed_frames[row]), 3),
                    "cost": round(float(final_cost[row, column]), 6),
                    "reason": "CANDIDATE" if accepted_by_gate[row, column] else "GEOMETRY_GATE_REJECTED",
                }
                self._association_diagnostics.append(event)
                LOGGER.debug("tracking_association=%s", event)

    def drain_association_diagnostics(self) -> list[dict[str, Any]]:
        """Return and clear the bounded, opt-in association diagnostic buffer."""
        events = list(self._association_diagnostics)
        self._association_diagnostics.clear()
        return events

    @staticmethod
    def _bottom_center(xyxy: NDArray) -> tuple[float, float]:
        return (float(xyxy[0]) + float(xyxy[2])) / 2.0, float(xyxy[3])

    def reset(self) -> None:
        self._tracker = self._create_tracker()
        self._motion_by_track.clear()
        self._observation_history.clear()
        self._direction_by_track.clear()
        self._coast_misses.clear()
        self._last_source_frame_id = None
        self._current_source_timestamp = None
        self._association_diagnostics.clear()

    def update(
        self,
        detections: Sequence[Detection],
        frame: NDArray[np.uint8],
        *,
        frame_id: int | None = None,
        source_timestamp: float | None = None,
    ) -> list[TrackedObject]:
        self._set_source_timestamp(source_timestamp)
        height, width = frame.shape[:2]
        self._frame_diagonal = float(np.hypot(width, height))
        if frame_id is not None:
            source_frame_id = int(frame_id)
            if self._last_source_frame_id is None:
                self._tracker.frame_id = source_frame_id
            elif source_frame_id > self._last_source_frame_id + 1:
                # Realtime mode may drop capture frames while inference is
                # busy.  Advance active/lost Kalman states through the gap so
                # the next association uses the correct predicted position.
                gap = min(source_frame_id - self._last_source_frame_id - 1, 60)
                pools = list(self._tracker.tracked_stracks) + list(self._tracker.lost_stracks)
                for _ in range(gap):
                    self._tracker.multi_predict(pools)
                self._tracker.frame_id = source_frame_id
            else:
                self._tracker.frame_id = max(self._tracker.frame_id, source_frame_id)
            self._last_source_frame_id = source_frame_id
        results = _DetectionResults(
            xyxy=np.asarray([item.xyxy for item in detections], dtype=np.float32),
            confidence=np.asarray([item.confidence for item in detections], dtype=np.float32),
            class_ids=np.asarray([item.class_id for item in detections], dtype=np.float32),
        )
        tracked = self._tracker.update(results, img=frame)

        # Damp Kalman filter velocity for stationary pedestrians (e.g. buying tickets, queues)
        # to prevent bounding box drift when partially occluded or standing still.
        if getattr(self._config, "stationary_boost", True):
            for strack in getattr(self._tracker, "tracked_stracks", []):
                if hasattr(strack, "mean") and len(strack.mean) >= 6:
                    vx, vy = float(strack.mean[4]), float(strack.mean[5])
                    if math.hypot(vx, vy) < self._config.stationary_speed_threshold:
                        strack.mean[4] *= 0.1
                        strack.mean[5] *= 0.1

        output = [self._to_tracked_object(row, width, height, observed=True) for row in tracked]
        visible_ids = {item.track_id for item in output}

        if self._config.lost_track_grace_frames:
            for item in self._tracker.lost_stracks:
                missed_frames = self._tracker.frame_id - item.end_frame
                grace_frames = self._config.lost_track_grace_frames
                if (
                    self._config.stationary_boost
                    and self._config.stationary_lost_track_grace_frames is not None
                    and hasattr(item, "mean")
                    and len(item.mean) >= 6
                    and math.hypot(float(item.mean[4]), float(item.mean[5]))
                    < self._config.stationary_speed_threshold
                ):
                    grace_frames = self._config.stationary_lost_track_grace_frames
                if (
                    item.is_activated
                    and item.tracklet_len > 0
                    and item.track_id not in visible_ids
                    and 0 < missed_frames <= grace_frames
                ):
                    output.append(self._to_tracked_object(
                        item.result, width, height, observed=False
                    ))

        output = self._update_tracking_history(output)
        for item in output:
            if item.observed:
                self._coast_misses.pop(item.track_id, None)
        return sorted(output, key=lambda item: item.track_id)

    def coast(
        self,
        frame: NDArray[np.uint8],
        *,
        frame_id: int | None = None,
        source_timestamp: float | None = None,
    ) -> list[TrackedObject]:
        """Advance predictions without treating an unsearched frame as a miss.

        Calling ``BYTETracker.update([])`` marks active tracks lost.  That is
        correct after a detector searched their location and found nothing, but
        incorrect when the motion scheduler intentionally skipped inference.
        This method advances only the Kalman state and labels every output as
        ``NOT_SEARCHED_BY_POLICY``; it does not create Common Path evidence.
        """
        self._set_source_timestamp(source_timestamp)
        height, width = frame.shape[:2]
        self._frame_diagonal = float(np.hypot(width, height))
        if frame_id is None:
            steps = 1
            self._last_source_frame_id = (
                0 if self._last_source_frame_id is None else self._last_source_frame_id + 1
            )
        else:
            source_frame_id = int(frame_id)
            if self._last_source_frame_id is None:
                steps = 1
            else:
                steps = max(0, source_frame_id - self._last_source_frame_id)
            self._last_source_frame_id = source_frame_id

        pools = list(self._tracker.tracked_stracks) + list(self._tracker.lost_stracks)
        for _ in range(min(steps, 60)):
            self._tracker.multi_predict(pools)
        if steps:
            self._tracker.frame_id += min(steps, 60)

        output: list[TrackedObject] = []
        seen: set[int] = set()
        grace_default = max(1, int(self._config.lost_track_grace_frames))
        for track in pools:
            track_id = int(track.track_id)
            if track_id in seen or not track.is_activated or track.tracklet_len <= 0:
                continue
            misses = self._coast_misses.get(track_id, 0) + max(1, steps)
            self._coast_misses[track_id] = misses
            grace = grace_default
            if (
                self._config.stationary_lost_track_grace_frames is not None
                and hasattr(track, "mean")
                and len(track.mean) >= 6
                and math.hypot(float(track.mean[4]), float(track.mean[5]))
                < self._config.stationary_speed_threshold
            ):
                grace = max(grace, int(self._config.stationary_lost_track_grace_frames))
            if misses > grace:
                continue
            row = track.result
            item = self._to_tracked_object(
                row,
                width,
                height,
                observed=False,
                observation_coverage="NOT_SEARCHED_BY_POLICY",
            )
            output.append(item)
            seen.add(track_id)
        return sorted(self._update_tracking_history(output), key=lambda item: item.track_id)

    def _set_source_timestamp(self, source_timestamp: float | None) -> None:
        if source_timestamp is not None and not math.isfinite(float(source_timestamp)):
            raise ValueError("source_timestamp must be finite when provided")
        self._current_source_timestamp = (
            None if source_timestamp is None else float(source_timestamp)
        )

    def _observation_timestamp(self, current_frame: int) -> float:
        """Use source media time; direct users get an explicit configurable fallback."""
        if self._current_source_timestamp is not None:
            return self._current_source_timestamp
        return current_frame / self._config.direction_fallback_fps

    def _update_tracking_history(self, output: list[TrackedObject]) -> list[TrackedObject]:
        current_frame = int(self._tracker.frame_id)
        current_timestamp = self._observation_timestamp(current_frame)
        updated_output: list[TrackedObject] = []
        for item in output:
            if item.observed:
                center = np.asarray(item.bottom_center, dtype=np.float32)
                previous = self._motion_by_track.get(item.track_id)
                if previous is None:
                    velocity = np.zeros(2, dtype=np.float32)
                    quality = 0.0
                else:
                    elapsed = max(1, current_frame - previous.frame_id)
                    measured = (center - previous.center) / float(elapsed)
                    alpha = self._config.motion_history_alpha
                    velocity = alpha * previous.velocity + (1.0 - alpha) * measured
                    quality = min(
                        1.0,
                        previous.quality + (0.35 if elapsed == 1 else 0.15),
                    )
                self._motion_by_track[item.track_id] = _MotionState(
                    center=center,
                    velocity=velocity.astype(np.float32),
                    frame_id=current_frame,
                    quality=quality,
                )
                if self._config.direction_diagnostics_enabled:
                    item = self._update_direction_estimate(
                        item,
                        current_frame=current_frame,
                        timestamp_s=current_timestamp,
                        center=center,
                    )
            elif self._config.direction_diagnostics_enabled:
                # A prediction may display the latest observed estimate but it
                # never appends history or raises the estimate's confidence.
                estimate = self._direction_by_track.get(item.track_id)
                if estimate is not None:
                    item = replace(
                        item,
                        direction_vector=estimate.vector,
                        direction_state=estimate.state,
                        direction_quality=estimate.quality,
                        direction_observed_span_s=estimate.observed_span_s,
                    )
            updated_output.append(item)

        active_ids = {
            int(track.track_id)
            for track in list(self._tracker.tracked_stracks) + list(self._tracker.lost_stracks)
        }
        self._motion_by_track = {
            track_id: history
            for track_id, history in self._motion_by_track.items()
            if track_id in active_ids
        }
        self._observation_history = {
            track_id: history
            for track_id, history in self._observation_history.items()
            if track_id in active_ids
        }
        self._direction_by_track = {
            track_id: estimate
            for track_id, estimate in self._direction_by_track.items()
            if track_id in active_ids
        }
        return updated_output

    def _update_direction_estimate(
        self,
        item: TrackedObject,
        *,
        current_frame: int,
        timestamp_s: float,
        center: NDArray[np.float32],
    ) -> TrackedObject:
        history = self._observation_history.setdefault(item.track_id, [])
        if history and (
            timestamp_s <= history[-1].timestamp_s
            or timestamp_s - history[-1].timestamp_s
            > self._config.direction_max_observation_gap_seconds
        ):
            # A source-time rewind or an unobserved gap creates a fresh
            # direction window.  It is not a new tracker segment and cannot
            # alter Common Path, but prevents a prediction from validating a
            # later turn or reappearance.
            history.clear()
        history.append(_ObservedPoint(current_frame, timestamp_s, center))
        cutoff = timestamp_s - self._config.direction_history_seconds
        while len(history) > 1 and history[0].timestamp_s < cutoff:
            history.pop(0)

        unknown = _DirectionEstimate(None, "unknown", 0.0, 0.0)
        if len(history) < self._config.direction_min_observations:
            self._direction_by_track[item.track_id] = unknown
            return replace(
                item,
                direction_vector=None,
                direction_state="unknown",
                direction_quality=0.0,
                direction_observed_span_s=0.0,
            )

        oldest = history[0]
        displacement = center - oldest.center
        span_s = timestamp_s - oldest.timestamp_s
        displacement_length = float(np.linalg.norm(displacement))
        if span_s < self._config.direction_min_observed_span_seconds:
            self._direction_by_track[item.track_id] = unknown
            return replace(
                item,
                direction_vector=None,
                direction_state="unknown",
                direction_quality=0.0,
                direction_observed_span_s=max(0.0, span_s),
            )

        speed = displacement_length / span_s
        previous_state = self._direction_by_track.get(item.track_id, unknown).state
        if previous_state == "moving":
            state = (
                "stationary"
                if speed <= self._config.direction_stationary_enter_speed_pixels_s
                else "moving"
            )
        else:
            state = (
                "moving"
                if speed >= self._config.direction_stationary_exit_speed_pixels_s
                else "stationary"
            )
        sample_factor = min(
            1.0,
            len(history) / float(self._config.direction_min_observations + 2),
        )
        span_factor = min(1.0, span_s / self._config.direction_history_seconds)
        quality = sample_factor * span_factor
        vector = None
        if state == "moving" and displacement_length >= self._config.direction_min_displacement_pixels:
            vector = (
                float(displacement[0] / displacement_length),
                float(displacement[1] / displacement_length),
            )
        estimate = _DirectionEstimate(vector, state, quality, span_s)
        self._direction_by_track[item.track_id] = estimate
        return replace(
            item,
            direction_vector=estimate.vector,
            direction_state=estimate.state,
            direction_quality=estimate.quality,
            direction_observed_span_s=estimate.observed_span_s,
        )

    @staticmethod
    def _to_tracked_object(
        row: Sequence[float],
        width: int,
        height: int,
        *,
        observed: bool = True,
        observation_coverage: str | None = None,
    ) -> TrackedObject:
        max_x = float(width - 1)
        max_y = float(height - 1)
        return TrackedObject(
            x1=min(max(float(row[0]), 0.0), max_x),
            y1=min(max(float(row[1]), 0.0), max_y),
            x2=min(max(float(row[2]), 0.0), max_x),
            y2=min(max(float(row[3]), 0.0), max_y),
            track_id=int(row[4]),
            confidence=float(row[5]),
            class_id=int(row[6]),
            observed=observed,
            observation_coverage=(
                observation_coverage
                if observation_coverage is not None
                else "MEASURED" if observed else "SEARCHED_NOT_FOUND"
            ),
        )
