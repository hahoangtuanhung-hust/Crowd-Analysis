from __future__ import annotations

import math
from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any

import numpy as np
from numpy.typing import NDArray

from backend.app.core.config import TrackerConfig
from backend.app.schemas import Detection, TrackedObject


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
        # Last observed bottom-center and velocity per ID.  ByteTrack's
        # Kalman state is excellent for ordinary motion, but retaining this
        # short history prevents an ID from jumping to the opposite person
        # when two pedestrians overlap and the detector briefly drops one.
        self._motion_by_track: dict[int, tuple[np.ndarray, np.ndarray, int]] = {}
        self._coast_misses: dict[int, int] = {}
        self._last_source_frame_id: int | None = None
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
    ) -> NDArray[np.float32]:
        from ultralytics.trackers.utils import matching

        cost = np.asarray(matching.iou_distance(tracks, detections), dtype=np.float32)
        if self._config.association_mode == "hybrid" and tracks and detections:
            track_points = np.asarray(
                [self._bottom_center(item.xyxy) for item in tracks],
                dtype=np.float32,
            )
            detection_points = np.asarray(
                [self._bottom_center(item.xyxy) for item in detections],
                dtype=np.float32,
            )
            distances = np.linalg.norm(
                track_points[:, None, :] - detection_points[None, :, :],
                axis=2,
            )
            maximum = max(
                1.0,
                self._frame_diagonal * self._config.max_center_distance_ratio,
            )
            center_cost = np.clip(distances / maximum, 0.0, 1.0).astype(np.float32)
            # When bounding boxes have zero overlap (cost >= 1.0), penalize center distance
            # so we do not greedily hijack an adjacent person's detection in crowded scenes.
            gated_center = np.where(
                cost < 1.0,
                np.minimum(cost, center_cost),
                np.clip(center_cost + 0.15, 0.0, 1.0),
            )
            cost = gated_center.astype(np.float32)

            motion_weight = self._config.motion_cost_weight
            if motion_weight > 0.0 and self._motion_by_track:
                current_frame = int(self._tracker.frame_id)
                detection_heights = np.asarray(
                    [float(item.xyxy[3] - item.xyxy[1]) for item in detections],
                    dtype=np.float32,
                )
                last_centers = np.zeros((len(tracks), 2), dtype=np.float32)
                predicted_points = np.zeros((len(tracks), 2), dtype=np.float32)
                velocities = np.zeros((len(tracks), 2), dtype=np.float32)
                elapsed_frames = np.ones(len(tracks), dtype=np.float32)
                track_heights = np.ones(len(tracks), dtype=np.float32)
                has_history = np.zeros(len(tracks), dtype=bool)
                for row, track in enumerate(tracks):
                    history = self._motion_by_track.get(int(track.track_id))
                    if history is None:
                        continue
                    last_center, velocity, last_frame = history
                    elapsed = max(1, current_frame - last_frame)
                    last_centers[row] = last_center
                    predicted_points[row] = last_center + velocity * float(elapsed)
                    velocities[row] = velocity
                    elapsed_frames[row] = float(elapsed)
                    track_heights[row] = max(1.0, float(track.xyxy[3] - track.xyxy[1]))
                    has_history[row] = True
                if np.any(has_history):
                    distance = np.linalg.norm(
                        predicted_points[:, None, :] - detection_points[None, :, :],
                        axis=2,
                    )
                    median_detection_height = max(1.0, float(np.median(detection_heights)))
                    velocity_speed = np.linalg.norm(velocities, axis=1)
                    scale = np.maximum.reduce(
                        [
                            np.full(len(tracks), 6.0, dtype=np.float32),
                            np.full(
                                len(tracks),
                                self._frame_diagonal * self._config.max_center_distance_ratio,
                                dtype=np.float32,
                            ),
                            1.5 * velocity_speed * elapsed_frames,
                            0.5
                            * np.maximum(track_heights, median_detection_height)
                            * elapsed_frames,
                        ]
                    )
                    motion_cost = np.clip(distance / scale[:, None], 0.0, 1.0)
                    displacement = detection_points[None, :, :] - last_centers[:, None, :]
                    direction = np.sum(displacement * velocities[:, None, :], axis=2)
                    reverse = direction < -0.25 * velocity_speed[:, None] * np.maximum(
                        np.linalg.norm(displacement, axis=2), 1.0
                    )
                    motion_cost = np.where(
                        reverse & (velocity_speed[:, None] > 1.0),
                        np.minimum(1.0, motion_cost + self._config.motion_direction_penalty),
                        motion_cost,
                    )
                    cost[has_history] = (
                        (1.0 - motion_weight) * cost[has_history]
                        + motion_weight * motion_cost[has_history]
                    )

        if fuse_score:
            cost = np.asarray(matching.fuse_score(cost, detections), dtype=np.float32)
        return cost

    @staticmethod
    def _bottom_center(xyxy: NDArray) -> tuple[float, float]:
        return (float(xyxy[0]) + float(xyxy[2])) / 2.0, float(xyxy[3])

    def reset(self) -> None:
        self._tracker = self._create_tracker()
        self._motion_by_track.clear()
        self._coast_misses.clear()
        self._last_source_frame_id = None

    def update(
        self,
        detections: Sequence[Detection],
        frame: NDArray[np.uint8],
        *,
        frame_id: int | None = None,
    ) -> list[TrackedObject]:
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

        self._update_motion_history(output)
        for item in output:
            if item.observed:
                self._coast_misses.pop(item.track_id, None)
        return sorted(output, key=lambda item: item.track_id)

    def coast(
        self,
        frame: NDArray[np.uint8],
        *,
        frame_id: int | None = None,
    ) -> list[TrackedObject]:
        """Advance predictions without treating an unsearched frame as a miss.

        Calling ``BYTETracker.update([])`` marks active tracks lost.  That is
        correct after a detector searched their location and found nothing, but
        incorrect when the motion scheduler intentionally skipped inference.
        This method advances only the Kalman state and labels every output as
        ``NOT_SEARCHED_BY_POLICY``; it does not create Common Path evidence.
        """
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
        return sorted(output, key=lambda item: item.track_id)

    def _update_motion_history(self, output: list[TrackedObject]) -> None:
        current_frame = int(self._tracker.frame_id)
        alpha = self._config.motion_history_alpha
        for item in output:
            if not item.observed:
                continue
            center = np.asarray(item.bottom_center, dtype=np.float32)
            previous = self._motion_by_track.get(item.track_id)
            if previous is None:
                velocity = np.zeros(2, dtype=np.float32)
            else:
                previous_center, previous_velocity, previous_frame = previous
                elapsed = max(1, current_frame - previous_frame)
                measured = (center - previous_center) / float(elapsed)
                velocity = alpha * previous_velocity + (1.0 - alpha) * measured
            self._motion_by_track[item.track_id] = (center, velocity, current_frame)

        active_ids = {
            int(track.track_id)
            for track in list(self._tracker.tracked_stracks) + list(self._tracker.lost_stracks)
        }
        self._motion_by_track = {
            track_id: history
            for track_id, history in self._motion_by_track.items()
            if track_id in active_ids
        }

    @staticmethod
    def _to_tracked_object(
        row: Sequence[float],
        width: int,
        height: int,
        *,
        observed: bool = True,
        observation_coverage: str | None = None,
    ) -> TrackedObject:
        return TrackedObject(
            x1=float(np.clip(row[0], 0, width - 1)),
            y1=float(np.clip(row[1], 0, height - 1)),
            x2=float(np.clip(row[2], 0, width - 1)),
            y2=float(np.clip(row[3], 0, height - 1)),
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
