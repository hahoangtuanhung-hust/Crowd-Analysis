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
            cost = np.minimum(cost, center_cost)

        if fuse_score:
            cost = np.asarray(matching.fuse_score(cost, detections), dtype=np.float32)
        return cost

    @staticmethod
    def _bottom_center(xyxy: NDArray) -> tuple[float, float]:
        return (float(xyxy[0]) + float(xyxy[2])) / 2.0, float(xyxy[3])

    def reset(self) -> None:
        self._tracker = self._create_tracker()

    def update(
        self,
        detections: Sequence[Detection],
        frame: NDArray[np.uint8],
    ) -> list[TrackedObject]:
        height, width = frame.shape[:2]
        self._frame_diagonal = float(np.hypot(width, height))
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

        output = [self._to_tracked_object(row, width, height) for row in tracked]
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
                    output.append(self._to_tracked_object(item.result, width, height))

        return sorted(output, key=lambda item: item.track_id)

    @staticmethod
    def _to_tracked_object(row: Sequence[float], width: int, height: int) -> TrackedObject:
        return TrackedObject(
            x1=float(np.clip(row[0], 0, width - 1)),
            y1=float(np.clip(row[1], 0, height - 1)),
            x2=float(np.clip(row[2], 0, width - 1)),
            y2=float(np.clip(row[3], 0, height - 1)),
            track_id=int(row[4]),
            confidence=float(row[5]),
            class_id=int(row[6]),
        )
