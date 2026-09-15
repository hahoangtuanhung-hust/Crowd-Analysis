from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from backend.app.analytics.spatial import SpatialTransformer
from backend.app.core.config import ZoneConfig
from backend.app.schemas import FrameResult, HeatmapSnapshot, Trajectory


@dataclass(frozen=True, slots=True)
class OverlayOptions:
    detection: bool = False
    tracking: bool = True
    trajectory: bool = True
    heatmap: bool = False
    zones: bool = True


class FrameRenderer:
    def render(
        self,
        result: FrameResult,
        trajectories: tuple[Trajectory, ...],
        heatmap: HeatmapSnapshot,
        zones: tuple[ZoneConfig, ...],
        transformer: SpatialTransformer,
        options: OverlayOptions,
        *,
        processing_fps: float,
    ) -> NDArray[np.uint8]:
        frame = result.packet.image.copy()
        if options.heatmap:
            frame = self._draw_heatmap(frame, heatmap.occupancy, transformer)
        if options.zones:
            self._draw_zones(frame, zones, transformer)
        if options.detection:
            for detection in result.detections:
                point = (
                    round((detection.x1 + detection.x2) / 2.0),
                    round(detection.y2),
                )
                cv2.circle(frame, point, 3, (32, 190, 245), 1, cv2.LINE_AA)

        trajectory_by_id = {item.track_id: item for item in trajectories}
        if options.trajectory:
            for trajectory in trajectories:
                if len(trajectory.points) < 2:
                    continue
                points = np.asarray(
                    [(round(point.x), round(point.y)) for point in trajectory.points],
                    dtype=np.int32,
                ).reshape((-1, 1, 2))
                cv2.polylines(frame, [points], False, (52, 211, 255), 2, cv2.LINE_AA)
                if len(points) >= 3:
                    cv2.arrowedLine(
                        frame,
                        tuple(points[-3, 0]),
                        tuple(points[-1, 0]),
                        (52, 211, 255),
                        2,
                        cv2.LINE_AA,
                        tipLength=0.35,
                    )
        if options.tracking:
            for track in result.tracks:
                trajectory = trajectory_by_id.get(track.track_id)
                if trajectory is not None and trajectory.points:
                    last = trajectory.points[-1]
                    point = (round(last.x), round(last.y))
                else:
                    x, y = track.bottom_center
                    point = (round(x), round(y))
                cv2.circle(frame, point, 5, (64, 202, 142), -1, cv2.LINE_AA)
                cv2.circle(frame, point, 7, (245, 247, 250), 1, cv2.LINE_AA)
                cv2.putText(
                    frame,
                    f"ID {track.track_id}",
                    (point[0] + 8, max(18, point[1] - 7)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

        latency_ms = result.inference_ms + result.tracking_ms
        label = (
            f"People {len(result.tracks)}   FPS {processing_fps:.1f}   "
            f"Latency {latency_ms:.0f} ms"
        )
        cv2.rectangle(frame, (12, 12), (430, 46), (15, 20, 25), -1)
        cv2.putText(
            frame, label, (22, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (245, 247, 250), 2, cv2.LINE_AA
        )
        return frame

    def _draw_zones(
        self,
        frame: NDArray[np.uint8],
        zones: tuple[ZoneConfig, ...],
        transformer: SpatialTransformer,
    ) -> None:
        for index, zone in enumerate(zones):
            camera_points = [transformer.inverse_transform(x, y) for x, y in zone.points]
            points = np.asarray(camera_points, dtype=np.int32).reshape((-1, 1, 2))
            color = ((71 + index * 47) % 220, (156 + index * 31) % 220, (235 + index * 19) % 255)
            cv2.polylines(frame, [points], True, color, 2, cv2.LINE_AA)
            anchor = tuple(points[0, 0])
            cv2.putText(
                frame, zone.name, anchor, cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA
            )

    @staticmethod
    def _draw_heatmap(
        frame: NDArray[np.uint8],
        values: NDArray[np.float32],
        transformer: SpatialTransformer,
    ) -> NDArray[np.uint8]:
        maximum = float(values.max(initial=0.0))
        if maximum <= 0:
            return frame
        normalized = np.clip(values / maximum * 255.0, 0, 255).astype(np.uint8)
        colored = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
        if transformer.mode == "pixel":
            colored = cv2.resize(
                colored, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR
            )
            mask = cv2.resize(
                normalized, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR
            )
        else:
            scale = np.asarray(
                [
                    [values.shape[1] / transformer.width, 0.0, 0.0],
                    [0.0, values.shape[0] / transformer.height, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )
            camera_to_grid = scale @ transformer.matrix
            grid_to_camera = np.linalg.inv(camera_to_grid)
            size = (frame.shape[1], frame.shape[0])
            colored = cv2.warpPerspective(colored, grid_to_camera, size)
            mask = cv2.warpPerspective(normalized, grid_to_camera, size)
        blended = cv2.addWeighted(frame, 0.45, colored, 0.55, 0.0)
        cv2.copyTo(blended, mask, frame)
        return frame
