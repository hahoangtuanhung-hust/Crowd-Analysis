from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Protocol

import cv2
import numpy as np
from numpy.typing import NDArray

from backend.app.analytics.spatial import SpatialTransformer
from backend.app.core.config import VisualizationConfig, ZoneConfig
from backend.app.schemas import (
    CommonPath,
    CommonPathSnapshot,
    DirectedFlowSnapshot,
    FrameResult,
    HeatmapSnapshot,
    Trajectory,
)


class PointLike(Protocol):
    track_id: int
    x: float
    y: float


@dataclass(frozen=True, slots=True)
class _CurrentPoint:
    track_id: int
    x: float
    y: float


@dataclass(frozen=True, slots=True)
class OverlayOptions:
    detection: bool = False
    tracking: bool = True
    trajectory: bool = True
    heatmap: bool = False
    zones: bool = False
    points: bool = True
    track_ids: bool = False
    trajectory_tails: bool = False
    grid: bool = False
    edge_flows: bool = False
    candidate_paths: bool = False
    active_paths: bool = True
    direction_arrows: bool = True
    debug_metrics: bool = True

    @classmethod
    def from_visualization(cls, config: VisualizationConfig) -> OverlayOptions:
        return cls(
            detection=config.show_bounding_boxes,
            tracking=True,
            trajectory=config.show_individual_trajectories,
            heatmap=config.show_heatmap,
            zones=config.show_zones,
            points=config.show_tracking_points,
            track_ids=config.show_track_ids,
            trajectory_tails=config.show_individual_trajectories,
            grid=config.show_grid_debug,
            edge_flows=config.show_edge_flow_debug,
            candidate_paths=config.show_candidate_path,
            active_paths=config.show_common_path,
            direction_arrows=config.show_direction_arrows,
            debug_metrics=config.show_metrics,
        )


@dataclass(slots=True)
class _DisplayPathState:
    path: CommonPath
    signature: tuple[str, tuple[tuple[float, float], ...]]
    source: NDArray[np.float64]
    target: NDArray[np.float64]
    started_at: float


class FrameRenderer:
    def __init__(self, visualization: VisualizationConfig | None = None) -> None:
        self.visualization = visualization or VisualizationConfig()
        self._display_paths: dict[tuple[str, str], _DisplayPathState] = {}

    def render(
        self,
        result: FrameResult,
        current_points: Sequence[PointLike],
        heatmap: HeatmapSnapshot,
        zones: tuple[ZoneConfig, ...],
        transformer: SpatialTransformer,
        options: OverlayOptions,
        *,
        processing_fps: float,
        common_paths: CommonPathSnapshot | None = None,
        directed_flows: DirectedFlowSnapshot | None = None,
        debug_trajectories: tuple[Trajectory, ...] = (),
    ) -> NDArray[np.uint8]:
        frame = result.packet.image.copy()
        if options.heatmap:
            frame = self._draw_heatmap(frame, heatmap.occupancy, transformer)

        # Overlay points must belong to this exact frame; smoothed histories are analytics state.
        points = [_CurrentPoint(track.track_id, *track.bottom_center)
                  for track in result.tracks if track.observed]

        return self.render_point_only_frame(
            frame,
            points,
            common_paths,
            zones,
            transformer,
            options,
            people_count=len(result.tracks),
            processing_fps=processing_fps,
            latency_ms=result.e2e_latency_ms,
            timestamp=result.packet.source_timestamp,
            directed_flows=directed_flows,
            debug_trajectories=debug_trajectories,
        )

    def render_point_only_frame(
        self,
        frame: NDArray[np.uint8],
        current_points: Sequence[PointLike],
        common_paths: CommonPathSnapshot | None,
        zones: tuple[ZoneConfig, ...],
        transformer: SpatialTransformer,
        options: OverlayOptions,
        *,
        people_count: int,
        processing_fps: float,
        latency_ms: float,
        timestamp: float,
        directed_flows: DirectedFlowSnapshot | None = None,
        debug_trajectories: tuple[Trajectory, ...] = (),
    ) -> NDArray[np.uint8]:
        rendered = frame.copy()
        if options.grid and directed_flows is not None:
            self._draw_grid(rendered, directed_flows, transformer)
        if options.edge_flows and directed_flows is not None:
            self._draw_edge_flows(rendered, directed_flows, transformer)
        if options.trajectory and options.trajectory_tails:
            self._draw_debug_trajectories(rendered, debug_trajectories)

        visible_paths: tuple[CommonPath, ...] = ()
        if common_paths is not None:
            visible_paths = self.display_common_paths(common_paths, timestamp)
        if options.active_paths:
            self._draw_confirmed_paths(rendered, visible_paths, options.direction_arrows)
            if common_paths is not None and not visible_paths:
                self._draw_collecting_state(rendered)
        if options.candidate_paths and common_paths is not None:
            candidates = tuple(path for path in common_paths.paths if path.state == "candidate")
            self._draw_candidate_paths(rendered, candidates, options.direction_arrows)

        if options.tracking and options.points:
            self._draw_points(rendered, current_points, show_ids=options.track_ids)
        if options.zones:
            self._draw_zones(rendered, zones, transformer)
        if options.debug_metrics:
            self._draw_metrics(
                rendered,
                people_count=people_count,
                processing_fps=processing_fps,
                latency_ms=latency_ms,
            )
        return rendered

    def display_common_paths(
        self, snapshot: CommonPathSnapshot, timestamp: float
    ) -> tuple[CommonPath, ...]:
        selected: dict[tuple[str, str], CommonPath] = {}
        for path in snapshot.paths:
            if path.state not in {"active", "cooling"}:
                continue
            key = (path.path_id, "tracklet") if path.color else (path.origin_zone, path.destination_zone)
            previous = selected.get(key)
            if previous is None or (previous.state == "cooling" and path.state == "active"):
                selected[key] = path

        visible_keys = set(selected)
        for key in tuple(self._display_paths):
            if key not in visible_keys:
                del self._display_paths[key]

        displayed = [
            self._display_path(key, path, timestamp)
            for key, path in sorted(selected.items())
        ]
        displayed.sort(key=lambda path: (-path.score, path.path_id))
        return tuple(displayed)

    def _display_path(
        self,
        key: tuple[str, str],
        path: CommonPath,
        timestamp: float,
    ) -> CommonPath:
        if path.color:
            return path
        signature = (path.path_id, path.polyline)
        target = self._resample_polyline(path.polyline)
        state = self._display_paths.get(key)
        if state is None:
            state = _DisplayPathState(path, signature, target, target, timestamp)
            self._display_paths[key] = state
        elif state.signature != signature:
            current = self._interpolated_polyline(state, timestamp)
            state = _DisplayPathState(path, signature, current, target, timestamp)
            self._display_paths[key] = state
        else:
            state.path = path

        polyline = self._interpolated_polyline(state, timestamp)
        return replace(
            state.path,
            polyline=tuple((float(x), float(y)) for x, y in polyline),
        )

    def _interpolated_polyline(
        self, state: _DisplayPathState, timestamp: float
    ) -> NDArray[np.float64]:
        duration = self.visualization.common_path_style.transition_milliseconds / 1000.0
        if duration <= 0:
            return state.target
        progress = min(1.0, max(0.0, (timestamp - state.started_at) / duration))
        eased = progress * progress * (3.0 - 2.0 * progress)
        return state.source * (1.0 - eased) + state.target * eased

    @staticmethod
    def _resample_polyline(
        points: Sequence[tuple[float, float]], count: int = 32
    ) -> NDArray[np.float64]:
        array = np.asarray(points, dtype=np.float64)
        if len(array) == 0:
            return np.zeros((count, 2), dtype=np.float64)
        if len(array) == 1:
            return np.repeat(array, count, axis=0)
        distances = np.linalg.norm(np.diff(array, axis=0), axis=1)
        cumulative = np.concatenate(([0.0], np.cumsum(distances)))
        if cumulative[-1] <= 1e-9:
            return np.repeat(array[:1], count, axis=0)
        samples = np.linspace(0.0, cumulative[-1], count)
        return np.column_stack(
            (
                np.interp(samples, cumulative, array[:, 0]),
                np.interp(samples, cumulative, array[:, 1]),
            )
        )

    def _draw_confirmed_paths(
        self,
        frame: NDArray[np.uint8],
        paths: Sequence[CommonPath],
        show_arrows: bool,
    ) -> None:
        style = self.visualization.common_path_style
        for path in paths:
            points = np.asarray(path.polyline, dtype=np.int32).reshape((-1, 1, 2))
            if len(points) < 2:
                continue
            center_width = round(
                style.min_width_pixels
                + path.score * (style.max_width_pixels - style.min_width_pixels)
            )
            center_width = max(style.min_width_pixels, min(style.max_width_pixels, center_width))
            if path.color:
                center_width = 4
            cooling = path.state == "cooling"
            center_color = (145, 153, 162) if cooling else (
                tuple(bytes.fromhex(path.color[1:])[::-1]) if path.color else style.centerline_color_bgr
            )
            corridor_color = (32, 32, 32) if path.color else ((92, 98, 104) if cooling else style.corridor_color_bgr)

            corridor = frame.copy()
            cv2.polylines(
                corridor,
                [points],
                False,
                corridor_color,
                center_width + (4 if path.color else 14),
                cv2.LINE_AA,
            )
            corridor_alpha = (0.75 if path.color else style.corridor_opacity) * (0.55 if cooling else 1.0)
            cv2.addWeighted(corridor, corridor_alpha, frame, 1.0 - corridor_alpha, 0.0, frame)

            centerline = frame.copy()
            cv2.polylines(
                centerline,
                [points],
                False,
                center_color,
                center_width,
                cv2.LINE_AA,
            )
            if show_arrows:
                # Tracklet paths are curved polylines, so direction must be
                # visible along the route rather than only at the last vertex.
                self._draw_spaced_arrows(
                    centerline,
                    np.asarray(path.polyline, dtype=np.float64),
                    center_color,
                    max(2, center_width // 2),
                    style.arrow_spacing_pixels,
                )
            line_alpha = (1.0 if path.color else style.centerline_opacity) * (0.65 if cooling else 1.0)
            cv2.addWeighted(centerline, line_alpha, frame, 1.0 - line_alpha, 0.0, frame)
            self._draw_path_label(frame, path, center_color)

    def _draw_candidate_paths(
        self,
        frame: NDArray[np.uint8],
        paths: Sequence[CommonPath],
        show_arrows: bool,
    ) -> None:
        color = (0, 196, 255)
        style = self.visualization.common_path_style
        for path in paths:
            points = np.asarray(path.polyline, dtype=np.int32).reshape((-1, 1, 2))
            if len(points) < 2:
                continue
            corridor = frame.copy()
            cv2.polylines(
                corridor,
                [points],
                False,
                color,
                style.min_width_pixels + 8,
                cv2.LINE_AA,
            )
            cv2.addWeighted(corridor, 0.22, frame, 0.78, 0.0, frame)
            cv2.polylines(frame, [points], False, color, max(2, style.min_width_pixels - 2), cv2.LINE_AA)
            if show_arrows:
                self._draw_spaced_arrows(
                    frame,
                    np.asarray(path.polyline, dtype=np.float64),
                    color,
                    2,
                    self.visualization.common_path_style.arrow_spacing_pixels,
                )

    @staticmethod
    def _draw_spaced_arrows(
        frame: NDArray[np.uint8],
        points: NDArray[np.float64],
        color: tuple[int, int, int],
        thickness: int,
        spacing: int,
    ) -> None:
        if len(points) < 2:
            return
        segments = np.diff(points, axis=0)
        lengths = np.linalg.norm(segments, axis=1)
        total = float(lengths.sum())
        if total <= 1e-9:
            return
        # Keep a visible arrowhead near the destination as well as the repeated
        # directional cues along a long path. The length scales with spacing so
        # the arrow remains legible at both 720p and larger source resolutions.
        arrow_length = max(26.0, min(52.0, spacing * 0.45))
        distances = list(np.arange(spacing * 0.65, total, spacing))
        if not distances:
            distances = [max(total * 0.65, total - arrow_length * 0.5)]
        end_distance = total - arrow_length * 0.5
        if end_distance > 0 and all(abs(end_distance - item) > arrow_length * 0.35 for item in distances):
            distances.append(end_distance)
        cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
        for distance in distances:
            segment_index = min(len(lengths) - 1, int(np.searchsorted(cumulative, distance) - 1))
            segment_length = lengths[segment_index]
            if segment_length <= 1e-9:
                continue
            offset = (distance - cumulative[segment_index]) / segment_length
            center = points[segment_index] + segments[segment_index] * offset
            direction = segments[segment_index] / segment_length
            start = center - direction * (arrow_length * 0.5)
            end = center + direction * (arrow_length * 0.5)
            cv2.arrowedLine(
                frame,
                tuple(np.rint(start).astype(int)),
                tuple(np.rint(end).astype(int)),
                color,
                max(2, thickness),
                cv2.LINE_AA,
                tipLength=0.38,
            )

    @staticmethod
    def _draw_path_label(
        frame: NDArray[np.uint8], path: CommonPath, color: tuple[int, int, int]
    ) -> None:
        anchor_x, anchor_y = map(round, path.polyline[0])
        if path.color:
            label = f"{path.path_id} | {path.support_tracks} IDs"
        elif path.origin_zone in {"dominant_direction", "dominant_live_flow"}:
            direction = path.destination_zone.replace("_", " ").title()
            label = f"Dominant {direction} | {path.unique_tracks_short} moving"
        else:
            origin = path.origin_zone.replace("_", " ").title()
            destination = path.destination_zone.replace("_", " ").title()
            label = f"{origin} > {destination} | {path.unique_tracks_long} IDs"
        (text_width, text_height), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
        )
        position = (
            min(frame.shape[1] - text_width - 8, max(8, anchor_x + 10)),
            min(frame.shape[0] - 8, max(text_height + 8, anchor_y - 10)),
        )
        cv2.putText(
            frame,
            label,
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (20, 24, 28),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            label,
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA,
        )

    def _draw_points(
        self,
        frame: NDArray[np.uint8],
        points: Sequence[PointLike],
        *,
        show_ids: bool,
    ) -> None:
        style = self.visualization.point_style
        outline_extra = 1 if style.radius_pixels <= 3 else 2
        for item in points:
            point = (round(item.x), round(item.y))
            cv2.circle(
                frame,
                point,
                style.radius_pixels + outline_extra,
                style.outline_color_bgr,
                -1,
                cv2.LINE_AA,
            )
            cv2.circle(
                frame,
                point,
                style.radius_pixels,
                style.color_bgr,
                -1,
                cv2.LINE_AA,
            )
            if show_ids:
                cv2.putText(
                    frame,
                    f"ID {item.track_id}",
                    (point[0] + 8, max(18, point[1] - 7)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

    @staticmethod
    def _draw_debug_trajectories(
        frame: NDArray[np.uint8], trajectories: Sequence[Trajectory]
    ) -> None:
        for trajectory in trajectories:
            if len(trajectory.points) < 2:
                continue
            points = np.asarray(
                [(round(point.x), round(point.y)) for point in trajectory.points],
                dtype=np.int32,
            ).reshape((-1, 1, 2))
            cv2.polylines(frame, [points], False, (52, 211, 255), 1, cv2.LINE_AA)

    @staticmethod
    def _draw_collecting_state(frame: NDArray[np.uint8]) -> None:
        label = "Collecting movement flow..."
        height, _ = frame.shape[:2]
        baseline = max(28, height - 22)
        cv2.rectangle(frame, (12, baseline - 24), (274, baseline + 8), (12, 16, 20), -1)
        cv2.putText(
            frame,
            label,
            (22, baseline),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (210, 218, 225),
            1,
            cv2.LINE_AA,
        )

    @staticmethod
    def _draw_metrics(
        frame: NDArray[np.uint8],
        *,
        people_count: int,
        processing_fps: float,
        latency_ms: float,
    ) -> None:
        label = f"People {people_count}   FPS {processing_fps:.1f}   Latency {latency_ms:.0f} ms"
        cv2.rectangle(frame, (12, 12), (430, 46), (15, 20, 25), -1)
        cv2.putText(
            frame,
            label,
            (22, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (245, 247, 250),
            2,
            cv2.LINE_AA,
        )

    @staticmethod
    def _draw_grid(
        frame: NDArray[np.uint8],
        snapshot: DirectedFlowSnapshot,
        transformer: SpatialTransformer,
    ) -> None:
        color = (78, 86, 96)
        for column in range(1, snapshot.grid_columns):
            x = transformer.width * column / snapshot.grid_columns
            start = transformer.inverse_transform(x, 0.0)
            end = transformer.inverse_transform(x, transformer.height)
            cv2.line(frame, tuple(map(round, start)), tuple(map(round, end)), color, 1)
        for row in range(1, snapshot.grid_rows):
            y = transformer.height * row / snapshot.grid_rows
            start = transformer.inverse_transform(0.0, y)
            end = transformer.inverse_transform(transformer.width, y)
            cv2.line(frame, tuple(map(round, start)), tuple(map(round, end)), color, 1)

    @staticmethod
    def _draw_edge_flows(
        frame: NDArray[np.uint8],
        snapshot: DirectedFlowSnapshot,
        transformer: SpatialTransformer,
    ) -> None:
        cell_width = transformer.width / snapshot.grid_columns
        cell_height = transformer.height / snapshot.grid_rows
        for edge in snapshot.edges[:200]:
            source_row, source_column = edge.from_cell
            target_row, target_column = edge.to_cell
            source = transformer.inverse_transform(
                (source_column + 0.5) * cell_width,
                (source_row + 0.5) * cell_height,
            )
            target = transformer.inverse_transform(
                (target_column + 0.5) * cell_width,
                (target_row + 0.5) * cell_height,
            )
            thickness = max(1, min(4, round(1 + edge.score * 3)))
            cv2.arrowedLine(
                frame,
                tuple(map(round, source)),
                tuple(map(round, target)),
                (96, 180, 232),
                thickness,
                cv2.LINE_AA,
                tipLength=0.3,
            )

    @staticmethod
    def _draw_zones(
        frame: NDArray[np.uint8],
        zones: tuple[ZoneConfig, ...],
        transformer: SpatialTransformer,
    ) -> None:
        for index, zone in enumerate(zones):
            camera_points = [transformer.inverse_transform(x, y) for x, y in zone.points]
            points = np.asarray(camera_points, dtype=np.int32).reshape((-1, 1, 2))
            color = (
                (71 + index * 47) % 220,
                (156 + index * 31) % 220,
                (235 + index * 19) % 255,
            )
            cv2.polylines(frame, [points], True, color, 2, cv2.LINE_AA)
            anchor = tuple(points[0, 0])
            cv2.putText(
                frame,
                zone.name,
                anchor,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
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
