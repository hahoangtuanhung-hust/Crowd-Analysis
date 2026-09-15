from __future__ import annotations

import json
import math
import time
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import asdict
from typing import Any

import cv2
import numpy as np

from backend.app.analytics import FlowAnalyzer, HeatmapAnalyzer, SpatialTransformer, ZoneAnalyzer
from backend.app.core.config import AnalyticsConfig, ZoneConfig
from backend.app.datasets.grand_central import (
    CoordinateMode,
    GrandCentralDataset,
    NormalizedTrajectoryPoint,
    TrajectorySource,
)
from backend.app.schemas import TrackPoint, Trajectory


class DatasetAnalyticsRunner:
    """Runs the existing incremental analytics over normalized offline points."""

    def __init__(
        self,
        dataset: GrandCentralDataset,
        config: AnalyticsConfig,
        *,
        coordinate_mode: CoordinateMode,
    ) -> None:
        self.dataset = dataset
        self.coordinate_mode = coordinate_mode
        self.transformer, self.config = self._build_spatial(config)

    def run(
        self,
        points: Iterable[NormalizedTrajectoryPoint],
        *,
        source: TrajectorySource,
        duration_label: str,
        persist: bool = True,
    ) -> dict[str, Any]:
        point_list = sorted(points, key=lambda point: (point.timestamp, point.track_id))
        if not point_list:
            raise ValueError("Cannot analyze an empty trajectory set")
        source_values = {point.source for point in point_list}
        if source_values != {source}:
            raise ValueError(
                f"Analytics source isolation failed: expected {source}, found {source_values}"
            )

        track_lengths = Counter(point.track_id for point in point_list)
        eligible = {
            track_id
            for track_id, count in track_lengths.items()
            if count >= self.config.min_confirmed_points
        }
        grouped: dict[tuple[float, int], list[NormalizedTrajectoryPoint]] = defaultdict(list)
        frame_counts: Counter[int] = Counter()
        for point in point_list:
            frame_counts[point.frame_id] += 1
            if point.track_id in eligible:
                grouped[(point.timestamp, point.frame_id)].append(point)

        heatmaps = HeatmapAnalyzer(self.config, self.transformer, retain_entire=True)
        flows = FlowAnalyzer(self.config, self.transformer, retain_entire=True)
        zones = ZoneAnalyzer(self.config, self.transformer, retain_entire=True)
        started = time.perf_counter()
        timeline: list[dict[str, Any]] = []
        for (timestamp, frame_id), frame_points in sorted(grouped.items()):
            trajectories = tuple(
                Trajectory(
                    track_id=point.track_id,
                    points=(self._track_point(point),),
                    age=track_lengths[point.track_id],
                    confirmed=True,
                    last_seen_timestamp=timestamp,
                )
                for point in frame_points
            )
            heatmaps.process(trajectories)
            flows.process(trajectories)
            zones.process(trajectories)
            timeline.append(
                {
                    "timestamp": timestamp,
                    "average_count": len(frame_points),
                    "peak_count": len(frame_points),
                    "frame_id": frame_id,
                }
            )
        flows.finalize_all()
        elapsed = time.perf_counter() - started

        heatmap = heatmaps.snapshot("entire")
        flow = flows.snapshot("entire")
        zone_snapshot = zones.snapshot()
        popular_paths = [self._path_payload(path, source) for path in flows.top_paths(5)]
        counts = list(frame_counts.values())
        result: dict[str, Any] = {
            "dataset": "grand-central",
            "source": source,
            "coordinate_mode": self.coordinate_mode,
            "coordinate_unit": "pixels"
            if self.coordinate_mode == "pixel_space"
            else "unverified_ground_unit",
            "duration": duration_label,
            "from_timestamp": point_list[0].timestamp,
            "to_timestamp": point_list[-1].timestamp,
            "summary": {
                "current_crowd_count": counts[-1],
                "average_crowd_count": round(sum(counts) / len(counts), 3),
                "peak_crowd_count": max(counts),
                "unique_track_count": len(track_lengths),
                "eligible_track_count": len(eligible),
                "processed_frames": len(frame_counts),
                "spatial_mode": self.transformer.mode,
                "calibration_required": False,
            },
            "metrics": {
                "processing_fps": round(len(grouped) / elapsed, 3) if elapsed else 0.0,
                "inference_ms": 0.0,
                "tracking_ms": 0.0,
                "analytics_ms": round(elapsed * 1000.0 / len(grouped), 3),
                "e2e_ms": round(elapsed * 1000.0 / len(grouped), 3),
                "dropped_frames": 0,
                "source_points": len(point_list),
                "filtered_short_track_points": sum(
                    count for track_id, count in track_lengths.items() if track_id not in eligible
                ),
            },
            "heatmap": {
                "source": source,
                "window": "entire",
                "spatial_mode": self.transformer.mode,
                "from_timestamp": heatmap.from_timestamp,
                "to_timestamp": heatmap.to_timestamp,
                "width": self.config.grid_width,
                "height": self.config.grid_height,
                "occupancy": heatmap.occupancy.tolist(),
                "movement": heatmap.movement.tolist(),
                "occupancy_max": float(heatmap.occupancy.max(initial=0.0)),
                "movement_max": float(heatmap.movement.max(initial=0.0)),
            },
            "flow": {
                "source": source,
                "window": "entire",
                "spatial_mode": self.transformer.mode,
                "dominant_direction": flow.dominant_direction,
                "width": self.config.grid_width,
                "height": self.config.grid_height,
                "vx": flow.vx.tolist(),
                "vy": flow.vy.tolist(),
                "samples": flow.samples.tolist(),
                "sample_count": int(flow.samples.sum()),
            },
            "paths": popular_paths,
            "zones": [asdict(metric) for metric in zone_snapshot.zones],
            "zone_flows": [
                {"source": source, **asdict(zone_flow)} for zone_flow in zone_snapshot.flows
            ],
            "timeline": timeline,
        }
        if persist:
            result["artifacts"] = self._persist(result, heatmaps, heatmap.occupancy)
        return result

    def _build_spatial(
        self, config: AnalyticsConfig
    ) -> tuple[SpatialTransformer, AnalyticsConfig]:
        video = self.dataset.video_metadata()
        zones_document = json.loads(
            (self.dataset.paths.processed / "zones.json").read_text(encoding="utf-8")
        )
        if self.coordinate_mode == "pixel_space":
            transformer = SpatialTransformer.pixel(video["width"], video["height"])
            zone_configs = [ZoneConfig.model_validate(zone) for zone in zones_document["zones"]]
            return transformer, config.model_copy(update={"zones": zone_configs})

        matrix = self.dataset.load_homography(required=True)
        assert matrix is not None
        transformed_corners = [
            self.dataset.transform_point(x, y, matrix)
            for x, y in (
                (0.0, 0.0),
                (float(video["width"]), 0.0),
                (float(video["width"]), float(video["height"])),
                (0.0, float(video["height"])),
            )
        ]
        world_width = math.ceil(max(point[0] for point in transformed_corners)) + 1.0
        world_height = math.ceil(max(point[1] for point in transformed_corners)) + 1.0
        transformer = SpatialTransformer(
            mode="ground", width=world_width, height=world_height, matrix=matrix
        )
        zone_configs = []
        for zone in zones_document["zones"]:
            transformed_points = [
                self.dataset.transform_point(float(x), float(y), matrix)
                for x, y in zone["points"]
            ]
            zone_configs.append(
                ZoneConfig(zone_id=zone["zone_id"], name=zone["name"], points=transformed_points)
            )
        adjusted = config.model_copy(
            update={
                "zones": zone_configs,
                "movement_threshold_pixels": 0.15,
                "max_movement_step_pixels": 12.0,
            }
        )
        return transformer, adjusted

    def _persist(
        self,
        result: dict[str, Any],
        analyzer: HeatmapAnalyzer,
        occupancy: np.ndarray,
    ) -> dict[str, str]:
        stem = f"{result['source']}_{result['coordinate_mode']}_{result['duration']}"
        paths = self.dataset.paths
        heatmap_json = paths.output / "heatmaps" / f"{stem}.json"
        heatmap_png = paths.output / "heatmaps" / f"{stem}.png"
        paths_json = paths.output / "popular_paths" / f"{stem}.json"
        zones_json = paths.output / "zone_flows" / f"{stem}.json"
        result_json = paths.output / f"{stem}.json"
        heatmap_json.write_text(json.dumps(result["heatmap"]), encoding="utf-8")
        rendered = analyzer.render(occupancy)
        rendered = cv2.resize(rendered, (1920, 1080), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(str(heatmap_png), rendered)
        paths_json.write_text(json.dumps(result["paths"], indent=2), encoding="utf-8")
        zones_json.write_text(
            json.dumps(
                {
                    "source": result["source"],
                    "coordinate_mode": result["coordinate_mode"],
                    "zones": result["zones"],
                    "flows": result["zone_flows"],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        artifact_map = {
            "result_json": str(result_json.resolve()),
            "heatmap_json": str(heatmap_json.resolve()),
            "heatmap_png": str(heatmap_png.resolve()),
            "popular_paths_json": str(paths_json.resolve()),
            "zone_flows_json": str(zones_json.resolve()),
        }
        serializable = {**result, "artifacts": artifact_map}
        result_json.write_text(json.dumps(serializable), encoding="utf-8")
        latest = paths.output / f"latest_{result['source']}.json"
        latest.write_text(json.dumps(serializable), encoding="utf-8")
        return artifact_map

    def _track_point(self, point: NormalizedTrajectoryPoint) -> TrackPoint:
        return TrackPoint(
            track_id=point.track_id,
            timestamp=point.timestamp,
            frame_id=point.frame_id,
            x=point.foot_x,
            y=point.foot_y,
            raw_x=point.foot_x,
            raw_y=point.foot_y,
            confidence=point.confidence if point.confidence is not None else 1.0,
        )

    def _path_payload(self, path: Any, source: TrajectorySource) -> dict[str, Any]:
        representative = []
        for region in path.regions:
            try:
                row, column = region.removeprefix("R").split("C")
                representative.append(
                    [
                        (int(column) - 0.5) / self.config.path_grid_width,
                        (int(row) - 0.5) / self.config.path_grid_height,
                    ]
                )
            except (ValueError, AttributeError):
                continue
        return {
            "path_id": path.path_id,
            "source": source,
            "from_zone": path.regions[0] if path.regions else None,
            "to_zone": path.regions[-1] if path.regions else None,
            "label": path.label,
            "count": path.count,
            "percentage": path.percentage,
            "representative_path": representative,
            "regions": list(path.regions),
            "kind": path.kind,
        }
