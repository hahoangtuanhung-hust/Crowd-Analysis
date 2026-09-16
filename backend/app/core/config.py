from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DetectorConfig(StrictModel):
    provider: Literal["ultralytics"] = "ultralytics"
    model: str = "yolo26n.pt"
    imgsz: int = Field(default=1280, ge=32)
    confidence: float = Field(default=0.05, ge=0.0, le=1.0)
    iou: float = Field(default=0.6, ge=0.0, le=1.0)
    max_det: int = Field(default=1000, ge=1)
    classes: list[int] = Field(default_factory=lambda: [0])
    device: str = "auto"
    precision: Literal["fp32", "fp16"] = "fp32"


class TrackerConfig(StrictModel):
    type: Literal["bytetrack"] = "bytetrack"
    track_high_thresh: float = Field(default=0.15, ge=0.0, le=1.0)
    track_low_thresh: float = Field(default=0.05, ge=0.0, le=1.0)
    new_track_thresh: float = Field(default=0.15, ge=0.0, le=1.0)
    track_buffer: int = Field(default=60, ge=1)
    match_thresh: float = Field(default=0.75, ge=0.0, le=1.0)
    fuse_score: bool = False
    association_mode: Literal["iou", "hybrid"] = "hybrid"
    max_center_distance_ratio: float = Field(default=0.035, gt=0.0, le=1.0)
    second_match_thresh: float = Field(default=0.6, ge=0.0, le=1.0)
    lost_track_grace_frames: int = Field(default=3, ge=0)

    @model_validator(mode="after")
    def validate_threshold_order(self) -> TrackerConfig:
        if self.track_low_thresh > self.track_high_thresh:
            raise ValueError("track_low_thresh must not exceed track_high_thresh")
        if self.new_track_thresh < self.track_high_thresh:
            raise ValueError("new_track_thresh must not be below track_high_thresh")
        return self


class VideoConfig(StrictModel):
    inference_interval: int = Field(default=1, ge=1)
    queue_size: int = Field(default=4, ge=1, le=128)
    stale_frame_policy: Literal["drop_oldest", "block"] = "drop_oldest"
    reconnect_initial_seconds: float = Field(default=1.0, gt=0.0)
    reconnect_max_seconds: float = Field(default=30.0, gt=0.0)


class ZoneConfig(StrictModel):
    zone_id: str = Field(min_length=1, max_length=64)
    name: str = Field(min_length=1, max_length=128)
    points: list[tuple[float, float]] = Field(min_length=3)


class ServerConfig(StrictModel):
    host: str = "0.0.0.0"
    port: int = Field(default=8000, ge=1, le=65535)
    upload_directory: str = "data/uploads"
    max_upload_mb: int = Field(default=500, ge=1)
    jpeg_quality: int = Field(default=82, ge=30, le=100)
    websocket_interval_ms: int = Field(default=250, ge=50, le=5000)
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:5173"])


class AnalyticsConfig(StrictModel):
    grid_width: int = Field(default=64, ge=4)
    grid_height: int = Field(default=36, ge=4)
    trajectory_history_points: int = Field(default=120, ge=2)
    trajectory_smoothing_alpha: float = Field(default=0.35, gt=0.0, le=1.0)
    min_confirmed_points: int = Field(default=3, ge=1)
    max_active_tracks: int = Field(default=2048, ge=1)
    inactive_track_ttl_seconds: float = Field(default=10.0, gt=0.0)
    movement_threshold_pixels: float = Field(default=3.0, ge=0.0)
    max_movement_step_pixels: float = Field(default=250.0, gt=0.0)
    max_point_gap_seconds: float = Field(default=2.0, gt=0.0)
    current_window_seconds: int = Field(default=10, ge=1)
    live_retention_seconds: int = Field(default=300, ge=10)
    gaussian_sigma: float = Field(default=1.5, ge=0.0)
    path_grid_width: int = Field(default=8, ge=2, le=64)
    path_grid_height: int = Field(default=5, ge=2, le=64)
    max_route_cells: int = Field(default=32, ge=2, le=256)
    max_completed_paths: int = Field(default=2048, ge=1)
    zone_debounce_points: int = Field(default=2, ge=1, le=30)
    zones: list[ZoneConfig] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_zones(self) -> AnalyticsConfig:
        zone_ids = [zone.zone_id for zone in self.zones]
        if len(zone_ids) != len(set(zone_ids)):
            raise ValueError("zone_id values must be unique")
        return self


class AppConfig(StrictModel):
    server: ServerConfig = Field(default_factory=ServerConfig)
    detector: DetectorConfig = Field(default_factory=DetectorConfig)
    tracker: TrackerConfig = Field(default_factory=TrackerConfig)
    video: VideoConfig = Field(default_factory=VideoConfig)
    analytics: AnalyticsConfig = Field(default_factory=AnalyticsConfig)

    @model_validator(mode="after")
    def validate_detection_tracking_thresholds(self) -> AppConfig:
        if self.detector.confidence > self.tracker.track_low_thresh:
            raise ValueError(
                "detector.confidence must not exceed tracker.track_low_thresh; "
                "otherwise ByteTrack cannot use its low-confidence recovery stage"
            )
        return self


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}
    return AppConfig.model_validate(raw)
