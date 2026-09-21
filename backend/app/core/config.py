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
    tiled_inference: bool = False
    tile_rows: int = Field(default=2, ge=1, le=4)
    tile_columns: int = Field(default=2, ge=1, le=4)
    tile_overlap: float = Field(default=0.2, ge=0.0, lt=0.5)
    tile_merge_iou: float = Field(default=0.45, gt=0.0, le=1.0)
    tile_include_full_frame: bool = True
    ignore_regions: list[tuple[float, float, float, float]] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_ignore_regions(self) -> DetectorConfig:
        for x1, y1, x2, y2 in self.ignore_regions:
            if not all(0.0 <= value <= 1.0 for value in (x1, y1, x2, y2)):
                raise ValueError("detector ignore_regions coordinates must be normalized")
            if x1 >= x2 or y1 >= y2:
                raise ValueError("detector ignore_regions must use x1 < x2 and y1 < y2")
        return self


class TrackerConfig(StrictModel):
    type: Literal["bytetrack"] = "bytetrack"
    track_high_thresh: float = Field(default=0.15, ge=0.0, le=1.0)
    track_low_thresh: float = Field(default=0.05, ge=0.0, le=1.0)
    new_track_thresh: float = Field(default=0.15, ge=0.0, le=1.0)
    track_buffer: int = Field(default=120, ge=1)
    match_thresh: float = Field(default=0.75, ge=0.0, le=1.0)
    fuse_score: bool = False
    association_mode: Literal["iou", "hybrid"] = "hybrid"
    max_center_distance_ratio: float = Field(default=0.045, gt=0.0, le=1.0)
    second_match_thresh: float = Field(default=0.65, ge=0.0, le=1.0)
    lost_track_grace_frames: int = Field(default=8, ge=0)
    stationary_lost_track_grace_frames: int | None = Field(default=None, ge=0)
    stationary_speed_threshold: float = Field(default=3.0, ge=0.0)
    stationary_boost: bool = True

    @model_validator(mode="after")
    def validate_threshold_order(self) -> TrackerConfig:
        if self.track_low_thresh > self.track_high_thresh:
            raise ValueError("track_low_thresh must not exceed track_high_thresh")
        if self.new_track_thresh < self.track_high_thresh:
            raise ValueError("new_track_thresh must not be below track_high_thresh")
        if (
            self.stationary_lost_track_grace_frames is not None
            and self.stationary_lost_track_grace_frames < self.lost_track_grace_frames
        ):
            raise ValueError(
                "stationary_lost_track_grace_frames must not be below "
                "lost_track_grace_frames"
            )
        return self


class VideoConfig(StrictModel):
    inference_interval: int = Field(default=1, ge=1)
    queue_size: int = Field(default=4, ge=1, le=128)
    analytics_queue_size: int = Field(default=1000, ge=1, le=100_000)
    stale_frame_policy: Literal["drop_oldest", "block"] = "drop_oldest"
    reconnect_initial_seconds: float = Field(default=1.0, gt=0.0)
    reconnect_max_seconds: float = Field(default=30.0, gt=0.0)


class ZoneConfig(StrictModel):
    zone_id: str = Field(min_length=1, max_length=64)
    name: str = Field(min_length=1, max_length=128)
    zone_type: Literal["entry", "exit", "area"] = "area"
    points: list[tuple[float, float]] = Field(min_length=3)


class ServerConfig(StrictModel):
    host: str = "0.0.0.0"
    port: int = Field(default=8000, ge=1, le=65535)
    upload_directory: str = "data/uploads"
    max_upload_mb: int = Field(default=500, ge=1)
    jpeg_quality: int = Field(default=82, ge=30, le=100)
    websocket_interval_ms: int = Field(default=250, ge=50, le=5000)
    zones_path: str | None = None
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:5173"])


class CommonPathConfig(StrictModel):
    enabled: bool = True
    engine: Literal["legacy", "directional_grid", "shadow"] = "legacy"
    shadow_display: Literal["legacy", "directional_grid"] = "legacy"
    max_active_paths: int = Field(default=1, ge=1, le=5)
    grid_columns: int = Field(default=8, ge=2, le=128)
    grid_rows: int = Field(default=5, ge=2, le=128)
    cell_hysteresis_ratio: float = Field(default=0.12, ge=0.0, lt=0.5)
    cell_confirmation_points: int = Field(default=2, ge=1, le=10)
    bucket_seconds: int = Field(default=1, ge=1, le=60)
    short_window_seconds: int = Field(default=30, ge=1)
    long_window_seconds: int = Field(default=180, ge=2)
    short_weight: float = Field(default=0.40, ge=0.0, le=1.0)
    long_weight: float = Field(default=0.60, ge=0.0, le=1.0)
    update_interval_seconds: float = Field(default=3.0, gt=0.0)
    flow_update_interval_seconds: float = Field(default=1.0, gt=0.0)
    min_unique_tracks: int = Field(default=5, ge=1)
    switch_margin: float = Field(default=0.20, ge=0.0)
    confirmation_seconds: float = Field(default=8.0, ge=0.0)
    cooling_seconds: float = Field(default=20.0, ge=0.0)
    min_path_edge_support: int = Field(default=3, ge=1)
    path_similarity_threshold: float = Field(default=0.70, ge=0.0, le=1.0)
    centerline_ema_alpha: float = Field(default=0.20, gt=0.0, le=1.0)
    min_track_duration_seconds: float = Field(default=1.0, ge=0.0)
    min_track_distance_pixels: float = Field(default=30.0, ge=0.0)
    min_track_points: int = Field(default=6, ge=2)
    max_stationary_speed_pixels_second: float = Field(default=3.0, ge=0.0)
    max_step_pixels: float = Field(default=250.0, gt=0.0)
    track_lost_timeout_seconds: float = Field(default=4.0, gt=0.0)
    zone_min_inside_frames: int = Field(default=3, ge=1, le=30)
    zone_debounce_seconds: float = Field(default=1.0, ge=0.0)
    max_path_cells: int = Field(default=128, ge=2, le=1024)
    top_k: int = Field(default=5, ge=1, le=20)

    @model_validator(mode="after")
    def validate_windows_and_weights(self) -> CommonPathConfig:
        if self.short_window_seconds >= self.long_window_seconds:
            raise ValueError("short_window_seconds must be below long_window_seconds")
        if self.short_weight + self.long_weight <= 0:
            raise ValueError("short_weight and long_weight cannot both be zero")
        return self


class LocalCorridorConfig(StrictModel):
    corridor_id: str = Field(default="local-corridor", min_length=1, max_length=64)
    coordinate_space: Literal["image_normalized", "image_pixels"] = "image_normalized"
    roi: list[tuple[float, float]] = Field(min_length=3)
    source_gate: list[tuple[float, float]] = Field(min_length=3)
    target_gate: list[tuple[float, float]] = Field(min_length=3)
    allow_reverse: bool = True

    @model_validator(mode="after")
    def validate_coordinates(self) -> LocalCorridorConfig:
        if self.coordinate_space == "image_normalized":
            for polygon in (self.roi, self.source_gate, self.target_gate):
                if any(not 0.0 <= coordinate <= 1.0 for point in polygon for coordinate in point):
                    raise ValueError("normalized corridor coordinates must be within [0, 1]")
        return self


class DirectionalGridConfig(StrictModel):
    display_policy: Literal["validated_route", "dominant_direction"] = "validated_route"
    dominant_min_support_tracks: int = Field(default=1, ge=1)
    columns: int = Field(default=32, ge=2, le=128)
    rows: int = Field(default=18, ge=2, le=128)
    direction_bins: int = Field(default=8, ge=4, le=16)
    direction_sample_seconds: float = Field(default=0.3, gt=0)
    min_track_age_seconds: float = Field(default=0.5, ge=0)
    min_displacement_cell_fraction: float = Field(default=0.15, gt=0)
    max_observation_gap_seconds: float = Field(default=0.5, gt=0)
    max_step_cells: float = Field(default=2.5, gt=0)
    short_window_seconds: float = Field(default=30, gt=0)
    long_window_seconds: float = Field(default=180, gt=0)
    min_edge_unique_tracks: int = Field(default=3, ge=1)
    min_support_tracks: int = Field(default=5, ge=1)
    min_complete_tracks: int = Field(default=3, ge=1)
    min_ordered_coverage: float = Field(default=0.70, gt=0, le=1)
    cell_tolerance: int = Field(default=1, ge=0, le=2)
    confirmation_seconds: float = Field(default=8, ge=0)
    cooling_seconds: float = Field(default=20, ge=0)
    switch_relative_margin: float = Field(default=0.20, ge=0)
    switch_absolute_margin: float = Field(default=0.01, ge=0)
    update_interval_seconds: float = Field(default=3, gt=0)
    max_candidates: int = Field(default=5, ge=1, le=20)
    beam_width: int = Field(default=20, ge=1, le=100)
    max_path_cells: int = Field(default=128, ge=2, le=1024)
    min_path_cells: int = Field(default=4, ge=2)
    max_tracks: int = Field(default=2048, ge=1)
    max_points_per_track: int = Field(default=256, ge=2)
    max_evidence_keys: int = Field(default=250000, ge=1)
    route_scope: Literal["global_od", "local_corridor"] = "global_od"
    local_corridor: LocalCorridorConfig | None = None
    diagnostics_enabled: bool = False
    max_diagnostic_evaluations: int = Field(default=5000, ge=1, le=100000)
    max_diagnostic_evidence_samples: int = Field(default=20, ge=0, le=100)

    @model_validator(mode="after")
    def validate_windows(self) -> DirectionalGridConfig:
        if self.short_window_seconds >= self.long_window_seconds:
            raise ValueError("directional short window must be shorter than long window")
        if self.min_path_cells > self.max_path_cells:
            raise ValueError("min_path_cells must not exceed max_path_cells")
        if self.route_scope == "local_corridor" and self.local_corridor is None:
            raise ValueError("local_corridor config is required for local_corridor scope")
        return self


class DominantLiveFlowConfig(StrictModel):
    mode: Literal["dominant_live_flow", "validated_route"] = "validated_route"
    columns: int = Field(default=32, ge=2, le=128)
    rows: int = Field(default=18, ge=2, le=128)
    direction_bins: int = Field(default=8, ge=4, le=16)
    direction_min_seconds: float = Field(default=0.3, gt=0)
    direction_max_seconds: float = Field(default=0.8, gt=0)
    direction_smoothing_alpha: float = Field(default=0.4, gt=0, le=1)
    min_displacement_cell_fraction: float = Field(default=0.15, gt=0)
    observation_timeout_seconds: float = Field(default=0.6, gt=0)
    history_seconds: float = Field(default=3.0, gt=0)
    evaluation_interval_seconds: float = Field(default=0.5, gt=0)
    min_active_tracks: int = Field(default=3, ge=1)
    confirmation_seconds: float = Field(default=2.0, ge=0)
    challenger_margin_tracks: int = Field(default=1, ge=1)
    stale_seconds: float = Field(default=1.5, ge=0)
    max_tracks: int = Field(default=4096, ge=1)
    max_points_per_track: int = Field(default=64, ge=2)
    min_geometry_cells: float = Field(default=1.5, gt=0)

    @model_validator(mode="after")
    def validate_timing(self) -> DominantLiveFlowConfig:
        if self.direction_min_seconds >= self.direction_max_seconds:
            raise ValueError("direction_min_seconds must be below direction_max_seconds")
        if self.direction_max_seconds > self.history_seconds:
            raise ValueError("direction_max_seconds must not exceed history_seconds")
        return self


class CommonPathStyleConfig(StrictModel):
    centerline_color_bgr: tuple[int, int, int] = (0, 255, 0)
    corridor_color_bgr: tuple[int, int, int] = (0, 180, 0)
    corridor_opacity: float = Field(default=0.25, ge=0.0, le=1.0)
    centerline_opacity: float = Field(default=0.75, ge=0.0, le=1.0)
    min_width_pixels: int = Field(default=6, ge=1, le=64)
    max_width_pixels: int = Field(default=18, ge=1, le=96)
    arrow_spacing_pixels: int = Field(default=80, ge=20, le=500)
    transition_milliseconds: int = Field(default=500, ge=0, le=5000)

    @model_validator(mode="after")
    def validate_widths_and_colors(self) -> CommonPathStyleConfig:
        if self.min_width_pixels > self.max_width_pixels:
            raise ValueError("min_width_pixels must not exceed max_width_pixels")
        for color in (self.centerline_color_bgr, self.corridor_color_bgr):
            if any(channel < 0 or channel > 255 for channel in color):
                raise ValueError("BGR color channels must be between 0 and 255")
        return self


class PointStyleConfig(StrictModel):
    radius_pixels: int = Field(default=2, ge=1, le=30)
    color_bgr: tuple[int, int, int] = (0, 255, 255)
    outline_color_bgr: tuple[int, int, int] = (0, 0, 0)

    @model_validator(mode="after")
    def validate_colors(self) -> PointStyleConfig:
        for color in (self.color_bgr, self.outline_color_bgr):
            if any(channel < 0 or channel > 255 for channel in color):
                raise ValueError("BGR color channels must be between 0 and 255")
        return self


class VisualizationConfig(StrictModel):
    show_bounding_boxes: bool = False
    show_track_ids: bool = False
    show_tracking_points: bool = True
    show_individual_trajectories: bool = False
    show_common_path: bool = True
    show_direction_arrows: bool = True
    show_candidate_path: bool = False
    show_zones: bool = False
    show_heatmap: bool = False
    show_grid_debug: bool = False
    show_edge_flow_debug: bool = False
    show_metrics: bool = True
    common_path_style: CommonPathStyleConfig = Field(default_factory=CommonPathStyleConfig)
    point_style: PointStyleConfig = Field(default_factory=PointStyleConfig)


class CommonPathTestConfig(StrictModel):
    inference_provider: Literal["modal"] = "modal"
    allow_local_inference: bool = False
    mode: Literal["offline_fast", "realtime"] = "offline_fast"
    smoke_duration_seconds: int = Field(default=25, ge=1, le=30)
    integration_duration_seconds: int = Field(default=75, ge=30, le=90)
    max_gpu_jobs: int = Field(default=3, ge=1, le=3)
    total_compute_budget_seconds: int = Field(default=900, ge=1, le=900)
    job_timeout_seconds: int = Field(default=600, ge=1, le=600)
    cache_policy: Literal["reuse", "refresh"] = "reuse"
    inspect_artifacts: bool = True


class AnalyticsConfig(StrictModel):
    grid_width: int = Field(default=64, ge=4)
    grid_height: int = Field(default=36, ge=4)
    trajectory_history_points: int = Field(default=150, ge=2)
    trajectory_smoothing_alpha: float = Field(default=0.35, gt=0.0, le=1.0)
    trajectory_min_point_distance_pixels: float = Field(default=0.0, ge=0.0)
    trajectory_max_history_seconds: float = Field(default=30.0, gt=0.0)
    min_confirmed_points: int = Field(default=3, ge=1)
    max_active_tracks: int = Field(default=2048, ge=1)
    inactive_track_ttl_seconds: float = Field(default=15.0, gt=0.0)
    movement_threshold_pixels: float = Field(default=2.5, ge=0.0)
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
    ddcrp_enabled: bool = False
    ddcrp_alpha: float = Field(default=0.1, gt=0.0)
    ddcrp_theta_min: float = Field(default=0.1) # radians
    ddcrp_theta_max: float = Field(default=0.5) # radians
    ddcrp_delta_min: float = Field(default=10.0) # pixels
    ddcrp_delta_max: float = Field(default=100.0) # pixels
    ddcrp_tracklet_seconds: float = Field(default=1.5, gt=0.0)
    ddcrp_stationary_threshold: float = Field(default=18.0, ge=0.0)
    common_path: CommonPathConfig = Field(default_factory=CommonPathConfig)
    directional_grid: DirectionalGridConfig = Field(default_factory=DirectionalGridConfig)
    dominant_live_flow: DominantLiveFlowConfig = Field(default_factory=DominantLiveFlowConfig)
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
    visualization: VisualizationConfig = Field(default_factory=VisualizationConfig)
    test: CommonPathTestConfig = Field(default_factory=CommonPathTestConfig)

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
