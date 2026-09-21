export type SessionStatus = "idle" | "starting" | "running" | "completed" | "stopped" | "error";
export type HeatmapMetric = "occupancy" | "movement";
export type TimeWindow = "current" | "1m" | "5m" | "entire";

export interface SessionSnapshot {
  camera_id: string;
  source_kind: string;
  status: SessionStatus;
  error: string | null;
  frame_version: number;
  frame_id: number;
  media_timestamp_s: number;
  stream_epoch: string;
  width: number;
  height: number;
}

export interface RuntimeInfo {
  mode: "live" | "replay" | "modal-live";
  source_mode: string;
  inference_executed: boolean | null;
  detector_calls: number | null;
  cache_reads: number | null;
  cache_resets?: number;
  cache_reader_instances?: number;
  cache_skipped_rows?: number;
  cache_mismatch_count?: number;
  detector_workers?: number;
  last_cache_frame_id?: number;
  last_cache_timestamp_s?: number;
}

export interface ModalLiveMetadata {
  source_name: string;
  session_id: string;
  stream_epoch: string;
  processing_mode: "realtime_pts";
  frame_id: number;
  media_time_s: number;
  path_state: "learning" | "confirming" | "active" | "cooling";
  path_reason: string;
  path_revision: number | null;
  evidence_until_s: number;
  preview_fps: number;
  processing_fps: number;
  source_fps: number;
  direction: string | null;
  active_count: number;
  challenger_direction: string | null;
  challenger_count: number;
  confirmation_elapsed_s: number;
  confirmation_required_s: number;
  moving_track_count: number;
  dropped_input_frames: number;
  dropped_preview_frames: number;
}

export interface CrowdSummary {
  current_crowd_count: number;
  average_crowd_count: number;
  peak_crowd_count: number;
  unique_track_count: number;
  processed_frames: number;
  spatial_mode: "pixel" | "ground";
  calibration_required: boolean;
}

export interface PerformanceMetrics {
  input_fps: number;
  processing_fps: number;
  decode_ms: number | null;
  inference_ms: number | null;
  inference_ms_p95?: number | null;
  tracking_ms: number | null;
  analytics_ms: number | null;
  render_ms: number | null;
  encoding_ms: number | null;
  e2e_latency_ms: number | null;
  e2e_latency_ms_p95?: number | null;
  queue_size: number;
  dropped_frames: number;
  cpu_percent: number;
  ram_mb: number;
  gpu_utilization: number | null;
  gpu_memory_mb: number | null;
  common_path_compute_ms?: number | null;
  common_path_compute_ms_p95?: number | null;
  common_path_switches?: number;
  candidate_rejections?: number;
  active_tracks?: number;
  completed_tracks?: number;
  valid_tracks?: number;
}

export interface PopularPath {
  path_id: string;
  label: string;
  count: number;
  percentage: number;
  regions: string[];
  kind: "zone" | "grid";
}

export type CommonPathState = "candidate" | "active" | "cooling" | "retired";

export interface CommonPath {
  path_id: string;
  origin_zone: string;
  destination_zone: string;
  state: CommonPathState;
  unique_tracks_short: number;
  unique_tracks_long: number;
  score: number;
  confidence: number;
  direction: string;
  polyline: [number, number][];
  updated_at: number;
}

export interface ZoneMetric {
  zone_id: string;
  name: string;
  current_people: number;
  unique_tracks: number;
  entry_count: number;
  exit_count: number;
  average_dwell_seconds: number;
  peak_occupancy: number;
}

export interface ZoneFlow {
  from_zone: string;
  to_zone: string;
  count: number;
}

export interface TimelinePoint {
  timestamp: number;
  average_count: number;
  peak_count: number;
}

export interface LiveSnapshot {
  type: "snapshot";
  session: SessionSnapshot | null;
  summary: CrowdSummary;
  metrics: PerformanceMetrics;
  paths: PopularPath[];
  common_paths: CommonPath[];
  points: TrackingPoint[];
  visualization: VisualizationSettings;
  zones: ZoneMetric[];
  zone_flows?: ZoneFlow[];
  timeline: TimelinePoint[];
  frame_url?: string;
}

export interface HeatmapData {
  metric: HeatmapMetric;
  window: TimeWindow;
  spatial_mode: "pixel" | "ground";
  calibration_required: boolean;
  from_timestamp: number;
  to_timestamp: number;
  grid_width: number;
  grid_height: number;
  max: number;
  sum: number;
  values: number[][];
}

export interface FlowData {
  window: TimeWindow;
  spatial_mode: "pixel" | "ground";
  dominant_direction: string;
  vx: number[][];
  vy: number[][];
  samples: number[][];
}

export interface OverlaySettings {
  detection: boolean;
  tracking: boolean;
  trajectory: boolean;
  heatmap: boolean;
  zones: boolean;
  points: boolean;
  track_ids: boolean;
  trajectory_tails: boolean;
  grid: boolean;
  edge_flows: boolean;
  candidate_paths: boolean;
  active_paths: boolean;
  direction_arrows: boolean;
  debug_metrics: boolean;
  popular_paths: boolean;
}

export interface TrackingPoint {
  track_id: number;
  x: number;
  y: number;
  confidence: number;
}

export interface VisualizationSettings {
  show_bounding_boxes: boolean;
  show_track_ids: boolean;
  show_tracking_points: boolean;
  show_individual_trajectories: boolean;
  show_common_path: boolean;
  show_direction_arrows: boolean;
  show_candidate_path: boolean;
  show_zones: boolean;
  show_heatmap: boolean;
  show_grid_debug: boolean;
  show_edge_flow_debug: boolean;
  show_metrics: boolean;
  common_path_style: {
    centerline_color_bgr: [number, number, number];
    corridor_color_bgr: [number, number, number];
    corridor_opacity: number;
    centerline_opacity: number;
    min_width_pixels: number;
    max_width_pixels: number;
    arrow_spacing_pixels: number;
    transition_milliseconds: number;
  };
  point_style: {
    radius_pixels: number;
    color_bgr: [number, number, number];
    outline_color_bgr: [number, number, number];
  };
}

export interface PolygonZone {
  zone_id: string;
  name: string;
  points: [number, number][];
}

export type DatasetSource = "ground_truth" | "prediction";
export type CoordinateMode = "pixel_space" | "ground_plane";
export type DatasetDuration = "1m" | "5m" | "all";

export interface DatasetAnalyticsResult {
  dataset: "grand-central";
  source: DatasetSource;
  coordinate_mode: CoordinateMode;
  coordinate_unit: "pixels" | "unverified_ground_unit";
  duration: DatasetDuration;
  summary: CrowdSummary & { eligible_track_count?: number };
  metrics: Partial<PerformanceMetrics> & { e2e_ms?: number };
  heatmap: {
    source: DatasetSource;
    window: "entire";
    spatial_mode: "pixel" | "ground";
    from_timestamp: number;
    to_timestamp: number;
    width: number;
    height: number;
    occupancy: number[][];
    movement: number[][];
    occupancy_max: number;
    movement_max: number;
  };
  flow: FlowData & { source: DatasetSource; width: number; height: number; sample_count: number };
  paths: PopularPath[];
  zones: ZoneMetric[];
  zone_flows: ZoneFlow[];
  timeline: TimelinePoint[];
  artifacts: Record<string, string>;
}
