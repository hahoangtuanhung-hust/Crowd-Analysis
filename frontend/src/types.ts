export type SessionStatus = "idle" | "starting" | "running" | "completed" | "stopped" | "error";
export type HeatmapMetric = "occupancy" | "movement";
export type TimeWindow = "current" | "1m" | "5m" | "entire";

export interface SessionSnapshot {
  camera_id: string;
  source_kind: string;
  status: SessionStatus;
  error: string | null;
  frame_version: number;
  width: number;
  height: number;
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
}

export interface PopularPath {
  path_id: string;
  label: string;
  count: number;
  percentage: number;
  regions: string[];
  kind: "zone" | "grid";
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
  popular_paths: boolean;
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
