import type {
  CoordinateMode,
  DatasetAnalyticsResult,
  DatasetDuration,
  DatasetSource,
  FlowData,
  HeatmapData,
  HeatmapMetric,
  RuntimeInfo,
  TimeWindow,
  VisualizationSettings
} from "./types";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, init);
  if (!response.ok) {
    const payload = await response.json().catch(() => null) as { detail?: string } | null;
    throw new Error(payload?.detail ?? `${response.status} ${response.statusText}`);
  }
  return response.json() as Promise<T>;
}

export async function uploadVideo(file: File): Promise<{ source_token: string }> {
  const form = new FormData();
  form.append("video", file);
  return request("/api/video/upload", { method: "POST", body: form });
}

export function startStream(source: string, cameraId: string, realtime?: boolean): Promise<unknown> {
  return request("/api/stream/start", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ source, camera_id: cameraId, realtime })
  });
}

export function getRuntimeInfo(): Promise<RuntimeInfo> {
  return request("/api/runtime");
}

export function stopStream(): Promise<unknown> {
  return request("/api/stream/stop", { method: "POST" });
}

export function getVisualization(): Promise<VisualizationSettings> {
  return request("/api/config/visualization");
}

export function patchVisualization(
  settings: Partial<Omit<VisualizationSettings, "common_path_style" | "point_style">>
): Promise<VisualizationSettings> {
  return request("/api/config/visualization", {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(settings)
  });
}

export function getGrandCentralResult(source: DatasetSource, coordinateMode: CoordinateMode, duration: DatasetDuration): Promise<DatasetAnalyticsResult> {
  const params = new URLSearchParams({ source, coordinate_mode: coordinateMode, duration });
  return request(`/api/datasets/grand-central/results?${params}`);
}

export function analyzeGrandCentral(source: DatasetSource, coordinateMode: CoordinateMode, duration: DatasetDuration): Promise<DatasetAnalyticsResult> {
  return request("/api/datasets/grand-central/analyze", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ source, coordinate_mode: coordinateMode, duration })
  });
}

export function getHeatmap(metric: HeatmapMetric, window: TimeWindow, signal?: AbortSignal): Promise<HeatmapData> {
  const params = new URLSearchParams({ metric, window });
  return request(`/api/analytics/heatmap?${params}`, { signal });
}

export function getFlow(window: TimeWindow, signal?: AbortSignal): Promise<FlowData> {
  return request(`/api/analytics/flow?window=${window}`, { signal });
}

export function setCalibration(points: [number, number][], width: number, height: number): Promise<unknown> {
  return request("/api/calibration", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ points, width, height })
  });
}

export function setZones(zones: { zone_id: string; name: string; points: [number, number][] }[]): Promise<unknown> {
  return request("/api/zones", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ zones, coordinate_space: "camera" })
  });
}
