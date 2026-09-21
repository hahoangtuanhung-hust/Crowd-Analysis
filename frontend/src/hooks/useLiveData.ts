import { useEffect, useRef, useState } from "react";

import type { LiveSnapshot } from "../types";

export const EMPTY_SNAPSHOT: LiveSnapshot = {
  type: "snapshot",
  session: null,
  summary: {
    current_crowd_count: 0,
    average_crowd_count: 0,
    peak_crowd_count: 0,
    unique_track_count: 0,
    processed_frames: 0,
    spatial_mode: "pixel",
    calibration_required: true
  },
  metrics: {
    input_fps: 0,
    processing_fps: 0,
    decode_ms: null,
    inference_ms: null,
    tracking_ms: null,
    analytics_ms: null,
    render_ms: null,
    encoding_ms: null,
    e2e_latency_ms: null,
    queue_size: 0,
    dropped_frames: 0,
    cpu_percent: 0,
    ram_mb: 0,
    gpu_utilization: null,
    gpu_memory_mb: null
  },
  paths: [],
  common_paths: [],
  points: [],
  visualization: {
    show_bounding_boxes: false,
    show_track_ids: false,
    show_tracking_points: true,
    show_individual_trajectories: false,
    show_common_path: true,
    show_direction_arrows: true,
    show_candidate_path: false,
    show_zones: false,
    show_heatmap: false,
    show_grid_debug: false,
    show_edge_flow_debug: false,
    show_metrics: true,
    common_path_style: {
      centerline_color_bgr: [0, 255, 0],
      corridor_color_bgr: [0, 180, 0],
      corridor_opacity: 0.25,
      centerline_opacity: 0.75,
      min_width_pixels: 6,
      max_width_pixels: 18,
      arrow_spacing_pixels: 80,
      transition_milliseconds: 500
    },
    point_style: {
      radius_pixels: 2,
      color_bgr: [0, 255, 255],
      outline_color_bgr: [0, 0, 0]
    }
  },
  zones: [],
  zone_flows: [],
  timeline: []
};

export function useLiveData(enabled = true): { data: LiveSnapshot; connected: boolean } {
  const [data, setData] = useState<LiveSnapshot>(EMPTY_SNAPSHOT);
  const [connected, setConnected] = useState(false);
  const retryRef = useRef(500);

  useEffect(() => {
    if (!enabled) return;
    let active = true;
    let socket: WebSocket | null = null;
    let timer: number | undefined;

    const connect = () => {
      const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
      socket = new WebSocket(`${protocol}//${window.location.host}/ws/live`);
      socket.onopen = () => {
        if (!active) return;
        retryRef.current = 500;
        setConnected(true);
      };
      socket.onmessage = (event) => {
        if (!active) return;
        setData(JSON.parse(event.data) as LiveSnapshot);
      };
      socket.onclose = () => {
        if (!active) return;
        setConnected(false);
        timer = window.setTimeout(connect, retryRef.current);
        retryRef.current = Math.min(8000, retryRef.current * 2);
      };
    };

    connect();
    return () => {
      active = false;
      if (timer !== undefined) window.clearTimeout(timer);
      socket?.close();
    };
  }, [enabled]);

  return { data, connected };
}
