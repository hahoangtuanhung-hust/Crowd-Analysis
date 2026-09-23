import { useCallback, useEffect, useRef, useState } from "react";

import type { CommonPath, LiveSnapshot, ModalLiveMetadata, SessionStatus } from "../types";
import { EMPTY_SNAPSHOT } from "./useLiveData";

interface RemotePath extends CommonPath {
  revision: number;
  support_tracks: number;
  validated_complete_tracks: number;
  evidence_until_s: number;
}

interface FrameHeader extends ModalLiveMetadata {
  version: number;
  type: "frame";
  event_seq: number;
  frame_width: number;
  frame_height: number;
  observed_people: number;
  unique_track_ids: number;
  inference_ms: number;
  queue_depth: number;
  dropped_input_frames: number;
  jpeg_bytes: number;
  common_path: RemotePath | null;
  paths?: RemotePath[];
}

interface ModalEvent {
  type: "event";
  event: string;
  event_seq: number;
  session_id?: string;
  stream_epoch?: string;
  message?: string;
  manifest?: { status?: string };
  applied_max_paths?: number;
}

const endpointValue = (import.meta.env.VITE_MODAL_LIVE_WS_URL as string | undefined)?.trim() ?? "";
const token = (import.meta.env.VITE_MODAL_LIVE_TOKEN as string | undefined)?.trim() ?? "";
const requested = Boolean(endpointValue || token);
const modalConfiguration = validateModalConfiguration(endpointValue, token);
const endpoint = modalConfiguration.endpoint;
const LIVE_DURATION_SECONDS = 200;
const defaultDuration = boundedNumber(
  import.meta.env.VITE_MODAL_LIVE_DURATION_SECONDS,
  LIVE_DURATION_SECONDS,
  1,
  LIVE_DURATION_SECONDS,
);
const defaultPreviewFps = boundedNumber(import.meta.env.VITE_MODAL_LIVE_PREVIEW_FPS, 10, 1, 15);
const configuredRunId = (import.meta.env.VITE_MODAL_LIVE_RUN_ID as string | undefined)?.trim() ?? "";

export function useModalLive() {
  const enabled = modalConfiguration.error === null && Boolean(endpoint && token);
  const socketRef = useRef<WebSocket | null>(null);
  const imageUrlRef = useRef<string | undefined>(undefined);
  const lastFrameRef = useRef(-1);
  const peakRef = useRef(0);
  const terminalRef = useRef(false);
  const [connected, setConnected] = useState(false);
  const [frameUrl, setFrameUrl] = useState<string>();
  const [metadata, setMetadata] = useState<ModalLiveMetadata>();
  const [status, setStatus] = useState<SessionStatus>(modalConfiguration.error ? "error" : "idle");
  const [error, setError] = useState<string | null>(modalConfiguration.error);
  const [identity, setIdentity] = useState({ sessionId: "", epoch: "" });
  const [data, setData] = useState<LiveSnapshot>(EMPTY_SNAPSHOT);
  const [maxPaths, setMaxPaths] = useState(1);
  const [appliedMaxPaths, setAppliedMaxPaths] = useState(1);

  useEffect(() => {
    if (!enabled) return;
    let active = true;
    const url = new URL(endpoint!);
    const socket = new WebSocket(url);
    socket.binaryType = "arraybuffer";
    socketRef.current = socket;

    socket.onopen = () => {
      if (!active) return;
      setError(null);
      socket.send(JSON.stringify({ action: "authenticate", token }));
    };
    socket.onmessage = async (message) => {
      if (!active) return;
      if (typeof message.data === "string") {
        const event = JSON.parse(message.data) as ModalEvent;
        if (event.event === "ready") {
          setConnected(true);
          setStatus((current) => current === "error" ? "idle" : current);
        }
        if (event.event === "max_paths_applied" && event.applied_max_paths) {
          setAppliedMaxPaths(event.applied_max_paths);
        }
        if (event.event === "control_error") setError(event.message ?? "Invalid path limit");
        if (event.event === "loading_model") {
          setIdentity({ sessionId: event.session_id ?? "", epoch: event.stream_epoch ?? "" });
          setStatus("starting");
        }
        if (event.event === "ended") {
          terminalRef.current = true;
          setStatus(event.manifest?.status === "stopped" ? "stopped" : "completed");
        }
        if (event.event === "error") {
          setError(event.message ?? "Remote inference failed");
          setStatus("error");
        }
        return;
      }

      const packet = message.data as ArrayBuffer;
      if (packet.byteLength < 5 || packet.byteLength > 2_000_000) return;
      const headerLength = new DataView(packet).getUint32(0, false);
      if (headerLength < 2 || headerLength > 65_536 || 4 + headerLength >= packet.byteLength) return;
      const header = JSON.parse(new TextDecoder().decode(packet.slice(4, 4 + headerLength))) as FrameHeader;
      if (header.version !== 1 || header.type !== "frame" || header.frame_id <= lastFrameRef.current) return;
      const jpeg = new Blob([packet.slice(4 + headerLength)], { type: "image/jpeg" });
      const decoded = await createImageBitmap(jpeg);
      decoded.close();
      if (header.frame_id <= lastFrameRef.current) return;
      lastFrameRef.current = header.frame_id;
      peakRef.current = Math.max(peakRef.current, header.observed_people);
      const nextUrl = URL.createObjectURL(jpeg);
      if (imageUrlRef.current) URL.revokeObjectURL(imageUrlRef.current);
      imageUrlRef.current = nextUrl;
      setFrameUrl(nextUrl);
      setMetadata(header);
      if (header.applied_max_paths) setAppliedMaxPaths(header.applied_max_paths);
      setStatus("running");
      setData({
        ...EMPTY_SNAPSHOT,
        session: {
          camera_id: "shibuya-01",
          source_kind: "modal-volume",
          status: "running",
          error: null,
          frame_version: header.frame_id + 1,
          frame_id: header.frame_id,
          media_timestamp_s: header.media_time_s,
          stream_epoch: header.stream_epoch,
          width: header.frame_width,
          height: header.frame_height
        },
        summary: {
          ...EMPTY_SNAPSHOT.summary,
          current_crowd_count: header.observed_people,
          peak_crowd_count: peakRef.current,
          unique_track_count: header.unique_track_ids,
          processed_frames: header.frame_id + 1
        },
        metrics: {
          ...EMPTY_SNAPSHOT.metrics,
          input_fps: 30,
          processing_fps: header.processing_fps,
          inference_ms: header.inference_ms,
          queue_size: header.queue_depth,
          dropped_frames: header.dropped_input_frames
        },
        common_paths: header.paths ?? (header.common_path ? [header.common_path] : []),
        frame_url: nextUrl
      });
      socket.send(JSON.stringify({ action: "frame_ack", frame_id: header.frame_id, event_seq: header.event_seq }));
    };
    socket.onerror = () => {
      if (active && !terminalRef.current) setError("Cannot connect to the authenticated Modal live endpoint");
    };
    socket.onclose = () => {
      if (!active) return;
      setConnected(false);
      setStatus((current) => terminalRef.current || current === "completed" || current === "stopped" ? current : "error");
    };
    return () => {
      active = false;
      socket.close();
      if (imageUrlRef.current) URL.revokeObjectURL(imageUrlRef.current);
    };
  }, [enabled]);

  const start = useCallback(() => {
    const runId = createRunId(configuredRunId);
    peakRef.current = 0;
    lastFrameRef.current = -1;
    terminalRef.current = false;
    setError(null);
    setStatus("starting");
    socketRef.current?.send(JSON.stringify({
      action: "start",
      run_id: runId,
      duration_seconds: defaultDuration,
      preview_fps: defaultPreviewFps,
      processing_mode: "realtime_pts",
      client_kind: "browser-ui",
      max_paths: maxPaths
    }));
  }, [maxPaths]);

  const changeMaxPaths = useCallback((value: number) => {
    if (!Number.isInteger(value) || value < 1 || value > 5) return;
    setMaxPaths(value);
    if (socketRef.current?.readyState === WebSocket.OPEN && status === "running") {
      socketRef.current.send(JSON.stringify({ action: "set_max_paths", max_paths: value }));
    }
  }, [status]);

  const stop = useCallback(() => socketRef.current?.send(JSON.stringify({ action: "stop" })), []);
  const mergedData = data.session ? { ...data, session: { ...data.session, status, error } } : data;
  return { requested, enabled, connected, data: mergedData, frameUrl, metadata, status, error, identity, start, stop,
    maxPaths, appliedMaxPaths, changeMaxPaths };
}

function createRunId(prefix: string): string {
  const base = prefix || "live-tracklet";
  const suffix = `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 7)}`;
  const maxBaseLength = Math.max(1, 64 - suffix.length - 1);
  return `${base.slice(0, maxBaseLength)}-${suffix}`;
}

function validateModalConfiguration(
  rawEndpoint: string,
  rawToken: string,
): { endpoint: string | null; error: string | null } {
  if (!rawEndpoint && !rawToken) return { endpoint: null, error: null };
  if (!rawEndpoint) return { endpoint: null, error: "Missing VITE_MODAL_LIVE_WS_URL" };
  if (!rawToken) return { endpoint: null, error: "Missing VITE_MODAL_LIVE_TOKEN" };
  if (/[<>]/.test(rawEndpoint) || /[<>]/.test(rawToken)) {
    return { endpoint: null, error: "Modal configuration still contains a placeholder; set the real WSS URL and session token" };
  }
  try {
    const parsed = new URL(rawEndpoint);
    if (parsed.protocol !== "ws:" && parsed.protocol !== "wss:") {
      return { endpoint: null, error: "VITE_MODAL_LIVE_WS_URL must use ws:// or wss://" };
    }
    return { endpoint: parsed.toString(), error: null };
  } catch {
    return { endpoint: null, error: "VITE_MODAL_LIVE_WS_URL is not a valid WebSocket URL" };
  }
}

function boundedNumber(value: string | undefined, fallback: number, minimum: number, maximum: number): number {
  const parsed = Number(value ?? fallback);
  return Number.isFinite(parsed) && parsed >= minimum && parsed <= maximum ? parsed : fallback;
}
