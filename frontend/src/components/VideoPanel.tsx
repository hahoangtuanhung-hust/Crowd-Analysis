import { useEffect, useRef, useState } from "react";
import { Crosshair, MapPinned, VideoOff } from "lucide-react";

import type { ModalLiveMetadata, OverlaySettings, RuntimeInfo, SessionSnapshot, SessionStatus } from "../types";
import { Toggle } from "./Toggle";

interface VideoPanelProps {
  session: SessionSnapshot | null;
  frameUrl?: string;
  overlay: OverlaySettings;
  onOverlayChange: (next: OverlaySettings) => void;
  onCalibrate: () => void;
  onZones: () => void;
  editable?: boolean;
  runtime: RuntimeInfo;
  modalMetadata?: ModalLiveMetadata;
  maxPaths?: number;
  appliedMaxPaths?: number;
  onMaxPathsChange?: (value: number) => void;
  commonPathOnly?: boolean;
}

/**
 * Hiển thị video live stream bằng MJPEG khi session đang running/starting,
 * hoặc hiện frame tĩnh cuối cùng khi session completed/stopped/error,
 * hoặc placeholder khi chưa có stream.
 */
export function VideoPanel({ session, frameUrl, overlay, onOverlayChange, onCalibrate, onZones, editable = true, runtime, modalMetadata, maxPaths, appliedMaxPaths, onMaxPathsChange, commonPathOnly = false }: VideoPanelProps) {
  const status: SessionStatus | undefined = session?.status;
  const isLive = status === "running" || status === "starting";

  // Khi chuyển sang live, dùng MJPEG stream; khi dừng, giữ frame tĩnh cuối cùng
  const [mjpegUrl, setMjpegUrl] = useState<string | null>(null);
  const imgRef = useRef<HTMLImageElement>(null);

  useEffect(() => {
    if (isLive) {
      // Thêm cache-buster để browser không dùng cache khi reconnect
      setMjpegUrl(`/api/stream.mjpg?t=${Date.now()}`);
    } else {
      // Khi session dừng, xóa MJPEG URL để hiện frame tĩnh cuối
      setMjpegUrl(null);
    }
  }, [isLive]);

  // URL hiển thị: ưu tiên MJPEG stream khi live, fallback sang frame tĩnh
  const displayUrl = runtime.mode === "modal-live" ? frameUrl : mjpegUrl ?? frameUrl;

  function update(key: keyof OverlaySettings, value: boolean) {
    onOverlayChange({ ...overlay, [key]: value });
  }

  return (
    <section className="panel video-panel">
      <header className="panel-header video-header">
        <div>
          <p className="eyebrow">Camera</p>
          <h2>{session?.camera_id ?? "Camera 01"}</h2>
        </div>
        <div className="video-tools">
          {maxPaths !== undefined && (
            <label className="path-limit">Số đường hiển thị
              <select aria-label="Số đường hiển thị" data-testid="path-limit" data-applied={appliedMaxPaths} value={maxPaths}
                onChange={(event) => onMaxPathsChange?.(Number(event.target.value))}>
                {[1, 2, 3, 4, 5].map((count) => <option value={count} key={count}>{count}</option>)}
              </select>
              {maxPaths !== appliedMaxPaths && <span aria-live="polite">Đang áp dụng</span>}
            </label>
          )}
          {isLive && (
            <span className={runtime.mode === "replay" ? "live-badge replay-badge" : "live-badge"} aria-label={runtime.mode === "replay" ? "Cache replay" : "Live streaming"}>
              {runtime.mode !== "replay" && <span className="live-pulse" />}
              {runtime.mode === "replay" ? "REPLAY" : runtime.mode === "modal-live" ? "MODAL GPU" : "LIVE"}
            </span>
          )}
          <button className="icon-command" type="button" onClick={onCalibrate} disabled={!displayUrl || !editable} title="Perspective calibration">
            <Crosshair size={17} aria-hidden="true" /> Calibrate
          </button>
          {!commonPathOnly && (
            <button className="icon-command" type="button" onClick={onZones} disabled={!displayUrl || !editable} title="Edit zones">
              <MapPinned size={17} aria-hidden="true" /> Zones
            </button>
          )}
        </div>
      </header>
      <div className="video-stage">
        {displayUrl ? (
          <>
            <img
              ref={imgRef}
              src={displayUrl}
              alt={runtime.mode === "replay" ? "Cached crowd tracking replay" : "Live crowd tracking output"}
            />
            {runtime.mode === "replay" && session && (
              <div className="replay-frame-meta" data-testid="replay-frame-meta">
                epoch {session.stream_epoch.slice(0, 8)} · frame {session.frame_id} · t={session.media_timestamp_s.toFixed(3)}s
              </div>
            )}
            {runtime.mode === "modal-live" && modalMetadata && (
              <>
                <div className="dominant-flow-status" data-testid="dominant-flow-status">
                  <strong>Common Paths</strong>
                  <span>
                    {modalMetadata.path_state === "active"
                      ? `${modalMetadata.active_count} track quan sát`
                      : modalMetadata.path_state === "confirming"
                        ? `Đang xác nhận ${modalMetadata.challenger_direction ?? "dòng mới"} · ${modalMetadata.challenger_count} người`
                        : "Đang tích lũy, chưa đủ dữ liệu"}
                  </span>
                  {modalMetadata.challenger_direction && modalMetadata.direction && (
                    <small>
                      Challenger {modalMetadata.challenger_direction}: {modalMetadata.challenger_count} · xác nhận {modalMetadata.confirmation_elapsed_s.toFixed(1)}/{modalMetadata.confirmation_required_s.toFixed(1)}s
                    </small>
                  )}
                </div>
                <div className="replay-frame-meta modal-frame-meta" data-testid="modal-frame-meta">
                  {modalMetadata.source_name} · frame {modalMetadata.frame_id} · t={modalMetadata.media_time_s.toFixed(2)}s · {modalMetadata.path_state} · {modalMetadata.processing_fps.toFixed(1)} FPS · drop {modalMetadata.dropped_input_frames}
                </div>
              </>
            )}
          </>
        ) : (
          <div className="empty-video">
            <VideoOff size={30} aria-hidden="true" />
            <span>{runtime.mode === "modal-live" && isLive ? "Đang kết nối và tích lũy dữ liệu..." : "No active stream"}</span>
          </div>
        )}
      </div>
      <div className="overlay-controls" aria-label="Video overlays">
        <Toggle label="Tracking Points" checked={overlay.points} disabled={!session || !editable} onChange={(value) => update("points", value)} />
        <Toggle label="Common Path" checked={overlay.active_paths} disabled={!session || !editable} onChange={(value) => update("active_paths", value)} />
        <Toggle label="Direction Arrows" checked={overlay.direction_arrows} disabled={!session || !overlay.active_paths || !editable} onChange={(value) => update("direction_arrows", value)} />
        {!commonPathOnly && <Toggle label="Zones" checked={overlay.zones} disabled={!session || !editable} onChange={(value) => update("zones", value)} />}
        <Toggle label="Track IDs" checked={overlay.track_ids} disabled={!session || !editable} onChange={(value) => update("track_ids", value)} />
        <Toggle label="Individual Paths (debug)" checked={overlay.trajectory_tails} disabled={!session || !editable} onChange={(value) => update("trajectory_tails", value)} />
        {!commonPathOnly && <Toggle label="Heatmap" checked={overlay.heatmap} disabled={!session || !editable} onChange={(value) => update("heatmap", value)} />}
        {!commonPathOnly && <Toggle label="Debug Grid" checked={overlay.grid} disabled={!session || !editable} onChange={(value) => update("grid", value)} />}
        {!commonPathOnly && <Toggle label="Debug Edge Flow" checked={overlay.edge_flows} disabled={!session || !editable} onChange={(value) => update("edge_flows", value)} />}
        <Toggle label="Debug Candidate" checked={overlay.candidate_paths} disabled={!session || !editable} onChange={(value) => update("candidate_paths", value)} />
        <Toggle label="Metrics" checked={overlay.debug_metrics} disabled={!session || !editable} onChange={(value) => update("debug_metrics", value)} />
      </div>
    </section>
  );
}
