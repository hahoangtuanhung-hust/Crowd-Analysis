import { lazy, Suspense, useEffect, useState } from "react";
import { CheckCircle2, CircleAlert, Radio, Server, X } from "lucide-react";

import { getRuntimeInfo, getVisualization, patchVisualization } from "./api";
import { FlowField } from "./components/FlowField";
import { HeatmapPanel } from "./components/HeatmapPanel";
import { MetricStrip } from "./components/MetricStrip";
import { PathsPanel } from "./components/PathsPanel";
import { SourceControls } from "./components/SourceControls";
import { SpatialEditor } from "./components/SpatialEditor";
import { VideoPanel } from "./components/VideoPanel";
import { ZonesPanel } from "./components/ZonesPanel";
import { useLiveData } from "./hooks/useLiveData";
import { useModalLive } from "./hooks/useModalLive";
import { useSpatialData } from "./hooks/useSpatialData";
import type {
  HeatmapMetric,
  OverlaySettings,
  RuntimeInfo,
  TimeWindow,
  VisualizationSettings,
} from "./types";

const TimelinePanel = lazy(() => import("./components/TimelinePanel").then((module) => ({ default: module.TimelinePanel })));

const DEFAULT_OVERLAY: OverlaySettings = {
  detection: false,
  tracking: true,
  trajectory: false,
  heatmap: false,
  zones: false,
  points: true,
  track_ids: false,
  trajectory_tails: false,
  grid: false,
  edge_flows: false,
  candidate_paths: false,
  active_paths: true,
  direction_arrows: true,
  debug_metrics: true,
  popular_paths: true,
};

const DEFAULT_RUNTIME: RuntimeInfo = {
  mode: "live",
  source_mode: "configured_input",
  inference_executed: null,
  detector_calls: null,
  cache_reads: null,
};

export default function App() {
  const modalLive = useModalLive();
  const modalMode = modalLive.requested;
  const localLive = useLiveData(!modalMode);
  const data = modalMode ? modalLive.data : localLive.data;
  const connected = modalMode ? modalLive.connected : localLive.connected;
  const [metric, setMetric] = useState<HeatmapMetric>("occupancy");
  const [timeWindow, setTimeWindow] = useState<TimeWindow>("current");
  const [overlay, setOverlay] = useState<OverlaySettings>(DEFAULT_OVERLAY);
  const [editor, setEditor] = useState<"calibration" | "zones" | null>(null);
  const [notice, setNotice] = useState("");
  const [runtime, setRuntime] = useState<RuntimeInfo>(DEFAULT_RUNTIME);

  const status = modalMode ? modalLive.status : data.session?.status ?? "idle";
  const hasConfirmedCommonPath = data.common_paths.some((path) => path.state === "active" || path.state === "cooling");
  const terminal = status === "completed" || status === "stopped";
  const statusConnected = connected || terminal;
  const connectionLabel = terminal
    ? (status === "completed" ? "Modal Session Completed" : "Modal Session Stopped")
    : connected
      ? (runtime.mode === "replay" ? "Replay Backend Connected" : runtime.mode === "modal-live" ? "Modal GPU Connected" : "Backend Connected")
      : "Connecting Backend...";

  useEffect(() => {
    if (modalMode) {
      setRuntime({
        mode: "modal-live",
        source_mode: "modal-volume-live-inference",
        inference_executed: true,
        detector_calls: null,
        cache_reads: 0
      });
      return;
    }
    let active = true;
    getVisualization()
      .then((settings) => {
        if (active) setOverlay(overlayFromVisualization(settings));
      })
      .catch((cause) => {
        if (active) {
          setNotice(cause instanceof Error ? cause.message : "Unable to load visualization settings");
        }
      });
    getRuntimeInfo()
      .then((info) => {
        if (active) setRuntime(info);
      })
      .catch((cause) => {
        if (active) {
          setNotice(cause instanceof Error ? cause.message : "Unable to load runtime mode");
        }
      });
    return () => { active = false; };
  }, [modalMode]);

  // Khi phân tích hoàn tất (completed), tự động chuyển timeWindow sang 'entire' để hiển thị toàn bộ kết quả
  useEffect(() => {
    if (status === "completed") {
      setTimeWindow("entire");
    }
  }, [status]);

  const { heatmap, flow } = useSpatialData(metric, timeWindow, data.session?.frame_version ?? 0, !modalMode);

  const frameUrl = modalMode
    ? modalLive.frameUrl
    : data.session && data.session.frame_version > 0 ? data.frame_url : undefined;

  useEffect(() => {
    if (!notice) return;
    const timer = window.setTimeout(() => setNotice(""), 3500);
    return () => window.clearTimeout(timer);
  }, [notice]);

  async function changeOverlay(next: OverlaySettings) {
    if (modalMode) return;
    const previous = overlay;
    setOverlay(next);
    try {
      await patchVisualization(visualizationFromOverlay(next));
    } catch (cause) {
      setOverlay(previous);
      setNotice(cause instanceof Error ? cause.message : "Unable to update overlays");
    }
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand-block">
          <span className="brand-mark" aria-hidden="true">CA</span>
          <div>
            <h1>Crowd Analysis</h1>
            <p>{runtime.mode === "replay" ? "Cached Video Replay & Crowd Movement Analysis" : runtime.mode === "modal-live" ? "Shibuya · Causal Common Path · Modal GPU" : "Live Video & Realtime Crowd Movement Intelligence"}</p>
          </div>
        </div>
        <div className="system-status">
          <span className={statusConnected ? "status-dot connected" : "status-dot"} />
          <div>
            <strong>{connectionLabel}</strong>
            <span><Server size={13} /> {runtime.mode === "replay" ? "Cache-backed · no inference" : runtime.mode === "modal-live" ? "Remote WebSocket · realtime_pts" : "Configured inference runtime"}</span>
          </div>
          <span className={`session-pill ${status}`}>
            <Radio size={13} />{status}
          </span>
        </div>
      </header>

      <main>
        {/* Thanh điều khiển nguồn phát Live Source */}
        <SourceControls
          status={status}
          replayMode={runtime.mode === "replay"}
          modalLive={modalMode ? {
            connected: modalLive.connected,
            sourceName: modalLive.metadata?.source_name ?? "data-shibuya-test.mp4",
            onStart: modalLive.start,
            onStop: modalLive.stop
          } : undefined}
        />

        {/* Thông báo phân tích hoàn tất hoặc đang chạy */}
        {status === "completed" && (
          <div className="completion-banner" role="status">
            <div className="completion-icon">
              <CheckCircle2 size={24} />
            </div>
            <div className="completion-text">
              <h3>Phân tích video hoàn tất!</h3>
              <p>
                Đã nhận diện tổng cộng <strong>{data.summary.unique_track_count}</strong> người, 
                lưu lượng đỉnh <strong>{data.summary.peak_crowd_count}</strong> người cùng lúc.{" "}
                {hasConfirmedCommonPath
                  ? "Common Path và bản đồ nhiệt đã sẵn sàng xem bên dưới."
                  : "Bản đồ nhiệt đã sẵn sàng; chưa có Common Path được xác nhận tại thời điểm kết thúc."}
              </p>
            </div>
          </div>
        )}

        {status === "running" && (
          <div className="running-banner" role="status">
            <span className="pulse-dot" />
            <span>
              <strong>{runtime.mode === "replay" ? "Đang phát replay cache..." : runtime.mode === "modal-live" ? "Modal GPU đang inference và stream trực tiếp..." : "Đang phân tích realtime..."}</strong> Khung hình #{data.session?.frame_id ?? 0} ·
              FPS: {data.metrics.processing_fps.toFixed(1)} · Đang tracking {data.summary.current_crowd_count} người
            </span>
          </div>
        )}

        {(data.session?.error || modalLive.error) && (
          <div className="error-banner" role="alert">
            <CircleAlert size={17} />{data.session?.error || modalLive.error}
          </div>
        )}

        {/* Thông số hệ thống và thống kê đám đông */}
        <MetricStrip summary={data.summary} metrics={data.metrics} replayMode={runtime.mode === "replay"} />

        {/* Khung chính: Video trực tiếp & Bản đồ nhiệt Heatmap */}
        <div className="primary-grid">
          <VideoPanel
            session={data.session}
            frameUrl={frameUrl}
            overlay={overlay}
            onOverlayChange={changeOverlay}
            onCalibrate={() => setEditor("calibration")}
            onZones={() => setEditor("zones")}
            editable={!modalMode}
            runtime={runtime}
            modalMetadata={modalLive.metadata}
            maxPaths={modalMode ? modalLive.maxPaths : undefined}
            appliedMaxPaths={modalLive.appliedMaxPaths}
            onMaxPathsChange={modalLive.changeMaxPaths}
          />

          <HeatmapPanel
            data={heatmap}
            metric={metric}
            window={timeWindow}
            onMetricChange={setMetric}
            onWindowChange={setTimeWindow}
          />
        </div>

        {/* Khung thứ 2: Pathmap (Trường vector Flow) & Top Common Paths */}
        <div className="secondary-grid">
          <FlowField data={flow} />
          <PathsPanel paths={overlay.popular_paths ? data.common_paths : []} />
        </div>

        {/* Khung thứ 3: Biểu đồ số người theo thời gian & Phân bổ Zone */}
        <div className="secondary-grid lower-grid">
          <Suspense fallback={<section className="panel timeline-panel timeline-loading" aria-label="Loading crowd timeline" />}>
            <TimelinePanel timeline={data.timeline} />
          </Suspense>
          <ZonesPanel zones={data.zones} flows={data.zone_flows ?? []} />
        </div>
      </main>

      {editor && frameUrl && (
        <SpatialEditor
          mode={editor}
          frameUrl={frameUrl}
          onClose={() => setEditor(null)}
          onSaved={setNotice}
        />
      )}

      {notice && (
        <div className="toast" role="status">
          <span>{notice}</span>
          <button type="button" onClick={() => setNotice("")} title="Dismiss">
            <X size={15} />
            <span className="visually-hidden">Dismiss</span>
          </button>
        </div>
      )}
    </div>
  );
}

function overlayFromVisualization(settings: VisualizationSettings): OverlaySettings {
  return {
    detection: settings.show_bounding_boxes,
    tracking: true,
    trajectory: settings.show_individual_trajectories,
    heatmap: settings.show_heatmap,
    zones: settings.show_zones,
    points: settings.show_tracking_points,
    track_ids: settings.show_track_ids,
    trajectory_tails: settings.show_individual_trajectories,
    grid: settings.show_grid_debug,
    edge_flows: settings.show_edge_flow_debug,
    candidate_paths: settings.show_candidate_path,
    active_paths: settings.show_common_path,
    direction_arrows: settings.show_direction_arrows,
    debug_metrics: settings.show_metrics,
    popular_paths: true
  };
}

function visualizationFromOverlay(settings: OverlaySettings) {
  return {
    show_bounding_boxes: false,
    show_track_ids: settings.track_ids,
    show_tracking_points: settings.points,
    show_individual_trajectories: settings.trajectory_tails,
    show_common_path: settings.active_paths,
    show_direction_arrows: settings.direction_arrows,
    show_candidate_path: settings.candidate_paths,
    show_zones: settings.zones,
    show_heatmap: settings.heatmap,
    show_grid_debug: settings.grid,
    show_edge_flow_debug: settings.edge_flows,
    show_metrics: settings.debug_metrics
  };
}
