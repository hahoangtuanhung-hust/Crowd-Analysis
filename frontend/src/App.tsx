import { lazy, Suspense, useEffect, useState } from "react";
import { CheckCircle2, CircleAlert, Radio, Server, X } from "lucide-react";

import { updateOverlay } from "./api";
import { FlowField } from "./components/FlowField";
import { HeatmapPanel } from "./components/HeatmapPanel";
import { MetricStrip } from "./components/MetricStrip";
import { PathsPanel } from "./components/PathsPanel";
import { SourceControls } from "./components/SourceControls";
import { SpatialEditor } from "./components/SpatialEditor";
import { VideoPanel } from "./components/VideoPanel";
import { ZonesPanel } from "./components/ZonesPanel";
import { useLiveData } from "./hooks/useLiveData";
import { useSpatialData } from "./hooks/useSpatialData";
import type {
  HeatmapMetric,
  OverlaySettings,
  TimeWindow,
} from "./types";

const TimelinePanel = lazy(() => import("./components/TimelinePanel").then((module) => ({ default: module.TimelinePanel })));

const DEFAULT_OVERLAY: OverlaySettings = {
  detection: true,
  tracking: true,
  trajectory: true,
  heatmap: false,
  zones: true,
  popular_paths: true,
};

export default function App() {
  const { data, connected } = useLiveData();
  const [metric, setMetric] = useState<HeatmapMetric>("occupancy");
  const [timeWindow, setTimeWindow] = useState<TimeWindow>("current");
  const [overlay, setOverlay] = useState<OverlaySettings>(DEFAULT_OVERLAY);
  const [editor, setEditor] = useState<"calibration" | "zones" | null>(null);
  const [notice, setNotice] = useState("");

  const status = data.session?.status ?? "idle";

  // Khi phân tích hoàn tất (completed), tự động chuyển timeWindow sang 'entire' để hiển thị toàn bộ kết quả
  useEffect(() => {
    if (status === "completed") {
      setTimeWindow("entire");
    }
  }, [status]);

  const { heatmap, flow } = useSpatialData(metric, timeWindow, data.session?.frame_version ?? 0);

  const frameUrl = data.session && data.session.frame_version > 0 ? data.frame_url : undefined;

  useEffect(() => {
    if (!notice) return;
    const timer = window.setTimeout(() => setNotice(""), 3500);
    return () => window.clearTimeout(timer);
  }, [notice]);

  async function changeOverlay(next: OverlaySettings) {
    const previous = overlay;
    setOverlay(next);
    try {
      const { popular_paths: _popularPaths, ...serverOverlay } = next;
      await updateOverlay(serverOverlay);
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
            <p>Live Video & Realtime Crowd Movement Intelligence</p>
          </div>
        </div>
        <div className="system-status">
          <span className={connected ? "status-dot connected" : "status-dot"} />
          <div>
            <strong>{connected ? "Modal Backend Connected" : "Connecting Backend..."}</strong>
            <span><Server size={13} /> Cloud GPU Serverless</span>
          </div>
          <span className={`session-pill ${status}`}>
            <Radio size={13} />{status}
          </span>
        </div>
      </header>

      <main>
        {/* Thanh điều khiển nguồn phát Live Source */}
        <SourceControls status={data.session?.status} />

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
                lưu lượng đỉnh <strong>{data.summary.peak_crowd_count}</strong> người cùng lúc. 
                Các thuật toán <strong>DD-CRP</strong> đã gom nhóm thành công các tuyến lộ trình phổ biến nhất 
                và các cụm người đứng yên mua vé. Toàn bộ bản đồ nhiệt (Occupancy & Movement) đã sẵn sàng xem bên dưới.
              </p>
            </div>
          </div>
        )}

        {status === "running" && (
          <div className="running-banner" role="status">
            <span className="pulse-dot" />
            <span>
              <strong>Đang phân tích realtime...</strong> Khung hình #{data.session?.frame_version ?? 0} · 
              FPS: {data.metrics.processing_fps.toFixed(1)} · Đang tracking {data.summary.current_crowd_count} người
            </span>
          </div>
        )}

        {data.session?.error && (
          <div className="error-banner" role="alert">
            <CircleAlert size={17} />{data.session.error}
          </div>
        )}

        {/* Thông số hệ thống và thống kê đám đông */}
        <MetricStrip summary={data.summary} metrics={data.metrics} />

        {/* Khung chính: Video trực tiếp & Bản đồ nhiệt Heatmap */}
        <div className="primary-grid">
          <VideoPanel
            session={data.session}
            frameUrl={frameUrl}
            overlay={overlay}
            onOverlayChange={changeOverlay}
            onCalibrate={() => setEditor("calibration")}
            onZones={() => setEditor("zones")}
            editable={true}
          />

          <HeatmapPanel
            data={heatmap}
            metric={metric}
            window={timeWindow}
            onMetricChange={setMetric}
            onWindowChange={setTimeWindow}
          />
        </div>

        {/* Khung thứ 2: Pathmap (Trường vector Flow) & Top paths (DD-CRP) */}
        <div className="secondary-grid">
          <FlowField data={flow} />
          <PathsPanel paths={overlay.popular_paths ? data.paths : []} />
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
