import { lazy, Suspense, useEffect, useMemo, useState } from "react";
import { CircleAlert, Database, Radio, Server, X } from "lucide-react";

import { analyzeGrandCentral, getGrandCentralResult, updateOverlay } from "./api";
import { DatasetControls } from "./components/DatasetControls";
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
  CoordinateMode,
  DatasetAnalyticsResult,
  DatasetDuration,
  DatasetSource,
  HeatmapData,
  HeatmapMetric,
  LiveSnapshot,
  OverlaySettings,
  PerformanceMetrics,
  TimeWindow
} from "./types";

const TimelinePanel = lazy(() => import("./components/TimelinePanel").then((module) => ({ default: module.TimelinePanel })));

const DEFAULT_OVERLAY: OverlaySettings = {
  detection: false,
  tracking: true,
  trajectory: true,
  heatmap: false,
  zones: true,
  popular_paths: true
};

const EMPTY_PERFORMANCE: PerformanceMetrics = {
  input_fps: 0, processing_fps: 0, decode_ms: null, inference_ms: null,
  tracking_ms: null, analytics_ms: null, render_ms: null, encoding_ms: null,
  e2e_latency_ms: null, queue_size: 0, dropped_frames: 0, cpu_percent: 0,
  ram_mb: 0, gpu_utilization: null, gpu_memory_mb: null
};

export default function App() {
  const { data, connected } = useLiveData();
  const [workspace, setWorkspace] = useState<"dataset" | "live">("dataset");
  const [datasetSource, setDatasetSource] = useState<DatasetSource>("ground_truth");
  const [coordinateMode, setCoordinateMode] = useState<CoordinateMode>("pixel_space");
  const [datasetDuration, setDatasetDuration] = useState<DatasetDuration>("all");
  const [datasetResult, setDatasetResult] = useState<DatasetAnalyticsResult | null>(null);
  const [datasetBusy, setDatasetBusy] = useState(false);
  const [datasetError, setDatasetError] = useState("");
  const [metric, setMetric] = useState<HeatmapMetric>("occupancy");
  const [timeWindow, setTimeWindow] = useState<TimeWindow>("current");
  const [overlay, setOverlay] = useState(DEFAULT_OVERLAY);
  const [editor, setEditor] = useState<"calibration" | "zones" | null>(null);
  const [notice, setNotice] = useState("");
  const { heatmap, flow } = useSpatialData(metric, timeWindow, data.session?.frame_version ?? 0);

  async function loadDatasetResult(analyze = false) {
    setDatasetBusy(true);
    setDatasetError("");
    try {
      const result = analyze
        ? await analyzeGrandCentral(datasetSource, coordinateMode, datasetDuration)
        : await getGrandCentralResult(datasetSource, coordinateMode, datasetDuration);
      setDatasetResult(result);
    } catch (cause) {
      if (!analyze) {
        try {
          const result = await analyzeGrandCentral(datasetSource, coordinateMode, datasetDuration);
          setDatasetResult(result);
          return;
        } catch (retryCause) {
          setDatasetError(retryCause instanceof Error ? retryCause.message : "Unable to load dataset result");
        }
      } else {
        setDatasetError(cause instanceof Error ? cause.message : "Unable to analyze dataset");
      }
    } finally {
      setDatasetBusy(false);
    }
  }

  useEffect(() => { void loadDatasetResult(false); }, []);

  const datasetSnapshot = useMemo<LiveSnapshot | null>(() => {
    if (!datasetResult) return null;
    const metrics: PerformanceMetrics = {
      ...EMPTY_PERFORMANCE,
      ...datasetResult.metrics,
      e2e_latency_ms: datasetResult.metrics.e2e_latency_ms ?? datasetResult.metrics.e2e_ms ?? null
    };
    return {
      type: "snapshot",
      session: {
        camera_id: "grand-central-01",
        source_kind: datasetResult.source,
        status: "completed",
        error: null,
        frame_version: 1,
        width: 1920,
        height: 1080
      },
      summary: datasetResult.summary,
      metrics,
      paths: datasetResult.paths,
      zones: datasetResult.zones,
      zone_flows: datasetResult.zone_flows,
      timeline: datasetResult.timeline
    };
  }, [datasetResult]);

  const activeData = workspace === "dataset" && datasetSnapshot ? datasetSnapshot : data;
  const datasetHeatmap = useMemo<HeatmapData | null>(() => {
    if (!datasetResult) return null;
    const values = metric === "occupancy" ? datasetResult.heatmap.occupancy : datasetResult.heatmap.movement;
    return {
      metric,
      window: "entire",
      spatial_mode: datasetResult.heatmap.spatial_mode,
      calibration_required: false,
      from_timestamp: datasetResult.heatmap.from_timestamp,
      to_timestamp: datasetResult.heatmap.to_timestamp,
      grid_width: datasetResult.heatmap.width,
      grid_height: datasetResult.heatmap.height,
      max: metric === "occupancy" ? datasetResult.heatmap.occupancy_max : datasetResult.heatmap.movement_max,
      sum: values.flat().reduce((total, value) => total + value, 0),
      values
    };
  }, [datasetResult, metric]);
  const activeHeatmap = workspace === "dataset" ? datasetHeatmap : heatmap;
  const activeFlow = workspace === "dataset" ? datasetResult?.flow ?? null : flow;
  const frameUrl = workspace === "dataset"
    ? datasetResult ? `/api/datasets/grand-central/preview.jpg?source=${datasetResult.source}` : undefined
    : data.session && data.session.frame_version > 0 ? data.frame_url : undefined;

  useEffect(() => {
    if (!notice) return;
    const timer = window.setTimeout(() => setNotice(""), 3500);
    return () => window.clearTimeout(timer);
  }, [notice]);

  async function changeOverlay(next: OverlaySettings) {
    const previous = overlay;
    setOverlay(next);
    if (workspace === "dataset") return;
    try {
      const { popular_paths: _popularPaths, ...serverOverlay } = next;
      await updateOverlay(serverOverlay);
    } catch (cause) {
      setOverlay(previous);
      setNotice(cause instanceof Error ? cause.message : "Unable to update overlays");
    }
  }

  const status = activeData.session?.status ?? "idle";

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand-block">
          <span className="brand-mark" aria-hidden="true">CA</span>
          <div><h1>Crowd Analysis</h1><p>Realtime movement intelligence</p></div>
        </div>
        <div className="system-status">
          <span className={connected ? "status-dot connected" : "status-dot"} />
          <div><strong>{connected ? "API connected" : "Reconnecting"}</strong><span><Server size={13} /> local processing</span></div>
          <span className={`session-pill ${status}`}><Radio size={13} />{status}</span>
        </div>
      </header>

      <main>
        <section className="workspace-switch" aria-label="Analysis workspace">
          <div className="segmented">
            <button type="button" className={workspace === "dataset" ? "selected" : ""} onClick={() => setWorkspace("dataset")}><Database size={16} />Grand Central</button>
            <button type="button" className={workspace === "live" ? "selected" : ""} onClick={() => setWorkspace("live")}><Radio size={16} />Live source</button>
          </div>
          {workspace === "dataset" && datasetResult && <span className="dataset-context">{datasetResult.source === "ground_truth" ? "Ground truth" : "Prediction"} · {datasetResult.coordinate_unit} · {datasetResult.duration}</span>}
        </section>
        {workspace === "dataset" ? (
          <DatasetControls
            source={datasetSource}
            coordinateMode={coordinateMode}
            duration={datasetDuration}
            busy={datasetBusy}
            onSourceChange={(source) => { setDatasetSource(source); setDatasetResult(null); setDatasetError(""); if (source === "prediction") setDatasetDuration("1m"); }}
            onCoordinateModeChange={(mode) => { setCoordinateMode(mode); setDatasetResult(null); setDatasetError(""); }}
            onDurationChange={(duration) => { setDatasetDuration(duration); setDatasetResult(null); setDatasetError(""); }}
            onAnalyze={() => void loadDatasetResult(true)}
          />
        ) : <SourceControls status={data.session?.status} />}
        {(datasetError || activeData.session?.error) && <div className="error-banner" role="alert"><CircleAlert size={17} />{datasetError || activeData.session?.error}</div>}
        <MetricStrip summary={activeData.summary} metrics={activeData.metrics} />

        <div className="primary-grid">
          <VideoPanel session={activeData.session} frameUrl={frameUrl} overlay={overlay} onOverlayChange={changeOverlay} onCalibrate={() => setEditor("calibration")} onZones={() => setEditor("zones")} editable={workspace === "live"} />
          <HeatmapPanel data={activeHeatmap} metric={metric} window={workspace === "dataset" ? "entire" : timeWindow} onMetricChange={setMetric} onWindowChange={setTimeWindow} />
        </div>
        <div className="secondary-grid">
          <FlowField data={activeFlow} />
          <PathsPanel paths={overlay.popular_paths ? activeData.paths : []} />
        </div>
        <div className="secondary-grid lower-grid">
          <Suspense fallback={<section className="panel timeline-panel timeline-loading" aria-label="Loading crowd timeline" />}>
            <TimelinePanel timeline={activeData.timeline} />
          </Suspense>
          <ZonesPanel zones={activeData.zones} flows={activeData.zone_flows ?? []} />
        </div>
      </main>

      {workspace === "live" && editor && frameUrl && <SpatialEditor mode={editor} frameUrl={frameUrl} onClose={() => setEditor(null)} onSaved={setNotice} />}
      {notice && <div className="toast" role="status"><span>{notice}</span><button type="button" onClick={() => setNotice("")} title="Dismiss"><X size={15} /><span className="visually-hidden">Dismiss</span></button></div>}
    </div>
  );
}
