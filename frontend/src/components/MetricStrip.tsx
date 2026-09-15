import { Activity, Gauge, UserRoundCheck, Users } from "lucide-react";

import type { CrowdSummary, PerformanceMetrics } from "../types";

interface MetricStripProps {
  summary: CrowdSummary;
  metrics: PerformanceMetrics;
}

export function MetricStrip({ summary, metrics }: MetricStripProps) {
  const items = [
    { label: "Current", value: summary.current_crowd_count, icon: Users },
    { label: "Average", value: summary.average_crowd_count.toFixed(1), icon: Activity },
    { label: "Peak", value: summary.peak_crowd_count, icon: Gauge },
    { label: "Unique tracks", value: summary.unique_track_count, icon: UserRoundCheck }
  ];
  return (
    <>
      <section className="metric-grid" aria-label="Crowd summary">
        {items.map(({ label, value, icon: Icon }) => (
          <article className="metric-tile" key={label}>
            <Icon size={19} aria-hidden="true" />
            <div><span>{label}</span><strong>{value}</strong></div>
          </article>
        ))}
      </section>
      <section className="performance-strip" aria-label="Processing performance">
        <PerformanceItem label="Processing" value={`${metrics.processing_fps.toFixed(1)} FPS`} />
        <PerformanceItem label="Inference" value={formatMs(metrics.inference_ms)} />
        <PerformanceItem label="Tracking" value={formatMs(metrics.tracking_ms)} />
        <PerformanceItem label="Analytics" value={formatMs(metrics.analytics_ms)} />
        <PerformanceItem label="p95 latency" value={formatMs(metrics.e2e_latency_ms_p95 ?? metrics.e2e_latency_ms)} />
        <PerformanceItem label="Queue" value={String(metrics.queue_size)} />
        <PerformanceItem label="Dropped" value={String(metrics.dropped_frames)} alert={metrics.dropped_frames > 0} />
        <PerformanceItem label="RAM" value={`${metrics.ram_mb.toFixed(0)} MB`} />
      </section>
    </>
  );
}

function PerformanceItem({ label, value, alert = false }: { label: string; value: string; alert?: boolean }) {
  return <div className={alert ? "performance-item alert" : "performance-item"}><span>{label}</span><strong>{value}</strong></div>;
}

function formatMs(value: number | null | undefined) {
  return value == null ? "--" : `${value.toFixed(value >= 100 ? 0 : 1)} ms`;
}
