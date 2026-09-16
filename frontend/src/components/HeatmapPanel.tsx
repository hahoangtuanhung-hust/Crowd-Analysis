import { useEffect, useRef } from "react";
import { Info } from "lucide-react";

import type { HeatmapData, HeatmapMetric, TimeWindow } from "../types";

interface HeatmapPanelProps {
  data: HeatmapData | null;
  metric: HeatmapMetric;
  window: TimeWindow;
  onMetricChange: (metric: HeatmapMetric) => void;
  onWindowChange: (window: TimeWindow) => void;
}

export function HeatmapPanel({ data, metric, window, onMetricChange, onWindowChange }: HeatmapPanelProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !data) return;
    const context = canvas.getContext("2d");
    if (!context) return;
    canvas.width = data.grid_width;
    canvas.height = data.grid_height;
    const image = context.createImageData(data.grid_width, data.grid_height);
    const maximum = Math.max(data.max, 1e-9);
    for (let row = 0; row < data.grid_height; row += 1) {
      for (let column = 0; column < data.grid_width; column += 1) {
        const offset = (row * data.grid_width + column) * 4;
        const [red, green, blue] = heatColor(data.values[row]?.[column] / maximum || 0);
        image.data[offset] = red;
        image.data[offset + 1] = green;
        image.data[offset + 2] = blue;
        image.data[offset + 3] = 255;
      }
    }
    context.putImageData(image, 0, 0);
  }, [data]);

  return (
    <section className="panel spatial-panel">
      <header className="panel-header split-header">
        <div>
          <p className="eyebrow">Spatial Analysis · Heatmap</p>
          <h2>{metric === "occupancy" ? "Occupancy Heatmap (Mật độ đứng yên)" : "Movement Heatmap (Vận tốc di chuyển)"}</h2>
        </div>
        <div className="segmented compact" aria-label="Heatmap metric">
          <button type="button" className={metric === "occupancy" ? "selected" : ""} onClick={() => onMetricChange("occupancy")}>Occupancy</button>
          <button type="button" className={metric === "movement" ? "selected" : ""} onClick={() => onMetricChange("movement")}>Movement</button>
        </div>
      </header>
      <div className="heatmap-wrap">
        <canvas ref={canvasRef} aria-label={`${metric} heatmap`} />
        {!data || data.max === 0 ? <span className="canvas-empty">No spatial samples</span> : null}
      </div>
      <footer className="heatmap-footer">
        <div className="segmented compact" aria-label="Time window">
          {(["current", "1m", "5m", "entire"] as TimeWindow[]).map((item) => (
            <button type="button" key={item} className={window === item ? "selected" : ""} onClick={() => onWindowChange(item)}>{item}</button>
          ))}
        </div>
        <div className="heat-legend"><span>Low</span><i /><span>High</span></div>
        {data?.calibration_required && <span className="relative-badge" title="Pixel-space result; calibrate for ground-plane analysis"><Info size={14} /> Relative</span>}
      </footer>
    </section>
  );
}

function heatColor(value: number): [number, number, number] {
  const stops: [number, number, number][] = [
    [20, 24, 34], [35, 83, 145], [37, 169, 184], [115, 202, 92], [245, 194, 65], [221, 72, 58]
  ];
  const scaled = Math.max(0, Math.min(1, value)) * (stops.length - 1);
  const index = Math.min(stops.length - 2, Math.floor(scaled));
  const amount = scaled - index;
  return stops[index].map((channel, channelIndex) => Math.round(channel + (stops[index + 1][channelIndex] - channel) * amount)) as [number, number, number];
}
