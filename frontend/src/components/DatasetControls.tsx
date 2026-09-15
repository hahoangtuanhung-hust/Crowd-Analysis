import { Database, Play, RotateCw } from "lucide-react";

import type { CoordinateMode, DatasetDuration, DatasetSource } from "../types";

interface DatasetControlsProps {
  source: DatasetSource;
  coordinateMode: CoordinateMode;
  duration: DatasetDuration;
  busy: boolean;
  onSourceChange: (source: DatasetSource) => void;
  onCoordinateModeChange: (mode: CoordinateMode) => void;
  onDurationChange: (duration: DatasetDuration) => void;
  onAnalyze: () => void;
}

export function DatasetControls({
  source,
  coordinateMode,
  duration,
  busy,
  onSourceChange,
  onCoordinateModeChange,
  onDurationChange,
  onAnalyze
}: DatasetControlsProps) {
  return (
    <section className="dataset-bar" aria-label="Grand Central analysis controls">
      <label className="dataset-picker">
        <Database size={16} aria-hidden="true" />
        <span className="visually-hidden">Dataset</span>
        <select value="grand-central" disabled>
          <option value="grand-central">Grand Central Station</option>
        </select>
      </label>
      <div className="segmented compact dataset-source" aria-label="Trajectory source">
        <button type="button" className={source === "ground_truth" ? "selected" : ""} onClick={() => onSourceChange("ground_truth")}>Ground truth</button>
        <button type="button" className={source === "prediction" ? "selected" : ""} onClick={() => onSourceChange("prediction")}>YOLO + ByteTrack</button>
      </div>
      <div className="segmented compact" aria-label="Coordinate mode">
        <button type="button" className={coordinateMode === "pixel_space" ? "selected" : ""} onClick={() => onCoordinateModeChange("pixel_space")}>Pixel space</button>
        <button type="button" className={coordinateMode === "ground_plane" ? "selected" : ""} onClick={() => onCoordinateModeChange("ground_plane")}>Ground plane</button>
      </div>
      <div className="segmented compact" aria-label="Analysis duration">
        {(["1m", "5m", "all"] as DatasetDuration[]).map((item) => (
          <button type="button" key={item} className={duration === item ? "selected" : ""} onClick={() => onDurationChange(item)}>{item}</button>
        ))}
      </div>
      <button className="button primary dataset-run" type="button" onClick={onAnalyze} disabled={busy}>
        {busy ? <RotateCw className="spin" size={16} aria-hidden="true" /> : <Play size={16} fill="currentColor" aria-hidden="true" />}
        {busy ? "Analyzing" : "Analyze"}
      </button>
    </section>
  );
}
