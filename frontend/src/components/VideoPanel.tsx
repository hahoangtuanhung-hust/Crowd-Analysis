import { Crosshair, MapPinned, VideoOff } from "lucide-react";

import type { OverlaySettings, SessionSnapshot } from "../types";
import { Toggle } from "./Toggle";

interface VideoPanelProps {
  session: SessionSnapshot | null;
  frameUrl?: string;
  overlay: OverlaySettings;
  onOverlayChange: (next: OverlaySettings) => void;
  onCalibrate: () => void;
  onZones: () => void;
  editable?: boolean;
}

export function VideoPanel({ session, frameUrl, overlay, onOverlayChange, onCalibrate, onZones, editable = true }: VideoPanelProps) {
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
          <button className="icon-command" type="button" onClick={onCalibrate} disabled={!frameUrl || !editable} title="Perspective calibration">
            <Crosshair size={17} aria-hidden="true" /> Calibrate
          </button>
          <button className="icon-command" type="button" onClick={onZones} disabled={!frameUrl || !editable} title="Edit zones">
            <MapPinned size={17} aria-hidden="true" /> Zones
          </button>
        </div>
      </header>
      <div className="video-stage">
        {frameUrl ? (
          <img src={frameUrl} alt="Live crowd tracking output" />
        ) : (
          <div className="empty-video"><VideoOff size={30} aria-hidden="true" /><span>No active stream</span></div>
        )}
      </div>
      <div className="overlay-controls" aria-label="Video overlays">
        <Toggle label="Detection" checked={overlay.detection} disabled={!session} onChange={(value) => update("detection", value)} />
        <Toggle label="Tracking" checked={overlay.tracking} disabled={!session} onChange={(value) => update("tracking", value)} />
        <Toggle label="Trajectory" checked={overlay.trajectory} disabled={!session} onChange={(value) => update("trajectory", value)} />
        <Toggle label="Heatmap" checked={overlay.heatmap} disabled={!session} onChange={(value) => update("heatmap", value)} />
        <Toggle label="Zones" checked={overlay.zones} disabled={!session} onChange={(value) => update("zones", value)} />
        <Toggle label="Popular paths" checked={overlay.popular_paths} disabled={!session} onChange={(value) => update("popular_paths", value)} />
      </div>
    </section>
  );
}
