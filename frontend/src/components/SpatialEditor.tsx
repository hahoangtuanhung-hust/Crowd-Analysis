import { useEffect, useRef, useState } from "react";
import { Check, Plus, RotateCcw, Trash2, X } from "lucide-react";

import { setCalibration, setZones } from "../api";
import type { PolygonZone } from "../types";

type EditorMode = "calibration" | "zones";

interface SpatialEditorProps {
  mode: EditorMode;
  frameUrl: string;
  onClose: () => void;
  onSaved: (message: string) => void;
}

export function SpatialEditor({ mode, frameUrl, onClose, onSaved }: SpatialEditorProps) {
  const imageRef = useRef<HTMLImageElement>(null);
  const [editorFrame] = useState(frameUrl);
  const [imageSize, setImageSize] = useState({ width: 1, height: 1 });
  const [points, setPoints] = useState<[number, number][]>([]);
  const [zones, setLocalZones] = useState<PolygonZone[]>([]);
  const [name, setName] = useState("Zone 1");
  const [groundWidth, setGroundWidth] = useState(20);
  const [groundHeight, setGroundHeight] = useState(12);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    function escape(event: KeyboardEvent) {
      if (event.key === "Escape") onClose();
    }
    window.addEventListener("keydown", escape);
    return () => window.removeEventListener("keydown", escape);
  }, [onClose]);

  function addPoint(event: React.MouseEvent<HTMLDivElement>) {
    const image = imageRef.current;
    if (!image || (mode === "calibration" && points.length >= 4)) return;
    const bounds = image.getBoundingClientRect();
    const scale = Math.min(bounds.width / image.naturalWidth, bounds.height / image.naturalHeight);
    const renderedWidth = image.naturalWidth * scale;
    const renderedHeight = image.naturalHeight * scale;
    const offsetX = (bounds.width - renderedWidth) / 2;
    const offsetY = (bounds.height - renderedHeight) / 2;
    const localX = event.clientX - bounds.left - offsetX;
    const localY = event.clientY - bounds.top - offsetY;
    if (localX < 0 || localY < 0 || localX > renderedWidth || localY > renderedHeight) return;
    const x = localX / scale;
    const y = localY / scale;
    setPoints((current) => [...current, [x, y]]);
  }

  function addZone() {
    if (points.length < 3) return;
    setLocalZones((current) => [
      ...current,
      { zone_id: `zone-${current.length + 1}`, name: name.trim() || `Zone ${current.length + 1}`, points }
    ]);
    setPoints([]);
    setName(`Zone ${zones.length + 2}`);
  }

  async function save() {
    setBusy(true);
    setError("");
    try {
      if (mode === "calibration") {
        if (points.length !== 4) throw new Error("Four calibration points are required");
        await setCalibration(points, groundWidth, groundHeight);
        onSaved("Ground-plane calibration applied");
      } else {
        const nextZones = points.length >= 3
          ? [...zones, { zone_id: `zone-${zones.length + 1}`, name: name.trim() || `Zone ${zones.length + 1}`, points }]
          : zones;
        if (!nextZones.length) throw new Error("Add at least one zone");
        await setZones(nextZones);
        onSaved(`${nextZones.length} zone${nextZones.length === 1 ? "" : "s"} applied`);
      }
      onClose();
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Unable to save");
    } finally {
      setBusy(false);
    }
  }

  const allPolygons = mode === "zones" ? zones : [];

  return (
    <div className="modal-backdrop" role="presentation" onMouseDown={(event) => event.target === event.currentTarget && onClose()}>
      <section className="spatial-editor" role="dialog" aria-modal="true" aria-labelledby="editor-title">
        <header className="editor-header">
          <div>
            <p className="eyebrow">Spatial setup</p>
            <h2 id="editor-title">{mode === "calibration" ? "Perspective calibration" : "Zone editor"}</h2>
          </div>
          <button className="icon-button" type="button" onClick={onClose} title="Close"><X size={19} /><span className="visually-hidden">Close</span></button>
        </header>

        <div className="editor-stage" onClick={addPoint}>
          <img ref={imageRef} src={editorFrame} alt="Camera frame for spatial setup" draggable={false} onLoad={(event) => setImageSize({ width: event.currentTarget.naturalWidth, height: event.currentTarget.naturalHeight })} />
          <svg viewBox={`0 0 ${imageSize.width} ${imageSize.height}`} preserveAspectRatio="xMidYMid meet" aria-hidden="true">
            {allPolygons.map((zone) => <polygon className="saved-polygon" key={zone.zone_id} points={zone.points.map((point) => point.join(",")).join(" ")} />)}
            {points.length > 1 && <polyline className="draft-polygon" points={points.map((point) => point.join(",")).join(" ")} />}
            {points.map(([x, y], index) => <g key={`${x}-${y}-${index}`}><circle cx={x} cy={y} r="7" /><text x={x + 11} y={y - 9}>{index + 1}</text></g>)}
          </svg>
        </div>

        <div className="editor-controls">
          {mode === "calibration" ? (
            <div className="dimension-fields">
              <label><span>Ground width (m)</span><input type="number" min="1" step="0.5" value={groundWidth} onChange={(event) => setGroundWidth(Number(event.target.value))} /></label>
              <label><span>Ground height (m)</span><input type="number" min="1" step="0.5" value={groundHeight} onChange={(event) => setGroundHeight(Number(event.target.value))} /></label>
            </div>
          ) : (
            <div className="zone-builder">
              <label><span>Zone name</span><input value={name} maxLength={128} onChange={(event) => setName(event.target.value)} /></label>
              <button className="button secondary" type="button" onClick={addZone} disabled={points.length < 3}><Plus size={17} /> Add zone</button>
            </div>
          )}
          <div className="point-status"><strong>{points.length}</strong> / {mode === "calibration" ? 4 : "3+"} points</div>
          <button className="icon-command" type="button" onClick={() => setPoints([])} disabled={!points.length} title="Clear draft"><RotateCcw size={17} /> Reset</button>
        </div>

        {mode === "zones" && zones.length > 0 && (
          <div className="zone-chips" aria-label="Draft zones">
            {zones.map((zone, index) => (
              <span key={zone.zone_id}>{zone.name}<button type="button" title={`Delete ${zone.name}`} onClick={() => setLocalZones((current) => current.filter((_, item) => item !== index))}><Trash2 size={14} /><span className="visually-hidden">Delete {zone.name}</span></button></span>
            ))}
          </div>
        )}
        {error && <p className="editor-error" role="alert">{error}</p>}
        <footer className="editor-actions">
          <button className="button secondary" type="button" onClick={onClose}>Cancel</button>
          <button className="button primary" type="button" onClick={save} disabled={busy || (mode === "calibration" && points.length !== 4)}><Check size={17} /> {busy ? "Saving" : "Apply"}</button>
        </footer>
      </section>
    </div>
  );
}
