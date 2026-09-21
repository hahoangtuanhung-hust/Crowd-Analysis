import { Activity, Route, Users } from "lucide-react";

import type { CommonPath } from "../types";

export function PathsPanel({ paths }: { paths: CommonPath[] }) {
  return (
    <section className="panel paths-panel">
      <header className="panel-header split-header">
        <div>
          <p className="eyebrow">Stable directed routes</p>
          <h2>Common Paths</h2>
        </div>
        <span className="badge-pill">Grid Flow</span>
      </header>
      {paths.length ? (
        <ol className="path-list">
          {paths.map((path, index) => {
            return (
              <li key={`${path.path_id}-${path.state}`}>
                <span className="path-rank">{String(index + 1).padStart(2, "0")}</span>
                <div className="path-content">
                  <div className="path-line">
                    <div className="path-title-wrap">
                      <span className={`path-tag ${path.state}`}><Activity size={12} /> {path.state}</span>
                      <strong title={path.direction}>{path.origin_zone} → {path.destination_zone}</strong>
                    </div>
                    <span className="path-count-badge">
                      <Users size={12} /> {path.unique_tracks_long} IDs
                    </span>
                  </div>
                  <div className="bar-track">
                    <span
                      style={{
                        width: `${Math.max(3, path.score * 100)}%`
                      }}
                    />
                  </div>
                  <small>
                    Short {path.unique_tracks_short} · Score {path.score.toFixed(2)} · Confidence {path.confidence.toFixed(2)}
                  </small>
                </div>
              </li>
            );
          })}
        </ol>
      ) : (
        <div className="panel-empty">
          <Route size={22} aria-hidden="true" />
          <span>Waiting for a confirmed common path</span>
        </div>
      )}
    </section>
  );
}
