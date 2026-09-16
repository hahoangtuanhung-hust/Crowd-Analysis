import { Compass, MapPin, Route, Users } from "lucide-react";

import type { PopularPath } from "../types";

export function PathsPanel({ paths }: { paths: PopularPath[] }) {
  return (
    <section className="panel paths-panel">
      <header className="panel-header split-header">
        <div>
          <p className="eyebrow">Ranked routes & Clusters</p>
          <h2>Top paths & Cụm đứng yên</h2>
        </div>
        <span className="badge-pill">DD-CRP Algorithm</span>
      </header>
      {paths.length ? (
        <ol className="path-list">
          {paths.map((path, index) => {
            const isStationary = path.kind === "ddcrp_stationary" || path.label.includes("Quầy vé") || path.label.includes("Xếp hàng");
            return (
              <li key={path.path_id} className={isStationary ? "stationary-item" : ""}>
                <span className="path-rank">{String(index + 1).padStart(2, "0")}</span>
                <div className="path-content">
                  <div className="path-line">
                    <div className="path-title-wrap">
                      {isStationary ? (
                        <span className="path-tag amber"><MapPin size={12} /> Đứng yên</span>
                      ) : (
                        <span className="path-tag blue"><Compass size={12} /> Lộ trình</span>
                      )}
                      <strong title={path.label}>{path.label}</strong>
                    </div>
                    <span className="path-count-badge">
                      <Users size={12} /> {path.count} lượt
                    </span>
                  </div>
                  <div className="bar-track">
                    <span
                      style={{
                        width: `${Math.max(3, path.percentage)}%`,
                        background: isStationary ? "var(--amber)" : "var(--blue)"
                      }}
                    />
                  </div>
                  <small>
                    {path.percentage.toFixed(1)}% tổng số người quan sát được
                  </small>
                </div>
              </li>
            );
          })}
        </ol>
      ) : (
        <div className="panel-empty">
          <Route size={22} aria-hidden="true" />
          <span>Đang theo dõi và gom nhóm các luồng di chuyển...</span>
        </div>
      )}
    </section>
  );
}
