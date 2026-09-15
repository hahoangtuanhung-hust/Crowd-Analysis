import { Route } from "lucide-react";

import type { PopularPath } from "../types";

export function PathsPanel({ paths }: { paths: PopularPath[] }) {
  return (
    <section className="panel paths-panel">
      <header className="panel-header"><div><p className="eyebrow">Ranked routes</p><h2>Top paths</h2></div></header>
      {paths.length ? (
        <ol className="path-list">
          {paths.map((path, index) => (
            <li key={path.path_id}>
              <span className="path-rank">{String(index + 1).padStart(2, "0")}</span>
              <div className="path-content">
                <div className="path-line"><strong title={path.label}>{path.label}</strong><span>{path.count}</span></div>
                <div className="bar-track"><span style={{ width: `${Math.max(2, path.percentage)}%` }} /></div>
                <small>{path.percentage.toFixed(1)}% of completed routes</small>
              </div>
            </li>
          ))}
        </ol>
      ) : (
        <div className="panel-empty"><Route size={22} aria-hidden="true" /><span>No completed paths</span></div>
      )}
    </section>
  );
}
