import { Map } from "lucide-react";

import type { ZoneFlow, ZoneMetric } from "../types";

export function ZonesPanel({ zones, flows }: { zones: ZoneMetric[]; flows: ZoneFlow[] }) {
  return (
    <section className="panel zones-panel">
      <header className="panel-header"><div><p className="eyebrow">Regions of interest</p><h2>Zone analytics</h2></div></header>
      {zones.length ? (
        <div className="table-scroll">
          <table>
            <thead><tr><th>Zone</th><th>Now</th><th>Entries</th><th>Exits</th><th>Dwell</th></tr></thead>
            <tbody>{zones.map((zone) => (
              <tr key={zone.zone_id}>
                <th>{zone.name}</th><td>{zone.current_people}</td><td>{zone.entry_count}</td><td>{zone.exit_count}</td><td>{zone.average_dwell_seconds.toFixed(1)}s</td>
              </tr>
            ))}</tbody>
          </table>
          {flows.length > 0 && <div className="zone-flows">{flows.slice(0, 5).map((flow) => <div key={`${flow.from_zone}-${flow.to_zone}`}><span>{flow.from_zone} <b>→</b> {flow.to_zone}</span><strong>{flow.count}</strong></div>)}</div>}
        </div>
      ) : (
        <div className="panel-empty"><Map size={22} aria-hidden="true" /><span>No zones configured</span></div>
      )}
    </section>
  );
}
