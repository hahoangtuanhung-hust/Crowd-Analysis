import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

import type { TimelinePoint } from "../types";

export function TimelinePanel({ timeline }: { timeline: TimelinePoint[] }) {
  return (
    <section className="panel timeline-panel">
      <header className="panel-header"><div><p className="eyebrow">Processed timeline</p><h2>Crowd over time</h2></div></header>
      <div className="chart-frame">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={timeline} margin={{ top: 8, right: 14, bottom: 0, left: -18 }}>
            <CartesianGrid stroke="#28313a" vertical={false} />
            <XAxis dataKey="timestamp" stroke="#7f8a96" tickLine={false} axisLine={false} tickFormatter={formatTime} minTickGap={28} />
            <YAxis stroke="#7f8a96" tickLine={false} axisLine={false} allowDecimals={false} width={42} />
            <Tooltip contentStyle={{ background: "#171d23", border: "1px solid #343e48", borderRadius: 6 }} labelFormatter={(value) => formatTime(Number(value))} />
            <Line type="monotone" dataKey="average_count" name="Average" stroke="#6b9cff" strokeWidth={2} dot={false} isAnimationActive={false} />
            <Line type="stepAfter" dataKey="peak_count" name="Peak" stroke="#e0a13d" strokeWidth={1.5} strokeDasharray="5 4" dot={false} isAnimationActive={false} />
          </LineChart>
        </ResponsiveContainer>
        {timeline.length < 2 && <span className="canvas-empty">Waiting for timeline samples</span>}
      </div>
    </section>
  );
}

function formatTime(value: number) {
  const minutes = Math.floor(value / 60);
  const seconds = Math.floor(value % 60);
  return `${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`;
}
