import { useEffect, useRef } from "react";
import { Navigation } from "lucide-react";

import type { FlowData } from "../types";

export function FlowField({ data }: { data: FlowData | null }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const context = canvas.getContext("2d");
    if (!context) return;
    canvas.width = 640;
    canvas.height = 300;
    context.fillStyle = "#12171c";
    context.fillRect(0, 0, canvas.width, canvas.height);
    context.strokeStyle = "#28313a";
    context.lineWidth = 1;
    for (let x = 0; x <= canvas.width; x += 80) {
      context.beginPath(); context.moveTo(x, 0); context.lineTo(x, canvas.height); context.stroke();
    }
    for (let y = 0; y <= canvas.height; y += 60) {
      context.beginPath(); context.moveTo(0, y); context.lineTo(canvas.width, y); context.stroke();
    }
    if (!data || data.samples.length === 0) return;
    const rows = data.samples.length;
    const columns = data.samples[0].length;
    const cellWidth = canvas.width / columns;
    const cellHeight = canvas.height / rows;
    context.strokeStyle = "#4ad3a5";
    context.fillStyle = "#4ad3a5";
    context.lineWidth = 1.5;
    for (let row = 0; row < rows; row += 1) {
      for (let column = 0; column < columns; column += 1) {
        const samples = data.samples[row][column];
        if (!samples) continue;
        const dx = data.vx[row][column] / samples;
        const dy = data.vy[row][column] / samples;
        const magnitude = Math.hypot(dx, dy);
        if (!magnitude) continue;
        const length = Math.min(cellWidth, cellHeight) * 0.42;
        const startX = (column + 0.5) * cellWidth;
        const startY = (row + 0.5) * cellHeight;
        const endX = startX + dx / magnitude * length;
        const endY = startY + dy / magnitude * length;
        drawArrow(context, startX, startY, endX, endY);
      }
    }
  }, [data]);
  const direction = data?.dominant_direction ?? "stationary";
  return (
    <section className="panel flow-panel">
      <header className="panel-header split-header">
        <div><p className="eyebrow">Vector field · Pathmap</p><h2>Pathmap (Hướng di chuyển)</h2></div>
        <span className="direction-badge"><Navigation size={15} style={{ transform: `rotate(${directionAngle(direction)}deg)` }} />Hướng chủ đạo: {direction}</span>
      </header>
      <canvas ref={canvasRef} aria-label={`Movement flow field, dominant direction ${direction}`} />
    </section>
  );
}

function drawArrow(context: CanvasRenderingContext2D, x1: number, y1: number, x2: number, y2: number) {
  const angle = Math.atan2(y2 - y1, x2 - x1);
  context.beginPath(); context.moveTo(x1, y1); context.lineTo(x2, y2); context.stroke();
  context.beginPath();
  context.moveTo(x2, y2);
  context.lineTo(x2 - 4 * Math.cos(angle - Math.PI / 6), y2 - 4 * Math.sin(angle - Math.PI / 6));
  context.lineTo(x2 - 4 * Math.cos(angle + Math.PI / 6), y2 - 4 * Math.sin(angle + Math.PI / 6));
  context.closePath(); context.fill();
}

function directionAngle(direction: string) {
  return ({ east: 90, "south-east": 135, south: 180, "south-west": 225, west: 270, "north-west": 315, north: 0, "north-east": 45 } as Record<string, number>)[direction] ?? 0;
}
