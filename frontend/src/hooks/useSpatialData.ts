import { useEffect, useState } from "react";

import { getFlow, getHeatmap } from "../api";
import type { FlowData, HeatmapData, HeatmapMetric, TimeWindow } from "../types";

export function useSpatialData(metric: HeatmapMetric, timeWindow: TimeWindow, version: number) {
  const [heatmap, setHeatmap] = useState<HeatmapData | null>(null);
  const [flow, setFlow] = useState<FlowData | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    const timer = globalThis.window.setTimeout(() => {
      Promise.all([
        getHeatmap(metric, timeWindow, controller.signal),
        getFlow(timeWindow, controller.signal)
      ]).then(([nextHeatmap, nextFlow]) => {
        setHeatmap(nextHeatmap);
        setFlow(nextFlow);
      }).catch((error: unknown) => {
        if (!(error instanceof DOMException && error.name === "AbortError")) {
          console.error(error);
        }
      });
    }, 350);
    return () => {
      globalThis.window.clearTimeout(timer);
      controller.abort();
    };
  }, [metric, timeWindow, Math.floor(version / 2)]);

  return { heatmap, flow };
}
