from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any

import numpy as np
import psutil

from backend.app.schemas import FrameResult
from backend.app.video.pipeline import PipelineStats


class PerformanceMonitor:
    def __init__(self, history_size: int = 2048) -> None:
        self._lock = threading.Lock()
        self._process = psutil.Process()
        self._started = time.monotonic()
        self._finished: float | None = None
        self._processed_at: deque[float] = deque(maxlen=history_size)
        self._stages: dict[str, deque[float]] = {
            name: deque(maxlen=history_size)
            for name in (
                "decode_ms",
                "inference_ms",
                "tracking_ms",
                "analytics_ms",
                "render_ms",
                "encoding_ms",
                "e2e_latency_ms",
            )
        }
        self._process.cpu_percent(interval=None)

    def record(
        self,
        result: FrameResult,
        *,
        analytics_ms: float,
        render_ms: float,
        encoding_ms: float,
    ) -> None:
        completed = time.monotonic()
        with self._lock:
            self._processed_at.append(completed)
            self._stages["decode_ms"].append(result.packet.decode_ms)
            self._stages["inference_ms"].append(result.inference_ms)
            self._stages["tracking_ms"].append(result.tracking_ms)
            self._stages["analytics_ms"].append(analytics_ms)
            self._stages["render_ms"].append(render_ms)
            self._stages["encoding_ms"].append(encoding_ms)
            self._stages["e2e_latency_ms"].append(
                max(0.0, (completed - result.packet.captured_monotonic) * 1000.0)
            )

    def finish(self) -> None:
        with self._lock:
            self._finished = time.monotonic()

    def processing_fps(self) -> float:
        with self._lock:
            if len(self._processed_at) < 2:
                return 0.0
            duration = self._processed_at[-1] - self._processed_at[0]
            return (len(self._processed_at) - 1) / duration if duration > 0 else 0.0

    def snapshot(
        self,
        pipeline: PipelineStats,
        *,
        analytics_queue_size: int,
        analytics_dropped_frames: int,
    ) -> dict[str, Any]:
        with self._lock:
            elapsed = max(1e-6, (self._finished or time.monotonic()) - self._started)
            metrics: dict[str, Any] = {
                "input_fps": round(pipeline.captured_frames / elapsed, 3),
                "processing_fps": round(self._fps_unlocked(), 3),
                "preprocessing_ms": None,
                "queue_size": pipeline.queue_size + analytics_queue_size,
                "capture_queue_size": pipeline.queue_size,
                "analytics_queue_size": analytics_queue_size,
                "dropped_frames": pipeline.dropped_frames + analytics_dropped_frames,
                "capture_dropped_frames": pipeline.dropped_frames,
                "analytics_dropped_frames": analytics_dropped_frames,
                "cpu_percent": round(self._process.cpu_percent(interval=None), 2),
                "ram_mb": round(self._process.memory_info().rss / (1024 * 1024), 2),
            }
            for name, values in self._stages.items():
                metrics[name] = self._percentile(values, 50)
                metrics[f"{name}_p95"] = self._percentile(values, 95)
            gpu = self._gpu_metrics()
            metrics.update(gpu)
            return metrics

    def _fps_unlocked(self) -> float:
        if len(self._processed_at) < 2:
            return 0.0
        duration = self._processed_at[-1] - self._processed_at[0]
        return (len(self._processed_at) - 1) / duration if duration > 0 else 0.0

    @staticmethod
    def _percentile(values: deque[float], percentile: int) -> float | None:
        if not values:
            return None
        return round(float(np.percentile(np.asarray(values, dtype=np.float64), percentile)), 3)

    @staticmethod
    def _gpu_metrics() -> dict[str, float | None]:
        try:
            import torch

            if not torch.cuda.is_available():
                return {"gpu_utilization": None, "gpu_memory_mb": None}
            return {
                "gpu_utilization": None,
                "gpu_memory_mb": round(torch.cuda.memory_allocated() / (1024 * 1024), 2),
            }
        except (ImportError, OSError, RuntimeError):
            return {"gpu_utilization": None, "gpu_memory_mb": None}
