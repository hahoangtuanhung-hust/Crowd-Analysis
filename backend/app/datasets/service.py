from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import cv2

from backend.app.core.config import AnalyticsConfig
from backend.app.datasets.analytics import DatasetAnalyticsRunner
from backend.app.datasets.grand_central import (
    CoordinateMode,
    GrandCentralDataset,
    GrandCentralPaths,
    TrajectorySource,
)
from backend.app.datasets.storage import read_trajectory_points


class GrandCentralService:
    def __init__(
        self,
        analytics_config: AnalyticsConfig,
        paths: GrandCentralPaths | None = None,
    ) -> None:
        self.dataset = GrandCentralDataset(paths)
        self.analytics_config = analytics_config
        self._lock = threading.RLock()
        self._status: dict[str, Any] = {
            "state": "ready" if self.available else "missing",
            "last_request": None,
            "error": None,
        }

    @property
    def available(self) -> bool:
        return self.dataset.paths.video.is_file() and any(
            self.dataset.paths.annotations.glob("*.txt")
        )

    def descriptor(self) -> dict[str, Any]:
        return {
            "dataset_id": "grand-central",
            "name": "Grand Central Station",
            "available": self.available,
            "sources": ["ground_truth", "prediction"],
            "coordinate_modes": ["pixel_space", "ground_plane"],
            "durations": ["1m", "5m", "all"],
            "license": "not clearly specified; research/education/internal benchmark only",
        }

    def metadata(self) -> dict[str, Any]:
        if not self.available:
            raise FileNotFoundError("Grand Central dataset is not installed")
        return self.dataset.load_metadata()

    def prepare(self) -> dict[str, Any]:
        if not self.available:
            raise FileNotFoundError("Grand Central dataset is not installed")
        validation = self.dataset.validate()
        paths = self.dataset.paths
        artifacts = {
            "trajectories_parquet": (paths.processed / "trajectories.parquet").is_file(),
            "trajectories_csv": (paths.processed / "trajectories.csv").is_file(),
            "zones": (paths.processed / "zones.json").is_file(),
            "homography": (paths.processed / "homography.json").is_file(),
        }
        return {
            "dataset_id": "grand-central",
            "prepared": all(
                artifacts[name]
                for name in ("trajectories_parquet", "zones", "homography")
            ),
            "source": "ground_truth",
            "validation": validation,
            "artifacts": artifacts,
        }

    def analyze(
        self,
        *,
        source: TrajectorySource,
        coordinate_mode: CoordinateMode,
        duration: str,
    ) -> dict[str, Any]:
        with self._lock:
            self._status = {
                "state": "running",
                "last_request": {
                    "source": source,
                    "coordinate_mode": coordinate_mode,
                    "duration": duration,
                },
                "error": None,
            }
        try:
            duration_seconds = self._duration_seconds(duration)
            trajectory_path = self._trajectory_path(source)
            points = read_trajectory_points(
                trajectory_path,
                source=source,
                duration_seconds=duration_seconds,
            )
            if source == "prediction":
                maximum = max((point.timestamp for point in points), default=-1.0)
                tolerance = 1.0 / float(self.dataset.video_metadata()["fps"])
                if maximum + tolerance < duration_seconds:
                    raise ValueError(
                        f"Prediction trajectories cover only {maximum:.1f}s; run "
                        f"scripts/run_grand_central_inference.py --duration {duration} first."
                    )
            runner = DatasetAnalyticsRunner(
                self.dataset,
                self.analytics_config,
                coordinate_mode=coordinate_mode,
            )
            result = runner.run(
                points,
                source=source,
                duration_label=duration,
                persist=True,
            )
            result = self._merge_inference_metrics(result)
            with self._lock:
                self._status = {
                    "state": "completed",
                    "last_request": self._status["last_request"],
                    "error": None,
                    "result": result["artifacts"]["result_json"],
                }
            return result
        except Exception as exc:
            with self._lock:
                self._status = {
                    "state": "error",
                    "last_request": self._status["last_request"],
                    "error": str(exc),
                }
            raise

    def result(
        self,
        *,
        source: TrajectorySource,
        coordinate_mode: CoordinateMode,
        duration: str,
    ) -> dict[str, Any]:
        path = (
            self.dataset.paths.output
            / f"{source}_{coordinate_mode}_{duration}.json"
        )
        if not path.is_file():
            raise FileNotFoundError(
                f"No {source}/{coordinate_mode}/{duration} result. Run analyze first."
            )
        result = json.loads(path.read_text(encoding="utf-8"))
        return self._merge_inference_metrics(result)

    def status(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._status)

    def preview(self, source: TrajectorySource) -> Path:
        if source == "prediction":
            path = (
                self.dataset.paths.output
                / "tracking"
                / "prediction_interval_1_1m_preview.jpg"
            )
            if not path.is_file():
                raise FileNotFoundError("Prediction preview has not been generated")
            return path
        path = self.dataset.paths.output / "tracking" / "ground_truth_preview.jpg"
        if path.is_file():
            return path
        capture = cv2.VideoCapture(str(self.dataset.paths.video))
        try:
            ok, frame = capture.read()
        finally:
            capture.release()
        if not ok:
            raise ValueError("Unable to read the Grand Central preview frame")
        for point in self.dataset.iter_annotations(duration_seconds=0.0):
            if point.frame_id != 0:
                continue
            center = (round(point.x), round(point.y))
            cv2.drawMarker(
                frame,
                center,
                (54, 211, 163),
                cv2.MARKER_CROSS,
                16,
                2,
                cv2.LINE_AA,
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), frame)
        return path

    def _duration_seconds(self, duration: str) -> float:
        video_duration = float(self.dataset.video_metadata()["duration_seconds"])
        values = {"1m": 60.0, "5m": 300.0, "all": video_duration}
        if duration not in values:
            raise ValueError(f"Unsupported duration: {duration}")
        return min(values[duration], video_duration)

    def _trajectory_path(self, source: TrajectorySource) -> Path:
        if source == "ground_truth":
            return self.dataset.paths.processed / "trajectories.parquet"
        return self.dataset.paths.output / "tracking" / "prediction_trajectories.parquet"

    def _merge_inference_metrics(self, result: dict[str, Any]) -> dict[str, Any]:
        if result.get("source") != "prediction":
            return result
        duration = result.get("duration", "1m")
        metrics_path = (
            self.dataset.paths.output
            / "benchmarks"
            / f"prediction_interval_1_{duration}.json"
        )
        if not metrics_path.is_file():
            return result
        measured = json.loads(metrics_path.read_text(encoding="utf-8"))
        result["metrics"].update(
            {
                "processing_fps": measured["processing_fps"],
                "inference_ms": measured["inference_ms"],
                "tracking_ms": measured["tracking_ms"],
                "e2e_latency_ms": measured["e2e_ms"],
                "dropped_frames": measured["dropped_frames"],
                "cpu_percent": measured["cpu_percent"],
                "ram_mb": measured["ram_mb"],
            }
        )
        return result
