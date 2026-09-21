from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Full
from typing import Literal

import cv2

from backend.app.analytics import AnalyticsEngine, SpatialTransformer
from backend.app.core.config import (
    AnalyticsConfig,
    AppConfig,
    DetectorConfig,
    VisualizationConfig,
    ZoneConfig,
)
from backend.app.inference import PersonDetector, UltralyticsPersonDetector
from backend.app.metrics import PerformanceMonitor
from backend.app.schemas import FrameResult, TrackPoint
from backend.app.tracking import ByteTrackTracker
from backend.app.video import FrameRenderer, OpenCVVideoSource, OverlayOptions, TrackingPipeline
from backend.app.video.latest_queue import BoundedFrameQueue

LOGGER = logging.getLogger(__name__)
_ANALYTICS_END = object()
SessionStatus = Literal["idle", "starting", "running", "completed", "stopped", "error"]


@dataclass(frozen=True, slots=True)
class SessionSnapshot:
    camera_id: str
    source_kind: str
    status: SessionStatus
    error: str | None
    frame_version: int
    frame_id: int
    media_timestamp_s: float
    stream_epoch: str
    width: int
    height: int


class ProcessingSession:
    def __init__(
        self,
        *,
        camera_id: str,
        source_uri: str,
        source_kind: str,
        realtime: bool,
        detector: PersonDetector,
        config: AppConfig,
        visualization: VisualizationConfig | None = None,
        max_frames: int | None = None,
    ) -> None:
        self.camera_id = camera_id
        self.source_uri = source_uri
        self.source_kind = source_kind
        self.realtime = realtime
        self.stream_epoch = uuid.uuid4().hex
        self.config = config
        self.source = OpenCVVideoSource(
            source_uri,
            max_frames=max_frames,
            pace=realtime and source_kind != "rtsp",
        )
        metadata = self.source.open()
        self.source.close()
        self.metadata = metadata
        spatial = SpatialTransformer.pixel(metadata.width, metadata.height)
        self.analytics = AnalyticsEngine(
            config.analytics, spatial, retain_entire=not realtime,
            camera_id=camera_id, stream_epoch=self.stream_epoch,
        )
        self.visualization = visualization or config.visualization
        self.renderer = FrameRenderer(self.visualization)
        self.monitor = PerformanceMonitor()
        self._overlay = OverlayOptions.from_visualization(self.visualization)
        self._analytics_queue: BoundedFrameQueue[FrameResult | object] = BoundedFrameQueue(
            maxsize=config.video.analytics_queue_size,
            drop_oldest=realtime,
        )
        self.pipeline = TrackingPipeline(
            source=self.source,
            detector=detector,
            tracker=ByteTrackTracker(config.tracker),
            queue_size=config.video.queue_size,
            drop_oldest=realtime,
            inference_interval=config.video.inference_interval,
            reconnect=realtime and source_kind == "rtsp",
            reconnect_initial_seconds=config.video.reconnect_initial_seconds,
            reconnect_max_seconds=config.video.reconnect_max_seconds,
            on_result=self._enqueue_result,
        )
        self._stop = threading.Event()
        self._state_lock = threading.RLock()
        self._analytics_thread: threading.Thread | None = None
        self._supervisor_thread: threading.Thread | None = None
        self._status: SessionStatus = "idle"
        self._error: str | None = None
        self._latest_jpeg: bytes | None = None
        self._latest_points: tuple[TrackPoint, ...] = ()
        self._frame_version = 0
        self._latest_frame_id = -1
        self._latest_media_timestamp_s = 0.0

    def start(self) -> None:
        with self._state_lock:
            if self._status in {"starting", "running"}:
                raise RuntimeError("Session is already running")
            self._status = "starting"
            self._error = None
        self._analytics_thread = threading.Thread(
            target=self._analytics_loop,
            name=f"analytics-{self.camera_id}",
            daemon=True,
        )
        self._analytics_thread.start()
        try:
            self.pipeline.start()
        except Exception:
            self._stop.set()
            self._put_analytics_end()
            self._analytics_thread.join(timeout=5.0)
            with self._state_lock:
                self._status = "error"
            raise
        with self._state_lock:
            self._status = "running"
        self._supervisor_thread = threading.Thread(
            target=self._supervise,
            name=f"supervisor-{self.camera_id}",
            daemon=True,
        )
        self._supervisor_thread.start()

    def stop(self) -> None:
        self._stop.set()
        self.pipeline.stop()
        with self._state_lock:
            if self._status not in {"completed", "error"}:
                self._status = "stopped"

    def wait(self, timeout: float | None = None) -> bool:
        if self._supervisor_thread is None:
            return True
        self._supervisor_thread.join(timeout)
        return not self._supervisor_thread.is_alive()

    def set_overlay(self, options: OverlayOptions) -> None:
        with self._state_lock:
            self._overlay = options

    def current_points(self) -> tuple[TrackPoint, ...]:
        with self._state_lock:
            return self._latest_points

    def set_calibration(
        self,
        source_points: list[tuple[float, float]],
        *,
        width: float,
        height: float,
    ) -> None:
        transformer = SpatialTransformer.ground_plane(source_points, width=width, height=height)
        config_data = self.analytics.config.model_dump()
        config_data["zones"] = []
        analytics_config = AnalyticsConfig.model_validate(config_data)
        with self._state_lock:
            self.analytics = AnalyticsEngine(
                analytics_config,
                transformer,
                retain_entire=not self.realtime,
                camera_id=self.camera_id,
                stream_epoch=uuid.uuid4().hex,
            )

    def set_zones(self, zones: list[ZoneConfig], *, camera_coordinates: bool = True) -> None:
        with self._state_lock:
            transformer = self.analytics.transformer
            if camera_coordinates:
                zones = [
                    ZoneConfig(
                        zone_id=zone.zone_id,
                        name=zone.name,
                        zone_type=zone.zone_type,
                        points=[transformer.transform(x, y) for x, y in zone.points],
                    )
                    for zone in zones
                ]
            config_data = self.analytics.config.model_dump()
            config_data["zones"] = [zone.model_dump() for zone in zones]
            analytics_config = AnalyticsConfig.model_validate(config_data)
            self.analytics = AnalyticsEngine(
                analytics_config,
                transformer,
                retain_entire=not self.realtime,
                camera_id=self.camera_id,
                stream_epoch=uuid.uuid4().hex,
            )

    def latest_frame(self) -> tuple[bytes | None, int]:
        with self._state_lock:
            return self._latest_jpeg, self._frame_version

    def snapshot(self) -> SessionSnapshot:
        with self._state_lock:
            return SessionSnapshot(
                camera_id=self.camera_id,
                source_kind=self.source_kind,
                status=self._status,
                error=self._error or self.pipeline.stats.error,
                frame_version=self._frame_version,
                frame_id=self._latest_frame_id,
                media_timestamp_s=self._latest_media_timestamp_s,
                stream_epoch=self.stream_epoch,
                width=self.metadata.width,
                height=self.metadata.height,
            )

    def metrics(self) -> dict:
        metrics = self.monitor.snapshot(
            self.pipeline.stats,
            analytics_queue_size=self._analytics_queue.size,
            analytics_dropped_frames=self._analytics_queue.dropped,
        )
        metrics["frame_queue_size"] = metrics["capture_queue_size"]
        metrics.update(self.analytics.common_path_metrics())
        return metrics

    def _enqueue_result(self, result: FrameResult) -> None:
        while not self._stop.is_set():
            try:
                self._analytics_queue.put(result)
                return
            except Full:
                continue

    def _put_analytics_end(self) -> None:
        while self._analytics_thread is not None and self._analytics_thread.is_alive():
            try:
                self._analytics_queue.put(_ANALYTICS_END)
                return
            except Full:
                self._stop.wait(0.05)

    def _supervise(self) -> None:
        self.pipeline.join()
        self._put_analytics_end()
        if self._analytics_thread is not None:
            self._analytics_thread.join()
        self.monitor.finish()
        with self._state_lock:
            if self.pipeline.stats.error:
                self._error = self.pipeline.stats.error
                self._status = "error"
            elif self._status not in {"stopped", "error"}:
                self._status = "completed"

    def _analytics_loop(self) -> None:
        try:
            while True:
                try:
                    item = self._analytics_queue.get(timeout=0.1)
                except Empty:
                    if self._stop.is_set() and not self.pipeline.stats.running:
                        break
                    continue
                if item is _ANALYTICS_END:
                    break
                assert isinstance(item, FrameResult)

                with self._state_lock:
                    engine = self.analytics
                    overlay = self._overlay
                started = time.perf_counter()
                trajectories = engine.process_frame(item)
                current_points = tuple(
                    trajectory.points[-1]
                    for trajectory in trajectories
                    if trajectory.points
                    and trajectory.points[-1].frame_id == item.packet.frame_id
                )
                heatmap = engine.heatmap("current")
                common_paths = engine.common_path_snapshot()
                directed_flows = engine.common_path_flows()
                analytics_ms = (time.perf_counter() - started) * 1000.0

                render_started = time.perf_counter()
                frame = self.renderer.render(
                    item,
                    current_points,
                    heatmap,
                    tuple(engine.config.zones),
                    engine.transformer,
                    overlay,
                    processing_fps=self.monitor.processing_fps(),
                    common_paths=common_paths,
                    directed_flows=directed_flows,
                    debug_trajectories=(
                        trajectories if overlay.trajectory_tails else ()
                    ),
                )
                render_ms = (time.perf_counter() - render_started) * 1000.0

                encoding_started = time.perf_counter()
                ok, encoded = cv2.imencode(
                    ".jpg",
                    frame,
                    [cv2.IMWRITE_JPEG_QUALITY, self.config.server.jpeg_quality],
                )
                encoding_ms = (time.perf_counter() - encoding_started) * 1000.0
                if not ok:
                    raise RuntimeError("JPEG encoding failed")
                self.monitor.record(
                    item,
                    analytics_ms=analytics_ms,
                    render_ms=render_ms,
                    encoding_ms=encoding_ms,
                )
                with self._state_lock:
                    self._latest_jpeg = encoded.tobytes()
                    self._latest_points = current_points
                    self._frame_version += 1
                    self._latest_frame_id = item.packet.frame_id
                    self._latest_media_timestamp_s = item.packet.source_timestamp

                if item.packet.frame_id % 30 == 0:
                    metrics = self.metrics()
                    LOGGER.info(
                        json.dumps(
                            {
                                "camera_id": self.camera_id,
                                "frame_id": item.packet.frame_id,
                                "people": len(item.tracks),
                                "fps": metrics["processing_fps"],
                                "inference_ms": metrics["inference_ms"],
                                "tracking_ms": metrics["tracking_ms"],
                                "queue_size": metrics["queue_size"],
                            },
                            separators=(",", ":"),
                        )
                    )
        except Exception as exc:
            LOGGER.exception("Analytics worker failed")
            with self._state_lock:
                self._error = f"analytics: {exc}"
                self._status = "error"
            self._stop.set()
            self.pipeline.stop()
        finally:
            with self._state_lock:
                self.analytics.finalize()


class SessionManager:
    def __init__(
        self,
        config: AppConfig,
        detector_factory: Callable[[DetectorConfig], PersonDetector] = UltralyticsPersonDetector,
    ) -> None:
        self.config = config
        self._detector_factory = detector_factory
        self._detector: PersonDetector | None = None
        self._session: ProcessingSession | None = None
        self._calibration: tuple[list[tuple[float, float]], float, float] | None = None
        self._zones: list[ZoneConfig] = []
        self._visualization = config.visualization.model_copy(deep=True)
        self._lock = threading.RLock()

    @property
    def current(self) -> ProcessingSession | None:
        with self._lock:
            return self._session

    def start(
        self,
        *,
        source_uri: str,
        source_kind: str,
        realtime: bool,
        camera_id: str = "camera-01",
        max_frames: int | None = None,
    ) -> ProcessingSession:
        with self._lock:
            if self._session is not None and self._session.snapshot().status in {
                "starting",
                "running",
            }:
                LOGGER.info("Stopping previous session before starting a new one")
                self._session.stop()
                self._session.wait(timeout=5.0)
            if self._detector is None:
                self._detector = self._detector_factory(self.config.detector)
            reset = getattr(self._detector, "reset", None)
            if callable(reset):
                reset()
            session = ProcessingSession(
                camera_id=camera_id,
                source_uri=source_uri,
                source_kind=source_kind,
                realtime=realtime,
                detector=self._detector,
                config=self.config,
                visualization=self._visualization,
                max_frames=max_frames,
            )
            if self._calibration is not None:
                points, width, height = self._calibration
                session.set_calibration(points, width=width, height=height)
            zones = self._zones or self._configured_zones(
                session.metadata.width, session.metadata.height
            )
            if zones:
                session.set_zones(zones, camera_coordinates=True)
            self._session = session
            session.start()
            return session

    def visualization(self) -> VisualizationConfig:
        with self._lock:
            return self._visualization.model_copy(deep=True)

    def update_visualization(self, changes: dict[str, bool]) -> VisualizationConfig:
        with self._lock:
            self._visualization = self._visualization.model_copy(update=changes)
            if self._session is not None:
                self._session.set_overlay(
                    OverlayOptions.from_visualization(self._visualization)
                )
            return self._visualization.model_copy(deep=True)

    def _configured_zones(self, width: int, height: int) -> list[ZoneConfig]:
        configured_path = self.config.server.zones_path
        if not configured_path:
            return []
        path = Path(configured_path)
        if not path.is_file():
            LOGGER.warning("Configured zones file does not exist: %s", path)
            return []
        document = json.loads(path.read_text(encoding="utf-8"))
        source_width = float(document.get("image_width", width))
        source_height = float(document.get("image_height", height))
        scale_x = width / source_width
        scale_y = height / source_height
        zones: list[ZoneConfig] = []
        for item in document.get("zones", []):
            zone_id = item.get("id", item.get("zone_id"))
            if not zone_id:
                raise ValueError("Every configured zone requires id or zone_id")
            zones.append(
                ZoneConfig(
                    zone_id=str(zone_id),
                    name=str(item.get("name", zone_id)),
                    zone_type=item.get("type", "area"),
                    points=[
                        (float(x) * scale_x, float(y) * scale_y)
                        for x, y in item["points"]
                    ],
                )
            )
        return zones

    def set_calibration(
        self,
        source_points: list[tuple[float, float]],
        *,
        width: float,
        height: float,
    ) -> None:
        with self._lock:
            session = self._session
            if session is None:
                raise RuntimeError("No processing session")
            points = [(float(x), float(y)) for x, y in source_points]
            session.set_calibration(points, width=width, height=height)
            if self._zones:
                session.set_zones(self._zones, camera_coordinates=True)
            self._calibration = (points, float(width), float(height))

    def set_zones(self, zones: list[ZoneConfig]) -> None:
        with self._lock:
            session = self._session
            if session is None:
                raise RuntimeError("No processing session")
            stored = [zone.model_copy(deep=True) for zone in zones]
            session.set_zones(stored, camera_coordinates=True)
            self._zones = stored

    def stop(self) -> bool:
        session = self.current
        if session is None:
            return False
        session.stop()
        session.wait(timeout=10.0)
        return True
