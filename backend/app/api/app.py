from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from backend.app.core.config import AppConfig, ZoneConfig, load_config
from backend.app.core.session import SessionManager
from backend.app.core.uploads import UploadStore
from backend.app.datasets import GrandCentralService
from backend.app.video import OverlayOptions


class RequestModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class StreamStartRequest(RequestModel):
    source: str = Field(min_length=1)
    camera_id: str = Field(default="camera-01", min_length=1, max_length=64)
    realtime: bool | None = None


class OverlayRequest(RequestModel):
    detection: bool = False
    tracking: bool = True
    trajectory: bool = True
    heatmap: bool = False
    zones: bool = False
    points: bool = True
    track_ids: bool = False
    trajectory_tails: bool = False
    grid: bool = False
    edge_flows: bool = False
    candidate_paths: bool = False
    active_paths: bool = True
    direction_arrows: bool = True
    debug_metrics: bool = True


class VisualizationPatchRequest(RequestModel):
    show_bounding_boxes: bool | None = None
    show_track_ids: bool | None = None
    show_tracking_points: bool | None = None
    show_individual_trajectories: bool | None = None
    show_common_path: bool | None = None
    show_direction_arrows: bool | None = None
    show_candidate_path: bool | None = None
    show_zones: bool | None = None
    show_heatmap: bool | None = None
    show_grid_debug: bool | None = None
    show_edge_flow_debug: bool | None = None
    show_metrics: bool | None = None


class CalibrationRequest(RequestModel):
    points: list[tuple[float, float]] = Field(min_length=4, max_length=4)
    width: float = Field(gt=0)
    height: float = Field(gt=0)


class ZoneRequest(RequestModel):
    zone_id: str = Field(min_length=1, max_length=64)
    name: str = Field(min_length=1, max_length=128)
    points: list[tuple[float, float]] = Field(min_length=3)


class ZonesRequest(RequestModel):
    zones: list[ZoneRequest]
    coordinate_space: Literal["camera", "active"] = "camera"


class DatasetAnalyzeRequest(RequestModel):
    source: Literal["ground_truth", "prediction"] = "ground_truth"
    coordinate_mode: Literal["pixel_space", "ground_plane"] = "pixel_space"
    duration: Literal["1m", "5m", "all"] = "all"


def create_app(
    config: AppConfig | None = None,
    manager: SessionManager | None = None,
) -> FastAPI:
    app_config = config or load_config(Path("configs/default.yaml"))
    session_manager = manager or SessionManager(app_config)
    uploads = UploadStore(
        app_config.server.upload_directory,
        app_config.server.max_upload_mb,
    )
    grand_central = GrandCentralService(app_config.analytics)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        session_manager.stop()

    app = FastAPI(
        title="Crowd Analysis API",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.config = app_config
    app.state.session_manager = session_manager
    app.state.uploads = uploads
    app.state.grand_central = grand_central
    app.add_middleware(
        CORSMiddleware,
        allow_origins=app_config.server.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health() -> dict:
        session = session_manager.current
        return {
            "status": "ok",
            "session": asdict(session.snapshot()) if session else None,
        }

    @app.get("/api/runtime")
    def runtime_info() -> dict:
        provider = getattr(session_manager, "runtime_info", None)
        if callable(provider):
            return provider()
        return {
            "mode": "live",
            "source_mode": "configured_input",
            "inference_executed": None,
            "detector_calls": None,
            "cache_reads": None,
            "common_path_only": False,
        }

    @app.get("/metrics")
    def metrics() -> dict:
        session = session_manager.current
        return session.metrics() if session else _empty_metrics()

    @app.get("/api/analytics/metrics")
    def analytics_metrics() -> dict:
        session = session_manager.current
        return session.metrics() if session else _empty_metrics()

    @app.get("/api/datasets")
    def datasets() -> dict:
        return {"datasets": [grand_central.descriptor()]}

    @app.get("/api/datasets/grand-central/metadata")
    def grand_central_metadata() -> dict:
        try:
            return grand_central.metadata()
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/datasets/grand-central/prepare")
    def prepare_grand_central() -> dict:
        try:
            return grand_central.prepare()
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/api/datasets/grand-central/analyze")
    def analyze_grand_central(request: DatasetAnalyzeRequest) -> dict:
        try:
            return grand_central.analyze(**request.model_dump())
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/api/datasets/grand-central/status")
    def grand_central_status() -> dict:
        return grand_central.status()

    @app.get("/api/datasets/grand-central/results")
    def grand_central_results(
        source: Literal["ground_truth", "prediction"] = Query(default="ground_truth"),
        coordinate_mode: Literal["pixel_space", "ground_plane"] = Query(
            default="pixel_space"
        ),
        duration: Literal["1m", "5m", "all"] = Query(default="all"),
    ) -> dict:
        try:
            return grand_central.result(
                source=source, coordinate_mode=coordinate_mode, duration=duration
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/datasets/grand-central/preview.jpg")
    def grand_central_preview(
        source: Literal["ground_truth", "prediction"] = Query(default="ground_truth"),
    ) -> FileResponse:
        try:
            path = grand_central.preview(source)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.post("/api/video/upload", status_code=201)
    async def upload_video(video: Annotated[UploadFile, File()]) -> dict:
        try:
            saved = await uploads.save(video)
        except OverflowError as exc:
            raise HTTPException(status_code=413, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=415, detail=str(exc)) from exc
        return asdict(saved)

    @app.post("/api/stream/start", status_code=202)
    def start_stream(request: StreamStartRequest) -> dict:
        try:
            source_uri, source_kind, inferred_realtime = uploads.resolve(request.source)
            session = session_manager.start(
                source_uri=source_uri,
                source_kind=source_kind,
                realtime=request.realtime if request.realtime is not None else inferred_realtime,
                camera_id=request.camera_id,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return asdict(session.snapshot())

    @app.post("/api/stream/stop")
    def stop_stream() -> dict:
        stopped = session_manager.stop()
        session = session_manager.current
        return {
            "stopped": stopped,
            "session": asdict(session.snapshot()) if session else None,
        }

    @app.put("/api/overlay")
    def update_overlay(request: OverlayRequest) -> dict:
        session = _require_session(session_manager)
        options = OverlayOptions(**request.model_dump())
        session.set_overlay(options)
        return request.model_dump()

    @app.get("/api/config/visualization")
    def visualization_config() -> dict:
        return session_manager.visualization().model_dump()

    @app.patch("/api/config/visualization")
    def patch_visualization(request: VisualizationPatchRequest) -> dict:
        changes = {
            key: value
            for key, value in request.model_dump().items()
            if value is not None
        }
        return session_manager.update_visualization(changes).model_dump()

    @app.post("/api/calibration")
    def update_calibration(request: CalibrationRequest) -> dict:
        _require_session(session_manager)
        try:
            session_manager.set_calibration(
                request.points, width=request.width, height=request.height
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {
            "spatial_mode": "ground",
            "width": request.width,
            "height": request.height,
            "analytics_reset": True,
            "applies_to_next_session": True,
        }

    @app.put("/api/zones")
    def update_zones(request: ZonesRequest) -> dict:
        _require_session(session_manager)
        zones = [ZoneConfig.model_validate(zone.model_dump()) for zone in request.zones]
        try:
            if request.coordinate_space != "camera":
                raise ValueError("Only camera-space zones can persist across sessions")
            session_manager.set_zones(zones)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {
            "zone_count": len(zones),
            "coordinate_space": request.coordinate_space,
            "analytics_reset": True,
            "applies_to_next_session": True,
        }

    @app.get("/api/analytics/summary")
    def analytics_summary() -> dict:
        session = session_manager.current
        if session is None:
            return {
                "summary": _empty_summary(),
                "timeline": [],
                "session": None,
            }
        return {
            "summary": asdict(session.analytics.summary()),
            "timeline": [asdict(item) for item in session.analytics.timeline()],
            "session": asdict(session.snapshot()),
        }

    @app.get("/api/analytics/heatmap")
    def analytics_heatmap(
        metric: Literal["occupancy", "movement"] = Query(default="occupancy"),
        window: Literal["current", "1m", "5m", "entire"] = Query(default="current"),
    ) -> dict:
        session = session_manager.current
        if session is None:
            values = np.zeros(
                (app_config.analytics.grid_height, app_config.analytics.grid_width),
                dtype=np.float32,
            )
            return _heatmap_payload(values, metric, window, "pixel", 0.0, 0.0)
        snapshot = session.analytics.heatmap(window)
        values = snapshot.occupancy if metric == "occupancy" else snapshot.movement
        return _heatmap_payload(
            values,
            metric,
            window,
            snapshot.spatial_mode,
            snapshot.from_timestamp,
            snapshot.to_timestamp,
        )

    @app.get("/api/analytics/flow")
    def analytics_flow(
        window: Literal["current", "1m", "5m", "entire"] = Query(default="current"),
    ) -> dict:
        session = session_manager.current
        if session is None:
            shape = (app_config.analytics.grid_height, app_config.analytics.grid_width)
            return {
                "window": window,
                "spatial_mode": "pixel",
                "dominant_direction": "stationary",
                "vx": np.zeros(shape).tolist(),
                "vy": np.zeros(shape).tolist(),
                "samples": np.zeros(shape, dtype=np.uint32).tolist(),
            }
        snapshot = session.analytics.flow(window)
        return {
            "window": snapshot.window,
            "spatial_mode": snapshot.spatial_mode,
            "from_timestamp": snapshot.from_timestamp,
            "to_timestamp": snapshot.to_timestamp,
            "dominant_direction": snapshot.dominant_direction,
            "vx": snapshot.vx.tolist(),
            "vy": snapshot.vy.tolist(),
            "samples": snapshot.samples.tolist(),
        }

    @app.get("/api/analytics/paths")
    def analytics_paths(limit: int = Query(default=5, ge=1, le=20)) -> dict:
        session = session_manager.current
        paths = session.analytics.top_paths(limit) if session else ()
        return {"paths": [asdict(item) for item in paths]}

    @app.get("/api/analytics/common-paths")
    def analytics_common_paths() -> dict:
        session = session_manager.current
        if session is None:
            return {"timestamp": 0.0, "paths": []}
        snapshot = session.analytics.common_path_snapshot()
        return {
            "timestamp": snapshot.timestamp,
            "paths": [asdict(item) for item in snapshot.paths],
        }

    @app.get("/api/analytics/flows")
    def analytics_directed_flows() -> dict:
        session = session_manager.current
        if session is None:
            settings = app_config.analytics.common_path
            return {
                "from_timestamp": 0.0,
                "to_timestamp": 0.0,
                "grid_columns": settings.grid_columns,
                "grid_rows": settings.grid_rows,
                "edges": [],
            }
        snapshot = session.analytics.common_path_flows()
        return asdict(snapshot)

    @app.get("/api/analytics/zones")
    def analytics_zones() -> dict:
        session = session_manager.current
        if session is None:
            return {"timestamp": 0.0, "zones": [], "flows": []}
        snapshot = session.analytics.zone_snapshot()
        return {
            "timestamp": snapshot.timestamp,
            "zones": [asdict(item) for item in snapshot.zones],
            "flows": [asdict(item) for item in snapshot.flows],
        }

    @app.get("/api/stream/frame.jpg")
    def latest_frame() -> Response:
        session = session_manager.current
        if session is None:
            return Response(status_code=204)
        frame, version = session.latest_frame()
        if frame is None:
            return Response(status_code=204)
        return Response(
            content=frame,
            media_type="image/jpeg",
            headers={"Cache-Control": "no-store", "X-Frame-Version": str(version)},
        )

    @app.get("/api/stream.mjpg")
    async def mjpeg_stream() -> StreamingResponse:
        async def frames():
            last_version = -1
            while True:
                session = session_manager.current
                if session is None:
                    await asyncio.sleep(0.2)
                    continue
                frame, version = session.latest_frame()
                if frame is not None and version != last_version:
                    last_version = version
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
                await asyncio.sleep(0.04)

        return StreamingResponse(
            frames(),
            media_type="multipart/x-mixed-replace; boundary=frame",
            headers={
                "Cache-Control": "no-store, no-cache, must-revalidate",
                "X-Accel-Buffering": "no",          # nginx proxy: không buffer
                "Connection": "keep-alive",
            },
        )

    @app.websocket("/ws/live")
    async def websocket_live(websocket: WebSocket) -> None:
        await websocket.accept()
        try:
            while True:
                session = session_manager.current
                if session is None:
                    payload = {
                        "type": "snapshot",
                        "session": None,
                        "summary": _empty_summary(),
                        "metrics": _empty_metrics(),
                        "paths": [],
                        "common_paths": [],
                        "points": [],
                        "visualization": session_manager.visualization().model_dump(),
                        "zones": [],
                        "timeline": [],
                    }
                else:
                    zone_snapshot = session.analytics.zone_snapshot()
                    common_paths = session.analytics.common_path_snapshot().paths
                    payload = {
                        "type": "snapshot",
                        "session": asdict(session.snapshot()),
                        "summary": asdict(session.analytics.summary()),
                        "metrics": session.metrics(),
                        "paths": [asdict(item) for item in session.analytics.top_paths()],
                        "common_paths": [
                            asdict(item)
                            for item in common_paths
                            if item.state in {"active", "cooling"}
                        ],
                        "points": [
                            {
                                "track_id": point.track_id,
                                "x": point.x,
                                "y": point.y,
                                "confidence": point.confidence,
                            }
                            for point in session.current_points()
                        ],
                        "visualization": session_manager.visualization().model_dump(),
                        "zones": [asdict(item) for item in zone_snapshot.zones],
                        "zone_flows": [asdict(item) for item in zone_snapshot.flows],
                        "timeline": [asdict(item) for item in session.analytics.timeline()],
                        "frame_url": f"/api/stream/frame.jpg?v={session.snapshot().frame_version}",
                    }
                await websocket.send_json(payload)
                await asyncio.sleep(app_config.server.websocket_interval_ms / 1000.0)
        except WebSocketDisconnect:
            return

    return app


def _require_session(manager: SessionManager):
    session = manager.current
    if session is None:
        raise HTTPException(status_code=409, detail="No processing session")
    return session


def _heatmap_payload(
    values: np.ndarray,
    metric: str,
    window: str,
    spatial_mode: str,
    from_timestamp: float,
    to_timestamp: float,
) -> dict:
    return {
        "metric": metric,
        "window": window,
        "spatial_mode": spatial_mode,
        "calibration_required": spatial_mode == "pixel",
        "from_timestamp": from_timestamp,
        "to_timestamp": to_timestamp,
        "grid_width": int(values.shape[1]),
        "grid_height": int(values.shape[0]),
        "max": float(values.max(initial=0.0)),
        "sum": float(values.sum()),
        "values": values.tolist(),
    }


def _empty_summary() -> dict:
    return {
        "current_crowd_count": 0,
        "average_crowd_count": 0.0,
        "peak_crowd_count": 0,
        "unique_track_count": 0,
        "processed_frames": 0,
        "spatial_mode": "pixel",
        "calibration_required": True,
    }


def _empty_metrics() -> dict:
    return {
        "input_fps": 0.0,
        "processing_fps": 0.0,
        "decode_ms": None,
        "preprocessing_ms": None,
        "inference_ms": None,
        "tracking_ms": None,
        "analytics_ms": None,
        "render_ms": None,
        "encoding_ms": None,
        "e2e_latency_ms": None,
        "queue_size": 0,
        "frame_queue_size": 0,
        "capture_queue_size": 0,
        "analytics_queue_size": 0,
        "dropped_frames": 0,
        "active_tracks": 0,
        "lost_tracks": 0,
        "completed_tracks": 0,
        "valid_tracks": 0,
        "discarded_tracks": 0,
        "flow_bucket_count": 0,
        "common_path_switches": 0,
        "candidate_rejections": 0,
        "common_path_compute_ms": None,
        "common_path_compute_ms_p95": None,
        "cpu_percent": 0.0,
        "ram_mb": 0.0,
        "gpu_utilization": None,
        "gpu_memory_mb": None,
    }
