from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
from fastapi.testclient import TestClient

from backend.app.api import create_app
from backend.app.core.config import AnalyticsConfig, AppConfig, ServerConfig, VideoConfig
from backend.app.core.session import SessionManager
from backend.app.schemas import Detection


class MovingDetector:
    def __init__(self, _config: object) -> None:
        self.frame_index = 0

    def detect(self, frame: np.ndarray) -> list[Detection]:
        offset = self.frame_index * 3
        self.frame_index += 1
        return [Detection(20 + offset, 15, 55 + offset, 100, 0.95)]


def make_video(path: Path) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (160, 120))
    assert writer.isOpened()
    for index in range(8):
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        cv2.rectangle(frame, (20 + index * 3, 15), (55 + index * 3, 100), (255, 255, 255), -1)
        writer.write(frame)
    writer.release()


def test_api_upload_process_and_analytics(tmp_path: Path) -> None:
    upload_directory = tmp_path / "uploads"
    config = AppConfig(
        server=ServerConfig(upload_directory=str(upload_directory)),
        video=VideoConfig(stale_frame_policy="block", queue_size=2),
        analytics=AnalyticsConfig(min_confirmed_points=2, movement_threshold_pixels=1),
    )
    manager = SessionManager(config, detector_factory=MovingDetector)
    app = create_app(config, manager)
    video_path = tmp_path / "input.mp4"
    make_video(video_path)

    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["session"] is None

        runtime = client.get("/api/runtime")
        assert runtime.status_code == 200
        assert runtime.json()["mode"] == "live"

        visualization = client.get("/api/config/visualization")
        assert visualization.status_code == 200
        assert visualization.json()["show_tracking_points"] is True
        assert visualization.json()["show_individual_trajectories"] is False
        assert visualization.json()["show_zones"] is False

        updated_visualization = client.patch(
            "/api/config/visualization",
            json={"show_zones": True, "show_direction_arrows": False},
        )
        assert updated_visualization.status_code == 200
        assert updated_visualization.json()["show_zones"] is True
        assert updated_visualization.json()["show_direction_arrows"] is False

        with video_path.open("rb") as stream:
            uploaded = client.post(
                "/api/video/upload",
                files={"video": ("input.mp4", stream, "video/mp4")},
            )
        assert uploaded.status_code == 201
        source = uploaded.json()["source_token"]

        started = client.post("/api/stream/start", json={"source": source})
        assert started.status_code == 202

        deadline = time.monotonic() + 10
        status = "running"
        while time.monotonic() < deadline and status in {"starting", "running"}:
            status = client.get("/health").json()["session"]["status"]
            time.sleep(0.02)
        assert status == "completed"
        session = client.get("/health").json()["session"]
        assert session["frame_id"] == 7
        assert session["media_timestamp_s"] >= 0.0
        assert len(session["stream_epoch"]) == 32

        summary = client.get("/api/analytics/summary").json()["summary"]
        assert summary["processed_frames"] == 8
        assert summary["peak_crowd_count"] == 1
        assert summary["unique_track_count"] == 1

        heatmap = client.get(
            "/api/analytics/heatmap",
            params={"metric": "movement", "window": "entire"},
        ).json()
        assert heatmap["sum"] > 0
        assert heatmap["grid_width"] == config.analytics.grid_width

        metrics = client.get("/metrics").json()
        assert metrics["tracking_ms"] is not None
        assert metrics["analytics_ms"] is not None
        assert metrics["gpu_utilization"] is None
        assert client.get("/api/analytics/metrics").status_code == 200

        common_paths = client.get("/api/analytics/common-paths").json()
        assert set(common_paths) == {"timestamp", "paths"}
        directed_flows = client.get("/api/analytics/flows").json()
        assert directed_flows["grid_columns"] == config.analytics.common_path.grid_columns
        assert isinstance(directed_flows["edges"], list)

        frame = client.get("/api/stream/frame.jpg")
        assert frame.status_code == 200
        assert frame.headers["content-type"] == "image/jpeg"
        assert len(frame.content) > 100

        with client.websocket_connect("/ws/live") as websocket:
            snapshot = websocket.receive_json()
            assert snapshot["type"] == "snapshot"
            assert snapshot["session"]["status"] == "completed"
            assert isinstance(snapshot["common_paths"], list)
            assert isinstance(snapshot["points"], list)
            assert snapshot["visualization"]["show_zones"] is True
            assert "trajectories" not in snapshot
            assert "trajectory_history" not in snapshot

        calibrated = client.post(
            "/api/calibration",
            json={
                "points": [[0, 0], [159, 0], [159, 119], [0, 119]],
                "width": 16,
                "height": 12,
            },
        )
        assert calibrated.status_code == 200
        assert calibrated.json()["analytics_reset"] is True
        assert client.get("/api/analytics/summary").json()["summary"]["spatial_mode"] == "ground"

        zones = client.put(
            "/api/zones",
            json={
                "coordinate_space": "camera",
                "zones": [
                    {
                        "zone_id": "entrance",
                        "name": "Entrance",
                        "points": [[0, 0], [70, 0], [70, 119], [0, 119]],
                    }
                ],
            },
        )
        assert zones.status_code == 200
        assert zones.json()["zone_count"] == 1

        restarted = client.post("/api/stream/start", json={"source": source})
        assert restarted.status_code == 202
        deadline = time.monotonic() + 10
        status = "running"
        while time.monotonic() < deadline and status in {"starting", "running"}:
            status = client.get("/health").json()["session"]["status"]
            time.sleep(0.02)
        assert status == "completed"
        assert client.get("/api/analytics/summary").json()["summary"]["spatial_mode"] == "ground"
        persisted_zones = client.get("/api/analytics/zones").json()["zones"]
        assert persisted_zones[0]["name"] == "Entrance"


def test_upload_rejects_non_video_extension(tmp_path: Path) -> None:
    config = AppConfig(server=ServerConfig(upload_directory=str(tmp_path / "uploads")))
    app = create_app(config, SessionManager(config, detector_factory=MovingDetector))
    with TestClient(app) as client:
        response = client.post(
            "/api/video/upload",
            files={"video": ("notes.txt", b"not a video", "text/plain")},
        )
    assert response.status_code == 415
