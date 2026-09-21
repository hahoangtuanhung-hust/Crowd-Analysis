from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from backend.app.core.config import (
    AnalyticsConfig,
    AppConfig,
    CommonPathConfig,
    TrackerConfig,
    VideoConfig,
)
from backend.app.core.session import ProcessingSession
from backend.app.schemas import Detection, FramePacket, FrameResult
from backend.app.tracking import ByteTrackTracker
from backend.app.video import OpenCVVideoSource, TrackingPipeline
from backend.app.video.latest_queue import BoundedFrameQueue


class FixedDetector:
    def __init__(self) -> None:
        self.frame_index = 0

    def detect(self, frame: np.ndarray) -> list[Detection]:
        offset = self.frame_index * 2
        self.frame_index += 1
        return [Detection(20 + offset, 20, 60 + offset, 100, 0.95)]


def make_video(path: Path, frame_count: int = 8) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10.0,
        (160, 120),
    )
    assert writer.isOpened()
    for index in range(frame_count):
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        cv2.rectangle(frame, (20 + index * 2, 20), (60 + index * 2, 100), (255, 255, 255), -1)
        writer.write(frame)
    writer.release()


def test_pipeline_processes_file_without_dropping(tmp_path: Path) -> None:
    video_path = tmp_path / "input.mp4"
    make_video(video_path)
    results: list[FrameResult] = []
    pipeline = TrackingPipeline(
        OpenCVVideoSource(video_path),
        FixedDetector(),
        ByteTrackTracker(TrackerConfig()),
        queue_size=2,
        drop_oldest=False,
        on_result=results.append,
    )

    pipeline.start()
    assert pipeline.join(timeout=10.0)

    stats = pipeline.stats
    assert stats.error is None
    assert stats.captured_frames == 8
    assert stats.processed_frames == 8
    assert stats.dropped_frames == 0
    assert len(results) == 8
    stable_ids = [track.track_id for result in results for track in result.tracks]
    assert len(stable_ids) >= 6
    assert len(set(stable_ids)) == 1


def test_pipeline_passes_frame_packet_to_cache_detector(tmp_path: Path) -> None:
    video_path = tmp_path / "packet.mp4"
    make_video(video_path, frame_count=3)

    class PacketDetector:
        def __init__(self) -> None:
            self.frame_ids: list[int] = []

        def detect(self, _frame: np.ndarray) -> list[Detection]:
            raise AssertionError("Cache-backed detection must receive FramePacket")

        def detect_packet(self, packet: FramePacket) -> list[Detection]:
            self.frame_ids.append(packet.frame_id)
            return [Detection(20, 20, 60, 100, 0.95)]

    detector = PacketDetector()
    pipeline = TrackingPipeline(
        OpenCVVideoSource(video_path),
        detector,
        ByteTrackTracker(TrackerConfig()),
        queue_size=2,
        drop_oldest=False,
    )

    pipeline.start()

    assert pipeline.join(timeout=10.0)
    assert pipeline.stats.error is None
    assert detector.frame_ids == [0, 1, 2]


def test_latest_queue_drops_oldest() -> None:
    queue: BoundedFrameQueue[int] = BoundedFrameQueue(maxsize=2, drop_oldest=True)
    queue.put(1)
    queue.put(2)
    queue.put(3)

    assert queue.dropped == 1
    assert queue.get() == 2
    assert queue.get() == 3


def test_invalid_source_fails_before_workers() -> None:
    pipeline = TrackingPipeline(
        OpenCVVideoSource("definitely-missing-video.mp4"),
        FixedDetector(),
        ByteTrackTracker(TrackerConfig()),
    )
    started = time.monotonic()
    try:
        pipeline.start()
    except ValueError as exc:
        assert "Unable to open video source" in str(exc)
    else:
        raise AssertionError("Expected invalid source error")
    assert time.monotonic() - started < 2.0


def test_video_source_honors_max_frames(tmp_path: Path) -> None:
    video_path = tmp_path / "input.mp4"
    make_video(video_path, frame_count=8)
    source = OpenCVVideoSource(video_path, max_frames=3)
    try:
        packets = list(source.frames())
    finally:
        source.close()
    assert [packet.frame_id for packet in packets] == [0, 1, 2]


def test_realtime_pipeline_reconnects_after_source_ends() -> None:
    class CyclingSource:
        def __init__(self) -> None:
            self.open_count = 0
            self.frame_id = 0

        def open(self) -> None:
            self.open_count += 1

        def close(self) -> None:
            return None

        def frames(self):
            frame_id = self.frame_id
            self.frame_id += 1
            yield FramePacket(
                frame_id=frame_id,
                source_timestamp=frame_id * 0.1,
                captured_monotonic=time.monotonic(),
                image=np.zeros((120, 160, 3), dtype=np.uint8),
            )

    source = CyclingSource()
    results: list[FrameResult] = []
    pipeline: TrackingPipeline

    def on_result(result: FrameResult) -> None:
        results.append(result)
        if len(results) >= 3:
            pipeline.stop()

    pipeline = TrackingPipeline(
        source,  # type: ignore[arg-type]
        FixedDetector(),
        ByteTrackTracker(TrackerConfig()),
        reconnect=True,
        reconnect_initial_seconds=0.01,
        reconnect_max_seconds=0.02,
        on_result=on_result,
    )
    pipeline.start()

    assert pipeline.join(timeout=5.0)
    assert source.open_count >= 2
    assert len(results) >= 3


def test_slow_analytics_worker_does_not_block_inference_worker(tmp_path: Path) -> None:
    video_path = tmp_path / "slow-analytics.mp4"
    make_video(video_path, frame_count=20)
    config = AppConfig(
        video=VideoConfig(
            queue_size=2,
            analytics_queue_size=100,
            stale_frame_policy="block",
        )
    )
    session = ProcessingSession(
        camera_id="test-camera",
        source_uri=str(video_path),
        source_kind="upload",
        realtime=False,
        detector=FixedDetector(),
        config=config,
    )
    original = session.analytics.process_frame

    def slow_process(result: FrameResult):
        time.sleep(0.03)
        return original(result)

    session.analytics.process_frame = slow_process  # type: ignore[method-assign]
    session.start()
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and session.pipeline.stats.processed_frames < 20:
        time.sleep(0.01)

    assert session.pipeline.stats.processed_frames == 20
    assert session.snapshot().status == "running"
    assert session.metrics()["analytics_queue_size"] > 0
    assert session.wait(timeout=10.0)
    assert session.snapshot().status == "completed"


def test_processing_session_runs_directional_grid_with_camera_id(tmp_path: Path) -> None:
    video_path = tmp_path / "directional-grid.mp4"
    make_video(video_path)
    config = AppConfig(
        analytics=AnalyticsConfig(
            common_path=CommonPathConfig(engine="directional_grid"),
        ),
    )
    session = ProcessingSession(
        camera_id="directional-camera",
        source_uri=str(video_path),
        source_kind="upload",
        realtime=False,
        detector=FixedDetector(),
        config=config,
    )

    session.start()

    assert session.wait(timeout=10.0)
    assert session.snapshot().status == "completed"
    assert session.analytics.directional_path.camera_id == "directional-camera"
    assert all(
        point.frame_id == session.snapshot().frame_id
        for point in session.current_points()
    )
