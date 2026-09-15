from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from backend.app.core.config import TrackerConfig
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
