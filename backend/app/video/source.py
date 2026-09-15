from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2

from backend.app.schemas import FramePacket


@dataclass(frozen=True, slots=True)
class VideoMetadata:
    width: int
    height: int
    fps: float
    frame_count: int


class OpenCVVideoSource:
    def __init__(self, source: str | Path, *, max_frames: int | None = None) -> None:
        if max_frames is not None and max_frames < 1:
            raise ValueError("max_frames must be positive")
        self.source = str(source)
        self.max_frames = max_frames
        self.is_rtsp = self.source.lower().startswith(("rtsp://", "rtsps://"))
        self._capture: cv2.VideoCapture | None = None
        self.metadata: VideoMetadata | None = None
        self._next_frame_id = 0

    def open(self) -> VideoMetadata:
        self.close()
        capture = cv2.VideoCapture(self.source)
        if not capture.isOpened():
            capture.release()
            raise ValueError(f"Unable to open video source: {self.source}")
        self._capture = capture
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        self.metadata = VideoMetadata(
            width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=fps if fps > 0 else 30.0,
            frame_count=int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        )
        return self.metadata

    def frames(self) -> Iterator[FramePacket]:
        if self._capture is None or self.metadata is None:
            self.open()
        assert self._capture is not None
        assert self.metadata is not None

        while True:
            if self.max_frames is not None and self._next_frame_id >= self.max_frames:
                break
            decode_started = time.perf_counter()
            ok, image = self._capture.read()
            decode_ms = (time.perf_counter() - decode_started) * 1000.0
            if not ok:
                break
            source_ms = float(self._capture.get(cv2.CAP_PROP_POS_MSEC))
            frame_id = self._next_frame_id
            source_timestamp = (
                frame_id / self.metadata.fps
                if self.is_rtsp
                else source_ms / 1000.0
                if source_ms > 0
                else frame_id / self.metadata.fps
            )
            yield FramePacket(
                frame_id=frame_id,
                source_timestamp=source_timestamp,
                captured_monotonic=time.monotonic(),
                image=image,
                decode_ms=decode_ms,
            )
            self._next_frame_id += 1

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
        self._capture = None
