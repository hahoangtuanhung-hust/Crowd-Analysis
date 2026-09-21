from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from queue import Empty, Full

from backend.app.inference import PersonDetector
from backend.app.schemas import FramePacket, FrameResult
from backend.app.tracking import MultiObjectTracker
from backend.app.video.latest_queue import BoundedFrameQueue
from backend.app.video.source import OpenCVVideoSource

LOGGER = logging.getLogger(__name__)
_END = object()


@dataclass(frozen=True, slots=True)
class PipelineStats:
    captured_frames: int
    processed_frames: int
    dropped_frames: int
    queue_size: int
    running: bool
    error: str | None


class TrackingPipeline:
    """Two-worker bounded pipeline for capture, detection, and tracking."""

    def __init__(
        self,
        source: OpenCVVideoSource,
        detector: PersonDetector,
        tracker: MultiObjectTracker,
        *,
        queue_size: int = 4,
        drop_oldest: bool = True,
        inference_interval: int = 1,
        reconnect: bool = False,
        reconnect_initial_seconds: float = 1.0,
        reconnect_max_seconds: float = 30.0,
        on_result: Callable[[FrameResult], None] | None = None,
    ) -> None:
        if inference_interval < 1:
            raise ValueError("inference_interval must be at least 1")
        self.source = source
        self.detector = detector
        self.tracker = tracker
        self._frames: BoundedFrameQueue[FramePacket | object] = BoundedFrameQueue(
            maxsize=queue_size,
            drop_oldest=drop_oldest,
        )
        self.inference_interval = inference_interval
        self.reconnect = reconnect
        self.reconnect_initial_seconds = reconnect_initial_seconds
        self.reconnect_max_seconds = reconnect_max_seconds
        self.on_result = on_result
        self._stop = threading.Event()
        self._done = threading.Event()
        self._capture_thread: threading.Thread | None = None
        self._processing_thread: threading.Thread | None = None
        self._captured_frames = 0
        self._processed_frames = 0
        self._error: str | None = None

    @property
    def stats(self) -> PipelineStats:
        return PipelineStats(
            captured_frames=self._captured_frames,
            processed_frames=self._processed_frames,
            dropped_frames=self._frames.dropped,
            queue_size=self._frames.size,
            running=not self._done.is_set() and self._capture_thread is not None,
            error=self._error,
        )

    def start(self) -> None:
        if self._capture_thread is not None and self._capture_thread.is_alive():
            raise RuntimeError("Pipeline is already running")
        self._stop.clear()
        self._done.clear()
        self._error = None
        self.tracker.reset()
        self.source.open()
        self._capture_thread = threading.Thread(
            target=self._capture_loop,
            name="camera-reader",
            daemon=True,
        )
        self._processing_thread = threading.Thread(
            target=self._processing_loop,
            name=("cache-replay-worker" if callable(getattr(self.detector, "detect_packet", None))
                  else "inference-worker"),
            daemon=True,
        )
        self._processing_thread.start()
        self._capture_thread.start()

    def stop(self) -> None:
        self._stop.set()
        self.source.close()
        self._put_end_marker()

    def join(self, timeout: float | None = None) -> bool:
        started = time.monotonic()
        for thread in (self._capture_thread, self._processing_thread):
            if thread is None:
                continue
            remaining = None
            if timeout is not None:
                remaining = max(0.0, timeout - (time.monotonic() - started))
            thread.join(remaining)
        return self._done.is_set()

    def _capture_loop(self) -> None:
        try:
            backoff = self.reconnect_initial_seconds
            while not self._stop.is_set():
                try:
                    for packet in self.source.frames():
                        if self._stop.is_set():
                            break
                        self._captured_frames += 1
                        while not self._stop.is_set():
                            try:
                                self._frames.put(packet)
                                break
                            except Full:
                                continue
                        backoff = self.reconnect_initial_seconds
                except Exception:
                    if not self.reconnect:
                        raise
                    LOGGER.exception("Video source read failed")
                if not self.reconnect or self._stop.is_set():
                    break
                self.source.close()
                LOGGER.warning("Video source disconnected; retrying in %.1f seconds", backoff)
                if self._stop.wait(backoff):
                    break
                try:
                    self.source.open()
                except Exception:
                    LOGGER.exception("Video source reconnect attempt failed")
                backoff = min(self.reconnect_max_seconds, backoff * 2.0)
        except Exception as exc:  # worker boundary must preserve the service process
            self._error = f"capture: {exc}"
            LOGGER.exception("Capture worker failed")
        finally:
            self.source.close()
            self._put_end_marker()

    def _put_end_marker(self) -> None:
        while not self._done.is_set():
            try:
                self._frames.put(_END)
                return
            except Full:
                self._stop.wait(0.05)

    def _processing_loop(self) -> None:
        try:
            while not self._stop.is_set():
                try:
                    item = self._frames.get(timeout=0.1)
                except Empty:
                    continue
                if item is _END:
                    break
                assert isinstance(item, FramePacket)
                if item.frame_id % self.inference_interval != 0:
                    continue

                inference_started = time.perf_counter()
                packet_detector = getattr(self.detector, "detect_packet", None)
                detections = (
                    packet_detector(item)
                    if callable(packet_detector)
                    else self.detector.detect(item.image)
                )
                inference_ms = (time.perf_counter() - inference_started) * 1000.0

                tracking_started = time.perf_counter()
                tracks = self.tracker.update(detections, item.image)
                tracking_ms = (time.perf_counter() - tracking_started) * 1000.0

                result = FrameResult(
                    packet=item,
                    detections=tuple(detections),
                    tracks=tuple(tracks),
                    inference_ms=inference_ms,
                    tracking_ms=tracking_ms,
                    processing_completed_monotonic=time.monotonic(),
                )
                self._processed_frames += 1
                if self.on_result is not None:
                    self.on_result(result)
        except Exception as exc:  # worker boundary must preserve the service process
            self._error = f"processing: {exc}"
            LOGGER.exception("Inference worker failed")
            self._stop.set()
        finally:
            self._done.set()
