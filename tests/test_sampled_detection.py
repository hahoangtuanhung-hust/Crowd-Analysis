"""Tests for sampled-detection pipeline: deadline-based scheduler,
coast() semantics, and metric accounting.

Covers the requirements from prompt_detection_theo_chu_ky_tracking_tracklet.md:
- N=1,3,5,10 select correct frames; first frame/EOF/short clip
- Invalid config rejected early
- Dropped deadline frames handled without starvation (deadline-based, not modulo)
- Coast vs update semantics: skip != miss, no double-predict
- policy_skipped_frames, association_updates, prediction_steps counted correctly
- Coast result emitted with NOT_SEARCHED_BY_POLICY (no Common Path evidence)
"""
from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from backend.app.core.config import TrackerConfig, VideoConfig
from backend.app.schemas import Detection, FramePacket, FrameResult
from backend.app.tracking import ByteTrackTracker
from backend.app.video import OpenCVVideoSource, TrackingPipeline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _StubDetector:
    """Detector that records which frame_ids it was called with."""

    def __init__(self, detections: list[Detection] | None = None) -> None:
        self.called_frame_ids: list[int] = []
        self._detections = detections or []

    def detect(self, _frame: np.ndarray) -> list[Detection]:
        return list(self._detections)

    def detect_packet(self, packet: FramePacket) -> list[Detection]:
        self.called_frame_ids.append(packet.frame_id)
        return list(self._detections)


class _CoastAwareTracker:
    """Minimal tracker that records update vs coast calls."""

    def __init__(self) -> None:
        self.update_calls: list[int] = []   # frame_ids passed to update()
        self.coast_calls: list[int] = []    # frame_ids passed to coast()

    def reset(self) -> None:
        self.update_calls.clear()
        self.coast_calls.clear()

    def update(
        self,
        detections,
        frame,
        *,
        frame_id: int | None = None,
        source_timestamp: float | None = None,
    ) -> list:
        if frame_id is not None:
            self.update_calls.append(frame_id)
        return []

    def coast(
        self,
        frame,
        *,
        frame_id: int | None = None,
        source_timestamp: float | None = None,
    ) -> list:
        if frame_id is not None:
            self.coast_calls.append(frame_id)
        return []


def _make_video(path: Path, frame_count: int = 20, fps: float = 30.0) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (160, 120),
    )
    assert writer.isOpened()
    for _ in range(frame_count):
        writer.write(np.zeros((120, 160, 3), dtype=np.uint8))
    writer.release()


def _run_pipeline(
    tmp_path: Path,
    *,
    inference_interval: int,
    frame_count: int = 20,
    tracker=None,
    detector=None,
) -> tuple[list[FrameResult], object]:
    """Run pipeline and return (results, stats)."""
    video_path = tmp_path / "test.mp4"
    _make_video(video_path, frame_count=frame_count)
    results: list[FrameResult] = []
    if tracker is None:
        tracker = ByteTrackTracker(TrackerConfig())
    if detector is None:
        detector = _StubDetector()

    pipeline = TrackingPipeline(
        OpenCVVideoSource(video_path),
        detector,
        tracker,
        queue_size=32,
        drop_oldest=False,
        inference_interval=inference_interval,
        on_result=results.append,
    )
    pipeline.start()
    assert pipeline.join(timeout=15.0)
    return results, pipeline.stats


# ---------------------------------------------------------------------------
# Scheduler correctness
# ---------------------------------------------------------------------------

def test_n1_detects_every_frame(tmp_path: Path) -> None:
    """N=1 must detect every frame — baseline unchanged."""
    detector = _StubDetector()
    results, stats = _run_pipeline(tmp_path, inference_interval=1, frame_count=10, detector=detector)
    assert len(results) == 10
    assert stats.association_updates == 10
    assert stats.prediction_steps == 0
    assert stats.policy_skipped_frames == 0
    # Every result should be FULL_COVERAGE
    assert all(r.observation_mode == "FULL_COVERAGE" for r in results)


def test_n3_detects_correct_frames(tmp_path: Path) -> None:
    """N=3: detect at frames 0, 3, 6, 9 … (first + every 3rd)."""
    tracker = _CoastAwareTracker()
    detector = _StubDetector()
    results, stats = _run_pipeline(
        tmp_path, inference_interval=3, frame_count=12, tracker=tracker, detector=detector
    )
    assert len(results) == 12, "All frames should produce a result (detect or coast)"
    # Detection frames: 0, 3, 6, 9 = 4 scans
    assert tracker.update_calls == [0, 3, 6, 9]
    # Coast frames: all others = 8
    assert len(tracker.coast_calls) == 8
    # Metrics
    assert stats.association_updates == 4
    assert stats.prediction_steps == 8
    assert stats.policy_skipped_frames == 8


def test_n5_detects_correct_frames(tmp_path: Path) -> None:
    """N=5: detect at frames 0, 5, 10, 15."""
    tracker = _CoastAwareTracker()
    results, stats = _run_pipeline(
        tmp_path, inference_interval=5, frame_count=20, tracker=tracker
    )
    assert len(results) == 20
    assert tracker.update_calls == [0, 5, 10, 15]
    assert stats.association_updates == 4
    assert stats.prediction_steps == 16


def test_n10_detects_correct_frames(tmp_path: Path) -> None:
    """N=10: detect at frames 0, 10."""
    tracker = _CoastAwareTracker()
    results, stats = _run_pipeline(
        tmp_path, inference_interval=10, frame_count=20, tracker=tracker
    )
    assert tracker.update_calls == [0, 10]
    assert stats.association_updates == 2
    assert stats.prediction_steps == 18


def test_first_frame_always_detected(tmp_path: Path) -> None:
    """First frame of epoch must be a detection scan regardless of N."""
    for n in (3, 5, 10):
        tracker = _CoastAwareTracker()
        _run_pipeline(tmp_path, inference_interval=n, frame_count=2, tracker=tracker)
        assert tracker.update_calls[0] == 0, f"N={n}: first frame must be detected"


def test_short_clip_eof(tmp_path: Path) -> None:
    """Short clip (fewer frames than N) must still produce results."""
    tracker = _CoastAwareTracker()
    results, stats = _run_pipeline(
        tmp_path, inference_interval=10, frame_count=3, tracker=tracker
    )
    assert len(results) == 3
    assert tracker.update_calls == [0]  # Only frame 0 detected
    assert stats.association_updates == 1


def test_invalid_inference_interval_rejected() -> None:
    """inference_interval < 1 must be rejected at construction time."""
    with pytest.raises(ValueError, match="inference_interval"):
        TrackingPipeline(
            OpenCVVideoSource("dummy.mp4"),
            _StubDetector(),
            ByteTrackTracker(TrackerConfig()),
            inference_interval=0,
        )


# ---------------------------------------------------------------------------
# Deadline-based scheduler: handles dropped frames
# ---------------------------------------------------------------------------

def test_deadline_scheduler_handles_dropped_deadline_frame(tmp_path: Path) -> None:
    """If the exact deadline frame is dropped, the next available frame
    should be detected (no permanent starvation from modulo mismatch)."""
    # Simulate: frames 0, 1, 2, 4, 5, 6, 8... (frame 3 and 7 dropped)
    # With N=3 and deadline-based:
    #   frame 0 -> detect (first)
    #   frame 1 -> coast
    #   frame 2 -> coast
    #   frame 4 -> detect (3 was dropped; 4-0=4 >= 3)
    #   frame 5 -> coast
    #   frame 6 -> coast
    #   frame 8 -> detect (7 was dropped; 8-4=4 >= 3)
    tracker = _CoastAwareTracker()

    class _PlainDetector:
        """Plain detect() detector so the pipeline uses _is_detection_frame path."""
        def detect(self, _frame: np.ndarray) -> list[Detection]:
            return []

    class DroppingSource:
        """Source that skips frames 3 and 7."""
        def open(self) -> None: pass
        def close(self) -> None: pass
        def frames(self):
            for fid in [0, 1, 2, 4, 5, 6, 8, 9, 10]:
                yield FramePacket(
                    frame_id=fid,
                    source_timestamp=fid / 30.0,
                    captured_monotonic=time.monotonic(),
                    image=np.zeros((120, 160, 3), dtype=np.uint8),
                )

    results: list[FrameResult] = []
    pipeline = TrackingPipeline(
        DroppingSource(),  # type: ignore[arg-type]
        _PlainDetector(),  # type: ignore[arg-type]
        tracker,
        inference_interval=3,
        queue_size=32,
        drop_oldest=False,
        on_result=results.append,
    )
    pipeline.start()
    assert pipeline.join(timeout=10.0)
    # Should detect at 0, 4 (deadline missed 3), 8 (deadline missed 7)
    assert tracker.update_calls == [0, 4, 8], (
        f"Expected detection at 0,4,8 but got {tracker.update_calls}"
    )
    assert pipeline.stats.detection_scan_deadline_misses == 2


# ---------------------------------------------------------------------------
# Skip semantics: coast != miss
# ---------------------------------------------------------------------------

def test_coast_frames_are_not_searched_by_policy(tmp_path: Path) -> None:
    """Coast/skip results must have observation_mode NOT_SEARCHED_BY_POLICY."""
    results, _ = _run_pipeline(tmp_path, inference_interval=3, frame_count=9)
    skip_results = [r for r in results if r.intentional_skip]
    detect_results = [r for r in results if not r.intentional_skip]
    assert len(skip_results) == 6, f"Expected 6 coast frames, got {len(skip_results)}"
    assert len(detect_results) == 3
    for r in skip_results:
        assert r.observation_mode == "NOT_SEARCHED_BY_POLICY"
        assert r.detector_images == 0
        # Coast results must not have observed=True tracks
        for track in r.tracks:
            assert not track.observed, "Coast tracks must not be marked observed"


def test_skip_does_not_increase_association_count(tmp_path: Path) -> None:
    """policy_skipped_frames and association_updates must be disjoint."""
    _, stats = _run_pipeline(tmp_path, inference_interval=5, frame_count=20)
    total = stats.association_updates + stats.policy_skipped_frames
    assert total == 20, (
        f"association_updates({stats.association_updates}) + "
        f"policy_skipped_frames({stats.policy_skipped_frames}) should equal "
        f"processed_frames(20)"
    )


def test_detection_results_are_full_coverage(tmp_path: Path) -> None:
    """Detection frames must have observation_mode FULL_COVERAGE."""
    results, _ = _run_pipeline(tmp_path, inference_interval=5, frame_count=10)
    detect_results = [r for r in results if not r.intentional_skip]
    for r in detect_results:
        assert r.observation_mode == "FULL_COVERAGE"
        assert not r.intentional_skip


# ---------------------------------------------------------------------------
# No double-predict: coast then update
# ---------------------------------------------------------------------------

def test_no_double_predict_after_coast(tmp_path: Path) -> None:
    """After coasting frames 1-4, update at frame 5 must not advance
    state by the full gap again. Verify by checking coast_calls then
    update at frame 5 only — not that update is called extra times."""
    tracker = _CoastAwareTracker()
    _run_pipeline(tmp_path, inference_interval=5, frame_count=10, tracker=tracker)
    # Frame 0 -> update, frames 1-4 -> coast, frame 5 -> update, frames 6-9 -> coast
    assert tracker.update_calls == [0, 5]
    assert len(tracker.coast_calls) == 8
    # No frame_id should appear in both lists
    update_set = set(tracker.update_calls)
    coast_set = set(tracker.coast_calls)
    assert update_set.isdisjoint(coast_set), "A frame must not appear in both update and coast"


# ---------------------------------------------------------------------------
# Metric consistency
# ---------------------------------------------------------------------------

def test_stats_processed_equals_captured_when_no_drop(tmp_path: Path) -> None:
    """All captured frames should produce a result (detect or coast)."""
    for n in (1, 3, 5):
        _, stats = _run_pipeline(tmp_path, inference_interval=n, frame_count=15)
        assert stats.processed_frames == 15, f"N={n}: expected 15 processed frames"
        assert stats.dropped_frames == 0


def test_detector_images_zero_for_coast_frames(tmp_path: Path) -> None:
    """detector_images counter must not include coast/skip frames."""

    class TiledDetector:
        """Plain detect() detector with tiled image count (no detect_packet)."""
        @staticmethod
        def detect(_frame: np.ndarray) -> list[Detection]:
            return []

        @staticmethod
        def reference_image_count(_w: int, _h: int) -> int:
            return 5

    detector = TiledDetector()
    results, stats = _run_pipeline(
        tmp_path, inference_interval=5, frame_count=10, detector=detector
    )
    # 2 detection scans (frame 0, 5) * 5 images = 10
    assert stats.detector_images == 10
    # Coast results must report 0 detector_images
    coast_results = [r for r in results if r.intentional_skip]
    for r in coast_results:
        assert r.detector_images == 0
