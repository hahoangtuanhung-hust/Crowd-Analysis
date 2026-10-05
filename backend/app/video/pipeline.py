from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from queue import Empty, Full

from backend.app.inference import PersonDetector
from backend.app.schemas import FramePacket, FrameResult
from backend.app.tracking import MultiObjectTracker
from backend.app.video.latest_queue import BoundedFrameQueue
from backend.app.video.motion_roi import MotionROIPlan, MotionROIPlanner
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
    reference_scans: int = 0
    roi_scans: int = 0
    intentionally_skipped_scans: int = 0
    detector_images: int = 0
    # Sampled detection metrics
    policy_skipped_frames: int = 0
    association_updates: int = 0
    prediction_steps: int = 0
    detection_scan_deadline_misses: int = 0
    detector_images_total: int = 0
    detector_images_far: int = 0
    detector_images_middle: int = 0
    detector_images_near: int = 0
    detector_images_full: int = 0
    detections_far_raw: int = 0
    detections_middle_raw: int = 0
    detections_near_raw: int = 0
    detections_after_merge: int = 0
    perspective_preprocess_ms: float = 0.0
    perspective_detector_ms: float = 0.0
    perspective_merge_ms: float = 0.0


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
        motion_scheduler: MotionROIPlanner | None = None,
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
        self.motion_scheduler = motion_scheduler
        self.on_result = on_result
        self._stop = threading.Event()
        self._done = threading.Event()
        self._capture_thread: threading.Thread | None = None
        self._processing_thread: threading.Thread | None = None
        self._captured_frames = 0
        self._processed_frames = 0
        self._error: str | None = None
        self._last_tracks = ()
        self._reference_scans = 0
        self._roi_scans = 0
        self._intentionally_skipped_scans = 0
        self._detector_images = 0
        self._policy_skipped_frames = 0
        self._association_updates = 0
        self._prediction_steps = 0
        self._detection_scan_deadline_misses = 0
        self._detector_images_total = 0
        self._detector_images_far = 0
        self._detector_images_middle = 0
        self._detector_images_near = 0
        self._detector_images_full = 0
        self._detections_far_raw = 0
        self._detections_middle_raw = 0
        self._detections_near_raw = 0
        self._detections_after_merge = 0
        self._perspective_preprocess_ms = 0.0
        self._perspective_detector_ms = 0.0
        self._perspective_merge_ms = 0.0
        # Deadline-based detection scheduler state
        # Tracks the source frame_id of the last detection scan so we use
        # "frames since last scan >= N" rather than frame_id % N == 0.
        # This handles live frame drops: if frame 5 was dropped, frame 6
        # is detected on schedule instead of waiting until frame 10.
        self._last_detection_frame_id: int | None = None

    @property
    def stats(self) -> PipelineStats:
        return PipelineStats(
            captured_frames=self._captured_frames,
            processed_frames=self._processed_frames,
            dropped_frames=self._frames.dropped,
            queue_size=self._frames.size,
            running=not self._done.is_set() and self._capture_thread is not None,
            error=self._error,
            reference_scans=self._reference_scans,
            roi_scans=self._roi_scans,
            intentionally_skipped_scans=self._intentionally_skipped_scans,
            detector_images=self._detector_images,
            policy_skipped_frames=self._policy_skipped_frames,
            association_updates=self._association_updates,
            prediction_steps=self._prediction_steps,
            detection_scan_deadline_misses=self._detection_scan_deadline_misses,
            detector_images_total=self._detector_images_total,
            detector_images_far=self._detector_images_far,
            detector_images_middle=self._detector_images_middle,
            detector_images_near=self._detector_images_near,
            detector_images_full=self._detector_images_full,
            detections_far_raw=self._detections_far_raw,
            detections_middle_raw=self._detections_middle_raw,
            detections_near_raw=self._detections_near_raw,
            detections_after_merge=self._detections_after_merge,
            perspective_preprocess_ms=self._perspective_preprocess_ms,
            perspective_detector_ms=self._perspective_detector_ms,
            perspective_merge_ms=self._perspective_merge_ms,
        )

    def _is_detection_frame(self, frame_id: int) -> bool:
        """Deadline-based detection scheduler.

        Instead of ``frame_id % N == 0`` (which misses the schedule when
        exactly that frame is dropped from the live queue), we trigger
        detection on the first frame that is *at least N source frames* away
        from the last detection scan.  The schedule is updated from the
        actual scanned frame so subsequent deadlines stay correctly spaced.

        On the very first frame of an epoch the scheduler always detects.
        """
        if self.inference_interval <= 1:
            return True
        if self._last_detection_frame_id is None:
            # First frame of epoch — always detect
            return True
        gap = frame_id - self._last_detection_frame_id
        if gap < 0:
            # frame_id went backwards (epoch reset) — detect to resync
            return True
        if gap >= self.inference_interval:
            if gap > self.inference_interval:
                # The exact deadline frame was dropped; we're running late
                self._detection_scan_deadline_misses += 1
            return True
        return False

    def start(self) -> None:
        if self._capture_thread is not None and self._capture_thread.is_alive():
            raise RuntimeError("Pipeline is already running")
        self._stop.clear()
        self._done.clear()
        self._error = None
        self.tracker.reset()
        if self.motion_scheduler is not None:
            self.motion_scheduler.reset()
        self._last_tracks = ()
        self._reference_scans = 0
        self._roi_scans = 0
        self._intentionally_skipped_scans = 0
        self._detector_images = 0
        self._policy_skipped_frames = 0
        self._association_updates = 0
        self._prediction_steps = 0
        self._detection_scan_deadline_misses = 0
        self._last_detection_frame_id = None
        self.source.open()
        self._capture_thread = threading.Thread(
            target=self._capture_loop,
            name="camera-reader",
            daemon=True,
        )
        self._processing_thread = threading.Thread(
            target=self._processing_loop,
            name=(
                "cache-replay-worker"
                if callable(getattr(self.detector, "detect_packet", None))
                else "inference-worker"
            ),
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

                # --- Sampled detection scheduler ---
                # Use deadline-based logic instead of modulo so live frame
                # drops do not permanently stall the detection schedule.
                run_detection = self._is_detection_frame(item.frame_id)
                if not run_detection:
                    # Coast: advance Kalman predictions without running YOLO.
                    # This keeps bbox positions reasonable for preview and
                    # ensures the state advances correctly before the next
                    # real measurement update (no double-predict).
                    coast_fn = getattr(self.tracker, "coast", None)
                    if callable(coast_fn):
                        coast_tracks = coast_fn(
                            item.image,
                            frame_id=item.frame_id,
                            source_timestamp=item.source_timestamp,
                        )
                        self._prediction_steps += 1
                        self._policy_skipped_frames += 1
                        # Emit a coast result so downstream gets state/preview.
                        # observed=False / NOT_SEARCHED_BY_POLICY — these
                        # points must NOT create Common Path evidence.
                        coast_result = FrameResult(
                            packet=item,
                            detections=(),
                            tracks=tuple(coast_tracks),
                            inference_ms=0.0,
                            tracking_ms=0.0,
                            processing_completed_monotonic=time.monotonic(),
                            observation_mode="NOT_SEARCHED_BY_POLICY",
                            searched_regions=(),
                            scheduler_state="CADENCE_SKIP",
                            scheduler_reasons=("cadence_skip",),
                            detector_images=0,
                            intentional_skip=True,
                        )
                        self._processed_frames += 1
                        if self.on_result is not None:
                            self.on_result(coast_result)
                    # If tracker has no coast method, silently skip (N=1 only
                    # reaches here if inference_interval==1, which is guarded).
                    continue

                # This is a detection frame — record it for the deadline scheduler.
                self._last_detection_frame_id = item.frame_id

                packet_detector = getattr(self.detector, "detect_packet", None)
                if callable(packet_detector) or self.motion_scheduler is None:
                    inference_started = time.perf_counter()
                    detections = (
                        packet_detector(item)
                        if callable(packet_detector)
                        else self.detector.detect(item.image)
                    )
                    inference_ms = (time.perf_counter() - inference_started) * 1000.0
                    tracking_started = time.perf_counter()
                    tracks = self.tracker.update(
                        detections,
                        item.image,
                        frame_id=item.frame_id,
                        source_timestamp=item.source_timestamp,
                    )
                    tracking_ms = (time.perf_counter() - tracking_started) * 1000.0
                    self._association_updates += 1
                    height, width = item.image.shape[:2]
                    observation_mode = "FULL_COVERAGE"
                    searched_regions = ((0, 0, width, height),)
                    scheduler_state = (
                        "DISABLED" if self.motion_scheduler is None else "CACHE_REPLAY"
                    )
                    scheduler_reasons: tuple[str, ...] = ()
                    motion_ms = 0.0
                    planning_ms = 0.0
                    union_ratio = 1.0
                    # Cache replay performs no model work.  A normal detector
                    # may execute a full frame plus several tiles, so never
                    # hard-code this counter to one.
                    if callable(packet_detector):
                        detector_images = 0
                    else:
                        image_count_provider = getattr(self.detector, "reference_image_count", None)
                        detector_images = (
                            int(image_count_provider(width, height))
                            if callable(image_count_provider)
                            else 1
                        )
                    self._detector_images += detector_images
                    intentional_skip = False
                else:
                    (
                        detections,
                        tracks,
                        inference_ms,
                        tracking_ms,
                        plan,
                        detector_images,
                    ) = self._process_scheduled(item)
                    observation_mode = {
                        "reference": "FULL_COVERAGE",
                        "tiles": "SELECTIVE_COVERAGE",
                        "skip": "NOT_SEARCHED_BY_POLICY",
                    }[plan.scan_type]
                    searched_regions = plan.searched_regions
                    scheduler_state = plan.state
                    scheduler_reasons = plan.reasons
                    motion_ms = plan.motion_ms
                    planning_ms = plan.planning_ms
                    union_ratio = plan.roi_union_ratio
                    intentional_skip = plan.scan_type == "skip"

                tracks = self._label_coverage(tracks, searched_regions, observation_mode)
                self._last_tracks = tuple(tracks)

                result = FrameResult(
                    packet=item,
                    detections=tuple(detections),
                    tracks=tuple(tracks),
                    inference_ms=inference_ms,
                    tracking_ms=tracking_ms,
                    processing_completed_monotonic=time.monotonic(),
                    observation_mode=observation_mode,
                    searched_regions=tuple(searched_regions),
                    scheduler_state=scheduler_state,
                    scheduler_reasons=scheduler_reasons,
                    motion_ms=motion_ms,
                    roi_planning_ms=planning_ms,
                    roi_union_ratio=union_ratio,
                    detector_images=detector_images,
                    intentional_skip=intentional_skip,
                )

                det_stats = getattr(self.detector, "last_inference_stats", {}) or {}
                self._detector_images_total += det_stats.get("detector_images_total", 0)
                self._detector_images_far += det_stats.get("detector_images_far", 0)
                self._detector_images_middle += det_stats.get("detector_images_middle", 0)
                self._detector_images_near += det_stats.get("detector_images_near", 0)
                self._detector_images_full += det_stats.get("detector_images_full", 0)
                self._detections_far_raw += det_stats.get("detections_far_raw", 0)
                self._detections_middle_raw += det_stats.get("detections_middle_raw", 0)
                self._detections_near_raw += det_stats.get("detections_near_raw", 0)
                self._detections_after_merge += det_stats.get("detections_after_merge", 0)
                self._perspective_preprocess_ms += det_stats.get("perspective_preprocess_ms", 0.0)
                self._perspective_detector_ms += det_stats.get("perspective_detector_ms", 0.0)
                self._perspective_merge_ms += det_stats.get("perspective_merge_ms", 0.0)

                self._processed_frames += 1
                if self.on_result is not None:
                    self.on_result(result)
        except Exception as exc:  # worker boundary must preserve the service process
            self._error = f"processing: {exc}"
            LOGGER.exception("Inference worker failed")
            self._stop.set()
        finally:
            self._done.set()

    def _process_scheduled(
        self,
        item: FramePacket,
    ) -> tuple[list, list, float, float, MotionROIPlan, int]:
        assert self.motion_scheduler is not None
        height, width = item.image.shape[:2]
        tile_provider = getattr(self.detector, "tile_regions", None)
        tile_regions = tuple(tile_provider(width, height)) if callable(tile_provider) else ()
        image_count_provider = getattr(self.detector, "reference_image_count", None)
        reference_image_count = (
            int(image_count_provider(width, height)) if callable(image_count_provider) else 1
        )
        plan = self.motion_scheduler.plan(
            item.image,
            item.source_timestamp,
            tile_regions=tile_regions,
            reference_image_count=reference_image_count,
            protected_boxes=tuple(
                (int(track.x1), int(track.y1), int(track.x2), int(track.y2))
                for track in self._last_tracks
            ),
        )
        if plan.scan_type == "tiles" and any(
            not self._box_fully_covered(track.xyxy, plan.searched_regions)
            for track in self._last_tracks
        ):
            # Selective inference is only safe for an ordinary ByteTrack update
            # when every previously published track was actually searched.
            # Otherwise an unmatched out-of-ROI track would be marked lost.
            plan = replace(
                plan,
                scan_type="reference",
                searched_regions=((0, 0, width, height),),
                reasons=plan.reasons + ("TRACK_COVERAGE_REFERENCE_FALLBACK",),
            )

        inference_started = time.perf_counter()
        if plan.scan_type == "reference":
            detections = self.detector.detect(item.image)
            detector_images = reference_image_count
            self._reference_scans += 1
        elif plan.scan_type == "tiles":
            region_detector = getattr(self.detector, "detect_regions", None)
            if not callable(region_detector):
                # A backend without selective-inference support must retain
                # reference coverage; never pretend that an ROI scan happened.
                detections = self.detector.detect(item.image)
                detector_images = reference_image_count
                self._reference_scans += 1
                plan = replace(
                    plan,
                    scan_type="reference",
                    searched_regions=((0, 0, width, height),),
                    reasons=plan.reasons + ("BACKEND_REFERENCE_FALLBACK",),
                )
            else:
                detections = region_detector(item.image, plan.selected_tiles)
                detector_images = len(plan.selected_tiles)
                self._roi_scans += 1
        else:
            detections = []
            detector_images = 0
            self._intentionally_skipped_scans += 1
        inference_ms = (time.perf_counter() - inference_started) * 1000.0
        self._detector_images += detector_images

        tracking_started = time.perf_counter()
        if plan.scan_type == "skip":
            coast_fn = getattr(self.tracker, "coast", None)
            if not callable(coast_fn):
                raise RuntimeError(
                    "motion ROI skip requires tracker.coast; refusing update([]) semantics"
                )
            tracks = coast_fn(
                item.image,
                frame_id=item.frame_id,
                source_timestamp=item.source_timestamp,
            )
            self._prediction_steps += 1
        else:
            tracks = self.tracker.update(
                detections,
                item.image,
                frame_id=item.frame_id,
                source_timestamp=item.source_timestamp,
            )
            self._association_updates += 1
        tracking_ms = (time.perf_counter() - tracking_started) * 1000.0
        return detections, tracks, inference_ms, tracking_ms, plan, detector_images

    @staticmethod
    def _label_coverage(tracks, searched_regions, observation_mode: str):
        labeled = []
        for track in tracks:
            if track.observed:
                coverage = "MEASURED"
            elif observation_mode == "NOT_SEARCHED_BY_POLICY":
                coverage = "NOT_SEARCHED_BY_POLICY"
            else:
                searched = TrackingPipeline._box_fully_covered(track.xyxy, searched_regions)
                coverage = "SEARCHED_NOT_FOUND" if searched else "NOT_SEARCHED_BY_POLICY"
            labeled.append(replace(track, observation_coverage=coverage))
        return labeled

    @staticmethod
    def _box_fully_covered(box, regions) -> bool:
        bx1, by1, bx2, by2 = map(float, box)
        box_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        if box_area <= 0.0:
            return False
        clipped = []
        for region in regions:
            x1, y1, x2, y2 = map(float, region)
            item = (max(bx1, x1), max(by1, y1), min(bx2, x2), min(by2, y2))
            if item[2] > item[0] and item[3] > item[1]:
                clipped.append(item)
        if not clipped:
            return False
        xs = sorted({value for item in clipped for value in (item[0], item[2])})
        covered = 0.0
        for left, right in zip(xs, xs[1:]):
            intervals = sorted(
                (top, bottom) for x1, top, x2, bottom in clipped if x1 < right and x2 > left
            )
            if not intervals:
                continue
            start, end = intervals[0]
            height = 0.0
            for top, bottom in intervals[1:]:
                if top <= end:
                    end = max(end, bottom)
                else:
                    height += end - start
                    start, end = top, bottom
            covered += (right - left) * (height + end - start)
        return covered >= box_area * (1.0 - 1e-6)
