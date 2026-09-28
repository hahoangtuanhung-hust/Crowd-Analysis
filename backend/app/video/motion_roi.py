from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from typing import Literal, Sequence

import cv2
import numpy as np
from numpy.typing import NDArray

from backend.app.core.config import MotionROIConfig

Rect = tuple[int, int, int, int]
PlannerState = Literal["WARMUP", "HYBRID", "FULL_COVERAGE", "RECOVER"]
ScanType = Literal["reference", "tiles", "skip"]


@dataclass(frozen=True, slots=True)
class MotionROIPlan:
    timestamp_s: float
    state: PlannerState
    scan_type: ScanType
    proposed_scan_type: ScanType
    reasons: tuple[str, ...]
    selected_tiles: tuple[int, ...]
    searched_regions: tuple[Rect, ...]
    motion_tiles: tuple[int, ...]
    protected_tiles: tuple[int, ...]
    foreground_ratio: float
    roi_union_ratio: float
    estimated_cost_ratio: float
    learning_rate: float
    motion_ms: float
    planning_ms: float
    mask_shape: tuple[int, int]
    shadow_mode: bool

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["reasons"] = list(self.reasons)
        value["selected_tiles"] = list(self.selected_tiles)
        value["searched_regions"] = [list(item) for item in self.searched_regions]
        value["motion_tiles"] = list(self.motion_tiles)
        value["protected_tiles"] = list(self.protected_tiles)
        return value


class MotionROIPlanner:
    """Video-time MOG2 planner which selects existing detector tiles.

    The mask only schedules detector work. It never produces detections,
    trajectories, counts, or Common Path evidence.
    """

    def __init__(self, config: MotionROIConfig) -> None:
        self.config = config
        self._subtractor = self._create_subtractor()
        self._first_timestamp: float | None = None
        self._last_timestamp: float | None = None
        self._last_reference_timestamp: float | None = None
        self._healthy_since: float | None = None
        self._recover_since: float | None = None
        self._cost_guard_active = False
        self._state: PlannerState = "WARMUP"
        self._last_mask: NDArray[np.uint8] | None = None

    @property
    def state(self) -> PlannerState:
        return self._state

    @property
    def last_mask(self) -> NDArray[np.uint8] | None:
        return None if self._last_mask is None else self._last_mask.copy()

    def reset(self) -> None:
        self._subtractor = self._create_subtractor()
        self._first_timestamp = None
        self._last_timestamp = None
        self._last_reference_timestamp = None
        self._healthy_since = None
        self._recover_since = None
        self._cost_guard_active = False
        self._state = "WARMUP"
        self._last_mask = None

    def _create_subtractor(self):
        background = self.config.background
        return cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=16.0,
            detectShadows=background.detect_shadows,
        )

    def plan(
        self,
        frame: NDArray[np.uint8],
        timestamp_s: float,
        *,
        tile_regions: Sequence[Rect],
        reference_image_count: int,
        protected_boxes: Sequence[Rect] = (),
    ) -> MotionROIPlan:
        started = time.perf_counter()
        timestamp_s = float(timestamp_s)
        reasons: list[str] = []
        if not math.isfinite(timestamp_s):
            raise ValueError("motion ROI timestamp must be finite")
        if self._last_timestamp is not None and timestamp_s < self._last_timestamp:
            self.reset()
            reasons.append("SOURCE_TIME_DISCONTINUITY")
        if self._first_timestamp is None:
            self._first_timestamp = timestamp_s

        motion_started = time.perf_counter()
        small = self._resize_for_analysis(frame)
        dt = 0.0 if self._last_timestamp is None else max(0.0, timestamp_s - self._last_timestamp)
        tau = self.config.background.learning_time_constant_s
        learning_rate = -1.0 if self._last_timestamp is None else min(
            0.5, max(1e-5, 1.0 - math.exp(-dt / tau))
        )
        raw_mask = self._subtractor.apply(small, learningRate=learning_rate)
        foreground = self._foreground_mask(raw_mask)
        self._last_mask = foreground
        motion_ms = (time.perf_counter() - motion_started) * 1000.0
        foreground_ratio = float(np.count_nonzero(foreground) / max(1, foreground.size))
        mask_healthy = (
            foreground.shape == small.shape[:2]
            and foreground_ratio <= self.config.background.foreground_ratio_max
        )
        # A planner call can be delayed by a bounded/drop-oldest pipeline.  A
        # mask computed from a frame that is too far from the previous sample
        # is not evidence that the intervening region was covered.  Fail safe
        # to reference mode instead of allowing a stale mask to schedule a
        # long skip interval.
        mask_stale = (
            self._last_timestamp is not None
            and timestamp_s - self._last_timestamp
            > self.config.background.max_mask_age_s
        )
        if mask_stale:
            mask_healthy = False
        if mask_healthy:
            if self._healthy_since is None:
                self._healthy_since = timestamp_s
        else:
            self._healthy_since = None

        # A valid video timeline normally starts at exactly 0.0.  Do not use
        # truthiness here: ``0.0 or timestamp_s`` would keep elapsed at zero
        # forever and leave every such video in WARMUP.
        first_timestamp = (
            self._first_timestamp
            if self._first_timestamp is not None
            else timestamp_s
        )
        elapsed = timestamp_s - first_timestamp
        periodic_due = (
            self._last_reference_timestamp is None
            or timestamp_s - self._last_reference_timestamp
            >= self.config.coverage.periodic_scan_interval_s - 1e-9
        )
        motion_tiles = self._motion_tiles(foreground, frame.shape[1], frame.shape[0], tile_regions)
        protected_tiles = self._protected_tiles(
            protected_boxes,
            tile_regions,
            frame.shape[1],
            frame.shape[0],
        )
        selected = tuple(sorted(motion_tiles | protected_tiles))
        union_ratio = self._union_ratio(selected, tile_regions, frame.shape[1], frame.shape[0])
        estimated_cost = len(selected) / max(1, reference_image_count)

        proposed: ScanType
        state: PlannerState
        healthy_duration = (
            timestamp_s - self._healthy_since
            if self._healthy_since is not None
            else 0.0
        )
        mask_stable = healthy_duration >= self.config.background.static_confirm_s

        if not tile_regions:
            proposed, state = "reference", "FULL_COVERAGE"
            reasons.append("REFERENCE_PROFILE_HAS_NO_TILES")
        elif elapsed < self.config.background.warmup_min_s:
            proposed, state = "reference", "WARMUP"
            reasons.append("BACKGROUND_WARMUP")
        elif not mask_healthy or not mask_stable:
            proposed = "reference"
            if mask_stale:
                state = "RECOVER"
                self._recover_since = self._recover_since or timestamp_s
                reasons.append("MASK_STALE")
            elif elapsed < self.config.background.warmup_max_s:
                # The minimum warmup is only the earliest point at which the
                # mask may be trusted.  Keep reference coverage while the
                # initial model is unhealthy, up to the configured maximum.
                state = "WARMUP"
                reasons.append(
                    "BACKGROUND_NOT_READY" if mask_healthy else "MASK_UNHEALTHY"
                )
            else:
                state = "RECOVER"
                if self._recover_since is None:
                    self._recover_since = timestamp_s
                reasons.append("MASK_UNSTABLE" if mask_healthy else "MASK_UNHEALTHY")
        elif self._recover_since is not None and (
            self._healthy_since is None
            or timestamp_s - self._healthy_since
            < self.config.fallback.soft_recovery_confirm_s
        ):
            proposed, state = "reference", "RECOVER"
            reasons.append("RECOVERY_HYSTERESIS")
        elif periodic_due:
            proposed, state = "reference", "FULL_COVERAGE"
            reasons.append("PERIODIC_COVERAGE_DEADLINE")
            self._recover_since = None
        else:
            self._recover_since = None
            if len(selected) > self.config.roi.max_selected_tiles:
                proposed, state = "reference", "FULL_COVERAGE"
                reasons.append("ROI_TILE_BUDGET_EXCEEDED")
            else:
                enter = self.config.fallback.area_ratio_enter
                exit_ratio = self.config.fallback.area_ratio_exit
                if self._cost_guard_active:
                    self._cost_guard_active = estimated_cost > exit_ratio
                elif estimated_cost >= enter:
                    self._cost_guard_active = True
                if (
                    self.config.fallback.enabled
                    and self.config.fallback.use_measured_cost_guard
                    and (
                        self._cost_guard_active
                        or union_ratio >= self.config.roi.max_roi_union_ratio
                    )
                ):
                    proposed, state = "reference", "FULL_COVERAGE"
                    reasons.append("REFERENCE_CHEAPER_THAN_ROIS")
                elif selected:
                    proposed, state = "tiles", "HYBRID"
                    if motion_tiles:
                        reasons.append("MOTION_WAKE")
                    if protected_tiles:
                        reasons.append("TRACK_PROTECTION")
                else:
                    proposed, state = "skip", "HYBRID"
                    reasons.append("NO_MOTION_OR_DUE_TRACK")

        scan_type = "reference" if self.config.shadow_mode else proposed
        if self.config.shadow_mode:
            reasons.append("SHADOW_REFERENCE_EXECUTION")
        # This is the scheduler's virtual reference clock.  In shadow mode the
        # executed scan is always reference, but that must not postpone the
        # reference scan the proposed hybrid schedule would have requested.
        # Otherwise shadow logs cannot audit periodic coverage deadlines.
        if proposed == "reference":
            self._last_reference_timestamp = timestamp_s
        if scan_type == "reference":
            searched_regions = ((0, 0, frame.shape[1], frame.shape[0]),)
        elif scan_type == "tiles":
            searched_regions = tuple(tile_regions[index] for index in selected)
        else:
            searched_regions = ()
        self._state = state
        self._last_timestamp = timestamp_s
        planning_ms = max(0.0, (time.perf_counter() - started) * 1000.0 - motion_ms)
        return MotionROIPlan(
            timestamp_s=timestamp_s,
            state=state,
            scan_type=scan_type,
            proposed_scan_type=proposed,
            reasons=tuple(reasons),
            selected_tiles=selected,
            searched_regions=searched_regions,
            motion_tiles=tuple(sorted(motion_tiles)),
            protected_tiles=tuple(sorted(protected_tiles)),
            foreground_ratio=foreground_ratio,
            roi_union_ratio=union_ratio,
            estimated_cost_ratio=estimated_cost,
            learning_rate=learning_rate,
            motion_ms=motion_ms,
            planning_ms=planning_ms,
            mask_shape=foreground.shape,
            shadow_mode=self.config.shadow_mode,
        )

    def _resize_for_analysis(self, frame: NDArray[np.uint8]) -> NDArray[np.uint8]:
        height, width = frame.shape[:2]
        target_width = min(width, self.config.background.analysis_width)
        target_height = max(1, round(height * target_width / max(1, width)))
        if (target_width, target_height) == (width, height):
            return frame
        return cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)

    def _foreground_mask(self, mask: NDArray[np.uint8]) -> NDArray[np.uint8]:
        foreground = np.where(mask == 255, 255, 0).astype(np.uint8)
        minimum = max(
            1,
            math.ceil(self.config.background.min_component_area_ratio * foreground.size),
        )
        if minimum <= 1 or not np.any(foreground):
            return foreground
        count, labels, stats, _ = cv2.connectedComponentsWithStats(foreground, connectivity=8)
        retained = np.zeros_like(foreground)
        for label in range(1, count):
            if int(stats[label, cv2.CC_STAT_AREA]) >= minimum:
                retained[labels == label] = 255
        return retained

    @staticmethod
    def _motion_tiles(
        mask: NDArray[np.uint8],
        frame_width: int,
        frame_height: int,
        tile_regions: Sequence[Rect],
    ) -> set[int]:
        mask_height, mask_width = mask.shape
        selected: set[int] = set()
        for index, (x1, y1, x2, y2) in enumerate(tile_regions):
            mx1 = max(0, min(mask_width, math.floor(x1 * mask_width / frame_width)))
            mx2 = max(mx1 + 1, min(mask_width, math.ceil(x2 * mask_width / frame_width)))
            my1 = max(0, min(mask_height, math.floor(y1 * mask_height / frame_height)))
            my2 = max(my1 + 1, min(mask_height, math.ceil(y2 * mask_height / frame_height)))
            if np.any(mask[my1:my2, mx1:mx2]):
                selected.add(index)
        return selected

    def _protected_tiles(
        self,
        boxes: Sequence[Rect],
        tile_regions: Sequence[Rect],
        width: int,
        height: int,
    ) -> set[int]:
        if not self.config.roi.protect_track_predictions:
            return set()
        padding = self.config.roi.bbox_padding_ratio
        selected: set[int] = set()
        for box in boxes:
            x1, y1, x2, y2 = map(float, box)
            pad_x = max(2.0, (x2 - x1) * padding)
            pad_y = max(2.0, (y2 - y1) * padding)
            expanded = (
                max(0.0, x1 - pad_x),
                max(0.0, y1 - pad_y),
                min(float(width), x2 + pad_x),
                min(float(height), y2 + pad_y),
            )
            for index, tile in enumerate(tile_regions):
                if self._intersection_area(expanded, tile) > 0:
                    selected.add(index)
        return selected

    @staticmethod
    def _intersection_area(first: Sequence[float], second: Sequence[float]) -> float:
        width = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
        height = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
        return width * height

    @classmethod
    def _union_ratio(
        cls,
        selected: Sequence[int],
        regions: Sequence[Rect],
        width: int,
        height: int,
    ) -> float:
        if not selected or width <= 0 or height <= 0:
            return 0.0
        xs = sorted({0, width, *(regions[index][0] for index in selected), *(regions[index][2] for index in selected)})
        ys = sorted({0, height, *(regions[index][1] for index in selected), *(regions[index][3] for index in selected)})
        area = 0.0
        for left, right in zip(xs, xs[1:]):
            for top, bottom in zip(ys, ys[1:]):
                center = ((left + right) / 2.0, (top + bottom) / 2.0)
                if any(
                    regions[index][0] <= center[0] <= regions[index][2]
                    and regions[index][1] <= center[1] <= regions[index][3]
                    for index in selected
                ):
                    area += (right - left) * (bottom - top)
        return min(1.0, area / float(width * height))
