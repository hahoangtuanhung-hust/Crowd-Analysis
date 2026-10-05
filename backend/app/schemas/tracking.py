from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import NDArray

FloatImage = NDArray[np.uint8]
ObservationCoverage = Literal[
    "MEASURED",
    "SEARCHED_NOT_FOUND",
    "NOT_SEARCHED_BY_POLICY",
]
DirectionState = Literal["unknown", "stationary", "moving"]


@dataclass(frozen=True, slots=True)
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int = 0

    @property
    def xyxy(self) -> tuple[float, float, float, float]:
        return self.x1, self.y1, self.x2, self.y2


@dataclass(frozen=True, slots=True)
class TrackedObject:
    track_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int = 0
    observed: bool = True
    observation_coverage: ObservationCoverage = "MEASURED"
    direction_vector: tuple[float, float] | None = None
    direction_state: DirectionState = "unknown"
    direction_quality: float = 0.0
    direction_observed_span_s: float = 0.0

    @property
    def xyxy(self) -> tuple[float, float, float, float]:
        return self.x1, self.y1, self.x2, self.y2

    @property
    def bottom_center(self) -> tuple[float, float]:
        return (self.x1 + self.x2) / 2.0, self.y2


@dataclass(frozen=True, slots=True)
class FramePacket:
    frame_id: int
    source_timestamp: float
    captured_monotonic: float
    image: FloatImage = field(repr=False, compare=False)
    decode_ms: float = 0.0


@dataclass(frozen=True, slots=True)
class FrameResult:
    packet: FramePacket = field(repr=False, compare=False)
    detections: tuple[Detection, ...]
    tracks: tuple[TrackedObject, ...]
    inference_ms: float
    tracking_ms: float
    processing_completed_monotonic: float
    observation_mode: str = "FULL_COVERAGE"
    searched_regions: tuple[tuple[int, int, int, int], ...] = ()
    scheduler_state: str = "DISABLED"
    scheduler_reasons: tuple[str, ...] = ()
    motion_ms: float = 0.0
    roi_planning_ms: float = 0.0
    roi_union_ratio: float = 1.0
    detector_images: int = 1
    intentional_skip: bool = False

    @property
    def e2e_latency_ms(self) -> float:
        return max(
            0.0,
            (self.processing_completed_monotonic - self.packet.captured_monotonic) * 1000.0,
        )


@dataclass(frozen=True, slots=True)
class TrackPoint:
    track_id: int
    timestamp: float
    frame_id: int
    x: float
    y: float
    raw_x: float
    raw_y: float
    confidence: float


@dataclass(frozen=True, slots=True)
class Trajectory:
    track_id: int
    points: tuple[TrackPoint, ...]
    age: int
    confirmed: bool
    last_seen_timestamp: float


@dataclass(frozen=True, slots=True)
class HeatmapSnapshot:
    window: str
    spatial_mode: str
    from_timestamp: float
    to_timestamp: float
    occupancy: NDArray[np.float32] = field(repr=False, compare=False)
    movement: NDArray[np.float32] = field(repr=False, compare=False)
