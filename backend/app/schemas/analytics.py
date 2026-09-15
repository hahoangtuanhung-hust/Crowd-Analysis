from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class PopularPath:
    path_id: str
    label: str
    count: int
    percentage: float
    regions: tuple[str, ...]
    kind: str


@dataclass(frozen=True, slots=True)
class FlowSnapshot:
    window: str
    spatial_mode: str
    from_timestamp: float
    to_timestamp: float
    dominant_direction: str
    vx: NDArray[np.float32] = field(repr=False, compare=False)
    vy: NDArray[np.float32] = field(repr=False, compare=False)
    samples: NDArray[np.uint32] = field(repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class ZoneMetric:
    zone_id: str
    name: str
    current_people: int
    unique_tracks: int
    entry_count: int
    exit_count: int
    average_dwell_seconds: float
    peak_occupancy: int


@dataclass(frozen=True, slots=True)
class ZoneFlow:
    from_zone: str
    to_zone: str
    count: int


@dataclass(frozen=True, slots=True)
class ZoneSnapshot:
    timestamp: float
    zones: tuple[ZoneMetric, ...]
    flows: tuple[ZoneFlow, ...]


@dataclass(frozen=True, slots=True)
class CrowdSummary:
    current_crowd_count: int
    average_crowd_count: float
    peak_crowd_count: int
    unique_track_count: int
    processed_frames: int
    spatial_mode: str
    calibration_required: bool


@dataclass(frozen=True, slots=True)
class TimelinePoint:
    timestamp: float
    average_count: float
    peak_count: int
