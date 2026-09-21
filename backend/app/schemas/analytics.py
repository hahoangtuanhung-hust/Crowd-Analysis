from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

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


Cell = tuple[int, int]


@dataclass(frozen=True, slots=True)
class DirectedFlowEdge:
    from_cell: Cell
    to_cell: Cell
    unique_tracks_short: int
    unique_tracks_long: int
    score: float


@dataclass(frozen=True, slots=True)
class DirectedFlowSnapshot:
    from_timestamp: float
    to_timestamp: float
    grid_columns: int
    grid_rows: int
    edges: tuple[DirectedFlowEdge, ...]


@dataclass(frozen=True, slots=True)
class CommonPath:
    path_id: str
    origin_zone: str
    destination_zone: str
    state: Literal["candidate", "active", "cooling", "retired"]
    unique_tracks_short: int
    unique_tracks_long: int
    score: float
    confidence: float
    direction: str
    polyline: tuple[tuple[float, float], ...]
    updated_at: float
    revision: int = 0
    coordinate_space: str = "image_pixels"
    support_tracks: int = 0
    validated_complete_tracks: int = 0
    evidence_until_s: float = 0.0
    stale: bool = False


@dataclass(frozen=True, slots=True)
class CommonPathSnapshot:
    timestamp: float
    paths: tuple[CommonPath, ...]
    version: int = 0
    evidence_until_s: float = 0.0


@dataclass(frozen=True, slots=True)
class CommonPathTimelinePoint:
    timestamp: float
    path_id: str
    state: str
    score: float
    confidence: float
    unique_tracks_short: int
    unique_tracks_long: int


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
