from __future__ import annotations

import numpy as np

from backend.app.analytics import SpatialTransformer
from backend.app.core.config import AnalyticsConfig
from backend.app.schemas import (
    Detection,
    FramePacket,
    FrameResult,
    HeatmapSnapshot,
    TrackedObject,
)
from backend.app.video import FrameRenderer, OverlayOptions


def test_heatmap_overlay_changes_only_sampled_area() -> None:
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    values = np.zeros((12, 16), dtype=np.float32)
    values[5:7, 7:9] = 1.0

    rendered = FrameRenderer._draw_heatmap(
        frame,
        values,
        SpatialTransformer.pixel(width=160, height=120),
    )

    assert rendered[60, 80].any()
    assert not rendered[0, 0].any()


def test_empty_heatmap_keeps_frame_unchanged() -> None:
    frame = np.full((24, 32, 3), 17, dtype=np.uint8)

    rendered = FrameRenderer._draw_heatmap(
        frame.copy(),
        np.zeros((6, 8), dtype=np.float32),
        SpatialTransformer.pixel(width=32, height=24),
    )

    assert np.array_equal(rendered, frame)


def test_tracking_overlay_draws_bottom_center_point_without_bbox() -> None:
    frame = np.zeros((200, 240, 3), dtype=np.uint8)
    packet = FramePacket(0, 0.0, 1.0, frame)
    result = FrameResult(
        packet=packet,
        detections=(Detection(100, 80, 120, 140, 0.9),),
        tracks=(TrackedObject(7, 100, 80, 120, 140, 0.9),),
        inference_ms=12.0,
        tracking_ms=2.0,
        processing_completed_monotonic=1.1,
    )
    analytics = AnalyticsConfig()
    heatmap = HeatmapSnapshot(
        "current",
        "pixel",
        0.0,
        0.0,
        np.zeros((analytics.grid_height, analytics.grid_width), dtype=np.float32),
        np.zeros((analytics.grid_height, analytics.grid_width), dtype=np.float32),
    )

    rendered = FrameRenderer().render(
        result,
        (),
        heatmap,
        (),
        SpatialTransformer.pixel(240, 200),
        OverlayOptions(detection=False, tracking=True, trajectory=False, zones=False),
        processing_fps=10.0,
    )

    assert rendered[140, 110].any()
    assert not rendered[80, 100].any()
