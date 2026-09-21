from __future__ import annotations

import cv2
import numpy as np

from backend.app.analytics import SpatialTransformer
from backend.app.core.config import AnalyticsConfig
from backend.app.schemas import (
    CommonPath,
    CommonPathSnapshot,
    Detection,
    FramePacket,
    FrameResult,
    HeatmapSnapshot,
    TrackedObject,
    TrackPoint,
    Trajectory,
)
from backend.app.video import FrameRenderer, OverlayOptions


def frame_result() -> FrameResult:
    frame = np.zeros((200, 240, 3), dtype=np.uint8)
    packet = FramePacket(0, 0.0, 1.0, frame)
    return FrameResult(
        packet=packet,
        detections=(),
        tracks=(),
        inference_ms=12.0,
        tracking_ms=2.0,
        processing_completed_monotonic=1.1,
    )


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


def test_active_common_path_is_drawn_from_snapshot() -> None:
    analytics = AnalyticsConfig(grid_width=8, grid_height=6)
    result = frame_result()
    heatmap = HeatmapSnapshot(
        "current",
        "pixel",
        0.0,
        1.0,
        np.zeros((analytics.grid_height, analytics.grid_width), dtype=np.float32),
        np.zeros((analytics.grid_height, analytics.grid_width), dtype=np.float32),
    )
    snapshot = CommonPathSnapshot(
        timestamp=1.0,
        paths=(
            CommonPath(
                path_id="a__b__01",
                origin_zone="a",
                destination_zone="b",
                state="active",
                unique_tracks_short=5,
                unique_tracks_long=8,
                score=0.8,
                confidence=0.9,
                direction="A_TO_B",
                polyline=((20.0, 160.0), (120.0, 160.0), (220.0, 160.0)),
                updated_at=1.0,
            ),
        ),
    )

    rendered = FrameRenderer().render(
        result,
        (),
        heatmap,
        (),
        SpatialTransformer.pixel(240, 200),
        OverlayOptions(
            detection=False,
            tracking=False,
            trajectory=False,
            zones=False,
            active_paths=True,
            debug_metrics=False,
        ),
        processing_fps=10.0,
        common_paths=snapshot,
    )

    assert rendered[160, 120].any()


def trajectory(track_id: int = 7) -> Trajectory:
    return Trajectory(
        track_id=track_id,
        points=(
            TrackPoint(track_id, 0.0, 0, 20.0, 40.0, 20.0, 40.0, 0.9),
            TrackPoint(track_id, 1.0, 1, 120.0, 40.0, 120.0, 40.0, 0.9),
        ),
        age=2,
        confirmed=True,
        last_seen_timestamp=1.0,
    )


def common_path(
    *,
    state: str = "active",
    path_id: str = "a__b__01",
    y: float = 120.0,
) -> CommonPath:
    return CommonPath(
        path_id=path_id,
        origin_zone="a",
        destination_zone="b",
        state=state,
        unique_tracks_short=8,
        unique_tracks_long=12,
        score=0.8,
        confidence=0.9,
        direction="A_TO_B",
        polyline=((20.0, y), (120.0, y), (220.0, y)),
        updated_at=1.0,
    )


def empty_heatmap() -> HeatmapSnapshot:
    return HeatmapSnapshot(
        "current",
        "pixel",
        0.0,
        1.0,
        np.zeros((6, 8), dtype=np.float32),
        np.zeros((6, 8), dtype=np.float32),
    )


def test_production_renderer_does_not_draw_individual_trajectory() -> None:
    target = trajectory()
    rendered = FrameRenderer().render(
        frame_result(),
        (target.points[-1],),
        empty_heatmap(),
        (),
        SpatialTransformer.pixel(240, 200),
        OverlayOptions(
            tracking=True,
            points=True,
            trajectory=True,
            trajectory_tails=False,
            active_paths=False,
            zones=False,
            debug_metrics=False,
        ),
        processing_fps=10.0,
        debug_trajectories=(target,),
    )

    assert not rendered[40, 70].any()
    assert not rendered[40, 120].any()  # point belongs to frame 1, not packet frame 0


def test_candidate_path_is_hidden_in_production_mode() -> None:
    snapshot = CommonPathSnapshot(timestamp=1.0, paths=(common_path(state="candidate"),))
    rendered = FrameRenderer().render(
        frame_result(),
        (),
        empty_heatmap(),
        (),
        SpatialTransformer.pixel(240, 200),
        OverlayOptions(
            tracking=False,
            active_paths=True,
            candidate_paths=False,
            zones=False,
            debug_metrics=False,
        ),
        processing_fps=10.0,
        common_paths=snapshot,
    )

    assert not rendered[120, 120].any()


def test_active_path_transition_interpolates_without_one_frame_jump() -> None:
    renderer = FrameRenderer()
    old = CommonPathSnapshot(timestamp=0.0, paths=(common_path(y=60.0),))
    new = CommonPathSnapshot(
        timestamp=1.0,
        paths=(common_path(path_id="a__b__02", y=140.0),),
    )

    renderer.display_common_paths(old, 0.0)
    transition_start = renderer.display_common_paths(new, 1.0)[0]
    transition_middle = renderer.display_common_paths(new, 1.25)[0]
    transition_end = renderer.display_common_paths(new, 1.5)[0]

    assert transition_start.polyline[16][1] == 60.0
    assert 60.0 < transition_middle.polyline[16][1] < 140.0
    assert transition_end.polyline[16][1] == 140.0


def test_cooling_path_remains_visible_after_short_detection_gap() -> None:
    renderer = FrameRenderer()
    active = CommonPathSnapshot(timestamp=1.0, paths=(common_path(),))
    cooling = CommonPathSnapshot(timestamp=2.0, paths=(common_path(state="cooling"),))

    renderer.display_common_paths(active, 1.0)
    displayed = renderer.display_common_paths(cooling, 2.0)

    assert len(displayed) == 1
    assert displayed[0].state == "cooling"


def test_point_only_video_output_preserves_fps_and_resolution(tmp_path) -> None:
    output = tmp_path / "point-only.mp4"
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10.0,
        (240, 200),
    )
    assert writer.isOpened()
    renderer = FrameRenderer()
    options = OverlayOptions(zones=False, debug_metrics=False, active_paths=False)
    for _ in range(3):
        frame = renderer.render(
            frame_result(),
            (),
            empty_heatmap(),
            (),
            SpatialTransformer.pixel(240, 200),
            options,
            processing_fps=10.0,
        )
        writer.write(frame)
    writer.release()

    capture = cv2.VideoCapture(str(output))
    assert capture.isOpened()
    assert capture.get(cv2.CAP_PROP_FRAME_WIDTH) == 240
    assert capture.get(cv2.CAP_PROP_FRAME_HEIGHT) == 200
    assert capture.get(cv2.CAP_PROP_FPS) == 10.0
    assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 3
    capture.release()
