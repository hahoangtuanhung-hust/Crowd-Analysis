from __future__ import annotations

import argparse
import json
import logging
import threading
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from backend.app.analytics import (
    FlowAnalyzer,
    HeatmapAnalyzer,
    SpatialTransformer,
    TrajectoryManager,
    ZoneAnalyzer,
)
from backend.app.core.config import load_config
from backend.app.inference import UltralyticsPersonDetector
from backend.app.schemas import FrameResult
from backend.app.tracking import ByteTrackTracker
from backend.app.video import OpenCVVideoSource, TrackingPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run person detection and ByteTrack on a video")
    parser.add_argument("source", help="MP4 path or RTSP URL")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output", default="data/outputs/milestone-1-overlay.mp4")
    parser.add_argument("--jsonl", default="data/outputs/milestone-1-tracks.jsonl")
    parser.add_argument("--occupancy-heatmap", default="data/outputs/occupancy-heatmap.png")
    parser.add_argument("--movement-heatmap", default="data/outputs/movement-heatmap.png")
    parser.add_argument("--flow-field", default="data/outputs/flow-field.png")
    parser.add_argument("--analytics-json", default="data/outputs/analytics-summary.json")
    parser.add_argument(
        "--realtime", action="store_true", help="Drop stale frames instead of blocking"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    config = load_config(args.config)
    source = OpenCVVideoSource(args.source)
    metadata = source.open()
    source.close()

    output_path = Path(args.output)
    jsonl_path = Path(args.jsonl)
    occupancy_path = Path(args.occupancy_heatmap)
    movement_path = Path(args.movement_heatmap)
    flow_path = Path(args.flow_field)
    analytics_path = Path(args.analytics_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    occupancy_path.parent.mkdir(parents=True, exist_ok=True)
    movement_path.parent.mkdir(parents=True, exist_ok=True)
    flow_path.parent.mkdir(parents=True, exist_ok=True)
    analytics_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        max(1.0, metadata.fps / config.video.inference_interval),
        (metadata.width, metadata.height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Unable to create output video: {output_path}")

    io_lock = threading.Lock()
    jsonl_stream = jsonl_path.open("w", encoding="utf-8")
    trajectories = TrajectoryManager(config.analytics)
    spatial = SpatialTransformer.pixel(metadata.width, metadata.height)
    heatmaps = HeatmapAnalyzer(
        config.analytics,
        spatial,
        retain_entire=True,
    )
    flows = FlowAnalyzer(config.analytics, spatial, retain_entire=True)
    zones = ZoneAnalyzer(config.analytics, spatial, retain_entire=True)

    def handle_result(result: FrameResult) -> None:
        frame = result.packet.image.copy()
        trajectory_snapshots = trajectories.update(
            result.tracks,
            frame_id=result.packet.frame_id,
            timestamp=result.packet.source_timestamp,
        )
        heatmaps.process(trajectory_snapshots)
        flows.process(trajectory_snapshots)
        zones.process(trajectory_snapshots)
        trajectory_by_id = {item.track_id: item for item in trajectory_snapshots}
        for track in result.tracks:
            trajectory = trajectory_by_id.get(track.track_id)
            if trajectory is not None and trajectory.points:
                point = (
                    round(trajectory.points[-1].x),
                    round(trajectory.points[-1].y),
                )
            else:
                x, y = track.bottom_center
                point = (round(x), round(y))
            cv2.circle(frame, point, 5, (66, 196, 143), -1, cv2.LINE_AA)
            cv2.circle(frame, point, 7, (245, 247, 250), 1, cv2.LINE_AA)
            cv2.putText(
                frame,
                f"ID {track.track_id}",
                (point[0] + 8, max(18, point[1] - 7)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            if trajectory is not None and len(trajectory.points) >= 2:
                points = np.asarray(
                    [(round(point.x), round(point.y)) for point in trajectory.points],
                    dtype=np.int32,
                ).reshape((-1, 1, 2))
                cv2.polylines(frame, [points], False, (52, 211, 255), 2, cv2.LINE_AA)
        payload = {
            "frame_id": result.packet.frame_id,
            "timestamp": round(result.packet.source_timestamp, 6),
            "people": len(result.tracks),
            "inference_ms": round(result.inference_ms, 3),
            "tracking_ms": round(result.tracking_ms, 3),
            "tracks": [
                {
                    "track_id": track.track_id,
                    "bbox": [round(value, 2) for value in track.xyxy],
                    "bottom_center": [
                        round(trajectory_by_id[track.track_id].points[-1].x, 2),
                        round(trajectory_by_id[track.track_id].points[-1].y, 2),
                    ],
                    "confidence": round(track.confidence, 4),
                    "age": trajectory_by_id[track.track_id].age,
                    "confirmed": trajectory_by_id[track.track_id].confirmed,
                    "trajectory": [
                        {
                            "frame_id": point.frame_id,
                            "timestamp": round(point.timestamp, 6),
                            "x": round(point.x, 2),
                            "y": round(point.y, 2),
                        }
                        for point in trajectory_by_id[track.track_id].points
                    ],
                }
                for track in result.tracks
            ],
        }
        with io_lock:
            writer.write(frame)
            jsonl_stream.write(json.dumps(payload, separators=(",", ":")) + "\n")

    detector = UltralyticsPersonDetector(config.detector)
    tracker = ByteTrackTracker(config.tracker)
    pipeline = TrackingPipeline(
        source=source,
        detector=detector,
        tracker=tracker,
        queue_size=config.video.queue_size,
        drop_oldest=args.realtime,
        inference_interval=config.video.inference_interval,
        on_result=handle_result,
    )
    try:
        pipeline.start()
        pipeline.join()
    finally:
        pipeline.stop()
        pipeline.join(timeout=5.0)
        with io_lock:
            writer.release()
            jsonl_stream.close()

    heatmap = heatmaps.snapshot("entire")
    cv2.imwrite(str(occupancy_path), heatmaps.render(heatmap.occupancy))
    cv2.imwrite(str(movement_path), heatmaps.render(heatmap.movement))
    flows.finalize_all()
    flow = flows.snapshot("entire")
    cv2.imwrite(str(flow_path), flows.render(flow))
    zone_snapshot = zones.snapshot()
    top_paths = zones.top_paths() or flows.top_paths()
    analytics_path.write_text(
        json.dumps(
            {
                "spatial_mode": spatial.mode,
                "dominant_direction": flow.dominant_direction,
                "top_paths": [asdict(item) for item in top_paths],
                "zones": [asdict(item) for item in zone_snapshot.zones],
                "zone_flows": [asdict(item) for item in zone_snapshot.flows],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    stats = pipeline.stats
    print(json.dumps(asdict(stats), indent=2))
    return 1 if stats.error else 0


if __name__ == "__main__":
    raise SystemExit(main())
