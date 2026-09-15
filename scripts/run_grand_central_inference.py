from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
import threading
import time
from pathlib import Path
from statistics import fmean

import cv2
import psutil

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.core.config import load_config
from backend.app.datasets import (
    DatasetAnalyticsRunner,
    GrandCentralDataset,
    GrandCentralPaths,
    NormalizedTrajectoryPoint,
)
from backend.app.datasets.storage import TrajectoryBatchWriter
from backend.app.inference import UltralyticsPersonDetector
from backend.app.schemas import FrameResult
from backend.app.tracking import ByteTrackTracker
from backend.app.video import OpenCVVideoSource, TrackingPipeline
from scripts.analyze_grand_central import load_points


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run YOLO and ByteTrack on the Grand Central video"
    )
    parser.add_argument("--root", type=Path, help="Override GC_DATASET_ROOT")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--duration", choices=("1m", "5m", "all"), default="1m")
    parser.add_argument("--inference-interval", type=int, choices=range(1, 31), default=1)
    parser.add_argument(
        "--coordinate-mode", choices=("pixel_space", "ground_plane"), default="pixel_space"
    )
    parser.add_argument("--csv", action="store_true")
    return parser.parse_args()


def draw_preview(result: FrameResult) -> object:
    frame = result.packet.image.copy()
    for track in result.tracks:
        x1, y1, x2, y2 = (round(value) for value in track.xyxy)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (58, 211, 163), 2)
        cv2.putText(
            frame,
            f"ID {track.track_id}",
            (x1, max(18, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return frame


def main() -> int:
    args = parse_args()
    paths = GrandCentralPaths.from_env(args.root)
    dataset = GrandCentralDataset(paths)
    config = load_config(args.config)
    video_metadata = dataset.video_metadata()
    video_duration = float(video_metadata["duration_seconds"])
    requested_seconds = {"1m": 60.0, "5m": 300.0, "all": video_duration}[args.duration]
    duration_seconds = min(requested_seconds, video_duration)
    max_frames = min(
        int(video_metadata["frame_count"]),
        max(1, round(duration_seconds * float(video_metadata["fps"]))),
    )
    video_config = config.video.model_copy(update={"inference_interval": args.inference_interval})
    config = config.model_copy(update={"video": video_config})

    stem = f"prediction_interval_{args.inference_interval}_{args.duration}"
    parquet_path = paths.output / "tracking" / f"{stem}.parquet"
    csv_path = paths.output / "tracking" / f"{stem}.csv" if args.csv else None
    preview_path = paths.output / "tracking" / f"{stem}_preview.jpg"
    canonical_path = paths.output / "tracking" / "prediction_trajectories.parquet"
    matrix = dataset.load_homography(required=False)
    zones = json.loads((paths.processed / "zones.json").read_text(encoding="utf-8"))["zones"]
    writer = TrajectoryBatchWriter(parquet_path, csv_path=csv_path, batch_size=5_000)
    inference_times: list[float] = []
    tracking_times: list[float] = []
    e2e_times: list[float] = []
    preview_frame = None
    write_lock = threading.Lock()

    def handle_result(result: FrameResult) -> None:
        nonlocal preview_frame
        points = []
        for track in result.tracks:
            foot_x, foot_y = track.bottom_center
            world_x = world_y = None
            if matrix is not None:
                world_x, world_y = dataset.transform_point(foot_x, foot_y, matrix)
            points.append(
                NormalizedTrajectoryPoint(
                    camera_id=dataset.camera_id,
                    source="prediction",
                    frame_id=result.packet.frame_id,
                    timestamp=result.packet.source_timestamp,
                    track_id=track.track_id,
                    x=foot_x,
                    y=foot_y,
                    x1=track.x1,
                    y1=track.y1,
                    x2=track.x2,
                    y2=track.y2,
                    foot_x=foot_x,
                    foot_y=foot_y,
                    world_x=world_x,
                    world_y=world_y,
                    confidence=track.confidence,
                    zone_id=dataset.zone_at(foot_x, foot_y, zones),
                )
            )
        with write_lock:
            writer.write(points)
            inference_times.append(result.inference_ms)
            tracking_times.append(result.tracking_ms)
            e2e_times.append(result.e2e_latency_ms)
            if result.tracks and (preview_frame is None or result.packet.frame_id % 300 == 0):
                preview_frame = draw_preview(result)

    detector = UltralyticsPersonDetector(config.detector)
    tracker = ByteTrackTracker(config.tracker)
    source = OpenCVVideoSource(paths.video, max_frames=max_frames)
    pipeline = TrackingPipeline(
        source,
        detector,
        tracker,
        queue_size=config.video.queue_size,
        drop_oldest=False,
        inference_interval=args.inference_interval,
        on_result=handle_result,
    )
    process = psutil.Process()
    process.cpu_percent(None)
    started = time.perf_counter()
    try:
        pipeline.start()
        pipeline.join()
    finally:
        pipeline.stop()
        pipeline.join(timeout=10.0)
        with write_lock:
            writer.close()
    elapsed = time.perf_counter() - started
    stats = pipeline.stats
    if stats.error:
        raise RuntimeError(stats.error)
    if writer.rows == 0:
        raise RuntimeError("Inference completed but produced no tracked trajectory points")
    if args.inference_interval == 1:
        shutil.copy2(parquet_path, canonical_path)
    if preview_frame is not None:
        cv2.imwrite(str(preview_path), preview_frame)

    points = load_points(parquet_path, source="prediction", duration_seconds=duration_seconds)
    runner = DatasetAnalyticsRunner(
        dataset, config.analytics, coordinate_mode=args.coordinate_mode
    )
    analytics = runner.run(
        points,
        source="prediction",
        duration_label=args.duration,
        persist=args.inference_interval == 1,
    )
    metrics = {
        "mode": "model_inference",
        "source": "prediction",
        "duration": args.duration,
        "duration_seconds": duration_seconds,
        "detector": config.detector.model,
        "tracker": config.tracker.type,
        "imgsz": config.detector.imgsz,
        "confidence": config.detector.confidence,
        "inference_interval": args.inference_interval,
        "captured_frames": stats.captured_frames,
        "processed_frames": stats.processed_frames,
        "trajectory_rows": writer.rows,
        "processing_fps": round(stats.processed_frames / elapsed, 3),
        "inference_ms": round(fmean(inference_times), 3),
        "tracking_ms": round(fmean(tracking_times), 3),
        "e2e_ms": round(fmean(e2e_times), 3),
        "elapsed_seconds": round(elapsed, 3),
        "cpu_percent": process.cpu_percent(None),
        "ram_mb": round(process.memory_info().rss / 1024**2, 3),
        "gpu_memory_mb": None,
        "dropped_frames": stats.dropped_frames,
        "hardware": {
            "processor": platform.processor() or platform.machine(),
            "physical_cores": psutil.cpu_count(logical=False),
            "logical_cores": psutil.cpu_count(),
            "gpu": None,
        },
        "artifacts": {
            "trajectories": str(parquet_path.resolve()),
            "canonical_trajectories": str(canonical_path.resolve())
            if args.inference_interval == 1
            else None,
            "preview": str(preview_path.resolve()),
            "analytics": analytics.get("artifacts", {}).get("result_json"),
        },
    }
    metrics_path = paths.output / "benchmarks" / f"{stem}.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
