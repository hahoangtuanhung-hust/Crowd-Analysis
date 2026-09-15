from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from statistics import fmean

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.core.config import load_config
from backend.app.datasets import GrandCentralDataset, GrandCentralPaths
from backend.app.datasets.evaluation import evaluate_predictions
from backend.app.datasets.grand_central import NormalizedTrajectoryPoint
from backend.app.datasets.storage import TrajectoryBatchWriter
from backend.app.inference import UltralyticsPersonDetector
from scripts.analyze_grand_central import load_points


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Grand Central predictions at annotated timestamps"
    )
    parser.add_argument("--root", type=Path, help="Override GC_DATASET_ROOT")
    parser.add_argument("--duration", choices=("1m", "5m", "all"), default="1m")
    parser.add_argument("--inference-interval", type=int, default=1)
    parser.add_argument("--threshold-pixels", type=float, default=60.0)
    parser.add_argument("--prediction", type=Path, help="Override prediction Parquet")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument(
        "--no-sampled-detection",
        action="store_true",
        help="Use tracked points instead of rerunning YOLO at annotation frames",
    )
    return parser.parse_args()


def sample_detections(
    dataset: GrandCentralDataset,
    ground_truth: list[NormalizedTrajectoryPoint],
    *,
    duration_label: str,
    inference_interval: int,
    config_path: Path,
) -> tuple[list[NormalizedTrajectoryPoint], dict[str, object]]:
    video_fps = float(dataset.video_metadata()["fps"])
    requested_frames = {round(point.timestamp * video_fps) for point in ground_truth}
    config = load_config(config_path)
    detector = UltralyticsPersonDetector(config.detector)
    matrix = dataset.load_homography(required=False)
    zones = json.loads(
        (dataset.paths.processed / "zones.json").read_text(encoding="utf-8")
    )["zones"]
    output_path = (
        dataset.paths.output
        / "tracking"
        / f"annotation_frame_detections_interval_{inference_interval}_{duration_label}.parquet"
    )
    points: list[NormalizedTrajectoryPoint] = []
    inference_times = []
    capture = cv2.VideoCapture(str(dataset.paths.video))
    if not capture.isOpened():
        raise ValueError(f"Unable to open video: {dataset.paths.video}")
    try:
        frame_id = 0
        last_frame = max(requested_frames)
        while frame_id <= last_frame:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_id in requested_frames:
                started = time.perf_counter()
                frame_detections = detector.detect(frame)
                inference_times.append((time.perf_counter() - started) * 1000.0)
                timestamp = frame_id / video_fps
                for index, detection in enumerate(frame_detections):
                    foot_x = (detection.x1 + detection.x2) / 2.0
                    foot_y = detection.y2
                    world_x = world_y = None
                    if matrix is not None:
                        world_x, world_y = dataset.transform_point(foot_x, foot_y, matrix)
                    points.append(
                        NormalizedTrajectoryPoint(
                            camera_id=dataset.camera_id,
                            source="prediction",
                            frame_id=frame_id,
                            timestamp=timestamp,
                            track_id=frame_id * 10_000 + index,
                            x=foot_x,
                            y=foot_y,
                            x1=detection.x1,
                            y1=detection.y1,
                            x2=detection.x2,
                            y2=detection.y2,
                            foot_x=foot_x,
                            foot_y=foot_y,
                            world_x=world_x,
                            world_y=world_y,
                            confidence=detection.confidence,
                            zone_id=dataset.zone_at(foot_x, foot_y, zones),
                        )
                    )
            frame_id += 1
    finally:
        capture.release()
    with TrajectoryBatchWriter(output_path, batch_size=1_000) as writer:
        writer.write(points)
    return points, {
        "source": "prediction",
        "sampled_frames": len(requested_frames),
        "detections": len(points),
        "mean_inference_ms": round(fmean(inference_times), 3),
        "artifact": str(output_path.resolve()),
    }


def main() -> int:
    args = parse_args()
    paths = GrandCentralPaths.from_env(args.root)
    dataset = GrandCentralDataset(paths)
    video_duration = float(dataset.video_metadata()["duration_seconds"])
    requested = {"1m": 60.0, "5m": 300.0, "all": video_duration}[args.duration]
    duration_seconds = min(requested, video_duration)
    prediction_path = args.prediction or (
        paths.output
        / "tracking"
        / f"prediction_interval_{args.inference_interval}_{args.duration}.parquet"
    )
    ground_truth = load_points(
        paths.processed / "trajectories.parquet",
        source="ground_truth",
        duration_seconds=duration_seconds,
    )
    prediction = load_points(
        prediction_path,
        source="prediction",
        duration_seconds=duration_seconds,
    )
    sampled_metrics = None
    detections = None
    if not args.no_sampled_detection:
        detections, sampled_metrics = sample_detections(
            dataset,
            ground_truth,
            duration_label=args.duration,
            inference_interval=args.inference_interval,
            config_path=args.config,
        )
    result = evaluate_predictions(
        dataset,
        ground_truth,
        prediction,
        detections=detections,
        threshold_pixels=args.threshold_pixels,
        duration_label=args.duration,
        inference_interval=args.inference_interval,
        persist=True,
    )
    result["sampled_detection_run"] = sampled_metrics
    Path(result["artifacts"]["evaluation_json"]).write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "dataset": result["dataset"],
        "duration": result["duration"],
        "ground_truth": result["ground_truth"],
        "prediction": result["prediction"],
        "timebase": result["timebase"],
        "matching": result["matching"],
        "metrics": result["metrics"],
        "sampled_detection_run": sampled_metrics,
        "limitations": result["limitations"],
        "artifacts": result["artifacts"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
