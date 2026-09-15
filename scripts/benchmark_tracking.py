from __future__ import annotations

import argparse
import csv
import math
import statistics
import time
from collections import defaultdict
from pathlib import Path

import cv2

from backend.app.core.config import TrackerConfig, load_config
from backend.app.datasets.grand_central import GrandCentralDataset
from backend.app.inference import UltralyticsPersonDetector
from backend.app.schemas import Detection
from backend.app.tracking import ByteTrackTracker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark YOLO person detection and ByteTrack")
    parser.add_argument("source", nargs="?", default="data/videos/data.mp4")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output", default="outputs/benchmark.csv")
    parser.add_argument("--frame-log", default="outputs/benchmark_frames.csv")
    parser.add_argument("--max-frames", type=int, default=150)
    parser.add_argument("--warmup-frames", type=int, default=10)
    parser.add_argument("--min-fps", type=float, default=3.0)
    return parser.parse_args()


def _greedy_matches(
    ground_truth: list[tuple[float, float]],
    predictions: list[tuple[float, float]],
    threshold: float = 60.0,
) -> int:
    candidates = []
    for gt_index, (gt_x, gt_y) in enumerate(ground_truth):
        for pred_index, (pred_x, pred_y) in enumerate(predictions):
            distance = math.hypot(gt_x - pred_x, gt_y - pred_y)
            if distance <= threshold:
                candidates.append((distance, gt_index, pred_index))
    used_gt: set[int] = set()
    used_predictions: set[int] = set()
    count = 0
    for _, gt_index, pred_index in sorted(candidates):
        if gt_index in used_gt or pred_index in used_predictions:
            continue
        used_gt.add(gt_index)
        used_predictions.add(pred_index)
        count += 1
    return count


def _ground_truth_by_frame(
    source: Path, max_frames: int
) -> dict[int, list[tuple[float, float]]]:
    capture = cv2.VideoCapture(str(source))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    capture.release()
    dataset = GrandCentralDataset()
    grouped: dict[int, list[tuple[float, float]]] = defaultdict(list)
    try:
        points = dataset.load_trajectories(duration_seconds=max_frames / fps)
        for point in points:
            frame_id = round(point.timestamp * fps)
            if frame_id < max_frames:
                grouped[frame_id].append((point.foot_x, point.foot_y))
    except FileNotFoundError:
        return {}
    return grouped


def _detections(detector: UltralyticsPersonDetector, frame) -> list[Detection]:
    return detector.detect(frame)


def main() -> int:
    args = parse_args()
    if args.max_frames < 1:
        raise ValueError("--max-frames must be positive")
    source = Path(args.source)
    if not source.is_file():
        raise FileNotFoundError(f"Video does not exist: {source}")

    config = load_config(args.config)
    ground_truth = _ground_truth_by_frame(source, args.max_frames)
    frame_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for imgsz in (640, 960, 1280):
        for confidence in (0.15, 0.20, 0.25):
            detector = UltralyticsPersonDetector(
                config.detector.model_copy(
                    update={
                        "imgsz": imgsz,
                        "confidence": confidence,
                        "iou": 0.60,
                        "max_det": 1000,
                        "classes": [0],
                    }
                )
            )
            tracker = ByteTrackTracker(
                TrackerConfig(
                    track_high_thresh=0.25,
                    track_low_thresh=0.05,
                    new_track_thresh=0.20,
                    track_buffer=60,
                    match_thresh=0.80,
                    fuse_score=True,
                )
            )
            capture = cv2.VideoCapture(str(source))
            previous_ids: set[int] = set()
            seen_ids: set[int] = set()
            own_rows: list[dict[str, object]] = []
            total_matches = total_predictions = total_ground_truth = 0
            for frame_id in range(args.max_frames):
                ok, frame = capture.read()
                if not ok:
                    break
                started = time.perf_counter()
                inference_started = time.perf_counter()
                detections = _detections(detector, frame)
                inference_ms = (time.perf_counter() - inference_started) * 1000.0
                tracks = tracker.update(detections, frame)
                processing_ms = (time.perf_counter() - started) * 1000.0
                active_ids = {track.track_id for track in tracks}
                new_ids = active_ids - seen_ids
                lost_ids = previous_ids - active_ids
                seen_ids.update(active_ids)
                row = {
                    "imgsz": imgsz,
                    "confidence": confidence,
                    "frame_id": frame_id,
                    "raw_person_detections": len(detections),
                    "active_tracks": len(active_ids),
                    "new_tracks": len(new_ids),
                    "lost_tracks": len(lost_ids),
                    "processing_fps": 1000.0 / max(processing_ms, 1e-9),
                    "inference_ms": inference_ms,
                    "processing_ms": processing_ms,
                }
                own_rows.append(row)
                frame_rows.append(row)
                if frame_id in ground_truth:
                    prediction_points = [
                        ((item.x1 + item.x2) / 2.0, item.y2) for item in detections
                    ]
                    truth = ground_truth[frame_id]
                    total_matches += _greedy_matches(truth, prediction_points)
                    total_predictions += len(prediction_points)
                    total_ground_truth += len(truth)
                previous_ids = active_ids
            capture.release()
            measured = own_rows[min(args.warmup_frames, len(own_rows)) :]
            if not measured:
                raise RuntimeError("Video ended before benchmark measurements began")
            processing_seconds = sum(float(row["processing_ms"]) for row in measured) / 1000.0
            summary_rows.append(
                {
                    "input": str(source),
                    "frames": len(own_rows),
                    "warmup_excluded": min(args.warmup_frames, len(own_rows)),
                    "imgsz": imgsz,
                    "confidence": confidence,
                    "iou": 0.60,
                    "max_det": 1000,
                    "track_high_thresh": 0.25,
                    "track_low_thresh": 0.05,
                    "new_track_thresh": 0.20,
                    "track_buffer": 60,
                    "match_thresh": 0.80,
                    "fuse_score": True,
                    "raw_detections_per_frame": statistics.fmean(
                        int(row["raw_person_detections"]) for row in measured
                    ),
                    "active_tracks_per_frame": statistics.fmean(
                        int(row["active_tracks"]) for row in measured
                    ),
                    "peak_active_tracks": max(int(row["active_tracks"]) for row in measured),
                    "unique_track_ids": len(seen_ids),
                    "processing_fps": len(measured) / processing_seconds,
                    "inference_ms": statistics.fmean(
                        float(row["inference_ms"]) for row in measured
                    ),
                    "point_precision_proxy": total_matches / total_predictions
                    if total_predictions
                    else "",
                    "point_recall_proxy": total_matches / total_ground_truth
                    if total_ground_truth
                    else "",
                    "annotated_gt_points": total_ground_truth,
                    "annotated_prediction_points": total_predictions,
                    "selected": False,
                }
            )

    eligible = [row for row in summary_rows if float(row["processing_fps"]) >= args.min_fps]
    candidates = eligible or summary_rows
    selected = max(
        candidates,
        key=lambda row: (float(row["active_tracks_per_frame"]), float(row["processing_fps"])),
    )
    selected["selected"] = True

    output = Path(args.output)
    frame_log = Path(args.frame_log)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame_log.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    with frame_log.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(frame_rows[0]))
        writer.writeheader()
        writer.writerows(frame_rows)
    print(f"Selected imgsz={selected['imgsz']} confidence={selected['confidence']}")
    print(f"Wrote {output.resolve()}")
    print(f"Wrote {frame_log.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
