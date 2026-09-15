from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from collections.abc import Iterable
from statistics import fmean, median
from typing import Any

from backend.app.datasets.grand_central import GrandCentralDataset, NormalizedTrajectoryPoint


def _greedy_point_matches(
    ground_truth: list[NormalizedTrajectoryPoint],
    prediction: list[NormalizedTrajectoryPoint],
    *,
    threshold_pixels: float,
) -> list[tuple[int, int, float]]:
    candidates = []
    for gt_index, gt in enumerate(ground_truth):
        for pred_index, pred in enumerate(prediction):
            distance = math.hypot(gt.foot_x - pred.foot_x, gt.foot_y - pred.foot_y)
            if distance <= threshold_pixels:
                candidates.append((distance, gt_index, pred_index))
    matched_gt: set[int] = set()
    matched_prediction: set[int] = set()
    matches = []
    for distance, gt_index, pred_index in sorted(candidates):
        if gt_index in matched_gt or pred_index in matched_prediction:
            continue
        matched_gt.add(gt_index)
        matched_prediction.add(pred_index)
        matches.append((gt_index, pred_index, distance))
    return matches


def evaluate_predictions(
    dataset: GrandCentralDataset,
    ground_truth: Iterable[NormalizedTrajectoryPoint],
    prediction: Iterable[NormalizedTrajectoryPoint],
    *,
    detections: Iterable[NormalizedTrajectoryPoint] | None = None,
    threshold_pixels: float = 60.0,
    duration_label: str,
    inference_interval: int,
    persist: bool = True,
) -> dict[str, Any]:
    if threshold_pixels <= 0:
        raise ValueError("threshold_pixels must be positive")
    gt_points = list(ground_truth)
    pred_points = list(prediction)
    detection_points = list(detections) if detections is not None else pred_points
    if not gt_points:
        raise ValueError("Ground-truth evaluation set is empty")
    if {point.source for point in gt_points} != {"ground_truth"}:
        raise ValueError("Ground-truth input contains another source")
    if pred_points and {point.source for point in pred_points} != {"prediction"}:
        raise ValueError("Prediction input contains another source")
    if detection_points and {point.source for point in detection_points} != {"prediction"}:
        raise ValueError("Detection input contains another source")

    video_fps = float(dataset.video_metadata()["fps"])
    gt_by_video_frame: dict[int, list[NormalizedTrajectoryPoint]] = defaultdict(list)
    for point in gt_points:
        mapped = round(point.timestamp * video_fps)
        gt_by_video_frame[mapped].append(point)
    tracked_by_frame: dict[int, list[NormalizedTrajectoryPoint]] = defaultdict(list)
    for point in pred_points:
        tracked_by_frame[point.frame_id].append(point)
    pred_by_frame: dict[int, list[NormalizedTrajectoryPoint]] = defaultdict(list)
    for point in detection_points:
        pred_by_frame[point.frame_id].append(point)

    absolute_count_errors = []
    percentage_count_errors = []
    count_bias = []
    match_distances = []
    matched_gt_tracks: set[int] = set()
    matched_prediction_tracks: set[int] = set()
    total_matches = 0
    total_gt = 0
    total_prediction = 0
    matched_frames = 0
    frames_with_prediction = 0
    per_frame = []
    for video_frame_id, gt_frame in sorted(gt_by_video_frame.items()):
        pred_frame = pred_by_frame.get(video_frame_id, [])
        matches = _greedy_point_matches(
            gt_frame, pred_frame, threshold_pixels=threshold_pixels
        )
        track_matches = _greedy_point_matches(
            gt_frame,
            tracked_by_frame.get(video_frame_id, []),
            threshold_pixels=threshold_pixels,
        )
        gt_count = len(gt_frame)
        pred_count = len(pred_frame)
        error = pred_count - gt_count
        absolute_count_errors.append(abs(error))
        if gt_count:
            percentage_count_errors.append(abs(error) / gt_count * 100.0)
        count_bias.append(error)
        total_gt += gt_count
        total_prediction += pred_count
        total_matches += len(matches)
        if pred_count:
            frames_with_prediction += 1
        if matches:
            matched_frames += 1
        for gt_index, pred_index, distance in matches:
            matched_gt_tracks.add(gt_frame[gt_index].track_id)
            match_distances.append(distance)
        tracked_frame = tracked_by_frame.get(video_frame_id, [])
        for _, pred_index, _ in track_matches:
            matched_prediction_tracks.add(tracked_frame[pred_index].track_id)
        per_frame.append(
            {
                "ground_truth_source": "ground_truth",
                "prediction_source": "prediction",
                "annotation_frame_id": gt_frame[0].frame_id,
                "video_frame_id": video_frame_id,
                "timestamp": gt_frame[0].timestamp,
                "ground_truth_count": gt_count,
                "prediction_count": pred_count,
                "count_error": error,
                "matches": len(matches),
                "false_positives": pred_count - len(matches),
                "false_negatives": gt_count - len(matches),
            }
        )

    gt_track_ids = {point.track_id for point in gt_points}
    pred_track_ids_at_annotations = {
        point.track_id
        for frame_id, points in tracked_by_frame.items()
        if frame_id in gt_by_video_frame
        for point in points
    }
    frame_count = len(per_frame)
    result = {
        "dataset": "grand-central",
        "evaluation": "timestamp_aligned_point_matching",
        "ground_truth": {
            "source": "ground_truth",
            "points": total_gt,
            "tracks": len(gt_track_ids),
        },
        "prediction": {
            "source": "prediction",
            "detection_points_at_annotated_frames": total_prediction,
            "tracked_points_at_annotated_frames": sum(
                len(tracked_by_frame.get(frame_id, [])) for frame_id in gt_by_video_frame
            ),
            "tracks_at_annotated_frames": len(pred_track_ids_at_annotations),
            "inference_interval": inference_interval,
        },
        "duration": duration_label,
        "timebase": {
            "annotation_source_fps": dataset.source_fps,
            "video_fps": video_fps,
            "mapping": "video_frame_id = round((annotation_frame_id / 25) * video_fps)",
            "annotated_frames": frame_count,
            "frames_with_prediction": frames_with_prediction,
        },
        "matching": {
            "method": "greedy one-to-one Euclidean matching of trajectory/bottom-center points",
            "threshold_pixels": threshold_pixels,
        },
        "metrics": {
            "person_count_mae": round(fmean(absolute_count_errors), 3),
            "person_count_mape_percent": round(fmean(percentage_count_errors), 3),
            "person_count_bias": round(fmean(count_bias), 3),
            "detection_point_precision": round(total_matches / total_prediction, 6)
            if total_prediction
            else 0.0,
            "detection_point_recall": round(total_matches / total_gt, 6) if total_gt else 0.0,
            "matched_point_count": total_matches,
            "matched_frame_coverage": round(matched_frames / frame_count, 6),
            "matched_ground_truth_track_coverage": round(
                len(matched_gt_tracks) / len(gt_track_ids), 6
            )
            if gt_track_ids
            else 0.0,
            "prediction_trajectory_coverage": round(
                len(matched_prediction_tracks) / len(pred_track_ids_at_annotations), 6
            )
            if pred_track_ids_at_annotations
            else 0.0,
            "mean_match_distance_pixels": round(fmean(match_distances), 3)
            if match_distances
            else None,
            "median_match_distance_pixels": round(median(match_distances), 3)
            if match_distances
            else None,
            "idf1": None,
            "hota": None,
        },
        "limitations": [
            "Ground truth contains sparse trajectory points, not bounding boxes; detection precision and recall are point-matching proxies, not IoU metrics.",
            "Ground-truth and prediction track IDs are unrelated and annotations are sampled at 1.25 Hz; IDF1 and HOTA are not valid here.",
            "The supplied MP4 is 30 fps while annotation frame IDs use the 25 fps source timebase.",
        ],
        "per_frame": per_frame,
    }
    if persist:
        stem = f"prediction_vs_ground_truth_interval_{inference_interval}_{duration_label}"
        json_path = dataset.paths.output / "benchmarks" / f"{stem}.json"
        csv_path = dataset.paths.output / "benchmarks" / f"{stem}.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=per_frame[0].keys())
            writer.writeheader()
            writer.writerows(per_frame)
        result["artifacts"] = {
            "evaluation_json": str(json_path.resolve()),
            "per_frame_csv": str(csv_path.resolve()),
        }
        json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
