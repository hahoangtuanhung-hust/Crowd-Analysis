"""Compare two GPU-run caches without treating the baseline as ground truth."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from itertools import zip_longest
from pathlib import Path
from typing import Any

import numpy as np
import yaml


@dataclass
class MatchTotals:
    frames: int = 0
    baseline_items: int = 0
    candidate_items: int = 0
    equal_count_frames: int = 0
    absolute_count_deltas: list[int] = field(default_factory=list)
    matches_iou50: int = 0
    matches_iou75: int = 0
    matched_ious: list[float] = field(default_factory=list)
    coordinate_errors: list[float] = field(default_factory=list)
    confidence_errors: list[float] = field(default_factory=list)
    numeric_changed_matches: int = 0
    same_ids: int = 0
    id_pair_counts: dict[tuple[int, int], int] = field(default_factory=dict)


def _boxes(items: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray(
        [[item["x1"], item["y1"], item["x2"], item["y2"]] for item in items],
        dtype=np.float64,
    ).reshape(-1, 4)


def _iou_matrix(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    if not len(first) or not len(second):
        return np.zeros((len(first), len(second)), dtype=np.float64)
    top_left = np.maximum(first[:, None, :2], second[None, :, :2])
    bottom_right = np.minimum(first[:, None, 2:], second[None, :, 2:])
    size = np.maximum(bottom_right - top_left, 0.0)
    intersection = size[..., 0] * size[..., 1]
    first_area = np.maximum(first[:, 2] - first[:, 0], 0.0) * np.maximum(
        first[:, 3] - first[:, 1], 0.0
    )
    second_area = np.maximum(second[:, 2] - second[:, 0], 0.0) * np.maximum(
        second[:, 3] - second[:, 1], 0.0
    )
    union = first_area[:, None] + second_area[None, :] - intersection
    return np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection),
        where=union > 0.0,
    )


def _greedy_matches(
    matrix: np.ndarray,
    threshold: float,
) -> list[tuple[int, int, float]]:
    rows, columns = np.nonzero(matrix >= threshold)
    if not len(rows):
        return []
    scores = matrix[rows, columns]
    order = np.argsort(-scores, kind="stable")
    used_rows: set[int] = set()
    used_columns: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for offset in order:
        row = int(rows[offset])
        column = int(columns[offset])
        if row in used_rows or column in used_columns:
            continue
        used_rows.add(row)
        used_columns.add(column)
        matches.append((row, column, float(scores[offset])))
    return matches


def _add_frame(
    totals: MatchTotals,
    baseline: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
    *,
    compare_ids: bool,
) -> None:
    totals.frames += 1
    totals.baseline_items += len(baseline)
    totals.candidate_items += len(candidate)
    totals.equal_count_frames += int(len(baseline) == len(candidate))
    totals.absolute_count_deltas.append(abs(len(baseline) - len(candidate)))
    baseline_boxes = _boxes(baseline)
    candidate_boxes = _boxes(candidate)
    matrix = _iou_matrix(baseline_boxes, candidate_boxes)
    matches50 = _greedy_matches(matrix, 0.50)
    totals.matches_iou50 += len(matches50)
    totals.matches_iou75 += len(_greedy_matches(matrix, 0.75))
    totals.matched_ious.extend(match[2] for match in matches50)
    for baseline_index, candidate_index, _ in matches50:
        coordinate_error = np.abs(
            baseline_boxes[baseline_index] - candidate_boxes[candidate_index]
        )
        totals.coordinate_errors.extend(coordinate_error.tolist())
        confidence_error = abs(
            float(baseline[baseline_index].get("confidence", 0.0))
            - float(candidate[candidate_index].get("confidence", 0.0))
        )
        totals.confidence_errors.append(confidence_error)
        totals.numeric_changed_matches += int(
            bool(np.any(coordinate_error > 1e-6)) or confidence_error > 1e-7
        )
        if compare_ids:
            baseline_id = int(baseline[baseline_index]["track_id"])
            candidate_id = int(candidate[candidate_index]["track_id"])
            totals.same_ids += int(baseline_id == candidate_id)
            pair = (baseline_id, candidate_id)
            totals.id_pair_counts[pair] = totals.id_pair_counts.get(pair, 0) + 1


def _percent(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return round(100.0 * numerator / denominator, 3)


def _finish(totals: MatchTotals, *, include_ids: bool) -> dict[str, Any]:
    def percentile(values: list[float], value: float) -> float | None:
        return round(float(np.percentile(values, value)), 6) if values else None

    output: dict[str, Any] = {
        "frames": totals.frames,
        "baseline_items": totals.baseline_items,
        "candidate_items": totals.candidate_items,
        "candidate_count_delta_percent": (
            round(100.0 * (totals.candidate_items / totals.baseline_items - 1.0), 3)
            if totals.baseline_items else None
        ),
        "frames_with_equal_count_percent": _percent(
            totals.equal_count_frames, totals.frames
        ),
        "mean_absolute_count_delta": (
            round(float(np.mean(totals.absolute_count_deltas)), 3)
            if totals.absolute_count_deltas else None
        ),
        "iou50_baseline_match_percent": _percent(
            totals.matches_iou50, totals.baseline_items
        ),
        "iou50_candidate_match_percent": _percent(
            totals.matches_iou50, totals.candidate_items
        ),
        "iou75_baseline_match_percent": _percent(
            totals.matches_iou75, totals.baseline_items
        ),
        "iou75_candidate_match_percent": _percent(
            totals.matches_iou75, totals.candidate_items
        ),
        "matched_iou_median": (
            round(float(np.median(totals.matched_ious)), 4)
            if totals.matched_ious else None
        ),
        "matched_coordinate_error_median_px": (
            round(float(np.median(totals.coordinate_errors)), 4)
            if totals.coordinate_errors else None
        ),
        "matched_coordinate_error_p95_px": percentile(
            totals.coordinate_errors, 95
        ),
        "matched_coordinate_error_max_px": (
            round(max(totals.coordinate_errors), 6)
            if totals.coordinate_errors else None
        ),
        "matched_confidence_error_median": percentile(
            totals.confidence_errors, 50
        ),
        "matched_confidence_error_p95": percentile(
            totals.confidence_errors, 95
        ),
        "matched_confidence_error_max": (
            round(max(totals.confidence_errors), 8)
            if totals.confidence_errors else None
        ),
        "iou50_matches": totals.matches_iou50,
        "mean_iou50_matches_per_frame": (
            round(totals.matches_iou50 / totals.frames, 3)
            if totals.frames else None
        ),
        "unmatched_baseline_items": totals.baseline_items - totals.matches_iou50,
        "unmatched_candidate_items": totals.candidate_items - totals.matches_iou50,
        "matched_items_with_numeric_change": totals.numeric_changed_matches,
    }
    if include_ids:
        output["same_track_id_among_iou50_matches_percent"] = _percent(
            totals.same_ids, totals.matches_iou50
        )
        output["same_track_id_matches"] = totals.same_ids
        used_baseline: set[int] = set()
        used_candidate: set[int] = set()
        mapped_observations = 0
        mapped_pairs = 0
        for (baseline_id, candidate_id), count in sorted(
            totals.id_pair_counts.items(), key=lambda item: item[1], reverse=True
        ):
            if baseline_id in used_baseline or candidate_id in used_candidate:
                continue
            used_baseline.add(baseline_id)
            used_candidate.add(candidate_id)
            mapped_observations += count
            mapped_pairs += 1
        output["mapped_track_id_agreement_percent"] = _percent(
            mapped_observations, totals.matches_iou50
        )
        output["mapped_track_id_observations"] = mapped_observations
        output["one_to_one_track_id_pairs"] = mapped_pairs
        output["track_id_mapping_method"] = (
            "greedy one-to-one mapping by matched observation count"
        )
    return output


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if not isinstance(value, dict):
        return {prefix: value}
    flattened: dict[str, Any] = {}
    for key, nested in value.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        flattened.update(_flatten(nested, path))
    return flattened


def _config_differences(
    baseline_path: Path | None,
    candidate_path: Path | None,
) -> list[dict[str, Any]]:
    if baseline_path is None or candidate_path is None:
        return []
    baseline = _flatten(yaml.safe_load(baseline_path.read_text(encoding="utf-8")) or {})
    candidate = _flatten(yaml.safe_load(candidate_path.read_text(encoding="utf-8")) or {})
    return [
        {
            "key": key,
            "baseline": baseline.get(key),
            "candidate": candidate.get(key),
        }
        for key in sorted(baseline.keys() | candidate.keys())
        if baseline.get(key) != candidate.get(key)
    ]


def compare_caches(
    baseline_cache: Path,
    candidate_cache: Path,
    *,
    baseline_config: Path | None = None,
    candidate_config: Path | None = None,
    frame_height: int = 720,
) -> dict[str, Any]:
    detections = MatchTotals()
    tracks = MatchTotals()
    all_tracks = MatchTotals()
    far_tracks = MatchTotals()
    region_detections = {
        name: MatchTotals() for name in ("TOP_FAR", "MIDDLE", "BOTTOM_NEAR")
    }
    region_tracks = {
        name: MatchTotals() for name in ("TOP_FAR", "MIDDLE", "BOTTOM_NEAR")
    }
    baseline_track_ids: set[int] = set()
    candidate_track_ids: set[int] = set()
    baseline_track_frames: dict[int, list[int]] = {}
    candidate_track_frames: dict[int, list[int]] = {}
    baseline_all_track_frames: dict[int, list[int]] = {}
    candidate_all_track_frames: dict[int, list[int]] = {}
    baseline_timestamps: dict[int, float] = {}
    candidate_scan_frames: list[int] = []
    baseline_prediction_rows = 0
    candidate_prediction_rows = 0
    exact_track_id_set_frames = 0
    with baseline_cache.open("r", encoding="utf-8") as baseline_stream, candidate_cache.open(
        "r", encoding="utf-8"
    ) as candidate_stream:
        for row_index, (baseline_line, candidate_line) in enumerate(
            zip_longest(baseline_stream, candidate_stream), start=1
        ):
            if baseline_line is None or candidate_line is None:
                raise ValueError(f"Cache row count differs at row {row_index}")
            baseline_row = json.loads(baseline_line)
            candidate_row = json.loads(candidate_line)
            if baseline_row["frame_id"] != candidate_row["frame_id"]:
                raise ValueError(f"Frame ID differs at row {row_index}")
            if abs(
                float(baseline_row["event_time_s"])
                - float(candidate_row["event_time_s"])
            ) > 1e-6:
                raise ValueError(f"Source timestamp differs at row {row_index}")
            candidate_plan = candidate_row.get("scheduler_decision") or {}
            candidate_scan_type = str(candidate_plan.get("scan_type", "reference"))
            frame_id = int(baseline_row["frame_id"])
            baseline_timestamps[frame_id] = float(baseline_row["event_time_s"])
            if candidate_scan_type != "skip":
                candidate_scan_frames.append(frame_id)
                baseline_detections = baseline_row.get("detections", [])
                candidate_detections = candidate_row.get("detections", [])
                _add_frame(
                    detections,
                    baseline_detections,
                    candidate_detections,
                    compare_ids=False,
                )
                for name, lower, upper in (
                    ("TOP_FAR", 0.0, 1.0 / 3.0),
                    ("MIDDLE", 1.0 / 3.0, 2.0 / 3.0),
                    ("BOTTOM_NEAR", 2.0 / 3.0, 1.0 + 1e-9),
                ):
                    def in_band(item: dict[str, Any]) -> bool:
                        center_y = (float(item["y1"]) + float(item["y2"])) / 2.0
                        normalized_y = center_y / max(1, frame_height)
                        return lower <= normalized_y < upper

                    _add_frame(
                        region_detections[name],
                        [item for item in baseline_detections if in_band(item)],
                        [item for item in candidate_detections if in_band(item)],
                        compare_ids=False,
                    )
            baseline_tracks = [
                item for item in baseline_row.get("tracks", [])
                if item.get("observed", True)
            ]
            candidate_tracks = [
                item for item in candidate_row.get("tracks", [])
                if item.get("observed", True)
            ]
            baseline_track_ids.update(int(item["track_id"]) for item in baseline_tracks)
            candidate_track_ids.update(int(item["track_id"]) for item in candidate_tracks)
            baseline_ids_this_frame = {int(item["track_id"]) for item in baseline_tracks}
            candidate_ids_this_frame = {int(item["track_id"]) for item in candidate_tracks}
            exact_track_id_set_frames += int(
                baseline_ids_this_frame == candidate_ids_this_frame
            )
            for track_id in baseline_ids_this_frame:
                baseline_track_frames.setdefault(track_id, []).append(frame_id)
            for track_id in candidate_ids_this_frame:
                candidate_track_frames.setdefault(track_id, []).append(frame_id)
            _add_frame(
                tracks,
                baseline_tracks,
                candidate_tracks,
                compare_ids=True,
            )
            baseline_all_tracks = baseline_row.get("tracks", [])
            candidate_all_tracks = candidate_row.get("tracks", [])
            baseline_prediction_rows += sum(
                not item.get("observed", True) for item in baseline_all_tracks
            )
            candidate_prediction_rows += sum(
                not item.get("observed", True) for item in candidate_all_tracks
            )
            for item in baseline_all_tracks:
                baseline_all_track_frames.setdefault(int(item["track_id"]), []).append(frame_id)
            for item in candidate_all_tracks:
                candidate_all_track_frames.setdefault(int(item["track_id"]), []).append(frame_id)
            _add_frame(
                all_tracks,
                baseline_all_tracks,
                candidate_all_tracks,
                compare_ids=True,
            )
            for name, lower, upper in (
                ("TOP_FAR", 0.0, 1.0 / 3.0),
                ("MIDDLE", 1.0 / 3.0, 2.0 / 3.0),
                ("BOTTOM_NEAR", 2.0 / 3.0, 1.0 + 1e-9),
            ):
                def track_in_band(item: dict[str, Any]) -> bool:
                    center_y = (float(item["y1"]) + float(item["y2"])) / 2.0
                    normalized_y = center_y / max(1, frame_height)
                    return lower <= normalized_y < upper

                _add_frame(
                    region_tracks[name],
                    [item for item in baseline_all_tracks if track_in_band(item)],
                    [item for item in candidate_all_tracks if track_in_band(item)],
                    compare_ids=True,
                )
            far_baseline = [
                item for item in baseline_all_tracks
                if (
                    (float(item["y1"]) + float(item["y2"])) / 2.0
                    / max(1, frame_height) < 1.0 / 3.0
                )
            ]
            far_candidate = [
                item for item in candidate_all_tracks
                if (
                    (float(item["y1"]) + float(item["y2"])) / 2.0
                    / max(1, frame_height) < 1.0 / 3.0
                )
            ]
            _add_frame(far_tracks, far_baseline, far_candidate, compare_ids=True)
    track_result = _finish(tracks, include_ids=True)
    track_result["baseline_unique_observed_track_ids"] = len(baseline_track_ids)
    track_result["candidate_unique_observed_track_ids"] = len(candidate_track_ids)
    track_result["changed_unique_track_ids"] = len(
        baseline_track_ids.symmetric_difference(candidate_track_ids)
    )
    track_result["frames_with_exact_observed_track_id_set_percent"] = _percent(
        exact_track_id_set_frames, tracks.frames
    )

    def continuity(track_frames: dict[int, list[int]]) -> dict[str, Any]:
        fragments = 0
        adjacent = 0
        possible = 0
        gap_events = 0
        for frames in track_frames.values():
            ordered = sorted(set(frames))
            if not ordered:
                continue
            fragments += 1
            possible += max(0, ordered[-1] - ordered[0])
            for previous, current in zip(ordered, ordered[1:]):
                adjacent += int(current == previous + 1)
                if current > previous + 1:
                    fragments += 1
                    gap_events += 1
        return {
            "fragments": fragments,
            "gap_events": gap_events,
            "adjacent_frame_continuity_percent": _percent(adjacent, possible),
        }

    track_result["baseline_continuity"] = continuity(baseline_track_frames)
    track_result["candidate_continuity"] = continuity(candidate_track_frames)
    scan_gaps = [
        current - previous
        for previous, current in zip(candidate_scan_frames, candidate_scan_frames[1:])
    ]
    candidate_interval = int(round(float(np.median(scan_gaps)))) if scan_gaps else 1

    def lifecycle(
        observed_frames: dict[int, list[int]],
        present_frames: dict[int, list[int]],
        expected_observation_gap: int,
    ) -> dict[str, Any]:
        durations: list[float] = []
        observed_scans: list[int] = []
        max_observation_gap = 0
        observation_gap_excess_events = 0
        presence_fragments = 0
        lost_recovered_events = 0
        for track_id, frames in present_frames.items():
            ordered = sorted(set(frames))
            if not ordered:
                continue
            durations.append(
                baseline_timestamps.get(ordered[-1], float(ordered[-1]))
                - baseline_timestamps.get(ordered[0], float(ordered[0]))
            )
            fragments = 1 + sum(
                current > previous + 1
                for previous, current in zip(ordered, ordered[1:])
            )
            presence_fragments += fragments
            lost_recovered_events += max(0, fragments - 1)
            observations = sorted(set(observed_frames.get(track_id, [])))
            observed_scans.append(len(observations))
            for previous, current in zip(observations, observations[1:]):
                gap = current - previous
                max_observation_gap = max(max_observation_gap, gap)
                observation_gap_excess_events += int(gap > expected_observation_gap)
        return {
            "unique_track_ids": len(present_frames),
            "observed_track_ids": len(observed_frames),
            "track_duration_p50_s": (
                round(float(np.percentile(durations, 50)), 4) if durations else None
            ),
            "track_duration_p95_s": (
                round(float(np.percentile(durations, 95)), 4) if durations else None
            ),
            "observed_scans_per_track_p50": (
                round(float(np.percentile(observed_scans, 50)), 3)
                if observed_scans else None
            ),
            "observed_scans_per_track_p95": (
                round(float(np.percentile(observed_scans, 95)), 3)
                if observed_scans else None
            ),
            "presence_fragments": presence_fragments,
            "lost_recovered_events": lost_recovered_events,
            "max_observation_gap_frames": max_observation_gap,
            "observation_gap_excess_events": observation_gap_excess_events,
            "expected_observation_gap_frames": expected_observation_gap,
        }

    all_track_result = _finish(all_tracks, include_ids=True)
    all_track_result["baseline_prediction_only_rows"] = baseline_prediction_rows
    all_track_result["candidate_prediction_only_rows"] = candidate_prediction_rows
    all_track_result["baseline_lifecycle"] = lifecycle(
        baseline_track_frames, baseline_all_track_frames, 1
    )
    all_track_result["candidate_lifecycle"] = lifecycle(
        candidate_track_frames, candidate_all_track_frames, candidate_interval
    )
    return {
        "quality_status": "REVIEW_REQUIRED_NO_GROUND_TRUTH",
        "matching_method": "greedy one-to-one matching by descending IoU",
        "limitations": [
            "The baseline is a regression reference, not ground truth.",
            "Track-ID agreement is a stability proxy, not IDF1 or HOTA.",
            "FAR is the fixed upper image third for this 1280x720 Shibuya clip.",
            "Common Path direction, branch choice, jitter and lifecycle require artifact review.",
        ],
        "candidate_detection_scan_frames": len(candidate_scan_frames),
        "candidate_inference_interval_observed": candidate_interval,
        "detections": _finish(detections, include_ids=False),
        "detections_by_normalized_region": {
            name: _finish(totals, include_ids=False)
            for name, totals in region_detections.items()
        },
        "observed_tracks": track_result,
        "all_tracks_including_predictions": all_track_result,
        "far_upper_third_tracks": _finish(far_tracks, include_ids=True),
        "tracks_by_normalized_region": {
            name: _finish(totals, include_ids=True)
            for name, totals in region_tracks.items()
        },
        "resolved_config_differences": _config_differences(
            baseline_config, candidate_config
        ),
    }


def compare_common_paths(
    baseline_summary: Path,
    candidate_summary: Path,
) -> dict[str, Any]:
    baseline = json.loads(baseline_summary.read_text(encoding="utf-8"))
    candidate = json.loads(candidate_summary.read_text(encoding="utf-8"))
    baseline_paths = baseline.get("active_paths", [])
    candidate_paths = candidate.get("active_paths", [])
    comparisons: list[dict[str, Any]] = []
    substantive = len(baseline_paths) != len(candidate_paths)
    for rank, (first, second) in enumerate(
        zip(baseline_paths, candidate_paths), start=1
    ):
        first_polyline = np.asarray(first.get("polyline", []), dtype=np.float64).reshape(-1, 2)
        second_polyline = np.asarray(second.get("polyline", []), dtype=np.float64).reshape(-1, 2)
        if len(first_polyline) == len(second_polyline) and len(first_polyline):
            distances = np.linalg.norm(first_polyline - second_polyline, axis=1)
            polyline_mean = float(np.mean(distances))
            polyline_max = float(np.max(distances))
        else:
            polyline_mean = polyline_max = None
        confidence_delta = abs(
            float(first.get("confidence", 0.0))
            - float(second.get("confidence", 0.0))
        )
        identity_changed = any(
            first.get(field) != second.get(field)
            for field in ("path_id", "direction", "support_tracks")
        )
        row_substantive = (
            identity_changed
            or len(first_polyline) != len(second_polyline)
            or confidence_delta > 0.01
            or (polyline_mean is not None and polyline_mean > 5.0)
        )
        substantive = substantive or row_substantive
        comparisons.append({
            "rank": rank,
            "baseline_path_id": first.get("path_id"),
            "candidate_path_id": second.get("path_id"),
            "baseline_direction": first.get("direction"),
            "candidate_direction": second.get("direction"),
            "baseline_support": first.get("support_tracks"),
            "candidate_support": second.get("support_tracks"),
            "baseline_confidence": first.get("confidence"),
            "candidate_confidence": second.get("confidence"),
            "confidence_absolute_delta": round(confidence_delta, 8),
            "baseline_polyline_points": len(first_polyline),
            "candidate_polyline_points": len(second_polyline),
            "polyline_mean_point_delta_px": (
                round(polyline_mean, 6) if polyline_mean is not None else None
            ),
            "polyline_max_point_delta_px": (
                round(polyline_max, 6) if polyline_max is not None else None
            ),
            "substantive_change": row_substantive,
        })
    return {
        "baseline_path_count": len(baseline_paths),
        "candidate_path_count": len(candidate_paths),
        "substantive_change": substantive,
        "criteria": {
            "identity_direction_or_support_changed": True,
            "confidence_absolute_delta_gt": 0.01,
            "polyline_mean_point_delta_px_gt": 5.0,
        },
        "paths": comparisons,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-cache", required=True, type=Path)
    parser.add_argument("--candidate-cache", required=True, type=Path)
    parser.add_argument("--baseline-config", type=Path)
    parser.add_argument("--candidate-config", type=Path)
    parser.add_argument("--baseline-summary", type=Path)
    parser.add_argument("--candidate-summary", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--frame-height", type=int, default=720)
    args = parser.parse_args()
    result = compare_caches(
        args.baseline_cache,
        args.candidate_cache,
        baseline_config=args.baseline_config,
        candidate_config=args.candidate_config,
        frame_height=args.frame_height,
    )
    if args.baseline_summary is not None or args.candidate_summary is not None:
        if args.baseline_summary is None or args.candidate_summary is None:
            parser.error("both --baseline-summary and --candidate-summary are required")
        result["common_path"] = compare_common_paths(
            args.baseline_summary, args.candidate_summary
        )
    rendered = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
