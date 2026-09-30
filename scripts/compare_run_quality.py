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
    same_ids: int = 0


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
        totals.coordinate_errors.extend(
            np.abs(
                baseline_boxes[baseline_index] - candidate_boxes[candidate_index]
            ).tolist()
        )
        if compare_ids:
            totals.same_ids += int(
                baseline[baseline_index]["track_id"]
                == candidate[candidate_index]["track_id"]
            )


def _percent(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return round(100.0 * numerator / denominator, 3)


def _finish(totals: MatchTotals, *, include_ids: bool) -> dict[str, Any]:
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
    }
    if include_ids:
        output["same_track_id_among_iou50_matches_percent"] = _percent(
            totals.same_ids, totals.matches_iou50
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
) -> dict[str, Any]:
    detections = MatchTotals()
    tracks = MatchTotals()
    baseline_track_ids: set[int] = set()
    candidate_track_ids: set[int] = set()
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
            _add_frame(
                detections,
                baseline_row.get("detections", []),
                candidate_row.get("detections", []),
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
            _add_frame(
                tracks,
                baseline_tracks,
                candidate_tracks,
                compare_ids=True,
            )
    track_result = _finish(tracks, include_ids=True)
    track_result["baseline_unique_observed_track_ids"] = len(baseline_track_ids)
    track_result["candidate_unique_observed_track_ids"] = len(candidate_track_ids)
    return {
        "quality_status": "REVIEW_REQUIRED_NO_GROUND_TRUTH",
        "matching_method": "greedy one-to-one matching by descending IoU",
        "limitations": [
            "The baseline is a regression reference, not ground truth.",
            "Track-ID agreement is a stability proxy, not IDF1 or HOTA.",
            "Common Path direction, branch choice, jitter and lifecycle require artifact review.",
        ],
        "detections": _finish(detections, include_ids=False),
        "observed_tracks": track_result,
        "resolved_config_differences": _config_differences(
            baseline_config, candidate_config
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-cache", required=True, type=Path)
    parser.add_argument("--candidate-cache", required=True, type=Path)
    parser.add_argument("--baseline-config", type=Path)
    parser.add_argument("--candidate-config", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = compare_caches(
        args.baseline_cache,
        args.candidate_cache,
        baseline_config=args.baseline_config,
        candidate_config=args.candidate_config,
    )
    rendered = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
