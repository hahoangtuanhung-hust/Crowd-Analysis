import json
from pathlib import Path

import pytest

from scripts.compare_run_quality import compare_caches


def _write_cache(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_compare_caches_reports_detection_track_and_config_differences(
    tmp_path: Path,
) -> None:
    baseline_cache = tmp_path / "baseline.jsonl"
    candidate_cache = tmp_path / "candidate.jsonl"
    baseline_rows = [{
        "frame_id": 0,
        "event_time_s": 0.0,
        "detections": [
            {"x1": 0, "y1": 0, "x2": 10, "y2": 20},
            {"x1": 30, "y1": 0, "x2": 40, "y2": 20},
        ],
        "tracks": [
            {"track_id": 7, "x1": 0, "y1": 0, "x2": 10, "y2": 20, "observed": True},
            {"track_id": 9, "x1": 50, "y1": 0, "x2": 60, "y2": 20, "observed": False},
        ],
    }]
    candidate_rows = [{
        "frame_id": 0,
        "event_time_s": 0.0,
        "detections": [
            {"x1": 0.5, "y1": 0, "x2": 10.5, "y2": 20},
        ],
        "tracks": [
            {"track_id": 8, "x1": 0.5, "y1": 0, "x2": 10.5, "y2": 20, "observed": True},
        ],
    }]
    _write_cache(baseline_cache, baseline_rows)
    _write_cache(candidate_cache, candidate_rows)
    baseline_config = tmp_path / "baseline.yaml"
    candidate_config = tmp_path / "candidate.yaml"
    baseline_config.write_text("detector:\n  precision: fp32\n", encoding="utf-8")
    candidate_config.write_text("detector:\n  precision: fp16\n", encoding="utf-8")

    result = compare_caches(
        baseline_cache,
        candidate_cache,
        baseline_config=baseline_config,
        candidate_config=candidate_config,
    )

    assert result["quality_status"] == "REVIEW_REQUIRED_NO_GROUND_TRUTH"
    assert result["detections"]["baseline_items"] == 2
    assert result["detections"]["candidate_items"] == 1
    assert result["detections"]["iou50_candidate_match_percent"] == 100.0
    assert result["observed_tracks"]["baseline_items"] == 1
    assert result["observed_tracks"]["same_track_id_among_iou50_matches_percent"] == 0.0
    assert result["resolved_config_differences"] == [{
        "key": "detector.precision",
        "baseline": "fp32",
        "candidate": "fp16",
    }]


def test_compare_caches_rejects_frame_or_row_mismatch(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.jsonl"
    candidate = tmp_path / "candidate.jsonl"
    _write_cache(baseline, [{"frame_id": 0, "event_time_s": 0, "detections": [], "tracks": []}])
    _write_cache(candidate, [{"frame_id": 1, "event_time_s": 0, "detections": [], "tracks": []}])

    with pytest.raises(ValueError, match="Frame ID differs"):
        compare_caches(baseline, candidate)


def test_sampled_comparison_limits_detection_to_candidate_scan_frames(
    tmp_path: Path,
) -> None:
    baseline = tmp_path / "baseline.jsonl"
    candidate = tmp_path / "candidate.jsonl"
    common_detection = {"x1": 1, "y1": 1, "x2": 5, "y2": 9, "confidence": 0.8}
    baseline_rows = []
    candidate_rows = []
    for frame_id in range(4):
        baseline_rows.append({
            "frame_id": frame_id,
            "event_time_s": frame_id / 10,
            "detections": [common_detection],
            "tracks": [{**common_detection, "track_id": 1, "observed": True}],
        })
        scanned = frame_id in (0, 3)
        candidate_rows.append({
            "frame_id": frame_id,
            "event_time_s": frame_id / 10,
            "detections": [common_detection] if scanned else [],
            "tracks": [{**common_detection, "track_id": 1, "observed": scanned}],
            "scheduler_decision": {"scan_type": "reference" if scanned else "skip"},
        })
    _write_cache(baseline, baseline_rows)
    _write_cache(candidate, candidate_rows)

    result = compare_caches(baseline, candidate)

    assert result["candidate_detection_scan_frames"] == 2
    assert result["candidate_inference_interval_observed"] == 3
    assert result["detections"]["frames"] == 2
    assert result["detections"]["candidate_count_delta_percent"] == 0.0
    assert result["all_tracks_including_predictions"]["candidate_prediction_only_rows"] == 2
    assert result["all_tracks_including_predictions"]["iou50_candidate_match_percent"] == 100.0
