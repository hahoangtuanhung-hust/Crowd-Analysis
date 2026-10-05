from __future__ import annotations

import json
from pathlib import Path

from scripts.replay_detection_cache import replay


def test_replay_detection_cache_rebuilds_tracks_with_source_time(tmp_path: Path) -> None:
    cache = tmp_path / "detections.jsonl"
    rows = [
        {
            "frame_id": frame_id,
            "event_time_s": frame_id * 0.1,
            "detections": [
                {
                    "x1": 20 + frame_id * 2,
                    "y1": 20,
                    "x2": 40 + frame_id * 2,
                    "y2": 70,
                    "confidence": 0.9,
                    "class_id": 0,
                }
            ],
            "scheduler_decision": {"source_frame_shape": [100, 160, 3]},
        }
        for frame_id in range(4)
    ]
    cache.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    output = tmp_path / "replay"

    summary = replay(cache, Path("configs/default.yaml"), output)

    assert summary["quality_status"] == "QUALITY_REVIEW_PENDING"
    assert summary["frames"] == 4
    assert summary["observed_track_frames"] >= 3
    assert (output / "tracking_cache.jsonl").is_file()
    assert (output / "metrics.csv").is_file()
    assert (output / "provenance.json").is_file()
