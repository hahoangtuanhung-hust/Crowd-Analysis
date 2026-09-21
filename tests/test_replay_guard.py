from __future__ import annotations

import subprocess
import sys
import json
import hashlib
from pathlib import Path

import cv2
import numpy as np

from backend.app.schemas import FramePacket
from scripts.replay_ui import CachedDetector, ReplayCounters, ReplaySessionManager


def test_renderer_import_does_not_eagerly_load_detector() -> None:
    command = (
        "import sys; import scripts.diagnose_common_path; "
        "assert not any(name.startswith('backend.app.inference') for name in sys.modules)"
    )
    result = subprocess.run([sys.executable, "-c", command], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_cached_detector_reset_rewinds_without_inference(tmp_path: Path) -> None:
    cache = tmp_path / "tracking_cache.jsonl"
    rows = [
        {"frame_id": 0, "event_time_s": 0.0, "detections": [{"x1": 1, "y1": 2, "x2": 3, "y2": 4, "confidence": 0.9}]},
        {"frame_id": 1, "source_timestamp": 0.1, "detections": []},
    ]
    cache.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    counters = ReplayCounters()
    detector = CachedDetector(cache, counters)

    assert len(detector.detect(np.zeros((4, 4, 3), dtype=np.uint8))) == 1
    assert detector.detect(np.zeros((4, 4, 3), dtype=np.uint8)) == []
    detector.reset()
    assert len(detector.detect(np.zeros((4, 4, 3), dtype=np.uint8))) == 1
    assert counters.detector_calls == 0
    assert counters.cache_reads == 3
    assert counters.cache_resets == 1


def test_cache_reader_aligns_dropped_frame_by_id_and_timestamp(tmp_path: Path) -> None:
    cache = tmp_path / "tracking_cache.jsonl"
    cache.write_text("".join(json.dumps({
        "frame_id": frame_id,
        "event_time_s": frame_id / 30,
        "detections": [{"x1": frame_id, "y1": 2, "x2": 3, "y2": 4, "confidence": 0.9}],
    }) + "\n" for frame_id in range(3)), encoding="utf-8")
    counters = ReplayCounters()
    detector = CachedDetector(cache, counters)
    frame = np.zeros((4, 4, 3), dtype=np.uint8)

    detections = detector.detect_packet(FramePacket(2, 2 / 30, 0.0, frame))

    assert detections[0].x1 == 2
    assert counters.cache_skipped_rows == 2
    assert counters.cache_mismatch_count == 0
    detector.reset()
    assert detector.detect_packet(FramePacket(0, 0.0, 0.0, frame))[0].x1 == 0


def test_replay_rejects_wrong_source_before_start(tmp_path: Path) -> None:
    from backend.app.core.config import AppConfig

    cache = tmp_path / "tracking_cache.jsonl"
    cache.write_text("", encoding="utf-8")
    cache.with_suffix(".meta.json").write_text(json.dumps({"source_hash": "wrong"}), encoding="utf-8")
    source = tmp_path / "video.mp4"
    source.write_bytes(b"not the cached source")
    manager = ReplaySessionManager(AppConfig(), cache, ReplayCounters())

    try:
        manager.start(source_uri=str(source), source_kind="upload", realtime=True)
    except ValueError as exc:
        assert "SHA-256" in str(exc)
    else:
        raise AssertionError("Mismatched replay source was accepted")


def test_replay_completes_at_cache_boundary_before_source_eof(tmp_path: Path) -> None:
    from backend.app.core.config import AppConfig

    source = tmp_path / "longer-source.mp4"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 10, (160, 120))
    assert writer.isOpened()
    for _ in range(8):
        writer.write(np.zeros((120, 160, 3), dtype=np.uint8))
    writer.release()

    cache = tmp_path / "tracking_cache.jsonl"
    cache.write_text("".join(json.dumps({"frame_id": index, "event_time_s": index / 10,
                                         "detections": []}) + "\n" for index in range(4)),
                     encoding="utf-8")
    cache.with_suffix(".meta.json").write_text(json.dumps({
        "source_hash": hashlib.sha256(source.read_bytes()).hexdigest(), "frame_count": 4,
    }), encoding="utf-8")
    counters = ReplayCounters()
    manager = ReplaySessionManager(AppConfig(), cache, counters)
    session = manager.start(source_uri=str(source), source_kind="upload", realtime=False)

    assert session.wait(timeout=10)
    assert session.snapshot().status == "completed"
    assert session.snapshot().frame_id == 3
    assert session.snapshot().error is None
    assert counters.cache_mismatch_count == 0
    assert counters.detector_calls == 0
