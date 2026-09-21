from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from scripts.live_common_path import LiveCommonPathProcessor


class EmptyDetector:
    def detect(self, _frame: np.ndarray) -> list:
        return []


class SlowEmptyDetector:
    def detect(self, _frame: np.ndarray) -> list:
        time.sleep(0.14)
        return []


def _write_video(path: Path) -> None:
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (160, 90)
    )
    assert writer.isOpened()
    for index in range(6):
        frame = np.full((90, 160, 3), 25 + index * 10, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def test_live_processor_streams_causal_frames_and_writes_replay_artifacts(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    model = tmp_path / "model.pt"
    output = tmp_path / "run"
    _write_video(source)
    model.write_bytes(b"fake-model-for-hash-only")
    processor = LiveCommonPathProcessor(
        source=source,
        config_path=Path("configs/shibuya.yaml"),
        model_path=model,
        output=output,
        run_id="unit-live",
        session_id="session-1",
        stream_epoch="epoch-1",
        duration_seconds=0.6,
        preview_fps=10,
        detector=EmptyDetector(),
        device_name="fake-cpu-no-inference",
    )

    packets = list(processor.frames())

    assert packets
    assert [metadata["frame_id"] for metadata, _ in packets] == list(range(len(packets)))
    assert all(metadata["evidence_until_s"] <= metadata["media_time_s"] for metadata, _ in packets)
    assert all(jpeg.startswith(b"\xff\xd8") for _, jpeg in packets)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    assert manifest["detector_calls"] == 6
    assert manifest["device"] == "fake-cpu-no-inference"
    assert (output / "tracking_cache.jsonl").is_file()
    assert (output / "path_timeline.jsonl").is_file()
    assert (output / "preview.mp4").stat().st_size > 0
    assert (output / "metrics.json").is_file()


def test_live_processor_marks_an_explicit_stop(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    model = tmp_path / "model.pt"
    _write_video(source)
    model.write_bytes(b"fake-model-for-hash-only")
    stop_event = threading.Event()
    stop_event.set()
    processor = LiveCommonPathProcessor(
        source=source,
        config_path=Path("configs/shibuya.yaml"),
        model_path=model,
        output=tmp_path / "stopped-run",
        run_id="unit-stopped",
        session_id="session-2",
        stream_epoch="epoch-2",
        duration_seconds=0.6,
        preview_fps=10,
        detector=EmptyDetector(),
        device_name="fake-cpu-no-inference",
        stop_event=stop_event,
    )

    assert list(processor.frames()) == []
    assert processor.final_manifest is not None
    assert processor.final_manifest["status"] == "stopped"
    assert processor.final_manifest["frames_processed"] == 0


def test_realtime_pts_drops_stale_input_frames_and_keeps_video_timeline(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    model = tmp_path / "model.pt"
    output = tmp_path / "realtime-run"
    _write_video(source)
    model.write_bytes(b"fake-model-for-hash-only")
    processor = LiveCommonPathProcessor(
        source=source,
        config_path=Path("configs/shibuya.yaml"),
        model_path=model,
        output=output,
        run_id="unit-realtime",
        session_id="session-3",
        stream_epoch="epoch-3",
        duration_seconds=0.6,
        preview_fps=10,
        detector=SlowEmptyDetector(),
        device_name="fake-cpu-no-inference",
        processing_mode="realtime_pts",
    )

    packets = list(processor.frames())
    ids = [metadata["frame_id"] for metadata, _ in packets]
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    capture = cv2.VideoCapture(str(output / "preview.mp4"))
    written_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()

    assert ids == sorted(ids)
    assert len(ids) < 6
    assert manifest["processing_mode"] == "realtime_pts"
    assert manifest["dropped_input_frames"] > 0
    assert written_frames == 6
