from __future__ import annotations

import argparse
import csv
import platform
import statistics
import time
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import psutil
from ultralytics import YOLO
from ultralytics.trackers.track import TRACKER_MAP
from ultralytics.utils import YAML, IterableSimpleNamespace

from backend.app.schemas import Detection
from backend.app.tracking.bytetrack_tracker import _DetectionResults

TRACKER_CONFIG_ROOT = Path(__import__("ultralytics").__file__).parent / "cfg" / "trackers"
TRACKERS = ("bytetrack", "botsort", "ocsort", "deepocsort", "tracktrack")
CSV_FIELDS = (
    "measured_at_utc",
    "hardware",
    "video",
    "detector",
    "tracker",
    "backend",
    "precision",
    "input_resolution",
    "inference_interval",
    "frames_total",
    "inference_calls",
    "processing_fps",
    "inference_ms_p50",
    "inference_ms_p95",
    "tracking_ms_p50",
    "tracking_ms_p95",
    "id_continuity_pct",
    "tracked_detection_coverage_pct",
    "ram_peak_mb",
    "vram_mb",
    "status",
    "notes",
)


@dataclass(slots=True)
class DetectionRun:
    detections: list[list[Detection]]
    elapsed_ms: list[float]
    peak_ram_mb: float


@dataclass(slots=True)
class TrackerRun:
    elapsed_ms: list[float]
    continuity_pct: float
    coverage_pct: float
    peak_ram_mb: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark crowd-analysis detector and trackers")
    parser.add_argument("--video", default="data/videos/example-people.mp4")
    parser.add_argument("--pytorch-model", default="yolo26n.pt")
    parser.add_argument("--onnx-model", default="yolo26n.onnx")
    parser.add_argument("--output", default="benchmarks/benchmark_results.csv")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--max-frames", type=int, default=76)
    parser.add_argument("--cycles", type=int, default=3)
    return parser.parse_args()


def load_frames(path: Path, max_frames: int, cycles: int) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open benchmark video: {path}")
    source: list[np.ndarray] = []
    while len(source) < max_frames:
        ok, frame = capture.read()
        if not ok:
            break
        source.append(frame)
    capture.release()
    if not source:
        raise RuntimeError(f"Benchmark video has no decodable frames: {path}")
    sequence = source + list(reversed(source[1:-1])) if len(source) > 2 else source
    return (sequence * max(1, cycles))[:max_frames]


def run_detector(
    model_path: Path, frames: list[np.ndarray], imgsz: int, interval: int
) -> DetectionRun:
    model = YOLO(str(model_path))
    model.predict(
        frames[0], imgsz=imgsz, conf=0.35, iou=0.7, classes=[0], device="cpu", verbose=False
    )
    process = psutil.Process()
    peak_ram_mb = process.memory_info().rss / 1024**2
    detections: list[list[Detection]] = [[] for _ in frames]
    elapsed: list[float] = []
    for index in range(0, len(frames), interval):
        started = time.perf_counter()
        result = model.predict(
            frames[index],
            imgsz=imgsz,
            conf=0.35,
            iou=0.7,
            classes=[0],
            device="cpu",
            verbose=False,
        )[0]
        elapsed.append((time.perf_counter() - started) * 1000.0)
        boxes = result.boxes
        if boxes is not None and len(boxes):
            xyxy = boxes.xyxy.cpu().numpy()
            confidence = boxes.conf.cpu().numpy()
            class_ids = boxes.cls.cpu().numpy()
            detections[index] = [
                Detection(
                    x1=float(box[0]),
                    y1=float(box[1]),
                    x2=float(box[2]),
                    y2=float(box[3]),
                    confidence=float(score),
                    class_id=int(class_id),
                )
                for box, score, class_id in zip(xyxy, confidence, class_ids, strict=True)
            ]
        peak_ram_mb = max(peak_ram_mb, process.memory_info().rss / 1024**2)
    return DetectionRun(detections, elapsed, peak_ram_mb)


def create_tracker(name: str) -> Any:
    config_path = TRACKER_CONFIG_ROOT / f"{name}.yaml"
    args = IterableSimpleNamespace(**YAML.load(config_path))
    args.device = "cpu"
    if hasattr(args, "with_reid"):
        args.with_reid = False
    return TRACKER_MAP[name](args=args)


def run_tracker(
    name: str, frames: list[np.ndarray], detections: list[list[Detection]]
) -> TrackerRun:
    tracker = create_tracker(name)
    elapsed: list[float] = []
    outputs: list[np.ndarray] = []
    process = psutil.Process()
    peak_ram_mb = process.memory_info().rss / 1024**2
    tracked_count = 0
    detection_count = 0
    for frame, frame_detections in zip(frames, detections, strict=True):
        if not frame_detections:
            continue
        results = _DetectionResults(
            xyxy=np.asarray([item.xyxy for item in frame_detections], dtype=np.float32),
            confidence=np.asarray([item.confidence for item in frame_detections], dtype=np.float32),
            class_ids=np.asarray([item.class_id for item in frame_detections], dtype=np.float32),
        )
        started = time.perf_counter()
        tracked = tracker.update(results, img=frame)
        elapsed.append((time.perf_counter() - started) * 1000.0)
        tracked = np.asarray(tracked, dtype=np.float32).reshape(-1, 8)
        outputs.append(tracked)
        tracked_count += len(tracked)
        detection_count += len(frame_detections)
        peak_ram_mb = max(peak_ram_mb, process.memory_info().rss / 1024**2)
    continuity = id_continuity(outputs)
    coverage = 100.0 * tracked_count / detection_count if detection_count else 0.0
    return TrackerRun(elapsed, continuity, coverage, peak_ram_mb)


def id_continuity(outputs: list[np.ndarray]) -> float:
    stable = 0
    comparable = 0
    for previous, current in pairwise(outputs):
        if len(previous) == 0 or len(current) == 0:
            continue
        candidates: list[tuple[float, int, int]] = []
        for old_index, old in enumerate(previous):
            for new_index, new in enumerate(current):
                overlap = iou(old[:4], new[:4])
                if overlap >= 0.3:
                    candidates.append((overlap, old_index, new_index))
        used_old: set[int] = set()
        used_new: set[int] = set()
        for _, old_index, new_index in sorted(candidates, reverse=True):
            if old_index in used_old or new_index in used_new:
                continue
            used_old.add(old_index)
            used_new.add(new_index)
            comparable += 1
            stable += int(int(previous[old_index, 4]) == int(current[new_index, 4]))
    return 100.0 * stable / comparable if comparable else 0.0


def iou(first: Iterable[float], second: Iterable[float]) -> float:
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    union = max(0.0, (ax2 - ax1) * (ay2 - ay1)) + max(0.0, (bx2 - bx1) * (by2 - by1)) - intersection
    return intersection / union if union else 0.0


def percentile(values: list[float], percentile_value: float) -> float:
    return float(np.percentile(values, percentile_value)) if values else 0.0


def measured_row(
    *,
    now: str,
    hardware: str,
    video: str,
    backend: str,
    tracker_name: str,
    interval: int,
    imgsz: int,
    frame_count: int,
    detector_run: DetectionRun,
    tracker_run: TrackerRun,
) -> dict[str, object]:
    compute_seconds = (sum(detector_run.elapsed_ms) + sum(tracker_run.elapsed_ms)) / 1000.0
    return {
        "measured_at_utc": now,
        "hardware": hardware,
        "video": video,
        "detector": "YOLO26n",
        "tracker": tracker_name,
        "backend": backend,
        "precision": "FP32",
        "input_resolution": imgsz,
        "inference_interval": interval,
        "frames_total": frame_count,
        "inference_calls": len(detector_run.elapsed_ms),
        "processing_fps": round(frame_count / compute_seconds, 3),
        "inference_ms_p50": round(statistics.median(detector_run.elapsed_ms), 3),
        "inference_ms_p95": round(percentile(detector_run.elapsed_ms, 95), 3),
        "tracking_ms_p50": round(statistics.median(tracker_run.elapsed_ms), 3),
        "tracking_ms_p95": round(percentile(tracker_run.elapsed_ms, 95), 3),
        "id_continuity_pct": round(tracker_run.continuity_pct, 2),
        "tracked_detection_coverage_pct": round(tracker_run.coverage_pct, 2),
        "ram_peak_mb": round(max(detector_run.peak_ram_mb, tracker_run.peak_ram_mb), 1),
        "vram_mb": "",
        "status": "measured",
        "notes": "CPU-only compute timing; decode/render/encode excluded",
    }


def unsupported_row(
    now: str, hardware: str, video: str, backend: str, precision: str, note: str
) -> dict[str, object]:
    row = {field: "" for field in CSV_FIELDS}
    row.update(
        measured_at_utc=now,
        hardware=hardware,
        video=video,
        detector="YOLO26n",
        tracker="ByteTrack",
        backend=backend,
        precision=precision,
        input_resolution=640,
        inference_interval=1,
        status="unsupported",
        notes=note,
    )
    return row


def main() -> None:
    args = parse_args()
    video_path = Path(args.video)
    frames = load_frames(video_path, args.max_frames, args.cycles)
    now = datetime.now(UTC).isoformat()
    hardware = f"{platform.processor() or platform.machine()} | {psutil.cpu_count(logical=False)}C/{psutil.cpu_count()}T | {round(psutil.virtual_memory().total / 1024**3, 1)}GB"
    rows: list[dict[str, object]] = []
    cached_pytorch: DetectionRun | None = None

    for interval in (1, 2, 3):
        print(f"Benchmarking PyTorch FP32, interval={interval} ...", flush=True)
        detector_run = run_detector(Path(args.pytorch_model), frames, args.imgsz, interval)
        tracker_run = run_tracker("bytetrack", frames, detector_run.detections)
        rows.append(
            measured_row(
                now=now,
                hardware=hardware,
                video=video_path.name,
                backend="PyTorch",
                tracker_name="ByteTrack",
                interval=interval,
                imgsz=args.imgsz,
                frame_count=len(frames),
                detector_run=detector_run,
                tracker_run=tracker_run,
            )
        )
        if interval == 1:
            cached_pytorch = detector_run

    assert cached_pytorch is not None
    for tracker_name in TRACKERS[1:]:
        print(f"Benchmarking tracker={tracker_name} ...", flush=True)
        tracker_run = run_tracker(tracker_name, frames, cached_pytorch.detections)
        rows.append(
            measured_row(
                now=now,
                hardware=hardware,
                video=video_path.name,
                backend="PyTorch",
                tracker_name={
                    "botsort": "BoT-SORT",
                    "ocsort": "OC-SORT",
                    "deepocsort": "Deep OC-SORT",
                    "tracktrack": "TrackTrack",
                }[tracker_name],
                interval=1,
                imgsz=args.imgsz,
                frame_count=len(frames),
                detector_run=cached_pytorch,
                tracker_run=tracker_run,
            )
        )

    onnx_path = Path(args.onnx_model)
    if onnx_path.exists():
        print("Benchmarking ONNX Runtime FP32 ...", flush=True)
        detector_run = run_detector(onnx_path, frames, args.imgsz, 1)
        tracker_run = run_tracker("bytetrack", frames, detector_run.detections)
        rows.append(
            measured_row(
                now=now,
                hardware=hardware,
                video=video_path.name,
                backend="ONNX Runtime",
                tracker_name="ByteTrack",
                interval=1,
                imgsz=args.imgsz,
                frame_count=len(frames),
                detector_run=detector_run,
                tracker_run=tracker_run,
            )
        )
    else:
        rows.append(
            unsupported_row(
                now,
                hardware,
                video_path.name,
                "ONNX Runtime",
                "FP32",
                "ONNX model was not exported",
            )
        )

    rows.extend(
        [
            unsupported_row(
                now, hardware, video_path.name, "PyTorch", "FP16", "No CUDA-capable GPU detected"
            ),
            unsupported_row(
                now,
                hardware,
                video_path.name,
                "TensorRT",
                "FP16",
                "No NVIDIA GPU or TensorRT runtime detected",
            ),
        ]
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {output_path}", flush=True)


if __name__ == "__main__":
    main()
