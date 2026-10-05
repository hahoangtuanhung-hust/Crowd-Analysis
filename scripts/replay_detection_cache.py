"""Replay a detector cache through ByteTrack without rerunning inference.

The cache written by ``common_path_clip.py`` contains detector boxes before the
tracker as well as its historical tracker output.  This command deliberately
uses only ``detections`` and source frame/time metadata, so tracker variants
are compared against identical detector evidence.  It reports operational and
continuity proxies only; it does not claim MOT accuracy without ground truth.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml

# Keep the documented ``python scripts/replay_detection_cache.py`` entry point
# usable as well as ``python -m scripts.replay_detection_cache``.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.core.config import AppConfig, load_config
from backend.app.schemas import Detection
from backend.app.tracking import ByteTrackTracker


def _parse_frame_shape(value: str) -> tuple[int, int]:
    try:
        width_text, height_text = value.lower().split("x", maxsplit=1)
        width, height = int(width_text), int(height_text)
    except (ValueError, AttributeError) as exc:
        raise argparse.ArgumentTypeError("frame shape must be WIDTHxHEIGHT") from exc
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("frame shape values must be positive")
    return width, height


def _frame_shape(row: dict[str, Any], fallback: tuple[int, int] | None) -> tuple[int, int]:
    scheduler = row.get("scheduler_decision")
    candidates = (
        row.get("source_frame_shape"),
        scheduler.get("source_frame_shape") if isinstance(scheduler, dict) else None,
    )
    for shape in candidates:
        if isinstance(shape, (list, tuple)) and len(shape) >= 2:
            height, width = int(shape[0]), int(shape[1])
            if width > 0 and height > 0:
                return width, height
    if fallback is not None:
        return fallback
    raise ValueError(
        "Detection cache row has no source_frame_shape; pass --frame-shape WIDTHxHEIGHT"
    )


def _config_text(config: AppConfig) -> str:
    return yaml.safe_dump(config.model_dump(mode="json"), sort_keys=True)


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            text=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _percentile(values: list[float], ratio: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * ratio))]


def replay(
    cache: Path,
    config_path: Path,
    output_dir: Path,
    *,
    max_frames: int | None = None,
    fallback_shape: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """Replay detector rows and write immutable tracker-variant artifacts."""
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_dir}")
    if not cache.is_file():
        raise FileNotFoundError(cache)
    if max_frames is not None and max_frames <= 0:
        raise ValueError("max_frames must be positive when provided")

    meta_path = cache.with_suffix(".meta.json")
    input_meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    config = load_config(config_path)
    config_text = _config_text(config)
    tracker = ByteTrackTracker(config.tracker)
    output_dir.mkdir(parents=True)

    frame_rows: list[dict[str, Any]] = []
    tracking_ms_values: list[float] = []
    observed_track_frames = 0
    predicted_track_frames = 0
    unique_track_ids: set[int] = set()
    previous_observed_ids: set[int] = set()
    observed_id_starts = 0
    observed_id_disappearances = 0
    first_frame_id: int | None = None
    last_frame_id: int | None = None
    first_timestamp_s: float | None = None
    last_timestamp_s: float | None = None
    last_seen_frame = -1
    last_seen_timestamp = float("-inf")
    diagnostic_count = 0

    with (
        cache.open("r", encoding="utf-8") as source,
        (output_dir / "tracking_cache.jsonl").open("w", encoding="utf-8") as track_rows,
        (output_dir / "association_diagnostics.jsonl").open("w", encoding="utf-8") as diagnostics,
    ):
        for line_number, line in enumerate(source, start=1):
            if max_frames is not None and len(frame_rows) >= max_frames:
                break
            if not line.strip():
                continue
            row = json.loads(line)
            frame_id = int(row["frame_id"])
            timestamp_value = row.get("event_time_s", row.get("source_timestamp"))
            if timestamp_value is None:
                raise ValueError(f"Cache row requires event_time_s (line {line_number})")
            timestamp_s = float(timestamp_value)
            if frame_id <= last_seen_frame:
                raise ValueError(f"Cache frame IDs must increase (line {line_number})")
            if timestamp_s < last_seen_timestamp:
                raise ValueError(f"Cache timestamps must not move backwards (line {line_number})")
            last_seen_frame, last_seen_timestamp = frame_id, timestamp_s
            width, height = _frame_shape(row, fallback_shape)
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            detections = [Detection(**item) for item in row.get("detections", [])]

            started = time.perf_counter()
            tracks = tracker.update(
                detections,
                frame,
                frame_id=frame_id,
                source_timestamp=timestamp_s,
            )
            tracking_ms = (time.perf_counter() - started) * 1000.0
            tracking_ms_values.append(tracking_ms)
            diagnostic_events = tracker.drain_association_diagnostics()
            for event in diagnostic_events:
                diagnostics.write(json.dumps(event, separators=(",", ":")) + "\n")
            diagnostic_count += len(diagnostic_events)

            observed = [track for track in tracks if track.observed]
            observed_ids = {track.track_id for track in observed}
            observed_track_frames += len(observed)
            predicted_track_frames += len(tracks) - len(observed)
            unique_track_ids.update(observed_ids)
            new_observed_ids = observed_ids - previous_observed_ids
            observed_id_starts += len(new_observed_ids)
            observed_id_disappearances += len(previous_observed_ids - observed_ids)
            previous_observed_ids = observed_ids
            frame_rows.append(
                {
                    "frame_id": frame_id,
                    "event_time_s": timestamp_s,
                    "detections": len(detections),
                    "high_detections": sum(
                        item.confidence >= config.tracker.track_high_thresh
                        for item in detections
                    ),
                    "low_detections": sum(
                        config.tracker.track_low_thresh <= item.confidence
                        < config.tracker.track_high_thresh
                        for item in detections
                    ),
                    "observed_tracks": len(observed),
                    "predicted_tracks": len(tracks) - len(observed),
                    "new_observed_ids": len(new_observed_ids),
                    "tracking_ms": round(tracking_ms, 6),
                }
            )
            track_rows.write(
                json.dumps(
                    {
                        "frame_id": frame_id,
                        "event_time_s": timestamp_s,
                        "detections": [asdict(item) for item in detections],
                        "tracks": [asdict(item) for item in tracks],
                    },
                    separators=(",", ":"),
                )
                + "\n"
            )
            first_frame_id = frame_id if first_frame_id is None else first_frame_id
            first_timestamp_s = timestamp_s if first_timestamp_s is None else first_timestamp_s
            last_frame_id = frame_id
            last_timestamp_s = timestamp_s

    if not frame_rows:
        raise ValueError("Detection cache did not contain any rows")
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(frame_rows[0]))
        writer.writeheader()
        writer.writerows(frame_rows)

    total_tracking_ms = sum(tracking_ms_values)
    summary = {
        "quality_status": "QUALITY_REVIEW_PENDING",
        "metric_scope": "operational and continuity proxies only; no ground-truth IDF1/HOTA",
        "input_detection_cache": str(cache),
        "frames": len(frame_rows),
        "first_frame_id": first_frame_id,
        "last_frame_id": last_frame_id,
        "first_event_time_s": first_timestamp_s,
        "last_event_time_s": last_timestamp_s,
        "detections": sum(int(row["detections"]) for row in frame_rows),
        "high_detections": sum(int(row["high_detections"]) for row in frame_rows),
        "low_detections": sum(int(row["low_detections"]) for row in frame_rows),
        "observed_track_frames": observed_track_frames,
        "predicted_track_frames": predicted_track_frames,
        "unique_observed_track_ids": len(unique_track_ids),
        "observed_id_starts_proxy": observed_id_starts,
        "observed_id_disappearances_proxy": observed_id_disappearances,
        "tracking_ms_total": round(total_tracking_ms, 6),
        "tracking_ms_p50": round(_percentile(tracking_ms_values, 0.5) or 0.0, 6),
        "tracking_ms_p95": round(_percentile(tracking_ms_values, 0.95) or 0.0, 6),
        "tracking_fps": round(1000.0 * len(frame_rows) / total_tracking_ms, 3)
        if total_tracking_ms > 0.0
        else None,
        "association_diagnostic_events": diagnostic_count,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "config_resolved.yaml").write_text(config_text, encoding="utf-8")
    provenance = {
        "schema": "tracking-cache-replay-v1",
        "source_detection_cache": str(cache),
        "source_cache_meta": input_meta,
        "config_path": str(config_path),
        "config_sha256": hashlib.sha256(config_text.encode("utf-8")).hexdigest(),
        "git_commit": _git_commit(),
        "versions": {
            "numpy": _version("numpy"),
            "ultralytics": _version("ultralytics"),
            "pydantic": _version("pydantic"),
        },
        "limitations": [
            "Detector boxes are fixed by the input cache; a lower detector floor requires a new detector run.",
            "No labelled MOT data was supplied, so this artifact cannot establish IDF1, HOTA, recall, or ID switches.",
            "This replay has no source video renderer; visual before/after validation requires the hashed input video.",
        ],
    }
    (output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True, help="Detector cache JSONL")
    parser.add_argument("--config", type=Path, required=True, help="Tracker config YAML")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--frame-shape", type=_parse_frame_shape)
    args = parser.parse_args()
    print(
        json.dumps(
            replay(
                args.cache,
                args.config,
                args.output_dir,
                max_frames=args.max_frames,
                fallback_shape=args.frame_shape,
            ),
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
