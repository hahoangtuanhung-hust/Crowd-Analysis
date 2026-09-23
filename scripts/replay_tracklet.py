"""Replay observed tracker rows from a hashed Modal cache without inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import cv2

from backend.app.analytics.directional_grid import GridTrackPoint
from backend.app.analytics.spatial import SpatialTransformer
from backend.app.analytics.tracklet_aggregation import TrackletAggregationEngine
from backend.app.core.config import load_config
from backend.app.schemas import TrackedObject
from backend.app.video.renderer import FrameRenderer, OverlayOptions


def run(cache: Path, source: Path, output: Path, config_path: Path, max_paths: int = 3) -> dict:
    if output.exists():
        raise FileExistsError(output)
    meta = json.loads(cache.with_suffix(".meta.json").read_text(encoding="utf-8"))
    # Batch caches created before the schema field was introduced are still
    # content-addressed and safe to replay when their required hashes exist.
    schema = meta.get("schema", "detections-tracklets-v1")
    if schema not in {"detections-tracklets-live-realtime-v1", "detections-tracklets-live-v1", "detections-tracklets-v1"}:
        raise ValueError("Unsupported tracking cache schema")
    hasher = hashlib.sha256()
    with source.open("rb") as video:
        for chunk in iter(lambda: video.read(1024 * 1024), b""):
            hasher.update(chunk)
    if hasher.hexdigest() != meta["source_hash"]:
        raise ValueError("Source video SHA-256 does not match cache")
    config = load_config(config_path)
    cap = cv2.VideoCapture(str(source))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if not cap.isOpened() or fps <= 0 or width <= 0 or height <= 0:
        raise ValueError("Invalid source video")
    output.mkdir(parents=True)
    transformer = SpatialTransformer.pixel(width, height)
    engine = TrackletAggregationEngine(config.analytics.common_path.tracklet_aggregation,
                                       transformer, stream_epoch="cache-replay", max_paths=max_paths)
    renderer = FrameRenderer(config.visualization)
    overlay = OverlayOptions(tracking=True, points=True, trajectory=False, trajectory_tails=False,
                             track_ids=False, candidate_paths=False, active_paths=True,
                             debug_metrics=False)
    writer = cv2.VideoWriter(str(output / "preview.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                             min(fps, 10), (width, height))
    if not writer.isOpened():
        raise RuntimeError("Cannot open preview writer")
    last_frame = -1
    last_time = -1.
    count = 0
    active_frames = 0
    last_preview = -1.
    try:
        with cache.open(encoding="utf-8") as rows, (output / "path_timeline.jsonl").open("w", encoding="utf-8") as timeline:
            for line in rows:
                row = json.loads(line)
                frame_id, timestamp = int(row["frame_id"]), float(row["event_time_s"])
                if frame_id <= last_frame or timestamp < last_time or "tracks" not in row:
                    raise ValueError("Cache needs ordered frame/time and observed tracks")
                last_frame, last_time = frame_id, timestamp
                tracks = [TrackedObject(**track) for track in row["tracks"]]
                points = [GridTrackPoint("cam01", "cache-replay", track.track_id, 0, frame_id,
                                         timestamp, *track.bottom_center, observed=track.observed) for track in tracks]
                snapshot = engine.update(points, timestamp)
                active_frames += bool(snapshot.paths)
                timeline.write(json.dumps({"camera_id": "cam01", "stream_epoch": "cache-replay",
                                           "source_frame_id": frame_id, "media_time_s": timestamp,
                                           "applied_max_paths": engine.max_paths,
                                           "paths": [asdict(path) for path in snapshot.paths]}) + "\n")
                count += 1
                if timestamp - last_preview < .1:
                    continue
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
                ok, frame = cap.read()
                if not ok:
                    raise ValueError(f"Cannot decode frame {frame_id}")
                current = [SimpleNamespace(track_id=track.track_id, x=track.bottom_center[0],
                                           y=track.bottom_center[1]) for track in tracks if track.observed]
                rendered = renderer.render_point_only_frame(frame, current, snapshot, (), transformer,
                                                             overlay, people_count=len(current),
                                                             processing_fps=0, latency_ms=0, timestamp=timestamp)
                writer.write(rendered)
                if snapshot.paths:
                    cv2.imwrite(str(output / "ui_common_path.jpg"), rendered)
                last_preview = timestamp
    finally:
        cap.release()
        writer.release()
    latencies = list(engine.compute_ms)
    metrics = {"source": str(source), "cache": str(cache), "inference_calls": 0,
               "processed_frames": count, "active_frames": active_frames,
               "analytics_p50_ms": round(statistics.median(latencies), 3) if latencies else None,
               "analytics_p95_ms": round(sorted(latencies)[int(.95 * (len(latencies)-1))], 3) if latencies else None,
               "candidate_count": engine.candidate_count, "active_count": len(engine.snapshot().paths),
               "rejections": dict(engine.rejections)}
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/shibuya.yaml"))
    parser.add_argument("--max-paths", type=int, default=3)
    args = parser.parse_args()
    print(json.dumps(run(args.cache, args.input, args.output_dir, args.config, args.max_paths), indent=2))


if __name__ == "__main__":
    main()
