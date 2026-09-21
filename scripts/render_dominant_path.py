"""Render a causal dominant-direction Common Path from an immutable tracking cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import cv2
import yaml

from backend.app.analytics.directional_grid import DirectionalGridEngine, GridTrackPoint
from backend.app.analytics.spatial import SpatialTransformer
from backend.app.core.config import load_config
from backend.app.schemas import TrackedObject
from backend.app.video.renderer import FrameRenderer, OverlayOptions


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def run(
    source: Path,
    cache: Path,
    config_path: Path,
    output: Path,
    run_id: str,
    *,
    execution_host: str = "local_cpu",
    gpu_device: str | None = None,
) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite run: {output}")
    meta_path = cache.with_suffix(".meta.json")
    if not source.is_file() or not cache.is_file() or not meta_path.is_file():
        raise FileNotFoundError("Source, tracking cache, or cache metadata is missing")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    source_hash = digest(source)
    if meta.get("source_hash") != source_hash:
        raise ValueError("Tracking cache source hash does not match the video")

    config = load_config(config_path)
    grid_config = config.analytics.directional_grid
    if grid_config.display_policy != "dominant_direction":
        raise ValueError("Directional grid display_policy must be dominant_direction")

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError(f"Cannot decode source video: {source}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    expected_frames = int(meta.get("frame_count", 0))
    if fps <= 0 or width <= 0 or height <= 0 or expected_frames <= 0:
        raise ValueError("Invalid source or cache media metadata")

    output.mkdir(parents=True)
    video_path = output / "dominant_common_path.mp4"
    writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError("Cannot open dominant Common Path MP4 writer")

    transformer = SpatialTransformer.pixel(width, height)
    engine = DirectionalGridEngine(
        grid_config,
        transformer,
        stream_epoch=run_id,
        zones=config.analytics.zones,
        run_id=run_id,
        variant="dominant_direction",
    )
    renderer = FrameRenderer(config.visualization)
    overlay = OverlayOptions.from_visualization(config.visualization)
    timeline_path = output / "common_path_timeline.jsonl"
    started = time.perf_counter()
    frames = active_frames = 0
    first_active_s: float | None = None
    last_active = False
    active_intervals: list[dict[str, float]] = []
    interval_start: float | None = None
    last_time = 0.0
    path_ids: set[str] = set()
    max_short_support = 0
    last_timeline_version = -1

    try:
        with cache.open("r", encoding="utf-8") as cache_reader, timeline_path.open(
            "w", encoding="utf-8"
        ) as timeline:
            for frame_id in range(expected_frames):
                ok, frame = capture.read()
                line = cache_reader.readline()
                if not ok or not line:
                    raise ValueError(f"Video/cache ended early at frame {frame_id}")
                entry = json.loads(line)
                if int(entry.get("frame_id", -1)) != frame_id:
                    raise ValueError(f"Cache frame mismatch at frame {frame_id}")
                timestamp = float(entry["event_time_s"])
                tracks = [TrackedObject(**item) for item in entry["tracks"]]
                observed = [track for track in tracks if track.observed]
                points = [
                    GridTrackPoint(
                        "cam01", run_id, track.track_id, 0, frame_id, timestamp,
                        *track.bottom_center, observed=True,
                    )
                    for track in observed
                ]
                snapshot = engine.update(points, timestamp)
                rendered = renderer.render_point_only_frame(
                    frame,
                    [
                        SimpleNamespace(
                            track_id=track.track_id,
                            x=track.bottom_center[0],
                            y=track.bottom_center[1],
                        )
                        for track in observed
                    ],
                    snapshot,
                    tuple(config.analytics.zones),
                    transformer,
                    overlay,
                    people_count=len(observed),
                    processing_fps=0.0,
                    latency_ms=0.0,
                    timestamp=timestamp,
                )
                writer.write(rendered)
                frames += 1
                last_time = timestamp

                active = next((path for path in snapshot.paths if path.state == "active"), None)
                if active is not None:
                    active_frames += 1
                    first_active_s = timestamp if first_active_s is None else first_active_s
                    path_ids.add(active.path_id)
                    max_short_support = max(max_short_support, active.unique_tracks_short)
                    if not last_active:
                        interval_start = timestamp
                elif last_active and interval_start is not None:
                    active_intervals.append({"start_s": interval_start, "end_s": timestamp})
                    interval_start = None
                last_active = active is not None

                if snapshot.version != last_timeline_version:
                    timeline.write(json.dumps({
                        "frame_id": frame_id,
                        "media_time_s": timestamp,
                        "display_policy": grid_config.display_policy,
                        "path": asdict(active) if active is not None else None,
                    }) + "\n")
                    last_timeline_version = snapshot.version

            if cache_reader.readline():
                raise ValueError("Tracking cache contains more rows than declared")
    finally:
        capture.release()
        writer.release()

    if last_active and interval_start is not None:
        active_intervals.append({"start_s": interval_start, "end_s": last_time})
    if frames != expected_frames:
        raise ValueError(f"Rendered {frames} frames; expected {expected_frames}")

    with (output / "path_events.jsonl").open("w", encoding="utf-8") as sink:
        for event in engine.events:
            sink.write(json.dumps(event) + "\n")
    with (output / "candidate_evaluations.jsonl").open("w", encoding="utf-8") as sink:
        for evaluation in engine.candidate_evaluations:
            sink.write(json.dumps(evaluation) + "\n")
    (output / "config_resolved.yaml").write_text(
        yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False), encoding="utf-8"
    )

    manifest: dict[str, object] = {
        "run_id": run_id,
        "status": "success",
        "run_type": "remote_cache_replay" if execution_host != "local_cpu" else "local_cache_replay",
        "execution_host": execution_host,
        "gpu_device": gpu_device,
        "stage_devices": {
            "cache_read": "cpu",
            "analytics": "cpu",
            "rendering": "cpu_opencv_mp4v",
            "modal_container_gpu": gpu_device,
        },
        "detector_calls": 0,
        "inference_executed": False,
        "source": source.name,
        "source_hash": source_hash,
        "cache": cache.name,
        "cache_key": meta.get("cache_key"),
        "cache_schema": meta.get("schema"),
        "display_policy": grid_config.display_policy,
        "dominant_window_seconds": grid_config.short_window_seconds,
        "frames": frames,
        "fps": fps,
        "duration_seconds": last_time,
        "active_frames": active_frames,
        "active_frame_ratio": round(active_frames / frames, 6),
        "first_active_s": first_active_s,
        "active_intervals": active_intervals,
        "displayed_path_ids": sorted(path_ids),
        "path_switches": sum(event["event"] == "switch" for event in engine.events),
        "max_short_support": max_short_support,
        "runtime_seconds": round(time.perf_counter() - started, 3),
        "artifacts": [
            "candidate_evaluations.jsonl",
            "common_path_timeline.jsonl",
            "config_resolved.yaml",
            "dominant_common_path.mp4",
            "manifest.json",
            "path_events.jsonl",
        ],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/shibuya.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    print(json.dumps(run(
        args.input, args.cache, args.config, args.output, args.run_id
    ), indent=2))


if __name__ == "__main__":
    main()
