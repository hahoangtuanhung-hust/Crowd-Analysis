"""Bounded GPU clip run or CPU analytics replay. Never infer on the local replay path."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import psutil
import yaml

from backend.app.analytics.common_path import CommonPathAnalyzer
from backend.app.analytics.directional_grid import DirectionalGridEngine, GridTrackPoint
from backend.app.analytics.spatial import SpatialTransformer
from backend.app.core.config import load_config
from backend.app.schemas import Detection, TrackedObject
from backend.app.video.renderer import FrameRenderer, OverlayOptions


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def cache_key(input_hash: str, model_hash: str, config: object, start: float,
              duration: float) -> str:
    data = dict(schema="detections-tracklets-v1", input_hash=input_hash,
                model_hash=model_hash, start=start, duration=duration,
                detector=config.detector.model_dump(exclude={"device", "model"}),
                tracker=config.tracker.model_dump())
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def run(input_path: Path, output_dir: Path, *, config_path: Path, model_path: Path | None,
        start_seconds: float, duration_seconds: float, run_id: str, engine: str,
        replay_cache: Path | None = None, input_hash: str | None = None,
        model_hash: str | None = None, rebuild_tracks: bool = False) -> dict:
    if not (0 <= start_seconds and 0 < duration_seconds <= 90):
        raise ValueError("Clip must be positive, <=90 seconds, and start >=0")
    if engine not in ("legacy", "directional_grid", "shadow"):
        raise ValueError("Invalid engine")
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite run: {output_dir}")
    output_dir.mkdir(parents=True)
    config = load_config(config_path)
    transformer: SpatialTransformer
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot decode input: {input_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0 or width <= 0 or height <= 0:
        raise ValueError("Source FPS/resolution unavailable")
    transformer = SpatialTransformer.pixel(width, height)
    grid = DirectionalGridEngine(config.analytics.directional_grid, transformer,
                                 stream_epoch=run_id, zones=config.analytics.zones)
    legacy = CommonPathAnalyzer(config.analytics, transformer)
    selected = config.analytics.common_path.shadow_display if engine == "shadow" else engine
    renderer = FrameRenderer(config.visualization)
    overlay = OverlayOptions.from_visualization(config.visualization)
    input_hash = input_hash or digest(input_path)
    model_hash = model_hash or (digest(model_path) if model_path else "unknown")
    key = cache_key(input_hash, model_hash, config, start_seconds, duration_seconds)
    cache_path = output_dir / "tracking_cache.jsonl" if replay_cache is None else replay_cache
    if rebuild_tracks and replay_cache is None:
        raise ValueError("--rebuild-tracks requires --replay-cache")
    device = "none (CPU analytics replay)"
    cuda_runtime = None
    detector = tracker = None
    if replay_cache is None:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for video inference; no CPU fallback")
        torch.cuda.reset_peak_memory_stats()
        cuda_runtime = torch
        device = torch.cuda.get_device_name(0)
        from backend.app.inference import UltralyticsPersonDetector
        from backend.app.tracking import ByteTrackTracker
        detection_config = config.detector.model_copy(
            update={"device": "cuda:0", "model": str(model_path or config.detector.model)}
        )
        detector = UltralyticsPersonDetector(detection_config)
        tracker = ByteTrackTracker(config.tracker)
    else:
        meta_path = replay_cache.with_suffix(".meta.json")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta["cache_key"] != key:
            raise ValueError("Tracking cache key mismatch; inference/tracker/input/clip changed")
        if rebuild_tracks:
            from backend.app.tracking import ByteTrackTracker
            tracker = ByteTrackTracker(config.tracker)
    writer = cv2.VideoWriter(str(output_dir / "tracked_points_common_path.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError("Cannot open MP4 writer")
    capture_frames: list[np.ndarray] = []
    timings: list[dict] = []
    ram_samples_mb: list[float] = []
    frame_ids: list[int] = []
    count_tracks: set[int] = set()
    latest_snapshot = grid.snapshot()
    legacy_snapshot = legacy.snapshot()
    first_frame: np.ndarray | None = None
    last_frame: np.ndarray | None = None
    last_time = start_seconds
    start_frame = round(start_seconds * fps)
    end_frame = min(round((start_seconds + duration_seconds) * fps),
                    int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 10**9)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    cache_reader = replay_cache.open("r", encoding="utf-8") if replay_cache else None
    cache_writer = cache_path.open("w", encoding="utf-8") if replay_cache is None else None
    rebuilt_cache_path = output_dir / "tracking_cache_rebuilt.jsonl" if rebuild_tracks else None
    rebuilt_cache_writer = (rebuilt_cache_path.open("w", encoding="utf-8")
                            if rebuilt_cache_path else None)
    began = time.perf_counter()
    try:
        for frame_id in range(start_frame, end_frame):
            t0 = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                break
            decode_ms = (time.perf_counter() - t0) * 1000
            if first_frame is None:
                first_frame = frame.copy()
            raw_pts = float(cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000
            timestamp = raw_pts if raw_pts > last_time else frame_id / fps
            if timestamp < last_time:
                raise ValueError("Media timestamps moved backwards")
            last_time = timestamp
            if cache_reader:
                entry = json.loads(cache_reader.readline())
                if entry["frame_id"] != frame_id:
                    raise ValueError("Replay cache frame mismatch")
                timestamp = entry["event_time_s"]
                detections = [Detection(**d) for d in entry["detections"]]
                inference_ms = 0.
                if rebuild_tracks:
                    t1 = time.perf_counter()
                    tracks = tracker.update(detections, frame)
                    tracking_ms = (time.perf_counter() - t1) * 1000
                    rebuilt_cache_writer.write(json.dumps(dict(
                        frame_id=frame_id, event_time_s=timestamp,
                        detections=[asdict(d) for d in detections],
                        tracks=[asdict(t) for t in tracks],
                    )) + "\n")
                else:
                    tracks = [TrackedObject(**t) for t in entry["tracks"]]
                    tracking_ms = 0.
            else:
                import torch
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                detections = detector.detect(frame)
                torch.cuda.synchronize()
                inference_ms = (time.perf_counter() - t1) * 1000
                t1 = time.perf_counter()
                tracks = tracker.update(detections, frame)
                tracking_ms = (time.perf_counter() - t1) * 1000
                cache_writer.write(json.dumps(dict(frame_id=frame_id, event_time_s=timestamp,
                    detections=[asdict(d) for d in detections],
                    tracks=[asdict(t) for t in tracks])) + "\n")
            count_tracks.update(t.track_id for t in tracks)
            points = [GridTrackPoint("cam01", run_id, t.track_id, 0, frame_id,
                                     timestamp, *t.bottom_center, observed=t.observed)
                      for t in tracks]
            t2 = time.perf_counter()
            latest_snapshot = grid.update(points, timestamp)
            directional_ms = (time.perf_counter() - t2) * 1000
            legacy_ms = 0.
            if engine in ("legacy", "shadow"):
                legacy_started = time.perf_counter()
                legacy_snapshot = legacy.process_points(
                    [SimpleNamespace(camera_id="cam01", track_id=t.track_id,
                                     frame_id=frame_id, timestamp=timestamp,
                                     x=t.bottom_center[0], y=t.bottom_center[1],
                                     confidence=t.confidence, zone_id=None)
                    for t in tracks if t.observed],
                    active_track_ids=(t.track_id for t in tracks), timestamp=timestamp)
                legacy_ms = (time.perf_counter() - legacy_started) * 1000
            analytics_ms = (time.perf_counter() - t2) * 1000
            snapshot = legacy_snapshot if selected == "legacy" else latest_snapshot
            t3 = time.perf_counter()
            rendered = renderer.render_point_only_frame(
                frame, [SimpleNamespace(track_id=t.track_id, x=t.bottom_center[0],
                                        y=t.bottom_center[1])
                        for t in tracks if t.observed],
                snapshot, tuple(config.analytics.zones), transformer, overlay,
                people_count=len(tracks), processing_fps=0, latency_ms=inference_ms,
                timestamp=timestamp)
            render_ms = (time.perf_counter() - t3) * 1000
            t4 = time.perf_counter()
            writer.write(rendered)
            encode_ms = (time.perf_counter() - t4) * 1000
            if len(frame_ids) % max(1, math.ceil((end_frame - start_frame) / 8)) == 0:
                cv2.putText(rendered, f"frame={frame_id} media={timestamp:.2f}s", (15, height - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 2)
                capture_frames.append(cv2.resize(rendered, (width // 3, height // 3)))
            frame_ids.append(frame_id)
            timings.append(dict(frame_id=frame_id, event_time_s=round(timestamp, 5),
                                detections=len(detections), tracks=len(tracks), decode_ms=decode_ms,
                                inference_ms=inference_ms, tracking_ms=tracking_ms,
                                directional_analytics_ms=directional_ms,
                                legacy_analytics_ms=legacy_ms, analytics_ms=analytics_ms,
                                render_ms=render_ms, encode_ms=encode_ms))
            ram_samples_mb.append(psutil.Process().memory_info().rss / 1024**2)
            last_frame = frame.copy()
    finally:
        cap.release()
        writer.release()
        if cache_reader:
            cache_reader.close()
        if cache_writer:
            cache_writer.close()
        if rebuilt_cache_writer:
            rebuilt_cache_writer.close()
    if not frame_ids:
        raise RuntimeError("Clip has no decoded frames")
    if cache_writer is not None:
        cache_path.with_suffix(".meta.json").write_text(
            json.dumps(dict(cache_key=key, source_hash=input_hash, model_hash=model_hash,
                            start_seconds=start_seconds, duration_seconds=duration_seconds,
                            frame_count=len(frame_ids)), indent=2), encoding="utf-8")
    if rebuilt_cache_path is not None:
        rebuilt_key = hashlib.sha256(f"{key}:tracklets-observed-v2".encode()).hexdigest()
        rebuilt_cache_path.with_suffix(".meta.json").write_text(
            json.dumps(dict(cache_key=rebuilt_key, source_cache_key=key,
                            source_hash=input_hash, model_hash=model_hash,
                            tracklet_schema="tracklets-observed-v2",
                            start_seconds=start_seconds, duration_seconds=duration_seconds,
                            frame_count=len(frame_ids)), indent=2), encoding="utf-8")
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as sink:
        metrics_writer = csv.DictWriter(sink, fieldnames=list(timings[0]))
        metrics_writer.writeheader()
        metrics_writer.writerows(timings)
    if capture_frames:
        sheet = np.vstack([np.hstack(capture_frames[i:i+3]) for i in range(0, len(capture_frames)-2, 3)])
        cv2.imwrite(str(output_dir / "preview_contact_sheet.jpg"), sheet)
    assert first_frame is not None and last_frame is not None
    cv2.imwrite(str(output_dir / "common_path_map.png"), renderer.render_point_only_frame(
        last_frame, (), latest_snapshot if selected == "directional_grid" else legacy_snapshot,
        tuple(config.analytics.zones), transformer, overlay,
        people_count=0, processing_fps=0, latency_ms=0, timestamp=last_time))
    for image_name in ("directional_field_debug.png", "edge_flow_debug.png"):
        debug = first_frame.copy()
        if image_name.startswith("directional"):
            for cell, bins in grid.histogram().items():
                x = int((cell[1]+.5)*width/config.analytics.directional_grid.columns)
                y = int((cell[0]+.5)*height/config.analytics.directional_grid.rows)
                strong = sorted(((direction, support[1]) for direction, support in bins.items()
                                 if support[1] >= config.analytics.directional_grid.min_edge_unique_tracks),
                                key=lambda item: -item[1])[:2]
                for direction, total in strong:
                    angle = direction*2*math.pi/config.analytics.directional_grid.direction_bins
                    cv2.arrowedLine(debug, (x,y), (x+int(17*math.cos(angle)),y+int(17*math.sin(angle))),
                                    (0,255,255), max(1,min(3,total)), tipLength=.35)
        else:
            for edge in grid.flow_snapshot().edges:
                if edge.unique_tracks_long < config.analytics.directional_grid.min_edge_unique_tracks:
                    continue
                a,b = edge.from_cell,edge.to_cell
                scale = lambda cell: (int((cell[1]+.5)*width/config.analytics.directional_grid.columns),
                                      int((cell[0]+.5)*height/config.analytics.directional_grid.rows))
                cv2.arrowedLine(debug,scale(a),scale(b),(40,220,40),
                                max(1,min(4,edge.unique_tracks_long)),tipLength=.4)
        cv2.imwrite(str(output_dir / image_name), debug)
    path_events = list(grid.events)
    if not path_events:
        path_events.append(dict(
            event="insufficient_data",
            reason="no complete boundary-to-boundary route met configured support",
            path_id="", event_time_s=last_time, evidence_until_s=last_time,
        ))
    with (output_dir / "path_events.jsonl").open("w", encoding="utf-8") as events:
        for event in path_events:
            events.write(json.dumps(event) + "\n")
    (output_dir / "config_resolved.yaml").write_text(
        yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False), encoding="utf-8")
    elapsed = time.perf_counter() - began
    def p50(field: str) -> float:
        return round(float(np.percentile([t[field] for t in timings], 50)), 3)
    def p95(field: str) -> float:
        return round(float(np.percentile([t[field] for t in timings], 95)), 3)
    switch_count = sum(event["event"] == "switch" for event in path_events)
    summary = dict(run_type="analytics_replay" if replay_cache else "gpu_pipeline",
                   engine=engine, selected_display=selected, frames=len(frame_ids),
                   first_frame=frame_ids[0], last_frame=frame_ids[-1],
                   media_start_s=timings[0]["event_time_s"], media_end_s=last_time,
                   device=device,
                   tracks_source=("rebuilt_from_cached_detections" if rebuild_tracks else
                                  "cached_tracks" if replay_cache else "live_gpu_pipeline"),
                   unique_track_ids=len(count_tracks),
                   result="active" if latest_snapshot.paths else "insufficient_data",
                   active_paths=[asdict(p) for p in latest_snapshot.paths],
                   legacy_paths=[asdict(p) for p in legacy_snapshot.paths] if engine != "directional_grid" else [],
                   directional_edges=len(grid.flow_snapshot().edges),
                   support_cap_drops=grid.overflow,
                   rejected_jump_segments=grid.rejected_jumps,
                   analytics_p50_ms=p50("analytics_ms"), analytics_p95_ms=p95("analytics_ms"),
                   decode_p50_ms=p50("decode_ms"), decode_p95_ms=p95("decode_ms"),
                   inference_p50_ms=p50("inference_ms"), inference_p95_ms=p95("inference_ms"),
                   tracking_p50_ms=p50("tracking_ms"), tracking_p95_ms=p95("tracking_ms"),
                   render_p50_ms=p50("render_ms"), render_p95_ms=p95("render_ms"),
                   encode_p50_ms=p50("encode_ms"), encode_p95_ms=p95("encode_ms"),
                   directional_analytics_p95_ms=p95("directional_analytics_ms"),
                   legacy_analytics_p95_ms=p95("legacy_analytics_ms"),
                   path_switches=switch_count,
                   switches_per_minute=round(switch_count / (duration_seconds / 60), 3),
                   validated_route_support=[path.validated_complete_tracks
                                            for path in latest_snapshot.paths],
                   polyline_jitter_px=None,
                   change_detection_delay_s=None,
                   processing_fps=round(len(frame_ids)/elapsed, 3),
                   ram_peak_mb=round(max(ram_samples_mb), 2),
                   vram_allocated_peak_mb=(round(cuda_runtime.cuda.max_memory_allocated()/1024**2, 2)
                                           if cuda_runtime else None),
                   vram_reserved_peak_mb=(round(cuda_runtime.cuda.max_memory_reserved()/1024**2, 2)
                                          if cuda_runtime else None),
                   limitations=["Short clip does not fill 180-second window",
                                "Pixel-space grid has uncalibrated perspective",
                                "Tracking ID fragmentation can undercount complete routes",
                                "Path jitter and change delay are N/A without an Active Path"])
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "inspection.md").write_text(
        "# Artifact inspection\n\nStatus: NOT_REVIEWED\n\n"
        "Review the MP4, contact sheet, both debug fields, path events, and summary.\n",
        encoding="utf-8",
    )
    manifest = dict(run_id=run_id, status="success", **{k: summary[k] for k in
                     ("run_type", "engine", "device", "frames")},
                    input_hash=input_hash, model_hash=model_hash, cache_key=key,
                    cache_file=cache_path.name if replay_cache is None else str(replay_cache),
                    rebuilt_cache_file=(rebuilt_cache_path.name if rebuilt_cache_path else None),
                    clip=dict(start_seconds=start_seconds, duration_seconds=duration_seconds),
                    artifacts=sorted(p.name for p in output_dir.iterdir()))
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--config", default="configs/default.yaml", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--start-seconds", type=float, default=0.)
    parser.add_argument("--duration-seconds", type=float, default=25.)
    parser.add_argument("--engine", choices=("legacy","directional_grid","shadow"), default="directional_grid")
    parser.add_argument("--replay-cache", type=Path)
    parser.add_argument("--source-hash")
    parser.add_argument("--model-hash")
    parser.add_argument("--rebuild-tracks", action="store_true",
                        help="Re-run ByteTrack from cached detections without inference")
    args = parser.parse_args()
    print(json.dumps(run(args.input, args.output_dir, config_path=args.config, model_path=args.model,
                         start_seconds=args.start_seconds, duration_seconds=args.duration_seconds,
                         run_id=args.run_id, engine=args.engine, replay_cache=args.replay_cache,
                         input_hash=args.source_hash, model_hash=args.model_hash,
                         rebuild_tracks=args.rebuild_tracks)))


if __name__ == "__main__":
    main()
