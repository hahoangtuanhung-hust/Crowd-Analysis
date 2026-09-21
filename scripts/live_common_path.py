"""Stateful Shibuya processor used by the Modal WebSocket endpoint."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import threading
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator

import cv2
import numpy as np
import psutil
import yaml

from backend.app.analytics.directional_grid import DirectionalGridEngine, GridTrackPoint
from backend.app.analytics.dominant_live_flow import DominantLiveFlowEngine
from backend.app.analytics.spatial import SpatialTransformer
from backend.app.core.config import load_config
from backend.app.schemas import Detection, TrackedObject
from backend.app.tracking import ByteTrackTracker
from backend.app.video.renderer import FrameRenderer, OverlayOptions


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def _path_payload(snapshot: Any) -> tuple[dict[str, Any] | None, str, str]:
    selected = next(
        (path for path in snapshot.paths if path.state in {"active", "cooling"}), None
    )
    if selected is not None:
        reason = "ACTIVE" if selected.state == "active" else "STALE_EVIDENCE"
        return asdict(selected), selected.state, reason
    candidate = next((path for path in snapshot.paths if path.state == "candidate"), None)
    if candidate is not None:
        return asdict(candidate), "confirming", "HYSTERESIS_CONFIRMING"
    return None, "learning", "INSUFFICIENT_DATA"


def _candidate_funnel(engine: Any) -> dict[str, Any]:
    evaluations = list(getattr(engine, "candidate_evaluations", ()))
    diagnostics = list(getattr(engine, "graph_diagnostics", ()))
    return {
        "unique_routes": len({item["candidate_id"] for item in evaluations}),
        "evaluation_count": len(evaluations),
        "decisions": dict(Counter(item["decision"] for item in evaluations)),
        "primary_reasons": dict(Counter(item["primary_reason"] for item in evaluations)),
        "graph_reasons": dict(Counter(
            item["primary_reason"] or "CANDIDATES_EVALUATED" for item in diagnostics
        )),
    }


def _percentile(rows: list[dict[str, Any]], field: str, percentile: int = 95) -> float | None:
    if not rows:
        return None
    return round(float(np.percentile([row[field] for row in rows], percentile)), 3)


class LiveCommonPathProcessor:
    """Causal GPU-inference processor with an optional PTS-paced input loop."""

    def __init__(
        self,
        *,
        source: Path,
        config_path: Path,
        model_path: Path,
        output: Path,
        run_id: str,
        session_id: str,
        stream_epoch: str,
        duration_seconds: float,
        preview_fps: float,
        source_name: str | None = None,
        detector: Any | None = None,
        device_name: str = "unknown",
        stop_event: threading.Event | None = None,
        processing_mode: str = "inspect_all_frames",
    ) -> None:
        if processing_mode not in {"inspect_all_frames", "realtime_pts"}:
            raise ValueError("processing_mode must be inspect_all_frames or realtime_pts")
        self.source = source
        self.config_path = config_path
        self.model_path = model_path
        self.output = output
        self.run_id = run_id
        self.session_id = session_id
        self.stream_epoch = stream_epoch
        self.duration_seconds = duration_seconds
        self.preview_fps = preview_fps
        self.source_name = source_name or source.name
        self.detector = detector
        self.device_name = device_name
        self.stop_event = stop_event or threading.Event()
        self.processing_mode = processing_mode
        self.final_manifest: dict[str, Any] | None = None

    def frames(self) -> Iterator[tuple[dict[str, Any], bytes]]:
        boot_started = time.perf_counter()
        if self.output.exists():
            raise FileExistsError(f"Refusing to overwrite live run: {self.output}")
        self.output.mkdir(parents=True)
        config = load_config(self.config_path)
        config.analytics.directional_grid.diagnostics_enabled = True
        config.visualization.show_bounding_boxes = False
        config.visualization.show_track_ids = False
        config.visualization.show_individual_trajectories = False
        config.visualization.show_candidate_path = False
        config.visualization.show_common_path = True
        config.visualization.show_tracking_points = True
        input_hash = digest(self.source)
        model_hash = digest(self.model_path)
        cap = cv2.VideoCapture(str(self.source))
        if not cap.isOpened():
            raise ValueError(f"Cannot decode source: {self.source}")
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        source_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0 or width <= 0 or height <= 0:
            raise ValueError("Source video metadata is incomplete")
        limit = min(source_frames, max(1, round(self.duration_seconds * fps)))
        transformer = SpatialTransformer.pixel(width, height)
        dominant_mode = config.analytics.dominant_live_flow.mode == "dominant_live_flow"
        if dominant_mode:
            engine: Any = DominantLiveFlowEngine(
                config.analytics.dominant_live_flow,
                transformer,
                camera_id="shibuya-01",
                stream_epoch=self.stream_epoch,
            )
            route_scope = "dominant_live_flow"
        else:
            engine = DirectionalGridEngine(
                config.analytics.directional_grid,
                transformer,
                camera_id="shibuya-01",
                stream_epoch=self.stream_epoch,
                zones=config.analytics.zones,
                run_id=self.run_id,
                variant="live_inference",
            )
            route_scope = config.analytics.directional_grid.route_scope
        tracker = ByteTrackTracker(config.tracker)
        renderer = FrameRenderer(config.visualization)
        overlay = OverlayOptions(
            tracking=True,
            points=True,
            track_ids=False,
            trajectory=False,
            trajectory_tails=False,
            candidate_paths=False,
            active_paths=True,
            direction_arrows=True,
            debug_metrics=True,
        )
        if self.detector is None:
            from backend.app.inference import UltralyticsPersonDetector

            detector_config = config.detector.model_copy(
                update={"device": "cuda:0", "model": str(self.model_path)}
            )
            self.detector = UltralyticsPersonDetector(detector_config)
        model_ready_s = time.perf_counter() - boot_started

        writer = cv2.VideoWriter(
            str(self.output / "preview.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError("Cannot open live preview writer")
        cache_file = (self.output / "tracking_cache.jsonl").open("w", encoding="utf-8")
        timeline_file = (self.output / "path_timeline.jsonl").open("w", encoding="utf-8")
        frame_index_file = (self.output / "frame_index.jsonl").open("w", encoding="utf-8")
        cuts_file = (self.output / "tracking_cuts.jsonl").open("w", encoding="utf-8")
        metrics_rows: list[dict[str, Any]] = []
        unique_tracks: set[int] = set()
        raw_last: dict[int, tuple[float, float, float, int]] = {}
        prediction_counts: Counter[int] = Counter()
        cut_reasons: Counter[str] = Counter()
        preview_count = 0
        preview_drops = 0
        input_drops = 0
        detector_calls = 0
        next_preview_s = 0.0
        last_timeline_s = -math.inf
        last_timeline_signature: tuple[Any, ...] | None = None
        last_source_frame_id = -1
        last_media_s = 0.0
        first_frame_s: float | None = None
        pending_rendered: np.ndarray | None = None
        pending_source_frame_id = -1
        peak_ram_mb = 0.0
        process = psutil.Process()
        started = time.perf_counter()
        wall_started = time.time()
        source_frame_id = 0
        error: str | None = None
        stopped = False
        try:
            while source_frame_id < limit:
                if self.stop_event.is_set():
                    stopped = True
                    break
                if self.processing_mode == "realtime_pts":
                    while True:
                        elapsed_before_frame = time.perf_counter() - started
                        due_frame = min(limit - 1, int(elapsed_before_frame * fps))
                        if due_frame >= source_frame_id:
                            break
                        if self.stop_event.wait(min(0.05, (source_frame_id / fps) - elapsed_before_frame)):
                            stopped = True
                            break
                    if stopped:
                        break
                    while source_frame_id < due_frame:
                        if not cap.grab():
                            source_frame_id = limit
                            break
                        source_frame_id += 1
                        input_drops += 1
                    if source_frame_id >= limit:
                        break

                stage_started = time.perf_counter()
                ok, frame = cap.read()
                decode_ms = (time.perf_counter() - stage_started) * 1000
                if not ok:
                    break
                media_s = source_frame_id / fps
                last_media_s = media_s

                infer_started = time.perf_counter()
                detections: list[Detection] = self.detector.detect(frame)
                detector_calls += 1
                inference_ms = (time.perf_counter() - infer_started) * 1000
                track_started = time.perf_counter()
                tracks: list[TrackedObject] = tracker.update(detections, frame)
                tracking_ms = (time.perf_counter() - track_started) * 1000
                unique_tracks.update(track.track_id for track in tracks)

                for track in tracks:
                    if not track.observed:
                        prediction_counts[track.track_id] += 1
                        continue
                    x, y = track.bottom_center
                    previous = raw_last.get(track.track_id)
                    if previous is None:
                        reason = "RAW_ID_START"
                        dt = displacement = step_cells = None
                    else:
                        previous_s, previous_x, previous_y, _ = previous
                        dt = media_s - previous_s
                        displacement = math.hypot(x - previous_x, y - previous_y)
                        step_cells = math.hypot(
                            (x - previous_x) / (width / config.analytics.directional_grid.columns),
                            (y - previous_y) / (height / config.analytics.directional_grid.rows),
                        )
                        if dt > config.analytics.directional_grid.max_observation_gap_seconds:
                            reason = "OBSERVATION_GAP"
                        elif step_cells > config.analytics.directional_grid.max_step_cells:
                            reason = "JUMP_THRESHOLD"
                        else:
                            reason = "CONTINUOUS"
                    if reason != "CONTINUOUS":
                        cut_reasons[reason] += 1
                        cuts_file.write(json.dumps({
                            "track_id": track.track_id,
                            "frame_id": source_frame_id,
                            "event_time_s": media_s,
                            "previous_frame_id": previous[3] if previous else None,
                            "delta_time_s": dt,
                            "displacement_pixels": displacement,
                            "step_cells": step_cells,
                            "reason": reason,
                        }) + "\n")
                    raw_last[track.track_id] = (media_s, x, y, source_frame_id)

                points = [
                    GridTrackPoint(
                        "shibuya-01", self.stream_epoch, track.track_id, 0,
                        source_frame_id, media_s, *track.bottom_center,
                        observed=track.observed,
                    )
                    for track in tracks
                ]
                analytics_started = time.perf_counter()
                snapshot = engine.update(points, media_s)
                analytics_ms = (time.perf_counter() - analytics_started) * 1000
                path, path_state, reason = _path_payload(snapshot)
                if dominant_mode:
                    status = engine.status
                    path_state, reason = status.state, status.reason
                    active_count = status.active_count
                    active_direction = status.active_direction
                    challenger_count = status.challenger_count
                    challenger_direction = status.challenger_direction
                    confirmation_elapsed_s = status.confirmation_elapsed_s
                    confirmation_required_s = status.confirmation_required_s
                    moving_track_count = status.moving_track_count
                else:
                    diagnostics = getattr(engine, "graph_diagnostics", ())
                    if diagnostics and path is None:
                        reason = diagnostics[-1]["primary_reason"] or "INSUFFICIENT_DATA"
                    active_count = path["support_tracks"] if path else 0
                    active_direction = path["direction"] if path else None
                    challenger_count = 0
                    challenger_direction = None
                    confirmation_elapsed_s = 0.0
                    confirmation_required_s = config.analytics.directional_grid.confirmation_seconds
                    moving_track_count = 0
                active_path = path if path_state in {"active", "cooling"} else None
                current_points = [
                    SimpleNamespace(
                        track_id=track.track_id,
                        x=track.bottom_center[0],
                        y=track.bottom_center[1],
                    )
                    for track in tracks if track.observed
                ]
                elapsed = max(time.perf_counter() - started, 1e-6)
                render_started = time.perf_counter()
                rendered = renderer.render_point_only_frame(
                    frame,
                    current_points,
                    snapshot,
                    (),
                    transformer,
                    overlay,
                    people_count=len(current_points),
                    processing_fps=detector_calls / elapsed,
                    latency_ms=inference_ms,
                    timestamp=media_s,
                    directed_flows=engine.flow_snapshot() if hasattr(engine, "flow_snapshot") else (),
                )
                render_ms = (time.perf_counter() - render_started) * 1000
                encode_started = time.perf_counter()
                if pending_rendered is not None:
                    for _ in range(source_frame_id - pending_source_frame_id):
                        writer.write(pending_rendered)
                pending_rendered = rendered
                pending_source_frame_id = source_frame_id
                encode_ms = (time.perf_counter() - encode_started) * 1000

                cache_file.write(json.dumps({
                    "frame_id": source_frame_id,
                    "event_time_s": media_s,
                    "detections": [asdict(item) for item in detections],
                    "tracks": [asdict(item) for item in tracks],
                }) + "\n")
                frame_index_file.write(json.dumps({
                    "processed_frame_index": detector_calls - 1,
                    "source_frame_id": source_frame_id,
                    "media_time_s": media_s,
                    "path_revision": active_path["revision"] if active_path else None,
                    "path_state": path_state,
                    "evidence_until_s": (
                        active_path["evidence_until_s"] if active_path else snapshot.evidence_until_s
                    ),
                }) + "\n")
                signature = (
                    active_path["path_id"] if active_path else None,
                    active_path["revision"] if active_path else None,
                    path_state,
                    reason,
                    active_count,
                    challenger_count,
                    challenger_direction,
                )
                if signature != last_timeline_signature or media_s - last_timeline_s >= 1.0:
                    timeline_file.write(json.dumps({
                        "media_time_s": media_s,
                        "frame_id": source_frame_id,
                        "path_id": active_path["path_id"] if active_path else None,
                        "path_revision": active_path["revision"] if active_path else None,
                        "path_state": path_state,
                        "scope": route_scope,
                        "direction": active_direction,
                        "active_count": active_count,
                        "challenger_direction": challenger_direction,
                        "challenger_count": challenger_count,
                        "moving_track_count": moving_track_count,
                        "confirmation_elapsed_s": round(confirmation_elapsed_s, 3),
                        "confirmation_required_s": confirmation_required_s,
                        "polyline": active_path["polyline"] if active_path else [],
                        "evidence_until_s": (
                            active_path["evidence_until_s"] if active_path else snapshot.evidence_until_s
                        ),
                        "reason": reason,
                    }) + "\n")
                    last_timeline_s = media_s
                    last_timeline_signature = signature

                peak_ram_mb = max(peak_ram_mb, process.memory_info().rss / 1024 / 1024)
                metrics_rows.append({
                    "frame_id": source_frame_id,
                    "media_time_s": media_s,
                    "detections": len(detections),
                    "tracks": len(tracks),
                    "observed_tracks": len(current_points),
                    "moving_tracks": moving_track_count,
                    "decode_ms": decode_ms,
                    "inference_ms": inference_ms,
                    "tracking_ms": tracking_ms,
                    "analytics_ms": analytics_ms,
                    "render_ms": render_ms,
                    "encode_ms": encode_ms,
                })
                last_source_frame_id = source_frame_id

                if media_s + 1e-9 >= next_preview_s:
                    next_preview_s = media_s + 1.0 / self.preview_fps
                    jpeg_started = time.perf_counter()
                    encoded, jpeg = cv2.imencode(
                        ".jpg", rendered,
                        [cv2.IMWRITE_JPEG_QUALITY, config.server.jpeg_quality],
                    )
                    if not encoded:
                        raise RuntimeError(f"JPEG encode failed at frame {source_frame_id}")
                    jpeg_bytes = jpeg.tobytes()
                    preview_count += 1
                    elapsed = max(time.perf_counter() - started, 1e-6)
                    if first_frame_s is None:
                        first_frame_s = model_ready_s + elapsed
                    evidence_until_s = (
                        active_path["evidence_until_s"] if active_path else snapshot.evidence_until_s
                    )
                    metadata = {
                        "version": 1,
                        "type": "frame",
                        "session_id": self.session_id,
                        "stream_epoch": self.stream_epoch,
                        "input_hash": input_hash,
                        "source_name": self.source_name,
                        "mode": "live_inference",
                        "processing_mode": self.processing_mode,
                        "frame_id": source_frame_id,
                        "media_time_s": media_s,
                        "frame_width": width,
                        "frame_height": height,
                        "path_id": active_path["path_id"] if active_path else None,
                        "path_revision": active_path["revision"] if active_path else None,
                        "path_state": path_state,
                        "path_scope": route_scope,
                        "path_updated_at_s": active_path["updated_at"] if active_path else None,
                        "evidence_until_s": evidence_until_s,
                        "path_support": active_count,
                        "path_complete_tracks": active_count,
                        "path_reason": reason,
                        "direction": active_direction,
                        "active_count": active_count,
                        "challenger_direction": challenger_direction,
                        "challenger_count": challenger_count,
                        "confirmation_elapsed_s": confirmation_elapsed_s,
                        "confirmation_required_s": confirmation_required_s,
                        "moving_track_count": moving_track_count,
                        "common_path": active_path,
                        "processing_fps": detector_calls / elapsed,
                        "preview_fps": preview_count / elapsed,
                        "source_fps": fps,
                        "inference_ms": inference_ms,
                        "queue_depth": 0,
                        "dropped_input_frames": input_drops,
                        "dropped_preview_frames": preview_drops,
                        "observed_people": len(current_points),
                        "unique_track_ids": len(unique_tracks),
                        "jpeg_bytes": len(jpeg_bytes),
                        "jpeg_encode_ms": (time.perf_counter() - jpeg_started) * 1000,
                    }
                    if evidence_until_s > media_s + 1e-9:
                        raise RuntimeError("Path evidence is newer than the rendered frame")
                    yield metadata, jpeg_bytes
                else:
                    preview_drops += 1
                source_frame_id += 1
        except Exception as exc:
            error = str(exc)
            raise
        finally:
            cap.release()
            if pending_rendered is not None:
                repeats = 1
                if not stopped and error is None:
                    repeats = max(1, limit - pending_source_frame_id)
                for _ in range(repeats):
                    writer.write(pending_rendered)
            writer.release()
            for sink in (cache_file, timeline_file, frame_index_file, cuts_file):
                sink.close()
            if metrics_rows:
                with (self.output / "metrics.csv").open("w", newline="", encoding="utf-8") as sink:
                    csv_writer = csv.DictWriter(sink, fieldnames=list(metrics_rows[0]))
                    csv_writer.writeheader()
                    csv_writer.writerows(metrics_rows)
            with (self.output / "candidate_evaluations.jsonl").open("w", encoding="utf-8") as sink:
                for item in getattr(engine, "candidate_evaluations", ()):
                    sink.write(json.dumps(item) + "\n")
            with (self.output / "graph_diagnostics.jsonl").open("w", encoding="utf-8") as sink:
                for item in getattr(engine, "graph_diagnostics", ()):
                    sink.write(json.dumps(item) + "\n")
            with (self.output / "path_events.jsonl").open("w", encoding="utf-8") as sink:
                for item in engine.events:
                    sink.write(json.dumps(item) + "\n")
            (self.output / "config_resolved.yaml").write_text(
                yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False),
                encoding="utf-8",
            )
            cache_schema = "detections-tracklets-live-realtime-v1"
            cache_key = hashlib.sha256(json.dumps({
                "schema": cache_schema,
                "input_hash": input_hash,
                "model_hash": model_hash,
                "detector": config.detector.model_dump(exclude={"device", "model"}),
                "tracker": config.tracker.model_dump(),
                "duration_seconds": self.duration_seconds,
                "processing_mode": self.processing_mode,
            }, sort_keys=True).encode()).hexdigest()
            _write_json(self.output / "tracking_cache.meta.json", {
                "cache_key": cache_key,
                "schema": cache_schema,
                "source_hash": input_hash,
                "model_hash": model_hash,
                "start_seconds": 0.0,
                "duration_seconds": self.duration_seconds,
                "processed_frame_count": detector_calls,
                "last_source_frame_id": last_source_frame_id,
            })
            elapsed = time.perf_counter() - started
            video_seconds = (last_source_frame_id + 1) / fps if last_source_frame_id >= 0 else 0.0
            metrics = {
                "source_fps": round(fps, 3),
                "source_frames_in_window": limit,
                "frames_processed": detector_calls,
                "processing_fps": round(detector_calls / max(elapsed, 1e-6), 3),
                "preview_frames_sent": preview_count,
                "preview_fps": round(preview_count / max(elapsed, 1e-6), 3),
                "effective_media_rate": round(video_seconds / max(elapsed, 1e-6), 3),
                "dropped_input_frames": input_drops,
                "dropped_preview_frames": preview_drops,
                "queue_max_depth": 1,
                "peak_ram_mb": round(peak_ram_mb, 3),
                "model_ready_seconds": round(model_ready_s, 3),
                "first_frame_seconds": round(first_frame_s, 3) if first_frame_s else None,
                "decode_p95_ms": _percentile(metrics_rows, "decode_ms"),
                "inference_p95_ms": _percentile(metrics_rows, "inference_ms"),
                "tracking_p95_ms": _percentile(metrics_rows, "tracking_ms"),
                "analytics_p95_ms": _percentile(metrics_rows, "analytics_ms"),
                "render_p95_ms": _percentile(metrics_rows, "render_ms"),
                "encode_p95_ms": _percentile(metrics_rows, "encode_ms"),
            }
            _write_json(self.output / "metrics.json", metrics)
            manifest = {
                "run_id": self.run_id,
                "session_id": self.session_id,
                "stream_epoch": self.stream_epoch,
                "status": "error" if error else "stopped" if stopped or self.stop_event.is_set() else "completed",
                "error": error,
                "source": self.source_name,
                "input_hash": input_hash,
                "model_hash": model_hash,
                "mode": "dominant_live_flow" if dominant_mode else "validated_route",
                "processing_mode": self.processing_mode,
                "device": self.device_name,
                "detector_calls": detector_calls,
                "frames_processed": detector_calls,
                "preview_frames_sent": preview_count,
                "dropped_input_frames": input_drops,
                "dropped_preview_frames": preview_drops,
                "media_end_s": last_media_s,
                "runtime_seconds": round(elapsed, 3),
                "processing_fps": metrics["processing_fps"],
                "preview_fps": metrics["preview_fps"],
                "inference_p95_ms": metrics["inference_p95_ms"],
                "tracking_p95_ms": metrics["tracking_p95_ms"],
                "analytics_p95_ms": metrics["analytics_p95_ms"],
                "render_p95_ms": metrics["render_p95_ms"],
                "unique_track_ids": len(unique_tracks),
                "prediction_only_frames": sum(prediction_counts.values()),
                "tracking_cut_reasons": dict(cut_reasons),
                "candidate_funnel": _candidate_funnel(engine),
                "active_paths": [
                    asdict(path) for path in engine.snapshot().paths
                    if path.state in {"active", "cooling"}
                ],
                "versions": {
                    "python": platform.python_version(),
                    "opencv": cv2.__version__,
                    "numpy": np.__version__,
                },
                "wall_started_unix_s": wall_started,
                "wall_completed_unix_s": time.time(),
            }
            (self.output / "report.md").write_text(
                "# Dominant live flow run\n\n"
                f"- Run: `{self.run_id}`\n"
                f"- Source: `{self.source_name}` (`{input_hash}`)\n"
                f"- Device: `{self.device_name}`\n"
                f"- Mode: `{manifest['mode']}` / `{self.processing_mode}`\n"
                f"- Detector calls: {detector_calls}; input drops: {input_drops}\n"
                "- Visual review: NOT_REVIEWED (updated after download)\n"
                "- Limitation: counts are tracking-ID estimates and can be affected by ID switches.\n",
                encoding="utf-8",
            )
            manifest["artifacts"] = sorted(
                path.name for path in self.output.iterdir() if path.is_file()
            )
            _write_json(self.output / "manifest.json", manifest)
            self.final_manifest = manifest
