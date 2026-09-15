from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from backend.app.analytics import CompletedTracklet, PointTrackletManager
from backend.app.core.config import ZoneConfig, load_config
from backend.app.inference import UltralyticsPersonDetector
from backend.app.tracking import ByteTrackTracker

CSV_FIELDS = ("camera_id", "track_id", "frame_id", "timestamp", "x", "y", "zone_id")
FRAME_FIELDS = (
    "frame_id",
    "raw_person_detections",
    "active_tracks",
    "new_tracks",
    "lost_tracks",
    "processing_fps",
    "inference_ms",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create point-only tracks and entry/exit maps")
    parser.add_argument("source", nargs="?", default="data/videos/data.mp4")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--zones", default="configs/zones.json")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--camera-id", default="cam01")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--batch-size", type=int, default=1000)
    return parser.parse_args()


def load_zones(path: Path, width: int, height: int) -> tuple[ZoneConfig, ...]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    source_width = float(raw.get("image_width", width))
    source_height = float(raw.get("image_height", height))
    scale_x = width / source_width
    scale_y = height / source_height
    zones = []
    for item in raw.get("zones", []):
        zone_id = item.get("id", item.get("zone_id"))
        if not zone_id:
            raise ValueError("Every zone needs an 'id' or 'zone_id'")
        zones.append(
            ZoneConfig(
                zone_id=str(zone_id),
                name=str(item.get("name", zone_id)),
                points=[
                    (float(x) * scale_x, float(y) * scale_y) for x, y in item["points"]
                ],
            )
        )
    return tuple(zones)


class BatchedCsvWriter:
    def __init__(self, path: Path, fields: tuple[str, ...], batch_size: int) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self._stream = path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._stream, fieldnames=fields)
        self._writer.writeheader()
        self._batch: list[dict[str, object]] = []
        self._batch_size = batch_size

    def add(self, row: dict[str, object]) -> None:
        self._batch.append(row)
        if len(self._batch) >= self._batch_size:
            self.flush()

    def flush(self) -> None:
        if self._batch:
            self._writer.writerows(self._batch)
            self._batch.clear()
            self._stream.flush()

    def close(self) -> None:
        self.flush()
        self._stream.close()


def draw_zones(frame: np.ndarray, zones: tuple[ZoneConfig, ...]) -> None:
    for index, zone in enumerate(zones):
        points = np.asarray(zone.points, dtype=np.int32).reshape((-1, 1, 2))
        color = ((73 + index * 41) % 220, (165 + index * 29) % 220, (239 + index * 17) % 255)
        cv2.polylines(frame, [points], True, color, 2, cv2.LINE_AA)
        cv2.putText(
            frame,
            zone.name,
            tuple(points[0, 0]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            color,
            1,
            cv2.LINE_AA,
        )


def draw_completed_route(layer: np.ndarray, tracklet: CompletedTracklet) -> None:
    if not tracklet.confirmed or len(tracklet.points) < 2:
        return
    points = np.asarray([(round(p.x), round(p.y)) for p in tracklet.points], dtype=np.int32)
    color = (
        64 + tracklet.track_id * 37 % 160,
        64 + tracklet.track_id * 67 % 160,
        64 + tracklet.track_id * 97 % 160,
    )
    cv2.polylines(layer, [points.reshape((-1, 1, 2))], False, color, 2, cv2.LINE_AA)
    cv2.circle(layer, tuple(points[0]), 3, (80, 220, 120), -1, cv2.LINE_AA)
    cv2.arrowedLine(
        layer,
        tuple(points[-2]),
        tuple(points[-1]),
        color,
        2,
        cv2.LINE_AA,
        tipLength=0.4,
    )


def render_live_frame(
    frame: np.ndarray,
    tracks,
    histories,
    zones: tuple[ZoneConfig, ...],
    *,
    processing_fps: float,
    latency_ms: float,
) -> np.ndarray:
    rendered = frame.copy()
    draw_zones(rendered, zones)
    for track in tracks:
        history = histories.get(track.track_id, ())
        if len(history) >= 2:
            points = np.asarray([(round(p.x), round(p.y)) for p in history], dtype=np.int32)
            cv2.polylines(
                rendered,
                [points.reshape((-1, 1, 2))],
                False,
                (52, 211, 255),
                2,
                cv2.LINE_AA,
            )
            if len(points) >= 3:
                cv2.arrowedLine(
                    rendered,
                    tuple(points[-3]),
                    tuple(points[-1]),
                    (52, 211, 255),
                    2,
                    cv2.LINE_AA,
                    tipLength=0.35,
                )
        if history:
            point = (round(history[-1].x), round(history[-1].y))
        else:
            x, y = track.bottom_center
            point = (round(x), round(y))
        cv2.circle(rendered, point, 5, (64, 202, 142), -1, cv2.LINE_AA)
        cv2.circle(rendered, point, 7, (245, 247, 250), 1, cv2.LINE_AA)
        cv2.putText(
            rendered,
            f"ID {track.track_id}",
            (point[0] + 8, max(18, point[1] - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    label = f"People {len(tracks)}   FPS {processing_fps:.1f}   Latency {latency_ms:.0f} ms"
    cv2.rectangle(rendered, (12, 12), (430, 46), (15, 20, 25), -1)
    cv2.putText(
        rendered,
        label,
        (22, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (245, 247, 250),
        2,
        cv2.LINE_AA,
    )
    return rendered


def main() -> int:
    args = parse_args()
    source = Path(args.source)
    if not source.is_file():
        raise FileNotFoundError(f"Video does not exist: {source}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config)

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError(f"Unable to open video: {source}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    zones = load_zones(Path(args.zones), width, height)
    detector = UltralyticsPersonDetector(config.detector)
    tracker = ByteTrackTracker(config.tracker)
    tracklets = PointTrackletManager(
        config.analytics,
        config.tracker,
        zones,
        camera_id=args.camera_id,
    )
    density = np.zeros(
        (config.analytics.grid_height, config.analytics.grid_width), dtype=np.float32
    )

    video_path = output_dir / "tracked_points.mp4"
    video_writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        source_fps,
        (width, height),
    )
    if not video_writer.isOpened():
        capture.release()
        raise RuntimeError(f"Unable to create output video: {video_path}")
    trajectory_writer = BatchedCsvWriter(
        output_dir / "trajectories.csv", CSV_FIELDS, args.batch_size
    )
    frame_writer = BatchedCsvWriter(
        output_dir / "frame_metrics.csv", FRAME_FIELDS, args.batch_size
    )

    route_layer = np.zeros((height, width, 3), dtype=np.uint8)
    background: np.ndarray | None = None
    recent_fps: deque[float] = deque(maxlen=30)
    previous_ids: set[int] = set()
    seen_ids: set[int] = set()
    measured_inference: list[float] = []
    measured_processing: list[float] = []
    raw_counts: list[int] = []
    active_counts: list[int] = []
    frame_id = 0
    try:
        while args.max_frames is None or frame_id < args.max_frames:
            ok, frame = capture.read()
            if not ok:
                break
            if background is None:
                background = frame.copy()
            started = time.perf_counter()
            inference_started = time.perf_counter()
            detections = detector.detect(frame)
            inference_ms = (time.perf_counter() - inference_started) * 1000.0
            tracking_started = time.perf_counter()
            tracks = tracker.update(detections, frame)
            tracking_ms = (time.perf_counter() - tracking_started) * 1000.0
            processing_ms = (time.perf_counter() - started) * 1000.0
            instant_fps = 1000.0 / max(processing_ms, 1e-9)
            recent_fps.append(instant_fps)

            active_ids = {track.track_id for track in tracks}
            new_ids = active_ids - seen_ids
            lost_ids = previous_ids - active_ids
            seen_ids.update(active_ids)
            timestamp = frame_id / source_fps
            update = tracklets.update(tracks, frame_id=frame_id, timestamp=timestamp)
            for point in update.persist_points:
                trajectory_writer.add(point.as_csv_row())
                column = min(
                    config.analytics.grid_width - 1,
                    max(0, int(point.x / width * config.analytics.grid_width)),
                )
                row = min(
                    config.analytics.grid_height - 1,
                    max(0, int(point.y / height * config.analytics.grid_height)),
                )
                density[row, column] += 1.0
            for completed in update.completed:
                draw_completed_route(route_layer, completed)

            processing_fps = statistics.fmean(recent_fps)
            frame_writer.add(
                {
                    "frame_id": frame_id,
                    "raw_person_detections": len(detections),
                    "active_tracks": len(active_ids),
                    "new_tracks": len(new_ids),
                    "lost_tracks": len(lost_ids),
                    "processing_fps": round(processing_fps, 4),
                    "inference_ms": round(inference_ms, 4),
                }
            )
            histories = tracklets.histories(active_ids)
            video_writer.write(
                render_live_frame(
                    frame,
                    tracks,
                    histories,
                    zones,
                    processing_fps=processing_fps,
                    latency_ms=inference_ms + tracking_ms,
                )
            )
            raw_counts.append(len(detections))
            active_counts.append(len(active_ids))
            measured_inference.append(inference_ms)
            measured_processing.append(processing_ms)
            previous_ids = active_ids
            frame_id += 1
            if frame_id % 100 == 0:
                print(
                    f"processed={frame_id} detections={len(detections)} "
                    f"tracks={len(active_ids)} fps={processing_fps:.2f}",
                    flush=True,
                )
    finally:
        capture.release()
        for completed in tracklets.finalize_all():
            draw_completed_route(route_layer, completed)
        trajectory_writer.close()
        frame_writer.close()
        video_writer.release()

    if frame_id == 0 or background is None:
        raise RuntimeError("Input video contains no decodable frames")

    dark_background = cv2.addWeighted(background, 0.28, np.zeros_like(background), 0.72, 0.0)
    path_map = cv2.add(dark_background, route_layer)
    draw_zones(path_map, zones)
    flows = tracklets.zone_flows()
    cv2.putText(
        path_map,
        f"Completed routes: {tracklets.completed_tracklets}",
        (22, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (245, 247, 250),
        2,
        cv2.LINE_AA,
    )
    for index, flow in enumerate(flows[:5], start=1):
        label = (
            f"{index}. {flow['from_name']} -> {flow['to_name']}: "
            f"{flow['unique_track_ids']} IDs"
        )
        cv2.putText(
            path_map,
            label,
            (22, 38 + index * 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (245, 247, 250),
            1,
            cv2.LINE_AA,
        )
    cv2.imwrite(str(output_dir / "path_map.png"), path_map)

    smoothed_density = density
    if config.analytics.gaussian_sigma > 0:
        smoothed_density = cv2.GaussianBlur(
            density,
            ksize=(0, 0),
            sigmaX=config.analytics.gaussian_sigma,
            sigmaY=config.analytics.gaussian_sigma,
        )
    maximum = float(smoothed_density.max(initial=0.0))
    if maximum > 0:
        normalized = np.clip(smoothed_density / maximum * 255.0, 0, 255).astype(np.uint8)
        heatmap_small = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    else:
        heatmap_small = np.zeros((*density.shape, 3), dtype=np.uint8)
    heatmap_full = cv2.resize(heatmap_small, (width, height), interpolation=cv2.INTER_LINEAR)
    heatmap_output = cv2.addWeighted(background, 0.35, heatmap_full, 0.65, 0.0)
    draw_zones(heatmap_output, zones)
    cv2.imwrite(str(output_dir / "heatmap.png"), heatmap_output)

    processing_seconds = sum(measured_processing) / 1000.0
    summary = {
        "input": str(source.resolve()),
        "frames": frame_id,
        "source_fps": source_fps,
        "detector": config.detector.model_dump(),
        "tracker": config.tracker.model_dump(),
        "raw_detections_per_frame": statistics.fmean(raw_counts),
        "active_tracks_per_frame": statistics.fmean(active_counts),
        "peak_active_tracks": max(active_counts),
        "unique_track_ids": len(seen_ids),
        "processing_fps": frame_id / processing_seconds,
        "inference_ms": statistics.fmean(measured_inference),
        "completed_tracklets": tracklets.completed_tracklets,
        "discarded_short_tracklets": tracklets.discarded_short_tracklets,
        "zone_flows": flows,
    }
    (output_dir / "zone_flows.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
