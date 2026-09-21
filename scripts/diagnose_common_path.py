"""Replay-only Common Path diagnostics. This module intentionally has no detector import."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import statistics
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import cv2
import numpy as np
import psutil
import yaml

from backend.app.analytics.directional_grid import DirectionalGridEngine, GridTrackPoint
from backend.app.analytics.spatial import SpatialTransformer
from backend.app.core.config import AppConfig, DirectionalGridConfig, LocalCorridorConfig, load_config
from backend.app.schemas import TrackedObject
from backend.app.video.renderer import FrameRenderer, OverlayOptions


@dataclass(slots=True)
class _TrackAudit:
    track_id: int
    segment_id: int
    first_frame: int
    last_frame: int
    first_time_s: float
    last_time_s: float
    observed_frames: int = 0
    prediction_frames: int = 0
    first_observed_frame: int | None = None
    last_observed_frame: int | None = None
    first_observed_s: float | None = None
    last_observed_s: float | None = None
    first_position: tuple[float, float] | None = None
    second_position: tuple[float, float] | None = None
    previous_position: tuple[float, float] | None = None
    last_position: tuple[float, float] | None = None
    previous_observed_s: float | None = None
    valid_distance_px: float = 0.0
    max_observation_gap_s: float = 0.0
    cells: list[tuple[int, int]] = field(default_factory=list)


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def _percentile(values: Iterable[float], percentile: float) -> float | None:
    data = list(values)
    return round(float(np.percentile(data, percentile)), 4) if data else None


def _normalized_polygon(points: Iterable[tuple[float, float]], width: int,
                        height: int, coordinate_space: str) -> np.ndarray:
    scale = (width, height) if coordinate_space == "image_normalized" else (1, 1)
    return np.asarray([(round(x * scale[0]), round(y * scale[1])) for x, y in points],
                      dtype=np.int32)


def _inside(point: tuple[float, float], polygon: np.ndarray) -> bool:
    return cv2.pointPolygonTest(polygon.astype(np.float32), point, False) >= 0


def _gate_order(cells: Iterable[tuple[int, int]], source: set[tuple[int, int]],
                target: set[tuple[int, int]]) -> bool:
    started = False
    for cell in cells:
        if cell in source:
            started = True
        elif started and cell in target:
            return True
    return False


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, values: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as sink:
        for value in values:
            sink.write(json.dumps(value) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    fields = fields or (list(rows[0]) if rows else [])
    with path.open("w", newline="", encoding="utf-8") as sink:
        writer = csv.DictWriter(sink, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _diagnostic_config(config: AppConfig, scope: str) -> DirectionalGridConfig:
    updates: dict[str, Any] = {
        "route_scope": scope,
        "diagnostics_enabled": True,
    }
    if scope == "local_corridor":
        # Cache evidence showed no gate connectivity at 3, but connected seeds at 2.
        # Complete-track, coverage, support, and hysteresis thresholds remain unchanged.
        updates["min_edge_unique_tracks"] = 2
    return config.analytics.directional_grid.model_copy(update=updates)


def _draw_corridor(frame: np.ndarray, corridor: LocalCorridorConfig) -> None:
    height, width = frame.shape[:2]
    roi = _normalized_polygon(corridor.roi, width, height, corridor.coordinate_space)
    source = _normalized_polygon(corridor.source_gate, width, height, corridor.coordinate_space)
    target = _normalized_polygon(corridor.target_gate, width, height, corridor.coordinate_space)
    overlay = frame.copy()
    cv2.fillPoly(overlay, [roi], (90, 90, 15))
    cv2.addWeighted(overlay, .12, frame, .88, 0, frame)
    cv2.polylines(frame, [roi], True, (0, 215, 255), 2, cv2.LINE_AA)
    cv2.polylines(frame, [source], True, (255, 180, 0), 3, cv2.LINE_AA)
    cv2.polylines(frame, [target], True, (80, 230, 80), 3, cv2.LINE_AA)
    cv2.putText(frame, "S", tuple(source[0]), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (255, 180, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "T", tuple(target[0]), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (80, 230, 80), 2, cv2.LINE_AA)


def _draw_debug(frame: np.ndarray, engine: DirectionalGridEngine,
                points: list[SimpleNamespace], frame_id: int, timestamp: float) -> np.ndarray:
    rendered = frame.copy()
    corridor = engine.config.local_corridor
    if corridor is not None:
        _draw_corridor(rendered, corridor)
    height, width = rendered.shape[:2]
    cell_width, cell_height = width / engine.config.columns, height / engine.config.rows
    if engine.valid_cells:
        for row, col in engine.valid_cells:
            x1, y1 = round(col * cell_width), round(row * cell_height)
            x2, y2 = round((col + 1) * cell_width), round((row + 1) * cell_height)
            cv2.rectangle(rendered, (x1, y1), (x2, y2), (105, 105, 105), 1)
    evaluations = {item["evaluation_id"]: item for item in engine.candidate_evaluations[-20:]}
    colors = ((255, 80, 80), (40, 220, 255), (255, 80, 220))
    for index, route in enumerate(engine.latest_routes[:3]):
        route_points = np.asarray([
            (round((cell[1] + .5) * cell_width), round((cell[0] + .5) * cell_height))
            for cell in route.cells
        ], dtype=np.int32)
        if len(route_points) < 2:
            continue
        color = colors[index % len(colors)]
        cv2.polylines(rendered, [route_points], False, color, 3, cv2.LINE_AA)
        evaluation = evaluations.get(route.evaluation_id, {})
        label = (f"{route.candidate_id[-8:]} {route.direction} "
                 f"sup={route.support} complete={route.complete} "
                 f"{evaluation.get('primary_reason', 'UNEVALUATED')}")
        anchor = tuple(route_points[min(1, len(route_points)-1)])
        cv2.putText(rendered, label, anchor, cv2.FONT_HERSHEY_SIMPLEX, .45,
                    color, 2, cv2.LINE_AA)
    for item in points:
        cv2.circle(rendered, (round(item.x), round(item.y)), 3, (0, 255, 255), -1,
                   cv2.LINE_AA)
        cv2.putText(rendered, str(item.track_id), (round(item.x)+3, round(item.y)-3),
                    cv2.FONT_HERSHEY_SIMPLEX, .32, (255, 255, 255), 1, cv2.LINE_AA)
    graph = engine.graph_diagnostics[-1] if engine.graph_diagnostics else {}
    status = graph.get("primary_reason") or "CANDIDATES_EVALUATED"
    cv2.rectangle(rendered, (0, 0), (width, 56), (15, 15, 15), -1)
    cv2.putText(rendered, f"DEBUG REPLAY frame={frame_id} media={timestamp:.2f}s scope=local_corridor",
                (12, 22), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(rendered,
                f"{status} edges={graph.get('directed_edges_after_filter', 0)} "
                f"seeds={graph.get('start_seeds', 0)} routes={graph.get('candidate_routes_found', 0)}",
                (12, 46), cv2.FONT_HERSHEY_SIMPLEX, .5, (180, 220, 255), 1, cv2.LINE_AA)
    return rendered


def _update_track_audit(audits: dict[tuple[int, int], _TrackAudit],
                        segment_state: dict[int, int], tracks: list[TrackedObject],
                        frame_id: int, timestamp: float, transformer: SpatialTransformer,
                        grid: DirectionalGridConfig) -> None:
    cell_width = transformer.width / grid.columns
    cell_height = transformer.height / grid.rows
    for track in tracks:
        segment_id = segment_state.get(track.track_id, 0)
        audit = audits.get((track.track_id, segment_id))
        if audit is None:
            audit = _TrackAudit(track.track_id, segment_id, frame_id, frame_id,
                                timestamp, timestamp)
            audits[(track.track_id, segment_id)] = audit
        audit.last_frame, audit.last_time_s = frame_id, timestamp
        if not track.observed:
            audit.prediction_frames += 1
            continue
        x, y = track.bottom_center
        if audit.last_position is not None and audit.previous_observed_s is not None:
            gap = timestamp - audit.previous_observed_s
            step_cells = math.hypot((x-audit.last_position[0])/cell_width,
                                    (y-audit.last_position[1])/cell_height)
            if gap > grid.max_observation_gap_seconds or step_cells > grid.max_step_cells:
                segment_id += 1
                segment_state[track.track_id] = segment_id
                audit = _TrackAudit(track.track_id, segment_id, frame_id, frame_id,
                                    timestamp, timestamp)
                audits[(track.track_id, segment_id)] = audit
        audit.observed_frames += 1
        if audit.first_observed_frame is None:
            audit.first_observed_frame = frame_id
            audit.first_observed_s = timestamp
            audit.first_position = (x, y)
        elif audit.second_position is None:
            audit.second_position = (x, y)
        if audit.last_position is not None and audit.previous_observed_s is not None:
            gap = timestamp - audit.previous_observed_s
            audit.max_observation_gap_s = max(audit.max_observation_gap_s, gap)
            step_cells = math.hypot((x-audit.last_position[0])/cell_width,
                                    (y-audit.last_position[1])/cell_height)
            if gap <= grid.max_observation_gap_seconds and step_cells <= grid.max_step_cells:
                audit.valid_distance_px += math.hypot(x-audit.last_position[0],
                                                      y-audit.last_position[1])
        audit.previous_position = audit.last_position
        audit.last_position = (x, y)
        audit.previous_observed_s = timestamp
        audit.last_observed_frame = frame_id
        audit.last_observed_s = timestamp
        cell = transformer.grid_cell(x, y, grid_width=grid.columns, grid_height=grid.rows)
        if cell is not None and (not audit.cells or audit.cells[-1] != cell):
            audit.cells.append(cell)


def _tracking_outputs(audits: dict[tuple[int, int], _TrackAudit], frame_count: int, fps: float,
                      active_counts: list[int], observed_counts: list[int],
                      source_cells: set[tuple[int, int]], target_cells: set[tuple[int, int]],
                      output: Path) -> tuple[dict[str, Any], dict[str, str]]:
    rows, corridor_classes = [], {}
    for (track_id, segment_id), audit in sorted(audits.items()):
        observed_span = (audit.last_observed_s - audit.first_observed_s
                         if audit.first_observed_s is not None and audit.last_observed_s is not None
                         else 0.)
        total = audit.observed_frames + audit.prediction_frames
        start_censored = audit.first_frame == 0
        end_censored = audit.last_frame == frame_count - 1
        s_to_t = _gate_order(audit.cells, source_cells, target_cells)
        t_to_s = _gate_order(audit.cells, target_cells, source_cells)
        if s_to_t:
            corridor_class = "S_TO_T_COMPLETE"
        elif t_to_s:
            corridor_class = "T_TO_S_COMPLETE"
        elif any(cell in source_cells | target_cells for cell in audit.cells):
            corridor_class = "PARTIAL_GATE_EVIDENCE"
        else:
            corridor_class = "OUTSIDE_OR_NO_GATE_EVIDENCE"
        track_key = f"cam01:cache-epoch:{track_id}:{segment_id}"
        corridor_classes[track_key] = corridor_class
        rows.append(dict(
            camera_id="cam01", stream_epoch="cache-epoch", track_id=track_id,
            segment_id=segment_id,
            first_observed_time_s=audit.first_observed_s,
            last_observed_time_s=audit.last_observed_s,
            observed_span_seconds=round(observed_span, 4),
            observed_frames=audit.observed_frames,
            prediction_only_frames=audit.prediction_frames,
            observation_ratio=round(audit.observed_frames/max(1, total), 4),
            max_observation_gap_seconds=round(audit.max_observation_gap_s, 4),
            valid_distance_pixels=round(audit.valid_distance_px, 3),
            entry_cell=str(audit.cells[0]) if audit.cells else "",
            exit_cell=str(audit.cells[-1]) if audit.cells else "",
            termination_reason=("clip_end_censored" if end_censored else "tracker_disappeared"),
            start_censored=start_censored, end_censored=end_censored,
            corridor_class=corridor_class,
        ))
    _write_csv(output / "tracking_quality.csv", rows)
    spans = [float(row["observed_span_seconds"]) for row in rows]
    gaps = [float(row["max_observation_gap_seconds"]) for row in rows]
    new_bins: Counter[int] = Counter(int(audit.first_time_s // 10) for audit in audits.values())
    classes = Counter(corridor_classes.values())
    summary = dict(
        distinct_track_ids=len({audit.track_id for audit in audits.values()}),
        distinct_track_keys=len(rows),
        denominator_rule="all gap/jump-segmented track keys in one camera epoch",
        observed_span_median_s=_percentile(spans, 50), observed_span_p95_s=_percentile(spans, 95),
        shorter_than_1s=dict(count=sum(value < 1 for value in spans), denominator=len(spans),
                              ratio=round(sum(value < 1 for value in spans)/max(1, len(spans)), 4)),
        shorter_than_2s=dict(count=sum(value < 2 for value in spans), denominator=len(spans),
                              ratio=round(sum(value < 2 for value in spans)/max(1, len(spans)), 4)),
        active_tracks_per_frame=dict(median=_percentile(active_counts, 50),
                                     p95=_percentile(active_counts, 95), max=max(active_counts)),
        observed_tracks_per_frame=dict(median=_percentile(observed_counts, 50),
                                       p95=_percentile(observed_counts, 95), max=max(observed_counts)),
        new_tracks_per_10_seconds={f"{start*10}-{start*10+10}": count
                                   for start, count in sorted(new_bins.items())},
        max_observation_gap_median_s=_percentile(gaps, 50),
        max_observation_gap_p95_s=_percentile(gaps, 95),
        clip_start_censored=sum(bool(row["start_censored"]) for row in rows),
        clip_end_censored=sum(bool(row["end_censored"]) for row in rows),
        completed_before_clip_end=sum(not bool(row["end_censored"]) for row in rows),
        corridor_classes=dict(classes),
        corridor_complete_tracks=classes["S_TO_T_COMPLETE"] + classes["T_TO_S_COMPLETE"],
        corridor_complete_ratio=round(
            (classes["S_TO_T_COMPLETE"] + classes["T_TO_S_COMPLETE"])/max(1, len(rows)), 4
        ),
        fps=fps,
        caution="Observed spans are clip-bounded observations, not true person lifetimes.",
    )
    _write_json(output / "tracking_summary.json", summary)
    return summary, corridor_classes


def _suspected_switches(audits: dict[tuple[int, int], _TrackAudit], fps: float, input_path: Path,
                        output: Path) -> list[dict[str, Any]]:
    starts: dict[int, list[_TrackAudit]] = defaultdict(list)
    for audit in audits.values():
        if audit.first_observed_frame is not None:
            starts[audit.first_observed_frame].append(audit)
    candidates = []
    max_frames = max(1, round(.65 * fps))
    for before in audits.values():
        if before.last_observed_frame is None or before.last_position is None:
            continue
        for frame in range(before.last_observed_frame + 1, before.last_observed_frame + max_frames + 1):
            for after in starts.get(frame, ()):
                if after.track_id == before.track_id or after.first_position is None:
                    continue
                distance = math.dist(before.last_position, after.first_position)
                if distance > 80:
                    continue
                alignment = None
                if before.previous_position and after.second_position:
                    a = np.subtract(before.last_position, before.previous_position)
                    b = np.subtract(after.second_position, after.first_position)
                    norm = float(np.linalg.norm(a) * np.linalg.norm(b))
                    alignment = float(np.dot(a, b)/norm) if norm > 1e-6 else None
                    if alignment is not None and alignment < -.2:
                        continue
                delta = (after.first_observed_frame - before.last_observed_frame) / fps
                candidates.append((distance + delta*40, before, after, distance, delta, alignment))
    candidates.sort(key=lambda item: item[0])
    selected, used = [], set()
    for _, before, after, distance, delta, alignment in candidates:
        pair = (before.track_id, after.track_id)
        if pair in used:
            continue
        used.add(pair)
        selected.append(dict(
            label="suspected_id_switch", from_track_id=before.track_id,
            from_segment_id=before.segment_id, to_track_id=after.track_id,
            to_segment_id=after.segment_id, from_frame=before.last_observed_frame,
            to_frame=after.first_observed_frame, from_time_s=before.last_observed_s,
            to_time_s=after.first_observed_s, temporal_gap_s=round(delta, 4),
            endpoint_distance_px=round(distance, 3),
            direction_cosine=round(alignment, 4) if alignment is not None else None,
            reason="track ended and another began nearby with compatible direction",
            ground_truth=False,
        ))
        if len(selected) >= 12:
            break
    switch_dir = output / "suspected_id_switches"
    switch_dir.mkdir(exist_ok=True)
    cap = cv2.VideoCapture(str(input_path))
    montage = []
    for index, item in enumerate(selected):
        images = []
        for kind in ("from", "to"):
            frame_id = int(item[f"{kind}_frame"])
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ok, frame = cap.read()
            if not ok:
                continue
            audit = audits[(int(item[f"{kind}_track_id"]),
                            int(item[f"{kind}_segment_id"]))]
            position = audit.last_position if kind == "from" else audit.first_position
            assert position is not None
            resized = cv2.resize(frame, (640, 360))
            point = (round(position[0]/frame.shape[1]*640), round(position[1]/frame.shape[0]*360))
            cv2.circle(resized, point, 10, (0, 0, 255), 3, cv2.LINE_AA)
            cv2.putText(resized, f"{kind} ID={audit.track_id} frame={frame_id}", (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 2, cv2.LINE_AA)
            images.append(resized)
        if len(images) == 2:
            combined = np.hstack(images)
            relative = f"suspected_id_switches/switch-{index+1:02d}.jpg"
            cv2.imwrite(str(output / relative), combined)
            item["image"] = relative
            if len(montage) < 6:
                montage.append(cv2.resize(combined, (960, 270)))
    cap.release()
    if montage:
        cv2.imwrite(str(output / "suspected_id_switches_contact_sheet.jpg"), np.vstack(montage))
    result = dict(
        classification="suspected_id_switch", ground_truth_available=False,
        selection_rule="end-to-start <=0.65s, <=80px, non-opposing direction when measurable",
        count=len(selected), samples=selected,
        limitation="Proximity is ambiguous; IDs are not merged and this is not an IDF1/HOTA/MOTA score.",
    )
    _write_json(output / "suspected_id_switches.json", result)
    return selected


def _funnel(engine: DirectionalGridEngine, variant: str) -> dict[str, Any]:
    evaluations = list(engine.candidate_evaluations)
    decisions = Counter(item["decision"] for item in evaluations)
    primary = Counter(item["primary_reason"] for item in evaluations)
    secondary = Counter(reason for item in evaluations
                        for reason in item["evaluated_secondary_reasons"])
    generated = len(evaluations) + engine.diagnostic_dropped
    reconciled = sum(engine.candidate_decision_counts.values())
    return dict(
        variant=variant, unique_routes=len({item["candidate_id"] for item in evaluations}),
        generated_evaluations=generated, rejected=decisions["rejected"],
        pending=decisions["pending"], eligible_not_selected=decisions["eligible_not_selected"],
        selected=decisions["selected"], reconciled_total=reconciled,
        reconciliation_ok=generated == reconciled,
        dropped_evaluation_records=engine.diagnostic_dropped,
        primary_reason_counts=dict(primary), secondary_reason_counts=dict(secondary),
        graph_primary_reason_counts=dict(Counter(
            item["primary_reason"] or "CANDIDATES_EVALUATED" for item in engine.graph_diagnostics
        )),
    )


def run_diagnostics(input_path: Path, cache_path: Path, config_path: Path,
                    output: Path, run_id: str, *, execution_host: str = "local_cpu",
                    gpu_device: str | None = None) -> dict[str, Any]:
    if output.joinpath("manifest.json").exists():
        raise FileExistsError(f"Refusing to overwrite diagnostic run: {output}")
    output.mkdir(parents=True, exist_ok=True)
    baseline_dir, after_dir = output / "baseline", output / "after"
    baseline_dir.mkdir(exist_ok=True)
    after_dir.mkdir(exist_ok=False)
    config = load_config(config_path)
    corridor = config.analytics.directional_grid.local_corridor
    if corridor is None:
        raise ValueError("A local corridor definition is required")
    meta_path = cache_path.with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    source_hash = _digest(input_path)
    if meta["source_hash"] != source_hash:
        raise ValueError("Source video hash does not match immutable cache metadata")
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot decode source video: {input_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    transformer = SpatialTransformer.pixel(width, height)
    baseline_engine = DirectionalGridEngine(
        _diagnostic_config(config, "global_od"), transformer,
        stream_epoch="cache-epoch", run_id=run_id, variant="baseline_global_od",
    )
    after_engine = DirectionalGridEngine(
        _diagnostic_config(config, "local_corridor"), transformer,
        stream_epoch="cache-epoch", run_id=run_id, variant="after_local_corridor",
    )
    active_writer = cv2.VideoWriter(str(output / "active_common_path.mp4"),
                                    cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    debug_writer = cv2.VideoWriter(str(output / "debug_candidates.mp4"),
                                   cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not active_writer.isOpened() or not debug_writer.isOpened():
        raise RuntimeError("Cannot open diagnostic video writers")
    renderer = FrameRenderer(config.visualization)
    overlay = OverlayOptions(tracking=True, points=True, track_ids=False,
                             candidate_paths=False, active_paths=True,
                             direction_arrows=True, debug_metrics=True)
    audits: dict[tuple[int, int], _TrackAudit] = {}
    segment_state: dict[int, int] = {}
    active_counts, observed_counts = [], []
    frame_count = 0
    first_time = last_time = None
    capture_frames: list[np.ndarray] = []
    sample_frames = {0, 360, 720, 1080, 1440, int(meta["frame_count"])-1}
    detector_calls = 0
    started = time.perf_counter()
    try:
        with cache_path.open("r", encoding="utf-8") as cache:
            previous_frame, previous_time = -1, -math.inf
            for line in cache:
                entry = json.loads(line)
                frame_id, timestamp = int(entry["frame_id"]), float(entry["event_time_s"])
                if frame_id != previous_frame + 1 or timestamp < previous_time:
                    raise ValueError("Cache frame IDs/timestamps are not continuous and monotonic")
                ok, frame = cap.read()
                if not ok:
                    raise ValueError(f"Source decode ended before cached frame {frame_id}")
                tracks = [TrackedObject(**item) for item in entry["tracks"]]
                points = [GridTrackPoint("cam01", "cache-epoch", track.track_id, 0,
                                         frame_id, timestamp, *track.bottom_center,
                                         observed=track.observed) for track in tracks]
                baseline_engine.update(points, timestamp)
                after_snapshot = after_engine.update(points, timestamp)
                _update_track_audit(audits, segment_state, tracks, frame_id, timestamp, transformer,
                                    config.analytics.directional_grid)
                active_counts.append(len(tracks))
                observed_tracks = [track for track in tracks if track.observed]
                observed_counts.append(len(observed_tracks))
                current_points = [SimpleNamespace(track_id=track.track_id,
                                                  x=track.bottom_center[0], y=track.bottom_center[1])
                                  for track in observed_tracks]
                active = renderer.render_point_only_frame(
                    frame, current_points, after_snapshot, (), transformer, overlay,
                    people_count=len(observed_tracks), processing_fps=0., latency_ms=0.,
                    timestamp=timestamp,
                )
                debug = _draw_debug(frame, after_engine, current_points, frame_id, timestamp)
                active_writer.write(active)
                debug_writer.write(debug)
                if frame_id == round(54 * fps):
                    cv2.imwrite(str(output / "debug_frame_54s.jpg"), debug)
                    cv2.imwrite(str(output / "active_frame_54s.jpg"), active)
                if frame_id in sample_frames:
                    labelled = cv2.resize(active, (width//3, height//3))
                    cv2.putText(labelled, f"frame={frame_id} media={timestamp:.2f}s", (10, 24),
                                cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 2, cv2.LINE_AA)
                    capture_frames.append(labelled)
                previous_frame, previous_time = frame_id, timestamp
                first_time = timestamp if first_time is None else first_time
                last_time, frame_count = timestamp, frame_count + 1
    finally:
        cap.release()
        active_writer.release()
        debug_writer.release()
    if frame_count != int(meta["frame_count"]):
        raise ValueError(f"Cache frame count mismatch: {frame_count} != {meta['frame_count']}")
    if capture_frames:
        rows = [np.hstack(capture_frames[index:index+3])
                for index in range(0, len(capture_frames), 3)]
        cv2.imwrite(str(output / "preview_contact_sheet.jpg"), np.vstack(rows))
    cap = cv2.VideoCapture(str(input_path))
    ok, illustration = cap.read()
    cap.release()
    if ok:
        _draw_corridor(illustration, corridor)
        cv2.imwrite(str(output / "corridor_roi.png"), illustration)
    _write_json(output / "corridor_roi.json", {
        **corridor.model_dump(mode="json"),
        "source_hash": source_hash,
        "selection_before_algorithm_diagnostics": True,
    })
    tracking_summary, _ = _tracking_outputs(
        audits, frame_count, fps, active_counts, observed_counts,
        after_engine._source_cells, after_engine._target_cells, output,
    )
    switches = _suspected_switches(audits, fps, input_path, output)
    evaluations = [*baseline_engine.candidate_evaluations, *after_engine.candidate_evaluations]
    graph = [*baseline_engine.graph_diagnostics, *after_engine.graph_diagnostics]
    _write_jsonl(output / "candidate_evaluations.jsonl", evaluations)
    _write_jsonl(output / "graph_diagnostics.jsonl", graph)
    rejection_rows = [
        dict(variant=variant, reason=reason, count=count)
        for variant, engine in (("baseline_global_od", baseline_engine),
                                ("after_local_corridor", after_engine))
        for reason, count in sorted(engine.tracklet_rejection_counts.items())
    ]
    _write_csv(output / "tracklet_rejection_counts.csv", rejection_rows,
               ["variant", "reason", "count"])
    funnels = [_funnel(baseline_engine, "baseline_global_od"),
               _funnel(after_engine, "after_local_corridor")]
    _write_csv(output / "candidate_funnel.csv", [{
        key: (json.dumps(value, sort_keys=True) if isinstance(value, dict) else value)
        for key, value in funnel.items()
    } for funnel in funnels])
    _write_json(output / "rejection_summary.json", {item["variant"]: item for item in funnels})
    for directory, engine, funnel in ((baseline_dir, baseline_engine, funnels[0]),
                                      (after_dir, after_engine, funnels[1])):
        _write_json(directory / "diagnostic_summary.json", dict(
            route_scope=engine.config.route_scope, funnel=funnel,
            final_paths=[asdict(path) for path in engine.snapshot().paths],
            directed_edges=len(engine.flow_snapshot().edges),
            retained_points=engine.retained_points,
            support_cap_drops=engine.overflow, rejected_jump_segments=engine.rejected_jumps,
        ))
    local_evaluations = list(after_engine.candidate_evaluations)
    corridor_validation = dict(
        corridor_id=corridor.corridor_id, scope="local_corridor",
        complete_definition="same cache track crosses configured S then T (or T then S) in order",
        observed_track_counts=tracking_summary["corridor_classes"],
        max_candidate_support=max((item["support_tracks"] for item in local_evaluations), default=0),
        max_candidate_complete_tracks=max((item["complete_tracks"] for item in local_evaluations), default=0),
        active_paths=[asdict(path) for path in after_engine.snapshot().paths
                      if path.state in {"active", "cooling"}],
        algorithm_generated_only=True,
    )
    _write_json(output / "corridor_validation.json", corridor_validation)
    resolved = config.model_copy(update={"analytics": config.analytics.model_copy(update={
        "directional_grid": _diagnostic_config(config, "local_corridor")
    })})
    (output / "config_resolved.yaml").write_text(
        yaml.safe_dump(resolved.model_dump(mode="json"), sort_keys=False), encoding="utf-8"
    )
    elapsed = time.perf_counter() - started
    detector_module_loaded = "backend.app.inference" in sys.modules
    if detector_calls != 0 or detector_module_loaded:
        raise RuntimeError("Replay detector guard failed")
    manifest = dict(
        run_id=run_id, status="success", run_type="analytics_replay",
        execution_host=execution_host, gpu_device=gpu_device,
        stage_devices=dict(cache_read="cpu", tracking="cached_tracklets", analytics="cpu",
                           rendering="cpu", encoding="cpu_opencv_mp4v"),
        detector_calls=detector_calls, detector_module_loaded=detector_module_loaded,
        inference_executed=False, inference_timing_ms=None,
        source=str(input_path), source_hash=source_hash,
        cache=str(cache_path), cache_key=meta["cache_key"], cache_source_hash=meta["source_hash"],
        cache_model_hash=meta["model_hash"], cache_immutable=True,
        cache_schema="detections-tracklets-v1-with-observed-flag",
        frame_count=frame_count, fps=fps, media_start_s=first_time, media_end_s=last_time,
        coordinate_space="image_pixels", clock="media_event_time",
        variants=["baseline_global_od", "after_local_corridor"],
        candidate_funnels=funnels, tracking_summary=tracking_summary,
        suspected_id_switch_samples=len(switches), corridor_validation=corridor_validation,
        runtime_seconds=round(elapsed, 3), process_ram_mb=round(psutil.Process().memory_info().rss/1024**2, 2),
        versions=dict(python=sys.version.split()[0], opencv=cv2.__version__, numpy=np.__version__,
                      pydantic=importlib.metadata.version("pydantic")),
    )
    _write_json(output / "manifest.json", manifest)
    artifacts = sorted(str(path.relative_to(output)).replace("\\", "/")
                       for path in output.rglob("*") if path.is_file())
    manifest["artifacts"] = artifacts
    _write_json(output / "manifest.json", manifest)
    (output / "inspection.md").write_text(
        "# Diagnostic artifact inspection\n\nStatus: NOT_REVIEWED\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    print(json.dumps(run_diagnostics(args.input, args.cache, args.config,
                                     args.output, args.run_id), indent=2))


if __name__ == "__main__":
    main()
