# Crowd Analysis Common Path Refactor

## Result

The production profiles now run this path:

```text
Video source -> bounded capture queue -> person detector -> ByteTrack
  -> bounded trajectory/tracklet state -> tracklet Common Path
  -> solid polyline + tangent arrows -> MJPEG/WebSocket UI
```

`configs/default.yaml` and `configs/shibuya.yaml` select
`tracklet_aggregation`, use `max_paths: 1`, and disable auxiliary occupancy,
flow, zone, directional-grid, and DD-CRP processing. A caller can still opt
into the compatibility analyzers with `analytics.auxiliary_analytics_enabled`
or select an explicit legacy/grid engine.

The default UI follows the same contract. It renders the video, tracking
points, Common Path, arrows, and metrics. Heatmap, vector-flow, timeline, and
zone panels are hidden when the backend reports `common_path_only: true`; the
existing endpoints remain available for compatibility configurations.

## Common Path evidence

`TrackletAggregationEngine` keeps camera/session/track/frame/timestamp/point
observations in bounded active and closed segment buffers. It rejects invalid,
stationary, short, large-jump, zig-zag, and direction-inconsistent segments.
Compatible segments are linked by direction, overlap or directed endpoint
continuity, then merged into an ordered polyline and resampled by arc length.
No grid cell or heatmap value is used by this engine.

Each candidate contributes at most one support unit per temporary Track ID. Its
score is:

```text
unique_track_ids * (0.55 + 0.45 * temporal_coverage)
                  * (0.5 + 0.5 * mean_tracklet_quality)
```

`temporal_coverage` is capped from the number of valid tracklets relative to
the configured support threshold. This is evidence for an anonymous route,
not a claim that the same unique people completed the entire polyline.
Path IDs and colors are retained through EMA updates; candidates are promoted
after confirmation, can cool during a short evidence gap, and are retired
after the configured route-memory TTL. The renderer draws the representative
polyline and repeated tangent arrows in the observed travel direction.

## Changes

- `backend/app/analytics/engine.py`: conditionally constructs auxiliary
  analyzers; the tracklet engine is the only production Common Path owner;
  compatibility endpoints return bounded empty snapshots when those analyzers
  are disabled; tracklet compute/buffer/rejection metrics are exposed.
- `backend/app/analytics/tracklet_aggregation.py`: removes the dependency on
  the directional-grid module by introducing the grid-free `TrackletPoint`
  transport type and exposes bounded-buffer size.
- `scripts/common_path_clip.py`: only allocates the selected engine and only
  emits directional debug artifacts for an explicit directional-grid run.
- `scripts/live_common_path.py` and `scripts/replay_tracklet.py`: use the
  grid-free tracklet point type for tracklet runs.
- `configs/default.yaml`, `configs/shibuya.yaml`, `modal_common_path.py`, and
  the clip CLI: default to tracklet Common Path and Top 1; Shibuya DD-CRP is
  disabled in the production profile.
- `backend/app/core/session.py` and frontend components: expose the runtime
  contract and suppress dead auxiliary panels in Common Path-only mode.

No face, recognition, ReID, gallery, vector-store, TensorRT, or personal
identity code was added or loaded. Existing optional legacy/grid modules were
left intact because they are still referenced by compatibility scripts and
tests.

## Verification

Commands run:

```text
PYTHONPATH=. python -m pytest -q       128 passed
npm run build                          passed
python -m compileall -q backend scripts modal_common_path.py
git diff --check                       passed
```

The repository does not contain a fixed benchmark video/GPU execution in this
workspace suitable for a before/after production comparison. Therefore no FPS,
CPU/RAM/GPU, latency, or frame-drop improvement is claimed here. The live
pipeline already records decode, inference, tracking, analytics, rendering,
encoding, end-to-end p50/p95, queue size, and dropped-frame metrics through
`/metrics`; run the same video, hardware, and configuration before and after
deployment to populate that comparison.

## Run

```powershell
python -m uvicorn backend.main:app --reload --port 8000
cd frontend
npm run dev
```

For offline Common Path replay:

```powershell
python -m scripts.common_path_clip `
  --input data/videos/data-shibuya.mp4 `
  --config configs/shibuya.yaml `
  --output-dir outputs/common-path-refactor `
  --run-id shibuya-common-path
```

Important controls are `video.inference_interval`, `video.queue_size`,
`video.analytics_queue_size`, `analytics.common_path.tracklet_aggregation.*`,
and `analytics.common_path.max_paths` (1 by default; the Modal UI can change
Top K to 2-5 at runtime).

## Remaining limits

Track IDs are temporary per camera/session and can fragment or switch. Pixel
coordinates remain perspective-dependent until calibration is supplied. A
real hardware/video replay is still required to establish capacity and to
validate route changes, opposite-direction crossings, and long-duration TTL
behavior under the target deployment load.
