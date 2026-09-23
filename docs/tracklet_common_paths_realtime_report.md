# Tracklet Common Paths Realtime Report

## REUSED

- `PointTrackletManager`/bottom-center observation contract and `GridTrackPoint` camera + epoch key.
- Existing bounded/cardinality patterns; Common Path matching itself does not use DD-CRP or a grid.
- Existing Modal T4 YOLO/ByteTrack pipeline, binary frame protocol, renderer, replay cache and frontend stream.

## CHANGED

- Added `TrackletAggregationEngine`: bounded observed segments, gap/jump cuts, filtering for
  short/stationary/zig-zag motion, directed temporal/spatial matching, connected-component
  clustering, endpoint stitching, median overlap merge, polyline smoothing, hysteresis, cooling
  and stable path/color IDs. Candidate confirmation is held for 5 seconds by default. Route
  extensions are joined only through overlap or directed endpoint continuity; a weak match never
  shortens the remembered camera-entry-to-exit polyline. Score/confidence use a low-alpha EMA,
  short evidence gaps keep an active path unchanged, and `route_memory_seconds: 120` keeps the
  complete geometry in cooling before retirement.
- `configs/default.yaml` and `configs/shibuya.yaml` select `tracklet_aggregation` with `max_paths: 3`; old `directional_grid` and `dominant_live_flow` remain rollback options.
- Backend snapshot/renderer now carries `color`, `rank`, `support_tracks`, `evidence_until_s` and renders thin solid colored polylines with direction arrows.
- Modal WebSocket accepts `max_paths` at start and `set_max_paths` while running. The frontend control sends and displays the applied value, and the legend uses the backend snapshot.

## VERIFIED

- Narrow tests: `41 passed` across tracklet aggregation, engine selection, live processor, renderer, pipeline and config.
- Frontend `npm run build`: passed. Python compile and `git diff --check`: passed.
- Replay output: `outputs/common_path/tracklet-replay-20260922-final/`.
  - Cache source: `outputs/live-dominant-ui-fix-20260920-235206/tracking_cache.jsonl` + `data/videos/data-shibuya-test.mp4`.
  - 55 frames, 0 inference calls, 37 frames with active paths, 3 active paths at the end.
  - Analytics p50 280.812 ms, p95 846.617 ms.
  - Preview: `preview.mp4`; evidence timeline: `path_timeline.jsonl`; metrics: `metrics.json`; UI frame: `ui_common_path.jpg`.
- Modal GPU run: `outputs/common_path/tracklet-live-20260922-muc1yked-hy07e/`.
  - Tesla T4, 25 seconds, 38 processed preview frames, browser received changing 1280x720 frames.
  - Browser verification: `outputs/common_path/tracklet-browser-20260922/ui_verification.json` (`passed: true`). K=1/3/5 controls were acknowledged in the same stream; no console, HTTP or WebSocket request errors.
  - Source processing was 1.493 FPS with 712 dropped source frames; analytics p50 0.509 ms / p95 21.468 ms on the live run.

## LIMITATIONS

- The 25-second realtime clip did not accumulate three locally supported observed tracks for an active path: live `candidate_count: 2`, `active_paths: []`, primary rejection `INSUFFICIENT_LOCAL_SUPPORT` (15). This is reported as no evidence, not replaced by a synthetic overlay.
- Replay proves geometry and renderer behavior on the compatible cache; live proves Modal/backend/UI timing and K control, not long-term path quality.
- Pixel coordinates are normalized for matching but rendered in source pixels; perspective calibration and tracker ID fragmentation remain domain limitations.

## RUN / ROLLBACK

```powershell
python -m scripts.replay_tracklet `
  --cache outputs/live-dominant-ui-fix-20260920-235206/tracking_cache.jsonl `
  --input data/videos/data-shibuya-test.mp4 `
  --output-dir outputs/common_path/tracklet-replay-local
```

```powershell
$env:PYTHONIOENCODING = "utf-8"
$env:LIVE_SESSION_TOKEN = "<ephemeral-secret>"
modal serve modal_shibuya_live.py
# Set VITE_MODAL_LIVE_WS_URL, VITE_MODAL_LIVE_TOKEN, then:
npm --prefix frontend run dev -- --port 5176
```

For rollback, set `analytics.common_path.engine: directional_grid` and
`analytics.dominant_live_flow.mode: dominant_live_flow`, then start a new session. This resets
analytics state only; detector, precision and tracker are unchanged. Changing only `max_paths`
does not restart the session.
