# Realtime Common Path point-only UI report

## Root cause

The analytics implementation already used bottom-center points, bounded tracklets, unique-track
directed flow, short/long windows and stable Common Path states. The remaining production mismatch
was in delivery: realtime and batch renderers behaved differently, batch video always drew
individual tails and candidates, zones defaulted on, and raw snapshot polylines could change in one
frame. Visualization defaults were duplicated in Python and React and there was no configuration
API.

## Implemented

- Production renderer input is the raw frame, a current-point snapshot, confirmed Common Path
  snapshot, metrics and optional zones. Full trajectories are supplied only when the individual
  path debug option is enabled.
- Default output has no boxes, IDs, individual paths, zones, grid, edge flow or candidates.
  Tracking points, active/cooling confirmed paths, direction arrows and metrics are enabled.
- Confirmed paths use a translucent corridor, bounded-width centerline, spaced arrows, a short
  origin/destination/support label and a 500 ms resampled smooth transition.
- `GET/PATCH /api/config/visualization` is the shared source for backend and frontend settings.
  WebSocket snapshots contain only current points, active/cooling Common Paths, metrics and
  visualization state; no trajectory history is serialized.
- The batch command uses the same renderer and writes `realtime_point_common_path.mp4`.

## Changed areas

The primary changes are in `backend/app/video/renderer.py`, `backend/app/core/config.py`,
`backend/app/core/session.py`, `backend/app/api/app.py`, `scripts/process_point_tracks.py`, the
React visualization types/API/video controls, YAML configuration, artifact contract and renderer,
API, config and artifact tests.

## Tested

The complete suite has 79 passing tests. Coverage includes bottom-center/EMA, no bounding box,
no production individual trajectory, active rendering, candidate hiding, bidirectional unique-ID
flow, stationary/short-track filtering, cell jitter, promotion/cooling, 500 ms display transition,
short detection loss, bounded buckets, slow analytics isolation, video metadata and compact
WebSocket payload. The frontend production build and targeted Ruff checks pass.

The real output was visually inspected at 0.8 and 19 seconds. Before confirmation it contains
only current points and a collecting state. After confirmation it contains one green Common Path
corridor with arrows and 12-ID support; two internal candidates in `common_paths.json` are absent
from production video.

## Real-video metrics

Both matched rows use the same first 180 frames of `data/videos/data-test.mp4`, YOLO26n,
ByteTrack and CPU host. Runtime variation is dominated by inference, so changes are descriptive,
not an algorithmic speedup claim.

| Metric | Before | After point-only |
| --- | ---: | ---: |
| Processing FPS | 1.9855 | 2.1571 |
| Inference | 465.9661 ms | 426.1764 ms |
| Tracking | 37.6922 ms | 37.3969 ms |
| Analytics | 2.9329 ms | 2.8822 ms |
| Render | not recorded; isolated median 10.562 ms | 6.1846 ms average; isolated median 11.031 ms |
| Isolated render p95 | 15.884 ms | 13.681 ms |
| E2E | 506.5967 ms, excluded render/encode | 505.3080 ms, includes 6.1846 ms render and 32.6628 ms encode |
| RAM | 403.78 MB | 415.88 MB |
| CPU | 459.2% multi-core | 498.8% multi-core |
| GPU | unavailable | unavailable |

The final 600-frame run decoded 20 seconds at 1920x1080 and 30 FPS with zero dropped offline
frames. It measured 2.6544 processing FPS, 6.7936 ms render, 30.2990 ms video encode,
416.8526 ms E2E, 8.738/18.821 ms Common Path median/p95, 422.02 MB RAM and 614.0%
multi-core CPU. One active path appeared at 18 seconds with 12 unique tracks, score 1.0,
confidence 0.936497 and zero switches (0/minute). The first candidate appeared at 9 seconds.

## Selected parameters

- Grid: 8x5 macro cells
- Short/long windows: 30/180 seconds with 0.40/0.60 weights
- Switch margin: 20%
- Confirmation/cooling: 8/20 seconds
- Path transition: 500 ms
- Centerline width: 6-18 px; arrow spacing: 80 px
- Point radius: 5 px

## Outputs

The verified production video is `outputs/realtime_point_common_path.mp4`. Its full companion
artifact set is under `outputs/point-only-common-path/`, including trajectories, directed flows,
Common Path JSON/timeline/map and realtime benchmark. `ffprobe` confirmed 600 frames, 1920x1080,
30 FPS and 20.0 seconds.

## Limitations and next priorities

The CPU host cannot process this 30 FPS source in realtime; live mode remains bounded and drops
stale frames instead of accumulating latency. The 20-second clip does not fill the 180-second
window and contains no labeled sustained flow reversal. It demonstrates initial candidate at
9 seconds and promotion at 18 seconds, but no real-video flow-change response time can be claimed.
State-change behavior and 500 ms visual interpolation are covered deterministically. Next steps
are a multi-minute labeled flow-change recording, GPU benchmark, RTSP reconnect exercise and
overnight bounded-memory soak.
