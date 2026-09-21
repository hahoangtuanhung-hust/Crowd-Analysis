# Common Path Realtime - implementation report

## Root cause

The previous popular-path path used per-point edge counts and recomputed DDCRP output on every
analytics frame. Long or stationary tracks could dominate through repeated points, route rank
changes also changed IDs, and DDCRP retained growing tracklet/cluster state. There was no
short/long time-window lifecycle for a stable candidate, active, cooling, or retired path.

The pre-change 180-frame replay of `data/videos/data-test.mp4` measured 2.655 processing FPS,
66.069 ms median / 130.167 ms p95 analytics time, and 53.03 MB RSS growth. In six seconds DDCRP
retained 331 tracklets and created 10,776 cluster slots. The pre-change suite had 57 passing tests.

## Implemented architecture

The existing OpenCV -> YOLO26n -> ByteTrack pipeline remains intact. Analytics still uses the
EMA-smoothed bounding-box bottom center. Detection boxes are disabled in the default overlay.

`CommonPathAnalyzer` adds bounded per-track lifecycle state, grid-cell and zone hysteresis,
unique-Track-ID directed edges, one-second buckets, decayed short/long windows, positive-cost
Dijkstra extraction, and a candidate/active/cooling/retired state machine. Similar routes merge
by directed-edge overlap and update their centerline with EMA. Common Path calculation runs at a
three-second interval in the bounded analytics worker; the detector/tracker does not wait for it.
The older DDCRP path is retained as an opt-in compatibility feature and is disabled by default.

The API exposes `/api/analytics/common-paths`, `/api/analytics/flows`,
`/api/analytics/zones`, and `/api/analytics/metrics`; `/ws/live` publishes a compact snapshot once
per second. The UI includes independent point, ID, tail, zone, grid, edge-flow, candidate, active,
heatmap, and metrics controls. Active paths are green, candidates yellow, and cooling/retired paths
gray. Track IDs and tails default off so the active route stays readable in a crowded scene.

## Algorithm and selected parameters

The supplied Grand Central view uses an 8x5 macro grid, 1-second buckets, 30-second short and
180-second long windows, 0.40/0.60 weights, minimum 5 unique tracks, minimum edge support 3,
20% switch margin, 8-second confirmation, and 20-second cooling. These are configuration values,
not hard-coded algorithm constants.

The real 20-second trajectory replay compared short windows 15/30/60 seconds, long windows
120/180/300 seconds, margins 0.10/0.20/0.30, confirmations 3/8/15 seconds, and grids 8x5,
16x9, 32x18, and 64x36. The selected profile reached one active route at 18 seconds with zero
switches and 8.478 ms compute p95 after warm-up. A 3-second confirmation promoted two routes at
12 seconds; 15 seconds did not promote within the clip. Finer grids did not accumulate sufficient
connected support at the required thresholds. Full results are in
`outputs/common-path-realtime/parameter_benchmark.csv`.

## Tests

The final suite has 73 passing tests. It covers all 15 required behaviors, including direction,
unique-ID edge counting, short/stationary tracks, cell jitter, promotion, switching, cooling,
merging, window expiry, bounded buckets, empty video, flush, and slow-analytics isolation. The
frontend production build and Ruff checks pass. Deterministic tests validate state-machine cases;
performance conclusions below use only the real project video.

## Real-video result

The final run decoded exactly 600 frames (20.0 seconds), 1920x1080 at 30 FPS. It produced one
active `north_west_passage -> central_concourse` path with 12 unique tracks, score 1.0 and
confidence 0.936497, plus two candidates. First candidate appeared at 9 seconds, the active path
at 18 seconds, and there were zero switches or candidate rejections.

| Metric | Result |
| --- | ---: |
| Processing FPS | 2.9742 |
| Inference / tracking / analytics | 295.7149 / 40.5032 / 2.7855 ms |
| Common Path median / p95 | 9.316 / 18.144 ms |
| End-to-end latency | 339.008 ms |
| Dropped frames | 0 |
| Completed / valid tracks | 642 / 78 |
| CPU / RAM | 648.5% multi-core / 413.26 MB |
| GPU memory | unavailable on this host |

The default profile meets the initial Common Path p95 target below 20 ms. End-to-end processing
does not meet the 30 FPS input rate on this CPU. Offline file mode uses backpressure and preserved
all 600 frames; live mode uses the bounded latest-frame policy and drops stale frames rather than
allowing latency or memory to grow.

## Outputs

The verified non-empty artifacts are `tracked_points.mp4`, `trajectories.csv`,
`edge_flows.json`, `common_paths.json`, `common_path_map.png`,
`common_path_timeline.csv`, `realtime_benchmark.csv`, `frame_metrics.csv`, `heatmap.png`,
`path_map.png`, and `zone_flows.json` under `outputs/common-path-realtime/`. `ffprobe` confirmed
the generated video contains 600 frames. `frame_19s.png` is the inspected visual-QA frame.

## Known limitations and next priorities

The measured clip is only 20 seconds, so it does not fill the 180-second window or constitute a
long RTSP memory soak. It also contains no verified sustained real-world flow reversal; response
and cooling transitions are covered by deterministic tests, not claimed as a real-video result.
The configured polygons are camera-specific areas rather than ground-truth entry/exit annotations,
and tracking accuracy was not scored against labels. The next validation should use a multi-minute
RTSP/video sequence with a documented flow change, GPU measurements, and an overnight bounded-
memory soak before production capacity is set.
