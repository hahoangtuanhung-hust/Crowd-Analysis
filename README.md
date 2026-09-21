# Crowd Analysis Web Demo

Anonymous, single-camera crowd-flow analysis for MP4 files and RTSP streams. The system detects people, assigns session-local track IDs, extracts bottom-center trajectories, and produces occupancy/movement heatmaps, flow direction, popular routes, crowd timeline and zone statistics in a realtime web dashboard.

No face recognition, identity matching or cross-camera re-identification is implemented.

## Quick start with Docker

Prerequisites: Docker Desktop with Compose v2 and internet access on the first processing run so Ultralytics can download `yolo26n.pt`.

```bash
git clone <repository-url> crowd-analysis
cd crowd-analysis
cp .env.example .env
docker compose up --build
```

Open [http://localhost:8080](http://localhost:8080), choose `data/videos/example-people.mp4`, then press **Start**. The API is also exposed at [http://localhost:8000/docs](http://localhost:8000/docs).

The default image installs CPU PyTorch. On a compatible NVIDIA host with NVIDIA Container Toolkit:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

Set `TORCH_INDEX_URL` in `.env` to the CUDA wheel index compatible with the host driver. The detector uses `device: auto`, so the same application falls back to CPU when CUDA is unavailable.

Model weights (`*.pt`, `*.onnx`, `*.engine`) are intentionally ignored by Git and the Docker build context.

## Development mode

Prerequisites: Python 3.11-3.14 and Node.js 24+.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m uvicorn backend.main:app --reload --port 8000
```

In a second terminal:

```powershell
cd frontend
npm ci
npm run dev
```

Open [http://localhost:5173](http://localhost:5173). Vite proxies REST, WebSocket and MJPEG traffic to FastAPI.

For a non-web batch run:

```powershell
python -m scripts.run_tracking data/videos/example-people.mp4
```

Outputs are written to `data/outputs/`: overlay MP4, track JSONL, occupancy/movement PNGs, flow field and analytics JSON.

## Point-only tracking and route maps

Benchmark the test clip across `imgsz=640/960/1280` and
`confidence=0.05/0.10/0.15`:

```powershell
python -m scripts.benchmark_tracking data/videos/data-test.mp4 --max-frames 150
```

Process the clip with the measured default point-only production configuration:

```powershell
python -m scripts.process_point_tracks data/videos/data-test.mp4 `
  --zones configs/zones.json `
  --output-dir outputs/point-only-common-path `
  --max-frames 600
```

Replay the produced real trajectories across the documented Common Path parameter matrix:

```powershell
python -m scripts.benchmark_common_path
```

For a quick functional run, add `--max-frames 300 --output-dir outputs/smoke`. The command
writes these bounded tracking and Common Path artifacts:

- `common_path_map.png`: latest active/candidate/cooling paths over the camera view;
- `common_path_timeline.csv`: periodic path state, score, confidence and unique-ID support;
- `common_paths.json`: latest immutable Top Common Paths snapshot;
- `edge_flows.json`: directed grid edges with short/long unique-track support;
- `frame_metrics.csv`: per-frame detections, active/new/lost tracks, FPS and inference latency;
- `heatmap.png`: density accumulated from confirmed track points;
- `path_map.png`: completed routes and the top entry/exit flows ranked by unique Track ID;
- `realtime_point_common_path.mp4`: production view with current bottom-center points and confirmed
  Common Paths, without boxes, IDs, individual trajectories, zones, grids or candidates;
- `tracked_points.mp4`: compatibility copy of the point-only production video;
- `trajectories.csv`: confirmed point tracks with confidence and debounced zone membership;
- `realtime_benchmark.csv`: measured stage, queue, path-compute and resource metrics;
- `zone_flows.json`: measured run summary and deduplicated first-zone to last-zone flows;

The Modal entry point uses the same contract and defaults to `data/videos/data-test.mp4`:

```powershell
modal run modal_app.py --output-dir output_modal
```

### Directional Grid Common Path

`analytics.common_path.engine` selects the implementation without changing detection or
tracking: `directional_grid` is the new validated engine, `shadow` runs both engines from the
same observations and displays `shadow_display`, and `legacy` is the rollback setting.

Run the bounded 25-second GPU smoke on Modal (one T4, 600-second timeout):

```powershell
$env:PYTHONUTF8="1"
modal run --quiet modal_common_path.py `
  --input data/videos/data-test.mp4 `
  --start-seconds 0 --duration-seconds 25 `
  --engine shadow --mode offline_fast `
  --config configs/default.yaml `
  --run-id dg-smoke-YYYYMMDD-a --cache-policy reuse
```

The remote job saves a content-addressed detection/track cache and run artifacts in the
`crowd-analysis-data` Volume, then downloads only that run to `outputs/common_path/<run-id>/`.
Changing only grid/path analytics must use `scripts.common_path_clip --replay-cache ...`; that
path refuses a cache whose input/model/detector/tracker key does not match and does no inference.
Use `python -m scripts.replay_ui --cache <tracking_cache.jsonl>` to smoke the MJPEG pipeline from
the Modal cache without loading YOLO locally.

The Shibuya wide-angle view needs the tiled small-person profile. Run it on a GPU because each
frame includes one full-frame pass plus four overlapping tiles:

```powershell
$env:PYTHONUTF8="1"
modal run --quiet --timestamps modal_app.py `
  --video-path data/videos/data-shibuya-test.mp4 `
  --pipeline-type points `
  --config-path configs/shibuya.yaml `
  --zones-path configs/shibuya-zones.json `
  --output-dir output_modal/shibuya-sensitive
```

Use `--max-frames 300` for a short GPU smoke test. The empty Shibuya zone file intentionally
disables Grand Central polygons; add camera-specific polygons before interpreting zone flows.

### Live Shibuya from Modal

The live path runs YOLO, ByteTrack, dominant live-flow analytics and rendering in one ordered
Modal T4 session. The browser receives binary JPEG packets directly over an authenticated
WebSocket; the token below is an ephemeral session token, not a Modal account token.

Terminal 1 (repository root):

```powershell
$env:LIVE_SESSION_TOKEN = [Convert]::ToHexString([Security.Cryptography.RandomNumberGenerator]::GetBytes(24)).ToLower()
Write-Host $env:LIVE_SESSION_TOKEN
modal serve modal_shibuya_live.py
```

Copy both the generated session token and the `https://...modal.run` URL printed by Modal.
Change the URL scheme to `wss`, append `/ws/live`, then start the UI in Terminal 2:

Replace the angle-bracket examples below with real values. Do not run them literally: unresolved
`<modal-host>` or `<the-session-token-from-terminal-1>` values are rejected and shown as a UI
configuration error.

```powershell
Set-Location frontend
$env:VITE_MODAL_LIVE_WS_URL = "wss://<modal-host>/ws/live"
$env:VITE_MODAL_LIVE_TOKEN = "<the-session-token-from-terminal-1>"
$env:VITE_MODAL_LIVE_DURATION_SECONDS = "200"
$env:VITE_MODAL_LIVE_PREVIEW_FPS = "10"
$env:VITE_MODAL_LIVE_RUN_ID = "live-dominant-YYYYMMDD-smoke"
npm run dev -- --port 5176
```

Open `http://localhost:5176`, press **Start GPU inference**, and use **Stop** to finalize the
200-second Shibuya run early. For routine smoke tests, temporarily set `duration` to 20-30 seconds.
Each live run performs new CUDA
inference, so replay analytics changes from the downloaded `tracking_cache.jsonl` instead of
starting another GPU session. Stop both terminals after testing; `modal serve` then tears down
the ephemeral deployment.

The live endpoint accepts only `realtime_pts`: playback is paced by source timestamps and stale
input frames are dropped when CUDA inference cannot keep up. The output is stored remotely under
`/root/data/common_path/runs/<run-id>` and should be downloaded to
`outputs/common_path/<run-id>`. Set `analytics.dominant_live_flow.mode` back to
`validated_route` to roll the analytics engine back without changing the streaming transport.

To render the time-varying dominant movement direction from the immutable Shibuya cache on a
Modal T4, without rerunning YOLO:

```powershell
$env:PYTHONUTF8="1"
modal run --quiet --timestamps modal_dominant_path.py `
  --cache-run-id shibuya-live-integration-20260920 `
  --config configs/shibuya.yaml `
  --run-id shibuya-dominant-YYYYMMDD
```

`analytics.directional_grid.display_policy: dominant_direction` ranks recent direction bins by
deduplicated Track ID support in `short_window_seconds`, selects the strongest connected cell
component, and draws its centerline causally on each output frame. Set it back to
`validated_route` to restore the complete-route validator.

`configs/zones.json` uses pixel-space polygons for the fixed 1920x1080 camera. Each zone accepts
the documented `id`, `name`, and `points` fields. Coordinates are scaled when the input resolution
differs. Edit these polygons before using another camera view.

## Dashboard workflow

1. Select **Video** and upload MP4/MOV/AVI/MKV, or select **RTSP** and enter the stream URL.
2. Start the session. File input processes to completion; RTSP remains live and reconnects with exponential backoff.
3. The production view starts with tracking points, confirmed Common Paths, direction arrows and
   metrics. IDs, individual paths, zones, heatmap, grid, edge flow and candidates are opt-in debug
   overlays.
4. Use **Calibrate** to place four ordered ground-plane corners and enter real-world width/height.
5. Use **Zones** to draw and name one or more polygons.
6. Inspect occupancy vs movement heatmaps, current/1m/5m/entire windows, vector field, routes, timeline, zone transitions and stage latency.

Before calibration, spatial results are labeled `pixel` / `Relative`. Calibration resets prior aggregates because pixel-space and ground-space histories cannot be combined safely. Calibration and camera-space zones persist in the server process and are automatically applied when the video is run again, which is important for short files.

## Configuration

All thresholds and resource bounds live in [`configs/default.yaml`](configs/default.yaml):

- detector model, resolution, confidence, IoU, tiled inference, maximum detections, classes,
  device and precision;
- ByteTrack thresholds, hybrid IoU/point-distance association, moving/stationary grace and buffer;
- inference interval, queue size, stale-frame policy and reconnect backoff;
- trajectory smoothing/history/TTL/cardinality;
- heatmap grid, time retention, movement threshold and Gaussian sigma;
- route grid/cardinality, zone debounce, upload/JPEG/WebSocket settings.

Realtime Common Path uses an 8x5 camera-specific macro grid for the supplied Grand Central
time-lapse. Each directed edge counts a Track ID at most once. One-second buckets feed a
30-second short window and 180-second long window with exponential age decay. Candidate paths
are extracted with positive-cost Dijkstra every 3 seconds and must retain at least 5 unique
tracks for 8 seconds before promotion. A 20% margin is required to replace an active path;
similar directed-edge paths are merged and inactive paths cool for 20 seconds before retirement.
The grid remains configurable; finer 16x9 and 32x18 grids did not produce sufficient connected
edge support on the supplied trajectory data at the required support thresholds.
Measured implementation details, before/after results and limitations are in
[`docs/common_path_realtime_report.md`](docs/common_path_realtime_report.md).

Visualization defaults are shared by backend and frontend under `visualization` in the YAML
configuration. The renderer draws a bounded-width translucent corridor, centerline and regularly
spaced direction arrows. When a confirmed route changes, both polylines are resampled and
interpolated over 500 ms. Individual tracklets remain bounded internal analytics state and are
never passed to the production renderer unless the explicit individual-path debug toggle is on.

The `data-test.mp4` default is YOLO26n + ByteTrack, class `person`, `imgsz=1280`, detector
confidence `0.05`, new-track confidence `0.15`, and inference interval 1. Low-confidence boxes
can recover existing tracks but cannot start new ones. Hybrid IoU/bottom-center association handles
the large apparent step between adjacent time-lapse frames, while a three-frame grace keeps a
confirmed stationary person visible through a short detector miss. In a 30-frame local CPU
regression, active tracks increased from 11.2 to 62.3 per frame; throughput decreased from 5.34 to
3.05 FPS. A full 1,804-frame run averaged 74.8 active tracks and peaked at 103 at 2.39 CPU FPS.
These are PoC measurements, not a claim that all people can be recovered in every scene.
Ultralytics licensing is AGPL-3.0 or Enterprise;
proprietary distribution requires legal review, an Enterprise license, or a detector with
compatible terms.

## API

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/health` | Service and session status |
| GET | `/metrics` | p50/p95 stage timing, queues, drops, CPU/RAM/GPU |
| POST | `/api/video/upload` | Chunked, size/type-limited video upload |
| POST | `/api/stream/start` | Start uploaded token or RTSP URL |
| POST | `/api/stream/stop` | Stop active session |
| PUT | `/api/overlay` | Compatibility endpoint for the active session overlay |
| GET | `/api/config/visualization` | Current shared production/debug visualization settings |
| PATCH | `/api/config/visualization` | Update visualization settings for current and future sessions |
| POST | `/api/calibration` | Set four-point ground-plane homography |
| PUT | `/api/zones` | Replace polygon zone configuration |
| GET | `/api/analytics/summary` | Count summary and timeline |
| GET | `/api/analytics/heatmap` | Occupancy/movement grid by time window |
| GET | `/api/analytics/flow` | Grid vector field and dominant direction |
| GET | `/api/analytics/paths` | Ranked grid/zone routes |
| GET | `/api/analytics/common-paths` | Stable candidate/active/cooling Common Paths |
| GET | `/api/analytics/flows` | Directed grid edges with unique-track support |
| GET | `/api/analytics/zones` | Zone occupancy, entries, exits, dwell, flow |
| GET | `/api/analytics/metrics` | Pipeline, queue and Common Path metrics |
| GET | `/api/stream/frame.jpg` | Latest processed JPEG |
| GET | `/api/stream.mjpg` | MJPEG stream |
| WS | `/ws/live` | Live dashboard snapshot |

Interactive OpenAPI documentation is available at `/docs` on the API port.

## Grand Central dataset

The dashboard now has a **Grand Central** workspace alongside the existing live upload/RTSP workflow. It supports independent `ground_truth` and `prediction` results, pixel or ground-plane coordinates, and 1-minute/5-minute/all-clip selections. Ground truth drives the analytics directly; prediction uses YOLO26n and ByteTrack. Every persisted analytics artifact identifies one source and the two streams are not mixed.

The pedestrian walking-path data has no clearly stated license. Use it only for research, education, internal demos, and benchmarks. It is ignored by Git and excluded from Docker images. Full format notes, measured dataset statistics, timebase mapping, homography provenance, and limitations are in [`docs/grand_central_dataset.md`](docs/grand_central_dataset.md).

Set up from the supplied video and an existing OpenTraj dataset directory:

```powershell
python scripts/setup_grand_central.py --video data/videos/data.mp4 --opentraj-dir path/to/OpenTraj/datasets/GC
python scripts/prepare_grand_central.py --coordinate-mode ground_plane --duration all --csv
```

Run ground-truth analytics, model inference, timestamp-aligned evaluation, and all three measured benchmark rows:

```powershell
python scripts/analyze_grand_central.py --source ground_truth --coordinate-mode pixel_space --duration all
python scripts/run_grand_central_inference.py --duration 1m --inference-interval 1
python scripts/run_grand_central_inference.py --duration 1m --inference-interval 2
python scripts/evaluate_grand_central.py --duration 1m --inference-interval 1 --threshold-pixels 60
python scripts/benchmark_grand_central.py
```

The setup script also accepts `--archive path/to/dataset.rar`; it reports a missing `7z`/`unrar` clearly. `--download` is opt-in and displays the source, destination, size status, and license warning before downloading. Dataset paths can be overridden with `GC_DATASET_ROOT`, `GC_VIDEO_PATH`, `GC_ANNOTATION_DIR`, `GC_PROCESSED_DIR`, and `GC_OUTPUT_DIR` as shown in `.env.example`.

Dataset API endpoints:

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/api/datasets` | Installed dataset capabilities |
| GET | `/api/datasets/grand-central/metadata` | Source, checksum and validation metadata |
| POST | `/api/datasets/grand-central/prepare` | Revalidate prepared artifacts |
| POST | `/api/datasets/grand-central/analyze` | Run analytics for source/mode/duration |
| GET | `/api/datasets/grand-central/status` | Last dataset analysis state |
| GET | `/api/datasets/grand-central/results` | Read a persisted real result |
| GET | `/api/datasets/grand-central/preview.jpg` | GT or prediction preview frame |

## Architecture and data policy

See [`docs/architecture.md`](docs/architecture.md) for diagrams and the 1/10/100-camera scale path. The runtime uses bounded queues, fixed grids, time buckets, ring-buffer trajectories and TTL/cardinality caps. Slow inference may drop stale frames for live sources; it cannot grow capture memory without bound.

Raw detection data is separate from aggregate analytics. The demo stores uploads and explicitly requested batch outputs only. It does not persist faces or map track IDs to people. Avoid placing RTSP credentials in logs or committed config.

## Tests and benchmarks

```powershell
python -m pytest -q
cd frontend
npm run build
```

Reproduce the CPU benchmarks:

```powershell
python -m scripts.benchmark
python -m scripts.benchmark_pipeline
```

Results and conclusions:

- [`benchmarks/benchmark_results.csv`](benchmarks/benchmark_results.csv)
- [`benchmarks/pipeline_profile.json`](benchmarks/pipeline_profile.json)
- [`docs/performance-report.md`](docs/performance-report.md)
- [`docs/test-report.md`](docs/test-report.md)
- [`benchmarks/grand_central_results.csv`](benchmarks/grand_central_results.csv)

On the measured i5-10210U, ONNX Runtime reached 5.462 effective FPS versus PyTorch's 4.884 FPS at interval 1; neither reaches the 20 FPS target. GPU FP16/TensorRT rows are explicitly marked unsupported. No target-domain labels were available, so accuracy loss is reported only as temporal sample retention and observed smoke-test behavior, not invented mAP/HOTA values.

## Example data

`data/videos/example-people.mp4` is a 20-frame functional smoke clip derived by translating the public [Ultralytics bus example](https://ultralytics.com/images/bus.jpg). It verifies decode, person detection, tracking and visual output. It is too short and simple for accuracy conclusions; replace it with licensed, labeled CCTV footage for product evaluation.

## Project map

```text
backend/app/       FastAPI, pipeline, inference, tracking, analytics, rendering, metrics
frontend/src/      React dashboard and spatial editors
configs/           Typed runtime configuration
scripts/           Batch runner and reproducible benchmarks
tests/             Unit and API integration tests
benchmarks/        Measured CSV and pipeline profile
docs/              Research, architecture, performance and test reports
data/videos/       Example input
docker/            Backend/frontend images and Nginx proxy
```

## Current limitations

- One active camera session per process.
- No database or historic persistence beyond current file-session memory.
- Four-point planar homography assumes the walking surface is approximately planar.
- Grand Central has sparse point annotations rather than boxes; distance-based detection proxies are available, but valid mAP/HOTA/IDF1 and a real 30-minute RTSP soak remain unavailable.
- CUDA, TensorRT, FP16, GPU utilization and VRAM require a separate NVIDIA host.
- Client-supplied RTSP URLs require allowlisting/credential handling before internet-facing deployment.

Research rationale and the analysis of IJCAI paper 479 are in [`docs/phase-1-research-and-architecture.md`](docs/phase-1-research-and-architecture.md).
