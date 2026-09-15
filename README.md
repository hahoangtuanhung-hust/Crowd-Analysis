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

Benchmark the real Grand Central clip across `imgsz=640/960/1280` and
`confidence=0.15/0.20/0.25`:

```powershell
python -m scripts.benchmark_tracking data/videos/data.mp4 --max-frames 150
```

Process the full clip with the measured default configuration:

```powershell
python -m scripts.process_point_tracks data/videos/data.mp4 --zones configs/zones.json
```

For a quick functional run, add `--max-frames 300 --output-dir outputs/smoke`. The full command
writes these files to `outputs/`:

- `tracked_points.mp4`: bottom-center points, Track IDs, bounded tails, direction, count, FPS and latency; no person bounding boxes;
- `trajectories.csv`: confirmed point tracks with `camera_id,track_id,frame_id,timestamp,x,y,zone_id`;
- `path_map.png`: completed routes and the top entry/exit flows ranked by unique Track ID;
- `heatmap.png`: density accumulated from confirmed track points;
- `zone_flows.json`: measured run summary and deduplicated first-zone to last-zone flows;
- `benchmark.csv`: the nine detector benchmark rows. Per-frame diagnostics are in `benchmark_frames.csv` and `frame_metrics.csv`.

`configs/zones.json` uses pixel-space polygons for the fixed 1920x1080 camera. Each zone accepts
the documented `id`, `name`, and `points` fields. Coordinates are scaled when the input resolution
differs. Edit these polygons before using another camera view.

## Dashboard workflow

1. Select **Video** and upload MP4/MOV/AVI/MKV, or select **RTSP** and enter the stream URL.
2. Start the session. File input processes to completion; RTSP remains live and reconnects with exponential backoff.
3. Toggle detection, tracking, trajectory, video heatmap and zone overlays independently.
4. Use **Calibrate** to place four ordered ground-plane corners and enter real-world width/height.
5. Use **Zones** to draw and name one or more polygons.
6. Inspect occupancy vs movement heatmaps, current/1m/5m/entire windows, vector field, routes, timeline, zone transitions and stage latency.

Before calibration, spatial results are labeled `pixel` / `Relative`. Calibration resets prior aggregates because pixel-space and ground-space histories cannot be combined safely. Calibration and camera-space zones persist in the server process and are automatically applied when the video is run again, which is important for short files.

## Configuration

All thresholds and resource bounds live in [`configs/default.yaml`](configs/default.yaml):

- detector model, resolution, confidence, IoU, maximum detections, classes, device and precision;
- ByteTrack thresholds and lost-track buffer;
- inference interval, queue size, stale-frame policy and reconnect backoff;
- trajectory smoothing/history/TTL/cardinality;
- heatmap grid, time retention, movement threshold and Gaussian sigma;
- route grid/cardinality, zone debounce, upload/JPEG/WebSocket settings.

The measured `data.mp4` default is YOLO26n + ByteTrack, class `person`, `imgsz=960`, confidence
`0.15`, `max_det=1000`, inference interval 1, and a 60-frame lost-track buffer. On the local CPU,
960/0.15 was the fastest benchmark candidate above 3 FPS with the highest active-track count;
1280 detected more distant people but ran at about 2.3 FPS. These are PoC measurements, not a claim
that the defaults are universally best. Ultralytics licensing is AGPL-3.0 or Enterprise;
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
| PUT | `/api/overlay` | Update five overlay switches |
| POST | `/api/calibration` | Set four-point ground-plane homography |
| PUT | `/api/zones` | Replace polygon zone configuration |
| GET | `/api/analytics/summary` | Count summary and timeline |
| GET | `/api/analytics/heatmap` | Occupancy/movement grid by time window |
| GET | `/api/analytics/flow` | Grid vector field and dominant direction |
| GET | `/api/analytics/paths` | Ranked grid/zone routes |
| GET | `/api/analytics/zones` | Zone occupancy, entries, exits, dwell, flow |
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
