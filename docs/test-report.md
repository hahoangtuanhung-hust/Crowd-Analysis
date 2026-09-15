# Test Report

Run date: 2026-09-13

## Automated verification

```text
python -m pytest -q
34 passed
101 third-party deprecation warnings

npm run build
TypeScript passed
Vite production build passed
Entry bundle 257.81 kB; lazy timeline chunk 352.68 kB
```

The warnings originate from FastAPI 0.115 / Starlette calling an asyncio API deprecated by Python 3.14. They are not project test failures; pinning Python 3.12 in Docker avoids depending on the newest interpreter edge.

| Area | Evidence |
| --- | --- |
| Config | Valid defaults, strict unknown-key rejection, tracker threshold order |
| Detection/tracking integration | Real YOLO26n on the example clip; four people and stable IDs in the baseline smoke run |
| ByteTrack | Stable ID for moving box; empty detections handled |
| Pipeline | Lossless file processing, bounded drop-oldest queue, invalid source fail-fast, reconnect/backoff cycle |
| Trajectories | Bottom-center, EMA smoothing, missing observation, TTL, point/cardinality bounds |
| Heatmaps | Stationary affects occupancy only, movement segment rasterization, time retention, teleport-gap rejection |
| Perspective | Four-corner homography and boundary mapping |
| Flow/routes | Bidirectional routes, ranking, jitter rejection, active-route bound |
| Zones | Entry/exit/dwell/flow and boundary debounce |
| Rendering | Empty map no-op; mask restricts overlay to sampled area |
| API | Health, upload/process, summary, heatmap, metrics, JPEG, WebSocket, calibration and zones |
| Performance | Reproducible detector/tracker CSV and complete pipeline JSON profile |
| Grand Central adapter | Raw triples, x/y convention, timestamps, gap splitting, null fields, missing/empty/corrupt inputs, homography validation |
| Grand Central integration | Point-only GT analytics, empty scene, zone crossing, source isolation, evaluation with missing prediction frames |

## Live end-to-end verification

The final run used the actual development servers and YOLO26n rather than a test double:

```text
Upload: 618,956 bytes
First session: completed, 20 frames, peak 4, unique 4
Products: 1 popular path, movement heatmap sum 126.3, JPEG 236,708 bytes
Warm-service sample: 9.455 FPS, inference p50 104.853 ms, tracking p50 1.365 ms
Second session after setup: completed, 20 frames, unique 4
Persisted setup: spatial_mode=ground, calibration_required=false, zone=Walkway
```

The final backend and frontend logs contain no application errors. CPU and GPU Compose override files both pass `docker compose config`. An image build was attempted but Docker Desktop's Linux daemon was not running, so container execution remains unverified in this environment.

The Grand Central development endpoints were also exercised against the installed real data. `POST /api/datasets/grand-central/analyze` for GT/pixel/1m returned peak count 70, five popular paths, and 15 zone-flow pairs. Result JSON and both GT/prediction preview JPEGs returned HTTP 200 through the running API; the same result was fetched through the Vite proxy.

## Scenario coverage

| Scenario requested | Status | Current evidence / gap |
| --- | --- | --- |
| 0 people | Partially covered | Empty detection/tracker/heatmap paths are tested; no labeled empty CCTV clip |
| 5-10 people | Smoke covered | Real example image clip detects four people, just below target range |
| 30-100 people | Target-domain covered | Grand Central GT peak is 70 in the supplied clip; the model undercounts heavily |
| Occlusion/crossing | Logic only | Missing detections, debounce and paper/tracker research covered; requires labeled MOT sequence |
| Blur / low light | Not tested | Requires target-domain data |
| Strong perspective | Geometry covered | Homography is tested; quality needs measured ground points from a real camera |
| Stationary crowd | Unit covered | Occupancy remains nonzero while movement stays zero |
| Bidirectional flow | Unit covered | Opposing routes stay distinct and ranked |
| Video ended | Covered | Session reaches `completed` and finalizes active paths |
| Invalid source | Covered | Fails before worker start |
| RTSP disconnect | Simulated | Reconnect with bounded exponential backoff; no physical camera soak |
| Frontend disconnect | Covered by design | WebSocket loop exits without affecting processing; browser reconnection has backoff |
| GPU unavailable | Covered | CPU path runs; GPU/VRAM metrics return `null` |

## Resource stability

Bounded-memory invariants are directly tested for trajectory points, active track cardinality, heatmap time buckets, route state and both queues. The measured complete run used roughly 339-345 MB RSS after model load. A 30-minute real RTSP soak was not run on this laptop, so the production stability target remains open rather than being inferred from short tests.

## Known residual risks

- Grand Central provides sparse points rather than boxes. Count MAE and distance-based detection proxies are reported; valid mAP, HOTA, IDF1 and zone-boundary accuracy remain unavailable.
- No NVIDIA host, so CUDA OOM recovery, TensorRT, FP16, GPU utilization and VRAM are untested.
- RTSP retry was simulated; authentication, packet loss, codec errors and network timeout behavior need a real camera test.
- The included short example is derived from the public Ultralytics `bus.jpg` demo asset and is suitable for functional smoke testing, not model quality claims.
- In-app Browser automation was unavailable in this session. HTTP delivery and the complete upload/process/analytics/JPEG workflow passed, but desktop/mobile screenshot QA remains open.
