# 10-minute T4 bottleneck audit

Date: 2026-09-26

Status: `PARTIAL_MEASURED`; the final 10-minute live GPU run is `NOT_RUN` after
the earlier one-hour Modal timeout. No detector model, backend, tracker
association policy, or Common Path output contract was replaced.

## Provenance

The reference source is `data/videos/data-shibuya-test.mp4`, SHA-256
`ae75cf06369007e9745c0f951e06e07826a2abc5dcf36276bc1ef06050aca04c`, with
`yolo26n.pt`, SHA-256
`9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef`.
The full reference profile uses `configs/shibuya.yaml`: 1280x720, tiled FP32,
one full frame plus four tiles, `motion_roi.enabled=false`, and
`tracklet_aggregation.update_interval_seconds=0.5`.

The earlier full 65-second tiled T4 run (`shibuya-baseline-full-20260925-093636`)
processed 1,950 frames in 653.35s (2.987 pipeline FPS), with inference
p50/p95 233.679/267.433ms and analytics p95 331.457ms. Those values are from a
different code/config fingerprint and are retained as historical context, not
as a strict before/after comparison.

## Measured hotspots

The post-patch T4 smoke (`debug-t4-after-20260926-b`) processed the same source
for 150 frames with the reference tiled profile. It used a real Tesla T4; no
CPU fallback occurred.

| Stage | p50 | p95 | Evidence |
| --- | ---: | ---: | --- |
| Model predict | 112.839ms | 129.960ms | 5 detector images per source frame |
| Detector postprocess + tile merge | 28.259ms | 34.311ms | measured after model return |
| Inference total | 143.372ms | 161.741ms | includes CUDA synchronization |
| Tracking | 31.384ms | 36.606ms | 234 temporary IDs in 5s smoke |
| Common Path update | 0.572ms | 9.676ms | 10 actual computes in 5s |
| Common Path compute only | 44.704ms | 100.821ms | diagnostic compute samples |
| Render | 2.072ms | 2.567ms |  |
| Encode | 6.155ms | 9.432ms |  |
| Cache write | 2.336ms | 4.090ms | buffered JSONL |

The dominant live cost remains the detector workload: tiled FP32 executes five
images per source frame. The existing single-pass profile is an experiment,
not a production change. CPU utilization averaged 100.394%, GPU utilization
averaged 33.848%, RAM peaked at 5,220.367MB and VRAM peaked at 839MB.

## Common Path replay

The valid 1,950-frame single-pass tracking cache was replayed without detector
or tracker inference using the current Common Path code. The runner used the
explicit audit-only `--allow-cache-key-mismatch` flag because the cache was
created before the cache-key formula changed; source and model hashes matched.

Default cadence, 0.5s:

- 1,950 frames, 64.96s media time, 10.229 replay FPS.
- 130 Common Path computes; compute p50/p95 755.206/1,271.073ms.
- Common Path mean 48.438ms/frame because most frames are no-op between computes.
- Tracklet buffer peaked at 1,143 segments; RAM increased from 117.8MB to 189.4MB.

Audit cadence profile, 2.0s (`configs/shibuya-single-pass-t4.yaml`):

- 33 Common Path computes; total compute 20.416s versus 92.593s in the 0.5s replay.
- Replay FPS increased to 22.377; Common Path mean fell to 11.192ms/frame.
- Path identity/support changed materially (`path-087` versus `path-017`), so
  this profile is `REVIEW_PENDING` and is not enabled in `configs/shibuya.yaml`.

Artifacts:

- `outputs/common_path/debug-t4-after-20260926-b/debug-t4-after-20260926-b/`
- `outputs/common_path/debug-replay-runner-20260926/`
- `outputs/common_path/debug-replay-runner-20260926-t4/`

Each replay runner artifact includes `stage_metrics.csv` and
`long_run_metrics.csv`. The replay is an analytics measurement, not inference
FPS and not a person-level recall test.

## Patch

- Vectorized the source-aware duplicate-tile predicate while preserving the
  existing `_same_detection` geometry rules. The corrected instrumentation
  separates model predict, postprocess/merge, and total inference; merge time
  no longer includes model execution.
- Added a bounded fingerprint cache for repeated Common Path link scores and a
  cache-hit diagnostic. It does not alter direction, overlap, support, Top K,
  Path ID, or color rules.
- Increased tracking-cache JSONL buffering and measured cache-write latency.
- Added `stage_metrics.csv`, `long_run_metrics.csv`, state counters, and an
  audit-only cache-key override that still requires matching source/model
  hashes.
- Increased Modal batch worker timeout from 1 hour to 4 hours in
  `modal_common_path.py`; this only prevents premature cancellation of a long
  job and does not make the pipeline realtime.

## Reproduction

```powershell
$runId = "shibuya-10m-final-$(Get-Date -Format yyyyMMdd-HHmmss)"
modal run --quiet --timestamps modal_common_path.py `
  --input data/videos/data-shibuya-10m.mp4 `
  --engine tracklet_aggregation --mode offline_fast `
  --config configs/shibuya.yaml --run-id $runId --cache-policy reuse

python scripts/download_modal_artifacts.py `
  --run-id $runId --output-dir "outputs/common_path/$runId"
```

The earlier timeout run left partial `tracking_cache.jsonl` and MP4 files but
no complete manifest; it must not be treated as a successful 10-minute result.
No person-level labels are available, so detection recall, IDF1, direction
errors, jitter, and Common Path quality remain `REVIEW_PENDING` until the full
run is reviewed at matching timestamps.
