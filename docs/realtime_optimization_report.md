# Realtime optimization report

Updated: 2026-09-30 (Asia/Saigon)

## Result

Status: `NOT_MET`

A 25-second smoke on the same 750 source frames measured FP16 source batch 2 at 4.845 FPS.
That observed result is 17.9% faster than the new FP32 baseline, but the old runs used
different tracker motion weights and therefore are not a clean cross-precision A/B. Batch 2
is 6.0% faster than FP16 batch 1 under the same tracker config. The candidate remains far
below 30 FPS and is not promoted while quality is pending.

## Measured baseline

| Variant | GPU | Source/detector batch | Process FPS | Inference p50/p95 | Tracking p50/p95 | Result |
|---|---|---:|---:|---:|---:|---|
| FP32 baseline | Tesla T4 | 1 / 5 | 4.109 | 149.087 / 167.540 ms | 37.884 / 61.994 ms | Baseline |
| FP16 batch 1 | Tesla T4 | 1 / 5 | 4.569 | 129.329 / 150.086 ms | 36.165 / 60.680 ms | Faster |
| FP16 batch 2 | Tesla T4 | 2 / 10 | 4.845 | 108.807 / 131.343 ms | 38.116 / 64.108 ms | Selected; `QUALITY_PENDING` |
| FP16 batch 4 | Tesla T4 | 4 / 20 | 4.741 | 112.989 / 132.637 ms | 39.422 / 60.415 ms | Rejected |

Artifact review found 98.677% baseline-side detection matches at IoU >= 0.5 and only a
0.104% detection-count decrease. Rebuilding FP16 batch-2 detections with the baseline tracker
changed unique observed IDs from the invalid 484 to 438 versus baseline 439. Numeric track
identity and short-clip Common Paths still diverge, so this remains a regression proxy rather
than an IDF1/HOTA quality pass.

Baseline artifact:
`outputs/common_path/shibuya-10m-final-20260926-091731/shibuya-10m-final-20260926-091731`.

## Implemented changes

- Correct detector workload accounting for tiled inference and cache replay.
- Report detector images per source frame and per processed frame separately.
- Warm the exact detector instance used by web and Modal processing.
- Exclude optional preview render/encode samples from their latency distributions.
- Coalesce stale realtime preview frames while preserving analytics ingestion.
- Replace full Common Path neighbor sorting with partial selection plus deterministic ordering.
- Avoid evaluating detector ignore regions twice during tile merge.
- Export stage latency and processing FPS in Modal manifests and benchmark CSV files.
- Keep synchronized GPU profiling opt-in; throughput runs do not synchronize every frame.
- Compare cache frame/timestamp, detection IoU, observed tracks and resolved-config diffs with
  `scripts.compare_run_quality`.
- Add an FP32 batch-2 profile so batching and FP16 can be measured independently.
- Default Modal jobs to 2 physical CPUs, constrain PyTorch/BLAS threads to the request, record
  the requested CPU in the manifest, and retain `--cpu 8` as the rollback profile.

## Recommended GPU ablation

Run the small orthogonal matrix first, then a full run only for the winner:

```powershell
python scripts/benchmark_realtime.py `
  --input data/videos/data-shibuya-test.mp4 `
  --duration-seconds 25
```

The benchmark writes `outputs/benchmark/benchmark_summary_v2.csv` when an older incompatible
summary header already exists. The default matrix excludes the rejected batch-4 and duplicate
candidate profiles. Run clean FP32-batch-2 and aligned-candidate smoke jobs, then the 65-second
and full-clip checks only after recall/ID/path stability checks. Do not
describe sampled or dropped-frame operation as full 30 FPS processing.

## Verification

- Python tests: 174 passed.
- Python compileall: passed.
- Frontend production build: passed.
- Git whitespace validation: passed.
