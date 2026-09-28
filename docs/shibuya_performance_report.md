# Shibuya Performance Report

Date: 2026-09-25

## Scope

The same 5-second prefix of `data/videos/data-shibuya-test.mp4` was run on a Modal Tesla T4 with the same model (`yolo26n.pt`), tracker, engine (`tracklet_aggregation`) and 150 frames. The local machine has no CUDA device, so all inference measurements below are from Modal.

Input: 1280x720, approximately 30 FPS. Model/backend was not changed.

## Findings

The production Shibuya profile uses 2x2 tiled inference plus a full-frame pass:

```text
1 full frame + 4 tiles = 5 model passes per source frame
```

For a 65-second clip with 1,951 frames this is approximately 9,755 model passes. This is the dominant cause of the observed 20-minute processing time.

Common Path is not the bottleneck. In the baseline run its p95 was 4.335 ms, while inference p50 was 180.448 ms.

## A/B measurements

| Profile | GPU | Frames | Inference p50 | Inference p95 | Analytics p95 | Render p50 | Encode p50 | Pipeline FPS | Remote wall time |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline: tiled FP32, 1280 | Tesla T4 | 150 | 180.448 ms | 196.980 ms | 4.335 ms | 1.829 ms | 6.848 ms | 3.705 | 41.15 s |
| Tiled FP16, 1280 | Tesla T4 | 150 | 146.068 ms | 159.627 ms | 4.053 ms | 2.010 ms | 6.275 ms | 4.318 | 35.32 s |
| Single pass FP32, 1280 | Tesla T4 | 150 | 18.208 ms | 19.990 ms | 0.344 ms | 1.296 ms | 6.053 ms | 14.598 | 10.79 s |

The single-pass profile is about 9.9x faster in inference p50 and about 3.9x faster in end-to-end pipeline FPS for this smoke clip. FP16 alone on the tiled profile provides a smaller improvement, about 19% in inference p50.

## Artifacts

- Baseline: `outputs/common_path/shibuya-baseline-20260925-091031/`
- Tiled FP16: `outputs/common_path/shibuya-fp16-20260925-091721/`
- Single pass: `outputs/common_path/shibuya-single-pass-20260925-091412/`

Each run includes `manifest.json`, `summary.json`, `metrics.csv`, `tracking_cache.jsonl`, `tracked_points_common_path.mp4` and resolved configuration.

## Correctness observations

All three runs are only 5 seconds and report `insufficient_data`; there is no valid active Common Path in this short window. Therefore path quality, long-window support, jitter and adaptation are `NOT_RUN`, not passed. The single-pass run produced fewer unique temporary track IDs in this short clip (61 versus 248 baseline), so it must not become the default solely from the speed result. The difference requires a longer video review and detection/track quality comparison.

## Status

- Bottleneck diagnosis: **PASS**
- Modal GPU execution: **PASS** (Tesla T4)
- Common Path bottleneck hypothesis: **FAIL**
- Single-pass speed improvement: **PASS**
- FP16 speed improvement: **PASS**, but modest
- Full-video quality equivalence: **NOT_RUN**
- Browser/UI FPS: **NOT_RUN**
- TensorRT conversion: **NOT_RUN** by scope

## Recommended next experiment

Run baseline and single-pass on the full 65-second clip, then compare detection count, temporary Track ID coverage, tracklet support, first valid path time and rendered snapshots. If single-pass loses too much recall, test a reduced tile policy or lower tile frequency instead of returning to five model passes on every frame. Keep the current production profile unchanged until that quality comparison is complete.

## Full-video A/B result

The requested full 65-second A/B was run on the same Tesla T4. The local command timed out while downloading large artifacts, but the remote jobs completed successfully; manifests and summaries were read directly from the Modal Volume.

| Profile | Remote wall | Frames | Pipeline FPS | Inference p50/p95 | Analytics p95 | Unique temporary IDs | Active path |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Baseline tiled FP32 | 653.35 s | 1,950 | 2.987 | 233.679 / 267.433 ms | 331.457 ms | 1,287 | `path-095`, FORWARD, support 7 |
| Single-pass FP32 | 189.93 s | 1,950 | 10.290 | 19.252 / 20.104 ms | 600.622 ms | 715 | `path-046`, FORWARD, support 26 |

Both full runs completed and produced an active forward path. Single-pass reduced remote processing wall time by about 71% and did not lose the active forward route in this clip. However, temporary ID counts differ substantially (1,287 vs 715), so this is not sufficient to claim recall equivalence. A sampled detection/track review is still required before changing the production default.

The full baseline also explains why the observed delay can approach 20 minutes: 653 seconds remote processing for a 65-second clip, plus local artifact download. The dominant cost remains model inference; Common Path has occasional large p95 outliers but is not the primary average cost.

Full-run artifact namespaces:

- `common_path/runs/shibuya-baseline-full-20260925-093636`
- `common_path/runs/shibuya-single-pass-full-20260925-095318`

## Tracklet curve-join fix verification

After changing `_attach()` to compare a new segment with the local endpoint tangent, a Modal GPU verification run was completed:

```text
run: shibuya-tracklet-curve-fix-20260925-103045
GPU: Tesla T4
profile: configs/shibuya-single-pass.yaml
clip: 0-25 s
frames: 750
remote wall time: 66.63 s
pipeline FPS: 11.355
inference p50/p95: 20.036 / 21.604 ms
tracking p50/p95: 11.456 / 14.119 ms
Common Path analytics p50/p95: 0.357 / 36.335 ms
render p50/p95: 6.303 / 12.213 ms
encode p50/p95: 6.796 / 9.802 ms
active path: path-009, FORWARD, support 17
```

The contact sheet shows one continuous directed path in the final snapshots. This is a candidate-profile verification, not a full-video quality claim. The full artifact is in `outputs/common_path/shibuya-tracklet-curve-fix-20260925-103045/` and the remote run is under the matching `common_path/runs/` namespace.

## Reproduction

```powershell
modal run --quiet --timestamps modal_common_path.py `
  --input data/videos/data-shibuya-test.mp4 `
  --start-seconds 0 --duration-seconds 5 `
  --engine tracklet_aggregation --mode offline_fast `
  --config configs/shibuya.yaml `
  --run-id shibuya-baseline-<timestamp> --cache-policy refresh

modal run --quiet --timestamps modal_common_path.py `
  --input data/videos/data-shibuya-test.mp4 `
  --start-seconds 0 --duration-seconds 5 `
  --engine tracklet_aggregation --mode offline_fast `
  --config configs/shibuya-single-pass.yaml `
  --run-id shibuya-single-pass-<timestamp> --cache-policy reuse
```

## Detection/tracking correction verification

The detector/tracker correction was verified on Modal Tesla T4 without changing
`yolo26n.pt` or the Ultralytics backend. The detector now keeps strong
edge-of-tile evidence, merges only aligned cross-tile duplicates (same-source
boxes are not merged), and the tracker uses a short velocity history plus the
source frame ID to bridge dropped realtime frames. A synthetic crossing test
also covers two pedestrians crossing with a two-frame occlusion.

| Run | Frames | Detection avg/frame | Observed tracks avg/frame | Unique IDs | Track p50/p95 | Remote wall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Before correction, first 60 frames of long baseline | 60 | 173.77 | 127.45 | 171 | N/A | included in 605.94 s run |
| After correction, Modal smoke | 60 | 207.30 | 143.72 | 187 | 53 frames / 35.59 ms tracking p95 | 28.06 s |

The smoke contact sheet is
`outputs/common_path/shibuya-detect-track-final-smoke-20260925/preview_contact_sheet.jpg`.
This is evidence of fewer short-lived IDs in the tested prefix (3 in both
60-frame samples; the 10-second comparison fell from 5 to 2), not a measured
precision/recall score: the Shibuya clip has no person-level ground-truth labels.
The production profile remains tiled FP32, so a longer full-video run is still
needed before claiming realtime throughput or full recall.

## Hybrid Motion-ROI B/C verification

The interrupted run was resumed and completed with separate scheduler-fingerprinted
artifacts on the same Modal Tesla T4, using the full 1,951-frame Shibuya video.
The model, detector, tracker, Common Path engine, source and frame schedule were
held constant; only the top-level `motion_roi` profile changed.

| Profile | Config | Frames | Remote wall | Pipeline FPS | Inference p50/p95 | Tracking p95 | Scan decisions | Avg detector images/frame |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| B reference | `configs/shibuya.yaml` | 1,951 | 1,216.43 s | 1.605 | 502.270 / 651.392 ms | 78.012 ms | 1,951 reference | 5.00 |
| C hybrid | `configs/shibuya-motion-roi.yaml` | 1,951 | 1,316.36 s | 1.483 | 506.412 / 655.303 ms | 78.916 ms | 1,951 reference | 5.00 |

The C run is a valid hybrid execution, but it selected reference coverage for
all frames: the motion mask selected all four validated tiles after warmup and
the configured cost/coverage guard correctly fell back to the five-image
reference profile. It therefore produced **no measured inference saving** on
this dense Shibuya clip; wall time was 8.2% slower and FPS 7.6% lower than B.
This is evidence for keeping `motion_roi.enabled: false` in production by
default, not evidence that hybrid is faster. The scheduler and coverage trace
are still useful for testing sparser scenes and future tuning.

Required artifacts are present in the Modal Volume namespaces:

- `common_path/runs/shibuya-b-reference-20260925`
- `common_path/runs/shibuya-c-hybrid-20260925`

Each contains `scheduler_decisions.jsonl`, `motion_roi_metrics.csv`,
`observation_coverage_by_region.csv`, `region_funnel.csv`,
`candidate_decisions.jsonl`, `path_support.jsonl`, `provenance.json`, resource
metrics and the rendered MP4. The scheduler trace reports `ROI_TILE_BUDGET_EXCEEDED`
on 1,890 C frames and `BACKGROUND_WARMUP` on 61 frames. No person-level labels
are available, so recall, precision, IDF1, jitter and direction-error quality
remain `REVIEW_PENDING`, not PASS.

Full-video commands (results remain on the Volume; download separately):

```powershell
modal run --quiet --timestamps modal_common_path.py `
  --input data/videos/data-shibuya-test.mp4 --start-seconds 0 `
  --engine tracklet_aggregation --mode offline_fast `
  --config configs/shibuya.yaml `
  --run-id shibuya-b-reference-<timestamp> --cache-policy reuse

modal run --quiet --timestamps modal_common_path.py `
  --input data/videos/data-shibuya-test.mp4 --start-seconds 0 `
  --engine tracklet_aggregation --mode offline_fast `
  --config configs/shibuya-motion-roi.yaml `
  --run-id shibuya-c-hybrid-<timestamp> --cache-policy reuse
```
