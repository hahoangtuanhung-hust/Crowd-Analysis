# Shibuya Optimized B2 detector-worker A/B

## Scope and method

- Video: first 25 seconds of `data/videos/data-shibuya-5m.mp4`, 750 frames.
- Hardware: Modal Tesla T4, 2 requested physical CPU cores.
- Baseline: Optimized B2, `source_batch=2`, 10 detector images per batch, FP32.
- Model, detection thresholds/policy, ByteTrack, Common Path and TensorRT were unchanged.
- Each optimization was benchmarked separately. Ineffective variants were removed from the final code.
- Final accepted mode: `no_sync_profile`, enabled only by `configs/shibuya-overlap-batch2.yaml`; the system-wide default remains `baseline`.

The original CPU profile found 2,250 implicit CUDA barriers: Ultralytics `ops.Profile` synchronized on entry and exit of preprocess, inference and postprocess, six times for each of 375 batches. Those calls consumed 27.453 seconds in the profiled worker. The accepted change replaces these timers with non-synchronizing timers and CUDA events; it does not change execution order or tensor values.

## 1. Preprocess breakdown before/after

The baseline implementation performs, per batch, 10 resize/letterbox calls, one `np.stack`, one BGR→RGB/BHWC→BCHW contiguous copy, one `torch.from_numpy`, one H2D transfer and one FP32 conversion. Across the clip this is 3,750 letterboxes, 375 stacks, 375 contiguous full-batch copies, 375 tensor wrappers, 375 H2D submissions and 375 conversions.

| Comparable metric | Optimized B2 baseline | Final no-sync | Change |
|---|---:|---:|---:|
| preprocess total over 750 frames | 49.483 s | 40.539 s | -18.08% |
| preprocess p50 / p95 | 63.465 / 113.448 ms | 52.698 / 69.499 ms | -16.96% / -38.74% |
| crop/tile p50 / p95 | 0.078 / 0.094 ms | 0.083 / 0.104 ms | effectively unchanged |
| fused detector wall p50 / p95 | 143.150 / 215.146 ms | 132.813 / 197.603 ms | -7.22% / -8.15% |

The earlier cProfile totals supplied in the task (58.762 s preprocess, 37.339 s forward, 22.473 s postprocess) are cumulative profiler call times. The table above uses the same per-frame timing fields on both A/B runs; final phase values use CUDA events without stage barriers.

## 2. H2D timing

H2D could be isolated in the two rejected staging-buffer experiments:

| Variant | H2D p50 / p95 | CPU submit p50 / p95 |
|---|---:|---:|
| reusable pageable buffer | 5.919 / 8.720 ms | 5.792 / 8.414 ms |
| reusable pinned + non-blocking | 4.024 / 6.459 ms | 0.129 / 3.055 ms |

Pinned memory reduced H2D p50 by 32.0% and made submission asynchronous, but end-to-end FPS was 5.957 versus 6.230 for no-sync alone. It was rolled back.

## 3. Allocation and copy findings

- Accepted no-sync optimization: full-batch allocation/copy count is unchanged; CUDA stage barriers fall from 2,250 to zero.
- Reusable-buffer experiment: full-batch allocations fell from 750 to one, eliminating 749 allocations and 375 contiguous full-batch copies, about 18.432 GB copied over the clip. Despite lower preprocess timing, it reached only 5.984 FPS and was rolled back.
- Coalesced result-transfer experiment reduced `.cpu()`/`.numpy()` transfers from 30 to 10 per batch: 11,250 to 3,750 calls. Transfer p50/p95 improved from 2.033/19.240 ms on the paired control to 1.267/4.196 ms, but total FPS was 6.056, so it was rolled back.
- No hot-path `.item()` was found. Baseline uses 11,250 `.cpu()` and 11,250 `.numpy()` calls after GPU NMS; those transfers are synchronization points.

## 4–6. Throughput, GPU and CPU A/B

| Isolated run | FPS | GPU avg / peak | CPU avg / peak | Versus historical 6.078 FPS |
|---|---:|---:|---:|---:|
| historical Optimized B2 | 6.078 | 46.55% / 75% | 143.77% / 182.3% | baseline |
| paired instrumented control | 5.061 | 37.49% / 82% | 146.75% / 185.6% | -16.73% |
| no-sync, accepted | 6.223 | 41.23% / 75% | 141.08% / 176.0% | +2.39% |
| reusable pageable buffer | 5.984 | 45.30% / 74% | 141.71% / 174.8% | -1.55% |
| pinned + non-blocking | 5.957 | 42.24% / 74% | 142.93% / 171.2% | -1.99% |
| coalesced result transfer | 6.056 | 42.86% / 75% | 141.35% / 172.2% | -0.36% |

Three no-sync full runs were stable at 6.219–6.230 FPS. Their 1 Hz GPU samples ranged from 41.23% to 47.68% average while throughput changed by only 0.18%; therefore GPU average is sampling-sensitive and no utilization uplift is claimed. Peak remained 74–75%.

## 7. Preprocess / forward / postprocess p50–p95

| Stage | Baseline p50 / p95 | Final p50 / p95 | Total before / after |
|---|---:|---:|---:|
| preprocess including H2D | 63.465 / 113.448 ms | 52.698 / 69.499 ms | 49.483 / 40.539 s |
| YOLO FP32 forward | 67.690 / 73.714 ms | 67.837 / 73.006 ms | 51.828 / 51.476 s |
| YOLO postprocess | 4.928 / 66.109 ms | 4.762 / 66.931 ms | 13.072 / 13.417 s |
| result GPU→CPU transfer | 1.741 / 11.082 ms | 1.775 / 9.653 ms | 2.717 / 2.420 s |

Postprocess remains primarily CUDA tensor work plus a Python loop over 10 images. The deep profile attributed 18.814 s to NMS, including 2.071 s in `torchvision.ops.nms`; result construction/box scaling used 3.540 s. The project-level result conversion then performs `.cpu().numpy()` outside Ultralytics postprocess.

## 8. Quality gate

- 750/750 frames, IDs 0–749, no drop or reorder.
- 182,311 detections identical.
- 116,614 observed tracks and 413 unique Track IDs identical.
- Baseline and final 86.9 MB tracking caches have identical SHA-256: `8fcbbb42aff09ea8797c04e5cc77796e76f0b411755e52551e37bae0acb083de`.
- Common Path final snapshot is exact, including path IDs, directions, support, confidence and polylines.
- Full test suite: 201 passed.

## 9. New bottleneck

FP32 forward is now the largest stable detector phase at 67.837 ms p50, followed by preprocess at 52.698 ms. Postprocess median is small but retains a 66.931 ms p95 tail. E2E p50 improved from 1,584.659 to 1,472.157 ms; E2E p95 was effectively flat at 2,619.785 versus 2,614.539 ms.

True next-batch H2D/forward overlap was not retained. The current Ultralytics API owns a mutable predictor batch under one lock and executes preprocess→forward→NMS as one fused call on the default stream. A second CUDA stream would require splitting that API, double-buffering predictor inputs, and explicit event dependencies. Pinned/non-blocking submission alone did not improve FPS, so the larger refactor was not justified under the exact-output gate.

## 10. TensorRT FP16 decision

TensorRT FP16 is now worth an isolated next A/B. After removing timing barriers, FP32 forward is the largest stable p50 stage and did not improve (67.690→67.837 ms), while preprocess p50 fell to 52.698 ms. TensorRT must still be evaluated separately with the same cache SHA-256/Track ID/Common Path gate; it was not enabled in this task.
