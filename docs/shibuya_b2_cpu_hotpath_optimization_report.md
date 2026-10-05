# Shibuya B2 CPU hot-path optimization

## Test scope

- Video: `data/videos/data-shibuya-5m.mp4`, first 25 seconds, 750 frames at 30 FPS.
- Hardware: Modal Tesla T4, 2 physical CPU cores requested.
- Config: `configs/shibuya-overlap-batch2.yaml`; `source_batch=2`, 10 detector images per GPU batch, bounded queues of 2 batches.
- Git base commit: `4269fc8f469653e8a9b740245fe33be80b677467` with a dirty experimental worktree.
- Model, thresholds, detection policy, ByteTrack association semantics, Common Path and TensorRT were not changed.
- Baseline: `shibuya-25s-overlap-b2-20261002`.
- Deep profile: `shibuya-25s-b2-cpu-profile-20261002` (diagnostic only; cProfile overhead means its FPS is not used for A/B throughput).
- Optimized benchmark: `shibuya-25s-b2-cpu-opt-20261002`.

## Deep-profile findings

Percentages below are relative to the owning thread's 179.46 s consumer or 180.70 s GPU-worker profile. Cumulative stage roots are used; nested rows must not be added together.

### Consumer thread

| Stage/function | Cumulative CPU/profile time | Share |
|---|---:|---:|
| ByteTrack `update` | 48.351 s | 26.94% |
| tile materialize + merge | 35.233 s | 19.63% |
| recursive `dataclasses.asdict` serialization | 23.174 s | 12.91% |
| Common Path update | 22.261 s | 12.40% |
| result-queue wait | 9.568 s | 5.33% |
| render | 9.985 s | 5.56% |

The merge kernel itself consumed 33.197 s. Its hot loop repeatedly rebuilt selected-box area, center and diagonal arrays for every candidate.

### GPU worker / preprocess

| Sub-stage | Cumulative time | Share of GPU-worker profile |
|---|---:|---:|
| Ultralytics preprocess | 58.762 s | 32.52% |
| pre-transform/letterbox | 32.426 s | 17.95% |
| `copyMakeBorder` | 17.533 s | 9.70% |
| OpenCV resize | 14.698 s | 8.13% |
| NumPy stack/copy | 18.434 s | 10.20% |
| tensor `.to()` / H2D-related call | 4.714 s | 2.61% |
| model forward wrapper | 37.339 s | 20.66% |
| YOLO postprocess | 22.473 s | 12.44% |
| result-queue put wait in the profiled run | 29.680 s | 16.43% |

Tile slicing/planning itself was only 0.089 s total. The preprocess cost is therefore Ultralytics resize, letterbox and stacking, not project tile creation.

### ByteTrack breakdown

| Sub-stage | Time | Share of ByteTrack `update` |
|---|---:|---:|
| first association | 22.365 s | 46.25% |
| custom hybrid association cost | 17.561 s | 36.32% |
| Kalman update | 10.392 s | 21.49% |
| second association | 8.952 s | 18.51% |
| output bbox conversion/clipping | 5.835 s | 12.07% |
| Kalman multi-predict | 1.500 s | 3.10% |
| track-history update | 1.563 s | 3.23% |
| Hungarian/LAP assignment | 1.263 s | 2.61% |

Within the 17.561 s hybrid association cost, IoU construction used 3.076 s, distance/norm calls 2.125 s, and repeated bbox extraction for point/height arrays used 6.387 s. The remaining 5.973 s covered hybrid gates, allocations, motion costs, score fusion and other NumPy operations. LAP was not the dominant tracker cost.

## Optimizations implemented

1. Tile merge now caches accepted-box area, center and diagonal once. Candidate order, source-aware suppression and all thresholds are unchanged.
2. Hybrid association snapshots each Ultralytics `xyxy` property once per call and reuses it for bottom centers and heights.
3. Track output clamps four scalar coordinates with scalar min/max instead of allocating through `np.clip` four times per object.
4. Detection/track cache serialization now writes the same flat field schema directly, avoiding recursive `asdict` and `deepcopy` of scalar-only dataclasses.
5. Added opt-in, per-thread cProfile artifacts for producer, GPU worker and consumer. Profiling remains disabled by default.

## A/B benchmark

| Metric | B2 baseline | Optimized B2 | Change |
|---|---:|---:|---:|
| Processing FPS | 5.780 | 6.078 | +5.16% |
| GPU utilization avg / peak | 43.608% / 75.0% | 46.550% / 75.0% | +2.942 pp avg |
| CPU utilization avg / peak | 138.561% / 163.4% | 143.769% / 182.3% | +5.208 pp avg |
| preprocess p50 / p95 | 67.969 / 82.362 ms | 63.465 / 113.448 ms | -6.63% / +37.75% |
| tile creation p50 / p95 | 0.084 / 0.102 ms | 0.078 / 0.094 ms | -7.14% / -7.84% |
| tile merge p50 / p95 | 32.486 / 58.500 ms | 31.012 / 54.989 ms | -4.54% / -6.00% |
| ByteTrack p50 / p95 | 43.867 / 71.317 ms | 37.631 / 56.990 ms | -14.22% / -20.09% |
| result consumer wait p50 / p95 | 23.691 / 70.374 ms | 36.217 / 77.185 ms | +52.87% / +9.68% |
| result queue age p50 / p95 | 145.019 / 1243.895 ms | 134.371 / 1073.633 ms | -7.34% / -13.69% |
| E2E p50 / p95 | 1564.099 / 2923.554 ms | 1584.659 / 2619.785 ms | +1.31% / -10.39% |

The result queue backlog improved: result-producer blocked time fell from 6.137 s to 3.656 s (-40.43%), and mean queue age fell from 440.396 ms to 363.373 ms (-17.49%). Consumer wait increased because the faster consumer now more often waits for the unchanged GPU worker; this is evidence of the bottleneck shift rather than a larger result backlog. Mean E2E fell from 1884.827 ms to 1738.922 ms (-7.74%).

The preprocess p95 regression is a tail spike, while p50 and mean improved (mean 68.315 to 65.977 ms). It should be rechecked over a longer repeated benchmark before attributing it to code; none of the accepted optimizations changed preprocessing.

## Quality gate

- No frame drop/reorder: frames 0 through 749 in order.
- Detections: 182,311 / 182,311, exact.
- Observed tracks: 116,614 / 116,614; 413 unique IDs in both runs.
- Full 86.9 MB tracking caches have the same SHA-256: `8fcbbb42aff09ea8797c04e5cc77796e76f0b411755e52551e37bae0acb083de`.
- Common Path final snapshot is exact: the same path IDs, directions, support, confidence, polylines and zero switches.
- Full test suite: 201 tests passed.

## New bottleneck and TensorRT decision

After the CPU changes, the consumer more often waits for the GPU-worker result. The next bottleneck is the unchanged detector worker, especially CPU-side Ultralytics preprocess/letterbox/stack and YOLO postprocess around the GPU forward. GPU average utilization is still only 46.55%, so the T4 is not continuously fed.

TensorRT FP16 should not be the immediate next step. The measured model-forward wrapper was 37.339 s, but preprocess was 58.762 s and postprocess was 22.473 s in the same worker profile. First A/B a semantics-preserving preprocess/H2D path (preallocated or pinned buffers, fewer NumPy stacks/copies, and removal of avoidable synchronization) and then re-profile. TensorRT FP16 becomes justified after that feed path is no longer dominant and must receive the same detection/tracking/Common Path quality gate.
