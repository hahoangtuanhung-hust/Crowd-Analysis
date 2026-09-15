# Performance Report

Measured: 2026-09-13, Asia/Saigon

## Scope and hardware

This report separates model/tracker compute from the complete application pipeline. The measurements are real; empty values in the CSV mean the backend was unavailable.

```text
OS: Windows 11 Pro
CPU: Intel Core i5-10210U, 4 physical / 8 logical cores
RAM: 15.8 GB
GPU: none (Intel integrated graphics is not a CUDA device)
Python: 3.14.6
PyTorch: 2.13.0+cpu
Ultralytics: 8.4.116
ONNX Runtime: 1.28.0, CPUExecutionProvider
OpenCV: 5.0.0
```

Artifacts:

- `yolo26n.pt`: SHA-256 `9B09CC8BF347F0FC8A5F7657480587F25DB09B34BF33B0652110FB03A8AD4FEF`
- `yolo26n.onnx`: SHA-256 `F145C2F9611814E35ECF8D848ACA343868339922A5C6E0569EE5726A37715AB9`
- Input: `data/videos/example-people.mp4`, 20 frames, 810 x 1080, 8 FPS. The benchmark traverses the clip forward/backward up to 76 frames to reduce warmup sensitivity.
- Detector: person class, confidence 0.35, IoU 0.70, `imgsz=640`, batch 1.
- ReID is disabled for every tracker. This keeps the comparison CPU-compatible but does not measure the advertised appearance-assisted modes.

The full raw table is in `benchmarks/benchmark_results.csv`; the complete stage profile is in `benchmarks/pipeline_profile.json`.

## Grand Central one-minute benchmark

The target-domain table is in `benchmarks/grand_central_results.csv`. All rows use the supplied 1920 x 1080 `data.mp4` on the i5-10210U CPU; GPU fields are blank. Ground-truth throughput counts sparse annotation frames, while model throughput counts frames sent to the detector, so those rates are not directly interchangeable.

| Mode | Interval | Processing FPS | Inference | Tracking | Analytics | E2E | Dropped |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Ground-truth analytics | 0 | 210.567 | n/a | n/a | 4.749 ms | 4.749 ms | 0 |
| YOLO26n + ByteTrack | 1 | 7.671 | 125.834 ms | 3.091 ms | 0.076 ms | 768.064 ms | 0 |
| YOLO26n + ByteTrack | 2 | 7.244 | 132.525 ms | 3.093 ms | 0.066 ms | 400.391 ms | 0 |

Interval 2 processed 900 detector frames versus 1,800 at interval 1. Its end-to-end queue latency was lower, but it emitted only 606 trajectory points versus 1,403 at interval 1. The baseline remains interval 1 for analytics fidelity.

## Detector and frame interval

These timings exclude video decode, analytics, rendering and JPEG encoding.

| Backend | Interval | Effective input FPS | Inference p50 | Inference p95 | ByteTrack p50 | RAM peak |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PyTorch FP32 | 1 | 4.884 | 179.840 ms | 308.964 ms | 2.029 ms | 386.6 MB |
| PyTorch FP32 | 2 | 11.252 | 167.940 ms | 226.919 ms | 2.225 ms | 398.6 MB |
| PyTorch FP32 | 3 | 15.830 | 179.824 ms | 214.809 ms | 2.148 ms | 400.3 MB |
| ONNX Runtime FP32 | 1 | 5.462 | 166.169 ms | 261.560 ms | 3.656 ms | 489.0 MB |

ONNX Runtime improved interval-1 effective throughput by 11.8%, inference p50 by 7.6%, and p95 by 15.3% against PyTorch in this run. It used about 102 MB more peak process RSS. The gain is useful but insufficient for 20 FPS on this CPU.

Interval 2 and 3 increase effective input throughput because only 50.0% and 34.2% of frames receive detections. The current implementation does not emit tracker prediction-only samples on skipped frames, so those gains directly reduce temporal analytics sampling. Interval 3 also reduced the full-pipeline unique-track result from 4 to 3 on this small clip. Use interval 1 for fidelity; interval 2 is a CPU-preview fallback; interval 3 is not the default.

## Tracker comparison

All trackers consumed the same cached YOLO26n detections. `ID continuity` is an IoU-linked adjacent-frame proxy, not HOTA or IDF1.

| Tracker | Tracking p50 | Tracking p95 | ID continuity proxy | Output coverage |
| --- | ---: | ---: | ---: | ---: |
| ByteTrack | 2.029 ms | 2.842 ms | 100.0% | 94.26% |
| OC-SORT | 1.926 ms | 3.411 ms | 100.0% | 95.49% |
| Deep OC-SORT, no ReID | 1.712 ms | 3.012 ms | 100.0% | 95.49% |
| BoT-SORT, no ReID | 41.405 ms | 81.694 ms | 100.0% | 97.13% |
| TrackTrack, no ReID | 35.607 ms | 42.462 ms | 100.0% | 81.56% |

The clip is too easy to rank association accuracy: every tracker scored 100% on the continuity proxy. ByteTrack remains the baseline because it adds about 2 ms and has the simplest failure surface. OC-SORT and Deep OC-SORT remain low-cost challengers. BoT-SORT and TrackTrack spend substantial CPU on their default motion compensation/multi-cue paths without a measured benefit here; they require a labeled occlusion dataset before adoption.

## Full pipeline

The default profile uses tracking, trajectories and zone overlays. The all-overlays profile additionally enables detections and the video heatmap.

| Profile | Interval | Output FPS | Decode p50 | Infer p50 / p95 | Track p50 | Analytics p50 | Render p50 / p95 | Encode p50 | RSS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Default | 1 | 2.815 | 5.446 ms | 304.464 / 632.961 ms | 2.641 ms | 1.181 ms | 3.193 / 7.379 ms | 9.493 ms | 343.6 MB |
| Default | 2 | 2.882 | 5.537 ms | 343.146 / 493.157 ms | 3.101 ms | 1.197 ms | 3.669 / 37.058 ms | 10.508 ms | 339.3 MB |
| Default | 3 | 3.218 | 5.579 ms | 277.158 / 444.271 ms | 2.571 ms | 1.143 ms | 3.208 / 45.855 ms | 10.093 ms | 341.7 MB |
| All overlays | 1 | 2.646 | 6.026 ms | 322.593 / 866.344 ms | 2.702 ms | 1.140 ms | 14.236 / 18.892 ms | 10.448 ms | 344.6 MB |

Full-pipeline inference is slower than isolated inference because decode, Torch and rendering contend for a four-core CPU. End-to-end latency for the file profile is 806-3137 ms p95 because lossless offline mode applies backpressure and queues decoded frames. This is not an RTSP latency claim: live mode uses a bounded latest-frame queue and drops stale frames.

The first all-overlay profile measured render p50 at 110.219 ms. Profiling located a full-frame NumPy floating-point alpha blend in the heatmap overlay. Replacing it with OpenCV `addWeighted` plus a nonzero mask reduced measured p50 to 14.236 ms, an 87.1% reduction, while pixel tests verify that only sampled areas are changed.

## Target assessment

| Goal | Result |
| --- | --- |
| 20 FPS on common GPU | Not tested: no CUDA GPU available |
| p95 processing latency below 100 ms | Failed on this CPU; model p95 alone is 261.6 ms with ONNX |
| Bounded queues and trajectory history | Passed by design and automated tests |
| Non-blocking capture | Passed architecturally; live queues use drop-oldest |
| Stable 30-minute run | Not completed in this environment; short soak is reported in the test report |
| GPU and VRAM metrics | Gracefully return `null` |

At 20 input FPS and interval 1, this laptop cannot support one camera. Even isolated ONNX + ByteTrack provides only 5.46 effective FPS before decode/render overhead. With 30% capacity headroom, it is appropriate for one low-rate stream requiring at most about four detector calls per second, such as 8 FPS at interval 2. Camera-per-GPU capacity is unknown until the same end-to-end benchmark runs on the target GPU and codec.

## Technical decisions

- **Detector:** YOLO26n remains the PoC baseline. This benchmark validates integration and speed only, not detection accuracy. YOLO26s or RT-DETR-R18 needs a labeled target set before replacement.
- **Tracker:** ByteTrack remains default. OC-SORT is the first CPU challenger; TrackTrack should be evaluated on labeled dense/occlusion footage, not selected from paper numbers alone.
- **Resolution:** keep `imgsz=640`. Only 640 was measured; 512/960 are future accuracy/performance ablations.
- **Interval:** 1 for analytics correctness. Allow 2 as an explicit CPU preview compromise. Do not default to 3.
- **Backend:** PyTorch FP32 for development; ONNX Runtime for CPU deployment; TensorRT FP16 for an NVIDIA deployment benchmark. FP16 and TensorRT are marked unsupported in the CSV, not estimated.
- **Accuracy loss:** no labeled ground truth exists, so mAP/HOTA/IDF1 loss cannot be claimed. Measured temporal sample retention is 50.0% at interval 2 and 34.2% at interval 3; the latter missed one of four unique tracks in the full pipeline.
- **Popular paths:** grid-flow route compression plus zone transitions is the online MVP. Trajectory clustering/DD-CRP remains an offline challenger after perspective correction.

## Scaling recommendation

- **1 camera:** modular monolith as implemented; one bounded capture queue, one detector/tracker worker and one analytics/render worker.
- **10 cameras:** independent ingest workers, shared GPU inference service with dynamic batching, per-camera tracker state, central metrics, admission control based on measured p95 with at least 30% headroom.
- **100 cameras:** camera ingest services, partitioned GPU worker pool, durable event bus carrying anonymous tracks, stateless analytics consumers plus time-series/columnar storage, scheduler/autoscaling, fleet observability and fault isolation.

No multi-camera number should be derived from the published YOLO model latency alone. Decode, batching efficiency, scene density, tracker, overlays, codec and recovery headroom must be included on the target hardware.

## Warm service smoke check

After the benchmark matrix, the already-running development service processed the 20-frame example at 9.455 output FPS with inference p50 104.853 ms and tracking p50 1.365 ms. This confirms that power state, warm caches and concurrent load materially affect this laptop. It is retained as end-to-end smoke evidence, not substituted into the controlled matrix above; production sizing should use repeated target-host runs and conservative p95.
