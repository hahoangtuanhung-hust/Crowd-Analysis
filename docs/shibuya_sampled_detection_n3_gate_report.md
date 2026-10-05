# Shibuya sampled detection N=3 quality gate

Date: 2026-10-03  
Decision: **N=3 FAIL — keep N=1; N=5 not run**

## 1. Baseline N=1

- Run: `shibuya-25s-sampled-n1c-20261003`.
- Input: `data/videos/data-shibuya-5m.mp4`, 0–25 seconds, 1280×720,
  750/750 frames. Input SHA-256:
  `f3b845b6eb312086bb9dbc408986bea3277ba887e62a56212e7497a3fd5c9670`.
- Commit: `4269fc8f469653e8a9b740245fe33be80b677467` with the sampled-offline
  scheduler worktree changes described below.
- Backend/config: Tesla T4, PyTorch FP32, `yolo26n`, `source_batch=2`, overlap,
  `no_sync_profile`, `inference_interval=1`. No TensorRT.
- Controlled-run FPS: **5.960**. The locked production reference remains
  6.223 FPS; differences are run-to-run host/sampling noise and the added
  sampled-run instrumentation.
- 750 association updates, zero policy skips, 375 GPU batches, 3,750 detector
  images. Detector images rate: 150.361/source-second and 29.800/wall-second.
- CPU average/peak: 143.849% / 184.3%. GPU average/peak: 40.472% / 74%.
  VRAM peak: 1,718 MB.
- Preprocess p50/p95: 69.806/86.948 ms; forward: 67.027/68.604 ms;
  YOLO postprocess: 4.553/60.319 ms; detector wall: 187.037/259.369 ms.
- Tile merge: 27.028/49.989 ms; ByteTrack association:
  31.695/55.562 ms; queue age: 102.156/713.921 ms; E2E:
  1,510.395/2,329.358 ms.

## 2. N=3 performance

- Run: `shibuya-25s-sampled-n3c-20261003`.
- Scheduler invariant: **750/750 frames**, no drop/reorder, detection frames
  `0,3,6,…,747`; 250 association updates, 500 prediction steps, 125 GPU
  batches, and **0 detection deadline misses**.
- FPS: **12.525**, +110.151% against the controlled N=1 run and +101.269%
  against the locked 6.223 production reference. Processing time fell from
  125.830 s to 59.879 s.
- Detector rate: 10.024 scans/source-second, 3.312 scans/wall-second,
  50.120 images/source-second, and 20.875 images/wall-second.
- CPU average/peak: 143.408% / 182.2%, effectively unchanged. GPU average
  fell to 30.033% (−10.439 percentage points); sampled peak was 73%.
  VRAM peak was unchanged at 1,717 MB.
- Scan-only preprocess p50/p95: 72.160/84.968 ms; forward:
  66.956/67.827 ms; YOLO postprocess: 10.715/68.798 ms; detector wall:
  191.564/276.419 ms.
- Scan-only tile merge: 27.365/51.746 ms. ByteTrack association:
  28.310/46.807 ms; coast: 3.627/5.196 ms. Overall tracking, including
  coast: 4.587/41.444 ms.
- Queue age rose to 788.905/2,374.373 ms and E2E to
  2,152.806/4,834.943 ms. Sampling increases throughput, but the B2 detection
  group now spans six source frames and CPU analytics creates result backlog.
- Both final runs have the identical code fingerprint
  `e09d912fa2d409f99ff5935b37b9d2c424bcb8571e4ca0e0b57f792a556e20af`
  and identical 18-logical-CPU affinity. Only `video.inference_interval` differs.

## 3. N=3 tracking quality

- Detector equivalence at the 250 candidate scan frames is exact: 60,705 vs
  60,705 detections, 100% count equality, 100% IoU≥0.75 matching, and zero
  bbox/confidence difference. The quality regression is therefore downstream
  of the unchanged detector output.
- Unique observed Track IDs fell **413 → 315** (−23.729%). All track-state rows,
  including predictions, fell **171,038 → 134,512** (−21.355%).
- One-to-one mapped ID agreement over IoU≥0.5 matches was only **30.380%**.
- Presence fragments increased **1,867 → 4,749** (+154.365%); the
  lost/recovered proxy increased **1,454 → 4,434** (+204.952%).
- Candidate prediction-only rows: 97,193. Matched track bbox drift was modest
  (median 0.485 px, p95 3.501 px, max 29.097 px), but many tracks disappeared
  rather than merely drifting.
- FAR upper-third track-state rows fell **9,370 → 7,509** (−19.861%). Only
  59.210% of baseline FAR rows found an N=3 IoU≥0.5 match. This fails the
  small/far-person gate.
- In the dense crossing, fixed prediction-only frames show people counts lower
  by roughly 14–30% than N=1. Occlusion/re-entry produces visibly fewer retained
  boxes and more track lifecycle breaks. New entrants can also wait up to two
  source frames for their first observation, as required by N=3 semantics.

## 4. N=3 Common Path quality

- Both final snapshots contain three paths and retain the rank-level direction
  set `FORWARD, FORWARD, REVERSE`; no final FORWARD↔REVERSE flip was observed.
- All three final path IDs changed. Support changed 11→14, 6→9, and 3→8;
  confidence changed by 0.0553, 0.0411, and 0.0405. These are not equivalent
  evidence sets, so the larger candidate support is not an improvement claim.
- The final REVERSE polyline changed by **146.941 px mean / 173.369 px max**.
  The other two paths changed point count, so pointwise geometry is not directly
  comparable.
- Path creation events increased **45 → 59** (+31.111%) and cooling events
  increased 4→5. Earlier creation plus more path identities is evidence of
  lifecycle/path fragmentation and disappear/reappear instability.
- Common Path compute was 301.263/654.292 ms p50/p95 per scheduled compute
  versus 251.153/764.735 ms at N=1.

## 5. N=3 visual review

Reviewed identical timestamps and explicit prediction-only frames covering
waiting/stationary crowds, small people in FAR, new entrants, crossing,
occlusion, and late dense crossing.

- Amber boxes correctly identify `observed=false` prediction frames and those
  tracks do not produce Common Path evidence.
- Prediction coordinates generally coast smoothly for tracks that survive;
  obvious global frame reorder/drop was not seen.
- N=3 visibly retains fewer boxes in every reviewed dense-crossing frame.
  Missing tracks, not cosmetic bbox jitter, are the dominant regression.
- Common Path overlays appear earlier and follow materially different geometry,
  agreeing with the cache/path-event measurements.

Artifacts:

- N=1 video: `outputs/common_path/shibuya-25s-sampled-n1c-20261003/shibuya-25s-sampled-n1c-20261003/tracked_points_common_path.mp4`
- N=3 video: `outputs/common_path/shibuya-25s-sampled-n3c-20261003/shibuya-25s-sampled-n3c-20261003/tracked_points_common_path.mp4`
- N=3 quality JSON: `outputs/common_path/shibuya-25s-sampled-n3c-20261003/shibuya-25s-sampled-n3c-20261003/quality_vs_n1.json`
- Fixed-frame visual sheets: `visual_review_skip_frames.jpg` in each run directory.
- Both downloaded tracking caches match the SHA-256 recorded by their Modal
  manifests.

## 6. N=3 PASS/FAIL

**FAIL.** FPS improved clearly, but track continuity, fragmentation, FAR/small
person retention, ID stability and Common Path geometry/lifecycle regressed
materially. No thresholds, tracker, Common Path configuration or detection
policy were tuned to hide the result. N=3 is not promoted.

## 7. N=5

N=5 was **not run**. The explicit gate required N=3 to pass before a full N=5
quality evaluation, and N=3 failed. Consequently there is intentionally no N=5
video or quality artifact.

## 8. New bottleneck

Detector demand fell by two thirds, so GPU average utilization also fell rather
than rising. The limiting path moved toward CPU Common Path computation and
consumer/result backlog:

- Common Path compute p50/p95: 301.263/654.292 ms.
- Queue age p50/p95: 788.905/2,374.373 ms.
- E2E p50/p95: 2,152.806/4,834.943 ms.
- Scan-only detector wall remains 191.564/276.419 ms, but only 250 source
  frames invoke it. Detector bursts and periodic CPU Common Path computation
  now alternately fill the bounded result queue.

## 9. Recommended candidate

Keep **N=1** as production candidate. N=3 and N=5 are rejected/not evaluated,
respectively. Sampled scheduling remains available only as an experimental
profile.

## 10. Rollback command/config

Production rollback is the existing B2 configuration:

```powershell
modal run --quiet --timestamps modal_common_path.py `
  --input data/videos/data-shibuya-5m.mp4 `
  --start-seconds 0 --duration-seconds 25 `
  --engine tracklet_aggregation --mode offline_fast `
  --config configs/shibuya-overlap-batch2.yaml `
  --run-id shibuya-25s-n1-rollback `
  --cache-policy reuse `
  --detector-worker-mode no_sync_profile `
  --inference-backend pytorch_fp32 --cpu 2
```

Equivalent config setting: `video.inference_interval: 1`. This keeps PyTorch
FP32, B2 overlap, the same five-image tile policy, ByteTrack, hybrid association,
thresholds and Common Path configuration.

Validation: `python -m pytest -q` — **204 passed**.
