# Shibuya detector workload A/B: 5 images vs 3 images

Date: 2026-10-03  
Decision: **FAIL — keep production at 1 full frame + 4 tiles**

## Test controls

- Input: `data/videos/data-shibuya-5m.mp4`, 0–25 seconds, 1280×720,
  750/750 frames. SHA-256:
  `f3b845b6eb312086bb9dbc408986bea3277ba887e62a56212e7497a3fd5c9670`.
- Tesla T4, PyTorch FP32, `source_batch=2`, `inference_interval=1`, overlap and
  `no_sync_profile`. No TensorRT.
- Both runs used code fingerprint
  `5b54181bdc554d5685759baf55bd55bb9a118d889446237c6a7fd0be0ab9fbd2`
  and 18 logical CPU affinity.
- Model, confidence/NMS, ByteTrack thresholds/association and Common Path config
  are identical. Resolved config comparison reports exactly one difference:
  `detector.tile_regions_normalized`.
- Candidate normalized FAR crops:
  `[0,0,0.55625,0.555556]` and
  `[0.44375,0,1,0.555556]`. At 1280×720 these map to the baseline upper tiles
  `(0,0,712,400)` and `(568,0,1280,400)`. Crop mapping, tile-edge filtering and
  merge/NMS are unchanged.

## 1. Five-image baseline vs three-image candidate

Baseline run `shibuya-25s-tile5-a-20261003`:

- 1 full frame + 4 tiles; 5 detector images/frame and 10/GPU batch.
- 5.916 FPS; 126.774 processing seconds.
- Detector wall p50/p95: 186.743/262.576 ms.
- Preprocess: 65.650/87.342 ms; forward: 66.809/70.600 ms.
- Tile merge: 28.552/51.695 ms; ByteTrack: 33.945/57.823 ms.
- Queue age: 109.598/912.948 ms; E2E: 1,560.934/2,465.885 ms.
- CPU average/peak: 143.870%/182.4%; GPU: 45.648%/74%; VRAM peak 1,717 MB.

Candidate run `shibuya-25s-tile3-b-20261003`:

- 1 full frame + 2 FAR tiles; 3 detector images/frame and 6/GPU batch.
- 10.115 FPS; 74.149 processing seconds.
- Detector wall p50/p95: 105.396/151.342 ms.
- Preprocess: 30.282/41.787 ms; forward: 40.009/54.466 ms.
- Tile merge: 15.611/18.726 ms; ByteTrack: 19.885/26.018 ms.
- Queue age: 75.776/669.909 ms; E2E: 890.786/1,467.659 ms.
- CPU average/peak: 147.703%/187.7%; GPU: 40.493%/69%; VRAM peak 1,155 MB.

## 2. FPS gain

- FPS increased **5.916 → 10.115**, or **+70.977%**.
- Processing time fell 41.510%.
- Detector-wall p50/p95 fell 43.559%/42.363%.
- E2E p50/p95 fell 42.933%/40.482%.

## 3. Detector workload reduction

- Detector images: 3,750 → 2,250, exactly **−40%**.
- Detector invocations remain 375; association updates remain 750. Every source
  frame still runs detection, with zero policy skips and zero deadline misses.
- Preprocess p50 fell 53.874%; forward p50 fell 40.113%; tile-merge p50 fell
  45.324%.
- GPU average/peak fell 45.648%/74% → 40.493%/69%; shorter six-image batches
  finish faster. VRAM peak fell by 562 MB. CPU average rose 3.833 percentage
  points, within the same two-core allocation.

## 4. FAR detection regression

The TOP/FAR gate itself passes relative to the baseline:

- TOP/FAR detections: **10,482 → 10,482**, zero count delta; 100% IoU≥0.75
  match and zero bbox/confidence difference.
- TOP/FAR track-state rows: 9,370 → 9,371 (+0.011%). IoU≥0.5 baseline match
  is 91.441%; mapped ID agreement is 71.300%.

However, global detector quality fails because the full-frame pass cannot replace
the removed high-resolution lower tiles:

- Total detections: 182,311 → 108,914 (**−40.259%**).
- MIDDLE detections: 101,098 → 69,705 (**−31.052%**).
- BOTTOM/NEAR detections: 70,731 → 28,727 (**−59.386%**).
- Only 58.172% of baseline detections have a candidate IoU≥0.5 match.

## 5. Tracking regression

- Unique Track IDs: 413 → 240 (**−41.889%**).
- Observed track rows: 116,614 → 66,424 (−43.039%); all track-state rows:
  171,038 → 105,025 (−38.596%).
- Adjacent-frame observed continuity fell 59.290% → 53.538%.
- Mapped Track-ID agreement over all IoU≥0.5 track matches is only 35.746%.
- Absolute fragments fall 1,867 → 1,261 because many tracks never exist in the
  candidate. Normalized fragments/unique-ID worsen 4.521 → 5.254 (+16.225%).
- Lost/recovered events per unique ID worsen 3.521 → 4.254 (+20.836%).
- MIDDLE track-state rows fall 28.532%; BOTTOM/NEAR fall 59.004%.
- Visual review of TOP/FAR, MIDDLE, BOTTOM/NEAR, crossing and occlusion agrees:
  FAR boxes are stable, while middle/near crowds lose many boxes. At eight fixed
  frames, baseline people count spans 148–330 versus 107–179 for the candidate.

## 6. Common Path regression

- Both runs end with three paths, but all Path IDs change.
- Final directions change from `FORWARD, FORWARD, REVERSE` to
  `FORWARD, FORWARD, FORWARD`: the third path has a material
  **REVERSE → FORWARD flip**.
- Support changes 11→6, 6→5 and 3→4. The primary path loses 45.455% support.
- Confidence deltas are 0.00057, 0.03434 and 0.01212. All polylines change point
  count, so a direct pointwise distance would be misleading; the rendered route
  geometry is visibly different.
- Path-created events fall 45→34 and cooling events 4→2, consistent with less
  usable track evidence rather than improved stability.

## 7. Queue/backlog change

- Result queue age p50/p95 improves 109.598/912.948 ms →
  75.776/669.909 ms (−30.861%/−26.621%).
- Consumer wait p50/p95 improves 64.877/93.512 ms → 20.249/38.921 ms.
- E2E p50/p95 improves 1,560.934/2,465.885 ms → 890.786/1,467.659 ms.
- Common Path compute also falls 245.801/805.814 ms → 213.271/464.768 ms,
  mainly because the candidate supplies far fewer tracks/evidence.

## 8. PASS/FAIL

**FAIL.** The +70.977% FPS gain and exact FAR preservation do not compensate for
31–59% regional detection loss, 41.9% fewer track IDs, worse normalized
fragmentation/continuity, and a Common Path direction flip. The candidate is not
promoted.

Rollback requires no production edit: keep
`configs/shibuya-overlap-batch2.yaml`, whose
`detector.tile_regions_normalized` is empty and therefore retains the historical
1 full + 4 tile grid. The experimental config remains
`configs/shibuya-overlap-batch2-far2.yaml` for reproducibility only.

Validation: `python -m pytest -q` — **207 passed**.
