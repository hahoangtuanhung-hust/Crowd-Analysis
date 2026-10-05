# Shibuya 25-second T4 pipeline profile

Run ID: `shibuya-25s-t4-profile-20261002-143400`

## 1. Video/config/commit test

- Source: `data/videos/data-shibuya-5m.mp4`, interval `0.00–25.00 s`; 750 frames processed (`0.00–24.94 s`).
- Input SHA-256: `f3b845b6eb312086bb9dbc408986bea3277ba887e62a56212e7497a3fd5c9670`.
- Model: `yolo26n.pt`, SHA-256 `9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef`.
- Resolved config: `configs/shibuya.yaml`, config hash `e9958480fb9e1b0d861c7a7fe1f5129dedd582dc45edd1c4097a8a10dad5dc1c`.
- Git HEAD: `4269fc8f469653e8a9b740245fe33be80b677467`; code fingerprint captured by the run: `558d2240c4510d7864fbd7483ea534479ba034212aee0fe200f564e2dc51b3e5`.
- Hardware: Tesla T4; Modal CPU request 2 physical cores; FP32; tiled inference enabled; motion ROI disabled; inference interval 1.
- No detector/tracker/Common Path policy or TensorRT change was made. The only code change before measurement exported additional stage telemetry already produced by the detector.

## 2. Processing FPS

- Frame-loop time: `182.223 s` for 750 frames.
- End-to-end processing throughput: **4.116 FPS**.
- Source rate: about 30 FPS; processing is about **7.29× slower than realtime**.
- Warm-up: `3.947 s`; Modal remote wall time: `187.205 s`.

## 3. GPU avg/peak utilization

- GPU utilization: **31.269% average**, **57% peak**, p50 `38%`, p95 `46%` over 175 one-second samples.
- 30/175 samples (`17.1%`) were at or below 10%; 40/175 (`22.9%`) were at or below 20%.
- GPU memory: `1,187 MB` device peak; PyTorch allocated peak `769.36 MB`, reserved peak `1,028 MB`.
- Process CPU: `100.107%` average, `112.6%` peak; p50 `99.4%`, p95 `105.3%`. This is approximately one fully occupied CPU core despite a two-core request. Per-thread utilization was not emitted by the current sampler; configured counts were PyTorch intra-op 2, inter-op 2 and OpenCV 1.

## 4. Stage timing p50/p95

Values are milliseconds per source frame. Means and total wall contribution are included to rank bottlenecks without over-weighting rare p95 spikes.

- Decode/capture: p50 `1.191`, p95 `1.556`, mean `1.239`, total `0.929 s`.
- Tile list/crop-view preparation: p50 `0.141`, p95 `0.171`, mean `0.141`, total `0.106 s`.
- YOLO preprocess including host-to-device transfer: p50 `36.057`, p95 `39.122`, mean `36.236`, total `27.177 s`.
- YOLO model forward: p50 `71.241`, p95 `74.024`, mean `69.748`, total `52.311 s`.
- YOLO internal postprocess: p50 `5.044`, p95 `5.677`, mean `5.171`, total `3.878 s`.
- Result D2H and Python conversion (`detach().cpu().numpy()`): p50 `1.707`, p95 `2.176`, mean `1.774`, total `1.330 s`.
- Tile merge/NMS: p50 `30.699`, p95 `50.775`, mean `33.366`, total `25.024 s`.
- ByteTrack: p50 `38.531`, p95 `62.926`, mean `39.812`, total `29.859 s`.
- Analytics/Common Path aggregate: p50 `1.217`, p95 `23.535`, mean `13.794`, total `10.345 s`. The periodic Common Path compute ran 50 times with p50 `162.975` and p95 `473.374 ms`.
- Render: p50 `4.021`, p95 `20.262`, mean `9.195`, total `6.896 s`.
- Encode: p50 `7.193`, p95 `9.226`, mean `7.283`, total `5.462 s`.
- Cache write: p50 `3.162`, p95 `5.747`, mean `3.324`, total `2.493 s`.
- Unattributed synchronous pipeline overhead: p50 `20.834`, p95 `28.341`, mean `19.323`, total `14.492 s`.
- Full frame: p50 `226.807`, p95 `305.541`, mean `241.281 ms`.

H2D is not separately exposed by Ultralytics 8.4.116: its `preprocess()` performs NumPy stack/letterbox, layout/color conversion, `torch.from_numpy`, `.to(device)`, FP32 conversion and normalization as one timed region. Therefore the measured `36.236 ms` is an exact combined preprocess+H2D figure, not an invented H2D-only estimate. D2H is measured separately by the project wrapper.

## 5. GPU idle/starvation evidence

- Each source frame had exactly `model_invocations=1`, `detector_batch_size=5`, `inference_images=5`; totals were 750 invocations and 3,750 detector images. The full frame plus four tiles are therefore one GPU batch, not five sequential `predict()` calls.
- Mean GPU forward was `69.748 ms` of `241.281 ms` frame time (**28.91%**). The remaining `171.533 ms` per frame occurs outside the model forward. This duty-cycle estimate closely matches measured average GPU utilization `31.269%`.
- The batch path is synchronous and has `source_batch_size=1`: decode → one five-image predict → D2H/merge → ByteTrack → analytics → render → encode/cache. There is no offline prefetch queue and no next-frame GPU work overlapping these CPU stages. Queue wait is therefore not a hidden backlog metric; it is structurally absent. The measured unattributed/idle overhead is `19.323 ms` mean, while all CPU stages between model forwards also leave the GPU without the next batch.
- CPU p50 `99.4%` together with low GPU utilization and the serial stage order is direct starvation evidence: roughly one CPU core feeds the T4, and the next batch is not submitted while merge/tracking/analytics/render/encode execute.
- Project-level `profile_gpu` was disabled for the throughput run, so it did not add extra outer synchronization. However Ultralytics 8.4.116 itself calls accelerator synchronization at entry/exit of each internal preprocess, inference and postprocess timer. The project then calls `.detach().cpu().numpy()` separately for `xyxy`, confidence and class tensors for each of five results. Those transfers synchronize/copy results, but their measured `1.774 ms` mean is not a top bottleneck.

## 6. Top 3 bottlenecks

Ranked by non-overlapping total measured time:

1. **YOLO model forward:** `52.311 s` total, `69.748 ms/frame`, 28.91% of frame time.
2. **ByteTrack:** `29.859 s` total, `39.812 ms/frame`, 16.50%.
3. **YOLO preprocess + H2D:** `27.177 s` total, `36.236 ms/frame`, 15.02%.

Tile merge/NMS is a close fourth at `25.024 s`, `33.366 ms/frame`, 13.83%. Although YOLO forward is the largest single stage, the larger system bottleneck is the lack of overlap: all CPU work serializes the next GPU submission.

## 7. Why the GPU is underutilized

The T4 is not starved by video decode (`1.239 ms`) or creating tile views (`0.141 ms`). It is starved because the offline runner submits one five-image detector batch for one source frame, then waits synchronously through about `130 ms/frame` outside the Ultralytics wrapper before submitting the next batch. ByteTrack, tile merge, periodic Common Path work, rendering, encoding, cache logging and general loop overhead all run serially. Internal Ultralytics timing synchronization and result materialization enforce additional phase boundaries, but measured result D2H is small. Low VRAM use is consistent with batch 5 and does not itself prove a memory bottleneck.

## 8. Three optimizations to try next

No optimization below was implemented in this profile.

1. **Bounded multi-frame detector batching / CPU–GPU overlap.** A/B the existing `source_batch_size=2` candidate while keeping five detector images per frame and ordered tracker/analytics consumption. This targets the `~130 ms` CPU gap between GPU submissions and has the highest expected end-to-end ROI. Validate detection/track equivalence and queue drain counts.
2. **Optimize ByteTrack and tile merge hot paths without changing policy.** Profile assignment matrix construction/LAP and source-aware duplicate merge; vectorize/reuse buffers and reduce Python object churn. Together these stages consume `64.164 ms/frame` mean, nearly as much as GPU forward.
3. **Decouple render/encode/cache and periodic Common Path from detector submission with bounded queues.** Preserve frame order and backpressure, but let the T4 begin the next detector batch while CPU outputs for the previous batch finish. Measure queue age/drops, memory and full-drain correctness; do not increase Modal container concurrency for one video.

## Artefacts

- Raw per-frame metrics: `outputs/common_path/shibuya-25s-t4-profile-20261002-143400/metrics.csv`
- Stage percentiles: `outputs/common_path/shibuya-25s-t4-profile-20261002-143400/stage_metrics.csv`
- Resource samples/summary: `outputs/common_path/shibuya-25s-t4-profile-20261002-143400/resource_metrics.jsonl`, `resource_summary.json`
- Resolved configuration/provenance: `config_resolved.yaml`, `provenance.json`
- Rendered 25-second output: `tracked_points_common_path.mp4`
