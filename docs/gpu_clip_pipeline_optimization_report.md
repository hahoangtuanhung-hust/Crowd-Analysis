# Báo cáo tối ưu `gpu_clip`

Ngày cập nhật: 30/09/2026 (Asia/Saigon)

## Kết luận hiện tại

Trạng thái: `NOT_MET` — A/B mới đã cải thiện throughput nhưng vẫn còn xa mục tiêu 30 FPS.
Chất lượng FP16: `QUALITY_PENDING`.

Run lịch sử Shibuya 10 phút đạt 3,88 FPS trên Tesla T4. Smoke mới trên cùng đoạn 25 giây
(750 frame) cho kết quả throughput tốt nhất ở FP16 source batch 2 với 4,845 FPS. Đây mới là
performance candidate, chưa phải production candidate vì resolved tracker của baseline và
FP16 trong bốn run cũ không đồng nhất. Bottleneck đã đo là wrapper
Ultralytics predict khoảng 114 ms/frame, project postprocess/merge khoảng 33 ms/frame và
tracking khoảng 43 ms/frame. Decode khoảng 1,2 ms/frame, vì vậy chưa có bằng chứng để ưu
tiên thay decoder.

## Đường chạy và ownership

`modal_common_path.main` upload file theo hash và gọi một `gpu_clip` container. Bên trong
container, `scripts.common_path_clip.run` sở hữu duy nhất một `cv2.VideoCapture`, một detector,
một tracker và một engine Common Path. Đường batch hiện tại tuần tự:

```text
VideoCapture.read
  -> detector.detect_batch (mỗi source frame gồm 1 full frame + 4 tile)
  -> result transfer + source-aware tile merge
  -> ByteTrack.update theo frame ID nguồn
  -> Common Path update theo source timestamp
  -> render -> MP4 encode -> cache JSONL
```

Không có capture queue/prefetch bên trong `gpu_clip`; capture/inference workers của web
`TrackingPipeline` không được dùng ở đường Modal batch. Không có PTS pacing trong
`offline_fast`. Tracker, analytics và artifact writer đều single-writer.

## Kết quả audit batching

- Shibuya tiled đã batch năm ảnh của cùng một source frame vào **một** `model.predict` call.
- Actual source-image batch size là 5. Tensor shape sau Ultralytics letterbox chưa được wrapper
  public hiện tại cung cấp; artifact ghi rõ shape quan sát là shape ảnh nguồn trước letterbox.
- Baseline/FP16 ablation giữ source batch 1. Các profile experimental đã được bổ sung:
  `shibuya-fp32-batch2.yaml` cô lập batching ở FP32,
  `shibuya-fp16-batch2.yaml` dùng 2 source frame/10 detector images và
  `shibuya-fp16-batch4.yaml` dùng tối đa 4 source frame/20 detector images. Sau A/B,
  `shibuya-realtime-candidate.yaml` đã được chốt ở 2 source frame/10 detector images.
- Detector có thể infer nhiều source frame trong một model call, nhưng kết quả được tách theo
  frame và ByteTrack/Common Path vẫn update tuần tự theo frame ID/source timestamp.
- Batch cuối ở EOF được flush dù chưa đủ kích thước cấu hình.
- Motion ROI có feedback từ track trước; mọi thiết kế cross-frame sau này phải tự fallback về
  source batch 1 khi scheduler này bật.

## Thay đổi đã triển khai

- Warm-up đúng detector instance và đúng source/detector batch shape dùng trong timed loop.
- Throughput run không còn `cuda.synchronize()` mỗi frame; synchronized profiling là opt-in.
- Sửa detector image/invocation accounting cho tiled inference và cache replay.
- Tách metric `ultralytics_predict_ms`, `result_transfer_ms`, `merge_ms`; không mô tả wrapper
  Ultralytics như GPU-forward thuần.
- Manifest chứa processing FPS, stage p50/p95, tổng model invocation, tổng detector images và
  detector images/source frame.
- Bounded offline microbatch có hai giới hạn độc lập: `source_batch_size` và
  `max_detector_images_per_batch`; không nạp cả video vào RAM/VRAM.
- Cache fingerprint bao gồm cấu hình batch vì Ultralytics padding theo batch có thể đổi output.
- Motion ROI, cache replay hoặc detector không hỗ trợ batch tự fallback source batch 1 và ghi
  lý do vào summary/provenance.
- Source-aware tile merge chỉ kiểm tra ignore region một lần.
- Common Path nearest-neighbour dùng partial selection thay full sort.
- Realtime coalesce preview cũ nhưng không bỏ analytics evidence.
- Benchmark runner ghi stage metrics thay vì chỉ wall time/exit code.
- Quality diff kiểm tra frame/timestamp, bbox IoU, unmatched box, observed-track proxy và
  toàn bộ khác biệt trong hai file `config_resolved.yaml`.

## Baseline và candidate

| Variant | CPU request/actual | Source/detector batch | FPS E2E | GPU util | Quality |
|---|---|---|---:|---:|---|
| Historical FP32 tiled | Không đủ provenance CPU request / ~1 core process | 1 / 5 | 3,88 | 31,025% avg | REVIEW_PENDING |
| FP32 baseline | 8 physical cores / actual resource summary chưa đính kèm | 1 / 5 | 4,109 | NOT_MEASURED | REVIEW_PENDING |
| FP16 batch 1 | 8 / ~1,00 core process | 1 / 5 | 4,569 | 25,181% avg | QUALITY_PENDING |
| FP16 batch 2 — performance candidate | 8 / ~1,00 core process | 2 / 10 | 4,845 | 22,219% avg | QUALITY_PENDING |
| FP16 batch 4 | 8 / ~1,01 core process | 4 / 20 | 4,741 | 19,883% avg | QUALITY_PENDING |

Số đo quan sát cho batch 2 cao hơn FP32 baseline 17,9% FPS và thấp hơn 15,2% processing
time, nhưng không được quy toàn bộ chênh lệch này cho FP16/batching: baseline dùng tracker
weight `0.45/0.30`, còn ba run FP16 cũ dùng `0.35/0.20`. So sánh sạch trong nhóm FP16 cho
thấy batch 2 tăng khoảng 6,0% so với batch 1; batch 4 chậm hơn batch 2 khoảng 2,1% và dùng
1.883 MB GPU memory thay vì 1.044 MB, nên bị loại. Mọi run vẫn xử lý đủ 750 frame, 3.750
detector images; invocation giảm đúng 750 → 375 → 188.

## Kiểm tra chất lượng artifact

- Bounding box xuất hiện trên toàn bộ sáu mốc contact sheet; người nhỏ và box sát rìa vẫn
  được giữ. Candidate xuất đúng ba Common Path theo `max_paths: 3`.
- FP32 baseline so với FP16 batch 2: 182.311 so với 182.122 detection (−0,104%); greedy
  one-to-one match đạt 98,677% theo phía baseline ở IoU ≥ 0,5, IoU trung vị 0,9767 và sai số
  tọa độ trung vị 0,0855 px.
- Cache FP16 cũ có 484 observed Track IDs so với 439 của baseline, nhưng đây là A/B sai
  tracker config. Rebuild cùng detection batch 2 bằng tracker baseline cho 438 IDs, cho thấy
  phần lớn chênh lệch tổng ID đến từ config tracker.
- Dù tổng ID đã khớp, track identity vẫn phân kỳ trong cảnh quá đông: chỉ 5,528% spatial
  matches giữ cùng numeric Track ID. Common Path của clip 25 giây cũng khác nhánh giữa các
  run. Không có nhãn IDF1/HOTA và clip chưa đủ dài để kết luận ổn định, nên trạng thái vẫn là
  `QUALITY_PENDING`, không promote candidate.
- Bằng chứng tái lập nằm trong
  `outputs/benchmark/quality_fp32_vs_fp16_batch2_original.json` và
  `outputs/benchmark/quality_fp32_vs_fp16_batch2_aligned_tracker.json`.

Không có dữ liệu billing nên cost/video-minute là `NOT_MEASURED`. Modal mặc định hiện đã giảm
từ 8 xuống 2 physical CPU và đồng bộ PyTorch/BLAS thread limits theo request; có thể rollback
bằng `--cpu 8`. Hiệu năng/cost của CPU=2 vẫn phải A/B cùng clip trước khi công bố là tốt hơn.

## Lệnh benchmark tái lập

Smoke 25 giây:

```powershell
python scripts/benchmark_realtime.py `
  --input data/videos/data-shibuya-5m.mp4 `
  --duration-seconds 25
```

Runner thực hiện inference thật và ghi `outputs/benchmark/benchmark_summary_v2.csv` nếu CSV
cũ có header không tương thích. Mặc định chỉ chạy ma trận trực giao `baseline`,
`fp32-batch2`, `fp16`, `fp16-batch2`; batch 4 và candidate là opt-in. Không thêm
`--profile-gpu` cho throughput run. Artifact cần được tải và review trước khi promote:

```powershell
python scripts/download_modal_artifacts.py `
  --run-id <RUN_ID> `
  --output-dir outputs/common_path/<RUN_ID>
```

Đối chiếu cache và resolved config:

```powershell
python -m scripts.compare_run_quality `
  --baseline-cache <BASELINE_RUN>/tracking_cache.jsonl `
  --candidate-cache <CANDIDATE_RUN>/tracking_cache.jsonl `
  --baseline-config <BASELINE_RUN>/config_resolved.yaml `
  --candidate-config <CANDIDATE_RUN>/config_resolved.yaml
```

## Bước benchmark còn thiếu

1. Chạy smoke `fp32-batch2` để cô lập lợi ích batching và chạy lại `candidate` với tracker
   đã đồng nhất; không dùng bốn run cũ làm A/B precision sạch.
2. Chạy hai ứng viên sạch trên cùng clip 65 giây, review crossing/occlusion và Common Path,
   sau đó full clip hai lần và soak trước khi đổi
   production profile.
3. Trên batch 2, A/B CPU request 8/2; chỉ thử `--cpu 1` nếu 2 không làm GPU chờ thêm.
4. Đo cost/video-minute khi có dữ liệu billing.

Không được gọi trạng thái hiện tại là full-frame 30 FPS hoặc sampled-live pass.

## Kiểm chứng local

- 174 Python tests pass.
- `compileall` pass cho backend, scripts và Modal entrypoint.
- `git diff --check` pass.
- Benchmark CLI hỗ trợ `baseline`, `fp32-batch2`, `fp16`, `fp16-batch2`, `fp16-batch4` và
  `candidate`.
- Bốn smoke performance run đã hoàn thành và artifact đã review; quality cuối cùng và cost
  vẫn chưa đo do thiếu ground truth/IDF1/HOTA và billing.
