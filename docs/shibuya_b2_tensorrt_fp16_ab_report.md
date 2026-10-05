# Shibuya B2: PyTorch FP32 vs TensorRT FP16 trên Tesla T4

Ngày đo: 2026-10-03. Kết luận: **FAIL quality gate; không promote TensorRT FP16 lên production**.
Backend TensorRT được giữ sau explicit experimental switch để điều tra tiếp; default và
`configs/shibuya-overlap-batch2.yaml` vẫn dùng PyTorch FP32.

## 1. Video, config và provenance

- Input: `data/videos/data-shibuya-5m.mp4`, đoạn `0.00–25.00 s`.
- Output: 750/750 frame, frame ID 0–749, không drop/reorder; cả hai MP4 decode đủ
  750 frame, duration 24.998333 s.
- GPU: Tesla T4; CPU request: 2 physical cores.
- Config: `configs/shibuya-overlap-batch2.yaml`; `source_batch=2`, 5 detector images/frame,
  static detector batch 10, input `[10,3,1280,1280]`.
- Config hash hai nhánh giống nhau:
  `a372a1ae70dd7e2a71a80bb2ce1bc1847b72ab464e911e74baf02c0e37dc9055`.
- Scheduler trace hash hai nhánh giống nhau:
  `95ec90a11ceb83097a6b5b5d43622366065db6c3d9388568eb5fad9acf2a8ec6`.
- Git HEAD: `4269fc8f469653e8a9b740245fe33be80b677467`; worktree có thay đổi chưa commit.
- Remote code fingerprint: `573fe9852922ec3acc508cad67f64b6f30fcd2a36feb21e61be5a8bde841efe6`.
- Runtime: Python 3.11.12, Ultralytics 8.4.155, Torch 2.14.0 cho benchmark;
  TensorRT 10.16.1.11. Không dùng INT8/FP8.

Phép đo cuối chạy A rồi B liền nhau, không download xen giữa. Hai run có cùng CPU
affinity 18 logical CPUs; TensorRT bắt đầu 3.019 giây sau khi FP32 hoàn tất. Preprocess
p50 gần như bằng nhau (69.309 và 69.380 ms), nên phép so sánh không còn bị nhiễu CPU
host lớn như các run chẩn đoán trước.

## 2. Performance A/B cuối

| Metric | PyTorch FP32 B2 | TensorRT FP16 B2 | Thay đổi |
|---|---:|---:|---:|
| Processing FPS | 5.893 | 7.192 | **+22.043%** |
| Processing time | 127.262 s | 104.284 s | -18.056% |
| Preprocess p50 / p95 | 69.309 / 88.563 ms | 69.380 / 89.506 ms | gần như không đổi |
| H2D p50 / p95 | 5.600 / 11.298 ms | 5.576 / 10.577 ms | -0.429% / -6.382% |
| Forward p50 / p95 | 66.859 / 69.874 ms | 31.399 / 37.125 ms | **-53.037% / -46.868%** |
| YOLO postprocess p50 / p95 | 4.650 / 60.992 ms | 5.205 / 59.377 ms | +11.935% / -2.648% |
| Detector wall p50 / p95 | 147.157 / 207.956 ms | 111.460 / 179.531 ms | **-24.258% / -13.669%** |
| E2E p50 / p95 | 1540.910 / 2383.030 ms | 1203.241 / 2388.382 ms | -21.914% / +0.225% |
| CPU avg / peak | 142.298% / 181.1% | 146.711% / 183.0% | +3.101% / +1.049% |
| GPU avg / peak (1 Hz) | 41.640% / 74% | 28.210% / 40% | -13.430 pp / -34 pp |
| Consumer result wait p50 / p95 | 67.068 / 97.128 ms | 35.011 / 58.076 ms | -47.799% / -40.203% |

Baseline production đã khóa trước instrumentation là 6.223 FPS. So với mốc đó,
TensorRT đo được 7.192 FPS, tương đương **+15.571%**. Con số +22.043% phía trên là
phép A/B kiểm soát cùng instrumentation; không trộn hai loại baseline.

GPU utilization giảm không có nghĩa TensorRT chậm hơn: forward ngắn hơn một nửa nên
GPU hoạt động theo burst ngắn, trong khi sampler chỉ lấy mẫu mỗi giây và CPU preprocess
trở thành critical path. Đây là bằng chứng pipeline còn CPU-bound/GPU-starved sau TensorRT.

### VRAM và engine

- Isolated NVIDIA VRAM peak: FP32 1709 MB; TensorRT 1177 MB, giảm 532 MB (31.1%).
- Paired warm-container TensorRT báo 2189 MB do CUDA allocator của run FP32 còn resident;
  không dùng số này để so backend độc lập.
- Torch allocator peak trong paired run: 1014.38 MB FP32 và 604.18 MB TensorRT.
- Engine SHA-256:
  `69ed984f395cef890c74968d7e3883a337fac5034b28f7ab557606d21b5c2669`.
- Engine size: 7,129,044 bytes (6.799 MiB).
- One-time engine build trên T4: 603.348 s.
- Warm-container detector init/warmup TensorRT: 1.171 ms / 1223.124 ms.
- Isolated cold-container detector init/warmup TensorRT: 287.086 ms / 6459.963 ms.

## 3. Detection regression

So sánh one-to-one theo IoU, baseline chỉ là regression reference chứ không phải ground truth.

- Detection count: 182,311 FP32 và 181,939 TensorRT; giảm 372 (-0.204%).
- Chỉ 111/750 frame (14.8%) có count bằng nhau; mean absolute delta 1.981 detection/frame.
- 179,856 cặp match IoU≥0.5, tương đương 98.653% baseline và 98.855% candidate.
- Unmatched: 2,455 baseline observations và 2,083 candidate observations.
- Bbox absolute error: median 0.0827 px, p95 0.337341 px, max 21.115173 px.
- Confidence absolute error: median 0.000548, p95 0.002628, max 0.19741264.
- Mọi cặp match đều có ít nhất một numeric field lệch trên tolerance chặt của report;
  phần lớn lệch nhỏ, nhưng các outlier/count changes đủ để làm association rẽ nhánh.

## 4. Tracking regression

- Observed track rows: 116,614 FP32 và 116,201 TensorRT (-0.354%).
- Unique observed Track IDs: 413 và 412.
- Trong 112,132 track observations match IoU≥0.5, chỉ 7,254 (6.469%) giữ raw Track ID.
- Sau greedy one-to-one ID remapping để loại ảnh hưởng renumbering thuần, chỉ 39,287
  observations (35.036%) nhất quán. Vì vậy khác biệt không chỉ là offset ID.
- Fragment count: 7,845 → 7,888 (+43); gap events: 7,432 → 7,476 (+44).
- Adjacent-frame continuity proxy: 59.290% → 59.049% (-0.241 percentage point).

Không có ground-truth MOT nên đây không phải IDF1/HOTA. Tuy vậy, mức association divergence
và fragmentation tăng đã vượt điều kiện “sai khác số học nhỏ”. Tracking gate **FAIL**.

## 5. Common Path regression

| Rank | FP32 | TensorRT | Direction | Support | Confidence |
|---|---|---|---|---|---|
| 1 | path-019 | path-008 | FORWARD → FORWARD | 11 → 9 | 0.616800 → 0.669355 |
| 2 | path-020 | path-015 | FORWARD → FORWARD | 6 → 7 | 0.617543 → 0.647444 |
| 3 | path-006 | path-014 | **REVERSE → FORWARD** | 3 → 4 | 0.610917 → 0.675609 |

Cả ba polyline đổi số điểm (14→16, 10→13, 21→10). Rank 3 đổi hướng, vì vậy đây là
regression thực chất, không phải chỉ path ID renumbering. Common Path gate **FAIL**.

## 6. Bottleneck mới sau TensorRT

Forward không còn là bottleneck lớn nhất. Hot path mới là CPU preprocessing và các stage
consumer tuần tự:

- preprocess p50 69.380 ms, trong đó pre-transform 25.004 ms, stack 13.997 ms,
  layout/contiguous copy 18.996 ms và H2D 5.576 ms;
- consumer result wait p50 35.011 ms;
- ByteTrack p50 31.880 ms;
- tile merge p50 28.692 ms;
- TensorRT forward p50 31.399 ms.

GPU avg giảm còn 28.210% vì CPU không cấp batch liên tục sau khi forward được rút ngắn.
Result-queue age p95 tăng 836.537 → 1018.412 ms và E2E p95 không cải thiện, cho thấy tail
latency/backlog vẫn nằm ngoài TensorRT forward.

## 7. Quyết định

TensorRT FP16 đạt performance gate nhưng **không đạt quality gate**. Không dùng nó làm
production candidate ở trạng thái hiện tại; production/default tiếp tục là PyTorch FP32.
Không thử INT8/FP8 trong task này.

Nếu tiếp tục nghiên cứu, bước kế tiếp phải là điều tra nguồn divergence tại raw per-tile
outputs/NMS boundary và các detection gần confidence threshold, sau đó đánh giá trên tập
MOT có nhãn. Không được nới threshold hoặc đổi ByteTrack/Common Path để hợp thức hóa FP16.

## 8. Verification và artifact

- Full test suite: 201 passed.
- Quality JSON: `quality_vs_torch_fp32.json` trong thư mục TensorRT paired artifact.
- Hai video đều bật bounding boxes và đã được `ffprobe` xác nhận 750 frame.
- Recursive Modal directory download trên Windows làm hỏng newline của một bản JSONL dù
  giữ nguyên byte-size. Quality diff dùng bản TensorRT cache tải per-file có SHA khớp
  manifest; không dùng bản recursive-download sai hash.
