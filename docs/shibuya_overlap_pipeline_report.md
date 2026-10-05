# Shibuya T4 CPU–GPU overlap experiment

Ngày đo: 2026-10-02  
Nguồn: `data/videos/data-shibuya-5m.mp4`, đoạn `0–25 s`, 750 frame  
Model: `yolo26n.pt`, FP32, SHA-256 `9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef`  
Input SHA-256: `f3b845b6eb312086bb9dbc408986bea3277ba887e62a56212e7497a3fd5c9670`  
Git HEAD: `4269fc8f469653e8a9b740245fe33be80b677467`

## Thay đổi experimental

- Thêm producer đọc frame và GPU worker, nối bằng hai queue bounded theo số batch; cấu hình thử nghiệm dùng capacity 2.
- GPU worker chỉ chuẩn bị tile và chạy một `predict()` trên batch phẳng. Raw detection CPU được đưa vào result queue.
- Main consumer merge tile, update ByteTrack, Common Path, render, encode và ghi cache theo đúng `frame_id` tuần tự.
- Không drop/reorder frame; partial batch ở EOF được drain; worker lỗi được truyền về main thread.
- Mặc định `offline_overlap_enabled=false`, nên profile sản xuất cũ vẫn là rollback path.
- Không đổi model, detector thresholds/tile policy, tracker, Common Path hoặc TensorRT.

## A/B trên Tesla T4

| Run | Source batch | Detector batch | FPS | GPU avg/peak | CPU avg/peak | VRAM peak | E2E p50/p95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 1 | 5 | 4.116 | 31.269% / 57% | 100.107% / 112.6% | 1,187 MB | 226.807 / 305.541 ms* |
| Overlap B1 | 1 | 5 | 5.564 | 38.860% / 65% | 136.837% / 161.6% | 1,222 MB | 828.271 / 1,661.999 ms |
| Overlap B2 | 2 | 10 | **5.780** | **43.608% / 75%** | 138.561% / 163.4% | 1,709 MB | 1,564.099 / 2,923.554 ms |
| Overlap B3 | 3 | 15 | 5.916 | 41.156% / 100% | 142.989% / 175.5% | 2,356 MB | 2,200.725 / 4,306.113 ms |

`*` Baseline không có queue; frame total p50/p95 được dùng làm E2E tương đương.

So với baseline:

- B1: FPS `+35.18%`; GPU average `+7.591` điểm phần trăm.
- B2: FPS **`+40.43%`**; GPU average **`+12.339` điểm phần trăm** (`+39.46%` tương đối).
- B3: FPS `+43.73%`; GPU average `+9.887` điểm phần trăm. Peak 100% chỉ là đỉnh mẫu, không phản ánh mức sử dụng bền vững.

## Stage p50/p95 (ms trên mỗi source frame)

| Stage | Baseline | B1 | B2 | B3 |
|---|---:|---:|---:|---:|
| YOLO forward | 71.241 / 74.024 | 71.341 / 109.504 | 67.757 / 77.458 | 68.369 / 73.645 |
| Preprocess + H2D | 36.057 / 39.122 | 63.036 / 81.051 | 67.969 / 82.362 | 67.575 / 79.628 |
| Tile merge | 30.699 / 50.775 | 34.024 / 60.746 | 32.486 / 58.500 | 29.970 / 56.199 |
| ByteTrack | 38.531 / 62.926 | 45.866 / 70.716 | 43.867 / 71.317 | 39.214 / 66.356 |
| GPU input queue wait | N/A | 0.034 / 0.109 | 0.019 / 0.060 | 0.015 / 0.041 |
| Consumer result wait | N/A | 29.930 / 71.057 | 23.691 / 70.374 | 29.187 / 72.843 |
| Result queue age | N/A | 37.009 / 629.719 | 145.019 / 1,243.895 | 202.125 / 1,774.859 |

Preprocess/H2D và các CPU stage chậm hơn khi overlap vì GPU worker tranh CPU với merge/tracker/render. Dù latency từng frame tăng, tổng throughput tăng do CPU và GPU chạy đồng thời.

## Queue, idle và starvation

- Cả B1/B2/B3 đều đạt input/result queue peak 2/2 batch; queue không tăng vô hạn.
- B2 GPU input wait tổng `46.369 ms` trong `129.767 s`, p95 `0.060 ms/frame`: decode/producer không còn làm GPU đói dữ liệu.
- B2 result queue put wait tổng `6,137.428 ms` và result queue age p95 `1,243.895 ms`: GPU worker đôi lúc bị backpressure vì CPU consumer không drain kịp.
- Producer put wait B2 `117,608.189 ms` là backpressure có chủ đích của bounded queue, không phải drop. Tất cả run đều có đủ 750 frame, 3,750 detector images và frame/timestamp khớp baseline.
- GPU starvation đã giảm nhưng chưa hết: average tăng từ 31.269% lên 43.608%. Bottleneck hệ thống chuyển sang chuỗi CPU consumer (merge + ByteTrack + analytics/render/encode) và CPU preprocess/H2D trong GPU worker. Một model invocation vẫn đồng bộ ở mức Ultralytics.

## Regression detection, tracking và Common Path

So sánh toàn bộ cache 750 frame với baseline; đây là regression proxy, không phải ground truth/mAP/HOTA.

- B1: detection và observed tracks khớp 100%; Track ID agreement 100%; 413 unique IDs.
- B2: detection và observed tracks khớp 100%; Track ID agreement 100%; 413 unique IDs.
- B3: detection vẫn khớp 100%, nhưng observed-track count lệch `+0.008%`, equal-count frames còn `98.667%`, Track ID agreement còn `97.594%`, unique IDs tăng `413 → 414`.
- Final Common Path của B1/B2 khớp baseline ở 3 path ID, direction, support, confidence và số điểm polyline. B3 giữ cùng ba path nhưng confidence của `path-019` đổi `0.616800 → 0.617111`, phù hợp với lệch tracking nhỏ.
- Không run nào drop/reorder frame; comparator xác nhận đủ 750 frame và timestamp tương ứng.

## Kết luận sáu câu hỏi

1. **GPU utilization tăng bao nhiêu?** Ứng viên B2 tăng average từ `31.269%` lên `43.608%`: `+12.339` điểm phần trăm; peak `57% → 75%`.
2. **FPS tăng bao nhiêu?** `4.116 → 5.780 FPS`, tăng **40.43%**. Vẫn chậm hơn source 30 FPS khoảng 5.19 lần.
3. **GPU còn starvation không?** Có, nhưng đã giảm. GPU input queue gần như luôn có dữ liệu; phần starvation còn lại đến từ CPU preprocess/synchronization và backpressure của result queue khi merge/tracker/output không drain kịp.
4. **Bottleneck mới ở đâu?** CPU consumer và contention CPU: preprocess+H2D p50 `67.969 ms`, ByteTrack `43.867 ms`, merge `32.486 ms`; result queue age p95 `1.244 s` chứng minh backlog nằm sau inference.
5. **Source batch tối ưu trên T4?** **Batch 2** cho thử nghiệm này. B3 chỉ nhanh hơn B2 `2.35%`, GPU average thấp hơn `2.452` điểm, VRAM cao hơn 647 MB, E2E p95 cao hơn `47.3%` và có tracking regression.
6. **Regression tracking/Common Path?** Không thấy ở B1/B2 trên regression proxy 750 frame; B3 có regression nhỏ nhưng đo được, nên không được chọn.

## Kiểm thử và artefact

- Toàn bộ test suite: pass.
- Unit test mới kiểm tra queue capacity 1, partial EOF batch, không drop và thứ tự frame.
- Raw artefact:
  - Baseline: `outputs/common_path/shibuya-25s-t4-profile-20261002-143400/`
  - B1: `outputs/common_path/shibuya-25s-overlap-b1-20261002/`
  - B2: `outputs/common_path/shibuya-25s-overlap-b2-20261002/`
  - B3: `outputs/common_path/shibuya-25s-overlap-b3-20261002/`
  - Regression JSON: `quality_vs_baseline.json` trong từng thư mục B1/B2/B3.

## Khuyến nghị triển khai

Giữ `configs/shibuya-overlap-batch2.yaml` làm candidate experimental, chưa thay `configs/shibuya.yaml`. Bước tiếp theo nên tối ưu/drain CPU consumer và giảm E2E backlog trước khi cân nhắc batch lớn hơn; không có bằng chứng để dùng B3 hoặc TensorRT ở task này.
