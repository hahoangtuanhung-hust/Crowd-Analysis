# Tối ưu pipeline và kiểm thử Modal T4 — 06/10/2026

Đã chọn cấu hình `configs/shibuya-t4-optimized.yaml`: FPS tăng từ **5,982 lên
6,967 (+16,47%)**, giữ nguyên toàn bộ cache detection/tracking và lịch sử
support của Common Path. Profile phối cảnh nhanh hơn nhưng không đạt kiểm tra
chất lượng, được giữ riêng cho chẩn đoán.

## Thay đổi đã thực hiện

- Gộp box theo các khối NumPy có giới hạn bộ nhớ. Vẫn giữ thứ tự confidence,
  ngưỡng IoU/containment, phân biệt source và chỉ cho box đã được chọn loại box
  trùng. Frame ít box dùng kernel cũ để tránh chi phí tạo ma trận.
- Common Path tái sử dụng mẫu tracklet và vector hóa tính góc. Cache liên kết
  dùng đầy đủ hình học, tránh nhầm hai tuyến có cùng endpoint nhưng phần giữa
  khác nhau. Prune cache theo evidence còn hoạt động; mọi nhánh trả score 0
  cũng tuân thủ giới hạn cache. Sắp xếp event cùng thời điểm cho kết quả ổn định.
- Sửa lỗi detector phối cảnh bị bỏ qua trong `detect_batch` và pipeline overlap.
  Single frame, batch và overlap dùng cùng crop planner; tôn trọng số hàng FAR,
  nhóm inference theo cấu hình độ phân giải, rồi trả prediction đúng thứ tự
  source frame. Tọa độ box dùng kích thước resize thực tế riêng theo từng trục.
- Không cho kết hợp planner Motion ROI dùng lưới tile với detector phối cảnh
  có hình học khác, tránh báo sai vùng đã tìm kiếm.
- Giảm copy frame khi vẽ overlay; đường có opacity 1 vẽ trực tiếp vào output
  đã được copy, không copy/blend thêm toàn frame. Ảnh nguồn vẫn được bảo toàn.
- Áp dụng `detector_worker_mode` cho cả SessionManager và Modal live processor.
  Trước đây lựa chọn no-sync chủ yếu chỉ có tác dụng trong GPU clip runner.
- Sửa số batch nguồn để không bị nhân đôi bởi hai nhóm resolution; sửa overlay
  FPS đang bị hardcode 0 và đổi latency preview sang thời gian có hàng đợi.
- Modal CLI con dùng cùng Python interpreter với runner để tránh lệch client
  giữa môi trường `.venv` và Python cài toàn máy.

## Điều kiện đo

Ba job thực tế trên Modal **Tesla T4**, mỗi job yêu cầu **2 CPU physical cores**.
Video `data/videos/data-shibuya-5m.mp4`, đoạn 0–25 giây, 1280×720, 750 frame.
Source có FPS 30,002227; MP4 xuất có FPS 30,002. Đã decode đủ 750 frame của
cả ba video, kiểm tra contact sheet và frame cuối.

Model `yolo26n.pt`, PyTorch FP32, ByteTrack hybrid, Tracklet Aggregation,
detect mỗi frame, source batch 2, hàng đợi overlap 2 batch. Không chuyển sang
TensorRT, không bỏ frame trong benchmark. Warmup và khởi tạo model được báo
riêng, không nằm trong thời gian xử lý 750 frame.

- Baseline: `pipeline-baseline-20261006`, 125,378 giây, 5,982 FPS.
- Optimized: `pipeline-optimized-20261006`, 107,646 giây, 6,967 FPS.
- Perspective thử nghiệm: `pipeline-perspective-20261006`, 54,607 giây,
  13,735 FPS.

Đây là một lần chạy 25 giây cho mỗi profile; chưa phải benchmark lặp hoặc
kiểm tra vận hành liên tục 5–10 phút. FPS trên là throughput offline của toàn
pipeline có render, encode và ghi cache; chưa đạt realtime 30 FPS.

## Kết quả của cấu hình được chọn

- Tile merge p50: **27,276 → 15,426 ms**; p95: **50,643 → 27,883 ms**.
- Common Path compute p50: **224,795 → 164,099 ms**; p95:
  **753,423 → 426,477 ms**. Chỉ tính các lần compute theo lịch.
- End-to-end p50: **1.501,127 → 1.236,169 ms**; p95:
  **2.320,306 → 1.683,693 ms**, giảm khoảng 27,44% ở p95.
- GPU utilization trung bình: **42,927% → 49,613%**; CPU trung bình:
  **143,052% → 150,987%**. Utilization lấy mẫu mỗi giây; không coi đây là
  phép đo GPU liên tục. VRAM peak đều 1.717 MB.
- ByteTrack p50: **32,314 → 32,824 ms**; p95: **54,773 → 53,455 ms**.
  Không khẳng định bản thân thuật toán tracker chạy nhanh hơn đáng kể.

Detector forward FP32 vẫn khoảng 67 ms mỗi source frame sau amortization.
Preprocess vẫn là phần đáng kể. Giảm chi phí consumer giúp GPU được cấp việc
đều hơn; T4 chưa chạy liên tục ở mức sử dụng tối đa.

## Chất lượng detect, tracking và Common Path

- 750 frame theo đúng thứ tự 0–749; 750 association updates, 0 policy skips.
- **182.311 detection**, **413 unique Track ID** ở cả baseline và optimized.
- Cache detection/tracking có SHA-256 giống hệt:
  `de20748318ebd6db7a77abc5c50f861b3ad5f6f0b2a011bec1acfb1893a3a935`.
  Hash được tính lại từ các file đã tải xuống, khớp manifest Modal.
- `path_support.jsonl` giống từng byte. Ba path cuối giữ nguyên ID
  `path-019`, `path-020`, `path-006`, hướng FORWARD/FORWARD/REVERSE,
  support 11/6/3, confidence và toàn bộ tọa độ polyline. Sai lệch hình học 0 px.
- Cùng 49 path event với nội dung và source timestamp giống nhau; thứ tự
  một số event đồng thời khác nhau trong hai process. Source hiện đã sắp xếp
  thứ tự event. Cả hai run có 0 path switches.

Các lỗi về bỏ qua crop FAR khi chạy batch, tọa độ upscale, cache hình học và
coverage đã được sửa và có regression test. **Chưa có annotation Shibuya để
đo mức tăng recall, precision, IDF1 hoặc HOTA**; kết quả thực nghiệm của cấu
hình được chọn chứng minh giữ nguyên chất lượng tham chiếu và tăng tốc.
Số detection nhiều hơn không tự chứng minh chính xác hơn.

## Vì sao loại profile phối cảnh

Full frame + hai crop FAR 960, 3 detector images/frame, thực sự đã chạy cả hai
nhóm resolution trên T4 và xử lý đủ 750 frame. Tổng 375 source batches tạo
750 model invocations. Summary GPU cũ báo 750 inference batches do đếm theo
model invocation; source đã sửa riêng lỗi đếm này.

Tuy đạt 13,735 FPS, detection giảm **182.311 → 54.073 (−70,34%)**, unique ID
giảm **413 → 126**. Vùng FAR cũng giảm **10.482 → 6.449 detection** theo phép
phân vùng chẩn đoán upper-third. Common Path đổi ID, thứ hạng/hướng và hình học;
route cùng số điểm ở rank 2 lệch trung bình khoảng **364,5 px**. Contact sheet
cho thấy nhiều người trong đám đông không được giữ box. Vì vậy profile này
**không được chọn làm cấu hình chính**.

So sánh với baseline chỉ là regression reference, không phải ground truth.
Crop FAR của detector dùng y≤0,38, còn vùng FAR của công cụ quality comparison
là upper-third; không trộn hai định nghĩa khi đọc số liệu.

## Kiểm thử và phạm vi phiên bản

- Toàn bộ **216 test đạt**, gồm test mới cho phối cảnh single/batch/overlap,
  resize không nguyên, seam duplicate, merge nhiều khối, cache có endpoint
  giống nhau, giới hạn cache score 0 và tách batch nguồn/model invocation.
- Frontend `npm run build` đạt; Python compileall và `git diff --check` đạt.
- Lần kiểm tra cuối dùng `--basetemp tmp/pytest-final-20261006 -p no:cacheprovider`
  vì Windows từ chối xóa thư mục pytest cũ. Test với thư mục riêng đã đạt.
- Các artifact giữ code fingerprint của từng job trong manifest và file số
  liệu đính kèm. Sau các job GPU, đã chỉnh counter preview FPS/latency, counter
  source batch và thứ tự event; các chỉnh sửa này được kiểm tra local, không
  chạy thêm job GPU vượt số lượng được duyệt. FPS benchmark thuộc đúng code
  fingerprint ghi trong artifact.
- Video benchmark cũ còn hiển thị FPS 0 do counter cũ; dùng `summary.json`
  để đọc FPS đo được. Source hiện đã sửa counter.

## Cách chạy

```powershell
$env:PYTHONIOENCODING = "utf-8"
.\.venv\Scripts\python.exe -m modal run --quiet --timestamps modal_common_path.py `
  --input data/videos/data-shibuya-5m.mp4 `
  --start-seconds 0 --duration-seconds 25 `
  --engine tracklet_aggregation `
  --config configs/shibuya-t4-optimized.yaml `
  --run-id shibuya-t4-optimized-new-run --cache-policy reuse --cpu 2
```

Tải artifact bằng `python scripts/download_modal_artifacts.py --run-id
shibuya-t4-optimized-new-run`. Với backend live, chọn `CROWD_CONFIG` trỏ tới
`configs/shibuya-t4-optimized.yaml`; source batching vẫn chỉ áp dụng ở clip
runner, không coi benchmark batch offline là FPS live.

Số liệu có kiểm tra hash nằm tại
`benchmarks/pipeline_optimization_20261006.json`. Video, ảnh kiểm tra,
manifest, cache và quality comparison nằm dưới `outputs/optimization-20261006/`.
