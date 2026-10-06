# Tối ưu pipeline hướng tới 15 FPS trên Modal T4 — 2026-10-06

Cấu hình được chọn đạt **9.892 FPS**, so với **6.967 FPS** trước đó: tăng **41.98%**. **Chưa đạt 15 FPS.** Cache detection/tracking khớp từng byte; ba Common Path giữ nguyên ID, hướng, support, confidence và toàn bộ tọa độ.

## Phạm vi kiểm chứng

Video Shibuya 1280×720, 0–25 giây, 750 frame. YOLO26n, FP32, source batch 2, 5 ảnh/frame (full frame + 2×2 tile), imgsz 1280. Detect mọi frame; không bỏ frame, không giảm vùng tìm kiếm. ByteTrack hybrid và Tracklet Aggregation giữ các ngưỡng trước đây.

Bản trước dùng T4 + 2 CPU; bản mới dùng T4 + 4 CPU. Mức tăng là kết quả của cả cấu hình và thay đổi code, chưa phải A/B cô lập riêng CUDA Graph. Đây là smoke test offline; chưa xác nhận 15 FPS realtime hoặc toàn bộ video.

## Nút thắt và cách sửa

- Producer trước chỉ decode; crop và preprocessing vẫn nằm trong luồng GPU. Đã chuyển chuẩn bị batch sang producer, giữ hàng đợi 2 batch và thứ tự nguồn.
- Bỏ batch NHWC trung gian. Đóng gói trực tiếp BGR sang RGB NCHW, giống từng byte với preprocessing cũ. Mỗi batch sở hữu buffer riêng để tránh bị ghi đè.
- Lịch sử candidate/support Common Path bị deepcopy rồi băm lại ở mỗi frame. Profile 750 frame ghi nhận khoảng 10,3 giây deepcopy candidate và 3,8 giây canonical digest. Chỉ đọc và băm lịch sử khi engine vừa compute; trace xuất ra vẫn giống từng byte.
- Merge trước so sánh cả candidate đã bị loại và candidate ở block tương lai. Chỉ so với box đã được chọn và block hiện tại, giữ thứ tự confidence và quy tắc suppress cũ.
- IoU và motion association dùng chung một snapshot box Kalman, tránh tính xyxy lặp lại.
- CUDA Graph replay các kernel PyTorch với shape cố định, giữ FP32. Shape khác hoặc backend không hỗ trợ dùng inference thông thường. GPU test xác nhận replay trên cả 750 frame và cache không đổi.
- Sửa FPS theo cửa sổ: trước đây long_run_metrics.csv lấy số frame chia thời gian video nên luôn gần 30 FPS. Code hiện dùng thời gian wall-clock. GPU test này vẫn đọc throughput từ summary.json; các CSV cửa sổ cũ không được dùng để kết luận tốc độ.

## Kết quả đo

- Throughput: 6.967 → 9.892 FPS; thời gian xử lý: 107.646 → 75.819 giây.
- Preprocessing trong luồng GPU, p50: 58.244 → 7.505 ms/frame. CPU preprocessing vẫn được thực hiện, nhưng chạy song song ở producer.
- Merge, p50: 15.426 → 12.780 ms/frame.
- Tracking, p50: 32.824 → 30.209 ms/frame.
- E2E latency, p95: 1683.693 → 1770.697 ms.
- GPU utilization trung bình: 49.613% → 72.597%; peak 93.0%. VRAM peak 2126 MB.
- 182311 detection, 413 ID, 375 inference batch, 3.750 detector image, 750 association update, 0 skip/reorder.

Các percentile GPU là thời gian batch được chia cho số source frame. cProfile có overhead và các lời gọi chờ CUDA; không cộng cumulative time hoặc p95 lồng nhau để suy ra FPS.

Throughput tăng nhưng p95 latency tăng khoảng 5,17%. Cấu hình này ưu tiên throughput offline; chưa giải quyết xong độ trễ hàng đợi cho realtime.

## Các cấu hình đã loại

- T4 + 8 CPU với cProfile: 5.745 FPS. Chỉ dùng tìm nút thắt, không so throughput với bản không profile. Decode mất khoảng 1,8 giây; producer phần lớn chờ hàng đợi. Tăng CPU riêng không giải quyết phần forward.
- Tensor 768×1280, vẫn đủ tile/độ phóng: **11.001 FPS**, nhưng detection 182311 → 144977 và path hạng 3 đổi hướng REVERSE → FORWARD. Bỏ padding vẫn làm đổi dự đoán của model; không chọn.
- PyTorch FP16 + producer preprocessing: **10.094 FPS**, detection 182122, 415 ID. Common Path cũng đổi hướng/support/geometry; không chọn.

## Vì sao chưa đạt 15 FPS

15 FPS có ngân sách **66,67 ms/frame cho toàn pipeline**. Forward FP32 trên T4 vẫn có p50 **68.423 ms/frame**, chưa tính H2D, postprocess và các bước khác. CUDA Graph giảm overhead phát lệnh nhưng không giảm phép tính model; forward chưa nhanh hơn bản trước.

Muốn xác nhận 15 FPS với workload này cần tiếp tục tối ưu backend/khối lượng inference hoặc dùng GPU nhanh hơn, rồi kiểm chứng detection và Common Path. Các thử nghiệm giảm precision/padding hiện có hồi quy nên không được dùng làm cấu hình mặc định. Kết quả này không chứng minh rằng mọi backend trên T4 đều không thể đạt 15 FPS.

## Kiểm tra chất lượng và tests

SHA-256 cache cả bản trước và bản mới: `de20748318ebd6db7a77abc5c50f861b3ad5f6f0b2a011bec1acfb1893a3a935`. Đã kiểm tra hash trên file tải về, decode đủ 750 frame, xem contact sheet và bản đồ path. `path_support.jsonl` và `candidate_decisions.jsonl` giống từng byte. Replay CPU từ detection cũ xây lại cả 750 bản ghi tracking giống hoàn toàn và ba path không lệch tọa độ.

Toàn bộ suite 221 test đã qua khi kiểm tra lại trước commit. Unit test bao gồm preprocessing byte equality, quyền sở hữu buffer, crop scale/alignment, greedy merge đối chiếu reference, hàng đợi/thứ tự batch, cache key và FPS wall-clock. Không chạy model inference trên local.

Không có nhãn ground truth Shibuya, vì vậy chỉ kết luận không hồi quy so với baseline; không tuyên bố tăng mAP/recall/HOTA/IDF1.

Cache của hai cấu hình bị loại cũng đã kiểm tra SHA và so sánh đầy đủ. MP4 FP16 decode đủ 750 frame. Riêng bản tải video rectangular bị thiếu dữ liệu (308/750 frame, lỗi codec), đã đổi tên thành `*.incomplete-download.mp4`; không dùng video này để đánh giá chất lượng. Việc loại rectangular dựa trên cache/summary đã kiểm chứng, không dựa trên file video hỏng. Cache và MP4 của cấu hình được chọn đều đã kiểm chứng đầy đủ.

## Dùng cấu hình đã kiểm chứng

```powershell
.\.venv\Scripts\python.exe -m modal run modal_common_path.py --input data/videos/data-shibuya-5m.mp4 --duration-seconds 25 --config configs/shibuya-t4-cuda-graph.yaml --cpu 4 --run-id YOUR_UNIQUE_RUN_ID
```

Benchmark JSON: `benchmarks/pipeline_15fps_20261006.json`.
Artifact đã kiểm chứng: `outputs/optimization-15fps-20261006/graph/pipeline-graph-fp32-20261006/`.
Profile: `outputs/common_path/pipeline-cpu8-profile-20261006/cpu_profile_*.txt`.

GPU source fingerprint: `586ef326240dce9dd1517e8ba04b06cb72863173af5aa49b1a989cf951b3d420`. Sau GPU test chỉ sửa thống kê wall-time FPS, cache-key performance flags và tài liệu; không đổi thuật toán detection/tracking/Common Path.

## Chạy toàn bộ video Shibuya

Run `shibuya-10m-graph-fp32-20261006` đã hoàn tất trên Tesla T4 + 4 CPU với cùng cấu hình CUDA Graph FP32. File `data/videos/data-shibuya-10m.mp4` thực tế dài 565,3 giây, 1280×720, 30 FPS. Đã xử lý đủ 16.959 frame, không bỏ frame, có 16.959 lượt association và 84.795 ảnh detector.

Tốc độ xử lý toàn video là **7,647 FPS**, processing time **2.217,797 giây**, remote wall time **2.238,438 giây**. Kết quả này thấp hơn clip 25 giây và vẫn chưa đạt mục tiêu 15 FPS. Không có ground truth cho lượt chạy đầy đủ để kết luận độ chính xác tuyệt đối.

```powershell
.\.venv\Scripts\python.exe -m modal run modal_common_path.py --input data/videos/data-shibuya-10m.mp4 --config configs/shibuya-t4-cuda-graph.yaml --cpu 4 --detector-worker-mode no_sync_profile --inference-backend pytorch_fp32 --run-id shibuya-10m-graph-fp32-20261006
```

Artifact trên Volume `crowd-analysis-data`: `common_path/runs/shibuya-10m-graph-fp32-20261006/`. Chỉ MP4 được tải về `outputs/common_path/shibuya-10m-graph-fp32-20261006/tracked_points_common_path.mp4`: 1.104.264.119 byte, đã decode đủ 16.959 frame và xem frame cuối video để kiểm tra overlay detection/common path. Video và các output lớn không được đưa vào Git.
