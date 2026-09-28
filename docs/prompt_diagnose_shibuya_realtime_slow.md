# Prompt thực thi: Chẩn đoán và tối ưu tốc độ xử lý video Shibuya trên GPU Modal

Bạn là kỹ sư Computer Vision và Performance Engineering. Hãy đọc repository hiện tại, đo đúng pipeline đang chạy, xác định nguyên nhân khiến video Shibuya dài khoảng một phút mất khoảng 20 phút để xử lý, sau đó sửa trực tiếp code/cấu hình trong phạm vi cần thiết và kiểm chứng lại.

## 1. Mục tiêu và giới hạn

Mục tiêu là giảm thời gian xử lý nhưng vẫn giữ đúng Common Path, hướng di chuyển, Track ID tạm và chất lượng tracklet ở mức đã ghi nhận. Realtime phải được đánh giá bằng số đo thực tế, không suy ra chỉ từ FPS detector.

Giữ nguyên:

- model weights hiện có;
- tracker ByteTrack hiện có;
- backend inference hiện có;
- video và timeline dùng để benchmark;
- logic Common Path trừ khi profiling chứng minh đây là bottleneck.

Không đưa TensorRT, đổi model, ReID, face recognition hoặc thay đổi dữ liệu đầu vào vào cùng lượt tối ưu. Không dùng bỏ frame hoặc giảm chất lượng để tuyên bố thuật toán nhanh hơn nếu chưa báo rõ ảnh hưởng chất lượng.

Các mục tiêu trong prompt là mục tiêu kiểm chứng, không phải kết quả đã đạt.

## 2. Đầu vào bắt buộc

Ưu tiên video:

```text
data/videos/data-shibuya-test.mp4
```

Kiểm tra và ghi:

- SHA-256 video;
- thời lượng, FPS, số frame, kích thước;
- SHA-256 model weights;
- Python, PyTorch, Ultralytics, OpenCV;
- GPU Modal thực tế, CUDA runtime, VRAM;
- commit/config hash.

Nếu video hoặc model không tồn tại, ghi `BLOCKED`; không thay bằng video khác mà không được phép.

## 3. Đọc repository và lần theo pipeline

Đọc README, config, entrypoint Modal, detector, tracker, pipeline video, Common Path, renderer và frontend nếu có. Xác định file/hàm thực tế cho chuỗi:

```text
decode
 -> preprocess
 -> detector
 -> tracker
 -> tracklet/Common Path
 -> render
 -> encode/write
 -> output/UI
```

Đặc biệt kiểm tra:

- `configs/shibuya.yaml`;
- `backend/app/inference/ultralytics_detector.py`;
- `backend/app/video/pipeline.py`;
- `scripts/common_path_clip.py`;
- `scripts/live_common_path.py`;
- `modal_common_path.py`;
- `modal_shibuya_live.py`;
- `backend/app/analytics/tracklet_aggregation.py`;
- `backend/app/video/renderer.py`.

Không kết luận từ code riêng lẻ; phải đối chiếu với metrics runtime.

## 4. Đo baseline trước khi sửa

Chạy trên GPU Modal cùng video/config, ít nhất một smoke test 5-10 giây và một lượt đại diện 25 giây. Giữ nguyên sampling, resolution, model, precision, engine và GPU giữa các lượt.

Lệnh tham khảo:

```powershell
$runId = "shibuya-baseline-$(Get-Date -Format yyyyMMdd-HHmmss)"
modal run --quiet --timestamps modal_common_path.py `
  --input data/videos/data-shibuya-test.mp4 `
  --start-seconds 0 `
  --duration-seconds 25 `
  --engine tracklet_aggregation `
  --mode offline_fast `
  --config configs/shibuya.yaml `
  --run-id $runId `
  --cache-policy refresh
```

Lưu artifact riêng cho từng run. Không ghi đè run cũ.

Bắt buộc thu thập từ `manifest.json`, `metrics.csv` hoặc định dạng tương đương:

- số frame và số detector calls;
- decode p50/p95;
- preprocess p50/p95 nếu có;
- inference p50/p95/max;
- tracker p50/p95;
- Common Path/tracklet update p50/p95/max;
- render p50/p95;
- encode/write p50/p95;
- end-to-end latency p50/p95;
- FPS inference và FPS pipeline;
- queue depth, dropped frames, frame age;
- cold start, model load, warm processing;
- RAM RSS, VRAM allocated/reserved, GPU utilization;
- GPU name và xác nhận `torch.cuda.is_available()`.

GPU timing phải đồng bộ đúng cách. Có thể dùng CUDA events hoặc synchronize ở phạm vi benchmark; không thêm synchronize mỗi frame vào production chỉ để đo.

## 5. Kiểm tra nguyên nhân ưu tiên

Kiểm tra định lượng các giả thuyết sau:

### 5.1 Tiled inference

Với cấu hình Shibuya hiện tại, nếu `tile_include_full_frame: true`, `tile_rows: 2`, `tile_columns: 2`, xác nhận mỗi frame đang chạy 5 lượt model. Tính:

```text
detector_calls = frame_count * số lượt predict mỗi frame
```

Đo riêng thời gian và số calls của full-frame/tile nếu có thể.

### 5.2 Kích thước input và precision

Kiểm tra tác động của:

- `imgsz: 1280` so với 640 hoặc 768;
- `fp32` so với `fp16` trên GPU Modal.

Không đổi hai biến cùng lúc trong phép đo đầu tiên.

### 5.3 Sampling detector

Kiểm tra `inference_interval`. Nếu thử interval 2 hoặc lớn hơn, phải ghi:

- số detector calls;
- số frame thực sự được tracker/Common Path quan sát;
- frame bỏ hoặc dự đoán;
- thay đổi Track ID, coverage, path direction và path support.

Không gọi FPS tăng do bỏ frame là cải thiện không điều kiện.

### 5.4 Render và encode

Đo riêng `render_ms` và `encode_ms`. Kiểm tra có alpha blend/full-frame copy, vẽ lại geometry/mũi tên hoặc encode tuần tự không cần thiết ở mỗi frame hay không. Nếu thêm cache render, khóa cache bằng `path_version`, cấu hình hiển thị và kích thước frame.

### 5.5 Common Path

Chỉ tối ưu Common Path nếu `common_path/tracklet_update_ms` chiếm đáng kể tổng thời gian. Kiểm tra clustering toàn lịch sử, số candidate, số tracklet/đoạn được so sánh, truy vấn lặp và backlog tick. Không đánh đổi correctness để giảm số liệu.

## 6. Thiết kế thí nghiệm A/B

Tạo các profile độc lập, không sửa baseline trực tiếp:

1. `baseline`: config Shibuya hiện tại.
2. `single_pass`: tắt tiled inference, giữ nguyên các biến khác.
3. `fp16`: bật FP16, giữ tiled và resolution hiện tại.
4. `imgsz_640`: giảm resolution, giữ tiled/precision hiện tại.
5. `single_pass_fp16`: chỉ chạy sau khi đã đo riêng hai thay đổi.
6. `interval_2`: detector mỗi 2 frame, chỉ dùng để đánh giá trade-off realtime.

Mỗi profile phải dùng:

- cùng đoạn video;
- cùng GPU Modal;
- cùng model;
- cùng tracker;
- cùng Common Path config;
- cùng số lượt warmup;
- run ID riêng.

Chạy tối đa 3 lượt/profile nếu chi phí cho phép; báo p50/p95 và độ biến thiên. Không chọn duy nhất lượt nhanh nhất.

## 7. Tiêu chí chất lượng bắt buộc

Đối với mỗi profile, so sánh với baseline:

- số Track ID tạm và coverage;
- số tracklet hợp lệ;
- thời gian xuất hiện Common Path đầu tiên;
- số Path ID đổi;
- số lần đổi hướng vô lý;
- support/coverage của path;
- độ rung geometry trên snapshot;
- số frame bỏ và tuổi frame;
- số candidate/segment được xử lý;
- trạng thái warming/active/stale/expired.

Nếu chưa có ground truth, dùng replay/cache cùng chuỗi tracklet và ghi rõ metric nào là proxy, metric nào là đánh giá thủ công. Không gọi là accuracy nếu chưa có định nghĩa.

## 8. Sửa code/cấu hình

Chỉ sửa sau khi có baseline. Ưu tiên theo thứ tự:

1. loại bỏ công việc model dư thừa đã được đo;
2. bật FP16 trên GPU nếu backend hỗ trợ và chất lượng không suy giảm đáng kể;
3. điều chỉnh resolution có kiểm soát;
4. tách nhịp detector/tracker/render nếu cần;
5. giới hạn queue/buffer và tránh backlog;
6. cache geometry/overlay Common Path;
7. tối ưu Common Path nếu profiling chứng minh cần thiết.

Không reset tracker/Common Path khi đổi Top K hoặc style. Không cho kết quả worker cũ ghi đè snapshot mới. Giữ `path_id`, màu và hướng ổn định.

Mọi config mới phải có validation, đơn vị, default và mô tả ảnh hưởng. Nếu profile chỉ phù hợp preview chứ không phù hợp phân tích chính xác, ghi rõ trong config/report.

## 9. Kiểm thử

Chạy toàn bộ test hiện có và thêm test gọn cho thay đổi mới. Tối thiểu kiểm tra:

- profile không tiled giảm số detector calls đúng kỳ vọng;
- FP16 chỉ được áp dụng trên CUDA;
- detector interval không làm tăng support giả;
- queue không backlog vô hạn;
- snapshot/config version cũ không ghi đè snapshot mới;
- Common Path vẫn giữ hướng ngược chiều riêng biệt;
- Path ID/màu không đổi chỉ vì rank thay đổi.

Lệnh tối thiểu:

```powershell
$env:PYTHONPATH = "."
pytest -q
```

## 10. Báo cáo bàn giao

Tạo báo cáo ngắn, ví dụ `docs/shibuya_performance_report.md`, và các artifact:

- `metrics.json`;
- `summary.csv`;
- `manifest.json`;
- metrics theo frame hoặc stage;
- snapshot Common Path JSONL;
- video/screenshot trước và sau;
- config đã resolve cho từng profile.

Báo cáo phải có:

1. baseline và phần cứng;
2. bottleneck được chứng minh bằng số liệu;
3. file/hàm đã sửa;
4. bảng A/B trước/sau;
5. FPS, latency p50/p95, detector calls;
6. RAM/VRAM/GPU utilization;
7. frame drop/queue age;
8. chất lượng Track ID/Common Path;
9. giới hạn còn lại;
10. lệnh tái chạy.

Mỗi mục kết luận phải được đánh dấu một trong:

```text
PASS
FAIL
NOT_RUN
BLOCKED
```

Nếu không có GPU Modal, credential, video hoặc dependency thì ghi `BLOCKED`, không bịa số liệu. Nếu chưa đo FPS UI thì ghi `NOT_RUN`, không thay bằng FPS inference.

## 11. Kết quả mong đợi

Không tuyên bố hoàn thành chỉ vì thời gian chạy giảm. Chỉ kết luận tối ưu thành công khi:

- latency/FPS cải thiện trên cùng GPU và input;
- không tạo backlog tick;
- Common Path không đảo hướng hoặc đếm trùng;
- Path ID/màu ổn định;
- độ rung không tăng bất hợp lý;
- RAM/VRAM/queue có giới hạn;
- profile và artifact có thể tái lập.

Nếu bottleneck chính là tiled inference, báo rõ chi phí 5 pass/frame và trade-off recall khi tắt tile. Nếu bottleneck còn lại là model inference, không đổ lỗi cho Common Path. Nếu GPU Modal chưa chạy được, hoàn thiện phần profiling local/replay nhưng đánh dấu benchmark GPU là `BLOCKED`.
