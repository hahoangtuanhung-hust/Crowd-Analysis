# Phân tích luồng đám đông (Crowd Analysis Web Demo)

Hệ thống phân tích luồng di chuyển của đám đông qua một camera duy nhất, ẩn danh, dành cho các file MP4 và luồng RTSP (camera trực tiếp). Hệ thống phát hiện người, gán ID theo dõi cục bộ theo phiên, trích xuất quỹ đạo di chuyển (dựa vào điểm giữa - dưới của bounding box) và tạo ra các bản đồ nhiệt (heatmap) về mật độ/di chuyển, hướng dòng chảy, các tuyến đường phổ biến, dòng thời gian của đám đông và thống kê theo khu vực (zone) ngay trên bảng điều khiển web thời gian thực.

Hệ thống **không** triển khai nhận diện khuôn mặt, đối sánh danh tính hay nhận diện lại (re-id) qua nhiều camera.

---

## ⚠️ Những file cần tải thêm sau khi Clone từ GitHub

Vì một số file có kích thước lớn hoặc chứa thông tin nhạy cảm nên đã được loại trừ (bỏ qua bởi `.gitignore`). Sau khi `git clone`, bạn cần chuẩn bị các file sau để chương trình có thể chạy:

1. **File cấu hình môi trường (.env):**
   - Chạy lệnh copy file mẫu: `cp .env.example .env` (hoặc copy thủ công trên Windows).
   - Điền các thông số cần thiết vào file `.env` nếu có.
2. **Trọng số Mô hình AI (Model Weights):**
   - Các file có đuôi `*.pt`, `*.onnx`, `*.engine` không được lưu trên Git.
   - **Cách xử lý:** Mặc định dự án sử dụng `yolo26n.pt`. Lần đầu tiên bạn chạy chương trình, thư viện Ultralytics sẽ cần **kết nối internet** để tự động tải file này về máy.
3. **Các Video mẫu (Sample Videos):**
   - Tất cả các file `*.mp4`, `*.zip`, `*.rar` đều bị chặn đưa lên Git.
   - Các lệnh ví dụ trong tài liệu có sử dụng `data/videos/example-people.mp4`, `data/videos/data-test.mp4`, `data/videos/data-shibuya-test.mp4`. Bạn cần tự đưa video của riêng bạn vào thư mục `data/videos/` hoặc thay đổi đường dẫn video trong câu lệnh cho phù hợp.
4. **Tập dữ liệu Grand Central (Tùy chọn):**
   - Nếu bạn muốn đánh giá trên bộ dữ liệu Grand Central, bạn cần phải tải thủ công hoặc chạy script.
   - Bạn cần sử dụng lệnh tải: `python scripts/setup_grand_central.py --download` (Xem chi tiết ở phần dưới).

---

## Bắt đầu nhanh với Docker

**Yêu cầu:** Máy đã cài Docker Desktop (với Compose v2) và có kết nối internet ở lần chạy đầu tiên.

```bash
git clone <repository-url> crowd-analysis
cd crowd-analysis
cp .env.example .env
docker compose up --build
```

Sau khi chạy xong, hãy mở [http://localhost:8080](http://localhost:8080), chọn một video (ví dụ `data/videos/example-people.mp4`) và bấm **Start**. 
API backend có sẵn tại [http://localhost:8000/docs](http://localhost:8000/docs).

*Mặc định, Docker sẽ cài đặt PyTorch bản CPU. Nếu bạn dùng máy có card NVIDIA và đã cài NVIDIA Container Toolkit, chạy lệnh sau:*
```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```
Hãy thiết lập `TORCH_INDEX_URL` trong file `.env` thành đường dẫn CUDA tương thích.

---

## Chế độ Phát triển (Development mode)

**Yêu cầu:** Python 3.11-3.14 và Node.js 24+.

**Terminal 1 (Backend):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m uvicorn backend.main:app --reload --port 8000
```

**Terminal 2 (Frontend):**
```powershell
cd frontend
npm ci
npm run dev
```

Mở [http://localhost:5173](http://localhost:5173). Vite sẽ làm proxy chuyển tiếp các request REST, WebSocket và MJPEG tới FastAPI (cổng 8000).

**Chạy chế độ Batch (Không dùng Web):**
```powershell
python -m scripts.run_tracking data/videos/example-people.mp4
```
Kết quả (video overlay, jsonl track, heatmap) sẽ được lưu trong `data/outputs/`.

---

## Tracking Điểm & Tuyến Đường

Bạn có thể đánh giá (benchmark) hoặc chạy pipeline trích xuất điểm cho cấu hình sản xuất mặc định:

```powershell
# Chạy với tối đa 600 frames
python -m scripts.process_point_tracks data/videos/data-test.mp4 `
  --zones configs/zones.json `
  --output-dir outputs/point-only-common-path `
  --max-frames 600
```

Chạy trên Modal GPU (Cloud) để xuất video với Common Path (đường đi phổ biến):
```powershell
$env:PYTHONIOENCODING = "utf-8"
modal run --quiet --timestamps modal_common_path.py `
  --input data/videos/my-video.mp4 `
  --start-seconds 0 `
  --duration-seconds 25 `
  --engine tracklet_aggregation `
  --config configs/shibuya.yaml `
  --run-id my-video-tracklet-20260922 `
  --cache-policy reuse
```

---

## Kiến trúc và Chính sách Dữ liệu

Xem `docs/architecture.md` để biết về kiến trúc hệ thống và khả năng mở rộng.
Hệ thống sử dụng các hàng đợi (queues) có giới hạn, khung lưới (grid) cố định, và giới hạn tuổi thọ của bộ đệm.
**Lưu ý về dữ liệu:** Dữ liệu nhận dạng (raw detection) hoàn toàn tách biệt với dữ liệu tổng hợp phân tích. Hệ thống **không** lưu trữ ảnh khuôn mặt hay liên kết Track ID với danh tính người thực.

## Cấu hình

Tất cả các ngưỡng cấu hình và giới hạn tài nguyên đều nằm ở [`configs/default.yaml`](configs/default.yaml). Bạn có thể cấu hình:
- Mô hình YOLO, độ phân giải (`imgsz`), ngưỡng `confidence`, `iou`.
- Các thông số theo dõi của ByteTrack (Track buffer, match_thresh...).
- Cấu hình lưới phân tích Common Path, Tracklet Aggregation, Heatmap.

---

## Giới hạn hiện tại

- Chỉ xử lý một phiên camera (session) trên một process.
- Không có cơ sở dữ liệu lưu trữ dài hạn (khi tắt server, dữ liệu phiên hiện tại sẽ bị xóa).
- Việc ánh xạ tọa độ mặt phẳng (Homography) giả định rằng mặt sàn đi lại là một mặt phẳng tương đối.
- Để sử dụng tính năng tăng tốc GPU/CUDA đầy đủ, yêu cầu phải có máy chủ hoặc máy trạm NVIDIA riêng biệt.
