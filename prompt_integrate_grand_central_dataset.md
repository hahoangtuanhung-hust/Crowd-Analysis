# Prompt tích hợp Grand Central Station Dataset vào Crowd Analysis Project

## ROLE

Bạn là **Senior Computer Vision Engineer + Data Engineer + MLOps Engineer**.

Nhiệm vụ của bạn là tích hợp **Grand Central Station Dataset** vào project Crowd Analysis hiện tại để kiểm thử pipeline:

```text
Video
→ Person Detection
→ Multi-object Tracking
→ Trajectory Extraction
→ Heatmap
→ Zone Flow
→ Popular Path Analysis
→ Web Dashboard
```

Phải làm việc trực tiếp trên codebase hiện tại, giữ nguyên những chức năng đang hoạt động và không tạo một project mới nếu không thật sự cần thiết.

---

## 1. THÔNG TIN DATASET

Dataset sử dụng:

```text
Grand Central Station Dataset
```

Nguồn tham khảo:

```text
https://www.ee.cuhk.edu.hk/~xgwang/grandcentral.html
```

Thông tin tích hợp và loader tham khảo:

```text
https://github.com/crowdbotp/OpenTraj
https://github.com/crowdbotp/OpenTraj/tree/master/datasets/GC
Tôi đã chuyển thành video data.mp4 trong thư mục data/videos
```

Thông số tham khảo:

```text
Resolution: 1920 × 1080
Video FPS: 25
Total frames: khoảng 100.000
Annotated frames: khoảng 5.000
Annotation FPS: khoảng 1,25
Annotated pedestrians: 12.684
Average pedestrians per annotated frame: 123
Maximum pedestrians per frame: 332
```

Dataset không có giấy phép sử dụng rõ ràng. Chỉ sử dụng cho:

```text
research
education
internal demo
benchmark
```

Không đóng gói dataset vào Docker image, không commit dataset lên Git và không tuyên bố phù hợp cho mục đích thương mại.

---

## 2. KIỂM TRA PROJECT TRƯỚC KHI SỬA

Trước khi code, hãy:

1. Đọc toàn bộ `README.md`.
2. Tìm và đọc các file hướng dẫn như `AGENTS.md`, `CONTRIBUTING.md`.
3. Kiểm tra cấu trúc backend, frontend, config, Docker và tests.
4. Kiểm tra `requirements.txt`, `pyproject.toml`, `package.json`.
5. Kiểm tra `git status`.
6. Xác định pipeline detection, tracking và analytics hiện tại.
7. Tìm tất cả module đang xử lý:
   - video;
   - detector;
   - tracker;
   - trajectory;
   - heatmap;
   - zone;
   - popular path;
   - API;
   - frontend.
8. Không ghi đè hoặc xóa thay đổi hiện có của người dùng.

Trước khi triển khai, báo cáo ngắn:

```text
CURRENT ARCHITECTURE
- ...

REUSABLE MODULES
- ...

MISSING COMPONENTS
- ...

FILES TO CHANGE
- ...

IMPLEMENTATION PLAN
- ...
```

Sau đó tiếp tục triển khai mà không cần tạo lại toàn bộ project.

---

## 3. MỤC TIÊU TÍCH HỢP

Sau khi hoàn thành, project phải hỗ trợ hai luồng độc lập:

### Luồng A — Model Inference

```text
Grand Central Video
→ YOLO Person Detection
→ ByteTrack
→ Predicted Trajectories
→ Crowd Analytics
```

### Luồng B — Ground-truth Analytics

```text
Grand Central Annotations
→ Ground-truth Trajectories
→ Crowd Analytics
```

Luồng B dùng để kiểm tra riêng logic heatmap, zone flow và popular path mà không phụ thuộc vào chất lượng YOLO/ByteTrack.

Không trộn ground truth và prediction trong cùng một kết quả. Mọi output phải có trường:

```text
source = ground_truth | prediction
```

---

## 4. DATASET STRUCTURE

Ưu tiên tổ chức dữ liệu như sau, nhưng điều chỉnh theo kiến trúc hiện tại nếu cần:

```text
data/
└── grand-central/
    ├── raw/
    │   ├── video/
    │   ├── annotations/
    │   └── homography/
    ├── processed/
    │   ├── trajectories.parquet
    │   ├── trajectories.csv
    │   ├── metadata.json
    │   ├── zones.json
    │   └── homography.json
    └── outputs/
        ├── tracking/
        ├── heatmaps/
        ├── popular_paths/
        ├── zone_flows/
        └── benchmarks/
```

Không hardcode đường dẫn.

Sử dụng biến môi trường hoặc config:

```env
GC_DATASET_ROOT=./data/grand-central
GC_VIDEO_PATH=./data/grand-central/raw/video/grand_central.mp4
GC_ANNOTATION_DIR=./data/grand-central/raw/annotations
GC_PROCESSED_DIR=./data/grand-central/processed
GC_OUTPUT_DIR=./data/grand-central/outputs
```

Cập nhật `.env.example`, không sửa hoặc commit secret trong `.env`.

Cập nhật `.gitignore`:

```gitignore
data/grand-central/raw/
data/grand-central/processed/
data/grand-central/outputs/
*.rar
*.zip
*.mp4
```

Nếu project cần commit file placeholder, sử dụng `.gitkeep`.

---

## 5. DOWNLOAD VÀ SETUP DATASET

Tạo script:

```text
scripts/setup_grand_central.py
```

Script cần:

1. Kiểm tra dataset đã tồn tại chưa.
2. Tạo đúng cấu trúc thư mục.
3. Hỗ trợ nhận file dataset đã được tải thủ công.
4. Kiểm tra file lỗi hoặc thiếu.
5. Extract `.rar` nếu công cụ phù hợp có sẵn.
6. Không tải lại nếu dữ liệu đã tồn tại.
7. Có thông báo rõ nếu môi trường thiếu `unrar` hoặc `7z`.
8. Không dùng URL không đáng tin cậy.
9. Ghi metadata nguồn dữ liệu vào `metadata.json`.
10. Không tự động commit dữ liệu.

Hỗ trợ lệnh:

```bash
python scripts/setup_grand_central.py \
  --archive /path/to/cvpr2015_pedestrianWalkingPathDataset.rar
```

Nếu có thể tải trực tiếp hợp lệ, hỗ trợ thêm:

```bash
python scripts/setup_grand_central.py --download
```

Nhưng trước khi tải phải hiển thị:

```text
source URL
destination
estimated size nếu xác định được
license warning
```

Không giả định tải thành công. Luôn verify file sau khi tải hoặc extract.

---

## 6. KHẢO SÁT ĐỊNH DẠNG ANNOTATION

Không viết parser dựa trên phỏng đoán.

Trước tiên:

1. Liệt kê cấu trúc dataset sau khi extract.
2. Đọc tài liệu đi kèm.
3. Inspect một số annotation thực tế.
4. Xác định:
   - frame ID;
   - pedestrian/track ID;
   - timestamp;
   - tọa độ;
   - bounding box nếu có;
   - trajectory point;
   - trạng thái mất track;
   - sampling rate;
   - coordinate system.
5. So sánh với loader của OpenTraj.

Sau khi hiểu format, ghi lại trong:

```text
docs/grand_central_dataset.md
```

Tài liệu phải mô tả:

```text
raw format
field mapping
coordinate convention
FPS và annotation FPS
known limitations
normalization rules
```

---

## 7. DATASET ADAPTER

Tạo adapter riêng, ví dụ:

```text
backend/app/datasets/grand_central.py
```

hoặc đặt theo kiến trúc hiện tại.

Interface gợi ý:

```python
class GrandCentralDataset:
    def load_metadata(self): ...
    def iter_annotations(self): ...
    def load_trajectories(self): ...
    def load_homography(self): ...
    def validate(self): ...
```

Không để logic dataset nằm trực tiếp trong API endpoint hoặc UI.

Nếu OpenTraj loader đáp ứng tốt, có thể tái sử dụng. Tuy nhiên:

- không thêm toàn bộ OpenTraj chỉ để dùng một hàm nhỏ nếu dependency quá lớn;
- nếu viết parser riêng, phải có test;
- ghi rõ khác biệt giữa parser nội bộ và OpenTraj.

---

## 8. NORMALIZED TRAJECTORY SCHEMA

Chuẩn hóa về schema chung của project:

```text
camera_id
source
frame_id
timestamp
track_id
x
y
x1
y1
x2
y2
foot_x
foot_y
world_x
world_y
confidence
zone_id
```

Quy tắc:

```text
foot_x = (x1 + x2) / 2
foot_y = y2
timestamp = frame_id / source_fps
```

Chỉ tính `foot_x`, `foot_y` từ bounding box nếu raw annotation thật sự có bounding box.

Nếu annotation chỉ cung cấp trajectory point:

```text
x = raw trajectory x
y = raw trajectory y
foot_x = x
foot_y = y
```

Các trường không tồn tại phải để `null`, không tạo dữ liệu giả.

Lưu output ưu tiên ở Parquet, đồng thời có tùy chọn CSV để debug.

---

## 9. HOMOGRAPHY

Kiểm tra dữ liệu homography của Grand Central/OpenTraj.

Hỗ trợ hai chế độ:

```text
pixel_space
ground_plane
```

Nếu homography hợp lệ:

```text
(foot_x, foot_y)
→ perspective transform
→ (world_x, world_y)
```

Yêu cầu:

1. Kiểm tra shape của matrix.
2. Kiểm tra matrix không singular.
3. Test một số điểm.
4. Không apply homography hai lần.
5. Không coi đơn vị world coordinate là mét nếu chưa xác minh.
6. UI phải ghi rõ coordinate mode đang sử dụng.

Nếu không có hoặc không xác minh được homography, hệ thống vẫn chạy ở `pixel_space`.

---

## 10. VIDEO PROCESSING

Tích hợp video Grand Central vào pipeline hiện tại.

Baseline:

```yaml
detector:
  model: yolo26n.pt
  classes: [person]
  confidence: 0.4
  imgsz: 640

tracker:
  type: bytetrack

video:
  source_fps: 25
  inference_interval: 1
  queue_size: 4
  drop_stale_frames: true
```

Không mặc định YOLO26n chắc chắn tồn tại trong môi trường. Kiểm tra model/config đang được project hỗ trợ và fallback có giải thích nếu cần.

Pipeline phải:

- dùng bounded queue;
- không tích lũy frame vô hạn;
- giải phóng `VideoCapture`;
- xử lý video kết thúc bình thường;
- hỗ trợ stop/cancel;
- không giữ toàn bộ video trong RAM;
- lưu trajectory theo batch;
- tách inference khỏi analytics nếu kiến trúc hiện tại cho phép.

---

## 11. CROWD ANALYTICS

Sử dụng cả ground truth và prediction để tạo:

### Occupancy Heatmap

Cho biết người xuất hiện hoặc đứng nhiều ở đâu.

### Movement Heatmap

Chỉ cộng những đoạn trajectory có độ dịch chuyển vượt ngưỡng cấu hình.

### Grid Flow

Chia scene thành grid và đếm:

```text
cell_A → cell_B
```

Loại bỏ:

- chuyển động nhỏ do jitter;
- cạnh tự nối `A → A`;
- track quá ngắn;
- bước nhảy bất thường;
- trajectory thiếu quá nhiều frame.

### Popular Paths

Baseline ưu tiên:

```text
trajectory
→ grid sequence
→ remove consecutive duplicates
→ entry region
→ intermediate regions
→ exit region
→ aggregate
→ rank top paths
```

Không chỉ vẽ toàn bộ trajectory vì sẽ tạo spaghetti lines.

Output tối thiểu:

```json
{
  "path_id": "path_001",
  "source": "ground_truth",
  "from_zone": "entrance_a",
  "to_zone": "platform",
  "count": 142,
  "percentage": 37.0,
  "representative_path": []
}
```

### Zone Analytics

Hỗ trợ polygon zones từ `zones.json`.

Metric:

```text
current occupancy
unique tracks
entries
exits
average dwell time
peak occupancy
zone-to-zone flow
```

Không đếm nhiều lần khi một track dao động gần biên zone. Dùng debounce hoặc hysteresis phù hợp.

---

## 12. API VÀ WEB UI

Tích hợp theo conventions hiện tại.

API gợi ý:

```text
GET  /api/datasets
GET  /api/datasets/grand-central/metadata
POST /api/datasets/grand-central/prepare
POST /api/datasets/grand-central/analyze
GET  /api/datasets/grand-central/status
GET  /api/analytics/summary
GET  /api/analytics/heatmap
GET  /api/analytics/paths
GET  /api/analytics/zones
```

Không tạo endpoint trùng nếu project đã có API tương đương.

UI cần cho phép:

1. Chọn `Grand Central Dataset`.
2. Chọn `Ground Truth` hoặc `YOLO + ByteTrack`.
3. Chọn `Pixel Space` hoặc `Ground Plane`.
4. Chọn đoạn video: 1 phút, 5 phút hoặc toàn bộ video.
5. Bật/tắt detection, Track ID, trajectory, heatmap, zones và popular paths.
6. Hiển thị:
   - people count;
   - processing FPS;
   - inference latency;
   - number of tracks;
   - dropped frames;
   - Top 5 Paths;
   - zone flow.

Nếu project chưa có frontend hoàn chỉnh, chỉ thêm UI tối thiểu cần thiết; không rewrite toàn bộ giao diện.

---

## 13. EVALUATION

Do annotation có sampling rate thấp hơn video, không so sánh Track ID từng frame một cách máy móc.

### Analytics Validation

Dùng ground-truth trajectories để kiểm tra:

```text
heatmap generation
zone entry/exit
dwell time
flow direction
top-path ranking
```

### Model Evaluation

Tại các annotated frames, đánh giá những metric khả thi:

```text
person count MAE
person count MAPE khi hợp lệ
detection precision
detection recall
matched track coverage
trajectory coverage
```

Tracking metrics như IDF1/HOTA chỉ tính khi annotation và sampling rate thực sự đáp ứng yêu cầu. Nếu không đáp ứng, phải ghi rõ limitation.

Không tạo benchmark giả.

---

## 14. TESTS

Tạo unit tests tối thiểu cho:

```text
annotation parsing
timestamp conversion
bottom-center calculation
trajectory grouping
homography transformation
point-in-polygon
zone transition
dwell time
grid transition
jitter filtering
popular-path aggregation
empty annotations
missing fields
invalid path
```

Tạo integration test nhỏ sử dụng một fixture được phép lưu trong repository.

Không đưa video hoặc annotation lớn vào fixtures.

Các case bắt buộc:

```text
empty scene
single trajectory
short trajectory
stationary pedestrian
bidirectional flow
track crossing zone boundary
track oscillating at boundary
missing annotation frame
corrupted annotation
invalid homography
```

---

## 15. BENCHMARK

Chạy thử ít nhất ba cấu hình nếu môi trường cho phép:

```text
Ground-truth analytics only
YOLO + ByteTrack every frame
YOLO + ByteTrack every 2 frames
```

Ghi vào:

```text
benchmarks/grand_central_results.csv
```

Schema:

```csv
mode,detector,tracker,imgsz,inference_interval,processing_fps,inference_ms,tracking_ms,analytics_ms,e2e_ms,cpu_percent,ram_mb,gpu_memory_mb,dropped_frames
```

Không điền số giả nếu chưa chạy.

Nếu môi trường không có GPU, benchmark CPU và ghi rõ hardware.

---

## 16. DOCUMENTATION

Cập nhật `README.md` với:

```text
Grand Central Dataset overview
license warning
download/setup instructions
expected directory structure
how to run ground-truth analytics
how to run model inference
how to open dashboard
how to run tests
how to run benchmark
known limitations
```

Ví dụ lệnh phải dựa trên code thật đã triển khai, không dùng lệnh minh họa không chạy được.

---

## 17. ACCEPTANCE CRITERIA

Chỉ báo hoàn thành khi:

- Dataset adapter đọc được annotation thật.
- Có validation report.
- Ground-truth trajectory được chuẩn hóa.
- Pipeline video chạy được.
- YOLO + ByteTrack xuất được predicted trajectories.
- Ground-truth analytics chạy độc lập.
- Heatmap được tạo từ dữ liệu thật.
- Zone flow hoạt động.
- Top paths được tổng hợp.
- API trả dữ liệu hợp lệ.
- UI hiển thị ít nhất một kết quả thật.
- Tests liên quan đều pass.
- Không có dataset lớn trong Git.
- README có hướng dẫn chạy lại từ đầu.
- Không có benchmark giả.
- Không làm hỏng tính năng hiện tại.

---

## 18. CÁCH BÁO CÁO SAU MỖI MILESTONE

Sau mỗi milestone, báo cáo:

```text
DONE
- Những file đã tạo hoặc sửa

TESTED
- Các lệnh đã chạy
- Test nào đã pass

RESULT
- Kết quả thật
- Output được tạo ở đâu

LIMITATIONS
- Những phần chưa xác minh
- Giới hạn dataset hoặc môi trường

NEXT
- Bước tiếp theo
```

Nếu gặp lỗi, phải tìm nguyên nhân trước khi đổi kiến trúc hoặc thay dependency.

---

## 19. THỨ TỰ TRIỂN KHAI

Thực hiện theo thứ tự:

```text
Milestone 1
Inspect project và dataset

Milestone 2
Dataset setup + validation

Milestone 3
Annotation parser + normalized trajectories

Milestone 4
Ground-truth heatmap + grid flow + popular paths

Milestone 5
YOLO + ByteTrack inference trên video

Milestone 6
So sánh prediction với annotation

Milestone 7
API + Web UI

Milestone 8
Tests + benchmark + documentation
```

Không bắt đầu bằng cách viết lại toàn bộ project.

---

## LỆNH BẮT ĐẦU

Bắt đầu ngay với **Milestone 1 — Inspect project và dataset**. Sau khi báo cáo kiến trúc hiện tại và danh sách file dự kiến thay đổi, tiếp tục triển khai từng milestone.
