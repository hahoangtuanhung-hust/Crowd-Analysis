# Prompt nghiên cứu và triển khai Crowd Analysis Web Demo

## ROLE

Bạn là **Senior Computer Vision Engineer + AI Systems Engineer + MLOps Engineer**.

Nhiệm vụ của bạn là tự nghiên cứu, thiết kế và triển khai một **demo Crowd Analysis chạy trên Web** cho hệ thống camera giám sát.

Không chỉ cần code chạy được, mà phải tiếp cận theo hướng có khả năng phát triển thành **production product**: realtime, ổn định, có benchmark, dễ thay model/tracker, có profiling và tối ưu tài nguyên.

---

## 1. BỐI CẢNH BÀI TOÁN

Hệ thống nhận video từ camera giám sát đặt cố định tại các khu vực như:

- đường đi bộ;
- sảnh tòa nhà;
- khu đô thị;
- trung tâm thương mại;
- khu vực công cộng;
- hành lang;
- cổng ra vào.

Mục tiêu chính:

> Phân tích quỹ đạo di chuyển của người từ camera và xác định những khu vực/con đường được sử dụng nhiều nhất.

Không tập trung vào nhận diện danh tính.

Hệ thống chỉ cần:

1. Person Detection
2. Multi-object Tracking
3. Trajectory Extraction
4. Crowd Density Analysis
5. Movement Flow Analysis
6. Heatmap
7. Popular Path Analysis
8. Web Visualization
9. Performance Monitoring

Track ID chỉ tồn tại trong phiên xử lý video và không đại diện cho danh tính thật.

---

## 2. MỤC TIÊU DEMO

Input:

```text
MP4 video
hoặc
RTSP Camera Stream
```

Pipeline mong muốn:

```text
Camera / Video
      ↓
Frame Decoder
      ↓
Person Detection
      ↓
Multi Object Tracking
      ↓
Track ID + Coordinates
      ↓
Trajectory Extraction
      ↓
Spatial Aggregation
      ↓
Crowd Analytics
      ↓
Web Dashboard
```

Dashboard phải cho phép quan sát:

```text
Camera
   ↓
Person Detection
   ↓
Tracking
   ↓
Trajectories
   ├── Occupancy Heatmap
   ├── Movement Heatmap
   ├── Popular Paths
   ├── Crowd Count
   ├── Zone Statistics
   └── Direction / Flow
```

---

## 3. PHASE 1 — RESEARCH TRƯỚC KHI CODE

Trước khi triển khai, hãy research các giải pháp Computer Vision hiện tại phù hợp với bài toán.

Không được chọn công nghệ chỉ vì quen thuộc.

So sánh ít nhất:

### Person Detection

Nghiên cứu các lựa chọn hiện tại, ưu tiên model realtime/edge:

- YOLO26n
- các phiên bản YOLO phù hợp khác
- RT-DETR hoặc detector khác nếu có lý do tốt
- TensorRT-optimized detector nếu phù hợp

So sánh:

```text
Accuracy
FPS
Latency
VRAM
CPU performance
GPU performance
Model size
ONNX support
TensorRT support
Ease of deployment
License
```

---

## 4. MULTI OBJECT TRACKING RESEARCH

So sánh ít nhất:

```text
ByteTrack
BoT-SORT
OC-SORT
Deep OC-SORT
các tracker realtime mới nếu đáng cân nhắc
```

Đánh giá:

```text
FPS
ID stability
ID switches
Occlusion handling
CPU cost
GPU cost
ReID requirement
Suitability for fixed CCTV
Suitability for dense crowds
```

Baseline ưu tiên thử đầu tiên:

```text
Detector: YOLO26n
Tracker: ByteTrack
Input size: 640
Person class only
```

Nhưng không mặc định đây là giải pháp cuối cùng.

Phải benchmark trước khi kết luận.

---

## 5. ĐẶC BIỆT PHÂN BIỆT 2 BÀI TOÁN

Không được đánh đồng:

```text
Crowd Counting
```

và

```text
Crowd Flow Analysis
```

Crowd Counting:

```text
Có bao nhiêu người?
```

Crowd Flow Analysis:

```text
Người đang đi đâu?
Khu vực nào đông?
Luồng nào được sử dụng nhiều?
Hướng nào phổ biến?
Thời gian nào đông?
```

Product đang hướng tới bài toán thứ hai.

---

## 6. TRAJECTORY EXTRACTION

Với mỗi Track ID, lưu:

```json
{
  "track_id": 12,
  "timestamp": 10.42,
  "frame_id": 313,
  "x": 534,
  "y": 412
}
```

Không sử dụng tâm bounding box một cách máy móc.

Ưu tiên điểm:

```text
bottom-center của bounding box
```

vì gần tương ứng với vị trí chân người trên mặt đất.

Ví dụ:

```text
x = (x1 + x2) / 2
y = y2
```

Mỗi track tạo thành trajectory:

```text
T_i = [(x1,y1,t1), (x2,y2,t2), ..., (xn,yn,tn)]
```

Cần xử lý:

- smoothing trajectory;
- missing detections;
- track age;
- very short tracks;
- ID switches;
- jitter.

Không lưu trajectory vô hạn trong RAM.

Thiết kế bounded history/ring buffer.

---

## 7. PERSPECTIVE CORRECTION

Camera CCTV thường có perspective distortion.

Hai vùng cùng kích thước pixel nhưng diện tích ngoài đời có thể rất khác nhau.

Nghiên cứu và nếu cần triển khai:

```text
Homography / Bird's-eye View
```

Cho phép người dùng chọn 4 điểm ROI trên frame:

```text
Camera Plane
      ↓
Perspective Transform
      ↓
Ground Plane / Bird's-eye View
```

Sau đó thực hiện heatmap/path analysis trên ground plane.

Phải hỗ trợ:

```text
Mode 1:
Pixel-space analysis

Mode 2:
Ground-plane analysis sau calibration
```

Nếu chưa calibration, UI phải thông báo rõ kết quả chỉ mang tính tương đối.

---

## 8. CROWD HEATMAP

Chia mặt phẳng thành grid:

```text
W × H
```

Ví dụ:

```text
64 × 36
128 × 72
```

Mỗi trajectory point cộng giá trị vào cell tương ứng.

Có thể dùng:

```text
Gaussian Kernel Density Estimation
```

hoặc phương pháp tương đương để tạo heatmap mượt.

Output:

```text
Crowd Density Heatmap
```

Heatmap cần hỗ trợ:

```text
Current Window
Last 1 minute
Last 5 minutes
Entire video
```

Không được cộng dồn vô hạn trong livestream.

Sử dụng sliding/time window.

---

## 9. CROWD DENSITY VS MOVEMENT DENSITY

Triển khai riêng hai metric.

### Occupancy Heatmap

Cho biết:

```text
Người thường xuất hiện/đứng ở đâu?
```

### Movement Heatmap

Cho biết:

```text
Người thường di chuyển qua đâu?
```

Không được gộp hai khái niệm thành một.

---

## 10. POPULAR PATH ANALYSIS

Đây là feature quan trọng nhất.

Mục tiêu:

> Xác định các tuyến đường được nhiều người sử dụng nhất.

Không đơn giản chỉ vẽ toàn bộ trajectory vì sẽ tạo spaghetti lines.

Nghiên cứu và triển khai một phương pháp robust.

Có thể xem xét:

### Approach A — Grid Flow

Spatial grid:

```text
cell_A → cell_B
```

Mỗi lần trajectory đi từ cell A sang B:

```text
flow[A][B] += 1
```

Từ đó xác định các edge có traffic cao.

### Approach B — Trajectory Clustering

Chuẩn hóa trajectory rồi cluster bằng một trong:

```text
DBSCAN
HDBSCAN
KMeans nếu phù hợp
trajectory distance
Fréchet distance
DTW
```

Không sử dụng thuật toán phức tạp nếu không mang lại giá trị thực tế.

### Approach C — Flow Field

Mỗi grid cell lưu vector:

```text
vx
vy
magnitude
```

Tổng hợp thành vector field.

Ví dụ:

```text
→ → → →
→ → ↘
   ↓
   ↓
```

Cho biết hướng di chuyển chủ đạo.

Sau research, lựa chọn phương án phù hợp nhất cho demo.

Ưu tiên:

```text
simple
fast
interpretable
production-friendly
```

---

## 11. TOP PATHS

Dashboard cần hiển thị:

```text
Top 5 most popular paths
```

Ví dụ:

```text
Path 1
Entrance → Lobby
142 trajectories
37%

Path 2
Entrance → Elevator
96 trajectories
25%

Path 3
Lobby → Exit
53 trajectories
14%
```

Nếu không có semantic zone thì dùng:

```text
Grid region A → Grid region B
```

---

## 12. ROI / ZONE ANALYTICS

Cho phép người dùng vẽ polygon ROI trên camera.

Ví dụ:

```text
Zone A = Entrance
Zone B = Lobby
Zone C = Elevator
Zone D = Exit
```

Theo dõi:

```text
current people
unique tracks
entry count
exit count
average dwell time
peak occupancy
```

Và flow:

```text
Entrance → Lobby
Lobby → Elevator
Lobby → Exit
```

---

## 13. CROWD METRICS

Tính tối thiểu:

```text
Current Crowd Count

Average Crowd Count

Peak Crowd Count

Unique Track Count

Zone Occupancy

Entry Count

Exit Count

Average Dwell Time

Flow per Zone

Movement Direction

Top Paths
```

---

## 14. WEB DEMO

Xây dựng kiến trúc tách:

```text
Frontend
      ↓
REST / WebSocket
      ↓
Backend
      ↓
Video Processing Pipeline
      ↓
AI Inference
```

Ưu tiên:

Backend:

```text
Python
FastAPI
OpenCV
Ultralytics / ONNX Runtime / TensorRT
NumPy
```

Frontend:

```text
React + Vite
```

hoặc công nghệ tương đương nếu có lý do rõ ràng.

Không dùng frontend quá phức tạp cho PoC.

Nếu Streamlit/Gradio giúp tạo baseline nhanh hơn, có thể làm phiên bản đầu tiên nhưng architecture cuối cùng phải mô tả hướng FastAPI + Web frontend.

---

## 15. WEB UI

UI tối thiểu:

```text
------------------------------------------------
 Crowd Analysis Dashboard
------------------------------------------------

Camera: Camera 01

[ Live Video ]

Persons: 34
FPS: 28
Inference: 21 ms

------------------------------------------------

[ Crowd Heatmap ]

------------------------------------------------

[ Popular Paths ]

Path 1   ███████████ 37%
Path 2   ███████     25%
Path 3   ████        14%

------------------------------------------------

[ Crowd Over Time ]

------------------------------------------------

[ Zone Flow ]

Entrance → Lobby      142
Lobby → Elevator       96
Lobby → Exit           53
```

---

## 16. VIDEO OVERLAY

Video hiển thị:

```text
Bounding Box
Track ID
Trajectory tail
Zone boundaries
Person count
FPS
```

Cho phép bật/tắt riêng:

```text
Detection
Tracking
Trajectory
Heatmap
Zones
```

---

## 17. PERFORMANCE ARCHITECTURE

Không được viết pipeline kiểu:

```python
while True:
    read frame
    inference
    draw
    encode
    send web
```

nếu khiến toàn pipeline block nối tiếp.

Thiết kế tối thiểu:

```text
Camera Reader
       ↓
Frame Queue
       ↓
Inference Worker
       ↓
Tracking
       ↓
Analytics Worker
       ↓
Visualization
       ↓
Web Stream
```

Ưu tiên:

```text
bounded queues
drop stale frames
latest-frame strategy
```

Realtime camera cần ưu tiên:

```text
freshness > processing every frame
```

Nếu inference không kịp camera FPS, được phép drop frame hợp lý.

---

## 18. FRAME SKIPPING

Benchmark:

```text
Inference every frame

Inference every 2 frames

Inference every 3 frames
```

Xác định trade-off giữa:

```text
FPS
GPU usage
tracking stability
analytics accuracy
```

Tracker có thể tiếp tục prediction giữa các detection frames nếu thiết kế phù hợp.

---

## 19. INFERENCE OPTIMIZATION

Benchmark ít nhất:

```text
PyTorch FP32
PyTorch FP16 nếu GPU hỗ trợ
ONNX Runtime
TensorRT nếu môi trường NVIDIA cho phép
```

Không tối ưu mù quáng.

Đo baseline trước.

Metrics:

```text
decode latency
preprocessing latency
inference latency
tracking latency
analytics latency
rendering latency
encoding latency
end-to-end latency
```

---

## 20. PERFORMANCE METRICS

Dashboard hoặc `/metrics` phải có:

```text
input_fps
processing_fps
inference_ms
tracking_ms
analytics_ms
render_ms
e2e_latency_ms
queue_size
dropped_frames
cpu_percent
ram_mb
gpu_utilization
gpu_memory_mb
```

Nếu không có GPU thì gracefully disable GPU metrics.

---

## 21. TARGET BAN ĐẦU

Dùng làm engineering goal, không được giả vờ đã đạt nếu chưa benchmark:

```text
720p camera

>= 20 FPS với GPU phổ thông nếu phần cứng cho phép

p95 processing latency < 100 ms

Không tăng RAM vô hạn

Không tăng trajectory history vô hạn

Không block camera capture

Stable run >= 30 phút
```

Nếu hardware không đạt target:

Báo cáo chính xác bottleneck thay vì fake benchmark.

---

## 22. DỮ LIỆU ANALYTICS

Tách raw detections khỏi aggregated analytics.

Schema gợi ý:

```text
tracks
------
timestamp
camera_id
track_id
x
y
zone_id
confidence
```

Aggregated:

```text
zone_stats
----------
time_bucket
camera_id
zone_id
count
entries
exits
avg_dwell_time
```

Flow:

```text
flows
-----
time_bucket
camera_id
from_zone
to_zone
count
```

Không nhất thiết dùng database ngay trong MVP.

Có thể bắt đầu:

```text
in-memory
+
SQLite
```

nhưng thiết kế abstraction để sau này chuyển sang:

```text
PostgreSQL
TimescaleDB
ClickHouse
Kafka
```

---

## 23. PRIVACY

Không triển khai:

```text
face recognition
identity recognition
person re-identification across unrelated cameras
```

trừ khi được yêu cầu riêng.

Demo chỉ cần anonymous tracking.

Ví dụ:

```text
Track 53
```

không phải:

```text
Nguyen Van A
```

Không lưu frame/video ngoài nhu cầu demo nếu không cần thiết.

---

## 24. TEST CASES

Tạo test videos/test scenarios cho:

### Normal

```text
5–10 people
```

### Dense

```text
30–100 people
```

### Occlusion

```text
people crossing each other
```

### Low light

```text
dark CCTV
```

### Perspective

```text
far vs near people
```

### Stationary Crowd

```text
people standing
```

### Bidirectional Flow

```text
people moving opposite directions
```

### Empty Scene

```text
0 people
```

Đánh giá từng case.

---

## 25. EVALUATION

Không đánh giá chỉ bằng FPS.

### Detection

Nếu có ground truth:

```text
Precision
Recall
mAP
```

### Tracking

Nếu dataset phù hợp:

```text
IDF1
HOTA
MOTA
ID switches
```

### Product Analytics

Quan trọng hơn với bài toán này:

```text
Count error
Zone crossing accuracy
Flow direction accuracy
Top-path stability
Heatmap stability
```

---

## 26. BENCHMARK TABLE

Tạo:

```text
benchmark_results.csv
```

Ví dụ:

| Detector | Tracker | Backend | FPS | Inf ms | Track ms | VRAM | ID Stability |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| YOLO26n | ByteTrack | PyTorch | ... | ... | ... | ... | ... |
| YOLO26n | BoT-SORT | PyTorch | ... | ... | ... | ... | ... |
| YOLO26n | ByteTrack | ONNX | ... | ... | ... | ... | ... |

Không điền dữ liệu giả.

---

## 27. PROJECT STRUCTURE

Thiết kế repository sạch.

Ví dụ:

```text
crowd-analysis/
│
├── backend/
│   ├── app/
│   │   ├── api/
│   │   ├── core/
│   │   ├── inference/
│   │   ├── tracking/
│   │   ├── analytics/
│   │   ├── video/
│   │   ├── metrics/
│   │   └── schemas/
│   │
│   └── main.py
│
├── frontend/
│   └── ...
│
├── configs/
│
├── models/
│
├── data/
│   ├── videos/
│   └── outputs/
│
├── benchmarks/
│
├── tests/
│
├── scripts/
│
├── docker/
│
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── README.md
```

Điều chỉnh cấu trúc nếu tìm được kiến trúc tốt hơn.

---

## 28. CONFIG

Không hardcode parameters.

Ví dụ:

```yaml
detector:
  model: yolo26n.pt
  imgsz: 640
  confidence: 0.4
  device: cuda

tracker:
  type: bytetrack

video:
  inference_interval: 1
  queue_size: 4

analytics:
  grid_width: 64
  grid_height: 36
  trajectory_history: 60
```

---

## 29. API

Tạo tối thiểu:

```text
GET /health
GET /metrics

POST /api/video/upload

POST /api/stream/start
POST /api/stream/stop

GET /api/analytics/summary
GET /api/analytics/heatmap
GET /api/analytics/paths
GET /api/analytics/zones

WS /ws/live
```

Có thể thay đổi API nếu architecture yêu cầu.

---

## 30. DOCKER

Dockerize application.

Tối thiểu:

```text
backend
frontend
```

Nếu cần:

```text
redis
database
```

Docker phải hỗ trợ:

```bash
docker compose up
```

Không đưa model weights lớn vào Git.

---

## 31. OBSERVABILITY

Structured log:

```json
{
  "camera_id": "cam01",
  "frame_id": 12641,
  "people": 32,
  "fps": 27.3,
  "inference_ms": 22.5,
  "tracking_ms": 3.1,
  "queue_size": 2
}
```

Log error phải đủ thông tin debug nhưng không log dữ liệu nhạy cảm.

---

## 32. FAILURE HANDLING

Xử lý:

```text
camera disconnect
invalid RTSP
corrupted frame
model loading failure
CUDA OOM
GPU unavailable
frontend disconnect
queue overflow
video ended
```

RTSP disconnect cần retry với backoff.

Không được để toàn service crash chỉ vì camera disconnect.

---

## 33. IMPLEMENTATION PLAN

Không code toàn bộ ngay.

Làm tuần tự.

### Milestone 1

```text
Video
→ Detection
→ Tracking
→ Track IDs
```

Verify.

### Milestone 2

```text
Tracking
→ Trajectories
```

Verify bằng overlay.

### Milestone 3

```text
Trajectories
→ Heatmap
```

Verify.

### Milestone 4

```text
Trajectories
→ Flow Analysis
→ Popular Paths
```

Verify.

### Milestone 5

```text
FastAPI
→ Video Processing
→ API
```

Verify.

### Milestone 6

```text
Frontend
→ Video
→ Heatmap
→ Analytics
```

Verify.

### Milestone 7

```text
Benchmark
→ Profiling
→ Optimization
```

### Milestone 8

```text
Docker
→ Tests
→ Documentation
```

---

## 34. QUY TẮC LÀM VIỆC

Trong quá trình triển khai:

1. Research trước khi chọn công nghệ.
2. Ưu tiên tài liệu chính thức và paper/repository gốc.
3. Không giả định một model là "best".
4. Không đổi model chỉ vì benchmark lý thuyết.
5. Luôn đo trên video thực tế.
6. Không tối ưu trước khi profiling.
7. Không rewrite toàn project khi chỉ cần sửa module.
8. Sau mỗi milestone phải chạy test.
9. Nếu phát hiện design ban đầu sai, giải thích trade-off rồi refactor.
10. Không tạo fake benchmark.
11. Không báo feature hoàn thành nếu chưa test.
12. Giữ dependency tối thiểu.
13. Ưu tiên code dễ đọc, modular và có type hints.
14. Tất cả threshold phải configurable.
15. Ghi lại assumption và limitation trong README.

---

## 35. YÊU CẦU AGENT BÁO CÁO THEO TỪNG GIAI ĐOẠN

Trước khi code, trả về:

```text
1. Problem understanding

2. Research findings

3. Technology comparison

4. Recommended baseline

5. Architecture

6. Data flow

7. Repository structure

8. Implementation milestones

9. Evaluation plan

10. Production risks
```

Sau đó mới triển khai.

Sau mỗi milestone trả:

```text
DONE
- ...

TESTED
- ...

RESULT
- ...

ISSUES
- ...

NEXT
- ...
```

---

## 36. DELIVERABLE CUỐI CÙNG

Project phải có:

```text
Source code

README.md

Architecture diagram

Dockerfile / docker-compose

Config files

Demo web

Example video

Heatmap

Trajectory visualization

Popular path visualization

Zone analytics

benchmark_results.csv

Performance report

Test report
```

README phải hướng dẫn từ zero:

```bash
git clone ...
cd ...
cp .env.example .env
docker compose up
```

và cách chạy development mode.

---

## 37. EXPECTED PRODUCT DEMO

Khi user upload video hoặc nhập RTSP:

```text
Camera
   ↓
YOLO
   ↓
Tracker
   ↓
Anonymous Track IDs
   ↓
Trajectory
   ↓
Spatial Aggregation
   ↓
+-------------------------+
| Crowd Analysis          |
|                         |
| Current People          |
| Occupancy Heatmap       |
| Movement Heatmap        |
| Top Paths               |
| Direction Flow          |
| Zone Occupancy          |
| Entries / Exits         |
| Crowd Timeline          |
+-------------------------+
```

Demo phải giúp trả lời trực quan:

```text
1. Hiện tại có bao nhiêu người?

2. Khu vực nào thường đông nhất?

3. Người thường di chuyển qua đâu?

4. Hướng di chuyển chính là gì?

5. Tuyến đường nào được sử dụng nhiều nhất?

6. Khoảng thời gian nào đông nhất?

7. Camera/model đang xử lý với FPS và latency bao nhiêu?
```

---

## 38. PRODUCTION ROADMAP

Sau khi PoC hoàn thành, đề xuất roadmap:

```text
PoC
 ↓
Single Camera MVP
 ↓
RTSP Realtime
 ↓
GPU Optimization
 ↓
Multi-camera
 ↓
Central Analytics Service
 ↓
Monitoring
 ↓
Autoscaling
 ↓
Production
```

Phân biệt rõ:

```text
Feature cần cho Demo
Feature cần cho MVP
Feature cần cho Production
```

Không đưa toàn bộ complexity của production vào PoC.

---

## 39. CÂU HỎI CUỐI CÙNG AGENT PHẢI TRẢ LỜI

Sau khi hoàn thành nghiên cứu và benchmark, hãy đưa ra kết luận kỹ thuật:

```text
Detector nào nên dùng?

Tracker nào nên dùng?

Input resolution bao nhiêu?

Inference interval bao nhiêu?

PyTorch / ONNX / TensorRT nên dùng ở giai đoạn nào?

Một GPU/CPU có thể xử lý khoảng bao nhiêu camera?

Bottleneck hiện tại nằm ở đâu?

Độ chính xác giảm bao nhiêu khi tối ưu FPS?

Popular Path nên tính theo Grid Flow hay Trajectory Clustering?

Kiến trúc nào phù hợp khi scale từ 1 → 10 → 100 camera?
```

Mọi kết luận phải dựa trên benchmark hoặc research có nguồn, không dựa trên cảm tính.

---

## LỆNH BẮT ĐẦU

Bắt đầu bằng **Phase 1 — Research & Architecture**. Chưa viết toàn bộ hệ thống cho tới khi đã trình bày baseline, trade-off và kế hoạch benchmark.
