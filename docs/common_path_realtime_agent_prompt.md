# Prompt triển khai Common Path Realtime

## ROLE

Bạn là **Senior Computer Vision Engineer + Realtime Systems Engineer**.

Hãy sửa trực tiếp project Crowd Analysis hiện tại để triển khai tính năng **Common Path Realtime**: xác định tuyến đường được nhiều người sử dụng nhất, cập nhật theo lưu lượng mới nhưng không bị rung hoặc thay đổi liên tục sau vài frame.

Không viết lại toàn bộ project. Không tạo benchmark hoặc dữ liệu giả. Mọi kết luận phải được kiểm tra trên video/camera thật của project.

---

## 1. Mục tiêu

Pipeline cần đạt được:

```text
Camera/Video
    ↓
Person Detection
    ↓
Multi-object Tracking
    ↓
Bottom-center Point
    ↓
Point Tracklet
    ↓
Directed Grid Flow
    ↓
Short-term + Long-term Aggregation
    ↓
Stable Common Path
    ↓
Realtime Visualization
```

Common Path phải:

- phản ánh đúng luồng người hiện tại;
- phân biệt hướng `A → B` và `B → A`;
- không thay đổi vì một vài track ngẫu nhiên;
- không cập nhật ở từng frame;
- không bị người đứng lâu làm tăng sai độ phổ biến;
- thay đổi khi lưu lượng thực tế thay đổi đủ lâu;
- không tăng RAM vô hạn;
- không làm block detector/tracker.

---

## 2. Kiểm tra project trước khi sửa

Đọc:

```text
README.md
config files
video pipeline
detector module
tracker module
trajectory module
analytics module
websocket/API
frontend visualization
```

Xác định:

- file video hoặc RTSP thực tế đang được sử dụng;
- detector và tracker hiện tại;
- FPS đầu vào và FPS xử lý;
- point hiện tại lấy từ vị trí nào;
- cách lưu trajectory;
- logic heatmap/path hiện có;
- path có đang tính lại mỗi frame không;
- dữ liệu có bị cộng dồn vô hạn không;
- tracker có giữ ID ổn định không;
- pipeline có queue hay đang chạy tuần tự;
- có zone Entrance/Exit trong `zones.json` không.

Trước khi code, báo cáo:

```text
CURRENT PIPELINE
- ...

CURRENT COMMON PATH LOGIC
- ...

ROOT CAUSES
- ...

FILES TO CHANGE
- ...

IMPLEMENTATION PLAN
- ...
```

Không sửa code trước khi xác định được nguyên nhân path hiện tại không ổn định.

---

## 3. Không thay đổi tracking nếu không cần thiết

Detector và ByteTrack được phép sử dụng bounding box nội bộ. Không vẽ bounding box ra giao diện.

Vị trí của người dùng cho analytics là bottom-center:

```python
point_x = (x1 + x2) / 2
point_y = y2
```

Mỗi point phải có:

```python
TrajectoryPoint(
    camera_id: str,
    track_id: int,
    frame_id: int,
    timestamp: float,
    x: float,
    y: float,
    confidence: float,
    zone_id: str | None,
)
```

Smoothing point bằng EMA hoặc phương pháp nhẹ tương đương:

```python
smooth_x = alpha * new_x + (1 - alpha) * previous_x
smooth_y = alpha * new_y + (1 - alpha) * previous_y
```

Cho phép cấu hình:

```yaml
trajectory:
  smoothing_alpha: 0.30
  min_point_distance_px: 4
  max_history_seconds: 30
  min_track_duration_seconds: 1.0
  min_track_distance_px: 30
```

Không thêm point nếu khoảng cách so với point trước quá nhỏ.

---

## 4. Quản lý tracklet

Quản lý các track đang hoạt động:

```python
dict[int, deque[TrajectoryPoint]]
```

Mỗi track phải có trạng thái:

```text
tentative
active
lost
completed
discarded
```

Chỉ đưa track vào Common Path khi đạt tối thiểu:

- đủ thời gian tồn tại;
- đủ quãng đường di chuyển;
- có số point hợp lệ;
- không phải detection xuất hiện trong vài frame;
- không phải người gần như đứng yên.

Ví dụ:

```yaml
track_filter:
  min_duration_seconds: 1.0
  min_distance_px: 30
  min_points: 6
  max_stationary_speed_px_s: 3
```

Kết thúc tracklet khi:

- Track ID mất quá `track_buffer`;
- người đi vào zone kết thúc;
- video kết thúc;
- stream bị dừng.

Track hoàn chỉnh phải được ghi theo batch ra CSV hoặc Parquet. Không giữ toàn bộ lịch sử livestream trong RAM.

---

## 5. Zone và hướng vào–ra

Đọc polygon zone từ `zones.json`.

Ví dụ:

```json
{
  "zones": [
    {
      "id": "entrance_a",
      "name": "Entrance A",
      "type": "entry",
      "points": [[20, 300], [200, 260], [220, 500], [20, 500]]
    },
    {
      "id": "exit_b",
      "name": "Exit B",
      "type": "exit",
      "points": [[800, 250], [950, 250], [950, 500], [780, 500]]
    }
  ]
}
```

Mỗi track cần xác định:

```text
origin_zone
destination_zone
direction
first_seen_at
last_seen_at
```

Chống rung ở biên zone bằng:

- yêu cầu point nằm trong zone liên tiếp một số frame;
- debounce theo thời gian;
- hysteresis;
- chỉ ghi nhận một transition hợp lệ một lần cho mỗi Track ID.

```yaml
zones:
  min_inside_frames: 3
  debounce_seconds: 1.0
```

---

## 6. Milestone 1 — Chia không gian thành grid

Chia ROI hoặc ground plane thành grid:

```yaml
grid:
  columns: 32
  rows: 18
```

Mỗi point được ánh xạ vào cell:

```python
cell_x = int(point_x / frame_width * grid_columns)
cell_y = int(point_y / frame_height * grid_rows)
```

Nếu có homography, ưu tiên ánh xạ point sang ground plane trước khi đưa vào grid.

Không thêm transition khi:

- hai point vẫn nằm trong cùng cell;
- track nhảy qua khoảng cách bất thường;
- trajectory quay đi quay lại ở biên cell do jitter.

Dùng hysteresis hoặc yêu cầu point đi sâu vào cell mới trước khi công nhận chuyển cell.

Test bằng cách vẽ grid, cell hiện tại, chuỗi cell của từng Track ID và hướng di chuyển.

---

## 7. Milestone 2 — Directed Grid Flow

Mỗi khi trajectory di chuyển:

```text
cell_A → cell_B
```

cập nhật directed edge:

```python
flow[cell_A, cell_B] += 1
```

Phải phân biệt `A → B` và `B → A`.

Không cộng điểm ở từng frame. Mỗi Track ID chỉ được đóng góp tối đa một lần cho cùng một directed edge trong cùng một hành trình hoặc time bucket.

```python
visited_edges_by_track[track_id] = {
    ("A1", "A2"),
    ("A2", "B2"),
}
```

Popularity phải tính chủ yếu theo số unique Track ID đi qua edge, không tính theo số trajectory point.

Lưu dữ liệu theo time bucket:

```python
FlowBucket(
    start_time,
    end_time,
    edge_unique_tracks,
    origin_destination_tracks,
)
```

Sử dụng ring buffer có giới hạn.

---

## 8. Milestone 3 — Short Window và Long Window

Không sử dụng bộ đếm cộng dồn vô hạn.

```yaml
common_path:
  bucket_seconds: 1
  short_window_seconds: 30
  long_window_seconds: 180
  short_weight: 0.40
  long_weight: 0.60
```

- Short window phản ánh lưu lượng trong khoảng 20–30 giây gần nhất.
- Long window giữ xu hướng ổn định trong khoảng 2–5 phút.

Điểm mỗi edge:

```python
edge_score = (
    short_weight * normalized_short_flow
    + long_weight * normalized_long_flow
)
```

Có thể dùng exponential decay:

```python
decayed_weight = exp(-age_seconds / tau)
```

Dữ liệu cũ phải giảm ảnh hưởng dần, không bị xóa đột ngột. Cập nhật flow snapshot khoảng một lần mỗi giây, không cập nhật theo từng frame.

---

## 9. Milestone 4 — Trích xuất Common Path

Với mỗi cặp zone `origin_zone → destination_zone`, xây directed graph:

```text
Node = grid cell
Edge = chuyển động giữa hai cell
Weight = flow score
```

Không dùng thuật toán clustering nặng ở mỗi frame. Baseline ưu tiên:

```text
Directed Grid Flow + weighted graph search
```

Chuyển flow thành cost:

```python
edge_cost = edge_length / (edge_score + epsilon)
```

Tìm candidate path từ origin tới destination bằng Dijkstra hoặc thuật toán tương đương với cost luôn dương.

Yêu cầu:

- không tạo loop;
- bỏ edge có support quá thấp;
- giới hạn độ dài path;
- ưu tiên edge có nhiều unique tracks;
- có thể thêm penalty cho chuyển hướng quá gấp;
- kiểm tra path thực sự nối origin và destination.

Sau khi tìm được path dạng danh sách grid cell, chuyển thành polyline và làm mượt bằng Ramer–Douglas–Peucker, Chaikin, moving average hoặc phương pháp nhẹ tương đương.

Không để polyline thay đổi mạnh chỉ vì một cell nhiễu.

---

## 10. Milestone 5 — Ổn định Common Path

Không thay Active Path ngay khi candidate mới có điểm cao hơn.

Mỗi path có trạng thái:

```text
candidate
active
cooling
retired
```

Cấu hình khởi đầu:

```yaml
common_path:
  update_interval_seconds: 3
  min_unique_tracks: 5
  switch_margin: 0.20
  confirmation_seconds: 8
  cooling_seconds: 20
  min_path_edge_support: 3
  path_similarity_threshold: 0.70
  centerline_ema_alpha: 0.20
```

Candidate chỉ thay Active Path khi:

```python
candidate_score >= active_score * (1 + switch_margin)
```

và điều kiện được duy trì đủ lâu:

```python
candidate_duration >= confirmation_seconds
```

Nếu candidate gần giống Active Path, không tạo path mới. Baseline ưu tiên directed-edge overlap:

```python
similarity = len(edges_a & edges_b) / len(edges_a | edges_b)
```

Nếu `similarity >= 0.70`, coi là cùng một tuyến và chỉ cập nhật centerline bằng EMA.

Khi Active Path giảm lưu lượng:

```text
active → cooling → retired
```

Chỉ chuyển sang `retired` nếu lưu lượng giảm liên tục đủ lâu. Không để path biến mất vì camera mất detection trong thời gian ngắn.

---

## 11. Milestone 6 — Top Common Paths

Hỗ trợ tối thiểu Top 3 hoặc Top 5 Common Paths.   

Mỗi path trả về:

```json
{
  "path_id": "entrance_a__exit_b__01",
  "origin_zone": "entrance_a",
  "destination_zone": "exit_b",
  "state": "active",
  "unique_tracks_short": 12,
  "unique_tracks_long": 48,
  "score": 0.82,
  "confidence": 0.88,
  "direction": "A_TO_B",
  "polyline": [[100, 420], [250, 390], [500, 350], [850, 310]],
  "last_updated_at": 1720000000.0
}
```

Confidence phải dựa trên dữ liệu thật, ví dụ số unique tracks, chênh lệch với path đứng sau, độ ổn định qua nhiều lần cập nhật và độ liên tục của edge. Không dùng confidence ngẫu nhiên.

---

## 12. Milestone 7 — Kiến trúc realtime

Không để pipeline chạy tuần tự và tích tụ frame.

```text
Camera Reader
      ↓
Latest-frame Queue
      ↓
Detection + Tracking
      ↓
Point Event Queue
      ↓
Trajectory Worker
      ↓
Flow Aggregator
      ↓
Common Path Worker
      ↓
WebSocket/API
```

Yêu cầu:

- bounded queue;
- ưu tiên frame mới;
- drop stale frame nếu inference không kịp;
- analytics không block detector;
- Common Path không chạy ở từng frame;
- bảo vệ shared state bằng lock ngắn hoặc immutable snapshot;
- graceful shutdown;
- flush tracklet và bucket khi dừng.

```yaml
realtime:
  frame_queue_size: 2
  analytics_queue_size: 1000
  drop_stale_frames: true
  flow_update_interval_seconds: 1
  common_path_update_interval_seconds: 3
  websocket_publish_interval_seconds: 1
```

---

## 13. Milestone 8 — Visualization

Video realtime chỉ hiển thị:

- bottom-center point;
- Track ID;
- trajectory tail;
- polygon zone;
- Active Common Path;
- Candidate Path nếu debug được bật;
- hướng mũi tên;
- số unique tracks;
- score;
- processing FPS;
- end-to-end latency.

Quy ước:

```text
Active Path    = màu xanh lá
Candidate Path = màu vàng
Cooling Path   = màu xám
Direction      = mũi tên
```

Độ dày path phản ánh lưu lượng nhưng phải giới hạn min/max. Renderer đọc common-path snapshot gần nhất, không tính lại path ở từng frame.

Cho phép bật/tắt:

```text
points
track_ids
trajectory_tails
zones
grid
edge_flows
candidate_paths
active_paths
heatmap
debug_metrics
```

---

## 14. API

Tạo hoặc cập nhật:

```text
GET /api/analytics/common-paths
GET /api/analytics/flows
GET /api/analytics/zones
GET /api/analytics/metrics
WS  /ws/live
```

Response Common Path phải có:

```text
path_id
origin_zone
destination_zone
state
score
confidence
unique_tracks_short
unique_tracks_long
polyline
direction
updated_at
```

WebSocket không gửi toàn bộ trajectory history mỗi lần. Chỉ gửi snapshot cần thiết hoặc delta.

---

## 15. Output

Sinh tối thiểu:

```text
outputs/tracked_points.mp4
outputs/trajectories.csv
outputs/edge_flows.json
outputs/common_paths.json
outputs/common_path_map.png
outputs/common_path_timeline.csv
outputs/realtime_benchmark.csv
```

Schema timeline:

```csv
timestamp,path_id,state,score,confidence,unique_tracks_short,unique_tracks_long
```

---

## 16. Test bắt buộc

Viết unit test cho:

1. Hai point trong cùng cell không tạo edge.
2. `A → B` khác `B → A`.
3. Một Track ID không bị đếm lặp trên cùng edge.
4. Track quá ngắn không ảnh hưởng Common Path.
5. Người đứng lâu không làm tăng popularity.
6. Point rung ở biên cell không tạo nhiều transition giả.
7. Candidate mới không thay Active Path ngay lập tức.
8. Candidate đủ mạnh và đủ lâu sẽ trở thành Active.
9. Active Path chuyển sang Cooling trước khi Retired.
10. Hai path tương tự được merge.
11. Dữ liệu hết thời gian window được loại bỏ.
12. Ring buffer không tăng RAM vô hạn.
13. Video không có người không gây lỗi.
14. Stream kết thúc flush dữ liệu đúng cách.
15. Analytics worker chậm không block inference worker.

---

## 17. Kịch bản kiểm thử Common Path

### Case 1 — Luồng ổn định

```text
80% người đi A → B
20% người đi A → C
```

Kỳ vọng: `A → B` là Active Common Path và không đổi liên tục.

### Case 2 — Nhiễu ngắn hạn

Trong vài giây có 2–3 người đi `A → C`.

Kỳ vọng: `A → B` vẫn Active; `A → C` có thể là Candidate nhưng không thay ngay.

### Case 3 — Thay đổi thật

Trong ít nhất 20–30 giây, 70% người chuyển sang `A → C`.

Kỳ vọng:

```text
A → C trở thành Candidate
→ duy trì đủ confirmation time
→ A → C trở thành Active
→ A → B chuyển sang Cooling
```

### Case 4 — Mất detection ngắn

Kỳ vọng: Active Path không biến mất ngay.

### Case 5 — Hai hướng ngược nhau

Kỳ vọng: `A → B` và `B → A` là hai flow độc lập.

---

## 18. Metrics

Đo tối thiểu:

```text
input_fps
processing_fps
inference_ms
tracking_ms
analytics_ms
common_path_compute_ms
end_to_end_latency_ms
frame_queue_size
analytics_queue_size
dropped_frames
active_tracks
completed_tracks
valid_tracks
common_path_switches
candidate_rejections
cpu_percent
ram_mb
gpu_memory_mb
```

Theo dõi thêm:

```text
Common Path switches/phút
Thời gian phát hiện luồng mới
Tỷ lệ candidate bị loại
Số unique tracks/path
Path overlap giữa hai lần cập nhật
```

Mục tiêu ban đầu:

```text
Không cập nhật Common Path ở mỗi frame
Common Path compute < 20 ms nếu phần cứng cho phép
Không block detector/tracker
RAM không tăng liên tục
Không đổi Active Path vì dưới 5 track
Path mới chỉ active sau thời gian xác nhận
```

Không giả vờ đạt mục tiêu nếu chưa đo.

---

## 19. Benchmark tham số

Benchmark trên cùng một đoạn video:

```text
short_window: 15, 30, 60 giây
long_window: 120, 180, 300 giây
switch_margin: 0.10, 0.20, 0.30
confirmation: 3, 8, 15 giây
grid: 16×9, 32×18, 64×36
```

So sánh:

```text
Path stability
Response time
Common Path switches
False switches
Compute latency
RAM
CPU
```

Chọn cấu hình cân bằng giữa responsiveness, stability, compute cost và interpretability. Không chọn chỉ dựa trên FPS.

---

## 20. Thứ tự triển khai

Thực hiện tuần tự:

```text
Milestone 1: Point tracklet ổn định
Milestone 2: Grid mapping
Milestone 3: Directed edge flow
Milestone 4: Time buckets và sliding windows
Milestone 5: Candidate path extraction
Milestone 6: State machine và hysteresis
Milestone 7: Realtime worker/API
Milestone 8: Visualization
Milestone 9: Tests
Milestone 10: Benchmark và tuning
```

Sau mỗi milestone phải:

1. Chạy test.
2. Báo file đã sửa.
3. Báo kết quả đo được.
4. Nêu limitation.
5. Chỉ tiếp tục nếu milestone hiện tại hoạt động.

Format báo cáo:

```text
MILESTONE
- ...

CHANGED
- ...

TESTED
- ...

RESULT
- ...

METRICS
- ...

ISSUES
- ...

NEXT
- ...
```

---

## 21. Acceptance criteria

Chỉ báo hoàn thành khi:

- Common Path dựa trên unique Track ID, không dựa trên số point.
- Phân biệt được hướng `A → B` và `B → A`.
- Track quá ngắn và người đứng yên không ảnh hưởng kết quả.
- Common Path không cập nhật ở từng frame.
- Candidate mới không thay Active Path ngay lập tức.
- Path thay đổi khi lưu lượng mới duy trì đủ lâu.
- Có short window và long window.
- Có decay hoặc sliding expiration.
- Có state machine và hysteresis.
- Analytics không block inference.
- Queue và trajectory history có giới hạn.
- Không tăng RAM vô hạn.
- API và visualization hoạt động.
- Tests pass.
- Benchmark sử dụng dữ liệu thật.
- README có lệnh chạy thực tế.

---

## 22. Báo cáo cuối

Sau khi hoàn thành, báo cáo:

```text
ROOT CAUSE
- Vì sao Common Path cũ bị nhiễu hoặc nhảy liên tục

IMPLEMENTED
- Các module và thuật toán đã triển khai

ALGORITHM
- Cách tạo grid-flow
- Cách tính short/long score
- Cách chọn candidate
- Cách chuyển trạng thái path

TESTED
- Unit test
- Integration test
- Video/RTSP test

RESULT
- FPS
- Latency
- Common Path switches/phút
- Thời gian phản ứng khi luồng thay đổi
- CPU/RAM/GPU
- Output đã tạo

SELECTED PARAMETERS
- Grid size
- Window size
- Weights
- Switch margin
- Confirmation time
- Cooling time

LIMITATIONS
- ...

NEXT
- ...
```

Bắt đầu bằng việc đọc project, chạy baseline trên một đoạn ngắn của video hiện tại và xác định chính xác logic Common Path đang được tính ở đâu. Chưa thay đổi thuật toán trước khi có baseline.
