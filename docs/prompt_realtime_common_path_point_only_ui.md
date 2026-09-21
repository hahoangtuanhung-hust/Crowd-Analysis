# Prompt triển khai Realtime Common Path với Point-only Tracking

## ROLE

Bạn là **Senior Computer Vision Engineer + Realtime Systems Engineer**.

Hãy sửa trực tiếp project Crowd Analysis hiện tại để hệ thống:

1. Phân tích video hoặc RTSP theo thời gian thực.
2. Tracking người bằng bottom-center point.
3. Không hiển thị bounding box.
4. Không hiển thị trajectory/pathway riêng của từng người.
5. Vẫn lưu tracklet cá nhân ở bên trong để phân tích.
6. Tìm Common Path tổng hợp của đám đông.
7. Overlay Common Path tổng hợp trực tiếp lên video realtime trên UI.
8. Common Path cập nhật theo lưu lượng mới nhưng không rung hoặc nhảy liên tục.

Không viết lại toàn bộ project. Không tạo benchmark, dữ liệu hoặc kết quả kiểm thử giả.

---

## 1. Làm rõ yêu cầu hiển thị

Phải phân biệt:

### Individual trajectory

Là đường đi riêng của từng Track ID.

- Được lưu và xử lý nội bộ.
- Dùng để tính grid flow và Common Path.
- Không được vẽ lên video production.
- Chỉ xuất hiện trong debug mode nếu người dùng chủ động bật.

### Common Path

Là tuyến đường tổng hợp được nhiều unique Track ID sử dụng.

- Phải overlay trực tiếp lên video realtime.
- Chỉ hiển thị Active Common Path đã được xác nhận.
- Có mũi tên thể hiện hướng.
- Độ dày hoặc opacity phản ánh lưu lượng.
- Không thay đổi vì vài frame hoặc vài track nhiễu.

Cấu hình mặc định:

~~~yaml
visualization:
  show_bounding_boxes: false
  show_track_ids: false
  show_tracking_points: true
  show_individual_trajectories: false
  show_common_path: true
  show_candidate_path: false
  show_grid_debug: false
  show_edge_flow_debug: false
~~~

---

## 2. Kiểm tra project trước khi sửa

Đọc và kiểm tra:

~~~text
README.md
project structure
detector/tracker config
video hoặc RTSP reader
trajectory storage
analytics modules
renderer
WebSocket/API
frontend video component
~~~

Xác định:

- nguồn video thực tế đang chạy;
- detector và tracker đang sử dụng;
- FPS đầu vào và FPS xử lý;
- pipeline có bị block tuần tự không;
- cách lấy point hiện tại;
- cách lưu trajectory;
- Common Path hiện được tính ở đâu;
- path có đang tính lại mỗi frame không;
- backend render frame hay frontend tự vẽ overlay;
- queue có giới hạn không;
- trajectory có tăng RAM vô hạn không.

Trước khi sửa, báo cáo:

~~~text
CURRENT PIPELINE
- ...

CURRENT UI RENDERING
- ...

ROOT CAUSES
- ...

FILES TO CHANGE
- ...

IMPLEMENTATION PLAN
- ...
~~~

Chạy baseline trên một đoạn video thật trước khi thay đổi thuật toán.

---

## 3. Kiến trúc yêu cầu

~~~text
Camera/Video Reader
        ↓
Latest-frame Queue
        ↓
Person Detection
        ↓
ByteTrack/Internal Tracker
        ↓
Bottom-center Points
        ├── Current Point Snapshot ───────────────┐
        │                                         │
        └── Internal Tracklets                    │
                    ↓                             │
             Grid-flow Aggregator                 │
                    ↓                             │
             Common Path Worker                   │
                    ↓                             │
          Active Common Path Snapshot             │
                    └───────────────┬──────────────┘
                                    ↓
                              Video Renderer
                                    ↓
                               Web UI/Stream
~~~

Renderer chỉ nhận:

- frame mới nhất;
- current tracking points;
- Active Common Path snapshot;
- metrics snapshot;
- zones nếu được bật.

Renderer không đọc hoặc vẽ toàn bộ trajectory history của từng Track ID.

---

## 4. Point-only tracking

Detector và ByteTrack được phép dùng bounding box nội bộ để association nhưng không được vẽ rectangle.

Với mỗi bounding box:

~~~python
point_x = (x1 + x2) / 2
point_y = y2
~~~

Mỗi point có schema:

~~~python
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
~~~

Làm mượt point bằng EMA hoặc Kalman filter nhẹ:

~~~python
smooth_x = alpha * new_x + (1 - alpha) * previous_x
smooth_y = alpha * new_y + (1 - alpha) * previous_y
~~~

~~~yaml
tracking_point:
  smoothing_alpha: 0.30
  min_point_distance_px: 4
  radius_px: 5
  show_track_id: false
~~~

Không thêm point vào tracklet nếu khoảng dịch chuyển nhỏ hơn ngưỡng cấu hình.

---

## 5. Tracklet nội bộ

Quản lý bằng cấu trúc có giới hạn:

~~~python
dict[int, deque[TrajectoryPoint]]
~~~

Yêu cầu:

- giới hạn số point hoặc thời gian history;
- bỏ track quá ngắn;
- bỏ track gần như đứng yên;
- kết thúc tracklet khi mất ID quá track_buffer;
- ghi track hoàn chỉnh theo batch ra CSV hoặc Parquet;
- không giữ toàn bộ lịch sử livestream trong RAM;
- không đưa trajectory cá nhân vào renderer production.

~~~yaml
tracklet:
  max_history_seconds: 30
  min_duration_seconds: 1.0
  min_distance_px: 30
  min_points: 6
  stationary_speed_threshold_px_s: 3
~~~

Schema lưu:

~~~csv
camera_id,track_id,frame_id,timestamp,x,y,zone_id
~~~

---

## 6. Zone và hướng di chuyển

Đọc polygon zone từ zones.json.

Mỗi track cần xác định:

~~~text
origin_zone
destination_zone
direction
first_seen_at
last_seen_at
~~~

Chống rung tại biên zone bằng:

- yêu cầu point nằm trong zone liên tiếp một số frame;
- debounce theo thời gian;
- hysteresis;
- không đếm lặp cùng transition của cùng Track ID.

~~~yaml
zones:
  min_inside_frames: 3
  debounce_seconds: 1.0
~~~

---

## 7. Directed Grid Flow

Chia ROI hoặc ground plane thành grid:

~~~yaml
grid:
  columns: 32
  rows: 18
~~~

Ánh xạ point vào cell:

~~~python
cell_x = int(point_x / frame_width * grid_columns)
cell_y = int(point_y / frame_height * grid_rows)
~~~

Nếu có homography, chuyển point sang ground plane trước khi mapping.

Mỗi chuyển động tạo directed edge:

~~~text
cell_A → cell_B
~~~

Phải phân biệt A → B và B → A.

Popularity dựa trên số unique Track ID đi qua edge, không dựa trên tổng số point. Mỗi Track ID chỉ đóng góp một lần cho cùng directed edge trong một hành trình hoặc time bucket.

Không tạo edge giả khi point rung qua lại tại biên cell. Áp dụng cell hysteresis hoặc yêu cầu point đi đủ sâu vào cell mới.

---

## 8. Short Window và Long Window

Không dùng bộ đếm cộng dồn vô hạn. Dùng time bucket và ring buffer:

~~~yaml
common_path:
  bucket_seconds: 1
  short_window_seconds: 30
  long_window_seconds: 180
  short_weight: 0.40
  long_weight: 0.60
~~~

Short window phản ánh thay đổi mới. Long window duy trì xu hướng ổn định.

~~~python
edge_score = (
    short_weight * normalized_short_flow
    + long_weight * normalized_long_flow
)
~~~

Có thể áp dụng exponential decay:

~~~python
decayed_weight = exp(-age_seconds / tau)
~~~

Flow statistics cập nhật khoảng một lần mỗi giây, không tính lại theo từng video frame.

---

## 9. Trích xuất Common Path

Xây directed graph cho từng cặp origin–destination:

~~~text
Node = grid cell
Edge = chuyển động giữa hai cell
Weight = flow score
~~~

Baseline ưu tiên:

~~~text
Directed Grid Flow + weighted graph search
~~~

Chuyển flow score thành positive cost:

~~~python
edge_cost = edge_length / (edge_score + epsilon)
~~~

Tìm candidate route bằng Dijkstra hoặc thuật toán tương đương.

Yêu cầu:

- không tạo loop;
- bỏ edge có support thấp;
- ưu tiên edge có nhiều unique Track ID;
- giới hạn chiều dài tuyến;
- có thể thêm penalty cho góc rẽ quá gấp;
- candidate phải nối đúng origin và destination.

Chuyển danh sách cell thành polyline và làm mượt bằng Ramer–Douglas–Peucker, Chaikin hoặc moving average. Không chạy trajectory clustering nặng trên từng frame.

---

## 10. Ổn định Common Path

~~~yaml
common_path:
  update_interval_seconds: 3
  min_unique_tracks: 5
  switch_margin: 0.20
  confirmation_seconds: 8
  cooling_seconds: 20
  min_path_edge_support: 3
  path_similarity_threshold: 0.70
  centerline_ema_alpha: 0.20
~~~

Mỗi path có trạng thái:

~~~text
candidate
active
cooling
retired
~~~

Candidate chỉ thay Active Path khi:

~~~python
candidate_score >= active_score * (1 + switch_margin)
~~~

và điều kiện duy trì đủ lâu:

~~~python
candidate_duration >= confirmation_seconds
~~~

So sánh hai path bằng directed-edge overlap:

~~~python
similarity = len(edges_a & edges_b) / len(edges_a | edges_b)
~~~

Nếu similarity vượt threshold, coi là cùng tuyến và chỉ cập nhật centerline bằng EMA.

Khi lưu lượng giảm:

~~~text
active → cooling → retired
~~~

Không xóa Active Path ngay khi camera mất detection ngắn. Candidate Path chỉ dùng nội bộ; UI production chỉ hiển thị Active Path.

---

## 11. Realtime architecture

Không chạy toàn bộ trong một vòng lặp block nối tiếp.

~~~text
Camera Reader
      ↓
Bounded Latest-frame Queue
      ↓
Detection + Tracking
      ↓
Bounded Point Event Queue
      ↓
Trajectory/Flow Worker
      ↓
Common Path Worker
      ↓
Atomic/Immutable Snapshot
      ↓
Video Renderer + WebSocket
~~~

Yêu cầu:

- bounded queue;
- ưu tiên frame mới;
- drop stale frame nếu inference không kịp;
- analytics không block detector;
- renderer không chờ Common Path Worker;
- shared snapshot dùng lock ngắn hoặc immutable object;
- graceful shutdown;
- flush tracklet và bucket khi dừng;
- reconnect RTSP bằng backoff.

~~~yaml
realtime:
  frame_queue_size: 2
  point_event_queue_size: 1000
  drop_stale_frames: true
  flow_update_interval_seconds: 1
  common_path_update_interval_seconds: 3
  websocket_publish_interval_seconds: 1
~~~

---

## 12. Video renderer

Thứ tự layer:

~~~text
1. Raw video frame
2. Active Common Path corridor
3. Active Common Path centerline và arrows
4. Current tracking points
5. Optional zones
6. FPS, latency và people count
~~~

Không vẽ:

~~~text
bounding boxes
individual trajectory tails
completed tracklets
candidate paths trong production
grid debug trong production
edge-flow debug trong production
~~~

Common Path cần có:

- centerline đã làm mượt;
- corridor bán trong suốt;
- mũi tên thể hiện hướng;
- độ dày phản ánh lưu lượng với giới hạn min/max;
- label ngắn như Entrance A → Exit B;
- support tính theo unique tracks.

~~~yaml
common_path_style:
  centerline_color_bgr: [0, 255, 0]
  corridor_color_bgr: [0, 180, 0]
  corridor_opacity: 0.25
  centerline_opacity: 0.75
  min_width_px: 6
  max_width_px: 18
  arrow_spacing_px: 80
  animate_transition_ms: 500

point_style:
  radius_px: 5
  color_bgr: [0, 255, 255]
  outline_color_bgr: [0, 0, 0]
  show_id: false
~~~

Nếu chưa đủ dữ liệu, không vẽ path giả. Hiển thị:

~~~text
Đang thu thập dữ liệu luồng di chuyển...
~~~

---

## 13. Chuyển đổi path mượt trên UI

Khi Candidate được xác nhận thành Active:

- không thay polyline trong một frame;
- resample đường cũ và mới về cùng số điểm;
- nội suy trong khoảng 300–700 ms;
- hoặc cập nhật centerline bằng EMA;
- giữ direction arrow ổn định trong quá trình chuyển.

~~~python
display_path = (1 - transition_t) * old_path + transition_t * new_path
~~~

Renderer luôn dùng display_path, không dùng raw candidate.

---

## 14. UI controls

Tạo các toggle:

~~~text
Tracking Points         ON mặc định
Common Path             ON mặc định
Direction Arrows        ON mặc định
Zones                   OFF mặc định
Track IDs               OFF mặc định
Debug Grid              OFF mặc định
Debug Edge Flow         OFF mặc định
Debug Candidate Path    OFF mặc định
~~~

UI hiển thị tối thiểu:

~~~text
Current people
Processing FPS
End-to-end latency
Active Common Path
Unique tracks supporting path
Path direction
Common Path confidence/state
~~~

---

## 15. API và WebSocket

Tạo hoặc cập nhật:

~~~text
GET /api/analytics/common-paths
GET /api/analytics/flows
GET /api/analytics/metrics
GET /api/config/visualization
PATCH /api/config/visualization
WS /ws/live
~~~

Common Path response:

~~~json
{
  "path_id": "entrance_a__exit_b__01",
  "origin_zone": "entrance_a",
  "destination_zone": "exit_b",
  "state": "active",
  "direction": "A_TO_B",
  "score": 0.82,
  "confidence": 0.88,
  "unique_tracks_short": 12,
  "unique_tracks_long": 48,
  "polyline": [[100, 420], [250, 390], [500, 350], [850, 310]],
  "updated_at": 1720000000.0
}
~~~

Không gửi toàn bộ trajectory history qua WebSocket. Chỉ gửi frame hoặc frame URL, current point snapshot, Active Common Path snapshot, metrics và visualization state.

---

## 16. Output

Sinh tối thiểu:

~~~text
outputs/realtime_point_common_path.mp4
outputs/trajectories.csv
outputs/edge_flows.json
outputs/common_paths.json
outputs/common_path_timeline.csv
outputs/realtime_benchmark.csv
~~~

Video output phải giống UI production:

- có current tracking points;
- có Active Common Path;
- không có bounding box;
- không có trajectory riêng từng người.

---

## 17. Tests bắt buộc

Viết unit/integration test cho:

1. Bottom-center point được tính đúng.
2. Renderer không vẽ bounding box.
3. Renderer không vẽ individual trajectory.
4. Renderer vẽ được Active Common Path.
5. Candidate Path không xuất hiện trong production mode.
6. A → B khác B → A.
7. Một Track ID không bị đếm lặp trên cùng edge.
8. Người đứng lâu không làm tăng popularity.
9. Track quá ngắn không ảnh hưởng Common Path.
10. Rung ở biên cell không tạo flow giả.
11. Candidate không thay Active ngay lập tức.
12. Candidate đủ mạnh và đủ lâu sẽ thành Active.
13. Path transition trên UI không nhảy đột ngột.
14. Mất detection ngắn không xóa Active Path.
15. Ring buffer không tăng RAM vô hạn.
16. Analytics chậm không block inference.
17. Video output mở được và có đúng FPS/resolution.
18. WebSocket không gửi toàn bộ trajectory history.

---

## 18. Kịch bản kiểm thử

### Stable flow

~~~text
80% người đi A → B
20% người đi A → C
~~~

Kỳ vọng: UI hiển thị một Active Common Path A → B.

### Short noise

Trong vài giây có 2–3 người đi A → C.

Kỳ vọng: đường A → B vẫn hiển thị; Candidate không xuất hiện trên UI.

### Real flow change

Trong ít nhất 20–30 giây, phần lớn người chuyển sang A → C.

Kỳ vọng:

~~~text
A → C candidate nội bộ
→ duy trì đủ confirmation time
→ trở thành Active
→ UI chuyển mượt từ A → B sang A → C
~~~

### Empty scene

Kỳ vọng:

- không crash;
- không tạo path giả;
- giữ path cũ trong cooling period;
- sau cooling period mới làm mờ hoặc ẩn.

### Bidirectional flow

Kỳ vọng: A → B và B → A được phân tích độc lập.

---

## 19. Metrics và benchmark

Đo:

~~~text
input_fps
processing_fps
inference_ms
tracking_ms
flow_aggregation_ms
common_path_compute_ms
render_ms
encode_ms
end_to_end_latency_ms
frame_queue_size
dropped_frames
active_tracks
valid_tracks
common_path_switches_per_minute
candidate_rejections
cpu_percent
ram_mb
gpu_memory_mb
~~~

So sánh trước và sau khi thêm Common Path overlay:

| Metric | Before | After |
|---|---:|---:|
| Processing FPS | ... | ... |
| End-to-end latency | ... | ... |
| Render latency | ... | ... |
| RAM | ... | ... |
| CPU/GPU | ... | ... |
| Path switches/minute | N/A | ... |

Không điền số nếu chưa đo.

---

## 20. Thứ tự triển khai

~~~text
Milestone 1: Chạy baseline và đo pipeline hiện tại
Milestone 2: Point-only renderer, bỏ bbox và individual trajectory
Milestone 3: Bounded internal tracklet storage
Milestone 4: Zone và directed grid flow
Milestone 5: Short/long window aggregation
Milestone 6: Candidate/Active/Cooling state machine
Milestone 7: Active Common Path snapshot
Milestone 8: Overlay Common Path lên video UI
Milestone 9: Smooth path transition
Milestone 10: Tests, benchmark và README
~~~

Sau mỗi milestone, chạy test và báo cáo:

~~~text
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
~~~

---

## 21. Acceptance criteria

Chỉ báo hoàn thành khi:

- video realtime chạy ổn định;
- mỗi người chỉ hiển thị bằng bottom-center point;
- không có bounding box;
- không có individual trajectory/pathway trên video production;
- individual tracklets vẫn được lưu nội bộ;
- Active Common Path được overlay trực tiếp trên video;
- Common Path có mũi tên thể hiện hướng;
- Candidate Path không xuất hiện trên UI production;
- Common Path không nhảy vì vài frame hoặc vài track nhiễu;
- khi luồng thay đổi thật, UI chuyển sang path mới mượt;
- popularity tính theo unique Track ID;
- người đứng lâu không làm sai kết quả;
- analytics không block detector/tracker;
- queue và history có giới hạn;
- RAM không tăng vô hạn;
- API/WebSocket không gửi dữ liệu dư thừa;
- output video mở được;
- tests pass;
- benchmark sử dụng dữ liệu thật;
- README có lệnh chạy thật.

---

## 22. Báo cáo cuối

Sau khi hoàn thành, trả về:

~~~text
ROOT CAUSE
- ...

IMPLEMENTED
- Point-only tracking UI
- Internal tracklet processing
- Stable Common Path
- Realtime overlay
- Smooth path transition

CHANGED FILES
- ...

TESTED
- ...

RESULT
- FPS trước và sau
- End-to-end latency
- Render latency
- Common Path switches/phút
- Thời gian phản ứng khi flow thay đổi
- CPU/RAM/GPU
- Output đã tạo

SELECTED PARAMETERS
- Grid size
- Short/long window
- Switch margin
- Confirmation time
- Cooling time

LIMITATIONS
- ...

NEXT
- ...
~~~

Bắt đầu bằng cách chạy baseline trên một đoạn ngắn của video hiện tại. Sau đó triển khai point-only renderer trước, xác minh video không còn bounding box hoặc individual trajectory, rồi mới thêm Common Path Worker và overlay Common Path lên UI.

