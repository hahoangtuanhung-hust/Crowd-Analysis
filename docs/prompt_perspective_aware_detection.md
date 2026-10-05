# Prompt triển khai Perspective-Aware Detection theo TOP / MIDDLE / BOTTOM

Bạn đang làm việc trên project **Crowd Analysis / Common Path**.

## Bối cảnh hiện tại

Pipeline hiện tại:

```text
Video / RTSP
    ↓
Capture
    ↓
Detection scheduler
    ↓
YOLO26n
    ↓
ByteTrack hybrid
    ↓
Tracklet
    ↓
Common Path
```

Profile Shibuya hiện tại dùng detector tiled:

```text
1 full frame + 4 tiles (2x2, overlap khoảng 20%)
```

Mục tiêu của tiled inference là giữ khả năng detect người nhỏ và ở xa, nhưng workload rất lớn.

Hiện tại dự án cũng đã có sampled detection:

```text
inference_interval = N
```

Ví dụ N=3 hoặc N=5:

```text
Detection frame:
YOLO → ByteTrack association

Intermediate frame:
ByteTrack coast()
```

Không được phá semantics này.

Tracker hiện tại là ByteTrack hybrid và Common Path chỉ sử dụng observed tracking points, không sử dụng prediction-only points làm evidence.

---

# MỤC TIÊU

Triển khai cơ chế **Perspective-Aware Detection Regions**.

Chia frame theo chiều dọc thành 3 vùng:

```text
TOP / FAR
──────────────────────────────
người nhỏ / ở xa

→ tile nhỏ hơn
→ detector input resolution cao hơn
→ có thể optional upscale 2×
→ ưu tiên recall


MIDDLE
──────────────────────────────
người kích thước trung bình

→ tiled YOLO bình thường
→ cân bằng recall / performance


BOTTOM / NEAR
──────────────────────────────
người lớn / gần camera

→ full-frame hoặc ít tile hơn
→ không SR
→ ưu tiên giảm workload
```

Mục tiêu chính KHÔNG phải làm video đẹp hơn.

Mục tiêu là:

1. tăng khả năng detect người nhỏ ở vùng xa;
2. duy trì tracking ổn định;
3. giảm số detector images/frame so với tiled 2×2 cố định nếu có thể;
4. không làm sai tọa độ bbox;
5. không phá ByteTrack;
6. không phá sampled detection;
7. không thay đổi semantics Common Path;
8. có metric rõ ràng để A/B benchmark.

---

# 1. AUDIT CODE TRƯỚC KHI SỬA

Trước khi thay đổi code, hãy đọc và xác định source of truth của:

```text
backend/app/video/pipeline.py
backend/app/inference/
backend/app/tracking/
backend/app/core/config*
configs/shibuya.yaml
configs/shibuya-sampled-n3.yaml
configs/shibuya-sampled-n5.yaml
```

Tìm chính xác:

- detector được gọi ở đâu;
- tiled inference đang được tạo ở đâu;
- tile coordinates đang được map về full-frame như thế nào;
- duplicate detections được merge như thế nào;
- `imgsz` được truyền cho Ultralytics ở đâu;
- `inference_interval` được áp dụng ở đâu;
- `coast()` được gọi ở đâu;
- metric detector image count hiện được thu ở đâu.

Không giả định cấu trúc code nếu chưa đọc source.

---

# 2. KHÔNG HARD-CODE PIXEL ABSOLUTE

Perspective regions phải sử dụng normalized Y coordinates:

```text
0.0 = top frame
1.0 = bottom frame
```

Ví dụ config ban đầu:

```yaml
detector:
  perspective_regions:
    enabled: true

    regions:
      - name: far
        y_min: 0.0
        y_max: 0.35

      - name: middle
        y_min: 0.30
        y_max: 0.72

      - name: near
        y_min: 0.67
        y_max: 1.0
```

Cho phép overlap nhẹ giữa các region để tránh mất người ở boundary.

Không hard-code:

```text
y < 250
250 < y < 500
...
```

vì camera resolution có thể thay đổi.

---

# 3. FAR REGION

Đối với vùng TOP / FAR:

```text
0.0 → khoảng 0.35 frame height
```

mục tiêu là detect người rất nhỏ.

Implement profile kiểu:

```yaml
name: far

tile:
  enabled: true

tile_rows: 2
tile_cols: 2

input_size: 960

upscale:
  enabled: false
  scale: 2.0
  method: lanczos
```

Không nhất thiết phải đúng chính xác các con số trên nếu architecture hiện tại có representation khác.

Điều quan trọng:

- FAR phải được crop trước;
- FAR có thể chia thành tile nhỏ;
- tile được đưa vào YOLO với input resolution lớn hơn;
- detection cuối cùng phải map chính xác về original-frame coordinates.

Ví dụ:

```text
Original frame
1280×720

TOP:
y=0 → 252

crop:
1280×252

chia thành:
2 hoặc 4 tile

tile
    ↓
resize/upscale
    ↓
YOLO
    ↓
bbox tile coordinates
    ↓
inverse scale
    ↓
add crop offset
    ↓
original frame bbox
```

---

# 4. MIDDLE REGION

MIDDLE sử dụng strategy cân bằng:

```text
khoảng y=0.30 → 0.72
```

Có thể sử dụng:

```text
2 horizontal tiles

hoặc

2×2 nếu audit cho thấy cần thiết
```

Nhưng tránh mặc định tạo quá nhiều detector images.

Input size nên giữ gần profile hiện tại.

Ví dụ:

```yaml
name: middle

tile:
  enabled: true

tile_rows: 1
tile_cols: 2

input_size: 640
```

---

# 5. NEAR REGION

BOTTOM / NEAR:

```text
khoảng y=0.67 → 1.0
```

người thường đã đủ lớn.

Không sử dụng Super Resolution.

Ưu tiên một trong hai:

```text
A.
crop near region
→ single detector pass

hoặc

B.
rely on full-frame pass nếu full-frame pass vẫn được giữ
```

Không được tiled quá nhỏ nếu không có bằng chứng cần thiết.

Mục tiêu:

```text
minimum compute
+
không giảm recall đáng kể
```

---

# 6. THIẾT KẾ ƯU TIÊN: REGION-ONLY HOẶC FULL-FRAME + FAR

Audit và hỗ trợ A/B giữa ít nhất hai policy.

## POLICY A — Region-only

```text
FAR:
high-resolution tiled

MIDDLE:
medium tiled

NEAR:
single crop
```

Không chạy full-frame detector.

Ví dụ workload:

```text
far    = 2 images
middle = 2 images
near   = 1 image

TOTAL = 5
```

Sau đó tối ưu tiếp xuống 3–4 nếu quality cho phép.

---

## POLICY B — Full-frame + far refinement

```text
full frame
    +
far region high-resolution tiles
```

Ví dụ:

```text
1 full frame
+
2 FAR tiles

TOTAL = 3 detector images
```

Đây là policy rất đáng benchmark vì có thể giảm từ:

```text
5 detector images/frame
```

xuống:

```text
3 detector images/frame
```

nhưng vẫn dành thêm resolution cho pedestrian nhỏ phía xa.

Không mặc định policy nào tốt hơn trước benchmark.

---

# 7. KHÔNG DOUBLE COUNT DETECTION

Region overlap sẽ sinh duplicate bbox.

Phải sử dụng merge logic hiện tại nếu có thể.

Không viết thêm một NMS hoàn toàn mới nếu repository đã có source-aware merge.

Sau khi detections được map về full frame:

```text
far detections
+
middle detections
+
near detections
+
optional full frame detections

↓

existing merge / deduplication

↓

ByteTrack
```

Kiểm tra kỹ người đứng ở boundary:

```text
far ↔ middle

middle ↔ near
```

không được tạo hai person detections cho cùng một người.

---

# 8. OPTIONAL UPSCALE / SUPER RESOLUTION

Giai đoạn đầu CHƯA đưa Real-ESRGAN hoặc neural SR model vào production.

Chỉ support optional interpolation upscale:

```text
INTER_CUBIC

hoặc

INTER_LANCZOS4
```

Ví dụ:

```yaml
upscale:
  enabled: false
  scale: 2.0
  method: lanczos
```

Nếu bật:

```text
crop
↓
upscale 2×
↓
YOLO
↓
convert bbox coordinates về crop gốc
↓
convert tiếp về frame gốc
```

Phải có unit test cho inverse coordinate transform.

Neural SR để ở phase riêng sau benchmark.

---

# 9. KHÔNG PHÁ SAMPLED DETECTION

Perspective-aware detection chỉ được chạy trên detection frames.

Ví dụ N=5:

```text
frame 0
perspective detector
→ ByteTrack update()

frame 1
coast()

frame 2
coast()

frame 3
coast()

frame 4
coast()

frame 5
perspective detector
→ ByteTrack update()
```

KHÔNG chạy FAR detector riêng trong intermediate frames ở phase này.

Không sửa scheduler semantics.

---

# 10. KHÔNG PHÁ TRACKING

Tracker phải tiếp tục nhận:

```text
detections in original-frame coordinates
```

Tracker không được biết detection đến từ:

```text
far tile
middle tile
near crop
full frame
```

trừ khi metadata cần cho debugging.

Không thay đổi:

```text
ByteTrack association thresholds
track buffer
hybrid IoU/distance logic
Kalman behavior
coast()
```

trong vòng benchmark đầu tiên.

Mục tiêu là isolate ảnh hưởng của detector policy.

---

# 11. KHÔNG PHÁ COMMON PATH

Không sửa:

```text
TrackletAggregationEngine
Common Path thresholds
support semantics
Path ID lifecycle
direction logic
```

Prediction-only track vẫn:

```text
observed=false
```

và không được tạo Common Path support.

---

# 12. CONFIG

Không sửa trực tiếp `configs/shibuya.yaml` thành experimental behavior.

Tạo profile riêng, ví dụ:

```text
configs/shibuya-perspective.yaml
configs/shibuya-perspective-n3.yaml
configs/shibuya-perspective-n5.yaml
```

Ưu tiên dùng `extends` nếu config system đang hỗ trợ.

Ví dụ conceptual config:

```yaml
extends: shibuya.yaml

detector:

  perspective_regions:

    enabled: true

    strategy: full_frame_plus_far

    regions:

      far:
        y_min: 0.0
        y_max: 0.38

        tiles:
          rows: 1
          cols: 2

        imgsz: 960

        upscale:
          enabled: false
          scale: 2.0
          interpolation: lanczos


      middle:
        y_min: 0.32
        y_max: 0.72

        tiles:
          rows: 1
          cols: 2

        imgsz: 640


      near:
        y_min: 0.66
        y_max: 1.0

        tiles:
          rows: 1
          cols: 1

        imgsz: 640
```

Đây chỉ là design direction.

Adapt schema theo code thực tế.

---

# 13. METRICS BẮT BUỘC

Thêm metric để biết workload thật.

Ít nhất:

```text
detector_images_total

detector_images_far
detector_images_middle
detector_images_near
detector_images_full

detections_far_raw
detections_middle_raw
detections_near_raw

detections_after_merge

perspective_preprocess_ms
perspective_detector_ms
perspective_merge_ms
```

Nếu dễ implement, thêm:

```text
far_person_count
middle_person_count
near_person_count
```

dựa trên bbox bottom-center sau merge.

---

# 14. DEBUG VISUALIZATION

Thêm optional debug overlay:

```yaml
visualization:
  perspective_regions: false
```

Khi bật, vẽ:

```text
FAR
MIDDLE
NEAR
```

và boundary lines.

Ví dụ:

```text
┌────────────────────────────┐
│ FAR                        │
│                            │
├────────────────────────────┤
│ MIDDLE                     │
│                            │
├────────────────────────────┤
│ NEAR                       │
│                            │
└────────────────────────────┘
```

Chỉ phục vụ debug.

Không bật production mặc định.

---

# 15. TESTS

Bổ sung unit test ít nhất cho:

### Region generation

```text
normalized Y → exact pixel coordinates
```

Test:

```text
720p
1080p
odd dimensions
```

### Coordinate mapping

Ví dụ:

```text
crop:
x=100
y=50

bbox inside crop:
x1=20
y1=30
x2=70
y2=120

expected original:
x1=120
y1=80
x2=170
y2=170
```

### Upscaling inverse transform

Ví dụ crop upscale 2×:

```text
bbox detector:
[200, 100, 400, 300]

scale=2

crop bbox:
[100, 50, 200, 150]
```

sau đó apply original crop offset.

### Boundary duplicate

Một bbox xuất hiện ở:

```text
FAR
+
MIDDLE
```

sau merge chỉ còn một detection.

### Scheduler compatibility

Với:

```text
inference_interval=5
```

phải đảm bảo:

```text
perspective detector chỉ chạy ở detection frame
```

các frame khác gọi:

```text
coast()
```

### Disabled behavior

```yaml
perspective_regions:
  enabled: false
```

phải giữ behavior hiện tại.

---

# 16. BENCHMARK MATRIX

Sau khi implement, chạy ít nhất các profile:

```text
A
shibuya.yaml
N=1
current 1 full + 4 tiled

B
perspective
N=1

C
perspective
N=3

D
perspective
N=5
```

Không chạy N=10 trước khi N=5 quality gate pass.

Đo:

```text
processing FPS

detector scans/sec

detector images/sec

model_predict p50/p95

postprocess p50/p95

tracking p50/p95

CPU utilization

GPU utilization

GPU memory

temporary Track IDs

average observed detections / scan

track fragmentation proxies

Common Path support

Path ID changes
```

---

# 17. VISUAL QUALITY REVIEW

Đặc biệt review vùng FAR.

Chọn một số pedestrian ở:

```text
TOP
TOP/MIDDLE boundary
MIDDLE
MIDDLE/BOTTOM boundary
BOTTOM
```

Kiểm tra:

```text
person có được detect không

bbox có jitter không

Track ID có continuity không

ID có switch khi crossing không

người nhỏ có bị mất nhiều scan liên tiếp không

bbox có nhảy khi chuyển region không
```

Không kết luận từ số lượng detection đơn thuần.

---

# 18. QUALITY GATE

Perspective version chỉ được coi là tốt hơn baseline khi:

```text
performance tăng rõ ràng
```

và đồng thời không có regression rõ ràng về:

```text
small-person detection

track continuity

ID fragmentation

direction stability

Common Path stability
```

Nếu performance tốt nhưng FAR recall giảm rõ:

```text
QUALITY_FAIL
```

Nếu chưa đủ ground truth:

```text
QUALITY_PENDING
```

Không tuyên bố accuracy PASS dựa trên cảm quan.

---

# 19. KHÔNG LÀM TRONG TASK NÀY

Không:

```text
chuyển TensorRT

đổi YOLO model

thêm ReID

đổi ByteTrack sang tracker khác

tune Common Path

thêm neural Super Resolution

thay scheduler sampled detection

refactor lớn ngoài phạm vi
```

Task này chỉ tập trung:

```text
PERSPECTIVE-AWARE DETECTOR POLICY
```

---

# 20. ƯU TIÊN IMPLEMENTATION

Nếu code hiện tại cho phép, ưu tiên triển khai phiên bản đầu tiên:

```text
FULL FRAME
+
2 FAR TILES
```

tức:

```text
┌──────────────────────────────┐
│ FAR TILE 1 │ FAR TILE 2      │
├────────────┴─────────────────┤
│                              │
│          FULL FRAME          │
│                              │
└──────────────────────────────┘
```

Mục tiêu:

```text
baseline:
5 detector images / detection frame

candidate:
3 detector images / detection frame
```

Sau đó benchmark.

Nếu FAR recall vẫn chưa đủ mới thử:

```text
FAR imgsz = 960
```

sau đó mới thử:

```text
FAR interpolation upscale 2×
```

Không bật tất cả optimization cùng lúc.

---

# 21. DELIVERABLE

Sau khi hoàn thành, trả về report có cấu trúc:

```text
1. Files changed

2. Existing architecture discovered

3. Perspective detector architecture implemented

4. Region definitions

5. Coordinate transformation

6. Detector images/frame
   baseline vs candidate

7. Scheduler compatibility

8. Tracker/Common Path compatibility

9. Tests added

10. Test results

11. Benchmark commands

12. Performance results nếu đã chạy

13. Visual review status

14. Known limitations

15. Recommended next experiment
```

Phân loại cuối cùng:

```text
IMPLEMENTATION_PASS
QUALITY_PENDING
```

nếu code/tests đã đúng nhưng chưa đủ benchmark/ground truth.

Không tự động merge vào production config nếu chưa qua quality gate.
