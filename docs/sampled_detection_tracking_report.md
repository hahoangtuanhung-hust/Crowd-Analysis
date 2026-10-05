# Báo Cáo: Sampled Detection — Tracking theo chu kỳ

## 1. Commit/Model/Config Baseline

- **Baseline commit**: `main@f99ec01` (tài liệu 28/09/2026)
- **Model**: `yolo26n.pt` (Ultralytics, FP32)
- **Config baseline**: `configs/shibuya.yaml` (`inference_interval=1`)
- **Tracker**: ByteTrack hybrid + `coast()` adapter nội bộ

## 2. Phát hiện lỗi quan trọng và thay đổi

### Lỗi semantics nghiêm trọng (sửa trong đợt này)

**`inference_interval` cũ dùng `frame_id % N != 0 → continue`** — tức frame bị bỏ **hoàn toàn**, không gọi `coast()`, Kalman state không được advance. Hệ quả:
- Tracker không biết rằng N-1 frame đã trôi qua
- Frame detection tiếp theo sẽ advance Kalman từ `_last_source_frame_id` đến current (đúng), nhưng **không có output preview** cho user trong N-1 frame đó
- Nếu live mode drop đúng frame là bội số của N, detection schedule bị stall hoàn toàn (modulo problem)

### Các thay đổi triển khai

| File | Thay đổi |
|---|---|
| [`backend/app/video/pipeline.py`](file:///d:/AITHUCCHIEN/VinFast/project-crownd-analysis/backend/app/video/pipeline.py) | Deadline-based scheduler `_is_detection_frame()`, gọi `coast()` cho intermediate frames, thêm metric mới |
| [`backend/app/video/pipeline.py`](file:///d:/AITHUCCHIEN/VinFast/project-crownd-analysis/backend/app/video/pipeline.py) | `PipelineStats` thêm: `policy_skipped_frames`, `association_updates`, `prediction_steps`, `detection_scan_deadline_misses` |
| [`configs/shibuya-sampled-n3.yaml`](file:///d:/AITHUCCHIEN/VinFast/project-crownd-analysis/configs/shibuya-sampled-n3.yaml) | Profile N=3: ~10 scans/giây nguồn |
| [`configs/shibuya-sampled-n5.yaml`](file:///d:/AITHUCCHIEN/VinFast/project-crownd-analysis/configs/shibuya-sampled-n5.yaml) | Profile N=5: ~6 scans/giây nguồn |
| [`configs/shibuya-sampled-n10.yaml`](file:///d:/AITHUCCHIEN/VinFast/project-crownd-analysis/configs/shibuya-sampled-n10.yaml) | Profile N=10 (EXPERIMENTAL, cần gate N=5 trước) |
| [`tests/test_sampled_detection.py`](file:///d:/AITHUCCHIEN/VinFast/project-crownd-analysis/tests/test_sampled_detection.py) | 14 tests covering scheduler, coast semantics, metrics |

## 3. Semantics scheduler mới (`_is_detection_frame`)

```
Detect frame 0 (epoch start, _last_detection_frame_id=None)
Detect khi: current_frame_id - last_detection_frame_id >= N
Nếu deadline frame bị drop: detect frame khả dụng tiếp theo (không stall)
```

**Ví dụ N=3, frames thực tế [0,1,2,4,5,6,8,9,10] (frames 3,7 bị drop trong live):**
- frame 0 → detect ✓
- frame 1, 2 → coast
- frame 4 → detect ✓ (4-0=4 ≥ 3, deadline miss logged)
- frame 5, 6 → coast
- frame 8 → detect ✓ (8-4=4 ≥ 3, deadline miss logged)

## 4. Quy ước `inference_interval` (sửa so với tài liệu)

`inference_interval = N = khoảng cách giữa hai lần detection` (số source frames).

**Không phải** số frame bỏ qua (như DeepStream `interval`). N=3 có nghĩa detect ở frame 0, 3, 6, 9... với 30 FPS source → 10 scan/giây.

Cũ (modulo): N=3 detects frames 0, 3, 6, 9... nhưng frame 3 bị drop → tiếp theo detect ở frame 6 (đúng).

Mới (deadline): N=3 detects frames 0, 3, 6, 9... frame 3 bị drop → detect frame 4 ngay (tốt hơn).

## 5. Workload lý thuyết (không đo thực tế)

| N | Scans/giây nguồn | Ảnh detector/giây nguồn | Gap quan sát |
|---:|---:|---:|---:|
| 1 | 30 | 150 | 33ms |
| 3 | 10 | 50 | 100ms |
| 5 | 6 | 30 | 167ms |
| 10 | 3 | 15 | 333ms |

## 6. Quality thresholds cần check (N=5)

Tại N=5 (167ms gap), cần audit với config shibuya hiện tại:
- `max_observation_gap_seconds = 0.8s` → OK (167ms << 800ms)
- `min_track_duration_seconds = 0.35s` → cần ≥ 3 observed scans (~500ms)
- Discovery delay: người mới xuất hiện chờ tối đa 167ms trước khi được phát hiện
- Đây là tradeoff cần báo cáo, không phải bug

## 7. Các lệnh rollback và chạy

```powershell
# Baseline N=1 (không thay đổi)
$env:CROWD_CONFIG = "configs/shibuya.yaml"

# N=3 candidate
$env:CROWD_CONFIG = "configs/shibuya-sampled-n3.yaml"

# N=5 candidate  
$env:CROWD_CONFIG = "configs/shibuya-sampled-n5.yaml"

# Modal batch N=3
modal run --quiet modal_common_path.py `
  --input data/videos/data-shibuya-10m.mp4 `
  --engine tracklet_aggregation `
  --mode offline_fast `
  --config configs/shibuya-sampled-n3.yaml `
  --run-id shibuya-sampled-n3-$(Get-Date -Format yyyyMMdd-HHmmss)
```

## 8. Kết luận

**Trạng thái**: `QUALITY_PENDING`

Code, semantics, tests (14 pass) và configs đã hoàn tất. **Chưa đo** IDF1/HOTA/FPS thực tế trên Shibuya vì cần:
1. Ground truth annotations (chưa có)
2. Chạy Modal batch với N=3 và N=5 để so detection scans/giây thực đo
3. Visual review video để kiểm tra track continuity

**Lỗi baseline được sửa**: `inference_interval` cũ dùng `continue` mà không gọi `coast()` — đây là bug về semantics cần sửa ngay cả khi N=1 (không ảnh hưởng N=1 nhưng ảnh hưởng mọi N>1). Common Path engine/config không bị thay đổi.

**N=10**: Giữ lại ở experimental, chỉ thử sau khi N=5 pass quality gate.
