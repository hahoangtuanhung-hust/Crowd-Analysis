# Báo cáo tracking người nhỏ/xa — 02/10/2026

## Kết luận nghiệm thu

`QUALITY_REVIEW_PENDING`. Đã hoàn tất thay đổi tracker, cấu hình candidate, diagnostic có giới hạn, replay từ detection cache và regression test. Chưa có annotation MOT/GT cho Shibuya nên **không** công bố IDF1, HOTA, AssA, recall hay khẳng định tracking đã ổn định. Video nguồn có SHA-256 trong cache không có trong workspace, do đó không tạo video overlay trước/sau giả lập.

Common Path không bị sửa: không thay engine/config aggregation, ranking, Top K, Path ID, màu hay smoothing. `TrackletPoint` vẫn nhận bottom-center raw và chỉ nhận `confirmed=true`, `observed=true` theo contract hiện có; prediction vẫn `observed=false`.

## Audit thực tế

- HEAD khi audit: `4269fc8`; working tree đã có thay đổi dở dang ở tracker/schema/candidate config, được hoàn thiện trong phạm vi nhiệm vụ.
- Baseline `configs/shibuya.yaml`: detector floor `0.04`, tile 2×2 + full frame, overlap `0.20`; `track_low/high/new = 0.04/0.08/0.08`, `match_thresh=0.72`, low-stage `0.60`, buffer 180 frame.
- Đường detector có một điểm loại box đáng kể: box chạm biên tile nội bộ bị loại nếu confidence dưới `0.12`. Điều này mâu thuẫn với mục tiêu chuyển weak evidence tới ByteTrack, nhất là seam. Merge sau đó đã source-aware và không gộp box cùng source.
- Cache Shibuya thực tế `tracking_cache.jsonl` chứa cả trường `detections` trước tracker (vì vậy dùng được làm minimum detection cache) và output tracks lịch sử. Nhưng score thấp nhất đã lưu là `0.0400`; cache không thể đánh giá candidate detector floor `0.02` hoặc tile-edge range `0.02–0.04`.
- Trong 600 frame đầu/19.94 s: 132,546 detection; height p10/p50/p90 = `19.469/31.150/44.500 px`, width p10/p50/p90 = `8.900/13.350/18.913 px`; 30,749 box cao ≤24 px và 73,036 ≤32 px. Đây là phân bố box dự đoán, **không** phải định nghĩa small/far theo GT.
- Provenance input cache: source 1280×720, 30.002 FPS, model hash `9b09…4fef`, source hash `f3b8…9670`; xem `outputs/common_path/shibuya-20260930-092330/.../provenance.json`.

## Thay đổi triển khai

1. `ByteTrackTracker` giữ ByteTrack one-to-one assignment, nhưng candidate có gate bottom-center thích ứng theo chiều cao bbox, pixel noise floor, growth theo frame gap, uncertainty covariance có cap và hard geometry reject. Baseline vẫn giữ gate frame-diagonal cũ; không gọi đây là Mahalanobis matching.
2. Low-confidence floor giảm riêng cho recovery (`0.02`); high/new threshold giữ `0.08`, nên weak box không tự sinh track mới chỉ vì hạ floor.
3. Candidate hạ ngưỡng edge tile từ `0.12` xuống `0.04`; merge source-aware vẫn quyết định duplicate. Điều này cần detector rerun mới đo được.
4. Motion history dùng source frame gap; `TrackingPipeline`, Modal clip runner và live runner chuyển `source_timestamp`. Không trộn frame với giây.
5. Direction là diagnostic tracker output: lịch sử chỉ chứa observation thật, dùng cửa sổ source-time, reset khi gap/reorder, có `unknown/stationary/moving`, hysteresis, quality/span. Vector là unit vector ảnh (`+x` sang phải, `+y` xuống dưới), không phải hướng địa lý. Prediction có thể hiển thị estimate cuối cùng nhưng không thêm history hay tăng quality.
6. `association_debug` mặc định tắt; khi bật chỉ giữ buffer có cap và filter theo frame/track. Mỗi event có track/detection index, confidence, bbox/height, IoU, distance/gate, motion score, gap, cost và lý do.

Candidate production: `configs/shibuya-tracking-stable.yaml`. Sample debug tách riêng: `configs/shibuya-tracking-diagnostic.yaml` (frame 100, tối đa 200 events), không dùng làm profile production.

## Replay cùng detection input

Lệnh đã chạy (tái tạo track từ `detections`, không rerun detector):

```powershell
python scripts/replay_detection_cache.py --cache outputs/common_path/shibuya-20260930-092330/shibuya-20260930-092330/tracking_cache.jsonl --config configs/shibuya.yaml --output-dir outputs/tracking_stability/shibuya-cache-20260930-b0-600 --max-frames 600
python scripts/replay_detection_cache.py --cache outputs/common_path/shibuya-20260930-092330/shibuya-20260930-092330/tracking_cache.jsonl --config configs/shibuya-tracking-stable.yaml --output-dir outputs/tracking_stability/shibuya-cache-20260930-c1-600-v2 --max-frames 600
```

| Metric proxy (600 frame, same detector cache) | B0 | C1 | Nhận xét |
| --- | ---: | ---: | --- |
| Observed track-frame | 83,924 | 83,388 | -0.64%; không phải cải thiện coverage |
| Predicted track-frame | 37,266 | 35,840 | -3.83% |
| Unique observed Track ID | 380 | 535 | +40.8%; cần GT để biết là recovery hay false birth/fragmentation |
| Observed ID starts proxy | 5,241 | 4,422 | -15.6%; không phải ID switch metric |
| Observed ID disappearances proxy | 5,039 | 4,227 | -16.1%; không phải ID switch metric |
| Tracker p50/p95 ms | 51.46 / 107.68 | 41.74 / 86.91 | Chạy riêng không kiểm soát tài nguyên; không kết luận FPS |

Artifact gồm `metrics.json`, `metrics.csv`, `tracking_cache.jsonl`, `config_resolved.yaml`, `provenance.json` tại hai thư mục output trên. `metrics.json` gắn rõ `QUALITY_REVIEW_PENDING` và không gán proxy thành quality MOT.

Diagnostic đã chạy 110 frame với config debug riêng; ghi đúng 200 event tại frame 100: 187 `CANDIDATE`, 13 `GEOMETRY_GATE_REJECTED`. Ví dụ reject track 81 → detection 0: distance 1073.63 px, gate 47.50 px, IoU 0, cost 1.0. Xem `outputs/tracking_stability/shibuya-cache-20260930-diagnostic-110-v2/association_diagnostics.jsonl`.

## Ablation và giới hạn

- B0 và C1 đã chạy trên cùng cache. Trong cache này mọi box đều ≥0.04 và high threshold không đổi; vì vậy C1 thực chất cô lập chủ yếu association/lifecycle, không chứng minh detector floor/tile seam mới.
- T1 (threshold-only) và D1 (detector rerun) chưa có bằng chứng hợp lệ vì cache đã cắt dải <0.04. Muốn đo chúng cần chạy lại detector với floor 0.02, lưu bbox/confidence/source/frame/coverage trước tracker và tách tune/holdout theo thời gian.
- H1 chỉ xác nhận contract direction trong synthetic tests/replay; chưa có GT moving/stationary/turn để đo angular error, coverage hay turn delay.
- Không có video source tương ứng cache nên không thể tạo crop/video trước-sau trung thực. Cần cung cấp hoặc mount đúng file có hash `f3b8…9670` rồi render cùng timestamp với bbox/ID/observed-predicted/trail/direction.

## Kiểm thử

Đã pass 49 test liên quan: tracker (small shift, low confidence, crossing, recovery, source-frame gap, adaptive hard gate, direction, turning/reset, diagnostic), detector tile merge/edge, config, pipeline và replay cache. Đã chạy `python -m compileall -q backend scripts`.

Regression bổ sung xác nhận: box nhỏ lệch/far box không bị hijack qua hard gate; low-confidence recovery cũ; crossing/occlusion; observed vs prediction; direction unknown trước đủ span; source timestamp; tile seam weak candidate; config schema/hysteresis; debug buffer bounded; replay provenance.

## Bước nghiệm thu bắt buộc tiếp theo

Tạo tập tuning + holdout có annotation bbox/ID/visibility, đánh dấu small/far bằng GT/ROI trước benchmark; rerun detector cache floor 0.02; chạy TrackEval (IDF1/HOTA/AssA/CLEAR), detection recall/precision và metric direction. Chỉ promote `IMPROVED_AND_VERIFIED` nếu small/far identity/continuity tăng mà coverage, precision và người gần không hồi quy.
