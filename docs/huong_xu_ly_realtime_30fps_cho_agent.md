# Hướng triển khai tối ưu Crowd Analysis hướng tới realtime 30 FPS

Ngày: 28/09/2026. Mốc tài liệu đầu vào: `main@f99ec01`.

## 1. Nhiệm vụ dành cho agent

Bạn là agent triển khai trong repository Crowd Analysis. Hãy đọc code và artifact thực tế, thực hiện tối ưu có đối chứng, bổ sung benchmark và báo cáo kết quả. Không dừng ở việc đề xuất. Mục tiêu ưu tiên là xử lý đủ các frame nguồn Shibuya 1280×720, 30 FPS trên Tesla T4, đồng thời giữ chất lượng detection, tracking và Common Path trong ngưỡng nghiệm thu bên dưới.

Đây là kế hoạch triển khai dựa trên tài liệu hiện trạng và số liệu người dùng cung cấp; người soạn chưa trực tiếp kiểm tra repository hoặc artifact. Tên file mới và tham số mới trong tài liệu là đề xuất, không được giả định đã tồn tại. Xác minh API của đúng phiên bản dependency đã cài trước khi viết code.

Nếu không đạt 30 FPS trên T4 với chất lượng yêu cầu, phải báo cáo chính xác giới hạn đo được và phương án tiếp theo. Không đổi định nghĩa realtime, hạ chất lượng ngầm hoặc dùng FPS hiển thị để công bố thành công.

## 2. Baseline và cách diễn giải số liệu

Run: `shibuya-10m-final-20260926-091731`.

| Chỉ số | Giá trị được cung cấp |
|---|---:|
| Số frame | 16.959 |
| Thời lượng nguồn | 565,267 giây |
| Remote wall time | 4.372,237 giây |
| Processing throughput | 3,88 FPS |
| Tiled inference | 1 full frame + 4 tile / frame nguồn |
| Số ảnh detector ước tính khi chạy đầy đủ | 84.795 |
| Inference p50 / p95 | 146,991 / 169,179 ms |
| Model predict p50 / p95 | 113,788 / 118,921 ms |
| Postprocess/merge p50 / p95 | 30,292 / 49,996 ms |
| Tracking p50 / p95 | 37,823 / 69,950 ms |
| Analytics p50 / p95 | 2,348 / 61,068 ms |
| Common Path compute p50 / p95 | 103,877 / 464,130 ms |
| Render p50 / p95 | 15,579 / 17,798 ms |
| Encode p50 / p95 | 6,428 / 8,900 ms |
| GPU utilization trung bình | 31,025% |
| CPU process trung bình | 99,83% |

Single-pass từng đạt khoảng 10,29 FPS trên clip 65 giây theo thông tin bổ sung của người dùng. FP16 tiled mới có smoke test cải thiện inference p50 khoảng 19%. Motion ROI trên 1.951 frame không giảm số ảnh detector và tăng wall time khoảng 8,2%.

Các lưu ý bắt buộc khi phân tích:

- CPU gần 100% theo cách đo process thường tương đương khoảng một logical core; kiểm tra cách chuẩn hóa và số CPU được cấp. Chưa thể kết luận toàn bộ máy hết CPU.
- GPU utilization thấp gợi ý GPU có thời gian chờ. Chưa đủ chứng minh batch nhỏ, ghi cache hay một hàm cụ thể là nguyên nhân; phải profiling.
- Không cộng các p50/p95 thành tổng latency: các phân vị không cộng được, stage có thể lồng nhau hoặc chạy đồng thời. Xác minh `postprocess_ms` có bao gồm `merge_ms` và `analytics_ms` có bao gồm Common Path hay không.
- 33,33 ms là khoảng thời gian trung bình giữa hai frame ở 30 FPS. Với pipeline song song, throughput 30 FPS không bắt buộc E2E latency từng frame ≤33,33 ms. Tuy nhiên một worker tuần tự mất 147 ms cho mỗi frame vẫn không thể nhận đủ 30 frame/giây nếu không giảm chi phí hoặc thay kiến trúc.
- Offline có backpressure sẽ làm xử lý chậm, không nhất thiết tăng queue vô hạn. Live `drop_oldest` giữ queue giới hạn nhưng phải công bố frame bị bỏ.
- 3,88/30 ≈12,9% chỉ là tỷ lệ công suất; tỷ lệ frame thực sự phân tích trong live phải đo riêng.

## 3. Những hành vi phải giữ nguyên

1. Production Common Path vẫn là `tracklet_aggregation`, không thay bằng grid hoặc trajectory của một cá nhân.
2. Evidence chỉ từ điểm `confirmed=true`, `observed=true`, đúng thứ tự thời gian. Prediction/coast không tạo support.
3. Giữ ba trạng thái `MEASURED`, `SEARCHED_NOT_FOUND`, `NOT_SEARCHED_BY_POLICY`. Vùng không quét không được biến thành detector miss.
4. Support vẫn theo temporary Track ID; không gắn nhãn là số người duy nhất. Không dùng số sample hoặc track dài để tăng support giả.
5. Giữ session/stream epoch, analytics generation, Path ID/màu, lifecycle, route memory và Top K 1–5 không reset state.
6. Giữ bounded queue, backpressure offline, cơ chế realtime drop có metric, reconnect và shutdown sạch.
7. Không bật Motion ROI mặc định; không bật auxiliary analytics để phục vụ benchmark Common Path.
8. Tách profile thử nghiệm khỏi baseline. Không ghi đè cache/artifact cũ; thay precision/backend/scheduler phải thay fingerprint tương ứng.
9. Không thay threshold hoặc `max_det` chỉ để giảm detection và làm FPS đẹp hơn mà không qua quality gate.

## 4. Tiêu chí nghiệm thu

Các ngưỡng dưới đây là tiêu chí kỹ thuật đề xuất cho đợt triển khai, không phải kết quả đã đo hay yêu cầu sản phẩm đã được xác nhận. Đóng băng trong cấu hình benchmark trước khi chạy; nếu cần sửa phải ghi lý do, không nới ngưỡng sau khi thấy kết quả.

### 4.1 Hiệu năng

| Gate | Điều kiện |
|---|---|
| Full-frame throughput | Chạy toàn bộ clip, mỗi frame nguồn được detector tìm kiếm toàn khung ở mức coverage đã khai báo, có tracker update và analytics ingest; không dùng tracking cache, không bỏ frame; sustained throughput ≥30 FPS |
| Live | Replay nguồn paced 30 FPS hoặc RTSP thật; không bỏ frame ở capture/inference/analytics để đáp ứng mục tiêu đầy đủ; E2E p95 ≤250 ms; queue không tăng theo thời gian |
| Stability | Chạy ít nhất 30 phút bằng nguồn lặp có epoch/reset rõ ràng hoặc RTSP; không crash/OOM, state/cache có giới hạn, bộ nhớ không tăng liên tục sau warmup |
| Repeatability | Candidate cuối chạy clip đầy đủ ít nhất hai lần độc lập; báo cáo cả hai và giá trị xấu hơn, không chỉ chọn run nhanh nhất |
| Preview | Cho phép preview 10–15 FPS nếu công bố rõ; không được dùng việc giảm preview để bỏ analytics frame. Kiểm tra renderer riêng ở 30 FPS nếu cần output MP4 đủ frame |

Đo riêng cold-start/setup/upload, warmup, steady-state processing, drain/flush/commit, download. Công bố cả thời gian xử lý steady-state lẫn tổng job; không đưa thời gian download vào processing FPS. Với RTSP không có timestamp capture đồng bộ, chỉ gọi metric là latency từ lúc server nhận frame, không gọi là camera-to-browser latency.

### 4.2 Chất lượng

Tạo tập kiểm chứng cố định có người gần/xa, giao cắt, che khuất, đứng yên, rìa tile và vùng quảng cáo. Tối thiểu đề xuất: 100 frame gán bbox person phân bố theo thời gian và 3 đoạn liên tục 10–20 giây gán MOT đầy đủ. Chia phần tuning và holdout; lưu manifest để mọi variant dùng cùng tập. Nếu thiếu nhãn, vẫn tối ưu và đo đối chứng nhưng trạng thái chất lượng phải là `REVIEW_PENDING`.

| Metric | Gate tương đối so với baseline được đo trên cùng tập |
|---|---|
| Person precision/recall tại IoU 0,5 | Mỗi metric giảm không quá 2 điểm phần trăm tổng thể |
| Recall người nhỏ/vùng xa | Giảm không quá 3 điểm phần trăm; định nghĩa vùng/nhóm kích thước trước benchmark |
| IDF1/HOTA | Mỗi metric giảm không quá 2 điểm trên thang 0–100 |
| ID switch | Không tăng quá 10%; nếu baseline bằng 0 thì không chấp nhận switch mới trên holdout |
| Common Path | Không xuất hiện lỗi đảo hướng, nối nhầm hai nhánh giao cắt hoặc support từ prediction trong các cảnh review |
| Jitter | p95 không cao hơn baseline quá 10%, đồng thời đối chiếu hình học thực tế; nếu baseline gần 0 dùng dung sai số đã chốt trước |
| Change delay | Không tăng quá 1 giây so với baseline ở các sự kiện đổi luồng được gán mốc thời gian |

Đây là gate chống hồi quy, không chứng minh baseline có chất lượng tuyệt đối tốt. Không dùng output tiled làm ground truth. Không suy ra recall từ số detection hoặc suy ra IDF1 từ `unique_track_ids`. Nhãn sparse Grand Central không thay thế MOT bbox ground truth đầy đủ của Shibuya.

## 5. P0 — Xác minh hiện trạng và tạo benchmark có thể tái lập

### Công việc

- Đọc `AGENTS.md` nếu có; kiểm tra working tree, commit hiện tại và khác biệt so với `f99ec01`. Giữ nguyên thay đổi không thuộc nhiệm vụ.
- Đọc YAML resolved, `summary.json`, `resource_summary.json`, `stage_metrics.csv`, provenance và manifest của run dài. Tìm theo run ID, không hardcode đường dẫn Windows của người dùng.
- Đọc các phần liên quan trong `backend/app/inference/`, `tracking/`, `video/pipeline.py`, `core/session.py`, `analytics/tracklet_aggregation.py`, `modal_common_path.py` và script benchmark/replay hiện có.
- Review MP4/contact sheet ở đầu/giữa/cuối và các vùng gần/xa/giao cắt; cập nhật inspection bằng quan sát thực tế, kèm timestamp. Không đánh dấu PASS chỉ vì file tồn tại.
- Dùng clip ngắn 5–10 giây để smoke, cùng clip 65 giây để A/B, toàn bộ 565,267 giây chỉ cho ứng viên cuối. Ghi chính xác frame range/source hash.
- Benchmark production có output I/O thực tế; bổ sung diagnostic không encode/cache-write để tách chi phí. Không dùng diagnostic làm kết quả nghiệm thu production.

### Metric cần bổ sung nếu đang thiếu

`source_frames`, `captured_frames`, `detector_source_frames`, `detector_images`, `tracker_updates`, `analytics_ingested_frames`, `rendered_frames`, `published_frames`, drops theo từng queue, scheduler skip, input/processed FPS, queue age/size, detection count, tracker count, evidence age, CPU per-core, GPU util/VRAM, RAM và bytes cache ghi ra.

Phân biệt thời gian CPU enqueue với thời gian GPU thực thi. Chỉ thêm CUDA event/synchronize trong chế độ profiling phù hợp; không đặt synchronize mỗi stage vào production để đo rồi làm mất overlap. Gắn source frame ID/timestamp qua toàn pipeline để truy vết.

### Đầu ra

- Baseline tái lập hoặc báo cáo khác biệt môi trường khiến chưa tái lập được.
- Bảng chất lượng ban đầu và inspection có trạng thái đúng.
- Một lệnh benchmark xuất report JSON/CSV; ưu tiên mở rộng script sẵn có.

## 6. P1 — Tối ưu ít thay đổi ngữ nghĩa trước

### 6.1 Detector và CPU–GPU

1. Kiểm tra 5 ảnh detector đang được gọi tuần tự, list batch hay bị tách do shape. Đo thời gian crop/resize/letterbox, H2D, forward, NMS, D2H và merge thực tế.
2. Thử FP16 trên T4 bằng profile riêng, giữ các tham số còn lại để so sánh. Kiểm tra output dtype/toạ độ và NaN/Inf; quality gate bắt buộc.
3. Nếu hiện gọi rời, thử batch 5 ảnh cùng một source frame với shape/letterbox nhất quán và metadata ánh xạ tile đầy đủ. Nếu đã batch thì không triển khai lại. Không batch nhiều timestamp trước khi có lý do đo được vì có thể tăng live latency.
4. Loại bỏ copy/convert CPU↔GPU lặp lại và khởi tạo model/tensor không cần thiết trong vòng lặp. Chỉ thử pinned memory/nonblocking transfer sau profiling và kiểm tra lifetime buffer.
5. Đánh giá cấp thêm CPU Modal theo tài nguyên thực tế, cùng với số thread PyTorch/OpenCV để tránh oversubscription. A/B CPU trước khi đổi GPU; không giả định cấp nhiều core tự giải quyết Python GIL.
6. Giữ merge source-aware: không loại hai người cùng source chỉ vì overlap. Tối ưu vùng ứng viên/NumPy và allocation dựa trên profile; bảo toàn box ở rìa tile.

### 6.2 Tracking

1. Profile tạo cost matrix, gating, assignment và quản lý track, phân nhóm theo số detection/track.
2. Vector hóa tính khoảng cách/IoU/motion penalty và tránh cấp phát lại trong hot path.
3. Spatial gating chỉ loại cặp chắc chắn không hợp lệ theo chính rule hiện có; kiểm tra riêng chuyển động nhanh, frame gap, occlusion và perspective. Không đổi bài toán assignment thành greedy để đạt tốc độ.
4. Giữ high/low-confidence association, Kalman dt theo frame nguồn, stationary grace và `coast()` semantics.
5. Nếu prune track/cache, chỉ bỏ state đã hết TTL theo hợp đồng; đo số đối tượng bị prune và kiểm tra ID switch.

### 6.3 Render, encode và artifact

1. Tách nhịp preview khỏi detection/tracking/analytics ingest; preview latest-frame queue vẫn giới hạn 1. Không lấy preview cadence làm cadence evidence.
2. Tránh render/encode cùng một frame nhiều lần cho nhiều client; dùng immutable snapshot đã có.
3. Ghi cache có buffer/batch hoặc writer bounded nếu profile cho thấy I/O đáng kể. Offline audit dùng backpressure, không âm thầm bỏ tracking record. Ghi metric và flush/join khi kết thúc.
4. Đo có/không overlay và MP4 để biết phần chi phí; báo cáo rõ profile headless, preview và archival output.

### Gate P1

Chạy các regression test liên quan và A/B 65 giây. Chỉ ghép các thay đổi có lợi đã xác minh; không chạy run dài sau mỗi chỉnh sửa nhỏ. Không hứa một mức tăng FPS trước khi đo.

## 7. P2 — Giảm spike Common Path mà giữ evidence

Mục tiêu là giảm compute p95 và tránh worker analytics/render bị chặn hàng trăm ms; không bắt đầu bằng tăng update interval lên 2 giây vì thử nghiệm đó đã thay đổi Path ID/support.

1. Replay cùng tracking cache để cô lập Common Path; tách thời gian link graph, clustering, polyline assembly, identity matching và publish.
2. Kiểm tra link-cache fingerprint, dirty set và incremental recomputation. Chỉ tính lại những cặp/hình học bị thay đổi, TTL cleanup có giới hạn.
3. Giữ ingest toàn bộ observed point. Phân biệt ingest, compute route, publish snapshot và render; không bỏ evidence để tránh tắc queue.
4. Nếu tách compute sang worker riêng, đánh giá thread có thực sự giải phóng GIL; chọn process chỉ khi lợi ích lớn hơn chi phí serialize/copy. Không thêm multiprocessing mặc định khi chưa đo.
5. Compute nhận snapshot bất biến có epoch/generation, sequence và evidence cutoff. Chỉ một computation đang chạy; có thể gộp yêu cầu recompute chờ, nhưng không bỏ point ingest. Kết quả cũ/sai generation bị loại. Worker identity update có ownership duy nhất.
6. Giữ last valid path trong lifecycle hiện có và xuất `computed_at`, `evidence_cutoff`, `path_age`; không kéo dài active/cooling vô hạn để che compute chậm.
7. Chỉ A/B cadence 0,5/1/2 giây sau tối ưu code và khi có metric change delay. Các giá trị này là biến thử nghiệm, không phải cấu hình mặc định mới.

Định nghĩa metric trước replay:

- Jitter: match cùng tuyến/cùng hướng qua các snapshot, resample trên phần arc length chung rồi đo dịch chuyển vuông góc giữa snapshot liên tiếp trong đoạn ground truth cho thấy tuyến ổn định. Báo cả p50/p95, sample count và unmatched route; không gọi chuyển dịch tuyến thật là jitter.
- Change delay: thời gian nguồn từ sự kiện đổi luồng được gán nhãn đến tuyến mới được publish ổn định theo khoảng xác nhận chốt trước. Báo trường hợp không phát hiện, không loại khỏi thống kê.
- Path switch: cùng tuyến vật lý đổi identity ngoài lifecycle hợp lệ; kiểm tra theo annotation/matching, không chỉ đếm tất cả Path ID mới.

## 8. P3 — Backend inference và giảm workload có kiểm soát

Chỉ bắt đầu sau baseline chất lượng và profiling ổn định. Ưu tiên thử backend FP16 giữ workload trước khi giảm tần suất quét.

### 8.1 TensorRT FP16 — một giả thuyết cần benchmark

- Kiểm tra hỗ trợ export của đúng model YOLO26n và phiên bản Ultralytics/TensorRT/CUDA hiện có bằng tài liệu chính thức và smoke test. Không giả định model export được hoặc nhanh hơn.
- Thêm backend qua detector adapter, giữ data contract, tile mapping, class filtering và NMS/merge semantics. Xác minh engine có NMS tích hợp để tránh NMS hai lần.
- Đóng gói version và engine build provenance, model hash, precision, shape/batch, GPU compatibility. Build engine trong môi trường tương thích; tách thời gian build khỏi inference.
- So sánh PyTorch FP16 và TensorRT FP16 với cùng batch/shape/workload. Nếu FP32→FP16 và backend đổi cùng lúc, cần ablation để biết lợi ích từng yếu tố.
- Không thêm INT8 trước khi có tập calibration đại diện và quality gate riêng.

### 8.2 Single-pass và tile cadence — profile thử nghiệm riêng

Nếu tối ưu giữ workload vẫn chưa đạt mục tiêu, thử single-pass/full-frame mỗi frame và tile bổ sung theo chu kỳ. Ví dụ full-frame mỗi frame + toàn bộ 4 tile mỗi N frame, N∈{2,3,5}; hoặc luân phiên tile có coverage deadline. Đây là biến A/B, không mặc định sẽ đạt 30 FPS.

Yêu cầu:

- Không giảm `imgsz`, threshold và cadence cùng lúc; thay từng trục trước khi ghép.
- Ghi per-region `last_searched_at`, resolution/scan type và tuổi coverage. Full-frame low-resolution không tương đương tile về khả năng thấy người nhỏ.
- Xác định coverage cho từng track theo detector pass thực sự đã quét vùng của nó. Nếu full-frame đã tìm vùng đó thì vẫn là searched, dù tile không chạy. Không dùng `NOT_SEARCHED_BY_POLICY` để che detection miss thật; nếu cần phân biệt coverage theo scale, phải thiết kế/test contract mở rộng rõ ràng.
- Mọi frame không detector thì chỉ coast, prediction không vote. Profile này phải mang nhãn sampled analysis và không được nhận gate full-frame detection 30 FPS.
- Full-frame detection 30 FPS với tile cadence giảm có thể đạt gate throughput nhưng chỉ được promote khi quality gate vùng xa/nhỏ đạt; báo riêng detector images/source frame.
- Motion ROI tiếp tục là tùy chọn thử nghiệm cho cảnh phù hợp, không lặp lại kỳ vọng tiết kiệm trên cảnh dense khi chưa đổi policy có bằng chứng.

Nếu vẫn không đạt, đo candidate tốt nhất trên GPU mạnh hơn với cùng cấu hình sau khi xử lý CPU bottleneck. Báo FPS, chất lượng và chi phí thực đo từ run/billing; không tự bịa giá hoặc cam kết GPU mới giải quyết được.

## 9. Ma trận benchmark và cách tiết kiệm thời gian

| ID | Thay đổi chính | Phạm vi ban đầu |
|---|---|---|
| B0 | Tiled FP32 hiện tại | Artifact dài + baseline 65 giây |
| B1 | Tiled FP16, cùng workload | Smoke → 65 giây |
| B2 | Batch/copy/merge optimization | Smoke → 65 giây |
| B3 | Tracker optimization | Cache detection nếu có → end-to-end 65 giây |
| B4 | Common Path incremental/worker | Cùng tracking cache → end-to-end |
| B5 | TensorRT FP16, cùng workload | Export smoke → 65 giây |
| B6 | Single-pass FP32/FP16 | 65 giây + quality holdout |
| B7 | Tile cadence từng N | 65 giây + quality holdout |
| F | Tổ hợp các thay đổi thắng | Hai run đầy đủ + 30 phút stability |

Không chạy toàn bộ tổ hợp tích Descartes. Chọn ứng viên theo từng bước, lưu cả thất bại và lý do dừng. Warmup cùng chính sách, cố định phần cứng/resource allocation và kiểm tra không có job khác gây tranh chấp. Run detector dùng inference thực; replay chỉ đánh giá phần sau detector/tracker tùy loại cache. Sau profiling có instrumentation nặng, benchmark lại khi tắt profiler.

## 10. File/code dự kiến thay đổi

Ưu tiên mở rộng module hiện có thay vì dựng pipeline song song khó bảo trì.

| Vị trí | Mục đích |
|---|---|
| `backend/app/inference/` | Precision/backend/batching, profiling, bảo toàn detection contract |
| `backend/app/tracking/` | Vectorized association và regression tests |
| `backend/app/video/pipeline.py` | Accounting, timestamp, bounded queue và scheduling |
| `backend/app/core/session.py` | Ownership, generation guard, tách compute/render nếu cần |
| `backend/app/analytics/tracklet_aggregation.py` | Incremental/cache và giảm compute spike |
| `backend/app/metrics/` | Stage/queue/coverage và end-to-end metrics |
| `configs/` | Profile thử nghiệm có `extends`, mặc định baseline được giữ |
| `scripts/` | Benchmark/quality/replay runner tái lập |
| `modal_common_path.py` | Khai báo resource/backend và provenance nếu cần |
| `docs/realtime_optimization_report.md` | Kết quả, quyết định và giới hạn thực tế |

Tên config mới đề xuất: `shibuya-realtime-candidate.yaml`. Không tự thêm các key trong YAML mà chưa có schema validation và code tiêu thụ. Không cần sửa UI trừ khi để phân biệt input FPS, processed FPS, preview FPS, dropped frames và path age.

## 11. Kiểm thử và điều kiện promote

Chạy test hiện có trước/sau thay đổi liên quan. Mốc 147 test chỉ là lịch sử, không yêu cầu giữ đúng số lượng đó. Bổ sung test có ý nghĩa cho các rủi ro mới:

- Batch tile mapping/letterbox, box sát biên và duplicate source-aware.
- Precision/backend contract, kết quả hợp lệ và không sai toạ độ.
- Gating không làm mất association hợp lệ; ID switch ở crossing/occlusion/frame gap.
- Prediction không tạo support; phân biệt miss với not searched.
- Kết quả compute cũ không ghi đè generation/epoch mới; worker kết thúc và flush sạch.
- Top K không reset Path ID; ingest đủ frame dù compute/publish thưa hơn.
- Fingerprint thay đổi khi backend/precision/scheduler đổi; cache incompatible bị từ chối.

Lệnh kiểm tra nền tảng từ hiện trạng:

```powershell
python -m pytest -q
python -m compileall -q backend scripts modal_common_path.py
git diff --check
Set-Location frontend
npm run build
```

Xác minh CLI trước khi chạy benchmark. Lệnh tham chiếu hiện có:

```powershell
$runId = "shibuya-realtime-$(Get-Date -Format yyyyMMdd-HHmmss)"
modal run --quiet --timestamps modal_common_path.py `
  --input data/videos/data-shibuya-10m.mp4 `
  --engine tracklet_aggregation `
  --mode offline_fast `
  --config configs/shibuya-realtime-candidate.yaml `
  --run-id $runId `
  --cache-policy reuse
```

Quan trọng: `--cache-policy reuse` chỉ được dùng cho benchmark end-to-end nếu code xác nhận nó không tái sử dụng tracking/detection để bỏ qua model. Nếu có thể tái sử dụng kết quả inference, chọn chế độ cold/recompute thực sự mà CLI hỗ trợ hoặc bổ sung chế độ rõ ràng; ghi cache hit count và detector call count vào report. Không dùng lệnh tham chiếu này một cách mù quáng.

Giữ workflow GPU commit Volume rồi kết thúc, download bằng lệnh riêng. Trước run tốn tài nguyên, kiểm tra timeout/quota/ngân sách đã được chủ dự án cho phép; không tự tăng GPU hoặc thực hiện hàng loạt run dài khi chưa có phạm vi chi phí rõ ràng. Có thể hoàn tất code, replay local, test và kế hoạch run trước khi cần xác nhận chi phí.

Chỉ đổi profile production khi cả hiệu năng, chất lượng và stability đạt. Nếu thiếu nhãn hoặc visual sign-off, giữ candidate ở experimental và trạng thái `REVIEW_PENDING`. Rollback bằng profile baseline và commit đã ghi; không trộn cache của hai profile.

## 12. Deliverable bắt buộc từ agent

1. Code và config đã triển khai, commit/patch rõ theo từng nhóm thay đổi.
2. Báo cáo `docs/realtime_optimization_report.md`, gồm baseline, giả thuyết, ablation, kết quả thắng/thua và cấu hình khuyến nghị.
3. Artifact benchmark có config resolved, commit, input/model hash, backend/version, resource allocation, metrics theo stage, queue/drops và quality results.
4. Inspection video có timestamp và trạng thái review trung thực; metric thiếu phải là `NOT_MEASURED`/`REVIEW_PENDING`, không điền 0 hoặc PASS.
5. Lệnh chạy baseline/candidate/replay/download chính xác theo CLI đã kiểm tra và cách rollback.
6. Kết luận theo một trong các trạng thái:
   - `PASS_FULL_30FPS`: đạt xử lý đầy đủ ≥30 FPS cùng quality/live/stability gates.
   - `PASS_SAMPLED_LIVE_ONLY`: chỉ đáp ứng live có sampling/drop; ghi detector FPS và tỷ lệ coverage thực tế, không gọi là full 30 FPS.
   - `PERFORMANCE_PASS_QUALITY_PENDING`: đạt tốc độ nhưng chất lượng chưa đủ bằng chứng; không promote.
   - `NOT_MET`: chưa đạt, nêu bottleneck còn lại, FPS thực đo và thử nghiệm kế tiếp đáng làm nhất.

Mẫu bảng tổng kết bắt buộc:

| Variant | GPU/CPU | Detector images/frame | Process FPS | E2E p95 | Capture/analytics drops | Recall xa | IDF1/HOTA | Jitter/change delay | Kết luận |
|---|---|---:|---:|---:|---|---|---|---|---|
| Baseline | Điền từ run | 5 nếu được xác nhận | Đo lại | Đo | Đo | Chưa đo nếu thiếu nhãn | Chưa đo nếu thiếu nhãn | Đo/review | Baseline |
| Candidate | Điền từ run | Đo | Đo | Đo | Đo | Đo | Đo | Đo | Theo gate |

## 13. Thứ tự hành động ngay

1. Xác minh code/artifact và thiết lập baseline, review video.
2. Profile detector batch/CPU–GPU và tracking; triển khai FP16 cùng các tối ưu bảo toàn ngữ nghĩa có bằng chứng.
3. Tách preview/I/O bottleneck nếu đã đo thấy; tối ưu Common Path bằng replay cùng cache.
4. Khi baseline chất lượng đủ, thử TensorRT giữ workload; sau đó mới đánh giá single-pass/tile cadence bằng quality gate.
5. Chọn một candidate tốt nhất, chạy full/stability, báo cáo và promote hoặc rollback theo gate.

Nguyên tắc cuối: mục tiêu là phân tích đúng và kịp thời. Mọi mức tăng FPS phải đi kèm thông tin frame coverage, chất lượng và cấu hình thực sự đã chạy.
