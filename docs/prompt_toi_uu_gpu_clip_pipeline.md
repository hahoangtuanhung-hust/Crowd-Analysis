# Prompt triển khai tối ưu gpu_clip và pipeline Crowd Analysis

Ngày soạn: 30/09/2026. Ngôn ngữ báo cáo: tiếng Việt.

> Giao toàn bộ tài liệu này cho agent có quyền đọc và sửa repository. Đây là nhiệm vụ triển khai, kiểm thử và benchmark; không chỉ trả lời bằng lời khuyên. Các chỉ số đầu vào do người dùng/tài liệu cung cấp, chưa được người soạn đối chiếu với code và artifact thực tế.

## 1. Vai trò và mục tiêu

Bạn là kỹ sư tối ưu computer vision/PyTorch phụ trách Crowd Analysis. Hãy xác định đường chạy thật của `gpu_clip`, tìm stage khiến GPU phải chờ, thực hiện thay đổi có kiểm chứng và bàn giao code cùng báo cáo trước/sau.

Ưu tiên theo thứ tự:

1. Giữ đúng kết quả detection, tracking và Common Path; không mất hoặc đảo thứ tự evidence.
2. Tăng throughput end-to-end offline, giảm latency live, hướng tới phân tích đầy đủ nguồn 30 FPS trên T4.
3. Giảm chi phí cho cùng một phút video đã xử lý với cùng chất lượng.

GPU utilization và VRAM là metric chẩn đoán, không phải mục tiêu độc lập. Không cố làm GPU đạt 80–100% hoặc dùng hết VRAM nếu không cải thiện kết quả trên. Không cam kết đạt 30 FPS trước khi đo.

Hãy tự thực hiện các lựa chọn kỹ thuật thông thường và hoàn tất phần code/test có thể làm. Nếu thiếu GPU, credential hoặc dữ liệu, nêu đúng phần bị chặn, cung cấp lệnh chạy đã kiểm tra và tiếp tục các phần còn lại; không bịa kết quả.

## 2. Bối cảnh dự án cần giữ

Tài liệu `tong_quan_hien_trang_du_an.md` đối chiếu `main@f99ec01`, ngày 28/09/2026, mô tả:

- YOLO26n qua Ultralytics/PyTorch; ByteTrack có hybrid association.
- Profile Shibuya: một full frame + bốn tile = năm ảnh detector cho mỗi frame nguồn, FP32.
- Common Path production: `TrackletAggregationEngine`, không dùng grid để quyết định tuyến.
- Pipeline web/local đã có capture worker và inference/tracking worker riêng, capture queue mặc định 4; analytics queue riêng mặc định 1.000 item. Không mặc định đường batch `gpu_clip` dùng đúng kiến trúc đó.
- Offline dùng backpressure; live dùng `drop_oldest`. Các queue giới hạn không chứng minh hệ thống đủ tốc độ.
- Motion ROI và auxiliary analytics đang tắt trong production.
- GPU job ghi artifact vào Modal Volume, commit rồi kết thúc; download là bước riêng.

Run lịch sử `shibuya-10m-final-20260926-091731`: 16.959 frame/565,267 giây; wall time 4.372,237 giây; 3,88 FPS; inference p50 khoảng 147 ms, tracking 37,8 ms, render 15,6 ms, encode 6,4 ms; GPU trung bình khoảng 31%; CPU process khoảng 100%; RAM đỉnh khoảng 5,5 GB.

Quan sát mới do người dùng mô tả cho `gpu_clip`: GPU khoảng 20%, VRAM khoảng 512 MB, CPU Used khoảng một core so với request khoảng năm core, Pending Calls = 0, RAM khoảng 1 GB, network thấp. Chưa có run ID/cửa sổ đo kèm theo. Không trộn các số này với run lịch sử hoặc coi là cùng cấu hình.

### Những giả thuyết phải kiểm chứng

| Nhận định | Cách kiểm chứng |
|---|---|
| Decode/preprocess làm GPU chờ | Trace stage, queue empty time, CPU profile, so sánh input đã decode trên đoạn nhỏ |
| Batch đang bằng 1 | Đếm model invocation và ghi tensor shape/batch thực tế, không suy từ default config |
| Warning/log làm chậm | Lấy warning mẫu, số lần/giây, thời gian format/write; A/B logging có kiểm soát |
| Năm CPU đang cấp dư | Đo theo process/thread, quota/affinity và workload đã tối ưu; A/B request 5/2/1 |
| Không có nghẽn vì Pending Calls = 0 | Xác minh metric cấp Modal; đo riêng queue và thời gian chờ bên trong từng call |
| GPU còn 80% công suất sử dụng được | Không thể suy trực tiếp. Kiểm tra định nghĩa metric, thời điểm lấy mẫu và timeline kernel |

Theo định nghĩa NVIDIA [S1], nếu metric là NVML/nvidia-smi GPU utilization, nó phản ánh tỷ lệ thời gian có kernel chạy trong cửa sổ lấy mẫu, không trực tiếp đo phần trăm FLOPS đã khai thác. VRAM thấp không chứng minh thiếu batching. Pending Calls = 0 không đủ kết luận một call đang chạy không có backlog hoặc chờ I/O nội bộ. RAM/network thấp không loại trừ latency đọc/ghi tuần tự.

## 3. Ràng buộc bắt buộc

- Giữ `stream_epoch`, camera/session, source timestamp và frame ID. Không dùng thứ tự hoàn thành inference làm thứ tự tracker update.
- Common Path chỉ nhận điểm `confirmed` và `observed`; prediction/coast không tạo support.
- Giữ `MEASURED`, `SEARCHED_NOT_FOUND`, `NOT_SEARCHED_BY_POLICY`; không biến frame/vùng chưa quét thành miss thật.
- Giữ source-aware tile merge, người nhỏ/xa, box sát biên, hybrid association, Kalman dt và stationary grace.
- Giữ Path ID/màu/lifecycle/route memory và Top K 1–5 không reset state. Support theo temporary Track ID, không coi là số người duy nhất.
- Không tăng queue vô hạn, nạp cả video vào RAM/VRAM, bỏ frame âm thầm, giảm resolution/tile/threshold/cadence để làm số FPS tốt hơn.
- Không chạy nhiều bản model hoặc nhiều tracker cùng sửa state của một camera. Tăng Modal concurrent calls không phải tối ưu nội bộ một video.
- Mỗi thay đổi có profile/flag riêng và rollback; baseline không bị ghi đè. Thay backend/precision/preprocess/scheduler phải cập nhật cache fingerprint/provenance phù hợp.

## 4. Giai đoạn A — Audit và profiling trước khi sửa kiến trúc

### A1. Xác định đường chạy

Đọc `AGENTS.md` nếu có, kiểm tra branch/commit/working tree, giữ nguyên thay đổi ngoài nhiệm vụ. Dùng `rg` tìm `gpu_clip` và trace từ Modal entrypoint tới script/pipeline thực sự chạy. Đọc:

- `modal_common_path.py`, `scripts/common_path_clip.py` và các entrypoint thực tế tìm được;
- detector trong `backend/app/inference/`, tracker trong `backend/app/tracking/`;
- `backend/app/video/pipeline.py`, `backend/app/core/session.py`;
- `backend/app/analytics/tracklet_aggregation.py`, metric modules;
- YAML resolved, artifact và báo cáo benchmark liên quan.

Ghi sơ đồ ownership của thread/process/queue bằng mô tả ngắn. Xác nhận chỗ đã có concurrency, chỗ còn tuần tự, lock nào bao trùm toàn pipeline, có `sleep`/PTS pacing trong `offline_fast` hay không. Không thêm worker capture trùng với worker đã tồn tại.

### A2. Tạo baseline đúng phạm vi

- Dùng cùng video/model/config/hardware và cùng khoảng frame cho mỗi A/B. Ghi version Python, torch, Ultralytics, OpenCV, CUDA/driver, Modal SDK.
- Smoke 5–10 giây; clip đại diện khoảng 65 giây để chọn ứng viên; run dài chỉ dành cho ứng viên cuối.
- Tách startup/download/model load/warmup, xử lý, drain/encode flush/Volume commit và download artifact. Báo cả tổng thời gian job và processing time.
- Benchmark throughput dùng `offline_fast`, không PTS pacing; live chạy riêng có pacing. Kiểm tra inference interval và cache policy để model thực sự chạy trên mọi frame yêu cầu.
- Đo khi có output production. Các run tắt render/cache/log chỉ dùng phân rã chi phí, không thay số end-to-end.

### A3. Instrumentation cần có

Mở rộng metric sẵn có, tránh tạo bộ đo trùng lặp. Ghi p50/p95, số lần gọi và tổng thời gian cho: read/decode; queue wait; crop/resize/letterbox; tensor conversion; H2D; GPU forward; GPU/CPU postprocess thực sự tồn tại; D2H; tile merge; tracking; analytics/Common Path; render; encode; cache write; log emission.

Không giả định YOLO26n có NMS hoặc có cùng postprocess với model khác: xác minh runtime/model export đang dùng. Không cộng p50/p95 của các stage lồng nhau hoặc song song để suy E2E.

Ghi thêm:

- Source/captured/detected/tracked/analytics-ingested/rendered/encoded frames; detector images và invocation count.
- Queue size/bytes, high-water mark, empty/full wait, frame age, drops theo lý do, in-flight batch, actual batch size và tensor shape.
- CPU theo thread/process, CPU allocation/quota/affinity và thread count; GPU util cùng sampling interval, VRAM allocated/reserved/device-used; RAM, disk I/O; log lines/bytes mỗi giây.
- Timestamp monotonic cho latency thực thi và source PTS cho logic chuyển động. Với RTSP thiếu capture clock đồng bộ, gọi metric là latency từ server nhận frame.

Theo hướng dẫn PyTorch [S3], CUDA chạy bất đồng bộ: dùng CUDA events hoặc profiler CPU/CUDA trên cửa sổ ngắn. Phân biệt thời gian CPU enqueue với GPU execution; kiểm tra cả implicit sync do `.cpu()`, `.numpy()`, `.item()` và wrapper. Không để `CUDA_LAUNCH_BLOCKING=1` hoặc synchronize mọi stage trong production. Benchmark lại khi tắt profiler nặng.

Đầu ra giai đoạn A: bảng top bottleneck có bằng chứng và quyết định tối ưu nào cần làm trước. Nếu decode không đáng kể, không dành phần lớn công sức viết lại decoder.

## 5. Giai đoạn B — Batch inference đúng semantics

### B1. Ưu tiên gom các ảnh cùng một frame

Xác minh năm ảnh detector hiện đang vào năm model calls hay đã là batch. Nếu còn gọi rời, thử batch full-frame + tile trong một source frame. Nếu đã batch, tiếp tục tìm shape/copy/postprocess overhead.

Giữ metadata cho từng ảnh: `(camera_id, stream_epoch, frame_id, source_pts, tile_id, crop_rect, resize_scale, padding, original_shape)`.

Ghi actual input shape: theo tài liệu Ultralytics [S5], runtime có thể đổi từ rectangular padding sang padding đầy đủ khi các ảnh khác shape. Điều này có thể tăng pixel workload và thay đổi output. Chuẩn hóa shape hoặc grouping chỉ sau khi so sánh geometry/quality; không coi đổi letterbox là hoàn toàn trung tính.

### B2. Microbatch nhiều frame cho offline

Chỉ thực hiện nếu B1/profiling cho thấy có lợi. Tách hai khái niệm cấu hình:

- `source_batch_size`: số frame nguồn;
- `max_detector_images_per_batch`: số ảnh detector tối đa sau khi tạo tile.

Tên key là đề xuất; kiểm tra schema hiện có và bổ sung validation trước khi dùng.

Thử source batch 1 → 2 → 4; chỉ thử 8/16 nếu 4 còn cải thiện, đủ memory headroom và downstream không bị nghẽn. Với 5 ảnh/frame, 4 frame tạo 20 ảnh detector, 16 frame tạo 80 ảnh; không nhầm với batch 4/16 ảnh.

Thiết kế yêu cầu:

1. Một owner đọc `VideoCapture`; một owner model/GPU; kết quả detection được nhóm lại theo frame.
2. Hoàn thành/merge mọi tile của frame trước khi trả detection frame đó. Giữ source-aware merge và inverse transform chính xác.
3. Tracker và analytics xử lý tuần tự theo source frame/PTS. Có thể dùng GPU suy luận frame tiếp trong khi CPU xử lý batch trước nếu đo thấy lợi, nhưng in-flight phải có giới hạn.
4. Flush batch cuối chưa đủ ở EOF. Stop/error phải giải phóng tài nguyên; không lặp tracker update khi retry/split batch.
5. Nếu OOM, fail rõ hoặc chia nhỏ batch có giới hạn trước khi commit tracking state; ghi fallback vào report. Không tự giảm `imgsz` hoặc bỏ frame.
6. Không batch qua epoch/calibration generation/restart boundary; tách/drain hoặc hủy phần cũ theo lifecycle và metric rõ ràng.
7. Common Path dùng source timestamp cho evidence window/cadence/lifecycle theo hợp đồng; không gọi một lần cho cả batch rồi làm mất các cập nhật trước đó.

Motion ROI có quyết định dựa trên track của frame trước. Khi bật tính năng này, batch lookahead qua nhiều frame có thể làm scheduler dùng state cũ. Không giả định tương đương: mặc định chỉ bật cross-frame batch cho policy không phụ thuộc tracker tương lai (production ROI đang tắt), hoặc fallback source batch 1 khi policy có feedback. Không cache một kết quả scheduler sai semantics.

### B3. Live có latency budget riêng

Live ưu tiên batch các tile cùng frame. Cross-frame batching chỉ bật khi có deadline (`max_batch_wait_ms`) và chứng minh lợi ích latency/throughput. Không đợi batch đầy vô thời hạn.

Với nguồn 30 FPS và bắt đầu từ queue rỗng, đợi đủ bốn frame tạo khoảng 100 ms chờ cho frame đầu; 16 frame khoảng 500 ms, chưa tính inference. Vì vậy không áp batch offline 16 cho live mặc định. Deadline hết thì chạy batch chưa đầy; queue stale bị xử lý theo policy có metric.

Không dùng frame tương lai để tạo Common Path cho timestamp quá khứ. Model detection độc lập có thể batch offline, nhưng tracker/analytics phải tiến thời gian đúng thứ tự.

## 6. Giai đoạn C — Overlap CPU/GPU và giảm hot path

### C1. Decode và preprocessing

- Nếu `gpu_clip` bỏ qua capture worker hiện có, ưu tiên tái sử dụng abstraction đó hoặc bổ sung bounded prefetch trong đường batch. Không cho nhiều thread cùng đọc một `VideoCapture`.
- Nếu CPU preprocessing chiếm tỷ trọng lớn, A/B vector hóa/tái sử dụng buffer, worker chuẩn bị ảnh hoặc GPU preprocess; giữ BGR/RGB, normalization, interpolation, resize/letterbox và inverse coordinates đúng.
- Thread phù hợp khi phần việc thực sự có thể chạy song song hoặc chờ I/O. Nếu hot path Python bị GIL, cân nhắc process/vectorization sau profiling; đo copy/serialization/shared-memory overhead. Không chuyển toàn bộ video qua process queue không giới hạn.
- GPU decode/NVDEC chỉ là thử nghiệm tiếp theo nếu decode được chứng minh là bottleneck và environment hỗ trợ; không mặc định OpenCV headless đã có CUDA decode.

### C2. H2D và model runtime

- Xác nhận `eval`/inference mode và cơ chế backend hiện tại; tránh thêm wrapper trùng lặp.
- A/B FP16 bằng profile riêng, không trộn thay precision với batching ngay ở thí nghiệm đầu.
- Nếu H2D/copy đáng kể, thử buffer pinned tái sử dụng và nonblocking copy. Theo hướng dẫn PyTorch [S4], `.pin_memory()` ngay trên hot path cũng có chi phí; không coi là tăng tốc tự động.
- Chỉ thêm copy stream/double buffer khi profiling chứng minh overlap hữu ích; dùng event/stream dependency, giữ lifetime buffer và không tái sử dụng pinned buffer trước khi copy xong.
- Giảm round-trip GPU→CPU→GPU, chuyển kết quả cần thiết một lần thay vì từng box. Không `empty_cache()` mỗi frame để cố làm VRAM thấp.
- TensorRT là giai đoạn sau nếu profile chứng minh inference còn chi phối; không đổi backend, scheduler, precision và threshold cùng một lần.

### C3. Postprocess, tracking và analytics

- Vector hóa merge/gating/cost matrix nếu đây là hot path; giữ kết quả association và hạn chế candidate theo rule hiện có. Không thay assignment bằng greedy chỉ để tăng FPS.
- Không bỏ qua tracking p50 khoảng 37,8 ms ở run lịch sử: ngay cả model nhanh hơn, stage tuần tự này có thể giới hạn throughput. Đo lại trên run hiện tại trước khi kết luận.
- Replay cùng tracking cache để tối ưu Common Path compute/cache/incremental update. Cache replay không dùng để công bố end-to-end FPS.
- Nếu tách Common Path worker, giữ immutable snapshot, epoch/generation guard, single-writer identity và bounded pending compute. Có thể gộp yêu cầu recompute, không được bỏ evidence ingest.
- Chưa đổi cadence lên hai giây nếu chưa đo tác động Path ID/support, jitter và change delay.

### C4. Logging, encode và ghi artifact

- Xác định chính xác warning lặp: file/hàm/thư viện, warning class, stack một lần và tần suất. Sửa API lỗi thời nếu tương thích; không tự suy đoán là cảnh báo của AMP hay thư viện cụ thể.
- Với warning đã hiểu và không thể sửa ngay, lọc có scope hẹp/once kèm lý do. Không tắt toàn bộ warning/error.
- Tổng hợp progress mỗi 1–5 giây, không format/in stdout mỗi frame. Giữ counter và lỗi đầu tiên; nếu async logging dùng queue bounded, chỉ được giảm log debug lặp, không làm mất fatal error.
- Ghi cache theo buffer/chunk hoặc writer riêng nếu đo có lợi. Audit records không bị bỏ im lặng; offline dùng backpressure, cuối job drain/flush/join rồi mới commit Volume.
- Tách preview 10–15 FPS khỏi detection/tracking/analytics đủ frame. MP4 archival muốn đủ frame vẫn phải encode đủ và được đo ở profile riêng.
- Không bắt GPU chờ client download; duy trì workflow lưu Volume và tải riêng của dự án.

## 7. Giai đoạn D — Chọn CPU Modal bằng chi phí toàn job

Không đổi `cpu=5` thành `cpu=1` ngay đầu nhiệm vụ. Trước hết xác minh actual request/limit, scope metric và thread config của container.

Theo tài liệu Modal [S2], CPU/memory được tính theo phần lớn hơn giữa request và actual usage; `cpu` biểu diễn physical cores, và request còn ảnh hưởng các biến thread của thư viện. Vì vậy giảm request có thể giảm phí CPU nhưng cũng có thể làm GPU phải chờ lâu hơn.

Sau khi có pipeline candidate:

1. So sánh request 5 và 2 trên cùng clip/config; thử 1 nếu kết quả cho thấy đáng làm. Ghi CPU thực dùng, không coi request là hard cap hoặc CPU usage cố định.
2. Ghi/kiểm soát `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, PyTorch intra/inter-op và OpenCV thread count để phân biệt hiệu ứng request với hiệu ứng threading. Tránh oversubscription.
3. Giữ GPU T4, clip, model, batch, logging/output và phiên bản cố định. Chạy lại ứng viên thắng để loại nhiễu.
4. So sánh wall time, FPS, E2E, queue age, GPU wait và chi phí toàn run. Ưu tiên actual billing khi có; nếu chỉ ước tính, nêu đơn giá/nguồn/ngày và cách tính, không bịa giá.

Các metric chi phí chuẩn hóa:

```text
cost_per_video_minute = total_run_cost / (source_duration_seconds / 60)
gpu_seconds_per_video_minute = gpu_active_container_seconds / (source_duration_seconds / 60)
```

Chỉ chọn CPU thấp hơn khi tiết kiệm chi phí toàn run mà vẫn đáp ứng mục tiêu hiệu năng/chất lượng. Báo riêng chi phí CPU và GPU để tránh tiết kiệm CPU nhưng tăng chi phí GPU vì job kéo dài. Không tăng đồng thời số container để làm utilization nhìn cao hơn.

## 8. Benchmark tối thiểu và chất lượng

### 8.1 Ma trận theo từng bước

| Variant | Thay đổi | Phạm vi |
|---|---|---|
| B0 | Baseline hiện tại | Cùng clip 65 giây, inference thật |
| B1 | Sửa warning/progress/I/O đã xác định | Bật output production, so với B0 |
| B2 | Batch tile cùng frame | Cùng precision và preprocess đã kiểm tra |
| B3 | Offline source batch 2/4 | Chỉ mở rộng 8/16 nếu có lợi |
| B4 | Prefetch/CPU-GPU overlap nếu cần | Giữ batch của ứng viên tốt nhất |
| B5 | FP16 | So cùng workload, quality gate riêng |
| B6 | Request CPU 5/2/1 có chọn lọc | Trên một candidate cố định |
| F | Candidate cuối | Hai lần full clip + kiểm tra live/stability |

Không chạy tất cả tổ hợp. Tái sử dụng benchmark/module đang có. Thu đủ frame count, batch count và queue accounting; đối chiếu số frame ra vào sau drain. Warmup xử lý giống nhau, không loại tùy ý các frame chậm khỏi report. Profiler chỉ chạy trên đoạn ngắn rồi tắt khi đo cuối.

### 8.2 Kiểm tra kết quả

Đối với thay đổi concurrency/batch giữ model/precision:

- Đối chiếu bbox/class/confidence trên cùng frame; đặt dung sai tọa độ/float trước khi đo, báo cả unmatched box. Không yêu cầu bitwise equality giữa mọi kernel nhưng cũng không bỏ qua khác biệt detection count.
- Kiểm tra riêng rìa tile, người xa, overlap, crossing, occlusion, đứng yên và batch cuối chưa đầy.
- Chạy tracker/analytics trên cùng detection cache trước và sau refactor để xác minh thứ tự, timestamp, support và lifecycle không đổi do batching.
- Kiểm tra không duplicate/missing frame, không update state hai lần, không kết quả cũ ghi đè epoch/generation mới.

Đối với FP16 hoặc đổi preprocess/backend: thêm precision/recall và IDF1/HOTA nếu có nhãn phù hợp. Khi chưa có ground truth, review có timestamp và regression chỉ là bằng chứng hạn chế; ghi `QUALITY_PENDING`, không coi baseline detector là ground truth.

Ngưỡng đề xuất cho candidate thay đổi số học: precision/recall và IDF1/HOTA giảm không quá hai điểm trên thang 0–100; recall người nhỏ giảm không quá ba điểm. Các ngưỡng phải được chốt trước benchmark; không nới sau khi thấy kết quả. Đánh giá Common Path về hướng, nhánh giao cắt, jitter, identity và change delay. `unique_track_ids` không thay cho recall/IDF1.

### 8.3 Định nghĩa thành công

- Tối ưu hiệu năng: improvement vượt dao động giữa các run, không hồi quy chất lượng/state, không tăng backlog/drop; công bố phần trăm thực đo, không đặt PASS theo GPU utilization.
- Full-frame 30 FPS: tất cả frame được detector xử lý, tracker update và analytics ingest, không dùng cache inference, không bỏ frame, sustained end-to-end throughput ≥30 FPS. 33,33 ms là khoảng cách đầu ra trung bình; không đồng nghĩa mọi frame E2E ≤33,33 ms trong pipeline song song.
- Live: báo riêng input/detector/analytics/preview FPS, E2E p50/p95 và drops. Mục tiêu đề xuất p95 từ server nhận tới output sẵn sàng ≤250 ms, queue không tăng; đo network/browser riêng nếu có acknowledgement.
- Stability: full clip gần 10 phút không leak/deadlock; candidate đủ điều kiện chạy soak 30 phút. Modal live hiện giới hạn 200 giây: soak có thể dùng harness/local tương đương hoặc entrypoint thử nghiệm có giới hạn riêng; không âm thầm nới giới hạn production và gọi nhiều session ngắn là một soak liên tục.
- Nếu chỉ giữ live gần hiện tại bằng drop/sampling, gọi đúng `SAMPLED_LIVE`, không công bố full-frame 30 FPS.

## 9. Kiểm thử và triển khai an toàn

Bổ sung test cho rủi ro mới, tránh viết test chỉ lặp lại implementation:

1. Mapping tile/batch → source frame chính xác, flush partial batch EOF.
2. Output hoàn thành lệch thứ tự nhưng tracker vẫn tiến đúng frame/PTS.
3. Epoch/generation đổi, stop/cancel/error/worker crash, bounded queues không deadlock.
4. OOM fallback không lặp evidence hoặc mất frame; lỗi worker truyền về caller.
5. Single-writer state, prediction không vote, Top K không reset Path ID.
6. Motion ROI có feedback không dùng batch lookahead sai state.
7. Logging filter giữ cảnh báo/lỗi ngoài scope; cache writer flush đầy đủ trước commit.

Chạy suite liên quan, sau đó kiểm tra cuối:

```powershell
python -m pytest -q
python -m compileall -q backend scripts modal_common_path.py
git diff --check
```

Chạy frontend build nếu UI/contracts thay đổi. Mốc 147 test pass là số liệu lịch sử; không giả định code hiện tại có đúng 147 test. Không dùng test pass thay visual review.

Triển khai config candidate riêng có `extends` và schema validation; giữ rollback về baseline. Không ghi đè artifact cũ hoặc thay mặc định production khi quality chưa đủ bằng chứng. Các job Modal phải nằm trong quyền/ngân sách đã có; nếu thiếu phạm vi cho run tốn phí, hoàn tất patch, test và cấu hình/lệnh run cụ thể trước khi báo blocker.

## 10. Deliverable agent phải bàn giao

- Code/patch hoàn chỉnh cho những bottleneck được xác nhận; nêu module đổi và lý do.
- Profile candidate offline/live, batch/queue/thread limits và rollback; không để key cấu hình không được code sử dụng.
- Script benchmark có thể tái lập hoặc mở rộng script hiện có, CLI thực sự chạy được. Không bịa cờ CLI; kiểm tra `--help` và smoke.
- Báo cáo đề xuất `docs/gpu_clip_pipeline_optimization_report.md` và metrics JSON/CSV, profiler trace đoạn ngắn, config resolved, input/model hash, commit/version/resource provenance.
- Inspection có timestamp; các metric thiếu ghi `NOT_MEASURED`, chất lượng thiếu bằng chứng ghi `QUALITY_PENDING`.
- Quyết định CPU request dựa trên cost/video-minute và hiệu năng, không chỉ biểu đồ CPU trung bình.

Mẫu bảng báo cáo:

| Variant | CPU request/actual | Source batch / detector batch | FPS end-to-end | E2E p95 | GPU util | Queue wait/drops | VRAM/RAM peak | Cost/video-minute | Quality |
|---|---|---|---:|---:|---:|---|---|---:|---|
| Baseline | Đo | Đo | Đo | Đo | Đo | Đo | Đo | Đo/ước tính có nguồn | Trạng thái |
| Candidate | Đo | Đo | Đo | Đo | Đo | Đo | Đo | Đo/ước tính có nguồn | Trạng thái |

Kết luận phải trả lời: bottleneck thật ở đâu; batching có giúp không; GPU đã bớt chờ công đoạn nào; CPU tối ưu là bao nhiêu; tốc độ/chi phí thay đổi ra sao; đã đạt full 30 FPS hay chỉ sampled live; bằng chứng chất lượng nào còn thiếu.

Hãy bắt đầu từ audit `gpu_clip` và baseline, triển khai thay đổi nhỏ có tác động lớn nhất đã được đo, kiểm thử, rồi tiếp tục ứng viên tiếp theo. Không dừng sau khi chỉ đưa ra kế hoạch, và không đánh dấu thành công khi chưa có artifact chứng minh.

## 11. Tài liệu kỹ thuật để đối chiếu

Các nguồn chính thức được tra cứu khi soạn. Agent cần đối chiếu phiên bản đã cài; URL `stable`/trang hiện hành có thể khác dependency của repository. Các đề xuất worker, batch limits và ngưỡng nghiệm thu ở trên là thiết kế cho dự án, không phải cam kết hiệu năng của nhà cung cấp.

- **[S1] NVIDIA — nvidia-smi, Utilization:** định nghĩa GPU utilization và cửa sổ lấy mẫu. https://docs.nvidia.com/deploy/nvidia-smi/
- **[S2] Modal — Configuring CPU, memory, and disk:** CPU request, thread environment và billing CPU/memory. https://modal.com/docs/guide/resources
- **[S3] PyTorch — CUDA semantics:** bất đồng bộ, đo GPU time, stream và memory. https://docs.pytorch.org/docs/stable/notes/cuda.html
- **[S4] PyTorch — non_blocking và pin_memory:** lợi ích phụ thuộc cách copy, synchronization và chi phí pinning. https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html
- **[S5] Ultralytics — Predict:** loại input, batching và quy tắc rectangular padding. https://docs.ultralytics.com/modes/predict/

Nguồn dự án: `tong_quan_hien_trang_du_an.md` đính kèm, ngày 28/09/2026, và mô tả metrics `gpu_clip` người dùng cung cấp ngày 30/09/2026. Khi code hiện tại khác tài liệu, ghi rõ khác biệt và ưu tiên code/config resolved/provenance đúng run đang benchmark.
