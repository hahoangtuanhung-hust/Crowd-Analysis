# Prompt Agent — Tối ưu pipeline Crowd Analysis trên Modal GPU
## Bối cảnh: Shibuya Performance Report ngày 2026-09-25

Bạn là kỹ sư tối ưu hệ thống computer vision. Hãy đọc repository hiện tại, xác minh báo cáo bên dưới bằng artifact/code, rồi **sửa trực tiếp code và chạy kiểm chứng trên Modal Tesla T4**. Không chỉ đưa ra kế hoạch hoặc thêm tài liệu.

**Mục tiêu sản phẩm:** tìm các tuyến đường đám đông đi phổ biến nhất và vẽ Common Path nhanh, ổn định, có mũi tên đúng hướng; số đường Top K được cấu hình trên UI. Chỉ cần tracking ẩn danh phục vụ quỹ đạo, không xác định danh tính.

**Thứ tự ưu tiên:** đúng tuyến và đúng hướng → giảm thời gian xử lý toàn pipeline → ổn định khi chạy dài và đổi cấu hình. Không đánh đổi việc mất người ở vùng xa để lấy FPS đẹp mà không công bố.

Các phương án và tiêu chí bên dưới là yêu cầu/thí nghiệm cần kiểm chứng, **không phải kết quả đã đạt**. Không tự nhận đã xem artifact, chạy GPU hoặc sửa thành công khi chưa thực hiện.

---

## 1. Dữ kiện đã có và giới hạn kết luận

Nguồn: báo cáo Shibuya do người dùng cung cấp, không phải benchmark mới của prompt này.

- Video: `data/videos/data-shibuya-test.mp4`, 1280×720, khoảng 30 FPS; toàn clip khoảng 65 giây, 1.951 frame.
- Smoke test: cùng đoạn 0–5 giây, 150 frame, Modal Tesla T4.
- Model: `yolo26n.pt`; engine: `tracklet_aggregation`; tracker và backend không đổi.
- Máy local không có CUDA. Mọi kết luận về GPU phải dựa trên lần chạy Modal thực tế.
- Profile production: 1 full-frame + 4 tile 2×2, tức 5 ảnh inference trên mỗi source frame.

| Profile | Inference p50 / p95 | Analytics p95 | Render p50 | Encode p50 | Pipeline FPS | Remote wall |
|---|---:|---:|---:|---:|---:|---:|
| Tiled FP32, 1280 | 180,448 / 196,980 ms | 4,335 ms | 1,829 ms | 6,848 ms | 3,705 | 41,15 s |
| Tiled FP16, 1280 | 146,068 / 159,627 ms | 4,053 ms | 2,010 ms | 6,275 ms | 4,318 | 35,32 s |
| Single-pass FP32, 1280 | 18,208 / 19,990 ms | 0,344 ms | 1,296 ms | 6,053 ms | 14,598 | 10,79 s |

Artifact được báo cáo:
- `outputs/common_path/shibuya-baseline-20260925-091031/`
- `outputs/common_path/shibuya-fp16-20260925-091721/`
- `outputs/common_path/shibuya-single-pass-20260925-091412/`

Mỗi run được báo cáo có `manifest.json`, `summary.json`, `metrics.csv`, `tracking_cache.jsonl`, `tracked_points_common_path.mp4` và cấu hình đã resolve. Hãy kiểm tra sự tồn tại và nội dung thực tế; không giả định các đường dẫn này luôn có trên local.

**Các giới hạn bắt buộc giữ nguyên:**
- Cả ba smoke run đều `insufficient_data`, chưa có Common Path hợp lệ. Chất lượng đường, jitter, adaptation, full-video equivalence và Browser/UI FPS đều chưa được xác nhận.
- Single-pass có 61 temporary Track ID, baseline có 248. Đây không phải số người thật hoặc phép đo recall; có thể liên quan đến coverage, fragmentation, ID switch hoặc detection trùng. Phải kiểm tra dữ liệu.
- Common Path chưa phải bottleneck đã được chứng minh trong smoke test. Không dồn công sức tối ưu vài ms analytics trong khi bỏ qua inference.
- Theo phép ngoại suy đơn giản, 1.951 / 3,705 ≈ 526,6 giây, tức 8,8 phút. Đây **không phải dự báo full-run** nhưng chưa giải thích trọn mốc 20 phút. Phải đo phần khởi động, I/O, truyền dữ liệu, chờ, xử lý video dài và UI; không quy toàn bộ 20 phút cho tiling.
- Lệnh tái lập cũ dùng baseline `--cache-policy refresh`, single-pass `--cache-policy reuse`. Phải kiểm tra cache trước khi công nhận A/B. Khác policy không tự chứng minh phép đo sai, nhưng cũng không được bỏ qua.

---

## 2. Phạm vi và rào chắn

Giữ model/backend hiện có làm mốc. **Không gộp chuyển TensorRT, INT8, đổi model, đổi tracker hoặc đổi engine vào đợt này.** FP16 chỉ là một biến thí nghiệm riêng nếu backend hiện có thực sự hỗ trợ.

Giữ kiến trúc tối thiểu:
`Video → Person Detection → Tracking ẩn danh → Tracklet → Common Path → Render/Stream`.

Giữ khả năng chọn Top K, màu riêng và Path ID ổn định. K=1 chọn tuyến được hỗ trợ mạnh nhất theo định nghĩa support rõ ràng; K>1 hiển thị tối đa K tuyến đủ bằng chứng, không bịa đường để đủ số lượng.

Không thêm face recognition, ReID nhận dạng người, cơ sở dữ liệu danh tính hoặc dịch vụ nặng không cần thiết. Nếu repository có code không liên quan, chỉ xóa khi đã kiểm tra import, call site, cấu hình và dependency. Không xóa video, model, artifact chuẩn, log benchmark hoặc dữ liệu người dùng.

Không viết lại toàn bộ hệ thống, thêm queue service/phân tán hoặc GPU hóa analytics chỉ vì có GPU. Chọn thay đổi nhỏ, có đo lường. Giữ profile production hiện tại cho đến khi candidate qua cổng chất lượng.

---

## 3. Kiểm tra pipeline và benchmark trước khi tối ưu

### 3.1. Đọc code và tái hiện đường chạy thực tế

Đọc README, entrypoint Modal/video/UI, config Shibuya, detector, tiling/merge, tracker, tracklet aggregation, renderer và cache. Tìm tên file thực tế; không tự tạo một pipeline demo song song bỏ qua hệ thống hiện tại.

Vẽ luồng gọi ngắn và xác định:
- Model có bị load lại mỗi request/frame không; GPU có thực sự được sử dụng không.
- Tiling tạo bao nhiêu ảnh, kích thước tensor thực tế và số lần forward.
- Có preprocessing/resize/copy/NMS/encode lặp không.
- Có serialize dữ liệu lớn, RPC/ghi file theo từng frame, chờ đồng bộ hoặc queue tăng vô hạn không.
- UI đang nhận video/frame/path bằng cách nào; FPS xử lý và FPS hiển thị được tính ở đâu.
- Config, frame timestamp và state video/camera được quản lý thế nào.

### 3.2. Làm rõ cache và tính công bằng A/B

Phân biệt model/download cache, detection cache, tracking cache, Common Path replay và output video cache.

**Đo tốc độ pipeline:** baseline và candidate đều phải thực thi inference thật trên cùng đầu vào; không để candidate đọc detection/tracking cache rồi so với baseline inference mới. Có thể dùng chung cache tải model/asset nếu công bố và áp dụng nhất quán.

**Đo thuật toán bằng replay:** được dùng tracking cache của từng profile nhưng phải ghi `analytics_replay`, tách khỏi benchmark GPU/end-to-end.

Kiểm tra cache key đủ phân biệt video/đoạn thời gian, model hash, backend/precision, preprocessing/imgsz, tile geometry/policy, threshold/merge và phiên bản liên quan. Tracking cache còn phụ thuộc detector output, tracker config và timebase. Đổi Top K/render không được làm mất detection cache hợp lệ; đổi detector không được tái dùng kết quả cũ sai profile.

Ghi cache hit/miss, số source frame có inference thật và số ảnh đã forward. Phân biệt:
- `model_invocation_count`: số lần gọi model.
- `inference_image_count`: tổng số ảnh/full-frame/tile được xử lý.
Batch 5 ảnh trong một invocation vẫn là 5 ảnh inference, không được báo thành giảm workload 5 lần.

### 3.3. Bổ sung đo lường đủ giải thích end-to-end

Đo theo source frame hoặc batch, gồm:
`decode → preprocess → transfer → inference → merge/NMS → tracking → analytics → render → encode → I/O/network/wait`.

Ghi mean, p50, p95, số mẫu và phạm vi timer; phần nào không đo được phải ghi rõ. Kiểm tra timing GPU bất đồng bộ bằng cơ chế timing phù hợp với runtime đang dùng; tránh đo riêng thời gian enqueue hoặc đồng bộ GPU ở mọi bước làm chậm pipeline production.

Không cộng/trừ các percentile riêng lẻ để suy ra latency tổng. Với các stage chạy chồng nhau, dùng trace/timestamp để giải thích overlap và wait.

Tách riêng:
- Warm worker processing và cold start/model load/first request.
- Thời gian chạy trên Modal và wall time người dùng thấy, gồm upload/download.
- Source FPS, processed FPS, detection FPS, encoded FPS và browser displayed FPS.
- Số frame decoded, inferred, tracker-updated, dropped/skipped và encoded.
- RAM/VRAM peak, queue depth/age và lỗi/OOM.

---

## 4. Thí nghiệm tối ưu inference — ít vòng, đúng trọng tâm

Không chạy một ma trận lớn ngay từ đầu. Ưu tiên đúng hai full-run để xác nhận chất lượng, sau đó chỉ thử phương án giải quyết vấn đề đã quan sát.

### Vòng A — Baseline và single-pass trên toàn clip

Chạy baseline tiled FP32 và single-pass FP32 trên cùng toàn bộ clip khoảng 65 giây, cùng T4, tracker, engine, threshold, output/encode policy và warm-up policy. Xác minh số frame thực tế.

Không dùng cache inference để thay phép đo tốc độ. Warm-up phải tách khỏi đo và reset state tracker/path trước phần video đánh giá.

Đối chiếu chất lượng theo mục 7. Chưa đổi production chỉ vì single-pass nhanh trên 5 giây.

### Vòng B — Chọn một hướng từ kết quả Vòng A

**Nếu single-pass giữ chất lượng phù hợp:** tối ưu overhead còn lại trong single-pass. Có thể thử FP16 trên chính profile này như một biến riêng; không mặc định lợi ích giống tiled FP16.

**Nếu single-pass mất coverage ở vùng xa/đông:** ưu tiên policy trung gian, thử tối đa hai candidate có căn cứ:
- Full-frame mỗi frame có detection, tile bổ sung theo chu kỳ cấu hình.
- Full-frame kết hợp ROI/tile giới hạn ở vùng thực sự thiếu coverage.
- Giảm số tile/kích thước tile sau khi kiểm tra người nhỏ trong từng vùng.
- Batch các tile cùng timestamp, chỉ khi đo được lợi ích và bộ nhớ phù hợp.

Các policy trên là ứng viên, không phải yêu cầu triển khai tất cả. Bắt đầu policy cố định dễ tái lập; chỉ thêm adaptive tiling khi có tín hiệu kích hoạt rõ, giới hạn tải và thực nghiệm chứng minh cần thiết.

**Yêu cầu khi ghép detection:**
- Quy đổi toàn bộ bbox về cùng hệ tọa độ source trước khi merge.
- Gộp full-frame/tile và xử lý duplicate trước khi tracker nhận kết quả.
- Tracker nhận một tập detection thống nhất cho mỗi source timestamp, không update liên tiếp cùng frame theo từng tile.
- Kiểm tra người đứng sát nhau; không dùng merge quá mạnh để giảm số detection giả tạo.
- Không dùng tile cũ như detection hiện tại nếu không có cơ chế hợp lệ được kiểm chứng.
- Ghi lịch tile thực tế; kiểm tra dao động track/path giữa frame có tile và không có tile.

### Vòng C — Kiểm chứng candidate thắng

Chạy lại candidate được chọn ít nhất một lần trên toàn clip trong cùng điều kiện warm. Nếu kết quả khác biệt lớn hoặc sát cổng chất lượng, lặp lại baseline để kiểm tra biến động môi trường.

Nếu không candidate nào đáp ứng chất lượng, giữ production, báo rõ trade-off và đề xuất bước tiếp theo từ dữ liệu. Không hạ tiêu chí để tuyên bố thành công.

---

## 5. Tinh gọn pipeline và sử dụng Modal GPU

Chỉ triển khai các thay đổi phù hợp code hiện có và bottleneck đã đo:

- Tái sử dụng model trong vòng đời worker; state tracker/path phải riêng theo video/camera/session. Không trộn state giữa người dùng hoặc giữa lần replay.
- Tránh load model, dựng tracker, tạo object lớn, sao chép frame và serialize kết quả nặng không cần thiết trong vòng lặp.
- Tránh một remote call cho từng frame nếu code đang làm vậy và đo thấy đây là overhead; ưu tiên xử lý một phiên/đoạn video trong worker với luồng dữ liệu phù hợp.
- Chỉ tách decode, inference, analytics và encode bằng worker/queue khi có lợi ích. Queue phải có giới hạn, backpressure, cơ chế dừng và thu hồi tài nguyên; không tạo thread/task vô hạn.
- Giới hạn concurrency GPU và đo VRAM; không nhân bản model tùy tiện hoặc dùng chung state không an toàn.
- Hạn chế ghi/flush log và cache theo từng frame; ghi theo batch khi không làm mất tính tái lập.
- Cache hình học/lớp overlay Common Path, chỉ dựng lại khi path/config thay đổi; compositing lên video vẫn phải đúng frame và đúng tọa độ.

**Tách hai chế độ rõ ràng:**
- `offline_fast`: giữ đầy đủ trình tự, timestamp và duration đầu ra. Không bỏ frame ngầm để tăng FPS. Mọi thí nghiệm giảm tần suất detection phải được công bố riêng.
- `live`: queue giới hạn, có thể ưu tiên frame mới và bỏ frame quá hạn; ghi số frame bỏ, queue age và source-to-display latency. Tracker phải xử lý khoảng thời gian thực giữa các lần update.

Nếu thử detection stride, phải đánh giá lại coverage và Common Path. Không đưa empty detections vào tracker với ngữ nghĩa “không thấy ai” thay cho thao tác predict/skipped-update nếu tracker không hỗ trợ đúng hành vi đó. Kiểm tra API thực tế trước khi sửa.

Dùng **source timestamp** cho quỹ đạo, TTL, cửa sổ support và nhịp analytics khi replay offline. Không để tốc độ xử lý máy thay đổi kết quả thống kê. Với live, xử lý timestamp/frame đến trễ hoặc đảo thứ tự rõ ràng.

Xác minh cấu hình Modal bằng dependency/runtime của repository. Không tự nâng phiên bản, đổi GPU hoặc đổi backend rồi ghi lợi ích thành tối ưu code. Ghi GPU name, runtime, package versions, model/config/code hash trong manifest.

---

## 6. Common Path: ổn định, incremental, không tạo đường giả

Không thay engine sang grid/heatmap. Đường đầu ra phải là polyline/curve liên tục đại diện cho nhiều tracklet phù hợp, không phải đường nối tùy ý các tâm cụm.

### 6.1. Dữ liệu và cách tính support

- Giữ temporary Track ID để gom quỹ đạo, không dùng làm danh tính.
- Chọn điểm đại diện nhất quán theo pipeline hiện có; nếu đổi cách lấy điểm phải so sánh riêng, không thay ngầm trong A/B detector.
- Lọc tracklet lỗi, trùng, nhảy tọa độ, quá ngắn hoặc gần đứng yên theo ngưỡng cấu hình.
- Phân biệt **nối mảnh của cùng một track** với **tổng hợp tuyến chung của nhiều người ở các thời điểm khác nhau**. Chỉ bước nối cùng track mới cần ràng buộc thời gian chuyển tiếp tương ứng; không ép mọi người cùng tuyến phải có timestamp nối tiếp nhau.
- Định nghĩa support rõ: một temporary track chỉ đóng góp tối đa một lần cho một path trong cùng cửa sổ, không tăng phiếu theo frame/điểm.
- Không tuyên bố support là số người duy nhất tuyệt đối. Công bố hạn chế do fragmentation; chỉ gộp mảnh track khi có đủ bằng chứng, không suy đoán danh tính.
- Kiểm tra support theo đoạn và bằng chứng tại chỗ nối, không dùng support lớn ở một đoạn để hợp thức hóa cả tuyến không được quan sát.

### 6.2. Gom tuyến và ổn định theo thời gian

- Chỉ xử lý phần tracklet mới; dùng cửa sổ thời gian/buffer giới hạn. Không cluster toàn bộ lịch sử ở mỗi frame.
- Gom theo hình học và hướng chuyển động; tách chiều ngược nhau. Không nối nhầm ở giao cắt, qua vùng không có dữ liệu hoặc qua khoảng hở lớn.
- Dùng index không gian nếu profiling cho thấy cần; index chỉ phục vụ tìm lân cận, không biến thuật toán thành đường trên lưới.
- Matching path cũ/mới theo cả hình học và hướng để giữ Path ID/màu, không gán ID theo thứ tự Top K.
- Chỉ smoothing sau khi đã match đúng path và căn chỉnh các điểm tương ứng theo chiều dài đường. Không lấy trung bình các path khác hướng/khác tuyến hoặc các mảng điểm lệch vị trí.
- Có điều kiện xác nhận trước khi đổi/thêm path, thời gian giữ ngắn khi thiếu dữ liệu và TTL loại path cũ. Cấu hình được, dùng source time.
- Giảm rung nhưng không đóng băng tuyến: nếu luồng thật thay đổi đủ bằng chứng, phải cập nhật; không giữ đường cũ vô hạn để làm đẹp metric.
- Khi chưa đủ support, trả `insufficient_data`; không vẽ đường giả hoặc tự hạ ngưỡng chỉ để có hình.

### 6.3. Render và hot config

- Vẽ đường liền, có thể cong/gấp khúc đúng dữ liệu. Không smoothing tạo đường cắt góc qua vùng chưa quan sát.
- Mũi tên theo tiếp tuyến và chiều của path; có mũi tên cuối và bổ sung dọc đường khi cần. Tránh arrow ở đoạn dài bằng 0 hoặc bị đảo sau resampling.
- Kiểm tra mapping source → canvas/video, resize/letterbox và độ dày nét; không để resize UI làm lệch path.
- Tách dữ liệu candidate path khỏi lớp Top K hiển thị. Đổi Top K hoặc màu không reset tracking, không chạy lại detector, không chờ tích lũy lại từ đầu.
- Config có validation và version. Kết quả tính từ config cũ không được ghi đè config mới.
- Thay đổi thuật toán/cửa sổ có thể cần rebuild analytics từ buffer hợp lệ; khai báo rõ tác động. Đổi model/tile khác loại với đổi style/render.
- Khi đổi camera/video, seek ngược hoặc replay từ đầu, reset state liên quan đúng cách.

---

## 7. Cổng chất lượng và hiệu năng

**Baseline không phải ground truth.** Candidate ít Track ID hơn không tự động là kém, và nhiều detection hơn không tự động là tốt.

### 7.1. Đối chiếu tối thiểu trên full video

Tạo bảng baseline/candidate trên cùng timestamp:
- Detection theo vùng gần/xa/đông; số detection trùng và missed detections qua mẫu có nhãn/kiểm tra thủ công.
- Temporary Track ID, độ dài track, fragmentation, số tracklet hợp lệ; phân biệt metric có ground truth với proxy.
- Support theo path/đoạn, thời điểm có path hợp lệ đầu tiên tính bằng source seconds, thời gian path hợp lệ được duy trì.
- Hướng, tuyến, ranking Top K; kiểm tra không sinh tuyến đi tắt hoặc nối nhầm tại giao cắt.
- Path ID/color churn, độ rung hình học sau matching, biến mất/xuất hiện bất thường và đảo hướng.
- Thời gian thích nghi khi video thực sự có thay đổi luồng. Nếu không có tình huống đó, ghi `NOT_RUN`, không tự suy ra adaptation.
- FPS end-to-end, p95 frame processing, inference images/source frame, wall time, RAM/VRAM và queue.

Chọn một tập mẫu nhỏ cố định, ví dụ 12–20 frame trải đều video và có vùng người nhỏ. Dùng nhãn sẵn có; nếu không có, xuất bộ ảnh review và ghi rõ chưa có ground truth. Không gọi tỉ lệ detection-count là recall. Mọi ngưỡng chấp nhận phải được ghi trước khi chọn candidate; không điều chỉnh theo kết quả để làm đẹp báo cáo.

Xuất snapshot cùng timestamp, đề xuất 15/30/45/60 giây, cho baseline và candidate; kèm video overlay. Nếu thời điểm đó chưa có path, giữ trạng thái thực tế và giải thích.

### 7.2. Điều kiện chọn profile

Candidate chỉ được đề xuất thay production khi:
- Chạy hết clip và lần lặp xác nhận, không crash/OOM; frame/timestamp/output duration đúng chính sách.
- Không làm mất tuyến chính hoặc sai hướng ở các khoảng đã kiểm tra; không che giảm coverage ở vùng xa bằng số liệu trung bình toàn ảnh.
- Chất lượng đường và tính ổn định đạt ngưỡng đã công bố, có minh chứng thay vì chỉ có FPS.
- Có cải thiện end-to-end được đo trong cùng điều kiện; báo cả trade-off chất lượng và chi phí. Có thể đặt mục tiêu thử nghiệm ≥2× warm baseline, nhưng không xem đó là kết quả hoặc cam kết.
- State/buffer có giới hạn, được dọn sau phiên; không có tăng bộ nhớ không kiểm soát trong bài kiểm chứng.
- Kết quả full-video không còn chỉ là `insufficient_data`. Nếu vẫn chưa có path, phải điều tra dữ liệu/ngưỡng hoặc ghi chưa đạt mục tiêu sản phẩm.

### 7.3. UI và config

Kiểm tra tối thiểu:
- Top K: `1 → 3 → 2` khi đang chạy, không reset track/path không cần thiết; khi chỉ có ít tuyến hợp lệ thì hiển thị ít hơn K.
- Đổi liên tiếp cấu hình, không có kết quả cũ ghi đè; ID/màu ổn định.
- Camera/video switch và seek không giữ path của nguồn cũ.
- Browser đo displayed FPS, frame age/latency và thời gian config tới overlay; không suy ra từ remote processing FPS.
- Đặt mục tiêu thử nghiệm đổi Top K/style hiện lên trong ≤1 giây nếu candidate data sẵn có; ghi thời gian thực đo và nguyên nhân nếu chưa đạt.

Không ép “30 FPS inference” trên T4 bằng cách đếm lại frame cũ. Video hiển thị mượt, inference rate và độ trễ dữ liệu phải được báo riêng.

---

## 8. Kiểm thử đủ dùng, tránh tốn GPU vô ích

Tái dùng harness và artifact hiện có. Ưu tiên hai full-run ở Vòng A, một hoặc hai candidate có căn cứ, và lần lặp candidate thắng. Không chạy toàn bộ tổ hợp precision × imgsz × tile × stride.

Dùng replay đúng cache để test analytics/render/config; không cần inference lại chỉ vì đổi màu/Top K. Replay phải xử lý timestamp và state như luồng thật.

Bổ sung vài regression test tập trung:
- Cache không dùng sai kết quả giữa tiled/single-pass/precision.
- Tile merge không update tracker nhiều lần cho cùng timestamp.
- Luồng đối hướng/giao cắt không bị gộp sai; dữ liệu thiếu không sinh đường giả.
- Top K thay đổi không reset tracker; stale config không ghi đè.
- Queue/buffer cleanup, seek/reset, và detection-skipping timebase nếu tính năng đó được triển khai.

Nếu thiếu quyền Modal, dữ liệu, GPU hoặc browser, báo đúng bước bị chặn, hoàn thiện phần code có thể kiểm chứng và cung cấp lệnh tái lập. Không ghi `PASS` cho phần chưa chạy. Không để benchmark tiếp tục tiêu tốn tài nguyên ngoài phạm vi cần thiết.

---

## 9. Sản phẩm bàn giao bắt buộc

1. **Code patch thực tế** theo kiến trúc hiện có; danh sách file sửa/xóa và lý do. Không để hai pipeline trùng chức năng.
2. **Config candidate tách biệt** với production, chú thích policy inference, ngưỡng và giới hạn. Chưa đủ chất lượng thì candidate vẫn experimental.
3. **Lệnh chạy có thể sao chép** cho baseline, candidate, replay và UI. Kiểm tra CLI thực tế; không bịa flag. Tên run/cache namespace không ghi đè artifact chuẩn.
4. **Artifact của từng run:** manifest/config đã resolve, metrics/summary, tracking cache có provenance, overlay video/snapshot và trạng thái test.
5. **Báo cáo trước–sau:** bottleneck thực đo, thay đổi đã làm, bảng tốc độ/chất lượng, cold/warm và cache policy, giải thích thời gian end-to-end, phần chưa kiểm chứng.
6. **Kết luận triển khai và rollback:** profile đề xuất, bằng chứng qua cổng chất lượng, cách bật và cách quay về production cũ.

Trong báo cáo cuối, tách rõ `PASS`, `FAIL`, `NOT_RUN`, `BLOCKED`; không gộp “chạy được” với “chất lượng tương đương”. Không công bố “recall không giảm”, “UI mượt”, “Common Path ổn định” hoặc “đã nhanh hơn X lần” khi thiếu phép đo tương ứng.

**Bắt đầu bằng đọc artifact và kiểm tra cache/timing; chạy full-video A/B; tối ưu nút thắt thật; kiểm chứng tuyến/hướng trước khi đổi production. Ưu tiên kết quả ít thay đổi nhưng chạy đúng và có thể tái lập.**
