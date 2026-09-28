# Prompt thực thi: Tối ưu thuật toán và pipeline Common Path trên GPU Modal

Bạn là kỹ sư Computer Vision và Performance Engineering. Hãy đọc repository Crowd Analysis hiện tại, xác định luồng đang chạy, sửa trực tiếp code và kiểm chứng trên GPU Modal. Không chỉ đề xuất kiến trúc; không viết lại toàn bộ hệ thống nếu có thể cải tiến các module hiện có.

## 1. Mục tiêu và giới hạn

Mục tiêu chính: **tìm các tuyến di chuyển phổ biến của đám đông từ tracklet ẩn danh, vẽ mỗi Common Path thành một đường liền có mũi tên đúng hướng, cập nhật nhanh và ổn định khi chạy video hoặc thay đổi cấu hình.**

Mặc định hiển thị Top 1; cho phép người dùng chọn Top K trên UI. K là **số tuyến tối đa được hiển thị**, không phải yêu cầu thuật toán phải tạo đủ K cụm hoặc K đường. Không đủ bằng chứng thì hiển thị ít hơn K và trạng thái đang tích lũy dữ liệu.

Không cần xác định người đó là ai. Giữ Track ID tạm trong từng camera/session để gom quỹ đạo và hạn chế đếm trùng; không thêm face recognition, appearance ReID, liên kết danh tính nhiều camera hoặc LLM. Common Path phải xuất phát từ tracklet, không chuyển bài toán sang grid/heatmap hoặc chỉ vẽ một vector hướng trung bình của toàn cảnh.

Ưu tiên theo thứ tự:
1. Đúng tuyến và đúng hướng.
2. Ổn định hình học, Path ID, màu và thứ hạng hiển thị.
3. Pipeline nhanh, bộ nhớ có giới hạn và UI không bị nghẽn.
4. Đo được hiệu quả trước/sau trên GPU Modal.

Giữ detector, model weights, tracker và backend inference hiện có làm đối chứng. **Không đưa conversion TensorRT, đổi model hoặc hạ độ phân giải vào cùng lượt tối ưu thuật toán.** Nếu repo đang dùng TensorRT thì giữ backend đó; chỉ sửa lỗi tương thích thật sự và ghi rõ. Không lấy tăng FPS do bỏ nhiều frame hoặc giảm chất lượng đầu vào làm bằng chứng thuật toán tốt hơn.

Các thiết kế và ngưỡng trong prompt này là **đề xuất cần kiểm chứng**, không phải mô tả code hiện tại hoặc kết quả benchmark đã đạt.

## 2. Rà soát và tạo đối chứng trước khi sửa

Đọc README, entrypoint, cấu hình, worker video, API và frontend. Lần theo đúng pipeline thực tế:

```text
Video source / decode
  -> Preprocess + person detection trên GPU
  -> Tracking ẩn danh
  -> Tracklet mới / phần tracklet vừa cập nhật
  -> Lọc và ghép với các tuyến ứng viên có hướng
  -> Cập nhật Common Path theo chu kỳ
  -> Snapshot đường + hướng + score + phiên bản
  -> Render / encode / truyền / hiển thị
```

Chỉ ra file/hàm và bằng chứng cho các điểm chậm hoặc không ổn định: clustering toàn bộ lịch sử mỗi frame, vòng lặp so sánh mọi cặp, cache không hết hạn, cập nhật trùng tracklet, reset state khi đổi config, đảo thứ tự điểm, gán màu theo rank, sai hệ tọa độ, model load lặp, copy frame/encode lặp, queue tích tụ hoặc frontend vẽ lại quá nhiều.

Lưu baseline và cấu hình trước khi sửa. Tạo cache detection/tracklet có timestamp trên video cố định để replay cùng đầu vào khi kiểm tra riêng thuật toán. Cache phải ghi video hash, model/weights, config và schema version; không nhầm kết quả của các camera hoặc các lần chạy.

Refactor trong phạm vi Common Path và luồng video liên quan. Khi xóa code thừa, kiểm tra consumer/import/config/endpoint trước; không xóa dữ liệu, weights, secrets hoặc thay đổi chưa commit. Không dành phần lớn công việc cho dọn code ngoài phạm vi tối ưu này.

## 3. Tối ưu thuật toán Common Path

### 3.1. Chuẩn hóa dữ liệu và thời gian

Thống nhất tọa độ giữa đầu ra detector, frame gốc, ROI và lớp vẽ trên UI. Kiểm tra resize, letterbox, crop, scale, offset và tỷ lệ hiển thị trước khi chỉnh ngưỡng clustering.

Dùng một điểm đại diện nhất quán, ưu tiên tâm đáy bbox nếu phù hợp dữ liệu hiện có. Lưu timestamp nguồn/video PTS, frame ID, camera ID, session ID và Track ID tạm. Không dùng tọa độ bbox chưa chuyển khỏi ảnh letterbox để vẽ lên video gốc.

Ngưỡng không gian nên biểu diễn theo tỷ lệ đường chéo frame gốc hoặc hệ tọa độ đã hiệu chuẩn. Với đơn vị tỷ lệ đường chéo, tính khoảng cách trong tọa độ pixel gốc rồi chia cho `D = hypot(width, height)`; không coi khoảng cách Euclidean trong `(x/width, y/height)` là tương đương khi tỷ lệ khung hình khác nhau. Không gắn nhãn mét hoặc m/s nếu chưa có hiệu chuẩn.

Dùng **event time của nguồn video** để tính cửa sổ phân tích, tuổi tracklet và hết hạn bằng chứng. Dùng đồng hồ monotonic để đo thời gian thực thi. Khi replay nhanh hơn realtime, cửa sổ 60 giây vẫn phải tương ứng 60 giây nội dung video. Xử lý seek, timestamp lùi và đổi video bằng session/epoch mới, không trộn dữ liệu hai lượt phát.

### 3.2. Lọc tracklet nhưng không làm mất đường rẽ

Loại điểm trùng, bước nhảy bất thường và chuyển động chỉ do bbox rung. Xét đồng thời số quan sát, thời lượng, chiều dài quỹ đạo và chất lượng tracking; không chỉ dùng khoảng cách từ đầu đến cuối vì có thể loại nhầm đường cong hoặc quay đầu.

Giữ các đoạn rẽ thực sự, người đi chậm và quỹ đạo đủ bằng chứng. Ước lượng hướng từ chuyển động cục bộ đã lọc nhiễu; hướng trung bình toàn tracklet chỉ được dùng như một bước lọc sơ bộ nếu phù hợp. Một tuyến cong không bắt buộc mọi đoạn có cùng góc với đoạn đầu.

Cập nhật từ track đang hoạt động, không đợi tất cả track kết thúc. Mỗi lần chỉ đưa phần dữ liệu mới vào bộ tổng hợp. Nếu cắt một track thành nhiều tracklet chồng lấp, phải có khóa đóng góp để không cộng lặp cùng quan sát hoặc cùng lượt đi.

### 3.3. Ghép tăng dần, không clustering lại toàn bộ mỗi frame

Ưu tiên cải tiến thuật toán hiện có theo hướng:

```text
Tracklet delta
  -> Tách thành các đoạn có hướng, giữ điểm rẽ quan trọng
  -> Truy vấn một tập nhỏ tuyến/đoạn ứng viên gần nhau
  -> Kiểm tra tương thích vị trí + hướng cục bộ + phần chồng lấp
  -> Cập nhật cụm đoạn/tuyến bị ảnh hưởng
  -> Dựng lại các Common Path bị thay đổi
```

Sử dụng spatial index nhẹ như KD-tree, R-tree hoặc cấu trúc tương đương đã có để thu hẹp ứng viên. Index không gian chỉ phục vụ tìm lân cận; không biến đường đầu ra thành đường bám ô lưới. Giới hạn số ứng viên và số đoạn mỗi tick. Đo chi phí cập nhật index, không chỉ chi phí query.

Không chạy DBSCAN/HDBSCAN trên toàn bộ lịch sử hoặc tính DTW mọi cặp mỗi frame. Không thêm các thư viện nặng chỉ để đưa clustering lên GPU. Nếu cần kiểm tra hình dạng tốn chi phí, chỉ áp dụng sau các bước lọc rẻ trên tập ứng viên nhỏ.

Điều kiện ghép phải xét khoảng cách tới tuyến, hướng tiếp tuyến tại vùng tương ứng và thứ tự tiến dọc tuyến. Có thể dùng cosine hoặc góc giữa các tiếp tuyến; không dùng `abs(dot)` vì sẽ coi hai chiều ngược nhau là cùng hướng. Không ghép chỉ vì hai tracklet gần nhau hoặc cắt nhau tại một điểm.

Phân biệt rõ:
- **Ghép mảnh của cùng một lượt tracking:** kiểm tra khoảng trống thời gian, vị trí, chuyển động và tính một-một; không khẳng định danh tính.
- **Tổng hợp tuyến từ nhiều lượt đi:** tracklet có thể xuất hiện ở các thời điểm khác nhau trong cửa sổ; không yêu cầu thời gian các tracklet của những người khác nhau phải nối tiếp nhau.

Nếu dùng đồ thị đoạn có hướng, chỉ tạo cạnh chuyển tiếp khi có bằng chứng đi qua liên tục hoặc phép nối ngắn đã qua kiểm tra. Giữ thông tin đi vào/đi ra tại giao cắt để không tạo nhánh rẽ không tồn tại. Không chỉ lấy đường dài nhất trong graph hoặc nối mọi đoạn gần nhau.

### 3.4. Tính mức phổ biến đúng với bằng chứng

Không xếp hạng bằng số frame, số điểm, thời gian đứng yên hoặc số lần một tracklet được gửi lại. Một lượt tracking không được tăng support liên tục chỉ vì tồn tại lâu.

Duy trì đóng góp có khóa theo camera/session/lượt tracking và vùng tuyến được quan sát. Hết cửa sổ phải loại đóng góp khỏi **cả support và hình học đại diện**, không chỉ trừ bộ đếm nhưng giữ hình dạng cũ mãi. Cache chống trùng và thông tin nối mảnh cũng phải có TTL/giới hạn.

Ưu tiên support từ các lượt quan sát đi dọc tuyến với mức coverage đủ và tiến triển đúng hướng. Khi chỉ có tracklet cục bộ, được dùng support theo đoạn cùng mức bao phủ liên tục để xếp hạng; ghi rõ đó là ước lượng phổ biến theo đoạn, không phải số người duy nhất đi hết tuyến.

Không cộng support của A→B và B→C rồi kết luận tất cả người đã đi A→C. Không để một đoạn rất đông kéo dài thành đường giả qua vùng thiếu bằng chứng. Không ưu tiên tuyến dài chỉ vì cộng tổng support của nhiều đoạn.

Chọn và mô tả một công thức score rõ ràng: support/coverage nào được dùng, có chuẩn hóa chiều dài hay không, điều kiện tuyến đủ hợp lệ và cách xử lý các nhánh. So sánh các ứng viên trên cùng định nghĩa; không trộn “số lượt đi trọn tuyến” với “support cục bộ” thành cùng một bộ đếm. Loại các tuyến trùng hình học và cùng chiều để Top K không chứa nhiều bản sao của cùng một tuyến.

### 3.5. Dựng một đường đại diện liên tục có hướng

Mỗi Common Path là một polyline có thứ tự theo chiều di chuyển, không phải tập các tracklet rời hoặc đám mũi tên riêng lẻ.

Căn chỉnh theo phần chồng lấp, vị trí dọc tuyến và hướng trước khi tổng hợp. Có thể resample theo chiều dài cung, dùng đại diện robust/medoid hoặc cập nhật trọng số có giới hạn. **Không lấy trung bình hai tracklet khác điểm bắt đầu/kết thúc chỉ vì đã resample về cùng số điểm.**

Đối với đoạn mới nối tiếp, kiểm tra khoảng hở, tiếp tuyến đầu/cuối và bằng chứng hỗ trợ. Khi thiếu dữ liệu, giữ đoạn ngắn hợp lệ hoặc tách tuyến; không bắc cầu khoảng trống lớn để tạo cảm giác đường đã liên tục.

Làm mượt không được cắt góc qua vùng không có người đi, tạo vòng lặp giả hoặc làm mất nhánh rẽ. Ưu tiên polyline đơn giản hóa có kiểm soát sai số; chỉ dùng spline khi đã kiểm tra overshoot. Đường thẳng, gấp khúc và cong đều hợp lệ nếu phù hợp bằng chứng.

### 3.6. Ổn định qua các lần cập nhật

Ghép tuyến mới với tuyến cũ bằng hướng, hình học và phần chồng lấp; giữ Path ID và màu theo tuyến, không theo thứ hạng. Không tái sử dụng ngay ID của tuyến vừa hết hạn cho tuyến khác.

Làm mượt theo thời gian trên các điểm đã căn chỉnh. Có thể dùng:

```text
alpha = 1 - exp(-delta_t / tau)
new_geometry = (1 - alpha) * previous_geometry + alpha * candidate_geometry
```

Chỉ áp dụng công thức khi hai geometry biểu diễn cùng tuyến và các điểm tương ứng đã được căn chỉnh. Không blend hai nhánh khác nhau hoặc làm mượt hai hướng đối nghịch thành một tuyến ở giữa.

Dùng ngưỡng thay thế và xác nhận qua vài lần cập nhật để Top 1 không nhảy liên tục khi hai tuyến có support gần nhau. Tuy nhiên, thay đổi luồng thật phải được cập nhật; không “ổn định” bằng cách đóng băng đường.

Mỗi tuyến có trạng thái `warming / active / stale / expired` hoặc tương đương. Grace period chỉ chống mất đường do thiếu quan sát ngắn, không gia hạn vô hạn bằng chứng đã hết hạn. Đo cả độ rung khi luồng ổn định và độ trễ thích nghi khi luồng thay đổi.

## 4. Tối ưu pipeline video và vẽ Common Path

### 4.1. Tách nhịp xử lý, giới hạn bộ nhớ

Detection/tracking xử lý theo thứ tự thời gian; Common Path chạy theo chu kỳ riêng; renderer sử dụng snapshot hoàn chỉnh gần nhất. Không chặn inference hoặc UI để chờ clustering toàn bộ lịch sử.

Dùng queue/buffer có giới hạn, một job cập nhật Common Path tại một thời điểm cho mỗi session. Nếu tick mới đến khi job trước chưa xong, gộp các delta chưa xử lý thành lần kế tiếp có giới hạn; không tạo hàng dài job cũ, không làm mất bằng chứng đã nhận. Có cơ chế resync từ buffer còn giữ nếu delta overflow và phải ghi metric.

Snapshot cần tối thiểu:

```text
camera_id, session_id, config_version, path_version,
source_time_range, data_watermark, generated_at,
paths[
  path_id, ordered_points, direction,
  score, support_type, support_value, state, color
]
```

`path_version` và `config_version` tăng đơn điệu trong session. Frontend bỏ snapshot cũ hoặc sai session. Khi đổi cấu hình trong lúc job cũ đang chạy, kết quả cũ không được ghi đè kết quả mới.

Dữ liệu truyền sang renderer phải bất biến hoặc có quyền sở hữu rõ ràng. Không giữ lock trong suốt inference, clustering hoặc encode. Không tạo thread/process theo từng frame. Nếu profiling cho thấy Python CPU-bound bị nghẽn, ưu tiên vector hóa phần tính toán; chỉ tách process khi lợi ích vượt chi phí copy/serialize.

### 4.2. Realtime và offline là hai chế độ khác nhau

**Realtime:** queue ngắn, ưu tiên frame mới khi quá tải, ghi số frame bỏ và tuổi frame. Client preview chậm không được kéo nghẽn luồng phân tích.

**Offline/replay:** xử lý theo thứ tự, không bỏ frame âm thầm. Các bản so sánh phải dùng cùng lịch sampling. Nếu video được lấy mẫu thưa, ghi rõ số quan sát thực sự đã dùng.

Nếu thay đổi khoảng cách giữa các frame detector, tracker phải xử lý `delta_t` đúng hoặc sử dụng cơ chế predict được hỗ trợ. Không đưa một detection cũ vào nhiều frame rồi coi là nhiều quan sát mới. Kiểm chứng quỹ đạo khi có mất frame; không chỉ báo FPS tăng.

Giảm copy GPU–CPU, resize và chuyển màu lặp; chỉ chuyển bbox/score hoặc dữ liệu nhỏ cần cho tracker nếu phù hợp backend. Giữ model load một lần; dùng inference mode nếu backend hỗ trợ. Chỉ thử batching hoặc GPU decode khi profiling chứng minh có lợi và vẫn giữ thứ tự tracking/latency.

### 4.3. Vẽ đường nhanh, có hướng rõ

Vẽ một nét liền cho mỗi tuyến, mũi tên ở cuối và thêm mũi tên dọc tuyến khi đủ dài. Hướng mũi tên theo tiếp tuyến polyline đã được định hướng; bỏ các đoạn gần độ dài zero trước khi tính góc. Đặt mũi tên theo khoảng cách dọc đường, không theo chỉ số điểm không đều.

Giữ màu ổn định bằng Path ID. Mặc định chỉ hiện Common Path; bbox, Track ID và quỹ đạo từng người là debug tùy chọn.

Cache geometry và lớp overlay theo `path_version`, cấu hình hiển thị và kích thước khung hình. Chỉ tính lại mũi tên/lớp đường khi các yếu tố đó đổi; vẫn ghép lớp này với frame video mới. Không encode lại cùng frame cho từng người xem nếu có thể dùng chung an toàn.

Chọn một nơi vẽ chính phù hợp stack hiện có: frontend Canvas/SVG hoặc backend overlay. Không vẽ trùng cả hai. Kiểm tra resize, letterbox và vị trí lớp overlay trên video thật.

Khi xuất video kiểm chứng offline, dùng snapshot tương ứng timestamp của frame; không áp kết quả tổng hợp cuối video lên toàn bộ video rồi gọi đó là kết quả realtime.

## 5. Cấu hình nhanh mà không reset pipeline

UI chỉ cần các điều khiển chính: Top K, cửa sổ phân tích, chu kỳ cập nhật, ngưỡng hỗ trợ và mức làm mượt. Ngưỡng ghép nâng cao để trong nhóm riêng, không làm UI quá phức tạp.

Phân loại cách áp dụng:

| Nhóm thay đổi | Hành vi bắt buộc |
|---|---|
| Top K, màu, độ dày, mũi tên, bật/tắt debug | Áp dụng trên tập tuyến ứng viên/snapshot đã có; không rerun detection/tracking |
| Support, mức làm mượt, chu kỳ cập nhật | Áp dụng có version ở tick hợp lệ tiếp theo; không xóa toàn bộ state |
| Cửa sổ, ROI, ngưỡng ghép làm đổi ý nghĩa tuyến | Tính lại phần cần thiết từ buffer giới hạn; hiển thị warming nếu thiếu dữ liệu |
| Nguồn video, model, hệ tọa độ không tương thích | Restart phần bắt buộc theo session mới, thông báo rõ; không giữ đường của video cũ |

Tách `max_display_paths` khỏi số tuyến ứng viên nội bộ. Đổi Top 1→Top 3 không được làm thuật toán đổi bản chất hoặc reset toàn bộ cụm. Khi tăng K vượt số ứng viên đủ bằng chứng, chỉ hiện số đang có.

Tăng cửa sổ lớn hơn lịch sử đã lưu không thể tạo lại dữ liệu đã bị xóa: phải trả trạng thái tích lũy hoặc replay nguồn nếu chế độ đó được hỗ trợ. Không âm thầm tuyên bố đã áp dụng đầy đủ.

Validate kiểu, khoảng giá trị, đơn vị và quan hệ giữa tham số; trả rõ `accepted_config_version` và `applied_config_version`. Gộp các thay đổi slider liên tiếp để tránh phát sinh hàng loạt job. Không đòi restart app cho thay đổi hiển thị.

Cấu hình khởi đầu dưới đây chỉ để thử nghiệm khi chưa có giá trị hợp lý trong repo. Giữ baseline hiện tại trước khi thử preset này, ánh xạ sang schema thật và không hardcode:

```yaml
common_path:
  max_display_paths: 1
  candidate_limit: 32
  window_seconds: 60
  update_interval_ms: 500
  min_support_passages: 5
  resample_points: 32
  local_direction_gate_degrees: 35
  max_join_gap_frame_diagonal_ratio: 0.02
  temporal_smoothing_tau_seconds: 1.0
  switch_margin_ratio: 0.15
  switch_confirm_updates: 3
  stale_grace_seconds: 3

pipeline:
  realtime_queue_capacity_frames: 2

benchmark:
  repeats: 3
  warmup_inference_iterations: 20
  max_gpu_runs: 6
```

`min_support_passages` chỉ dùng khi thực sự đo lượt tracking; nếu dùng support theo đoạn, đặt tên/đơn vị riêng. Ngưỡng góc là góc cục bộ vùng ghép, không phải góc toàn tuyến. Bổ sung giới hạn track/tracklet/điểm và min-length theo dữ liệu thật; khi chạm giới hạn phải ghi eviction/overflow thay vì âm thầm thiên lệch kết quả.

## 6. Chạy và kiểm chứng trên GPU Modal

“Modal” ở đây là nền tảng cloud GPU Modal. Đọc cấu hình Modal đang có và tài liệu chính thức theo SDK thực tế; không viết API dựa trên phỏng đoán.

### 6.1. Môi trường có thể tái lập

Giữ image/dependency tương thích đang chạy, pin phiên bản liên quan; ghi Python, CUDA runtime, driver, detector backend và commit/config hash. Xác nhận GPU thật bằng tên thiết bị, VRAM và log backend sử dụng GPU; không chỉ kiểm tra container có GPU trong khi model chạy CPU.

Giữ GPU hiện có làm baseline. Khi chưa cấu hình GPU, có thể chọn **một L4 làm cấu hình thử ban đầu** nếu tài khoản hỗ trợ; đây không phải kết luận L4 tối ưu nhất. Modal hỗ trợ chọn loại GPU qua tham số `gpu` [M1]. Không tự chuyển sang GPU khác giữa hai bản benchmark; ghi thiết bị thực tế đã được cấp.

Dùng một GPU, giới hạn container và số lượt chạy cho benchmark ban đầu. Chỉ thử nhiều video đồng thời khi có nhu cầu và sau khi đã đo một video. Phân biệt hai người cùng xem một video với hai video có hai pipeline phân tích.

### 6.2. Model sống theo container, state sống theo session

Với class-based serving, có thể dùng `@modal.enter()` để khởi tạo model một lần mỗi container [M2]. Không load weights hoặc build engine mỗi frame/mỗi thay đổi config.

Ưu tiên một lời gọi xử lý trọn video hoặc một phiên stream trong worker giữ state; không gọi remote riêng cho từng frame. Tracker, tracklet buffer và Common Path state phải tách theo camera/session, được khởi tạo và dọn đúng vòng đời.

**Không giả định một object proxy hoặc nhiều lời gọi method remote luôn vào cùng container.** Không dùng `set_config.remote()` rồi mặc nhiên cho rằng nó đã cập nhật đúng worker đang chạy video. Xác định cơ chế định tuyến/điều khiển session thực sự hoạt động: control channel của phiên hiện có hoặc store/queue nhỏ có namespace session và version, được worker đọc theo chu kỳ. Không thêm broker lớn nếu stack không cần.

Bắt đầu với một video đang xử lý tại một thời điểm trên mỗi worker. Nếu bật `@modal.concurrent`, lưu ý input đồng bộ có thể chạy trên các thread khác nhau; model và state phải an toàn với cơ chế đó [M3]. Không dùng một tracker chung cho nhiều input đồng thời.

### 6.3. I/O, lưu kết quả và chi phí

Tái sử dụng storage hiện có. Modal Volume có thể dùng cho weights và artifact; weights cũng có thể khởi tạo từ Volume rồi giữ trong RAM/VRAM [M4]. Không tải lại toàn bộ video/weights qua từng tick và không dùng Volume như kênh truyền frame.

Mỗi run có thư mục riêng; commit/persist output theo cơ chế đúng của storage đang dùng. Lưu metrics, snapshot và video minh họa cần thiết, không ghi crop/ảnh từng người hoặc toàn bộ frame debug vô hạn. Không ghi secrets/token vào báo cáo.

Đo cold start, model load và warm processing riêng. Không tạo public deployment hoặc giữ GPU warm vô thời hạn chỉ để hoàn thành benchmark. Giới hạn thời gian/số run, dừng khi lỗi lặp; không chạy sweep nhiều GPU hoặc toàn bộ tổ hợp tham số.

Nếu thiếu credential, video, quyền GPU hoặc dependency, vẫn hoàn thiện phần code và replay/test chạy được, kèm trạng thái `BLOCKED` chính xác cho phần Modal. Không bịa GPU metrics, không báo “đã benchmark” khi chỉ chạy CPU hoặc synthetic data.

## 7. Benchmark trước/sau: đo thuật toán, pipeline và hiển thị

### 7.1. Bộ chạy tối thiểu

Chọn video đại diện có sẵn, ưu tiên một đoạn 60–120 giây chứa luồng chính, đường rẽ hoặc giao cắt. Giữ nguyên video, timeline, ROI, detector, weights, precision và sampling giữa baseline/optimized.

Thực hiện ba lớp kiểm chứng:
1. **Replay cùng tracklet cache:** cô lập thuật toán, support, hình học và tính ổn định, không bị nhiễu bởi detection.
2. **Pipeline đầy đủ trên GPU Modal:** video → detection → tracking → Common Path → output, đo đầu-cuối.
3. **Realtime/UI:** kiểm tra preview, queue, thay đổi config và snapshot thực sự xuất hiện trên màn hình.

Lớp 1 phải so sánh cùng chuỗi tracklet; lớp 2 phải xác nhận refactor không làm thay đổi đầu vào detector ngoài chủ đích. Lớp 3 không được đánh đồng FPS server với FPS trình duyệt.

Warm up inference cùng shape mà không làm nhiễm tracker/Common Path của lượt đo; reset session trước khi bắt đầu video. Thời gian tích lũy để xuất hiện Common Path đầu tiên vẫn phải được ghi nhận, không che nó bằng warmup.

Nếu môi trường đủ, chạy baseline và optimized mỗi bản 3 lần trên cùng điều kiện, tổng mặc định tối đa 6 GPU runs. Ghi độ biến thiên; không chọn duy nhất lượt đẹp nhất. Test thuật toán nhỏ chạy bằng replay/CPU khi không cần GPU.

Nếu không có video phù hợp, dùng dữ liệu tổng hợp để kiểm tra logic nhưng phải ghi rõ; không dùng nó thay cho bằng chứng chất lượng trên video thật.

### 7.2. Các chỉ số bắt buộc

| Nhóm | Chỉ số và cách hiểu |
|---|---|
| Throughput | FPS inference và FPS pipeline tách riêng; FPS = số frame thực xử lý / thời gian đo tương ứng |
| Thời gian theo stage | Decode, preprocess, inference, postprocess, tracker, tracklet update, Common Path tick, render, encode: p50/p95 |
| Độ trễ pipeline | Từ nhận frame đến output sẵn sàng, có tính chờ queue; không cộng các p95 stage để gọi là p95 đầu-cuối |
| Common Path | Tick time p50/p95/max, số ứng viên được so sánh, số đoạn/tracklet, thời gian ra đường đầu tiên đủ điều kiện |
| Realtime | Queue depth, frame age, tỷ lệ frame bỏ; FPS preview thực hiển thị nếu có phép đo trình duyệt |
| Cấu hình | UI/config request → ack và → snapshot áp dụng đúng version, tách hai khoảng này |
| Tài nguyên | CPU, RAM RSS, GPU utilization, VRAM; kích thước và số lần eviction của buffer/cache |
| Khởi động | Cold start, load model, warm processing riêng |

GPU timing phải xét thực thi bất đồng bộ. Với PyTorch, dùng CUDA events hoặc đồng bộ đúng ở phép đo profiling; không chỉ đặt CPU timer quanh lời gọi inference [P1]. Không thêm synchronize mỗi frame vào production chỉ để tiện ghi log.

VRAM từ `nvidia-smi`/NVML và `torch.cuda.max_memory_allocated` không phải cùng một chỉ số; ghi riêng khi có. Modal cũng cung cấp GPU metrics, nhưng utilization cao/thấp một mình không chứng minh đã tìm đúng bottleneck [M5]. Dùng profiler khi cần, không chỉ xem dashboard GPU.

Không trừ trực tiếp timestamp trên máy client với timestamp trên Modal nếu đồng hồ chưa được đồng bộ. Đo trong cùng clock domain hoặc dùng frame/request ID với mốc gửi–nhận–render phía client. Video PTS không phải wall clock.

### 7.3. Định lượng đường “ổn định” nhưng vẫn đúng

Đánh giá trên snapshot theo thời gian, không chỉ một ảnh cuối:
- **Độ rung hình học:** sau matching cùng tuyến/cùng hướng và căn chỉnh phần chồng lấp, đo khoảng cách điểm-tới-đường giữa hai snapshot, chuẩn hóa theo đường chéo frame; báo p50/p95.
- **Độ đúng với bằng chứng:** độ lệch với tuyến tham chiếu hoặc tracklet hỗ trợ, tỷ lệ chiều dài có hỗ trợ; xem xét cùng độ rung để không thưởng cho đường cố định nhưng sai.
- **Đảo hướng:** số lần mũi tên đổi hướng vô lý khi luồng thực không đổi; phân biệt với luồng thay đổi thật.
- **ID/màu và xếp hạng:** số lần ID/màu của cùng tuyến đổi không cần thiết, số lần Top 1 đổi trong giai đoạn luồng ổn định.
- **Khả năng thích nghi:** thời gian tuyến mới trở thành active/Top 1 khi có đủ bằng chứng và thời gian tuyến cũ hết hạn.

Chỉ tính jitter trên các tuyến match được; báo riêng tỷ lệ matched/missing/new/expired để không làm đẹp số bằng cách bỏ qua các tuyến hay biến mất. Không khớp đường ngược chiều chỉ vì hình học giống.

Chưa có ground truth thì dùng replay tổng hợp có đáp án và kiểm tra một số đoạn video có gán nhãn thủ công. Ghi rõ metric nào tự động, metric nào dựa vào đánh giá thủ công; không công bố “accuracy” không có định nghĩa.

## 8. Test ngắn, tập trung đúng lỗi

Dùng vài test replay gọn, tận dụng fixture hiện có; không dựng bộ test đồ sộ:

| Tình huống | Kết quả cần thấy |
|---|---|
| Một tuyến đông và một tuyến ít người | Top 1 chọn tuyến có bằng chứng phổ biến hơn theo score đã công bố |
| Hai luồng ngược chiều cùng vị trí | Hai tuyến có hướng riêng, không triệt tiêu thành vector gần zero |
| Giao cắt hoặc rẽ chữ L | Không ghép nhánh không có người đi; giữ góc rẽ và mũi tên đúng |
| Tracklet bị chia, chồng lấp, gửi lặp hoặc mất frame | Không tăng support giả; không nối khoảng trống bất hợp lý |
| Người đứng yên hoặc video rỗng | Không sinh đường do bbox rung; đường cũ chuyển stale/expired đúng |
| Đổi Top 1→3→1, kéo slider liên tục | Không reset tracker; không đổi ID/màu vô cớ; config mới thắng snapshot cũ |
| Luồng chính đổi và bằng chứng cũ hết cửa sổ | Tuyến mới được nhận, tuyến cũ hết hạn; không đóng băng để giảm jitter |

Test thời gian dài cho TTL/bộ nhớ bằng replay event-time vượt ít nhất vài cửa sổ, không nhất thiết giữ GPU chạy lâu. Test race/config và lệch tọa độ trên snapshot hoặc frontend. Nếu hệ thống thực sự phục vụ nhiều session, thêm một kiểm tra tách state giữa hai session.

## 9. Mục tiêu nghiệm thu

Các giá trị sau là **mục tiêu thử ban đầu**, không phải số đo hoặc cam kết mọi video/GPU đều đạt. Báo điều kiện và kết quả thật:

- Common Path tick p95 nhỏ hơn chu kỳ cập nhật; hướng tới dưới 100 ms với workload đại diện đã ghi rõ. Không có backlog tick tăng liên tục.
- Đổi Top K/style dùng candidate cache, không gọi lại detector/tracker. Mục tiêu cập nhật cục bộ trong khoảng 200 ms; tách riêng network latency.
- Config thuật toán đã có đủ dữ liệu được áp dụng trong tối đa khoảng 2 chu kỳ cập nhật ở workload mục tiêu; yêu cầu rebuild dài hơn phải có trạng thái hiển thị rõ.
- Không có đảo hướng vô lý, đổi Path ID/màu hoặc đếm trùng trên các fixture xác định ở mục 8.
- Độ rung giảm so với baseline mà sai lệch tuyến/coverage và độ trễ thích nghi vẫn trong ngưỡng đã ghi trước khi tuning.
- RAM, VRAM, queue, candidate pool và tracklet buffer có giới hạn; eviction không bị che giấu.
- So sánh FPS/latency trên cùng input và phần cứng, kèm chất lượng tuyến. Chỉ kết luận tối ưu thành công khi có cải thiện đo được mà không phá correctness.

Nếu baseline đã nhanh hơn một mục tiêu, không làm nó chậm đi chỉ để đổi kiến trúc. Nếu chưa đạt, chỉ ra bottleneck còn lại và phần đã cải thiện; không tuyên bố hoàn thành toàn bộ.

## 10. Bàn giao

Sửa trực tiếp code, giữ patch nhỏ và có thể rollback. Tái sử dụng layout repo; chỉ tạo mới những thành phần chưa có.

Bàn giao tối thiểu:
- Code thuật toán/pipeline/UI đã sửa; cấu hình có đơn vị, default và validation.
- Entry point hoặc script benchmark chạy được trên Modal; lệnh thực tế đã kiểm tra, không để command giả.
- `metrics.json` và `summary.csv` hoặc định dạng có sẵn tương đương; metadata video/GPU/config/commit đầy đủ.
- `common_path_snapshots.jsonl` hoặc tương đương, video/ảnh minh họa cùng mốc thời gian trước/sau.
- Báo cáo ngắn: nguyên nhân có bằng chứng, file đã sửa/xóa, cách tính support, bảng trước/sau, chất lượng/độ ổn định, test đã chạy và giới hạn còn lại.

Báo cáo phải phân biệt `PASS`, `FAIL`, `NOT_RUN`, `BLOCKED`. Chưa đo được FPS UI thì ghi `NOT_RUN`, không lấy FPS inference thế vào. Chưa chạy GPU Modal thì không ghi benchmark Modal đã hoàn tất.

**Thực hiện theo thứ tự: đọc code và đo baseline → sửa thuật toán trên replay cố định → tách nhịp pipeline và cache render → hot config có version → benchmark Modal → kiểm tra UI và bàn giao.**

## Tài liệu chính thức để đối chiếu API

Các nguồn dưới đây chỉ hỗ trợ phần Modal/PyTorch; thiết kế thuật toán và các mục tiêu hiệu năng ở trên là yêu cầu triển khai cần kiểm chứng trên repository.

- [M1] Modal — GPU acceleration: `https://modal.com/docs/guide/gpu`
- [M2] Modal — Container lifecycle hooks: `https://modal.com/docs/guide/lifecycle-functions`
- [M3] Modal — Input concurrency: `https://modal.com/docs/guide/concurrent-inputs`
- [M4] Modal — Storing model weights: `https://modal.com/docs/guide/model-weights`
- [M5] Modal — GPU Metrics: `https://modal.com/docs/guide/gpu-metrics`
- [P1] PyTorch — CUDA semantics, asynchronous execution và timing: `https://docs.pytorch.org/docs/stable/notes/cuda.html`

Kiểm tra lại tài liệu tương ứng phiên bản SDK/backend đang dùng trước khi viết API cụ thể.
