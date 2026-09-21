# Prompt triển khai Common Path realtime bằng Directional Grid

## 1. Vai trò và nhiệm vụ

Bạn là Senior Computer Vision Engineer + Backend Engineer. Hãy sửa trực tiếp dự án Crowd Analysis hiện tại để triển khai pipeline:

**Point Tracklets → Directional Histogram theo Grid → Directed Edge Flow → Candidate Path → Tracklet Validation → Hysteresis → Active Common Path.**

Mục tiêu: phát hiện tuyến di chuyển phổ biến theo lưu lượng quan sát được, cập nhật online nhưng không nhảy đường liên tục. Video phải hiển thị quá trình xử lý, không đứng ở một ảnh tĩnh.

Các yêu cầu ưu tiên:

1. Tiếp tục trên project hiện tại; tái sử dụng detector, tracker, reader, API và UI. Không viết lại toàn bộ.
2. Tách Common Path thành engine độc lập, có chế độ `legacy`, `directional_grid` và `shadow`.
3. Mỗi người chỉ hiển thị bottom-center point; Track ID nhỏ là tùy chọn. Không vẽ bounding box, individual trajectory hoặc candidate lên video mặc định.
4. Vẽ Active Common Path đã được xác nhận trực tiếp vào frame trên server; frontend phát stream đã render. Grid/mũi tên từng ô chỉ xuất hiện trong debug output hoặc chế độ debug.
5. Khi test có inference video, bắt buộc dùng GPU trên Modal. Không âm thầm chạy YOLO bằng CPU/local GPU khi Modal chưa sẵn sàng.
6. Chạy clip ngắn, xuất artifact và kiểm tra trong thư mục output sau mỗi vòng; không mặc định chạy hết video hay soak test dài.
7. Unit test và replay analytics từ cache được chạy CPU local: chúng không chạy lại model và giúp tiết kiệm thời gian/GPU.
8. Không tạo benchmark giả, không kết luận realtime ổn định chỉ vì xuất được MP4.

Trong lần thay đổi này, ưu tiên engine Common Path và tích hợp tối thiểu. Không tự đổi model, training model mới, xây lại frontend, thêm Kafka hoặc tách microservice.

## 2. Khảo sát project trước khi sửa

Đọc `AGENTS.md` nếu có, `README.md`, dependency/lockfile, config, các module video, tracking, analytics, rendering, streaming và tích hợp Modal đang có.

Kiểm tra:

- Git status; giữ nguyên thay đổi của người dùng. Chỉ tạo branch nếu phù hợp workflow hiện có, không reset/checkout mất dữ liệu.
- Điểm nối giữa tracker và Common Path hiện tại; ai sở hữu state của từng camera.
- Video nguồn thực tế: tìm `data.mp4` và kiểm tra config; nếu project dùng `output.mp4`, xác minh đây là video nguồn chứ không phải video đã vẽ overlay.
- Chọn một đường dẫn input thống nhất, ghi vào manifest. Nếu thiếu/không rõ video nguồn, hỏi người dùng; không giả định đã đọc video.
- Modal entrypoint, môi trường/account được cấu hình, GPU, Volume, model weights và cách lấy artifact.
- UI hiện nhận stream liên tục hay chỉ một JPEG/PNG; xác định bằng code/log, không đoán nguyên nhân.
- Detector/tracker đang dùng tham số nào, có lọc mất low-confidence detections trước ByteTrack không. Không thay detector cùng lúc với analytics nếu chưa có lý do và số đo.

Báo cáo ngắn:

```text
CURRENT PIPELINE
REUSABLE MODULES
FILES TO CHANGE
INPUT VIDEO / SELECTED CLIP
MODAL TEST PLAN
RISKS / MISSING REQUIREMENTS
```

Sau đó triển khai từng milestone. Chỉ dừng hỏi khi có blocker thực sự: thiếu input, quyền Modal, thông tin cần quyết định hoặc vượt ngân sách test.

## 3. Kiến trúc và contract

Giữ một pipeline xử lý theo thứ tự thời gian cho mỗi camera/session. Inference, tracking, analytics và render nằm trong cùng server/job ở bản tích hợp này; không RPC lên Modal cho từng frame khi chạy test clip.

Common Path engine không được phụ thuộc trực tiếp YOLO, FastAPI hoặc frontend. Cấu trúc gợi ý, điều chỉnh theo project:

| Module | Trách nhiệm |
| --- | --- |
| `schemas.py` | TrackPoint, PathSnapshot, diagnostics |
| `tracklet_store.py` | History hữu hạn, smoothing, quality filter |
| `directional_grid.py` | Ánh xạ tọa độ, histogram nhiều hướng |
| `edge_flow.py` | Cạnh có hướng, unique support, time windows |
| `candidate_path.py` | Đề xuất số lượng hữu hạn tuyến ứng viên |
| `tracklet_validator.py` | Kiểm chứng tuyến bằng tracklet thật |
| `hysteresis.py` | Candidate/Active/Cooling/Retired |
| `directional_engine.py` | Điều phối các bước và xuất snapshot |

Contract tối thiểu:

```text
TrackPoint:
    camera_id: str
    stream_epoch: str
    track_id: int
    segment_id: int
    frame_id: int
    event_time_s: float
    x: float
    y: float
    confirmed: bool
    observed: bool

PathSnapshot:
    path_id: str
    revision: int
    state: str
    polyline: list[tuple[float, float]]
    coordinate_space: str
    support_tracks: int
    validated_complete_tracks: int
    score: float
    updated_at_s: float
    evidence_until_s: float
    stale: bool
```

Đây là schema mô tả, không phải code Python hoàn chỉnh. Hãy triển khai dataclass/Pydantic/type phù hợp.

Interface cần có `update(points, event_time_s)`, `tick(event_time_s)`, `snapshot()`, `reset(stream_epoch)`. `tick` vẫn chạy khi frame không có người để expire evidence và retire path.

- Khóa track phải chứa camera/session, không dùng riêng `track_id` toàn cục. Nếu phải cắt track vì gián đoạn bất thường, tăng `segment_id`.
- Unique track/tracklet chỉ là đại diện cho lượt quan sát, không bảo đảm là số người thật do ID switch/fragmentation.
- Với MP4 dùng PTS/media timestamp; dùng `frame_id / source_fps` chỉ khi nguồn constant-frame-rate và không có PTS đáng tin cậy.
- Dùng thời gian monotonic riêng để đo runtime/latency. Không dùng tốc độ chạy máy để quyết định cửa sổ lưu lượng.
- Reconnect, seek, loop video, đổi góc camera/ROI/calibration phải reset state hoặc tạo epoch mới.
- Chặn timestamp/frame đến sai thứ tự; worker analytics chỉ nhận snapshot bất biến có version và thời điểm evidence.

## 4. Bước 1 — Point Tracklets

Từ bbox nội bộ, lấy point trong hệ tọa độ frame gốc sau khi hoàn nguyên resize/letterbox:

```python
x = (x1 + x2) / 2
y = y2
```

Yêu cầu:

- Giữ bbox cho detector/tracker nội bộ; không vẽ rectangle.
- Chỉ dùng track confirmed, có quan sát thật và đủ chất lượng để tạo evidence lưu lượng. Điểm dự đoán khi bị che khuất không tự sinh lượt qua ô.
- Smoothing causal, chỉ dùng dữ liệu quá khứ. Có thể dùng EMA phụ thuộc delta-time; lưu cả point thô và point đã lọc trong debug.
- Tính hướng từ một khoảng lịch sử khoảng 0.3 giây, không chỉ lấy chênh lệch hai frame kề nhau.
- Ngưỡng dịch chuyển phù hợp kích thước ô/perspective; không hardcode một ngưỡng pixel cho mọi camera.
- Bỏ jitter và jump không hợp lý. Khi gap quá dài, không nối đường thẳng qua khoảng mất dấu.
- Không bỏ thông tin timestamp/occupancy chỉ vì người đứng yên; chỉ loại đóng góp vào movement flow.
- Giới hạn lịch sử point, tracklet hoàn thành và tập dedup bằng TTL/cap. Giữ chuỗi cell đã nén đủ để validate tuyến. Nếu chạm cap, log việc mất evidence và giảm độ tin cậy; không âm thầm gọi số liệu bị cắt là thống kê đầy đủ.

Không tự nối hai Track ID khác nhau để làm đẹp kết quả. Nếu chưa có logic nối track được kiểm chứng, báo rõ khả năng thiếu support do fragmentation.

## 5. Bước 2 — Directional Histogram theo Grid

Baseline: grid 32 cột × 18 hàng, 8 hướng, chỉ xử lý trong ROI hợp lệ. Camera cố định; có homography thì dùng ground plane, chưa có thì dùng pixel-space và ghi rõ giới hạn phối cảnh.

Mỗi cell lưu:

- Histogram hướng trong cửa sổ ngắn và dài.
- Unique track support; tỷ lệ các mode hướng, support của từng mode.
- Occupancy riêng, không gộp với movement.
- Thời điểm evidence gần nhất; trạng thái insufficient_data nếu ít dữ liệu.

Quy ước hình ảnh: x tăng sang phải, y tăng xuống dưới. Test mapping E/S/W/N, các đường chéo và wrap-around góc.

Định nghĩa baseline để tránh thiên lệch FPS/người đứng lâu:

```text
H(cell, direction, W)
= số track_key khác nhau có chuyển động hợp lệ
  trong cell theo direction trong W giây gần nhất.
```

Một track có thể đóng góp cho nhiều hướng nếu thực sự đổi hướng; tổng các bin KHÔNG phải số người duy nhất trong cell.

Không cộng 1 mỗi frame. Lưu timestamp evidence theo khóa để dedup trong toàn cửa sổ; expire cả cell không còn người.

Không lấy trung bình tất cả vector rồi coi đó là một hướng duy nhất. Giữ ít nhất hai mode khi có hai luồng ngược chiều/giao nhau; nếu bằng nhau thì biểu diễn đa hướng, không chọn ngẫu nhiên.

## 6. Bước 3 — Directed Edge Flow

Mỗi cell là node; chuyển động được xác nhận từ A sang B là cạnh có hướng. A→B và B→A là hai cạnh riêng.

Yêu cầu:

1. Xác nhận đổi cell bằng spatial hysteresis/debounce để tránh rung tại biên. Không tạo lượt đi-về giả do jitter.
2. Lưu `edge → track_key → last_crossing_event_time`; distinct support được tính trong toàn cửa sổ.
3. Nếu chia thành bucket để tối ưu, phải hợp nhất khóa hoặc dùng reference count. Không cộng số distinct của các bucket rồi gọi đó là unique tracks.
4. Phân biệt unique-track support với passage count khi một người đi lại nhiều lần. Baseline dùng unique-track support.
5. Cho phép trace các cell trung gian chỉ với đoạn chuyển động ngắn, liên tục, hợp lý. Đánh dấu phần nội suy; không dựng edge qua gap lớn, ngoài ROI hoặc xuyên vật cản.
6. Ưu tiên sparse edges giữa các cell lân cận, không cấp phát ma trận mọi cặp cell.

Dùng rolling window 30 giây và 180 giây làm baseline. So sánh trên cùng đơn vị:

```text
rate_short = unique_support_short / observed_short_window_seconds
rate_long  = unique_support_long  / observed_long_window_seconds
edge_score = 0.65 * rate_short + 0.35 * rate_long
```

Trong warm-up, mẫu số là phần thời gian thật sự đã quan sát, luôn có chặn tránh chia 0. Đây là chỉ số support theo thời gian, không cam kết lưu lượng người thật tuyệt đối.

Không trộn trực tiếp count 30 giây với count 180 giây. Chưa cần thêm exponential decay lên sliding window trong phiên bản đầu để tránh khó đánh giá độ trễ phản hồi.

## 7. Bước 4 — Candidate Path

Tạo ứng viên từ đồ thị có hướng và histogram:

- Ưu tiên entrance/exit zone hiện có, giữ thứ tự hướng đi.
- Nếu chưa có zone, dùng điểm bắt đầu/kết thúc tracklet quan sát được ở biên ROI. Không đặt tên semantic giả.
- Giữ nhiều mode qua ô giao nhau. Có thể dùng trạng thái `(previous_cell, current_cell)` hoặc hướng vào để không đánh mất thông tin rẽ.
- Lọc cạnh ít support; ưu tiên continuity và lưu lượng; không chỉ nối mũi tên lớn nhất từng ô.
- Dùng tìm kiếm giới hạn như bounded beam search; giới hạn candidates, path length và thời gian tính. Không enumerate mọi đường.
- Không cho vòng lặp vô hạn. Không tự bắc cầu qua vùng không có evidence.
- Nếu dùng chi phí `-log(P(next_cell | current_cell))`, kết hợp support và penalty rẽ; lưu ý bias theo độ dài. Không coi shortest path là tuyến đông nhất.
- So sánh/rank các route theo O/D, chiều đi và support đã validate. Hai đường cùng O/D nhưng vòng hai phía vật cản là hai route khác nhau.

Candidate chỉ là giả thuyết, chưa được render lên luồng video mặc định.

## 8. Bước 5 — Tracklet Validation

Đối chiếu từng candidate với chuỗi cell của tracklet thật trong cửa sổ evidence:

- Cùng chiều và thứ tự cell; cho tolerance không gian nhỏ, có cấu hình.
- Không dùng khoảng cách hình học đơn thuần để nhận nhầm người đi ngược chiều.
- Tính coverage theo độ dài tuyến/thứ tự, không theo số point vì FPS và tốc độ người có thể khác nhau.
- Baseline partial coverage khoảng 70% để đánh giá hỗ trợ hình học; đồng thời ghi số track nối đủ entrance→exit.
- Mỗi track chỉ tính một lần cho mỗi candidate trong cửa sổ.
- Không cộng support của nhiều đoạn rời rạc rồi tuyên bố có một track đã đi toàn tuyến.

Phân biệt rõ:

```text
support_tracks: tracklets ủng hộ phần lớn hình học tuyến.
validated_complete_tracks: tracklets có bằng chứng đi liên tục từ đầu tới cuối.
```

Chỉ xác nhận một route vào–ra hoàn chỉnh khi có số track hoàn chỉnh tối thiểu theo config. Nếu chỉ có bằng chứng từng đoạn, trả về partial corridor/candidate và báo insufficient_data; không bịa ra tuyến hoàn chỉnh.

Nếu tracking thiếu support, giữ trạng thái chưa đủ dữ liệu và nêu hạn chế, không tự hạ ngưỡng chỉ để có path trên video.

## 9. Bước 6 — Hysteresis và Active Common Path

Triển khai state machine:

- `Candidate`: có tuyến ứng viên đang tích lũy support.
- `Active`: đã đạt support tối thiểu và thời gian xác nhận.
- `Cooling`: support giảm hoặc evidence cũ; tạm giữ hình học, đánh dấu stale.
- `Retired`: hết thời gian giữ hoặc không còn hợp lệ; ngừng vẽ.

Quy tắc:

1. Candidate đầu tiên cần đủ support và confirmation time trước khi Active.
2. Challenger thay Active khi score vượt khoảng 20%, vượt thêm absolute margin có cấu hình và giữ điều kiện đủ lâu.
3. Dùng event time và evidence mới; gọi `tick` nhiều lần trên cùng dữ liệu cũ không được tính thành nhiều lần xác nhận độc lập.
4. Reset bộ đếm xác nhận khi điều kiện bị phá hoặc challenger thay đổi.
5. Matching path giữa hai lần extract dựa trên O/D, direction và hình học. Cùng tuyến rung nhẹ phải giữ `path_id`.
6. Làm mượt hình học chỉ giữa các revision của cùng route; không nội suy tuyến cũ sang một nhánh khác khiến đường cắt vật cản.
7. Active vẫn được đánh giá lại nếu không còn là top candidate; score cũ không được giữ mãi.
8. `tick` phải làm hết hạn path khi cảnh trống; camera mất kết nối được báo stale/disconnected bằng watchdog riêng.

Baseline: cập nhật graph 1 giây; extract candidate 3 giây; confirmation 8 giây; cooling 20 giây. Đây là tham số thử, không phải kết quả đã tối ưu.

Đo riêng độ ổn định hình học, số lần đổi route và thời gian phản ứng khi luồng thật đổi. Path không đổi bao giờ cũng không được coi là tốt.

## 10. Tích hợp render và UI đang có

Render trên server theo frame packet:

- Point/detection/tracking phải thuộc đúng frame đang render; giữ `frame_id` xuyên pipeline.
- Common Path là thống kê của cửa sổ quá khứ: dùng snapshot mới nhất hợp lệ với `evidence_until_s <= frame.event_time_s`, không đợi tính lại path trên mỗi frame.
- Đọc snapshot bất biến; không để thread render đọc polyline đang bị analytics sửa dở.
- Mỗi frame đầu ra đều vẽ lại Active/Cooling Path phù hợp và point hiện tại.
- Không vẽ bbox, trajectory cá nhân, grid debug hoặc candidate mặc định.
- Nếu Common Path chưa có, video/point vẫn phải tiếp tục cập nhật, kèm trạng thái đang tích lũy dữ liệu.
- Luồng video và số liệu dashboard có thể đi riêng, nhưng không ghép tọa độ của frame cũ vào video mới trên frontend.

Tái sử dụng transport hiện có. Nếu hiện chỉ có endpoint ảnh tĩnh, sửa tối thiểu thành stream liên tục; có thể dùng MJPEG cho smoke test. Không bắt buộc chuyển WebRTC trong cùng đợt nếu chưa cần.

Một processing worker/camera được chia sẻ cho các client; không khởi tạo YOLO theo mỗi request xem video. Queue hữu hạn, disconnect/EOF có xử lý; client chậm không block capture.

Phân biệt hai mode:

- `offline_fast`: đọc clip tuần tự nhanh nhất có thể, không sleep theo FPS, không drop tùy tiện; analytics vẫn dùng media timestamp.
- `realtime`: reader được pace theo nguồn/live, queue hữu hạn và loại frame cũ khi quá tải; tracker xử lý gap đúng, không tự đẻ evidence từ prediction.

Không dùng latest-frame reader đọc MP4 không pace trong test offline: reader có thể chạy tới EOF trước khi inference xử lý đủ mẫu.

## 11. Test trên Modal: ngắn, có giới hạn, tái sử dụng cache

### 11.1. Chuẩn bị Modal

- Tái sử dụng cấu hình/account/Volume của project. Không đọc hay in token/`.env` vào log; không bake credentials vào image.
- Mọi lần chạy detector trên video test phải nằm trong Modal GPU job. GPU test mặc định một L4 nếu project chưa cấu hình GPU; nếu có thì giữ cấu hình đã được chấp thuận. Không tự nâng lên GPU đắt hơn.
- Kiểm tra CUDA và tên GPU ở remote worker trước inference. Nếu không có CUDA, auth/quota/billing/network lỗi, fail-fast và báo blocker; không fallback CPU/local inference.
- Chỉ đưa source cần thiết vào image. Input clip, weights và cache lưu riêng; không upload cả repo, dữ liệu cá nhân hoặc mọi video.
- Load model một lần mỗi container; nếu dùng class runner thì dùng lifecycle hook theo SDK đang cài. State tracker/analytics phải reset theo run, không lẫn giữa hai clip.
- Có input hash, model hash và version dependencies; cache tránh upload/tải weights lặp.
- Dùng ephemeral `modal run` cho test, không tự deploy endpoint công khai hay chạy service GPU vô hạn.
- Giới hạn một container, một job tại một thời điểm, không retry vô hạn. Có timeout và giới hạn tổng budget; không dùng `--detach` mặc định.

Modal có cấu hình GPU qua `gpu`, lifecycle hook `@modal.enter()` và ephemeral app qua `modal run`. Kiểm tra lại SDK/API trong project trước khi viết runner. [GPU](https://modal.com/docs/guide/gpu), [lifecycle](https://modal.com/docs/guide/lifecycle-functions), [ephemeral apps](https://modal.com/docs/guide/apps).

### 11.2. Runner bắt buộc

Tạo hoặc mở rộng entrypoint có các tùy chọn tương đương:

```text
--input
--start-seconds
--duration-seconds
--engine legacy|directional_grid|shadow
--mode offline_fast|realtime
--config
--run-id
--cache-policy reuse|refresh
```

Runner thực hiện trọn clip trong remote job: decode → GPU inference → tracking → analytics → render/encode → lưu artifact. Chỉ trả manifest nhỏ, không trả tất cả frame qua RPC.

Trước khi return:

1. Đóng video writer/file handle.
2. Kiểm tra sơ bộ artifact và lưu trạng thái run `success/failed/partial`.
3. Commit output Volume rõ ràng để tải được kết quả.
4. Local runner tải đúng thư mục run, không tải cả Volume.
5. Chỉ báo thành công khi artifact cần thiết đã có ở local output và đã được kiểm tra.

Modal Volume hỗ trợ lưu file, commit/reload và CLI lấy file. Dùng một writer cho thư mục run; `reload` trước khi đọc thay đổi từ worker khác, không reload khi đang mở file. [Volumes](https://modal.com/docs/guide/volumes), [Volume CLI](https://modal.com/docs/cli/latest/volume).

### 11.3. Chính sách test nhanh

| Vòng | Nơi chạy | Phạm vi | Điều kiện đi tiếp |
| --- | --- | --- | --- |
| Unit | CPU local | Tracklet tổng hợp có timestamp giả lập, không inference | Logic và invariant pass |
| Smoke GPU | Modal GPU | Một clip 20–30 giây có người di chuyển | Có detections, tracklets, video động và artifact hợp lệ |
| Analytics tuning | CPU replay | Cache thật từ smoke; tối đa 3 cấu hình mỗi vòng | Sửa đúng lỗi quan sát được |
| Integration GPU | Modal GPU | Một clip 60–90 giây giàu chuyển động | Đủ kiểm chứng path khi dữ liệu cho phép |
| UI smoke | Pipeline hiện có, inference vẫn ở Modal nếu chạy model | Xem stream khoảng 10–15 giây | Frame/point cập nhật và không lệch overlay |

Giới hạn mặc định: tối đa 3 GPU job tổng cộng trong một lượt triển khai, kể cả baseline, retry và UI job có inference; timeout mỗi job 600 giây, tổng ngân sách compute được đặt trước không vượt 15 phút. Chỉ chạy job mới khi còn budget; không dùng timeout từng job để lách tổng giới hạn. Build/cold start/transfer phải đo riêng. Nếu cần thêm GPU job hoặc test dài hơn, báo lý do và xin người dùng quyết định.

- Smoke 20–30 giây chỉ kiểm tra plumbing, không đủ xác nhận cửa sổ 180 giây đã ổn định.
- Unit test dùng đồng hồ mô phỏng để kiểm tra decay/window/hysteresis nhiều phút mà không sleep nhiều phút.
- Không giảm `confirmation_seconds` trong production config chỉ để smoke có path nhanh.
- Nếu clip ít người/chưa đủ support, xuất `insufficient_data`; chọn đoạn có evidence phù hợp trong budget, không cố tạo Active Path.
- Không tự chạy toàn bộ dataset, full-video benchmark hoặc soak test 30 phút trong vòng sửa.

### 11.4. Cache để tránh chạy lại YOLO

Lần GPU đầu lưu detections và point tracklets theo frame/timestamp:

- Chỉ đổi grid/path/validation/hysteresis: replay cache tracklets.
- Đổi tracker: replay cache detections rồi dựng lại tracklets.
- Đổi smoothing/point preprocessing: dựng lại point tracklets từ bbox/points thô đã lưu; không tái sử dụng cache đã lọc bằng config cũ.
- Đổi weights, detector params, image preprocessing, input clip/sampling: vô hiệu hóa cache liên quan và chạy Modal GPU lại nếu còn budget.
- Cache key bao gồm input/clip/hash, model hash, inference config, tracker config, point preprocessing và schema/coordinate transform version tương ứng.
- Cả `legacy` và `directional_grid` nhận cùng một đầu ra tracking; không chạy hai detector trong shadow mode.
- Replay nhanh phải giữ nguyên event timestamps, không nén thời gian phân tích theo tốc độ chạy máy.
- Ghi rõ kết quả là `analytics_replay`, `gpu_pipeline` hay `live_ui`. Không gộp FPS của replay với FPS end-to-end.

Nếu chỉ sửa analytics, ưu tiên replay và render lại đoạn preview từ video/cache, không khởi động GPU chỉ để vẽ đường.

## 12. Output và kiểm tra bắt buộc sau mỗi vòng

Tái sử dụng thư mục `output/` hoặc `outputs/` đã có. Nếu chưa có, dùng `outputs/common_path/<run_id>/`. Không ghi đè run cũ; mỗi run có ID riêng.

Các artifact:

| File | Nội dung |
| --- | --- |
| `manifest.json` | Run ID, loại test, source hash, clip, GPU, version, config/cache key, status, danh sách artifact |
| `config_resolved.yaml` | Config đã resolve, không chứa secret |
| `tracked_points_common_path.mp4` | Video động: point và Active/Cooling Path, không bbox/tail |
| `preview_contact_sheet.jpg` | 6–9 thời điểm đầu/giữa/cuối, ghi timestamp/frame ID |
| `directional_field_debug.png` | Grid và các mode hướng, support |
| `edge_flow_debug.png` | Cạnh có hướng và support |
| `common_path_map.png` | Tuyến Active hoặc chú thích chưa đủ dữ liệu |
| `path_events.jsonl` | Candidate/activate/switch/cooling/retire, lý do và evidence time |
| `summary.json` | Kết quả, lỗi, evidence, giới hạn chưa kiểm thử |
| `metrics.csv` | Timing/stage, track counts, bộ nhớ, queue/drop nếu có |
| `inspection.md` | Agent đã mở artifact nào, thấy gì, cần sửa gì |

Cache detections/tracklets có thể nằm trong thư mục cache riêng để không sao chép giữa các run. Ghi đường dẫn/hash trong manifest.

Không coi file tồn tại hoặc log `success` là đủ. Agent phải:

1. Kiểm tra metadata video bằng công cụ như ffprobe: codec, kích thước, số frame/thời lượng nếu có; thử decode đầu/giữa/cuối.
2. Mở contact sheet và các ảnh debug, xem đoạn preview ngắn nếu công cụ hỗ trợ; ghi nhận điểm bám người, hướng, path và frame thay đổi.
3. Đối chiếu metadata frame, pixel/frame change và chuyển động thực trong nguồn; không yêu cầu ảnh khác nhau khi nguồn vốn đứng yên.
4. Xác minh path không xuyên vùng cấm, không ghép hai nhóm rời rạc và không mất/hồi mỗi vài frame.
5. Mở `path_events.jsonl` và `summary.json`, kiểm tra transition đúng ngưỡng/đúng thời gian.
6. Ghi vài nhận xét có timestamp hoặc tên artifact vào `inspection.md`.

Nếu không có công cụ xem ảnh/video, báo rõ giới hạn kiểm tra, vẫn kiểm tra decode/schema nhưng không tuyên bố đã xác minh bằng mắt.

MP4 đúng không chứng minh UI stream đúng. Cuối cùng cần smoke test UI ngắn: xem frame ID tăng, stream có nhiều frame được decode, point thay đổi theo frame và video vẫn chạy khi chưa có Common Path. Nếu bị chặn bởi auth/network/browser, báo `UI_NOT_VERIFIED` và không báo hoàn thành UI.

## 13. Config baseline

Đây là config của ứng dụng cần triển khai, không phải các keyword có sẵn của Modal/ByteTrack. Hãy validate và map đúng sang SDK, không copy tham số không được hỗ trợ.

```yaml
common_path:
  engine: directional_grid
  shadow_display: legacy
  max_active_paths: 1

directional_grid:
  columns: 32
  rows: 18
  direction_bins: 8
  coordinate_space: image_pixels
  direction_sample_seconds: 0.3
  min_track_age_seconds: 0.5
  min_displacement_cell_fraction: 0.15
  max_observation_gap_seconds: 0.5
  observed_points_only: true

flow:
  short_window_seconds: 30
  long_window_seconds: 180
  short_weight: 0.65
  long_weight: 0.35
  update_interval_seconds: 1
  dedup_scope: entire_window

candidate:
  update_interval_seconds: 3
  max_candidates: 5
  beam_width: 20
  max_path_cells: 128
  min_path_cells: 4
  min_edge_unique_tracks: 3

validation:
  min_support_tracks: 5
  min_complete_tracks: 3
  min_ordered_coverage: 0.70
  cell_tolerance: 1
  require_complete_evidence_for_od_path: true

hysteresis:
  confirmation_seconds: 8
  switch_relative_margin: 0.20
  cooling_seconds: 20
  # Bổ sung absolute margin sau khi định nghĩa thang score.

render:
  draw_boxes: false
  draw_points: true
  draw_track_ids: false
  draw_individual_trajectories: false
  draw_candidates: false
  draw_active_common_path: true
  draw_grid_debug: false

test:
  inference_provider: modal
  allow_local_inference: false
  mode: offline_fast
  smoke_duration_seconds: 25
  integration_duration_seconds: 75
  max_gpu_jobs: 3
  total_compute_budget_seconds: 900
  job_timeout_seconds: 600
  cache_policy: reuse
  inspect_artifacts: true
```

Nếu grid/threshold không phù hợp video, giải thích và sửa một nhóm tham số mỗi vòng. Không chạy quét tổ hợp detector × tracker × grid × hysteresis.

## 14. Milestone và test cases

Làm tuần tự, mỗi bước có test tương ứng:

1. Khảo sát + baseline/Modal runner + cache/output inspection.
2. Contract TrackPoint + tracklet store + clock/reset.
3. Directional histogram + directed edges + window dedup.
4. Candidate extraction + tracklet validation.
5. State machine + path identity/geometry stability.
6. Shadow replay so sánh với legacy trên cùng dữ liệu.
7. Nối renderer/stream; GPU integration clip ngắn và UI smoke.
8. README/config/rollback; bàn giao artifact và phần chưa kiểm thử.

Unit tests bắt buộc:

- Một luồng A→B, đổi A→C, hai luồng ngược chiều cân bằng và giao nhau.
- Người đứng yên/jitter không tạo movement/path giả.
- Một track xuất hiện ở nhiều frame/bucket vẫn chỉ là một unique support trong cửa sổ.
- Evidence hết hạn; cảnh trống; timestamp nhảy lùi; reconnect/epoch mới.
- Gap/ID switch không tạo cạnh teleport; đoạn nội suy không đi qua vùng cấm.
- Những edge riêng lẻ có support lớn nhưng không có track đi toàn tuyến phải bị validation chặn.
- Cùng route thay đổi polyline nhẹ giữ path ID; challenger chập chờn không được switch.
- Challenger mạnh và bền phải switch; Active cũ mất evidence phải Cooling rồi Retired.
- Replay một dữ liệu ở tốc độ khác nhau tạo cùng sự kiện theo media timestamp.
- RAM/history/sets được giới hạn; stream tiếp tục khi path chưa xuất hiện; slow client không block pipeline.

Dữ liệu tổng hợp chỉ dùng chứng minh logic unit test, phải gắn nhãn synthetic. Không dùng nó thay benchmark/video thật.

Metrics cần báo: số track, validated route support, switches/minute trong đoạn luồng ổn định, độ rung polyline cùng route, change-detection delay, analytics p50/p95, processing FPS, decode/inference/tracking/render/encode timing, RAM/VRAM peak.

Đo inference GPU bằng CUDA events hoặc synchronization phù hợp vì kernel có thể chạy bất đồng bộ. Tách model warm-up, build/cold start, transfer và xử lý steady-state. Chỉ gọi là end-to-end latency khi đã đo đúng từ nguồn tới điểm hiển thị; offline throughput không đại diện network/UI latency.

Baseline và engine mới phải dùng cùng clip/input/config tracking. Tách overhead của shadow mode khỏi overhead engine mới. Không có số đo thì ghi N/A và lý do.

## 15. Acceptance criteria và báo cáo

Chỉ đánh dấu phần tương ứng hoàn thành khi có bằng chứng:

- Engine mới chạy đúng 7 bước, module độc lập; rollback về legacy được.
- Evidence là unique track support theo cửa sổ, giữ được nhiều hướng, không suy luận route hoàn chỉnh từ các đoạn rời rạc.
- Path đủ ổn định trên luồng ổn định nhưng vẫn đổi khi evidence chuyển luồng đủ mạnh.
- Tất cả inference video test đã chạy trên Modal GPU, có device/run metadata; không fallback local.
- Có artifact thật và `inspection.md`; video decode được và có tiến trình xử lý.
- UI stream được kiểm tra riêng; không hiện bbox/trajectory cá nhân trong mode mặc định.
- Không giữ history/queue vô hạn. Test ngắn không được coi là bằng chứng ổn định nhiều giờ.
- Unit tests pass; lỗi và bước chưa chạy được báo rõ.

Sau mỗi milestone báo ngắn: `CHANGED / TESTED / OUTPUT REVIEW / ISSUES / NEXT`.

Báo cáo cuối:

```text
IMPLEMENTED
- Module và file đã sửa
- Cách bật directional_grid / shadow / rollback

MODAL TESTS
- Input, clip, GPU thực tế, số job và thời gian
- Test nào chạy mới, test nào replay cache

OUTPUT REVIEW
- Đường dẫn local tới preview MP4, contact sheet, debug field và summary
- Nhận xét có evidence/timestamp

RESULTS
- So sánh legacy và directional_grid bằng số đo thật
- Support / switches / response delay / FPS / analytics p95 / memory

NOT VERIFIED / LIMITATIONS
- UI hoặc trường hợp dữ liệu chưa đủ, long-window chưa đầy, chưa soak test

HOW TO RUN
- Lệnh thực tế đã chạy cho unit test, Modal smoke, replay và tải output
- Không đưa lệnh ví dụ chưa triển khai rồi nói đã kiểm thử
```

**Bắt đầu bằng khảo sát project và chốt một clip ngắn. Thiết lập Modal GPU runner + cache + output inspection trước vòng benchmark đầu tiên; sau đó triển khai từng bước. Mỗi vòng sửa phải đọc/xem output hiện có trước khi quyết định chạy GPU lần tiếp theo.**
