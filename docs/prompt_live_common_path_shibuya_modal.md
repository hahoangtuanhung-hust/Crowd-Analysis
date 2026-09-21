# Prompt: xem Common Path theo thời gian ngay khi Modal xử lý Shibuya

## 1. Mục tiêu của lượt triển khai này

Bạn là Senior Computer Vision Engineer + Realtime Backend Engineer. Tiếp tục sửa project Crowd Analysis hiện tại dựa trên `diagnostic_report.md`.

**Khi Modal đang xử lý `data-shibuya-test.mp4`, người dùng phải xem được video đang xử lý và Common Path tương ứng với thời điểm hiện tại trên UI, không phải chờ xử lý xong rồi tải MP4 về phát lại.**

Giữ pipeline:

Point Tracklets → Directional Histogram theo Grid → Directed Edge Flow → Candidate Path → Tracklet Validation → Hysteresis → Active Common Path.

Yêu cầu hiển thị:

- Người được biểu diễn bằng bottom-center point, không bbox, không trajectory riêng.
- Active Common Path được render trên server lên frame đang xử lý.
- Common Path được cập nhật từ dữ liệu đã quan sát tới thời điểm frame đó, không dùng kết quả cuối clip để vẽ ngược lên đầu clip.
- Khi chưa đủ bằng chứng, video vẫn chạy và UI ghi trạng thái đang tích lũy/chưa đủ dữ liệu. Không tạo đường giả để lấp khoảng trống.
- Giữ chế độ legacy, replay và khả năng rollback; không viết lại project.

## 2. Những điều báo cáo hiện tại đã và chưa chứng minh

Đọc và xác minh lại các kết luận sau trong source/artifact:

- Báo cáo dùng `data/videos/data-test.mp4`, corridor Grand Central, không phải Shibuya.
- Global có 1 route/7 evaluations và local có 9 routes/10 evaluations; tất cả bị loại, không có Active Path.
- 2.954 tracker IDs đã bị chia thành 11.897 track segments. Median observed span 0.2667 giây; 83.26% segments dưới 1 giây.
- Có 14 S→T và 10 T→S gate-complete segments nhưng candidate support/complete vẫn bằng 0. Điều này cần truy vết; chưa đủ để khẳng định validator sai vì các tuyến có thể khác nhau.
- Short window đã sẵn sàng sau 30 giây, không bị chặn tới 180 giây.
- Modal lần trước chỉ replay cache; detector_calls=0, các stage chạy CPU trong container T4.
- UI đã kiểm chứng local replay; chưa kiểm chứng stream kết quả trực tiếp từ Modal tới trình duyệt.

Không tái sử dụng ROI/gates/calibration hoặc tracking cache Grand Central cho Shibuya. Không coi việc đã kiểm chứng local replay là hoàn thành yêu cầu mới.

## 3. Phạm vi inference mới và test

Lượt này cho phép chạy inference thật bằng GPU Modal trên đúng video Shibuya để kiểm chứng luồng mới. Quy tắc “detector_calls=0” của prompt chẩn đoán trước chỉ còn áp dụng cho replay; không áp dụng cho session live_inference mới.

Phân biệt:

- `live_inference`: đọc Shibuya, YOLO thực sự chạy trên CUDA trong Modal, tracking/analytics/render cùng session, UI nhận frame trong lúc job đang chạy.
- `remote_cache_replay`: dùng cache đúng video/config trên Modal để sửa/test nhanh; UI ghi rõ replay.
- `local_replay`: phát local để debug; không dùng làm bằng chứng hoàn thành Modal streaming.

Ngân sách mặc định cho lượt này: tối đa 2 processing sessions có GPU, gồm một smoke 15–20 giây và một integration tối đa 120 giây nội dung nguồn hoặc tới EOF nếu ngắn hơn. Timeout mỗi session tối đa 600 giây, tổng compute budget tối đa 900 giây. Retry tính vào số session. Dừng worker/serve process test khi xong; không để GPU chạy vô hạn.

Integration phải mở UI ngay từ đầu và quan sát qua giai đoạn tích lũy, không chỉ xem 10 giây đầu rồi kết luận. Chỉ mở thêm session khi còn ngân sách; cần chạy thêm thì báo lý do cụ thể.

Sau lần inference đầu, lưu cache Shibuya để mọi lần sửa analytics/render tiếp theo không phải gọi YOLO lại. Nếu thiếu file, quyền Modal hoặc GPU, báo blocker; không âm thầm thay video hay fallback inference local.

## 4. Khảo sát và xác minh đúng dữ liệu

Đọc AGENTS.md nếu có, README, diagnostics, Modal runner, analytics, SessionManager, renderer, streamer và frontend player hiện tại.

Trước khi sửa:

1. Giữ worktree của người dùng và rollback engine cũ.
2. Tìm đúng `data-shibuya-test.mp4`; ghi path, hash, thời lượng, FPS/PTS, kích thước và frame đầu.
3. Ghi config riêng cho Shibuya. Chọn ROI/gates trên ảnh thật, lưu minh họa; không sao chép các tọa độ Grand Central.
4. Kiểm tra decoder, timestamp, class filter, detector confidence và output coordinate transform.
5. Tách reader MP4 và reader cache rõ ràng, tránh nhận sai mode.
6. Kiểm tra browser có thể truy cập endpoint Modal ngay trong preflight; không đợi dùng hết GPU budget mới phát hiện thiếu khả năng mở UI.

Nếu video/camera bị rung hoặc thay đổi góc, ghi rõ giới hạn grid cố định; không ghép evidence ở hai hệ tọa độ khác nhau. Không đổi detector/tracker hàng loạt trước khi có số đo.

## 5. Streaming trong lúc xử lý

Thiết kế một pipeline stateful cho mỗi session:

```text
Shibuya trên Modal
→ Decode → YOLO/CUDA → ByteTrack → Point/Directional Grid
→ Snapshot Common Path theo media time
→ Render point + path vào chính frame đó
→ Publish frame ngay → UI đang mở
```

Writer MP4 và snapshot timeline chạy như output phụ; không được chặn publish stream cho tới EOF/volume commit.

Ưu tiên tái sử dụng transport đã hoạt động nếu chứng minh frame đi qua Modal tới browser liên tục. Nếu MJPEG qua proxy bị buffer, triển khai WebSocket binary JPEG cho test:

- Một FastAPI ASGI app trên Modal phục vụ WebSocket và session control.
- Một pipeline owner/camera; tách processing worker khỏi event loop gửi/nhận.
- State tracker/analytics nằm cùng owner trong suốt session. Không gọi một remote function mới cho từng frame.
- Một serving container trong đợt test và một pipeline xử lý video tại một thời điểm. Cho phép control/stream đồng thời nhưng khóa start để không tạo hai worker.
- Cùng WebSocket có thể nhận lệnh start/stop và trả event/frame, tránh request start ở container này nhưng stream ở container khác.
- Nếu dùng control HTTP riêng, bảo đảm requests đi tới cùng session owner; không giả định global dict chia sẻ giữa các process/container.
- Hai tab xem cùng session không được tự sinh hai detector jobs; nếu MVP chỉ hỗ trợ một viewer, trả thông báo rõ.
- Frontend phải nhận frame mới trước khi job hoàn tất. Không polling file MP4 đang viết hoặc chờ Modal Volume commit làm live transport.

Modal hỗ trợ ASGI/WebSocket; WebSocket giữ một function call cho mỗi connection. Kiểm tra giới hạn message và cấu hình concurrency theo SDK hiện dùng; tài liệu hiện nêu tối đa 2 MiB/message. Đặt giới hạn payload thấp hơn và kiểm tra trước khi gửi. [Modal Web Functions](https://modal.com/docs/guide/webhooks)

Nếu dùng HTTP streaming, kiểm tra buffering qua proxy và giới hạn request thực tế; không mặc định JPEG qua HTTP sẽ được giao realtime. Không áp nguyên giới hạn HTTP cho WebSocket. [Streaming endpoints](https://modal.com/docs/guide/streaming-endpoints), [Request timeouts](https://modal.com/docs/guide/webhook-timeouts)

Tái sử dụng cơ chế xác thực hiện có. Browser không được nhận Modal account token. Nếu cần backend relay để giữ credentials server-side, relay chỉ chuyển frame/metadata và phải ghi rõ topology; không chuyển YOLO/analytics về local.

Không tạo public endpoint video không xác thực. Với test dùng endpoint tạm hoặc endpoint test hiện có, không ghi đè production deployment.

## 6. Packet và đồng bộ frame/path

Định nghĩa packet có version, tối thiểu:

```text
session_id, stream_epoch, input_hash
source_name, mode, frame_id, media_time_s
frame_width, frame_height
path_id, path_revision, path_state, route_scope
path_updated_at_s, evidence_until_s, path_support
processing_fps, preview_fps, inference_ms
queue_depth, dropped_input_frames, dropped_preview_frames
rendered_jpeg_bytes
```

Nếu dùng WebSocket, ưu tiên một binary packet có header length + JSON metadata + JPEG để không ghép nhầm metadata/frame. Kiểm tra length/schema/payload trước khi decode. Giới hạn một hoặc vài packet chờ; không dùng base64 JSON cho toàn bộ video nếu không cần.

Điều kiện:

- Points lấy từ kết quả tracking của đúng `frame_id`.
- Path snapshot bất biến; `evidence_until_s <= media_time_s`.
- Snapshot chứa path hiện tại hoặc null, không dùng một PNG path cuối cùng suốt video.
- UI vẽ frame đã render; không ghép thêm điểm/path cũ ở frontend.
- Cùng snapshot dùng cho video và dashboard tại frame đó. Diagnostics mới hơn được tách riêng nếu cần.
- Frame cũ/out-of-order theo epoch/frame ID bị loại. Decode ảnh ở frontend cũng không được hoàn tất lệch thứ tự rồi vẽ đè frame mới.
- Reconnect/reset tạo epoch hoặc khôi phục session có quy tắc rõ; không trộn path cũ vào video mới.
- Giải phóng ImageBitmap/object URL/buffer sau dùng để tránh tăng RAM.

## 7. Common Path theo từng thời điểm

Giữ event-time analytics. Gợi ý ban đầu:

- Flow update mỗi khoảng 1 giây media time.
- Candidate extraction mỗi khoảng 2–3 giây media time.
- Cửa sổ 30/180 giây, hysteresis theo config đã kiểm chứng.
- Render snapshot hiện tại lên mọi frame output, không chỉ frame đúng lúc extract.

Ý nghĩa hiển thị: path tại giây t mô tả luồng gần đây đã quan sát tới t, không phải tuyến của toàn video hoặc dự báo tương lai.

Các trạng thái UI:

| Trạng thái | Hiển thị |
| --- | --- |
| starting/loading_model | Tiến trình khởi động; chưa giả số FPS/path |
| learning/insufficient_data | Video + points; ghi đang tích lũy và lý do chưa có path |
| confirming | Video + points; panel ghi candidate đang chờ xác nhận |
| active | Video + points + Common Path đã xác nhận |
| cooling | Giữ path mờ, ghi evidence đang cũ |
| retired | Bỏ path, video vẫn tiếp tục |
| ended/error/disconnected | Báo trạng thái nguồn/worker rõ ràng |

Mặc định không vẽ candidate như Active. Có thể có toggle debug vẽ candidate nét đứt và nhãn “ứng viên/chưa xác nhận”, grid và support; toggle này không thay đổi state thật.

Polyline cùng route cần ổn định và giữ path_id. Đổi tuyến chỉ sau đủ evidence/hysteresis; không nối morph hai nhánh khác nhau xuyên vật cản.

## 8. Giải quyết nguyên nhân chưa xuất hiện path

Không chỉ sửa streamer: nếu validator tiếp tục loại tất cả candidate, người xem vẫn sẽ không thấy Active Path.

### A. Audit segmentation

Báo cáo trước có 11.897 segments từ 2.954 IDs. Phân rã segment cuts theo:

- Observation gap.
- Jump threshold/coordinate scaling.
- Missing confidence/observed flag.
- Epoch/reset/ID change thật sự.

Xuất raw ID, segment_id, timestamp, delta-time, displacement, threshold và lý do. Kiểm tra đơn vị pixel/normalized và tracker frame rate so với FPS xử lý thực tế.

Không khẳng định mọi đoạn ngắn đều là lỗi ByteTrack. Không bỏ segment cuts để làm đẹp số liệu; chỉ sửa ngưỡng/lifecycle nếu có bằng chứng false cuts, đồng thời test chống teleport.

### B. Audit gate-complete nhưng route-support bằng 0

Trên dữ liệu có cache, chọn vài gate-complete tracks và candidate cùng scope:

1. So frame/PTS, epoch/track key và time-window membership.
2. So grid dimensions, cell indexing, x/y, pixel/normalized transform.
3. So chiều đi, source/target gates và loại route.
4. Xuất chuỗi cell nén của track/candidate, các cặp match, coverage và lý do order fail.
5. Kiểm tra candidate có đi cùng nhánh với track thật không.

Thêm regression test: candidate lấy từ chuỗi cell của một track hợp lệ phải được chính track đó nhận là có ordered support; không yêu cầu một track tự đủ min-support để Active. Test thêm hướng ngược và route khác phải bị loại.

Chỉ chuyển sang local_corridor cho Shibuya nếu có evidence phù hợp. Complete khi cùng track đi từ gate S tới T trong corridor, không buộc qua toàn bộ biên màn hình. Không ghép các đoạn của nhiều người thành complete route.

Nếu đủ bằng chứng để xác nhận local path, ghi scope rõ trên UI. Nếu vẫn không đủ, báo nguyên nhân thay vì giảm threshold tùy tiện hoặc vẽ đường tay.

## 9. Nhịp video và độ trễ

Mặc định dùng MP4 như nguồn realtime được pace theo source PTS:

- Bắt đầu clock nguồn sau khi model ready và viewer kết nối.
- Khi xử lý không kịp, giữ queue hữu hạn và bỏ input cũ có kiểm soát; tracker dùng đúng delta-time, không sinh flow từ prediction-only.
- Giới hạn preview khoảng 10–15 FPS làm điểm xuất phát. Cap preview là bỏ bớt frame hiển thị, không tự động giảm FPS analytics.
- Client chậm phải nhận frame mới nhất; không chờ gửi hết backlog. Có send timeout, heartbeat và cleanup.
- Nếu giảm input FPS làm tracking vỡ, báo trade-off và sửa bottleneck trước khi drop mạnh hơn.

Cung cấp mode phụ `inspect_all_frames`: xử lý tuần tự mọi frame và stream ngay khi xong, có thể chậm hơn thời gian video. UI phải ghi “đang xử lý — chậm hơn realtime” nếu không theo kịp; không coi đây là realtime 1×.

Không đọc MP4 tới EOF nhanh hết cỡ vào queue latest-frame trong mode realtime. Không dùng kết quả final rồi phát lại để giả live processing.

Đo riêng:

- Build/cold start/model load.
- Time-to-first-frame sau Start và sau model_ready.
- Decode/inference/tracking/analytics/render/encode.
- Processing FPS, preview FPS, input/preview drops, queue depth.
- Server frame age bằng timestamp cùng clock domain.
- Browser display progress và số frame hiển thị.

Không trừ wall clock browser và server chưa đồng bộ để tuyên bố e2e latency chính xác. FPS replay không đại diện FPS GPU inference.

## 10. Timeline artifact để xem lại path tại từng thời điểm

Song song live stream, ghi:

- `common_path_timeline.jsonl`: media_time, frame_id, path_id/revision/state/scope, polyline, support, evidence_until, reason.
- `path_events.jsonl`: activate/switch/cooling/retire và bằng chứng.
- `frame_index.jsonl`: liên kết output frame với source frame/PTS và path revision.
- `live_preview.mp4`: video đã vẽ overlay đúng lịch sử khi xử lý.
- `tracking_cache.jsonl` hoặc schema cache hiện có: phục vụ replay Shibuya.

Ghi timeline khi state/geometry thay đổi và snapshot định kỳ khoảng 1 giây; không giữ tất cả trong RAM. Nếu lưu MP4 có bỏ frame, bảo toàn PTS hoặc ghi rõ playback retiming và có frame_index.

Không bắt buộc xây thêm timeline slider trong lượt này. Nhưng artifact phải cho phép xác minh tại t=10/30/60 giây hệ thống biết gì, và tại sao path xuất hiện/đổi.

## 11. Test và bằng chứng hoàn thành

Trước GPU, kiểm tra logic streaming/protocol bằng cache phù hợp hoặc synthetic fixtures gắn nhãn rõ; không dùng chúng làm bằng chứng live inference.

Smoke Modal: xác minh đúng Shibuya, CUDA inference thật, frame đi tới UI trong lúc session running. Sau đó integration 90–120 giây nguồn nếu video/budget cho phép. Không loop clip để tạo evidence hoặc số người mới.

Browser test phải dùng endpoint Modal hoặc relay tới Modal, không local cached MP4:

1. Mở UI thật, nhấn Start Shibuya.
2. Hiển thị source_name, session_id, mode và media time.
3. Lưu browser evidence tại đầu session, lúc accumulating, lúc Active nếu có và gần cuối.
4. Chứng minh `first_browser_frame_received < job_completed` bằng event sequence/ACK và log tương quan session; không dựa vào trừ hai clock chưa đồng bộ.
5. Quan sát nhiều frame/points thay đổi, và path revision/state theo timeline.
6. Kiểm tra Start/Stop, EOF/disconnect, console/network errors; không sinh thêm pipeline ngoài budget khi kiểm tra.
7. Kiểm tra overlay không biến mất ở các frame giữa hai lần extract.
8. Nếu không có Active, stream có thể PASS nhưng mục tiêu Active Path phải là NOT_VERIFIED/INSUFFICIENT_DATA, không báo hoàn thành toàn bộ.

Không bắt path phải đổi liên tục để chứng minh động: nếu luồng ổn định, giữ cùng path là đúng. Dùng unit test chuyển luồng có timestamp để kiểm tra hysteresis; không nhận đó là bằng chứng trên video thật.

Nếu browser không có route tới Modal, báo blocker thật. Screenshot MP4 hoặc localhost replay không thay thế kiểm chứng remote stream.

## 12. Output và bàn giao

Output run mới: `outputs/common_path/shibuya-live-<run_id>/`. Giữ nguyên run diagnostic trước.

Tối thiểu có:

- Manifest, resolved config, input hash, Modal job/session/device.
- Live preview MP4, common_path_timeline, frame_index và path_events.
- Cache Shibuya, candidate funnel và tracking-cut diagnostics.
- Browser screenshots/recording ngắn, ui_verification.json.
- Metrics, first-frame evidence và report chỉ rõ inference/replay topology.

Báo cáo cuối tách các trạng thái:

```text
SOURCE: Shibuya verified / blocked
INFERENCE: Modal CUDA verified / blocked
REMOTE_STREAM: verified / not verified
COMMON_PATH: active verified / insufficient data / defect
BROWSER: verified / not verified
REALTIME_1X: measured pass / slower than realtime / not measured
```

Kèm file đã sửa, nguyên nhân, lệnh Start/Stop/test đã chạy, URL xem test nếu session còn được phép hoạt động, và đường dẫn artifact. Khi hết session budget, dừng test và cung cấp lệnh chạy lại; không giả URL còn live.

**Bắt đầu từ việc xác minh đúng Shibuya và sửa pipeline để publish frame ngay trong lúc Modal xử lý. Đồng thời truy vết fragmentation và validation; không giải quyết việc thiếu Common Path bằng cách chỉ phát lại video output cũ.**
