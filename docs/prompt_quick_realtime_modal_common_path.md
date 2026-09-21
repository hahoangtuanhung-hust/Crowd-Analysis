# Prompt triển khai nhanh: Modal GPU + video realtime + Common Path đổi theo dòng người

Bạn là Senior Computer Vision Engineer + Full-stack Engineer. Hãy sửa trực tiếp project hiện tại, triển khai và kiểm thử; không chỉ đưa kế hoạch. Giữ code đang hoạt động, thay đổi của người dùng và khả năng rollback. Không viết lại project.

## 1. Kết quả cần đạt

Người dùng nhấn Start trên UI → Modal xử lý video bằng GPU → UI nhận video ngay trong lúc xử lý, có:

- Một point tại chân mỗi người, không bounding box hoặc trajectory cá nhân.
- Một Common Path có mũi tên thể hiện dòng di chuyển đông nhất hiện tại.
- Khi dòng khác có nhiều người hơn và duy trì đủ thời gian xác nhận ngắn, Common Path chuyển sang dòng đó.
- Video vẫn phát khi chưa có Common Path; UI ghi rõ đang tích lũy/chưa đủ dữ liệu.

Ưu tiên input `data-shibuya-test.mp4`. Xác minh đúng file/hash; không thay bằng video Grand Central, không dùng ROI/cache của video khác.

## 2. Định nghĩa Common Path cho MVP này

**Common Path ở đây là dòng di chuyển cục bộ đang có nhiều người nhất, không bắt buộc là tuyến entrance→exit hoàn chỉnh.**

Đây là thay đổi định nghĩa sản phẩm so với validator global_od cũ. Tạo mode `dominant_live_flow` riêng, giữ mode global_od/legacy để rollback. Không hạ chuẩn của global_od rồi gọi kết quả là tuyến hoàn chỉnh.

Đếm số track đang được quan sát và đang di chuyển theo từng dòng, không đếm tổng số point, tổng số frame hay tổng người từ đầu video.

- Mỗi `(camera_id, epoch, track_id)` chỉ đóng góp tối đa một phiếu tại một thời điểm.
- Không cộng nhiều segment của cùng ID thành nhiều người.
- Đây là số người ước lượng từ tracking; ID switch vẫn có thể gây sai số, phải báo giới hạn.
- Người đứng yên, prediction-only hoặc đã mất quan sát quá timeout không tạo phiếu.
- Chỉ dùng dữ liệu tới timestamp frame hiện tại.

Ví dụ minh họa, không phải benchmark: dòng sang phải có 18 người, sang trái có 10 → chọn phải. Sau đó phải còn 8, trái có 20 và duy trì 2 giây → chuyển sang trái. Hai dòng bằng nhau thì giữ lựa chọn cũ nếu còn hợp lệ.

## 3. Thuật toán gọn, tận dụng Directional Grid có sẵn

1. YOLO + tracker hiện có → bottom-center point đã hoàn nguyên về tọa độ frame gốc.
2. Ước lượng hướng từ observation thật của mỗi track trong khoảng 0.3–0.8 giây; lọc jitter bằng smoothing causal và ngưỡng chuyển động có cấu hình.
3. Chia ROI thành grid, mỗi ô giữ histogram 8 hướng. Giữ các mode đối nhau, không lấy trung bình chúng thành một vector.
4. Dùng các đoạn chuyển cell hợp lệ gần đây để nối ô thành cụm dòng có hướng liên tục. Không gộp hai hành lang tách rời chỉ vì cùng hướng, không nối qua gap lớn/vật cản.
5. Gán mỗi track đang di chuyển vào tối đa một cụm phù hợp vị trí/hướng gần nhất. Score của cụm = số distinct track hiện tại được gán, không phải tổng support các edge.
6. Chọn cụm score lớn nhất. Dựng polyline/mũi tên ngắn theo corridor được evidence hỗ trợ; làm mượt trong cùng cụm. Nếu chỉ đủ hướng cục bộ thì hiển thị mũi tên cục bộ, không bịa tuyến xuyên toàn cảnh.
7. Tracklet validation kiểm tra evidence cục bộ có thật, đúng hướng/thứ tự, không teleport. Không yêu cầu track đi từ biên ảnh tới biên ảnh trong mode này.

Tách hai loại dữ liệu:

- History ngắn dùng ổn định hướng/hình học.
- Snapshot số người đang di chuyển dùng quyết định dòng đông nhất.

Không dùng count tích lũy 30/180 giây làm score chính cho yêu cầu “đông nhất hiện tại”. Không chờ đủ 180 giây mới hiển thị.

### Chống nhấp nháy

- Re-evaluate khoảng mỗi 0.5–1 giây media time.
- Dòng đầu tiên cần tối thiểu 3 track hợp lệ và tồn tại 2 giây.
- Challenger có nhiều hơn Active ít nhất 1 track và duy trì 2 giây thì switch.
- Dùng identity cụm ổn định theo overlap/hình học/hướng; không reset confirmation mỗi lần extract.
- Reset confirmation nếu challenger đổi hoặc không còn đông hơn.
- Hòa điểm: giữ Active. Không còn dòng đủ support: đánh dấu stale ngắn rồi ẩn, không giữ path cũ mãi.
- Không xác nhận từ cùng snapshot lặp lại; phải có observation mới.

Các giá trị trên là baseline có cấu hình. UI hiển thị thời gian xác nhận để người dùng hiểu độ trễ chủ ý; không gọi đây là phản ứng tức thì từng frame.

## 4. Backend Modal và live stream

Triển khai luồng:

```text
Video trên Modal → Decode → YOLO GPU → Tracking → Dominant live flow
→ Vẽ point + Common Path vào frame → Gửi ngay tới UI
```

- Inference thực sự chạy CUDA trên Modal; ghi GPU/device và detector_calls. Không chỉ thuê container GPU rồi replay CPU và gọi là inference GPU.
- Tracking/analytics có thể chạy CPU trong cùng container. Load model một lần; state riêng theo session.
- Tái sử dụng FastAPI/session/player hiện có. Dùng WebSocket binary JPEG nếu cần bổ sung transport; không chờ MP4 hoàn tất mới phát.
- Một pipeline/session, nhiều request/viewer không được tạo thêm detector workers. Với MVP giới hạn một session xử lý và một serving container, xử lý control/stream đồng thời.
- Không block event loop bằng inference/encode. Worker và queue hữu hạn; client chậm không kéo chậm toàn pipeline.
- Giữ authentication hiện có; không gửi Modal account token xuống frontend. Nếu relay qua backend local, backend đó chỉ relay, không inference.
- Không polling Modal Volume để giả stream; Volume dùng input/cache/output.

Modal hỗ trợ ASGI/WebSocket. Kiểm tra API SDK, concurrency, timeout và giới hạn packet theo [tài liệu Modal](https://modal.com/docs/guide/webhooks); không giả định state được chia sẻ giữa các container.

Mỗi packet gồm session/epoch, frame_id, media_time_s, JPEG đã overlay, path_id/state/revision/scope, hướng, số người của Active/challenger và FPS. Metadata và JPEG phải ghép theo cùng packet; frontend loại frame cũ và quản lý thứ tự decode.

## 5. UI và nhịp realtime

- Start/Stop, tên video, video đang xử lý, hướng chủ đạo, số người theo hướng, trạng thái path và FPS.
- Nhãn chính: “Dòng di chuyển đông nhất”. Grid/candidate/Track ID chỉ bật khi debug.
- Vẽ point/path trên server vào đúng frame; UI chỉ hiển thị frame đã nhận, tránh overlay lệch.
- Common Path tính lại định kỳ nhưng được vẽ trên mọi frame; không dùng một ảnh path tĩnh cuối video.
- Có trạng thái connecting/loading/running/ended/error; EOF không lặp ngầm.

MP4 realtime phải pace theo PTS sau khi model ready. Queue input hữu hạn, drop frame cũ có kiểm soát khi không theo kịp; tracker dùng delta-time/gap đúng. Không đọc MP4 tới EOF nhanh hết cỡ vào latest-frame queue.

Giới hạn preview khoảng 10–15 FPS làm điểm xuất phát; preview FPS tách khỏi processing FPS. Nếu không theo kịp thời gian nguồn, báo lag/drop thật, không phát chậm rồi gọi là realtime 1×. Giữ history/dedup/buffer có TTL và giới hạn RAM.

## 6. Thứ tự làm và test nhanh

1. Đọc README/AGENTS nếu có, source/config/report; báo tối đa vài dòng về file cần sửa rồi triển khai.
2. Thêm `dominant_live_flow` và unit tests bằng track points giả lập có nhãn synthetic.
3. Kiểm tra streaming/browser plumbing bằng cache đúng nguồn nếu có, không dùng làm bằng chứng inference thật.
4. Chạy Modal inference thật trên Shibuya: một smoke 15–20 giây, sau đó một integration tối đa 90 giây nếu đủ budget.
5. Mở UI ngay trong lúc integration đang chạy; lưu screenshot/recording và timeline.
6. Lưu detections/tracklets cache; sửa analytics/render tiếp theo bằng replay, không gọi lại YOLO vô ích.

Giới hạn lượt test: tối đa 2 processing sessions GPU, gồm cả retry, timeout mỗi session 600 giây, tổng compute budget 900 giây. Dừng test service/worker khi xong. Cần thêm budget thì báo lý do. Thiếu video/Modal/browser access phải báo blocker cụ thể.

Test quan trọng:

- Một dòng chiếm ưu thế; hai dòng ngược chiều không triệt tiêu nhau.
- Dòng khác trở nên đông hơn: giữ qua 2 giây xác nhận rồi switch đúng.
- Challenger chỉ tăng dưới 2 giây hoặc hai dòng hòa: không nhấp nháy.
- Người đứng yên/mất observation/ID tái sử dụng sau reset không làm tăng count.
- Cụm cùng hướng ở hai khu vực rời nhau không bị nối thành tuyến giả.
- UI nhận nhiều frame khi Modal vẫn running, không chờ job kết thúc.
- Video vẫn chạy khi chưa đủ 3 track hoặc không có Active.

Nếu video thật không có đổi hướng ưu thế trong đoạn test, báo “switch verified bằng unit test, chưa verified trên clip này”; không dựng kết quả giả. Nếu vẫn không có Active, ghi reason/count từng bước, không báo hoàn thành tính năng chỉ vì video phát được.

## 7. Output và acceptance

Lưu vào `outputs/common_path/live-dominant-<run_id>/`:

- `manifest.json`: input hash, GPU, mode, config, session, versions.
- `preview.mp4`: video point + Common Path đúng lịch sử thời gian.
- `path_timeline.jsonl`: timestamp, scope, hướng, count, challenger, path revision, switch reason.
- `metrics.json`: processing/preview FPS, stage latency, drops, RAM, first-frame timing.
- `ui_evidence/`: screenshot/recording từ browser nhận stream Modal thật.
- `report.md`: thay đổi, test đã chạy, output đã xem, giới hạn và lệnh chạy lại.

Chỉ công nhận khi đã có bằng chứng: đúng input; inference Modal CUDA; UI nhận frame trước khi job hoàn tất; path có evidence và đổi đúng quy tắc; không bbox/tail; không backlog/RAM tăng vô hạn trong test đã chạy.

Tách kết luận streaming, dominant-flow, switching và realtime performance. Không dùng FPS replay làm FPS inference; không khẳng định ổn định nhiều giờ từ clip ngắn.

**Bắt đầu sửa trên repo hiện tại. Ưu tiên một luồng chạy được end-to-end, hiển thị Common Path theo số người đang di chuyển và đổi theo ưu thế mới. Không mở rộng sang full-route clustering, multi-camera hoặc viết lại UI trong lượt này.**
