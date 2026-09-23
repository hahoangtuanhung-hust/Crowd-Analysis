# Prompt triển khai Common Path bằng tổng hợp tracklet

Bạn là **Senior Computer Vision Engineer + Realtime Systems Engineer**. Hãy sửa trực tiếp project Crowd Analysis hiện tại để quay lại hướng **tổng hợp các đoạn tracklet ngắn thành nhiều Common Path**, kế thừa phần đã triển khai trước khi chuyển sang Directional Grid.

## 1. Kết quả cần đạt

- Common Path biểu diễn tuyến chuyển động được nhiều tracklet quan sát ủng hộ trong cửa sổ thời gian gần đây.
- Tuyến có hình dạng thực tế: đường thẳng, đường gấp khúc hoặc đường cong; không ép mọi tuyến thành một mũi tên thẳng theo 8 hướng.
- UI cho phép chọn **số Common Path tối đa hiển thị K**, mặc định 3, phạm vi ban đầu 1–5.
- Mỗi đường có màu riêng, giữ ổn định khi thứ hạng thay đổi. Có legend tương ứng.
- Video cập nhật trực tiếp trong khi backend GPU Modal xử lý. Trên video chỉ vẽ điểm tracking của người và các Common Path đã xác nhận; tắt bbox, ID và trajectory cá nhân mặc định.
- Duy trì khả năng rollback bằng config. Không viết lại project, không xóa thay đổi đang có trong worktree.

**Chỉ hiển thị tối đa K đường có đủ bằng chứng. Nếu chỉ tìm được 2 đường hợp lệ khi K=5, hiển thị 2 đường; không tạo đường để lấp đủ số lượng.**

## 2. Kiểm tra và tái sử dụng triển khai cũ

Đọc README, config, analytics, renderer, Modal runner và thành phần frontend đang phát video. Dùng lịch sử Git nếu có để tìm cách tổng hợp tracklet trước đây. Chỉ đọc lịch sử hoặc khôi phục có chọn lọc; không reset toàn bộ repository.

Xác định ngắn gọn:

1. Engine nào đang chạy thật ở live Modal, batch và replay.
2. Module cũ nào có thể tái sử dụng để gom tracklet và dựng centerline.
3. Contract frame, tracking point, Common Path và cấu hình phiên hiện tại.
4. Backend inference thực tế là PyTorch hay TensorRT FP16. Giữ backend đang hoạt động, không mặc định TensorRT đã được triển khai chỉ vì có prompt trước đó.

Tên như `directional_grid.py`, `dominant_live_flow.py`, `engine.py`, `live_common_path.py`, `modal_shibuya_live.py`, `useModalLive.ts` là gợi ý tìm kiếm từ tài liệu trước; kiểm tra đường dẫn thực tế.

Báo cáo tối đa 10 dòng về module tái sử dụng, file cần sửa và thiết kế chọn, sau đó triển khai. Nếu không tìm thấy thuật toán cũ, thêm engine riêng qua interface hiện có. Không chỉ đổi tên `legacy` rồi coi như đã khôi phục đúng thuật toán.

## 3. Pipeline mục tiêu

`Observed tracking points → Short tracklets → Ghép các đoạn tương đồng có hướng → Tuyến đại diện → Kiểm chứng bằng tracklet → Hysteresis → Top K Active Common Paths`

Không dùng histogram 8 hướng theo grid làm nguồn quyết định hình dạng đường. Có thể dùng chỉ mục không gian để tìm hàng xóm nhanh, nhưng hình dạng tuyến phải xuất phát từ tracklet quan sát.

### 3.1. Tạo tracklet ngắn

- Tiếp tục dùng detector và tracker hiện tại. Điểm đại diện là bottom-center của bbox nội bộ.
- Khóa track phải gồm `camera_id`, `stream_epoch`, `track_id`; giữ state riêng cho từng camera/phiên.
- Lưu `frame_id`, timestamp nguồn, vị trí, trạng thái `observed/confirmed` và segment ID.
- Chỉ quan sát thật và track hợp lệ được tạo bằng chứng. Kalman prediction không được tạo support mới.
- Loại jitter, đoạn quá ngắn, không đủ điểm hoặc độ dịch chuyển quá nhỏ bằng ngưỡng configurable.
- Tách segment khi mất quan sát quá lâu hoặc có bước nhảy bất hợp lý. Không nối qua khoảng mất dữ liệu lớn.
- Có thể cắt track đang sống thành các đoạn nhỏ để cập nhật sớm, không cần chờ người ra khỏi toàn bộ khung hình.
- Giới hạn điểm/track, tracklet trong cửa sổ và tổng bộ nhớ. Số lần một track được cắt đoạn không làm tăng số track hỗ trợ tuyến.

### 3.2. Ghép tracklet theo vị trí, hướng và thứ tự

Ưu tiên tái sử dụng thuật toán cũ nếu nó đáp ứng các điều kiện sau; bổ sung đúng phần còn thiếu:

1. Tìm tracklet hoặc đoạn tuyến lân cận bằng chỉ mục không gian hoặc bước lọc rẻ; tránh so sánh mọi cặp toàn bộ lịch sử.
2. So khớp theo khoảng cách hình học, hướng tiếp tuyến cục bộ và thứ tự tiến dọc tuyến. Hướng cần được so trên phần chồng lấn, không chỉ theo vector đầu–cuối của tracklet.
3. Cho phép tracklet chỉ phủ một phần tuyến. Căn chỉnh các phần chồng lấn trước khi tổng hợp; không resample mọi tracklet về cùng số điểm rồi lấy trung bình theo chỉ số khi chúng nằm ở các đoạn khác nhau.
4. Mở rộng tuyến bằng đoạn có phần chồng lấn đủ dài và sự tiếp nối được nhiều track quan sát ủng hộ.
5. Tại giao cắt hoặc phân nhánh, chỉ nối sang nhánh mới nếu tracklet cho thấy chuyển động tiếp diễn qua điểm nối. Giao nhau về vị trí không đủ để suy ra có người rẽ.
6. Tách hai chiều ngược nhau và các hành lang song song khác nhau. Không gom hai tuyến chỉ vì cùng hướng chung.

Ví dụ: nhiều tracklet phủ A–B, nhiều tracklet khác phủ B–C và có bằng chứng quan sát nối tiếp quanh B thì có thể tạo tuyến tổng hợp A–B–C. Nếu các tracklet chỉ giao nhau tại một điểm nhưng không có bằng chứng chuyển tiếp, giữ thành các tuyến riêng.

Không yêu cầu tất cả track phải đi hết từ cổng vào đến cổng ra. Tuy nhiên, phải phân biệt **support của từng đoạn tuyến**, **track hỗ trợ một phần** và **track đi hết tuyến**. Không tuyên bố tất cả người ở đoạn A–B đều đã đi đến C.

### 3.3. Dựng đường đại diện

- Dùng centerline từ các đoạn đã căn chỉnh theo chiều chuyển động; ưu tiên trung vị hoặc phương pháp robust để giảm ảnh hưởng outlier.
- Giữ dạng polyline làm biểu diễn gốc. Giữ góc rẽ nếu tracklet thực tế có góc rẽ.
- Chỉ làm cong/làm mượt trong hành lang có bằng chứng, với giới hạn sai lệch configurable. Không để spline cắt qua vùng không có người đi hoặc tạo vòng giả.
- Chỉ kéo dài tuyến đến nơi có support. Nếu có khoảng trống lớn chưa được quan sát, cắt ngắn hoặc tách tuyến; không vẽ một đường liền qua đó.
- Lưu hình học, support và thời gian cập nhật cùng nhau để renderer nhận một snapshot nhất quán.

## 4. Support, xếp hạng và ổn định theo thời gian

### Support và độ phổ biến

- Tính bằng số **TrackKey quan sát khác nhau** trong cửa sổ gần đây, không tính số point, frame hoặc tổng số đoạn đã cắt.
- Dedupe một TrackKey trên cùng tuyến trong cửa sổ. Người đứng lâu không được tăng điểm chỉ vì có nhiều mẫu.
- Kiểm tra support trên các phần của tuyến, không dùng tổng support lớn để che đoạn nối yếu hoặc không có bằng chứng.
- Chọn công thức score đơn giản, ghi rõ trong code/report. Ưu tiên support gần đây, độ phủ và chất lượng khớp; không ưu tiên đường chỉ vì dài hơn hoặc bị phân mảnh thành nhiều đoạn.
- Ghi rõ support là track quan sát, chưa phải số người duy nhất nếu tracker còn đổi ID. Không tự hợp nhất ID chỉ vì hai tracklet đi cùng một đường.
- Loại bằng chứng quá cửa sổ theo timestamp nguồn. Xử lý reconnect, seek hoặc đổi video bằng epoch mới.

### Ổn định đường và Top K

- Mỗi tuyến có `path_id` bền trong phiên; match tuyến mới với tuyến trước theo hình dạng và hướng, không theo rank.
- Xác nhận candidate qua nhiều lần cập nhật trước khi active.
- Làm mượt hình học sau khi đã căn chỉnh phần tương ứng; không blend hai nhánh khác nhau thành đường trung gian.
- Áp dụng margin và thời gian giữ lợi thế khi một tuyến mới cạnh tranh vị trí trong Top K. Không thay toàn bộ danh sách ở mỗi frame.
- Có thời gian cooling/retire khi mất support, thể hiện rõ trạng thái; không giữ tuyến cũ vô hạn.
- Màu gắn với `path_id`, giữ nguyên qua thay đổi rank và thay đổi K. Khi split/merge, quy định rõ tuyến nào kế thừa ID; nhánh mới có ID mới.
- Không gộp hai nhánh chỉ vì có chung đoạn đầu; đồng thời lọc candidate trùng gần như hoàn toàn để một tuyến không chiếm nhiều vị trí Top K.

## 5. UI và đồng bộ realtime

Thêm điều khiển **Số đường hiển thị** cạnh video, mặc định 3. Áp dụng trực tiếp vào phiên đang chạy, không khởi động lại YOLO và không xóa lịch sử tracking.

- K là giới hạn hiển thị; duy trì pool candidate có giới hạn riêng để tăng K không phải học lại từ đầu.
- UI gửi K đến backend qua API/WebSocket hiện có; backend validate và xác nhận giá trị đã áp dụng.
- Nếu overlay được vẽ ở backend, K và màu phải áp dụng trong renderer backend, không chỉ lọc legend ở frontend.
- Dùng đường kẻ **liền**, độ dày vừa phải, có mũi tên nhỏ để nhận biết chiều. Có thể dùng viền tương phản mảnh để dễ nhìn trên nền video.
- Legend hiển thị màu, tên hoặc ID đường, support gần đây và trạng thái. Màu kết hợp nhãn để người xem vẫn phân biệt khi màu gần nhau.
- Không có đường hợp lệ thì thông báo đang tích lũy bằng chứng. Không hiển thị candidate/debug mặc định.

Snapshot tối thiểu của mỗi đường gồm:

```text
path_id, color, rank, state, polyline,
support_tracks, score, evidence_until_s
```

Snapshot cấp frame gồm `camera_id`, `stream_epoch`, `source_frame_id`, `media_time_s`, `paths` và `applied_max_paths`. Nêu rõ tọa độ polyline là pixel hay chuẩn hóa; chuyển đúng tỷ lệ khi resize/letterbox preview.

Giữ kiến trúc backend Modal xử lý và vẽ overlay lên đúng frame, frontend hiển thị ảnh đã xử lý. Không bổ sung overlay thứ hai gây vẽ trùng. Evidence của path không được mới hơn frame đang hiển thị.

Chạy analytics theo nhịp riêng, ví dụ 0,5 giây thời gian video; giữa các lần cập nhật, tiếp tục render snapshot hợp lệ gần nhất lên frame mới. Dùng giới hạn bộ nhớ, hàng đợi có giới hạn và timestamp thực của pipeline. Chu kỳ xác nhận path không được chặn truyền video.

## 6. Cấu hình và rollback

Ví dụ dưới đây là **schema đề xuất**, phải map vào config thực tế và ghi rõ ngưỡng chưa hiệu chỉnh:

```yaml
analytics:
  common_path:
    engine: tracklet_aggregation
    max_paths: 3
    tracklet_aggregation:
      evidence_window_seconds: 30.0
      update_interval_seconds: 0.5
      min_support_tracks: 3
      confirmation_seconds: 2.0
      switch_margin_tracks: 2
      cooling_seconds: 3.0
      max_candidates: 20
```

Bổ sung ngưỡng geometry, hướng, gap, coverage và giới hạn history bằng config có đơn vị rõ ràng. Với tọa độ pixel, chuẩn hóa theo kích thước frame hoặc cấu hình camera; không hardcode pixel của riêng video Shibuya.

Kiểm tra cả live Modal và replay đều chọn engine mới. Không chỉ thay `analytics.common_path.engine` nếu entrypoint live còn đi qua selector khác như `dominant_live_flow.mode`.

Giữ engine hiện tại làm rollback và ghi đúng các khóa cần đổi. Thay engine phải reset state liên quan an toàn; thay K chỉ đổi lựa chọn hiển thị. Không thay detector, precision hoặc độ phân giải cùng lượt này nếu không cần sửa lỗi tích hợp.

## 7. Kiểm tra nhanh và có giới hạn

Ưu tiên cache detection/track có sẵn để sửa analytics và render; không gọi lại YOLO mỗi lần chỉnh thuật toán. Nếu cache không đủ trường hoặc không đúng phiên bản, báo rõ.

Chạy một nhóm test nhỏ trực tiếp liên quan:

1. Các tracklet ngắn chồng lấn tạo một tuyến cong hoặc gấp khúc có support.
2. Hai luồng giao cắt không bị nối thành góc rẽ giả; hai hướng ngược nhau vẫn tách biệt.
3. Lặp cùng tracklet/TrackKey không làm tăng support; evidence hết hạn làm tuyến biến mất đúng thời gian.
4. Đổi rank và K không đổi màu/ID của tuyến vẫn tồn tại; không ép đủ K khi thiếu đường.

Sau replay, chạy **một lượt live ngắn 20–30 giây trên GPU Modal** với đoạn đại diện của `data-shibuya-test.mp4` hoặc video đang dùng thực tế. Đây là smoke test, không phải chứng minh chất lượng dài hạn. Dùng cửa sổ quan sát thực có trong clip; không bắt chờ đầy cửa sổ nếu đã đủ support hợp lệ và không hạ ngưỡng để tạo đường giả.

Trong cùng lượt, mở trình duyệt, thay K giữa 1/3/5 và xác minh:

- Video thay đổi liên tục khi job còn chạy.
- Common Path là polyline có hình dạng theo tracklet; không chỉ là một hướng thẳng lấy từ histogram.
- Số đường không vượt K; màu và legend khớp overlay, không reset tracking khi thay K.
- Frame, metadata và overlay khớp thời điểm.

Ghi ngắn FPS hiển thị của frame mới, analytics p50/p95 và số candidate/active. Không tính ảnh lặp là FPS mới. Nếu UI vẫn ≤2 FPS, ghi điểm nghẽn có bằng chứng; không mở một đợt tối ưu model ngoài phạm vi.

Không chạy full suite, soak test, nhiều GPU hoặc ma trận tham số. Chỉ chạy lại ca liên quan nếu sửa lỗi vừa phát hiện. Nếu không có trình duyệt/GPU/cache, hoàn thành phần khả dụng và ghi rõ phần chưa kiểm chứng.

## 8. Bàn giao và tiêu chí hoàn thành

Lưu một thư mục output của lượt kiểm chứng gồm:

- Video preview ngắn và ảnh UI có Common Path.
- Timeline JSONL: frame/time, path_id, color, state, support, polyline và K đang áp dụng.
- Metrics ngắn và lý do candidate bị loại chính.
- README cập nhật lệnh chạy thật, cách dùng K và cách rollback.

Chỉ báo hoàn thành phần đã kiểm chứng. Không giả lập đường trong video thật và không dùng video output cũ để thay bằng chứng live.

Báo cáo cuối ngắn:

```text
REUSED
- Module/thuật toán cũ đã tái sử dụng

CHANGED
- File thay đổi, cách gom tracklet và dựng đường
- Điều khiển K, màu/path_id, entrypoint live và replay

VERIFIED
- Test liên quan, replay và browser live đã kiểm tra
- FPS hiển thị, analytics latency và output

LIMITATIONS
- Fragmentation/ID switch, support cục bộ hay toàn tuyến
- Phần chưa kiểm chứng hoặc chưa đủ dữ liệu

RUN / ROLLBACK
- Lệnh và config thực tế
```

Bắt đầu bằng việc tìm triển khai tracklet cũ và xác nhận engine live thực tế. Sau đó sửa module analytics, nối config/UI/renderer, replay từ cache và kết thúc bằng một lượt live Modal ngắn.
