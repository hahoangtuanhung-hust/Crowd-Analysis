# Prompt: Tìm và sửa nút thắt video 10 phút trên Modal T4

Hãy đọc repository Crowd Analysis, đo trên code/config thực tế rồi sửa trực tiếp nguyên nhân làm video khoảng 10 phút chạy chậm. Ưu tiên patch nhỏ, không viết lại hệ thống hoặc chỉ đưa ra đề xuất.

## 1. Bối cảnh và phạm vi

Theo báo cáo Shibuya đã cung cấp, chưa phải phép đo của video 10 phút:
- Profile tiled chạy 1 full-frame + 4 tile, tức 5 ảnh inference/frame nguồn.
- Full-video 65 giây từng có analytics p95 331.457 ms và 600.622 ms ở hai profile khác nhau. Không kết luận Common Path luôn nhẹ từ smoke test 5 giây.
- Lượt B/C hybrid mới hơn đều chạy 5 ảnh detector/frame; hybrid không tiết kiệm inference và wall time chậm hơn 8.2%.
- Các lượt trên khác phiên bản/điều kiện; phải kiểm tra provenance trước khi gọi chênh lệch là regression.

Giữ model/backend, detector/tracker và đầu ra Common Path polyline gồm các đoạn thẳng ngắn, mũi tên có hướng, Top K, Path ID/màu. Không chuyển TensorRT, thêm ReID hoặc mở rộng tính năng. Giữ hybrid OFF cho baseline nếu chưa có bằng chứng lợi ích trên nguồn này; ghi resolved config thật.

## 2. Đo trước khi sửa

Lần theo entrypoint Modal, pipeline, detector, tracker, tracklet aggregation, renderer và xuất artifact. Xác nhận GPU thực, CPU/RAM container, model/device, precision, tiling, kích thước tensor, commit/config hash và cache policy. Không giả định cấu hình từ tên file.

Tách thời gian:
`chờ cloud/khởi tạo → lấy video/load model → xử lý → hoàn tất encode/lưu kết quả → tải artifact về máy`.

Trong vòng xử lý, đo riêng:
`decode → crop/preprocess/transfer → model inference → postprocess/merge tile → tracking → tracklet update → aggregation/ranking/stabilization → render → encode → logging/I/O/queue wait`.

Báo count, tổng thời gian, p50/p95 và throughput thực. Dùng phép đo GPU đúng; không nhầm thời gian enqueue với thực thi, không cộng p95 từng bước thành p95 end-to-end. Nếu stage chạy song song/lồng nhau, không cộng trùng thời gian; chỉ dùng profiler chi tiết ở cửa sổ ngắn rồi đo lại throughput khi tắt profiler.

Ghi số ảnh inference, batch/tensor shapes, detection, active/lost tracks, tracklet/candidate/điểm đang giữ, số cặp matching, queue depth, RAM/VRAM và CPU/GPU utilization theo thời gian. Analytics phải có thống kê riêng cho lần thực sự tính toán, không để các lần no-op che độ trễ.

## 3. Phân biệt chậm ngay từ đầu và chậm dần

Xuất thống kê mỗi 30–60 giây VIDEO của một lần chạy liên tục. Kiểm tra latency, bộ nhớ và kích thước state có tăng dần không.

So sánh đầu/giữa/cuối cùng với mật độ detection, track và lịch inference. Seek thẳng tới cuối với state rỗng không chứng minh pipeline ổn định sau 10 phút tích lũy.

Ưu tiên tìm và sửa tối đa 3 hotspot có bằng chứng:
- Detector: load model/remote call lặp, CPU fallback, tile/preprocess/copy trùng, hậu xử lý hoặc merge duplicate chậm; xác minh ý nghĩa metric `inference` giữa các phiên bản.
- Tracker: association quá rộng, track lost không được dọn, lịch sử tăng vô hạn, cập nhật/predict lặp. Prune/vectorize an toàn nhưng không phá che khuất và hướng chuyển động.
- Common Path: `_attach()`/matching so sánh toàn lịch sử, ingest lặp delta, rebuild/clustering/smoothing mọi frame, candidate tăng vô hạn. Dùng cập nhật gia tăng, neighbor search giới hạn, cửa sổ hữu hạn và expire cả support cũ.
- Pipeline/Modal: queue không giới hạn, decode/encode hay ghi JSON/Volume đồng bộ quá thường xuyên, xuất quá nhiều debug artifact, tải kết quả/UI làm chặn xử lý. Kiểm tra model reuse và vòng đời worker/session.

Không xóa mù lịch sử hoặc giới hạn state tùy tiện để tăng FPS. Cửa sổ, TTL, vận tốc và grace dùng PTS/video time; mỗi tracklet delta chỉ đóng góp một lần. Giữ thứ tự frame và state riêng từng session.

## 4. Nguyên tắc sửa

Sửa lãng phí tính toán/I/O trước, giữ nguyên workload và sampling để đo lợi ích thật. Cache overlay khi hình học/config không đổi; không tính lại polyline/mũi tên vô ích. Buffer/queue phải có giới hạn.

Offline: backpressure khi cần, không âm thầm drop frame. Chạy detector thưa hơn, đổi tiling/resolution/precision là experiment riêng sau feature flag; phải kiểm chứng coverage/người nhỏ và tracking, không tự đổi production.

Không bật hybrid để mặc định “sẽ nhanh hơn”; nếu vẫn scan toàn cảnh thì loại overhead không cần thiết bằng config phù hợp. Prediction không tạo support Common Path. Không đổi Top K, làm mất tuyến hoặc tắt analytics để làm đẹp benchmark.

## 5. Kiểm chứng tiết kiệm, nhưng đủ dài

1. Dùng baseline/artifact hiện có khi provenance phù hợp; nếu thiếu, đo một đoạn đại diện để định vị hotspot trước. Không chạy hàng loạt full-video experiments.
2. Replay detection cache để cô lập tracker hoặc tracking cache để cô lập Common Path khi hợp lệ. Đo sự tăng state trên toàn 10 phút; FPS replay không phải FPS inference.
3. Test hồi quy đúng phần sửa, rồi so trước/sau trên cùng đoạn, GPU, frame schedule, cấu hình và mức logging.
4. Chạy bản cuối liên tục hết video 10 phút với inference thật. Đối chiếu throughput đầu/giữa/cuối, bộ nhớ, queue, số frame và output path. Baseline full-run chỉ tái sử dụng nếu thật sự tương đương.
5. Đối chiếu người nhỏ, gap/fragmentation, hướng tuyến và Top K ở cùng timestamp. Không dùng raw Track ID làm recall; chưa review chất lượng thì ghi REVIEW_PENDING.

Tách FPS pipeline, FPS playback/UI, thời gian remote và tải artifact. Báo intentional detector skips riêng với dropped frames. Nếu hết ngân sách/timeout, ghi số frame đã xử lý và NOT_RUN/BLOCKED; không coi test ngắn là PASS cho video dài.

## 6. Bàn giao

- `bottleneck_report.md`: nguyên nhân có bằng chứng, file/hàm, trước–sau và giới hạn kết luận.
- Patch nhỏ + config rollback; không xóa code chưa kiểm tra phụ thuộc.
- `stage_metrics.csv`, `long_run_metrics.csv`, resolved config và lệnh tái lập thật.
- Bảng trước–sau: FPS, tổng wall time, latency từng stage, state size, RAM/VRAM; vài snapshot/clip cùng timestamp xác nhận không làm hỏng Common Path.

Bắt đầu bằng đọc code và profiling; không hỏi lại thông tin repository đã có. Chỉ hỏi khi thật sự thiếu video/quyền truy cập. Không hứa mức tăng tốc trước khi đo. Kết luận rõ: nghẽn ở đâu, patch giảm chi phí nào và còn phần nào chưa kiểm chứng.
