# Prompt: Tinh gọn pipeline video Crowd Analysis, chỉ giữ Common Path

Bạn là kỹ sư Computer Vision phụ trách refactor và tối ưu dự án Crowd Analysis hiện tại. Hãy đọc repository, xác định đúng luồng đang chạy, rồi sửa trực tiếp code. Không chỉ đưa ra phương án, không viết lại toàn bộ dự án khi chưa cần thiết.

## 1. Mục tiêu và phạm vi

Mục tiêu duy nhất của luồng này là **tìm tuyến đường có nhiều lượt người di chuyển qua nhất từ các tracklet ẩn danh**, hiển thị thành đường liên tục với mũi tên chỉ hướng rõ ràng.

- Không cần biết người đó là ai; không nhận diện khuôn mặt, đối chiếu cư dân, tìm lại một người giữa các camera hoặc lưu hồ sơ cá nhân.
- Vẫn giữ **Track ID tạm trong từng camera/phiên** để gom quỹ đạo và hạn chế đếm trùng. Track ID không phải danh tính, cũng không bảo đảm đếm chính xác số người duy nhất khi có mất track/đổi ID.
- Common Path phải được tổng hợp từ tracklet, **không chuyển sang grid/heatmap**. Mỗi path là một đường đại diện, không phải tập hợp nhiều vệt tracking rời rạc.
- Mặc định hiển thị **Top 1 tuyến phổ biến nhất**; giữ khả năng chọn Top K trên UI nếu đã có. Chỉ hiển thị tối đa K tuyến đủ bằng chứng, không tạo thêm đường cho đủ số lượng.
- Giữ model detection và backend inference hiện có để so sánh trước/sau. **Chưa convert/build TensorRT trong đợt refactor này**; không xóa backend TensorRT đang được sử dụng chỉ vì nó là TensorRT.

Pipeline đích:

```text
Video/Camera
  -> Đọc frame và điều tiết tải
  -> Person Detection
  -> Tracking ẩn danh
  -> Thu thập, lọc tracklet
  -> Tổng hợp Common Path có hướng
  -> Vẽ đường + mũi tên
  -> Stream/hiển thị trên UI
```

## 2. Rà soát trước khi sửa

Đọc README, entrypoint, cấu hình và các module được gọi thực tế. Lần theo luồng từ nguồn video đến trình duyệt, bao gồm worker, API và frontend. Không suy đoán rằng một module tồn tại chỉ từ tên công nghệ trong prompt này.

Lập danh sách ngắn: **giữ / tối ưu / loại bỏ**, kèm vị trí và phụ thuộc. Đo baseline bằng một đoạn video cố định trước khi thay đổi. Kiểm tra riêng thời gian đọc/giải mã, tiền xử lý, inference, hậu xử lý, tracking, Common Path, vẽ, encode và truyền/hiển thị để tìm nút thắt thật; không mặc định model là phần chậm nhất.

Nếu repository chứa nhiều ứng dụng, chỉ tinh gọn luồng Crowd Analysis. Không xóa module dùng chung đang phục vụ chức năng khác.

## 3. Gỡ bỏ code không phục vụ mục tiêu

Nếu có và không còn consumer hợp lệ, gỡ khỏi pipeline, rồi xóa implementation/config/dependency riêng tương ứng:

- Face detection, face alignment, face recognition, face embedding, ReID theo danh tính, face gallery, xác thực cư dân và liên kết danh tính nhiều camera.
- Database/vector store/FAISS chỉ dùng cho danh tính; agent/LLM, nghiệp vụ cảnh báo, báo cáo hoặc xuất dữ liệu không phục vụ Common Path.
- Nhánh grid/heatmap đã bỏ, pipeline cũ chạy trùng, model load trùng, tác vụ ghi hình/crop ảnh tự động không cần thiết và code chết.

Loại bỏ cả import, khởi tạo model, thread/job, endpoint, màn hình, cấu hình và dependency tương ứng; không chỉ tắt hiển thị hoặc comment code. Giữ logging lỗi, cấu hình nguồn video, quản lý camera, health check và phần đo hiệu năng tối thiểu đang cần.

**An toàn khi xóa:** kiểm tra tham chiếu trong code, cấu hình, frontend và test; không xóa dữ liệu/video, model weights, `.env`, lịch sử Git hoặc thay đổi chưa commit của người dùng. Không dùng lệnh dọn/xóa hàng loạt không phân biệt. Khi chưa chứng minh được module là thừa, cô lập khỏi luồng chạy và ghi rõ thay vì xóa mù quáng. Không làm gãy API mà UI cần.

## 4. Tối ưu pipeline video

**Model và tracking:** khởi tạo một lần và tái sử dụng; chỉ xử lý lớp person. Giữ tracker hiện có nếu phù hợp, ví dụ ByteTrack; không bổ sung nhận diện/ReID nặng chỉ để làm Common Path. Không reset tracker mỗi frame. Tránh chuyển GPU–CPU, resize, đổi màu và copy frame nhiều lần; giữ đúng preprocess/postprocess của model. Không chia sẻ tracker giữa các camera.

**Điều tiết tải:** phân biệt chế độ realtime và offline. Realtime dùng hàng đợi có giới hạn, ưu tiên frame mới để tránh tích tụ độ trễ; không để client chậm kéo nghẽn inference. Offline dùng để phân tích/so sánh phải xử lý theo thứ tự và mặc định không chủ động bỏ frame. Ghi nhận số frame bị bỏ trong realtime.

Nếu giảm tần suất detection, phải xử lý khoảng thời gian thực giữa các quan sát theo khả năng tracker, không coi các frame bị bỏ là các frame liên tiếp. Không cập nhật tracklet/count nhiều lần bằng cùng một detection cũ. Kiểm chứng tác động của thay đổi sampling tới Common Path, không chỉ nhìn FPS.

**Bộ nhớ và lịch cập nhật:** lưu tracklet trong buffer giới hạn, có cửa sổ thời gian/TTL và dọn track hết hạn. Không lưu toàn bộ frame hoặc lịch sử vô hạn. Cập nhật Common Path theo chu kỳ cấu hình, độc lập với tốc độ vẽ; dùng kết quả mới nhất đã hoàn tất để render. Ưu tiên cập nhật phần dữ liệu thay đổi, tránh clustering lại toàn bộ lịch sử mỗi frame.

**Hiển thị:** kiểm tra đường truyền đến trình duyệt, không chỉ FPS inference. Tránh encode lặp cùng một frame cho nhiều consumer nếu có thể tái sử dụng an toàn; giới hạn tốc độ/độ phân giải preview độc lập với phân tích. Không tạo thread/process mỗi frame hoặc đưa thêm broker/microservice nặng chỉ để tối ưu. Bảo đảm quyền sở hữu frame/buffer rõ ràng, tránh race condition.

## 5. Sửa và ổn định Common Path

### Dữ liệu đầu vào

Mỗi tracklet cần camera/session, Track ID tạm, tọa độ và timestamp/frame ID. Thống nhất hệ tọa độ giữa model, frame gốc và UI; kiểm tra resize, letterbox, offset và scale trước khi chỉnh thuật toán. Giữ quy ước điểm đại diện hiện có nếu hợp lý; nếu dùng tâm đáy bbox thì áp dụng nhất quán.

Loại điểm trùng, dao động do đứng yên, bước nhảy bất thường và tracklet thiếu bằng chứng. Ngưỡng phải cấu hình được, tránh loại nhầm người đi chậm. Xử lý track đang hoạt động theo phần dữ liệu mới, không đợi tất cả track kết thúc mới có Common Path.

### Ghép tracklet và dựng đường

Phân nhóm theo vị trí, hình dạng và hướng di chuyển cục bộ; chỉ nối các đoạn có đầu–cuối tương thích, khoảng cách và độ đổi hướng hợp lý. Tách luồng ngược chiều hoặc giao cắt; không nối chỉ vì hai đoạn ở gần nhau. Cho phép đường cong/gấp khúc theo dữ liệu, không ép thành đường thẳng.

Phân biệt hai việc: nối mảnh quỹ đạo có khả năng thuộc cùng lượt tracking phải kiểm tra thời gian và chuyển động; tổng hợp tuyến từ nhiều người đi qua ở các thời điểm khác nhau thì **không yêu cầu timestamp của các tracklet phải nối tiếp nhau**. Không dùng ghép tracklet như một cơ chế khẳng định danh tính.

Dựng một polyline có thứ tự theo chiều di chuyển cho mỗi tuyến. Căn chỉnh hướng và lấy mẫu theo chiều dài cung trước khi làm mượt/cập nhật các điểm tương ứng; không lấy trung bình trực tiếp theo chỉ số của các tracklet khác độ dài. Chỉ nối/làm mượt trong vùng có bằng chứng, không bắc cầu khoảng trống lớn hoặc vẽ đường xuyên khu vực chưa quan sát được người đi qua.

### Xác định tuyến phổ biến

Tính mức hỗ trợ từ các lượt tracking ẩn danh trong cửa sổ thời gian, **không tính bằng số frame, số điểm hoặc số lần cập nhật**. Một track không được đóng góp lặp vào cùng path/đoạn chỉ vì tồn tại lâu hoặc bị cắt thành nhiều tracklet chồng lấp.

Điểm xếp hạng phải phản ánh bằng chứng di chuyển dọc tuyến, không chỉ một đoạn đông ở đầu tuyến. Không cộng số người của nhiều đoạn rời rồi khẳng định tất cả đã đi trọn đường ghép. Nếu chỉ có bằng chứng theo đoạn, báo đó là mức hỗ trợ theo đoạn và số lượt ước lượng; không gắn nhãn là số người duy nhất đi hết tuyến. Nêu rõ cách tính score/support đã triển khai.

### Ổn định và vẽ hướng

Giữ Path ID và màu theo hình học/hướng qua các lần cập nhật, không gán lại theo thứ hạng. Có làm mượt theo thời gian và ngưỡng thay thế để Top 1 không nhảy liên tục khi hai tuyến có hỗ trợ gần nhau; dữ liệu cũ vẫn phải hết hạn, không giữ mãi tuyến đã không còn phổ biến.

Vẽ nét liền, có mũi tên ở cuối và thêm dọc đường khi đủ dài. Mũi tên theo tiếp tuyến và chiều di chuyển thực, không đảo đầu ngẫu nhiên. Mặc định chỉ hiện Common Path; bbox/Track ID/vệt từng người chỉ bật trong chế độ debug. Khi thiếu dữ liệu, hiển thị trạng thái đang tích lũy thay vì tạo đường giả.

## 6. Cấu trúc và cấu hình

Tái sử dụng cấu trúc repo, chỉ tách module khi giúp trách nhiệm rõ hơn: nguồn video, detector/tracker, tracklet buffer, Common Path, render/stream. Không dựng framework mới hoặc thêm abstraction không cần thiết.

Giữ cấu hình gọn: realtime/offline, kích thước inference, tần suất detection, ngưỡng detector/tracker cần thiết, cửa sổ tracklet, chu kỳ Common Path, ngưỡng hỗ trợ/ghép, Top K và debug overlay. Dùng giá trị hiện có làm baseline; ngưỡng mới phải có đơn vị và được kiểm chứng bằng video. Không tự hạ chất lượng model/ảnh để báo tăng FPS mà bỏ qua chất lượng tuyến.

## 7. Kiểm chứng tối thiểu, tập trung đúng lỗi

Chạy lại cùng video/cấu hình/phần cứng trước và sau. Tách kiểm chứng thuật toán bằng replay cùng detection/tracklet đầu vào khỏi kiểm chứng realtime có bỏ frame khi có thể. Chỉ cần test tập trung và smoke test, không tạo bộ test đồ sộ ngoài phạm vi.

Xác nhận: luồng chính đông hơn được chọn Top 1; hai chiều/giao cắt không bị nối nhầm; đường rẽ và tracklet ngắt ngắn vẫn có hướng đúng khi đủ bằng chứng; người đứng yên/khung hình rỗng không sinh đường giả; khi luồng đổi hoặc hết dữ liệu, kết quả cập nhật/hết hạn hợp lý. Kiểm tra Top K, màu và mũi tên trên UI, cùng việc buffer/queue có giới hạn.

Báo trước/sau: FPS xử lý, FPS preview thực nhận ở UI nếu đo được, latency pipeline p50/p95, CPU, RAM, GPU/VRAM nếu có, số frame bỏ và kích thước queue/buffer. Định nghĩa rõ từng FPS; với realtime, đo tuổi frame từ timestamp nhận frame tới output trong cùng miền đồng hồ, không trộn timestamp video với wall clock. Đo thời gian GPU có đồng bộ trong lần profiling riêng khi cần; không thêm đồng bộ gây chậm vào mọi frame production.

Không đặt cam kết FPS khi chưa đo trên phần cứng thật. Nếu chưa có video/GPU/môi trường chạy, ghi rõ phần chưa kiểm chứng, không bịa kết quả hoặc tuyên bố đã fix thành công.

## 8. Bàn giao

Hoàn tất thay đổi code và gửi một báo cáo ngắn gồm: pipeline còn lại; nguyên nhân nặng/không ổn định có bằng chứng; file/module/dependency đã xóa và lý do; bảng đo trước/sau; cách chạy và các tham số chính; test đã chạy và giới hạn còn lại. Kèm ảnh/frame minh họa Common Path và mũi tên nếu chạy được.

**Hoàn thành khi:** luồng mặc định không load/chạy tác vụ danh tính hoặc chức năng thừa; chỉ còn pipeline phục vụ Common Path; tài nguyên có giới hạn; đường được tổng hợp từ tracklet, có hướng rõ, ổn định theo thời gian và Top 1 có căn cứ từ dữ liệu. Ưu tiên thay đổi nhỏ, đo được và dễ bảo trì; không đánh đổi tính đúng để lấy FPS.
