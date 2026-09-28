# PROMPT TRIỂN KHAI — COMMON PATH CHÍNH XÁC VÀ HYBRID MOTION-ROI NHẸ, ỔN ĐỊNH
> **Bản mở rộng ngày 2026-09-25.** Giữ yêu cầu sửa Common Path của prompt trước và bổ sung mục 14–17 về tự nhận diện vùng ít chuyển động, lập lịch detector theo ROI, bảo vệ tracking và benchmark trên Modal T4.
>
> **Đọc cả hai phần:** mục 1–13 quy định chất lượng tuyến đường; mục 14–17 quy định tối ưu upstream có kiểm soát. Không dùng motion mask để vẽ đường hoặc bỏ hẳn vùng tĩnh. Mặc định production tắt tính năng mới cho đến khi có A/B đủ bằng chứng.
>
> Đây là yêu cầu triển khai, không phải xác nhận code hiện tại đã hỗ trợ hoặc giải pháp đã được đo nhanh hơn. Phần bổ sung kết hợp phương án hybrid do người dùng yêu cầu với các ràng buộc kỹ thuật đề xuất; nguồn công khai nằm ở mục 17.

## 1. Vai trò, mục tiêu và phạm vi

Bạn là kỹ sư Computer Vision phụ trách repository Crowd Analysis hiện tại. Hãy đọc code, kiểm tra artifact, xác định nguyên nhân có bằng chứng và **sửa trực tiếp hệ thống**, không chỉ viết đề xuất.

Mục tiêu: từ video, phát hiện các **tuyến di chuyển phổ biến có hướng**, tổng hợp từ nhiều tracklet ẩn danh. Mỗi Common Path là một đường liên tục, có thể cong hoặc gấp khúc, có mũi tên đúng hướng. UI chọn được Top K; đường và màu phải ổn định nhưng vẫn thích nghi khi luồng người thay đổi.

**Không cần nhận diện danh tính, khuôn mặt hoặc ReID.** Giữ Track ID tạm để hình thành quỹ đạo và hạn chế đếm trùng; không coi nó là định danh con người.

Lỗi cần điều tra: ảnh kết quả hiện tại cho thấy các Common Path nổi bật tập trung ở phía dưới/tiền cảnh, trong khi người dùng quan sát thấy nhiều người đi ở phía trên và giữa. **Không hardcode đường lên trên, không tăng điểm cho vùng giữa, không chia quota hiển thị theo vùng.** Phải xác định đường bằng dữ liệu chuyển động theo thời gian.

### Bối cảnh được người dùng cung cấp — chưa được kiểm chứng độc lập

- Model: `yolo26n.pt`; tracker hiện tại dựa trên ByteTrack; engine `tracklet_aggregation`; inference trên Modal Tesla T4.
- Agent trước báo đã sửa detection sát biên tile, duplicate giữa tile, motion association, frame bị bỏ qua và che khuất ngắn.
- Agent báo `133 passed`; synthetic test hai người giao nhau và mất detection 2 frame giữ được ID.
- Smoke run mới: 60 frame/2 giây video; remote wall `28.06 s`; detection trung bình `207.3 box/frame`; observed track trung bình `143.72/frame`; tracking p95 `35.59 ms`; peak VRAM `839 MB`; peak RAM `5180 MB`; GPU utilization peak `50%`.
- Ảnh có HUD `People 386`, `FPS 0.0`, `Latency 527 ms`, nhiều marker vàng và một số đường ở phía dưới. Cần truy code để biết chính xác marker và từng con số biểu diễn gì.
- Video đã dùng trước đó: `data/videos/data-shibuya-test.mp4`, khoảng 65 giây. Hãy đọc metadata thực tế, không suy đoán frame count.

Các số trên là bối cảnh điều tra, **không chứng minh Common Path đúng**. Một ảnh tĩnh không xác định được hướng hay lưu lượng của cả video. `133 passed` và một smoke run 2 giây không thay thế kiểm chứng tuyến đường trên cửa sổ thời gian đủ dài.

### Ràng buộc

Giữ các bản sửa detector/tracker hiện có làm baseline. Không tự đổi model, đổi backend, chuyển TensorRT hoặc thêm optical-flow model nặng. Ngoài sửa upstream khi có bằng chứng làm mất/sai dữ liệu, **được bổ sung motion-ROI scheduler và tracking coverage-aware theo mục 14–17**, đặt sau feature flag, đo riêng với baseline. Không âm thầm tăng số lượt inference; periodic scan/fallback thay thế kế hoạch ROI tương ứng, có thống kê chi phí và cơ chế quay lại profile tham chiếu.

Không chuyển sang đường suy ra thuần từ grid/heatmap. Có thể dùng spatial index hoặc heatmap để tăng tốc/chẩn đoán; kết quả chính vẫn phải xuất phát từ tracklet.

---

## 2. Định nghĩa đúng đầu ra trước khi sửa

**Common Path** là hành lang quỹ đạo có hướng được nhiều lượt di chuyển ẩn danh hỗ trợ trong cùng cửa sổ quan sát, không phải đường của một người, đường dài nhất hoặc vùng có nhiều điểm đứng yên nhất.

Phân biệt rõ:

- **Occupancy:** có bao nhiêu người đang hiện diện.
- **Flow/support:** có bao nhiêu lượt di chuyển quan sát được hỗ trợ một hướng/tuyến.
- **Geometry:** đường đại diện nằm ở đâu.
- **Confidence/coverage:** bằng chứng đủ tin cậy và phủ được bao nhiêu phần đường.

Không xếp hạng theo diện tích bbox, độ dài pixel, số frame tồn tại, tổng số điểm, số segment hoặc số lần cập nhật. Một người đứng lâu hoặc một track dài không được lấn át nhiều lượt đi qua.

Không gọi số raw Track ID là số người thật. Khi còn ID fragmentation, dùng tên như `observed_passage_support`/“hỗ trợ từ lượt di chuyển quan sát được”, ghi giới hạn ước lượng.

Hai dòng ngược chiều trong cùng hành lang là hai directed paths. Đường giao nhau không mặc nhiên thuộc một tuyến. Số đường xuất ra là **tối đa K**, không phải bắt buộc đủ K khi thiếu bằng chứng.

---

## 3. P0 — Truy vết nguyên nhân trước khi đổi thuật toán

### 3.1. Đọc đúng code và cấu hình đang chạy

Bắt đầu từ các đường dẫn tương đối sau nếu tồn tại, rồi lần theo call graph:

```text
backend/app/inference/ultralytics_detector.py
backend/app/tracking/bytetrack_tracker.py
backend/app/video/pipeline.py
backend/app/core/config.py
configs/shibuya.yaml

Sau đó tìm:
tracklet_aggregation, tracklet filtering, candidate generation,
support/ranking, Top K selection, temporal smoothing,
coordinate conversion, common-path rendering, frontend overlay.
```

Không dựa vào line number cũ. Đọc resolved config thật của Modal và UI; ghi code commit, config hash, model hash, cache provenance. Chỉ ra nơi dữ liệu đi qua từng bước:

```text
detections → observed tracks → tracklet deltas → accepted segments
→ directed candidates → support/score → selected paths → rendered paths
```

### 3.2. Tạo bản đồ mất dữ liệu theo vùng

Tạo các ROI chẩn đoán phía trên, giữa, dưới phù hợp phần mặt đường nhìn thấy. Các ROI chỉ phục vụ so sánh; không tham gia cộng điểm hay bắt buộc chọn đường.

Xuất `region_funnel.csv`, thống kê theo vùng và khoảng thời gian video:

```text
detections
observed_tracks / predicted_only_tracks
new_track_ids / active_track_age_distribution
tracklet_deltas_in
tracklet_deltas_accepted
rejected_by_reason
accepted_motion_segments
candidate_count_before_top_k
selected_path_count
supported_path_length
```

Với segment đi qua nhiều ROI, chia phần hình học hoặc phân bổ nhất quán; không đếm cả segment vào mọi vùng rồi cộng tổng. Giữ denominator rõ ràng. Tỷ lệ observed tracks/detections chỉ là tỷ lệ nội bộ, không được gọi là recall.

Bổ sung overlay debug có thể bật/tắt: marker hợp lệ/bị loại, reason code, segment có hướng, candidate trước Top K và đường sau Top K. Chẩn đoán cả các candidate bị loại, không chỉ đường cuối.

### 3.3. Kiểm tra các giả thuyết sau, không coi chúng là kết luận

| Giả thuyết cần kiểm tra | Bằng chứng phải thu thập |
|---|---|
| Sai transform tile/resize/letterbox/canvas | Điểm và bbox trước–sau transform, kiểm tra round-trip và overlay trên frame gốc |
| Ngưỡng pixel ưu tiên tiền cảnh | Phân bố displacement, bbox scale, lý do loại tracklet theo vùng |
| Chỉ phát tracklet khi track kết thúc | Tỷ lệ active tracks có đóng góp, thời điểm segment đầu tiên được phát |
| Dùng wall-clock thay video time | Clock dùng cho velocity, history, TTL, decay và warmup |
| Đếm lặp lịch sử track mỗi lần update | Cùng segment/passage được nạp lại có tăng support không |
| Lọc quá chặt người xa/che khuất | Confidence, observed duration, gap, retained motion theo vùng |
| Sắp xếp/prune candidate quá sớm | Candidate trước/sau pruning, thứ tự iteration, cutoff Top K |
| Gộp sai dòng giao cắt/ngược chiều | Segment membership, signed direction và transition evidence |
| Smoothing/Path ID giữ nhầm đường cũ | Raw candidate so với stabilized candidate, tuổi bằng chứng |
| Cache/UI state cũ hoặc khác camera | session ID, timestamp, config version, payload version |

**Đặc biệt kiểm tra thời gian:** 2 giây video có thể mất 28 giây xử lý, nhưng motion, history window, support decay và tuổi bằng chứng vẫn phải chạy theo timestamp video. Không để pipeline chậm làm lịch sử hết hạn hoặc velocity bị tính sai.

Nếu thiếu artifact/source/video, ghi rõ `BLOCKED` ở phần tương ứng. Không báo đã kiểm chứng chỉ vì code chạy được.

---

## 4. P1 — Sửa dữ liệu hình học, thời gian và tracklet

### 4.1. Chuẩn tọa độ duy nhất

Giữ tọa độ phân tích rõ ràng và có metadata nguồn. Audit đầy đủ:

```text
model/letterbox coordinates
→ tile-local coordinates
→ original frame coordinates
→ analysis coordinates
→ render/canvas coordinates
```

Không áp dụng tile offset hoặc scale hai lần; không dùng kích thước ảnh screenshot để suy ra kích thước video. Kiểm tra frontend resize, letterbox và device-pixel ratio.

Ưu tiên điểm chân bbox `((x1+x2)/2, y2)` làm anchor mặt đường khi phù hợp. Đây là lựa chọn cần kiểm chứng: bbox bị che/cắt có thể làm điểm chân không tin cậy. Gắn quality/uncertainty cho trường hợp đó, không coi điểm dự đoán là phép đo thật.

Lưu source coordinates để render và kiểm chứng. Nếu biến đổi sang mặt phẳng phân tích, chiếu ngược đường về frame bằng đúng transform.

### 4.2. Giảm thiên lệch phối cảnh mà không tạo dữ liệu giả

Triển khai phương án nhẹ nhất có bằng chứng:

**Có calibration hợp lệ:** dùng homography mặt đường cho vùng gần phẳng đã được hiệu chỉnh. Kiểm tra reprojection, vùng hợp lệ và điểm gần singularity. Không tự dựng bốn điểm rồi gọi kết quả là tọa độ mét nếu không có tham chiếu thực. Homography mặt phẳng có điều kiện hình học, không phải hiệu chỉnh đúng mọi bề mặt [R1].

**Chưa có calibration:** dùng ngưỡng khoảng cách/chuyển động thích nghi với scale cục bộ, chẳng hạn bbox height đã lọc nhiễu hoặc scale map từ các quan sát đáng tin cậy. Ghi đây là xấp xỉ, không phải metric world coordinates.

Yêu cầu cho fallback:

- Không dùng một `min_displacement_px`, `merge_distance_px` cho toàn cảnh nếu số liệu cho thấy loại mất người xa.
- Chuẩn hóa displacement bằng scale tham chiếu cục bộ có clamp, được giữ nhất quán khi so sánh các segment.
- Không tạo “tọa độ chuẩn hóa” bằng cách chia từng điểm tuyệt đối cho bbox height biến thiên của chính nó.
- Dùng nhiều quan sát theo thời gian để phân biệt chuyển động nhỏ với jitter; không đơn giản hạ mọi threshold.
- Không bù recall bằng cách nhân support người xa theo hệ số tùy ý.
- Nếu dùng homography phi-metric, không báo vận tốc m/s hoặc sai lệch theo mét.

### 4.3. Tracklet incremental, có hướng, không đếm lặp

Mỗi quan sát nên mang tối thiểu:

```text
camera/session, source_frame_id, video_timestamp,
temporary_track_id, bbox/anchor, observed_or_predicted,
quality, preprocessing_version
```

Phát segment từ cả **active tracks**, theo phần dữ liệu mới; không chờ người rời khung hình. Tách rõ `source_frame_id` và số thứ tự frame đã xử lý. Dùng PTS/video timestamp cho `dt`; chỉ fallback `frame_id/source_fps` khi phù hợp nguồn CFR.

Gán khóa idempotent cho segment, ví dụ:

```text
(session, track_id, generation, start_source_frame, end_source_frame)
```

Một delta chỉ được ingest một lần. Segment chồng lấn từ cùng track phải có cơ chế chống tăng support.

Loại hoặc giảm tin cậy cho jump, teleport, `dt <= 0`, duplicate points, gap quá lớn, đoạn chủ yếu dự đoán và chuyển động không vượt noise floor. Mỗi quyết định loại phải có reason code.

Điểm prediction có thể duy trì state qua che khuất ngắn, nhưng không tự tạo thêm lượt đi hay support mới. Không nối qua gap dài chỉ để làm đường đẹp.

### 4.4. Không nhầm nối ID với gom luồng

Đây là hai việc khác nhau:

- **Stitch track bị đứt:** chỉ nối khi thời gian, motion, vị trí và tính duy nhất của phép ghép đủ thuyết phục; không ghép hai observed tracks đồng thời thành một người. Khi mơ hồ, giữ tách và báo uncertainty.
- **Aggregate Common Path:** các người khác nhau đi cùng tuyến ở thời điểm khác nhau trong cửa sổ vẫn được góp support. Không yêu cầu timestamp của tracklet từ hai người phải liên tiếp như cùng một track.

Tách trajectory thành các đoạn chuyển động cục bộ; không yêu cầu toàn bộ track có một hướng duy nhất, vì người có thể rẽ.

---

## 5. P2 — Gom luồng có hướng và dựng đường có bằng chứng

Ưu tiên sửa engine hiện tại. Chỉ thay cấu trúc khi diagnostic chứng minh cách cũ không thể đáp ứng. Hướng triển khai gợi ý là **gom các đoạn quỹ đạo cục bộ rồi dựng đường đại diện**, thay vì chỉ gom toàn trajectory; đây là ý tưởng tham khảo từ partition-and-group, không yêu cầu triển khai nguyên bộ TRACLUS [R2].

### 5.1. Association dựa trên hình học và hướng cục bộ

So sánh segment bằng khoảng cách vuông góc tới corridor, overlap/khoảng cách dọc tuyến, scale, hướng tiếp tuyến và quality.

- Dùng hướng có dấu; không dùng `abs(dot)`/cosine tuyệt đối để gộp hai chiều đối nhau.
- Gần nhau trong ảnh chưa đủ để gộp, nhất là nơi hai dòng cắt nhau.
- Dùng tangent cục bộ; không áp một hướng trung bình toàn cục lên đường cong.
- Resample theo chiều dài cung trong hệ tọa độ phù hợp, có giới hạn số điểm và không tăng trọng số track nhiều frame.
- Dùng spatial index và giới hạn neighbor search; tránh all-pairs toàn lịch sử.
- Pruning bộ nhớ không được mặc nhiên giữ riêng track dài, bbox lớn hoặc vùng dưới.

### 5.2. Nối các đoạn thành tuyến mà không tạo “đường tưởng tượng”

Các phần nối phải có hỗ trợ chuyển tiếp từ quỹ đạo quan sát hoặc bằng chứng liên tục đủ mạnh, không chỉ vì hai đầu mút gần nhau.

Tại giao cắt/rẽ nhánh, giữ thông tin hướng vào–hướng ra từ segment của cùng lượt đi khi có. Không ghép nhánh vào đông nhất với nhánh ra đông nhất để tạo tuyến chưa được quan sát.

Không bắt buộc một người phải được track hết toàn tuyến mới được tạo Common Path. Có thể nối các common subpaths có phần overlap/transition được nhiều quan sát hỗ trợ. Nhưng nếu chỉ biết các phần rời rạc mà chưa có bằng chứng kết nối, xuất subpaths riêng.

Không kéo đường xuyên khoảng trống dài hoặc vượt khỏi vùng có support. Không ép mọi nhánh vào một đường.

### 5.3. Dựng centerline

Dựng polyline đại diện bằng medoid hoặc thống kê robust của các điểm đã căn chỉnh theo tiến trình dọc tuyến. Không lấy trung bình điểm cùng index của các tracklet khác độ dài/khác điểm bắt đầu.

Làm mượt có giới hạn sai lệch so với corridor được hỗ trợ. Spline chỉ dùng khi không overshoot, không cắt góc sai, không tạo vòng hoặc kéo đường ra khỏi evidence. Đường gấp khúc đúng tốt hơn đường cong đẹp nhưng sai.

Mỗi phần đường phải truy được segment/passage hỗ trợ. Xuất confidence/coverage theo chiều dài, không chỉ một score chung.

---

## 6. P3 — Xếp hạng theo lượt di chuyển, không theo độ “to/dài” của track

Đây là phần bắt buộc audit nếu candidate ở vùng trên/giữa tồn tại nhưng luôn thua Top K.

### 6.1. Support chống trùng

Một `anonymous_passage` là lượt di chuyển tạm quan sát được, không phải danh tính. Các segment cùng một lượt không được nhân support theo số frame hoặc tần suất update.

Với mỗi đoạn của candidate, đếm passage phù hợp corridor và hướng. Một passage góp tối đa một đơn vị tại đoạn đó trong một lần đi qua. Có thể tái ghi nhận lượt quay lại khi có quy tắc re-entry rõ ràng, không chỉ dựa vào ID mới.

Ghi riêng:

```text
raw_track_ids
raw_segments
observed_passage_support
deduplicated_support
weighted_support
local_support_along_path
```

Nếu stitching không chắc chắn, không giả vờ đã đếm đúng người. Chống ảnh hưởng bằng giới hạn đóng góp của fragment và công bố uncertainty. Không hard-merge người gần nhau để giảm số ID.

### 6.2. Score dễ giải thích

Đặt điều kiện hợp lệ trước khi ranking: đủ lượt hỗ trợ cục bộ, hướng nhất quán, đủ chiều dài có evidence, không nối qua gap không được hỗ trợ.

Sau đó chọn score có ý nghĩa, **không cộng thô support của tất cả segment dọc tuyến**, vì đường dài sẽ tự được thưởng nhiều lần.

Một phương án khởi đầu cần kiểm chứng:

```text
local_support(s)
    = support ẩn danh chống trùng tại vị trí dọc tuyến s

route_support_proxy
    = thống kê robust đã chuẩn hóa theo chiều dài của local_support(s)
      trên phần liên tục đủ bằng chứng

ranking
    = route_support_proxy là tiêu chí chính
      + coverage/direction consistency là gate hoặc tie-breaker
```

Nêu rõ thống kê đã chọn, cách xử lý phần yếu và tác động tới tuyến ngắn/dài. Đây là **proxy hỗ trợ luồng**, không mặc nhiên là tổng số người đi hết tuyến. Nếu báo flow rate thực, phải có định nghĩa lượt qua đường cắt/corridor và thời gian quan sát tương ứng.

Không lấy detector confidence làm hệ số tuyến tính khiến người xa luôn mất ưu thế; dùng nó cho quality/uncertainty có kiểm chứng. Không xếp hạng theo cumulative support toàn phiên khi UI đang yêu cầu luồng gần đây. Dùng cửa sổ thời gian chung; công bố rõ độ dài cửa sổ và quy tắc decay, nếu có.

### 6.3. Candidate và Top K

Sinh một pool candidate đủ rộng trước khi Top K, có giới hạn tài nguyên. Không dừng ngay khi tìm được K đường đầu tiên.

Loại đường trùng theo directed corridor overlap và support membership; không xóa hai chiều đối nhau hoặc hai nhánh thật chỉ vì chung một đoạn đầu.

Mỗi candidate được chọn/bị loại phải có thông tin:

```text
candidate_id, geometry, score_components, support,
evidence_coverage, direction_consistency, decision_reason
```

Không đặt quota theo vùng. Đường phía dưới vẫn phải thắng nếu evidence thực sự mạnh hơn; đường phía trên/giữa phải thắng khi support của chúng mạnh hơn.

---

## 7. P4 — Ổn định hình học, mũi tên, Top K và UI

Ghép candidate giữa các lần cập nhật bằng corridor, overlap, direction và vị trí; Path ID không phụ thuộc thứ hạng.

- Giữ màu theo Path ID, không theo index trong danh sách.
- Làm mượt hình học sau khi căn chỉnh chiều và vị trí tương ứng theo chiều dài cung; không EMA mù hai mảng điểm khác nghĩa.
- Tách raw geometry và stabilized geometry để đo lỗi; smoothing không được che một tuyến sai.
- Hysteresis chỉ tránh thay thế khi điểm gần nhau; phải cho đường có bằng chứng tốt hơn thay thế sau thời gian xác nhận hữu hạn.
- Khi support giảm dưới ngưỡng trong cửa sổ đang xét, chuyển trạng thái có kiểm soát rồi ẩn; không giữ vô hạn đường lịch sử.
- Phân biệt thiếu frame đầu vào với thực sự không còn lượt đi qua. Tuổi evidence dùng video time; heartbeat mạng có thể dùng wall time nhưng không được làm sai thống kê chuyển động.
- Đổi Top K chỉ chọn lại từ candidate pool, không reset detector/tracker/lịch sử.
- Đổi các tham số ảnh hưởng hình học cần rebuild có version từ buffer hợp lệ; không để kết quả cũ ghi đè config mới.
- Seek, loop video, đổi camera hoặc đổi nguồn phải xử lý discontinuity và state namespace rõ ràng.

Mũi tên được đặt theo tangent của **đường cuối cùng được render** và chiều chuyển động có support, không suy từ thứ tự sort theo x/y. Có mũi tên rõ ở cuối và tùy chọn nhắc lại dọc đường. Không đổi chiều chỉ vì nhiễu tức thời.

Tách layer debug và layer sản phẩm. Mặc định không phủ dày mọi marker vàng, nhãn IDs và đường thử nghiệm. Nhãn path gọn: ID, support quan sát, cửa sổ thời gian hoặc confidence khi cần. Không đổi ngữ nghĩa “People” âm thầm; sửa tên/tooltip nếu đang biểu diễn active tracks hay cumulative IDs.

Audit `FPS 0.0` và latency trong HUD, nhưng không kết luận từ một ảnh rằng pipeline bị treo. Ghi rõ FPS xử lý, FPS playback và FPS trình duyệt là các số khác nhau.

---

## 8. Pipeline triển khai: sửa đúng và không làm nặng thêm

```text
Video PTS
→ detector/tracker hiện tại
→ observed tracklet deltas
→ geometry/quality normalization
→ incremental directed aggregation
→ deduplicated local support
→ candidate ranking
→ temporal stabilization
→ versioned paths payload
→ cached overlay/render
```

Khi bật hybrid, detector/tracker trong sơ đồ trên được điều phối bằng motion-ROI scheduler ở mục 14. Giữ toàn cảnh cho bước motion nhẹ và periodic coverage scan; không cắt bỏ vĩnh viễn vùng ảnh khỏi nguồn video. Motion mask không đi thẳng vào aggregation, support hay ranking.

Chạy tổng hợp đường theo nhịp riêng; không clustering toàn lịch sử mỗi frame. Buffer và index phải có giới hạn, expire theo video time, cập nhật cả phần đóng góp bị loại khỏi cửa sổ. Không chỉ thêm mà quên trừ support cũ.

Render lại hình học overlay khi path/config thay đổi, không tạo spline/mũi tên lại vô ích trên mỗi frame. Không đẩy ảnh/base64 hoặc toàn bộ track history khi UI chỉ cần path delta. Giữ phương án truyền video đang hoạt động nếu chưa chứng minh nó là bottleneck.

Offline benchmark phải xử lý đầy đủ frame theo lịch đã khóa, không âm thầm drop để tăng FPS. Realtime có thể dùng bounded queue/latest-frame policy, nhưng phải công bố frame bị bỏ, dùng timestamp đúng và không cho inference worker cập nhật tracker sai thứ tự.

Không đưa toàn bộ Common Path lên GPU chỉ vì Modal có GPU. Profile rồi sửa hotspot; ưu tiên tính gia tăng, vectorization và neighbor indexing.

---

## 9. Cấu hình khởi đầu — không phải thông số đã tối ưu

Map các ý nghĩa sau vào schema hiện có; không tạo key bị silently ignored. Đây là ví dụ thử nghiệm, không được tự nhận là preset tốt nhất:

```yaml
common_path:
  top_k: 3
  max_candidates: 24
  history_window_s: 30.0
  update_interval_s: 0.5
  emit_active_tracklet_deltas: true

  geometry:
    anchor: bottom_center
    normalization: local_scale   # Hoac calibrated_ground_plane khi co calibration hop le.
    calibration_file: null

  support:
    unit: anonymous_passage
    min_local_passages: 5
    min_supported_length_ratio: 0.70

  stabilization:
    confirm_s: 1.5
    replace_relative_margin: 0.15
    smoothing_time_constant_s: 1.0

  render:
    directional_arrows: true
    show_debug_points: false
    show_rejected_segments: false

  diagnostics:
    region_funnel: true
    candidate_decisions: true
```

Xác định riêng các ngưỡng noise, gap, segment duration, distance, direction từ phân bố dữ liệu và kiểm chứng. Nếu cấu hình trên làm mất luồng hợp lệ, điều chỉnh có ablation; không chỉ hạ mọi ngưỡng để xuất được nhiều đường.

Snapshot resolved config và ghi rõ tham số nào vừa đổi. Validate quan hệ `max_candidates >= top_k`, đơn vị thời gian/khoảng cách và calibration mode. UI chỉ expose các tham số thật sự cần thao tác, không tạo thêm màn hình cấu hình lớn.

---

## 10. Kiểm chứng tiết kiệm GPU nhưng đủ chứng minh chất lượng

### 10.1. Replay cùng dữ liệu để cô lập thuật toán

Ưu tiên dùng tracking cache của bản detector/tracker mới nhất để so sánh Common Path cũ và mới trên **cùng quan sát**.

Kiểm tra cache có đủ frame IDs, timestamps, bbox/anchor, observed flag và version. Cache cũ thiếu thông tin không được tự suy thành dữ liệu đã quan sát. Khi cần, chạy lại một lần trên Modal T4 để tạo cache hợp lệ.

Cache key phải bao gồm video/source segment, model/config inference, tiling, preprocessing, tracker/config/version, lịch sampling và schema. Đổi downstream Common Path không tự làm mất cache upstream hợp lệ. Nếu thay detector/tracker, refresh tương ứng.

A/B replay chỉ chứng minh tác động downstream; không chứng minh detection recall. Đo end-to-end thật riêng, không báo FPS replay-cache như FPS inference.

**Ngoại lệ quan trọng khi đánh giá hybrid:** scheduler làm thay đổi những vùng thực sự được detector quan sát, nên phải chạy inference/tracking thật với cache riêng. Không dùng chung tracking cache đầy đủ rồi che bớt output để tuyên bố hybrid giữ recall hoặc tăng FPS. Replay chung chỉ dùng để cô lập sửa đổi downstream như ở trên; A/B upstream thực hiện theo mục 16.

### 10.2. Chạy toàn video và so sánh cùng timestamp

Chạy video khoảng 65 giây từ đầu đến cuối cho baseline hiện tại và bản sửa, giữ cùng dữ liệu đầu vào. Xuất snapshot tại cùng video timestamp và clip overlay động có mũi tên.

Đọc toàn video để chọn các khoảng có người di chuyển, chờ và chuyển luồng; không suy hướng từ ảnh tĩnh. Đánh giá phần đầu khi chưa đủ support riêng với phần sau khi cửa sổ đã tích lũy đủ dữ liệu.

Không dùng một run 2–5 giây để kết luận đường ổn định hoặc không có tuyến đông hơn.

### 10.3. Đối chiếu độc lập với chuyển động thực

Tạo bộ review gọn, phủ phía trên/giữa/dưới và những khoảng thời gian khác nhau. Dùng video gốc không có overlay để đánh dấu các corridor/hướng chuyển động rõ và một số lượt đi qua đại diện.

- Annotation tự sinh từ output đang kiểm tra không được gọi là ground truth.
- Dùng cùng bộ review cho baseline và bản sửa; không đổi ground truth cho khớp kết quả.
- Chỉ báo precision/recall/lưu lượng chính xác trên phần đã được gán nhãn đủ.
- Khi vùng bị che quá nhiều hoặc reviewer chưa xác minh, ghi `UNKNOWN`/`REVIEW_PENDING`.
- Nếu chưa có khả năng xem video hoặc xác nhận nhãn, vẫn xuất artifact để review nhưng không báo chất lượng `PASS`.

Nếu vùng trên/giữa có rất ít observed tracklets hợp lệ, ghi rõ bottleneck upstream và minh họa. Khi đó sửa riêng điều kiện lọc/association liên quan, không “vẽ bù” từ occupancy. Nếu cần kiểm tra ByteTrack, đối chiếu cách xử lý detection confidence thấp với cơ chế association hai bước gốc, thay vì mặc định bỏ toàn bộ detection yếu [R3].

### 10.4. Test hồi quy tập trung

Bổ sung test xác định được kỳ vọng, không mở rộng suite không liên quan:

| Tình huống | Kỳ vọng |
|---|---|
| Nhiều lượt cùng hướng ở vùng xa; ít lượt có bbox lớn ở vùng gần | Tuyến nhiều lượt thắng khi quality tương đương, không ưu tiên pixel length |
| Một nhóm đứng yên lâu và một nhóm đi qua | Nhóm đứng yên không tạo dominant directed path |
| Hai chiều đối nhau, hai luồng giao nhau, đường rẽ cong | Tách đúng hướng/nhánh; không tạo tuyến nối sai qua giao cắt |
| Track còn active, mất detection ngắn, ID fragmentation | Có đóng góp sớm; prediction không sinh support; giới hạn đếm trùng |
| Ingest lặp delta, đổi FPS/sampling, tốc độ xử lý khác nhau | Support không tăng giả; time window không phụ thuộc wall-clock |
| Cùng geometry được biểu diễn ở độ phân giải khác | Tọa độ render và thứ hạng tương đương trong tolerance đã định |
| Đổi Top K và candidate đổi thứ hạng | Không reset track; giữ ID/màu; tuyến tốt hơn không bị khóa ngoài |
| Luồng mới xuất hiện, luồng cũ hết evidence, seek/đổi nguồn | Thích nghi hữu hạn; không giữ path ma hoặc trộn session |
| Hai subpath gần nhau nhưng không có transition evidence | Không bịa đường liên tục nối chúng |

Giữ các test đã có. Báo test chạy thật, tên và kết quả; không chỉ ghi tổng số passed.

---

## 11. Metrics, nghiệm thu và giới hạn kết luận

### Chất lượng

Xuất tối thiểu:

```text
region_funnel và rejection reasons
candidate ranking trước/sau Top K
observed/deduplicated support theo path và dọc path
direction agreement với nhãn review hoặc observed segments
supported-length ratio và unsupported-gap length
first_valid_path_time theo video time + startup wall time riêng
geometry jitter trên các khoảng luồng ổn định
Path ID/color churn, unexpected direction flips
thời gian thích nghi khi luồng thay đổi
```

Đo jitter bằng các vị trí đã được căn chỉnh trên phần đường tương ứng; không tính việc kéo dài một path thành rung toàn đường. Direction agreement với chính tracks đầu vào chỉ là consistency nội bộ; tách khỏi độ đúng với video đã review.

Không dùng “vẽ nhiều đường hơn”, “có đường ở phía trên”, “nhiều IDs hơn” hoặc “đường nhìn mượt hơn” làm tiêu chí chính xác.

### Hiệu năng

Trên cùng Modal T4 và cùng điều kiện, báo p50/p95 cho inference, tracking, aggregation/ranking, stabilization, render, encode; thêm pipeline FPS, wall time, RAM/VRAM, queue depth và dropped frames. Phân biệt cold start/model load với steady-state. UI FPS chưa đo thì ghi `NOT_RUN`.

Đặt budget Common Path theo baseline thực và công bố trước khi kết luận. Mốc thử nghiệm có thể là aggregation + stabilization p95 dưới 20 ms mỗi lần cập nhật, nhưng **đây chỉ là mục tiêu cần đo**, không phải lời hứa hoặc lý do bỏ evidence.

Không so smoke run 2 giây với full-video run rồi kết luận nhanh/chậm theo một tỷ lệ trực tiếp.

### Điều kiện nghiệm thu

Bản sửa chỉ đủ điều kiện đề xuất làm mặc định khi:

1. Có ít nhất một nguyên nhân gốc được chứng minh bằng code/config và artifact; không chỉ liệt kê giả thuyết.
2. Tuyến xếp hạng phù hợp bằng chứng lượt di chuyển trong cửa sổ; không thiên lệch do pixel size, số frame hay vùng ảnh.
3. Các tuyến có thể kiểm chứng ở trên/giữa được phát hiện khi đủ observed support, và tuyến dưới không bị loại vô cớ.
4. Không nối sai dòng giao cắt/ngược chiều; mỗi đoạn render có support truy vết.
5. Top K, mũi tên, ID/màu và cập nhật theo thời gian hoạt động đúng.
6. Có video đối chiếu dài đủ, test hồi quy liên quan và benchmark cùng điều kiện.
7. Các vùng thiếu dữ liệu và phần chưa kiểm chứng được ghi rõ, không gán `PASS`.

Không cần giữ production sai mãi để chờ một nghiên cứu hoàn hảo, nhưng phải có feature flag/config rollback và bằng chứng đủ trước khi thay mặc định.

---

## 12. Thứ tự thực hiện và bàn giao

Thực hiện theo checkpoint, ưu tiên sửa tối thiểu trước khi viết lại:

```text
A. Đóng băng baseline hiện tại, đọc artifact và trace luồng dữ liệu.
B. Thêm funnel theo vùng, audit tọa độ/time/support/Top K.
C. Sửa lỗi đã chứng minh; replay cùng cache để đo tác động từng nhóm thay đổi.
D. Chỉ bổ sung directed local aggregation/scale-aware filtering nếu còn cần.
E. Kiểm tra mũi tên, ổn định, config và UI.
F. Full-video review + một lượt xác nhận end-to-end trên Modal T4.
G. Báo cáo, bàn giao patch và config rollback.
```

**Checkpoint bổ sung cho hybrid:** sau khi có baseline Common Path sửa đúng, chạy motion ở shadow mode để audit ROI coverage; tiếp theo bật gating thật và so sánh B/C theo mục 16. Tách commit/config sửa chất lượng đường khỏi commit/config tiết kiệm inference. Có thể thêm instrumentation sớm nhưng không bật gating để che lỗi cũ.

Không tổ hợp hàng chục GPU experiments. Dùng replay/cache và test nhỏ để loại phương án trước, chỉ chạy GPU khi cần dữ liệu upstream hoặc xác nhận cuối.

Bàn giao:

```text
1. Code patch, file/hàm đã sửa và lý do.
2. Root-cause report: evidence, thay đổi, kết quả trước/sau, giới hạn.
3. Resolved configs và lệnh chạy thực tế; không bịa CLI flag.
4. region_funnel.csv, candidate_decisions.jsonl, path_support.jsonl.
5. Metrics trước/sau, manifest có provenance và version.
6. Video overlay baseline/fixed đồng bộ timestamp; snapshot có giải thích.
7. Tests đã chạy, phần PASS/FAIL/NOT_RUN/BLOCKED/REVIEW_PENDING.
```

Các artifact debug phải có sampling/giới hạn dung lượng và không log toàn lịch sử vô hạn. Không commit model/video/cache dung lượng lớn vào source nếu repository không quy định.

Trong báo cáo cuối, trả lời rõ:
**“Vì sao trước đây đường tập trung ở phía dưới? Bằng chứng nào cho thấy bản sửa chọn đúng luồng hơn, thay vì chỉ dịch đường lên trên hoặc làm đường mượt hơn?”**

---

## 13. Nguồn tham khảo kỹ thuật và phạm vi sử dụng

Bối cảnh, số liệu chạy và nhận xét ảnh ở mục 1 đến từ người dùng/agent trước, chưa được xác minh bởi người soạn prompt. Các thuật toán, config và tiêu chí còn lại là yêu cầu thiết kế/kiểm chứng đề xuất, không phải mô tả code đã đọc hay kết quả đã đo.

- **[R1] OpenCV — Basic concepts of the homography explained with code.**
  `https://docs.opencv.org/4.x/d9/dab/tutorial_homography.html`
  Tham khảo điều kiện phép chiếu giữa mặt phẳng và hiệu chỉnh phối cảnh; không chứng nhận calibration của camera hiện tại.

- **[R2] Lee, Han, Whang — Trajectory Clustering: A Partition-and-Group Framework, SIGMOD 2007.**
  `https://hanj.cs.illinois.edu/pdf/sigmod07_jglee.pdf`
  Tham khảo ý tưởng tìm common sub-trajectories bằng phân đoạn và gom nhóm. Directed association, incremental update, support chống trùng và ranking trong prompt là yêu cầu riêng, không được coi là nguyên bản TRACLUS.

- **[R3] Zhang et al. — ByteTrack: Multi-Object Tracking by Associating Every Detection Box.**
  `https://arxiv.org/html/2110.06864v3`
  Tham khảo association detection confidence cao/thấp và duy trì track qua quan sát yếu; không chứng minh bản tracker tùy biến hiện tại đã đúng.

Đã tham khảo nguồn công khai ngày 2026-09-25. Không cần nâng phiên bản thư viện chỉ để khớp phiên bản trang tài liệu.
---

## 14. P5 — Tự nhận diện vùng tĩnh và điều phối HYBRID MOTION-ROI

### 14.1. Mục tiêu và năm nguyên tắc bắt buộc

Mục tiêu bổ sung: giảm các lượt inference không cần thiết và chi phí association/aggregation, **không đánh đổi mất luồng người xa, người đi chậm hoặc người đứng chờ rồi di chuyển**.

Triển khai phương án hybrid người dùng yêu cầu:

1. Detector quét toàn cảnh định kỳ để tìm người mới, tái quan sát track và kiểm tra coverage.
2. Các frame còn lại dùng motion mask để đề xuất ROI, kết hợp ROI bảo vệ track và vùng cần tái quan sát.
3. Tracker tiếp tục dự đoán qua khoảng không có phép đo trong một grace period hữu hạn.
4. Khi motion/background không đáng tin, mất track bất thường hoặc lợi ích ROI không còn, quay lại profile toàn cảnh đã kiểm chứng.
5. Common Path chỉ lấy bằng chứng chuyển động từ track đã xác nhận với quan sát detector hợp lệ; không lấy foreground blob hoặc prediction thuần làm support.

**Làm rõ “hiệu chỉnh background”:** background subtractor có quá trình cập nhật riêng từ frame; detector không tự huấn luyện/cập nhật background model. Kết quả quét toàn cảnh giúp kiểm tra vùng bị bỏ sót và bảo vệ track. Cơ chế dùng detection để ảnh hưởng background learning, nếu có, phải được triển khai riêng, kiểm chứng API và ghi rõ [R4, R5].

**Vùng tĩnh = vùng ít thay đổi trong một khoảng quan sát, không phải vùng chắc chắn không có người.** Background subtraction phù hợp nhất với camera cố định; người đứng lâu có thể bị mô hình nền hấp thụ [R4, R5]. Không tạo polygon loại trừ vĩnh viễn chỉ vì vài giây đầu không có chuyển động.

Đây là tối ưu lịch tính toán, không phải thuật toán xác định Common Path mới. Không chuyển sang heatmap đường đi, optical flow thay tracking, nhận diện danh tính hoặc ReID.

### 14.2. State machine tối thiểu và thứ tự xử lý

Dùng các trạng thái rõ ràng, có reason code:

```text
WARMUP        : học nền và chạy reference coverage; chưa tin mask để bỏ inference.
HYBRID        : chọn motion/track-protection ROI, vẫn có deadline quét toàn cảnh.
FULL_COVERAGE : quét toàn cảnh theo deadline hoặc do cost/coverage guard.
RECOVER       : mask/camera/coverage bất thường; dùng reference profile và học lại nếu cần.
```

`FULL_COVERAGE` có thể chỉ là một lượt quét trong trạng thái hybrid; không reset tracker sau mỗi lần đổi chế độ. Chỉ trở về `HYBRID` khi bằng chứng ổn định đủ lâu, tránh bật/tắt liên tục.

Luồng logic:

```text
frame gốc + source PTS
→ kiểm tra discontinuity, tạo ảnh nhỏ CHƯA CÓ overlay
→ motion/background update và mask-health
→ snapshot dự đoán track tại PTS hiện tại
→ scheduler chọn reference scan / ROI scan / intentional skip
→ detector trên ảnh/crop RGB gốc, không trên ảnh bị bôi đen
→ map bbox về frame gốc + merge duplicate
→ coverage-aware tracking, cập nhật phép đo và trạng thái
→ observed tracklet deltas
→ Common Path incremental + expire evidence theo video time
→ overlay có cache + UI
```

Kalman prediction/state advance phải xảy ra **đúng một lần** cho mỗi bước thời gian được xử lý. Nếu tracker hiện tại tự predict trong `update`, planner chỉ dùng snapshot dự đoán không mutate state hoặc tách API rõ; không predict hai lần vì thêm ROI planner.

Tách riêng `source_frame_id`, PTS và thời gian xử lý. Mọi deadline scan, tuổi mask, grace, static dwell và history dùng video time. Wall time chỉ dùng performance/network timeout.

### 14.3. Motion/background nhẹ, tự thích nghi nhưng không làm mất người xa

Bắt đầu bằng MOG2 có sẵn trong OpenCV trên CPU với frame nhỏ; KNN chỉ là phương án đối chiếu khi có lý do. Không thêm dependency nặng hoặc mặc định OpenCV CUDA đã có. Khóa và ghi phiên bản OpenCV thực tế [R4].

Yêu cầu:

- Tính mask từ frame gốc trước khi vẽ marker, HUD, arrow và Common Path. Overlay không được trở thành “chuyển động” đầu vào.
- Giữ đúng aspect ratio và transform mask ↔ frame gốc. Khởi đầu thử chiều rộng 640 cho nguồn 1280; kiểm tra riêng người nhỏ phía trên/giữa trước khi giảm tiếp.
- Mỗi frame đã xử lý có motion check nhẹ, hoặc cadence riêng được đo với tuổi mask tối đa. Không dùng mask cũ vô hạn cho frame mới.
- Đánh dấu vùng “ít chuyển động” sau một khoảng liên tục; chỉ cần có tín hiệu mới hợp lệ là đánh thức nhanh. Temporal smoothing không được trì hoãn người mới đến chỉ để mask trông đẹp.
- Lọc nhiễu/morphology nhỏ có kiểm soát. Không dùng `min_component_area` hay kernel lớn cố định làm xóa người xa; audit footprint của người nhỏ trong ảnh mask.
- Bóng đổ cần được phân biệt với foreground khi backend hỗ trợ. MOG2 mặc định phân biệt background 0, shadow 127 và foreground 255; đọc cấu hình thực tế, không mặc định mọi `mask > 0` là người/chuyển động cần detector [R5].
- Không dùng motion blob để đếm người. Xe, cây, bóng và màn hình cũng phải được coi là nguồn ROI cần đánh giá, không phải person.
- Mask rỗng, bão hòa hoặc nền chưa ổn định không tự động chứng minh toàn cảnh an toàn để skip.
- Không yêu cầu có người/track trước mới cho motion xuất hiện; bootstrap phải phát hiện được vùng mới chưa từng có track.

Đề xuất điều khiển thời gian học nền theo `dt` video, chẳng hạn `alpha = 1 - exp(-dt/tau)` có clamp. Đây là heuristic cần kiểm chứng, không phải bảo đảm MOG2 bất biến tuyệt đối với sampling. Audit cả `history` và learning rate để tránh dùng cùng giá trị nhưng thời gian học thực khác nhau giữa offline/realtime.

Không reset background mỗi lần full scan. Nếu muốn bảo vệ pixel trong bbox người khỏi background learning, kiểm tra overload/API của **bản OpenCV đang cài** và có test. Nếu không hỗ trợ, dùng track-protection ROI + periodic scan; không bịa tham số hoặc chèn pixel đen vào model nền [R5].

### 14.4. ROI phải có cơ chế bảo vệ và không tạo vòng lặp bỏ sót

ROI đề xuất không chỉ đến từ mask:

```text
candidate_ROIs =
    motion regions
  ∪ moving-track predicted corridors
  ∪ track regions đến hạn tái quan sát
  ∪ recovery regions của track còn trong grace
  ∪ vùng tạm uncertain/chưa được quan sát đủ
```

ROI track cần được mở rộng theo bbox scale, uncertainty và khoảng thời gian từ phép đo cuối; clamp hợp lý. Không đặt một padding pixel cố định cho mọi độ sâu. Vùng toàn cảnh luôn được tái quét theo deadline, kể cả khi mask không báo gì.

Với track đứng yên, có thể giảm nhịp detector trong grace nhưng vẫn giữ deadline tái quan sát. Không bắt mọi bbox đứng yên thành một crop chạy trên mọi frame; cũng không để track đứng yên không bao giờ được detector kiểm tra lại.

**Bảo vệ người mới/luồng mới:**

- Global scan không được phụ thuộc vào việc đã có Common Path, đã có Track ID hay ROI có score cao.
- Không giới hạn ROI theo Top K đang hiển thị. Đổi Top K không được đổi coverage detector.
- Không bỏ ROI nhỏ phía trên/giữa chỉ vì ROI phía dưới lớn hơn. Khi ngân sách chật, dùng thời hạn quan sát/độ bất định, không chỉ sắp xếp theo diện tích.
- ROI bị hoãn phải có deadline; đến hạn thì gộp hợp lý hoặc fallback, không bỏ âm thầm.
- Không nhân support để “bù” cho vùng ít được scan. Giữ nguyên quy tắc support chống trùng và công bố giới hạn coverage.

### 14.5. Quét toàn cảnh phải giữ khả năng nhìn thấy người nhỏ

“Full-frame” không mặc nhiên có chất lượng ngang profile tiled trước đó.

Định nghĩa **reference coverage profile** từ config/commit đã kiểm chứng. Nếu cần `1 full frame + 4 tiles` để giữ người xa, periodic refresh/fallback phải giữ coverage tương đương đó cho tới khi có bằng chứng profile rẻ hơn đủ tốt.

Cách triển khai ít rủi ro trước:

- Tái sử dụng geometry, overlap, input size và detector của các tile hiện có.
- Trên frame hybrid, chỉ kích hoạt tile/ROI đáp ứng motion hoặc track-protection; không nhất thiết thêm full-frame pass ở tất cả các frame này.
- Trên frame periodic/fallback, chạy reference scan thay cho kế hoạch ROI; không chạy hai kế hoạch trùng nhau rồi tính là “tối ưu”.
- Tile ở đây chỉ chia workload detector, không phải grid dùng để suy ra tuyến đường.
- Dynamic crop/input-size optimization là bước sau có A/B riêng; không đổi đồng thời sampling, resolution, precision và tracker thresholds.

Detection trong ROI phải map về hệ tọa độ gốc trước khi tracking. Giữ một tracker cho toàn cảnh, không tạo tracker độc lập mỗi ROI. Giữ các sửa lỗi tile border/duplicate đã có; không loại bbox hợp lệ chỉ vì motion overlap thấp, và không hard-merge hai người cạnh nhau.

### 14.6. Cost guard: phải giảm công việc thật, không chỉ tô mask

**Không bôi đen vùng tĩnh rồi vẫn infer cùng tensor và gọi đó là giảm tải.** Cũng không cắt hàng chục ROI rồi resize từng ROI lên cùng input size lớn và mặc định nhanh hơn. Ultralytics có bước resize/padding theo `imgsz`, `rect`, batch và backend; ghi kích thước tensor thực thay vì suy chi phí từ diện tích crop gốc [R6].

Đo riêng:

```text
motion_ms, roi_planning_ms, crop_preprocess_ms
model_invocations, inference_images, actual_tensor_shapes
sum_tensor_pixels, batch_size, padding_overhead
global_scans, selected_tiles, ROI_union_area, estimated/actual_ROI_cost
```

Một model call chứa batch N ảnh không phải chỉ xử lý một ảnh. `sum_tensor_pixels` là proxy workload, không phải FLOPs/latency chính xác.

Dùng giới hạn số ROI, merge overlap có kiểm soát và cost estimate từ đo thật. Khi ROI phủ gần toàn cảnh, quá phân mảnh hoặc ước tính đắt ngang reference, chọn reference scan. Diện tích union chỉ là một tín hiệu; không dùng nó làm cost model duy nhất.

Khi cả cảnh đông người chuyển động, hybrid có thể tiết kiệm ít hoặc không tiết kiệm. Chấp nhận giữ reference profile ở đoạn đó; không giảm recall để cố đạt con số speedup. Có hysteresis/cadence nhẹ cho cost guard để tránh overhead và dao động chế độ.

Không cam kết `2x/5x/30 FPS` khi chưa benchmark. Mục tiêu là giảm chi phí end-to-end ở các đoạn phù hợp, đồng thời không làm xấu đáng kể đoạn đông.

### 14.7. Tracking phải phân biệt “chưa quan sát” và “quan sát nhưng không thấy”

Thêm observation coverage theo frame/ROI. Với mỗi track, phân biệt:

```text
MEASURED                  : có detection thực được association.
SEARCHED_NOT_FOUND        : vùng dự kiến đã được detector quan sát đủ nhưng không có match.
NOT_SEARCHED_BY_POLICY    : scheduler chủ động chưa chạy detector ở vùng này.
```

Coverage phải xét bbox/corridor dự đoán và biên crop; ROI chạm một phần bbox không mặc nhiên là quan sát đầy đủ. Xuất reason code cho ca mơ hồ.

Không gọi `update([])` theo cách tracker hiểu mọi track đều bị miss khi detector thực ra không chạy ở đó. ByteTrack gốc có xử lý unmatched/lost bên trong `update`; phải audit wrapper/tùy biến đang dùng trước khi thay lịch inference [R7].

Yêu cầu state:

- `last_measurement_pts` chỉ đổi khi có observation thực, không được refresh bằng prediction hoặc trạng thái “vùng tĩnh”.
- `NOT_SEARCHED_BY_POLICY` không bị tính như detector miss, nhưng tuổi phép đo và uncertainty vẫn tăng; không được bất tử.
- Đặt deadline tái quan sát trước khi hết grace. Nếu không thể giữ deadline vì ngân sách, fallback hoặc expire đúng hạn, không kéo dài grace âm thầm.
- Giữ track qua che khuất ngắn bằng prediction, nhưng không ép giữ cùng ID khi hai người giao cắt mà association mơ hồ.
- Điều chỉnh mô hình chuyển trạng thái và covariance/process noise phù hợp `dt`; chỉ nhân displacement với số frame bỏ qua là chưa đủ để mặc định filter đã đúng.
- Association chỉ xét neighbor hợp lý theo không gian, thời gian và coverage; giữ cơ chế low-score association có ích. Không ghép lại toàn bộ mọi track với mọi detection khi có thể prune an toàn.
- Không tạo trajectory point giả “đã quan sát” từ Kalman prediction. Phân biệt số người/track observed hiện tại với số track đang coasting trong HUD.

Grace để vượt gap ngắn, không để suy ra quỹ đạo dài khi người đứng yên đã bị model nền hấp thụ.

### 14.8. Common Path không được phụ thuộc trực tiếp vào mask

Chỉ phát delta từ các phép đo hợp lệ của track đã xác nhận và đủ bằng chứng chuyển động. Giữ lọc noise, scale-aware geometry, signed direction và support chống trùng của mục 4–7.

Một cặp điểm observed trước/sau gap ngắn có thể tạo segment nếu thỏa điều kiện liên tục đã kiểm chứng. Không nội suy qua gap dài/khúc rẽ không quan sát rồi biến thành route evidence.

Vùng mask tĩnh không được xóa Common Path lập tức. Ngược lại, khi không có delta mới, history/TTL/support decay vẫn phải tiến theo video time; không giữ đường cũ vô hạn bằng cách bỏ toàn bộ analytics.

Tối ưu phần phân tích bằng:

```text
có delta mới hoặc đến hạn expire/support/stabilization
    → cập nhật phần bị ảnh hưởng
không có thay đổi và chưa đến hạn
    → tái dùng snapshot/path overlay
```

Không nhân trọng số một observed segment vì nó đại diện cho nhiều frame bị skip. Không để thay đổi nhịp inference tự làm tuyến có nhiều lượt detector hơn thắng Top K.

### 14.9. Trigger phục hồi an toàn

Bật reference scan khi đến deadline, nền chưa đủ tin cậy, ROI budget vượt ngưỡng hoặc có dấu hiệu suy giảm coverage.

Mất một track trong cảnh đông **không bắt buộc global scan ngay mọi lần**. Ưu tiên local rescue; global fallback dựa trên tỷ lệ/số lượng track `SEARCHED_NOT_FOUND` bất thường trong một cửa sổ, có minimum sample count. Track chủ động chưa được scan không tính vào tỷ lệ detector miss.

Phân biệt:

- Thay đổi ánh sáng/auto-exposure: reference scan, giảm tin mask, cập nhật/relearn nền có kiểm soát; không tự reset mọi track/path.
- Camera rung/di chuyển: không tin motion toàn ảnh. Fallback; nếu hệ tọa độ thực sự đổi mà chưa bù được, invalidate state không còn hợp lệ.
- Seek/loop/đổi camera/scene cut: xử lý discontinuity, namespace và reset phù hợp; không nối path giữa hai cảnh khác nhau.
- Xe đi qua/cây rung/màn hình: không reset liên tục chỉ vì foreground ratio cao; xem đây là tình huống motion mask ít chọn lọc và dùng cost guard.
- Mask lỗi/NaN/sai shape/quá cũ: fail-safe về reference, log có giới hạn; không crash cả stream.

Có cooldown/hysteresis cho trigger mềm, nhưng camera discontinuity hoặc deadline an toàn phải được ưu tiên. Ghi rõ nguyên nhân và thời gian phục hồi; không che fallback kéo dài bằng cách báo đang hybrid.

### 14.10. Tích hợp gọn vào pipeline/Modal

Đặt module nhỏ trong cấu trúc hiện có: motion estimator, ROI scheduler và coverage metadata. Không xây thêm service/distributed framework chỉ cho tính năng này.

Mỗi camera/session có background, scheduler, tracker và timestamp state riêng. Model có thể được reuse theo kiến trúc hiện tại, nhưng không chia sẻ background hoặc Track ID giữa hai video. Chặn kết quả frame/config cũ ghi vào session mới.

Không copy frame GPU↔CPU nhiều lần chỉ để tạo mask. Dùng frame đã decode, tái dùng buffer ảnh nhỏ và ROI metadata. Đo cả CPU/RAM, crop/encode và transfer overhead, không chỉ GPU utilization.

Bật/tắt feature qua config/UI cần version hóa. `Top K` chỉ tác động chọn đường. Thay config motion ảnh hưởng background phải có warmup/reset thích hợp; không reload model không cần thiết.

Mặc định production giữ `enabled: false`. Chỉ enable config thử nghiệm; có rollback trả đúng reference behavior và test parity khi tắt.

---

## 15. Cấu hình hybrid khởi đầu — đề xuất, chưa được đo

Map vào schema thực tế; đặt tên hợp lý theo repository, validate và xuất resolved config. Không thêm key rồi để silently ignored. Các số dưới là điểm bắt đầu cho thử nghiệm, không phải thông số production được chứng nhận.

```yaml
motion_roi:
  enabled: false                   # Bat trong config experiment, khong tu bat production.
  shadow_mode: false               # True: tinh ROI de audit, van chay reference detector.

  background:
    method: mog2
    analysis_width: 640
    warmup_min_s: 2.0
    warmup_max_s: 6.0
    learning_time_constant_s: 10.0
    detect_shadows: true
    static_confirm_s: 2.0
    max_mask_age_s: 0.15

  coverage:
    periodic_scan_interval_s: 0.5
    use_validated_reference_profile: true

  roi:
    reuse_existing_tiles_first: true
    max_selected_tiles: 4
    bbox_padding_ratio: 0.15
    protect_track_predictions: true
    prioritize_observation_deadlines: true

  tracking:
    max_observation_gap_s: 1.0
    prediction_grace_s: 1.5
    distinguish_not_searched_from_missed: true

  fallback:
    enabled: true
    use_measured_cost_guard: true
    area_ratio_enter: 0.70
    area_ratio_exit: 0.50
    soft_recovery_confirm_s: 1.0

  diagnostics:
    log_scheduler_decisions: true
    log_coverage_by_region: true
    debug_overlay: false
```

Quan hệ cần validate:

- Periodic coverage interval không vượt observation deadline; observation deadline nhỏ hơn grace với margin phục hồi.
- `max_selected_tiles` phù hợp reference tiling thật; không áp 4 khi camera dùng layout khác.
- Warmup minimum không đồng nghĩa nền đã tốt. Hết warmup maximum mà nền chưa ổn định thì dùng reference mode, không buộc bật gating.
- Padding và area thresholds phải kiểm chứng theo scale/cost; không dùng diện tích thay latency tensor.
- Ngưỡng track-loss/camera-change/noise được suy từ diagnostic và ghi vào config, không hardcode tùy ý.
- Tiết lộ các tham số đang tính theo giây video, pixel source, pixel mask hay ratio.

UI chỉ cần toggle hybrid, interval và lớp debug khi cần. Các tham số kỹ thuật khác để trong config; không làm UI cấu hình rối.

---

## 16. Kiểm chứng hybrid: nhanh hơn phải đi cùng coverage và tuyến đúng

### 16.1. Ba mốc tách biệt, không trộn nguồn lợi ích

```text
A = pipeline hiện tại trước sửa Common Path, giữ để chẩn đoán lỗi cũ.
B = Common Path đã sửa + reference detector/tracker, hybrid OFF.
C = cùng Common Path/model/precision với B + hybrid ON.
```

Dùng A/B để đánh giá sửa thuật toán tuyến; dùng B/C để đánh giá lợi ích/rủi ro gating. Không kết luận hybrid tốt từ việc C tốt hơn A khi Common Path đã đổi cùng lúc.

Thực hiện tiết kiệm:

1. Unit test trọng tâm và shadow mode trên các đoạn đại diện. Shadow mode vẫn chạy reference để biết ROI planner định bỏ gì, không dùng FPS đó làm kết quả hybrid.
2. Sửa vấn đề coverage đặc biệt ở trên/giữa, người nhỏ và người đứng chờ.
3. Chạy B/C thật từ đầu đến cuối cùng video Shibuya, trên cùng Modal T4, điều kiện tương đương. Tạo cache detection/tracking riêng có scheduler fingerprint.
4. Dùng cùng tập review độc lập và cùng timestamp để đối chiếu path, không để mask/candidate của C tự làm nhãn chuẩn.
5. Xác nhận lại nếu kết quả sát ngưỡng hoặc chênh thời gian không ổn định; không mở hàng chục cấu hình GPU không cần thiết.

Khởi tạo background từ đầu mỗi run, không dùng frame tương lai hoặc nền học từ cuối video để xử lý đầu video. Cache không được rò state giữa các run. Ghi warmup/cold start riêng nhưng vẫn tính vào wall time toàn video.

Offline giữ cùng frame schedule; intentional detector skip phải được ghi lại, không biến thành frame dropped. Realtime benchmark riêng với queue/drop policy và PTS đúng.

### 16.2. Test bắt buộc

| Tình huống | Kỳ vọng |
|---|---|
| Người đứng từ đầu, chờ lâu rồi bước đi | Periodic scan vẫn quan sát; motion wake đúng; không mất người chỉ vì bị hấp thụ vào nền |
| Người rất nhỏ/đi chậm ở phía trên hoặc giữa | Không bị morphology, area gate hoặc ROI budget loại bất công; không tái tạo thiên lệch tiền cảnh |
| Người mới vào vùng trước đó tĩnh hoặc đi nhanh qua vùng | Có motion wake/global scan; đo cả lượt bỏ sót, không chỉ latency của lượt bắt được |
| Track bị scheduler bỏ quan sát so với đã scan nhưng mất detection | Hai lý do khác nhau; không xóa track hàng loạt bằng empty-update |
| Hai người giao cắt, che khuất ngắn, đi ngược chiều | Association không bị motion gate làm xấu; prediction không tạo support/route giả |
| Người đi qua biên ROI/tile | Bbox mapping, duplicate merge và Track ID không phụ thuộc tile |
| Cả cảnh đông chuyển động; rất nhiều ROI nhỏ | Cost guard quay reference hợp lý; không thêm hàng chục model inputs |
| Cảnh gần như tĩnh, bóng/xe/cây/màn hình chuyển động | Không vẽ path từ mask; vẫn periodic scan; không global-reset liên tục |
| Ánh sáng đổi, camera rung/scene cut, seek hoặc đổi nguồn | Fallback/reset đúng phạm vi, không nối geometry sai hệ tọa độ |
| Skip nhiều frame, chạy nhanh/chậm, bật/tắt hybrid và đổi Top K | PTS/grace/support nhất quán, không predict hai lần, không trộn config/session |
| Không còn quan sát chuyển động | Path expire đúng video time; không đông cứng đường cũ vô hạn |

Test dữ liệu tổng hợp chỉ kiểm chứng logic có kỳ vọng biết trước; không thay review video thực.

### 16.3. Metrics và artifact bổ sung

Ngoài mục 11, báo:

```text
Performance:
  motion/scheduler/preprocess/inference/tracking/analytics/render p50/p95
  wall time, processed FPS, RAM/VRAM, queue depth, dropped source frames
  global scan count, ROI scan count, intentional detector skip count
  model invocations AND inference images/tensor shapes
  periodic/fallback fraction, ROI union ratio, estimated vs measured cost

Coverage and tracking:
  processed-region coverage theo tren/giua/duoi
  observed / searched-not-found / not-searched track counts
  observation age, deadline violations, grace expirations
  new-person discovery delay (video time) + runtime/stream delay rieng
  permanently missed passages, track gaps/fragmentation
  measured detector recall tren subset co annotation; khong dung raw IDs lam recall

Common Path:
  support/coverage theo tuyen, Top-K direction/route agreement voi review
  first-valid-path time, unsupported gaps, geometry jitter, direction flips
  Path ID/color churn, adaptation time
  region funnel truoc/sau gating de tim starvation
```

Latency phải mô tả phần đo và cách đồng bộ GPU; không dùng thời gian enqueue CPU làm GPU inference time. Không cộng p95 từng stage để gọi là p95 end-to-end.

Thêm artifact có giới hạn dung lượng:

```text
scheduler_decisions.jsonl
motion_roi_metrics.csv
observation_coverage_by_region.csv
hybrid_ab_report.md
sampled_motion_roi_overlay.mp4
matched_timestamp_path_comparison/
resolved_config + manifest + cache provenance
```

Frame debug nên phân biệt motion mask, ROI thật đã scan, ROI bảo vệ track, track predicted-only và reference/fallback reason. Không vẽ mọi lớp dày lên video sản phẩm mặc định.

### 16.4. Gate trước khi đề xuất bật mặc định

Đặt budget/quality tolerance bằng số trong kế hoạch trước lượt benchmark cuối, phù hợp kích thước tập review và baseline; nêu ngưỡng nào người dùng chưa xác nhận. Không tự đổi ngưỡng sau khi thấy kết quả để tạo PASS.

Chỉ đề xuất bật hybrid khi:

- Có lợi ích end-to-end đã đo ở phạm vi sử dụng dự kiến; nêu rõ đoạn tĩnh/đông được và mất gì.
- Không mất luồng chính đã review, không làm hỏng hướng/mũi tên, và không làm vùng xa thiếu dữ liệu hơn vượt tolerance đã công bố.
- Observation deadlines/grace hoạt động; không có track/path ma từ prediction.
- Fallback và rollback được test, overhead đoạn đông nằm trong budget.
- Chất lượng đo trên video đủ dài; phần chưa gán nhãn/chưa xem ghi `REVIEW_PENDING`, không báo PASS.

Nếu gating không tiết kiệm trong cảnh Shibuya vì chuyển động phủ rộng, báo đúng kết quả và giữ B/reference cho cảnh đó. Nếu chỉ tiết kiệm khi vắng, có thể đề xuất chế độ tự thích nghi theo cost đã đo; không quảng cáo nhanh hơn cho mọi video.

Trong báo cáo cuối bổ sung:
**“Hybrid đã bỏ được bao nhiêu lượt/tensor inference? Có bỏ mất người nhỏ hoặc lượt đi qua nào không? Top K và hướng tuyến có còn đúng theo cùng bằng chứng review? Khi nào hệ thống tự tắt gating?”**

---

## 17. Nguồn cho phần hybrid và ranh giới kết luận

Năm nguyên tắc hybrid ở mục 14.1 xuất phát từ yêu cầu người dùng. State machine, config, thời hạn, A/B và điều kiện nghiệm thu là thiết kế đề xuất cần triển khai/kiểm chứng; các nguồn sau không chứng nhận hiệu quả trên repository hiện tại.

- **[R4] OpenCV — How to Use Background Subtraction Methods.**
  `https://docs.opencv.org/4.x/d1/dc5/tutorial_background_subtraction.html`
  Cơ sở foreground/background cho camera cố định và cập nhật model nền từ frame; có ví dụ MOG2/KNN.

- **[R5] OpenCV — BackgroundSubtractorMOG2 class reference.**
  `https://docs.opencv.org/4.x/d7/d7b/classcv_1_1BackgroundSubtractorMOG2.html`
  Tham chiếu history/background ratio, learningRate, shadow values và API. Kiểm tra phiên bản cài đặt trước khi dùng overload bảo vệ foreground.

- **[R6] Ultralytics — Predict, inference arguments and input padding.**
  `https://docs.ultralytics.com/modes/predict/`
  Tham chiếu `imgsz`/`rect`/batch và tensor shape thực. Không suy speedup chỉ từ phần trăm ảnh đã che hoặc crop.

- **[R7] ByteTrack — reference tracker implementation.**
  `https://github.com/FoundationVision/ByteTrack/blob/main/yolox/tracker/byte_tracker.py`
  Tham chiếu prediction, unmatched/lost và tuổi track trong triển khai gốc. Coverage-aware skipping/grace trong prompt là phần tích hợp đề xuất, không phải tuyên bố ByteTrack có sẵn hỗ trợ toàn bộ.

Tham khảo công khai ngày 2026-09-25. Không yêu cầu nâng phiên bản model/thư viện hoặc đổi backend chỉ để triển khai phần này.
