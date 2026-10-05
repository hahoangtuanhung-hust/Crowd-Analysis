# Prompt: tối ưu tracking ổn định cho người nhỏ ở xa

Ngày: 02/10/2026 — Dự án Crowd Analysis / Common Path.

> Giao toàn bộ nội dung dưới đây cho agent triển khai trong repository. Nhiệm vụ đợt này là nâng chất lượng detection đầu vào và tracking, đặc biệt người nhỏ ở xa. Hoàn tất code, cấu hình, kiểm thử và đối chứng; không chỉ đưa ra lời khuyên. Tài liệu hiện trạng được cung cấp là mốc tham khảo, chưa phải kết quả audit code hiện tại.

## 1. Mục tiêu và giới hạn công việc

Hãy giúp hệ thống phát hiện người nhỏ/ở xa và theo dõi đúng cùng một người qua thời gian, giảm mất ID, đổi ID, ghép nhầm hai người và rung điểm tracking. Từ những quan sát đúng, tạo đầu ra tracklet đủ liên tục và hướng chuyển động đủ tin cậy để Common Path hiện có sử dụng.

Thứ tự ưu tiên:

1. Đúng người, đúng liên kết danh tính qua frame.
2. Đủ quan sát thực cho người nhỏ/xa, kể cả khi confidence dao động và che khuất ngắn.
3. Tracklet và hướng chuyển động đáng tin cậy, không nối nhầm để làm đường dài hơn.
4. Ghi nhận chi phí/FPS để tránh hồi quy quá lớn, nhưng không triển khai đợt tối ưu realtime riêng.

**Phạm vi chính là tracking.** Chỉ điều chỉnh detector/preprocess/tile merge khi có bằng chứng nó làm mất đầu vào cần thiết cho tracker. ByteTrack là thuật toán association/motion/lifecycle; không mặc định cần “train lại model tracking”.

Không sửa thuật toán gom/vẽ Common Path, ranking/support, Top K, Path ID, màu, smoothing/cadence/threshold của engine Common Path. Không chuyển sang grid, không refactor toàn bộ pipeline, không tối ưu Modal/CPU/batching/TensorRT trong đợt này. Common Path giữ nguyên để kiểm tra tương thích downstream. Có thể thêm overlay chẩn đoán tracking/hướng ở chế độ debug.

## 2. Bối cảnh cần xác minh bằng code

Theo `tong_quan_hien_trang_du_an.md` ngày 28/09/2026, mốc `main@f99ec01`:

- Detector YOLO26n qua Ultralytics/PyTorch; Shibuya dùng full frame + 4 tile, overlap 20%.
- ByteTrack adapter đã có hybrid IoU + bottom-center distance, motion history, grace cho che khuất/đứng yên và Kalman advance theo frame ID.
- Điểm tracking là trung điểm cạnh dưới bbox. Common Path production là `TrackletAggregationEngine`.
- Common Path chỉ nhận điểm `confirmed=true`, `observed=true`. Prediction không tạo support.
- Đã có `MEASURED`, `SEARCHED_NOT_FOUND`, `NOT_SEARCHED_BY_POLICY` và `coast()`.
- Motion ROI đang tắt; Track ID có phạm vi camera/session/stream epoch.
- Chưa có ground truth Shibuya đầy đủ để công bố IDF1/HOTA hay recall.

Đọc `AGENTS.md`, kiểm tra HEAD và working tree, giữ nguyên sửa đổi ngoài nhiệm vụ. Đọc detector, tracker adapter, schema, YAML resolved, pipeline và chỗ tạo `TrackletPoint`; ghi rõ khác biệt với tài liệu. Tránh triển khai lại tính năng đã tồn tại.

## 3. Chẩn đoán trước khi điều chỉnh tham số

Chọn các đoạn có: người rất nhỏ, người nhỏ đi chậm, đi ngược chiều/giao cắt, che khuất rồi xuất hiện, đi qua biên tile, dừng rồi đi và đổi hướng thật. Phóng to crop chỉ để quan sát, đánh giá bằng tọa độ ảnh nguồn.

Với từng lỗi, phân loại và lưu timestamp/GT ID nếu có:

| Loại lỗi | Cần kiểm tra |
|---|---|
| Detector không có box | Ảnh đầu vào, kích thước sau resize, confidence thô, tile coverage, chất lượng nguồn |
| Box có nhưng bị loại | Confidence filtering, min size/area, max_det, ignore region, postprocess/merge |
| Box tới tracker nhưng không match | Stage association, cost/gate, dt, scale, covariance, competition với người bên cạnh |
| Không tạo được track mới | Ngưỡng khởi tạo thực tế, tentative/confirmation, quy tắc high/low-confidence |
| ID đổi khi che khuất/giao cắt | Recovery gate, timeout, uncertainty, direction penalty, sai liên kết |
| Điểm/hướng rung dù ID đúng | Bbox localization, điểm chân bị che, filter và cửa sổ hướng |

Thêm log/debug có giới hạn cho association: track/detection ID, confidence, bbox size, IoU, distance/gate, motion score, time gap, match/reject reason. Chỉ bật cho đoạn/track cần kiểm tra, không log toàn bộ cost matrix mỗi frame trong production.

## 4. Đảm bảo người nhỏ không bị mất trước khi tracking

### 4.1 Audit chuỗi confidence

ByteTrack dùng detection confidence thấp để hỗ trợ nối track hiện có [S1]. Kiểm tra tất cả tầng lọc detector → tile merge → adapter → high/low association. Nếu detector loại box trước, giảm `track_low_thresh` phía tracker sẽ không phục hồi được box đó.

- Lập bảng giá trị thực: detector confidence floor, `track_low_thresh`, `track_high_thresh`, `new_track_thresh`, confirmation và threshold riêng trong adapter. Xác minh điều kiện biên và ý nghĩa theo phiên bản đang cài [S2].
- Đảm bảo detection floor cho phép dải confidence thấp dự kiến tới tracker. Không hạ mọi ngưỡng cùng lúc.
- Dùng box yếu để recovery phải qua motion/geometry gating; không biến mọi box yếu thành ID mới được confirmed.
- Audit ngưỡng khởi tạo hiệu dụng: trong luồng ByteTrack chuẩn, hạ `new_track_thresh` dưới high threshold có thể vẫn không cho box chỉ nằm ở low pool khởi tạo. Nếu người xa luôn ở low pool, phải chứng minh điểm nghẽn đó trước khi đổi chính sách.
- Nếu thử khởi tạo tentative từ bằng chứng yếu lặp lại, đặt sau flag riêng, yêu cầu quan sát thực nhất quán và kiểm soát false birth; không coi đó là ByteTrack mặc định. Track chưa confirmed không tạo evidence cho Common Path.

### 4.2 Giữ thông tin ảnh và hình học

- Đo bbox width/height trên ảnh nguồn và kích thước thực sau resize/letterbox. Không hardcode “mọi người nhỏ đều ở nửa trên”. Phân biệt kích thước ảnh nhỏ với khoảng cách thật nếu chưa calibration.
- Kiểm tra crop, padding, scale và phép đưa tọa độ tile về full frame, đặc biệt biên tile và làm tròn tọa độ.
- Tận dụng tiled inference hiện có. Chỉ A/B tăng độ phân giải hoặc crop/tile vùng khó nếu detector recall thật sự thiếu; báo chi phí. Nghiên cứu slicing là cơ sở để thử, không chứng minh cấu hình mới tốt hơn trên Shibuya [S3].
- Giữ source-aware merge: giảm duplicate từ full frame/tile nhưng không gộp hai người gần nhau; không loại box chỉ vì overlap lớn hoặc kích thước nhỏ. Kiểm tra dao động bbox khi nguồn detection thay đổi giữa các tile.
- Không bật generative super-resolution để tạo chi tiết không có thật. Upscale không được coi là bảo đảm khôi phục người thiếu thông tin.
- Không mặc định đổi model hoặc fine-tune. Nếu audit chỉ ra detector không đủ tốt, báo phần lỗi không thể sửa bằng association; thử phương án đầu vào nhỏ nhất trước, còn huấn luyện/dataset lớn là bước riêng.

## 5. Tối ưu association và vòng đời track

### 5.1 Association thích ứng kích thước và độ bất định

Ưu tiên cải thiện ByteTrack adapter hiện có, không thay tracker ngay từ đầu.

- Với bbox nhỏ, kiểm tra trường hợp lệch vài pixel khiến IoU giảm mạnh. Kết hợp IoU với distance/motion đã có; tránh chỉ tăng `match_thresh` cho toàn cảnh.
- Distance nên có scale theo bbox/perspective phù hợp, một noise floor theo pixel và giới hạn hợp lý theo dt/uncertainty. Bbox quá nhỏ không được làm gate co về 0; gate quá rộng không được nuốt người bên cạnh.
- Nếu dùng Mahalanobis gating, xác minh state/measurement model và covariance; không chỉ thêm tên công thức vào một ngưỡng Euclidean. Tune measurement/process noise trên residual thực, kiểm tra covariance ổn định.
- Confidence thấp thường cần thận trọng hơn với localization. Không tăng trọng số box yếu chỉ để bám ID; nếu tăng uncertainty thì vẫn phải giữ hard gate hình học và giới hạn phục hồi.
- Direction/motion penalty là tín hiệu mềm khi hướng cũ đáng tin. Giảm tác dụng khi mới sinh track, đứng yên, gap dài hoặc người đang rẽ; không khóa người vào hướng ban đầu.
- Association một-một, có bước assignment phù hợp; không cho nhiều track nhận cùng detection hoặc một track nhận nhiều box cùng frame. Ambiguous match phải được ghi nhận, không cưỡng ép match để tăng track length.

### 5.2 Thời gian và che khuất

- Dùng source timestamp/frame gap đúng contract. Nếu Kalman dùng đơn vị frame, quy đổi dt nhất quán; không trộn frame-count với giây và không advance hai lần khi skip/update.
- Phân biệt mất detection thật với frame/vùng không được tìm. `coast()` và prediction phải giữ `observed=false`.
- Tune confirmation, lost buffer và grace theo thời lượng nguồn; kiểm tra FPS thật và inference interval. Không tăng buffer thật dài để giữ ghost track.
- Trong che khuất ngắn, giữ identity với uncertainty tăng có kiểm soát. Khi tái xuất hiện, chỉ nối lại nếu geometry, thời gian và motion tương thích; cạnh tranh nhiều người phải xử lý bảo thủ.
- Không giữ velocity cũ vô hạn cho người đứng yên. Giảm drift nhưng vẫn cho phép người bắt đầu đi lại.
- Khi không thể phân biệt sau che khuất dài, cho phép kết thúc/khởi tạo track thay vì gán chắc cùng ID. Không hứa không bao giờ ID switch trong cảnh không đủ thông tin.

### 5.3 Tracker khác chỉ là đối chứng có điều kiện

Nếu đã có bằng chứng ByteTrack adapter còn yếu ở crossing/nonlinear motion, có thể benchmark một tracker đối chứng sau flag, dùng cùng detection đầu vào. OC-SORT là một hướng motion/observation-based để tham khảo [S4]. Không tích hợp nhiều tracker cùng lúc hoặc đổi production theo tên thuật toán.

ReID không bật mặc định: crop rất nhỏ có thể thiếu đặc trưng để phân biệt. Chỉ thử local appearance nếu có bằng chứng crop đủ thông tin và metrics cải thiện. Không thêm face recognition, gallery danh tính thật hoặc ReID xuyên camera. Không mặc định phiên bản thư viện đang cài hỗ trợ tracker được nêu trong tài liệu web hiện hành.

## 6. Hướng chuyển động và tracklet đầu ra

Phần này nằm ở tracker output/diagnostic; giữ nguyên thuật toán Common Path và tiêu chí gom tuyến hiện tại.

### 6.1 Tách quan sát, ước lượng và prediction

- Giữ bbox/điểm quan sát gốc để audit. `observed=true` chỉ khi frame hiện tại có detection thực được association; prediction, nội suy hoặc optical flow nếu thử không được giả làm detection thực.
- Bottom-center bbox là điểm đại diện, chưa chắc chân thật khi bị che/cắt ảnh. Ghi chất lượng hoặc reason ở diagnostic; không tự dựng vị trí chân chắc chắn từ box bị cụt.
- Nếu smoothing online, dùng filter nhân quả, phụ thuộc dt và giữ raw point. Đo độ trễ khi bắt đầu đi/rẽ; không dùng future frame hoặc smooth xuyên gap/segment để làm video đẹp hơn.
- Giữ hành vi export hiện có theo mặc định. Nếu thay điểm cung cấp cho downstream bằng filtered measurement, phải có ablation và provenance; không ghi đè raw point hoặc thay `observed` semantics.

### 6.2 Ước lượng hướng từ nhiều quan sát

- Ước lượng vector vận tốc bằng fit/local displacement trên cửa sổ source time của các điểm observed đáng tin, thay vì chỉ lấy chênh lệch hai frame sát nhau. Cửa sổ là tham số để tune, không cố định chung cho mọi tốc độ/kích thước.
- Chỉ dùng lịch sử cùng track/segment, timestamp tăng và không chứa gap/teleport không hợp lệ. Không cho prediction tự xác nhận hướng của chính nó.
- Có trạng thái `unknown`, `stationary`, `moving` hoặc tương đương và độ tin cậy/chất lượng hướng. Không có đủ evidence thì báo unknown; không ép mọi track có mũi tên.
- Ngưỡng bắt đầu/chấm dứt chuyển động có hysteresis và noise floor theo kích thước/độ rung, nhưng không quá cao đến mức người xa đi chậm luôn bị coi là đứng yên.
- Khi có rẽ/quay đầu thật, cập nhật hướng sau đủ quan sát, đo delay. Không loại tracklet chỉ vì nó không đi thẳng; không dùng các Common Path hiện có làm prior bắt tracker đi theo đường đã vẽ.
- Nêu quy ước image coordinates: x sang phải, y xuống dưới. Hướng trong ảnh không phải hướng địa lý/ground-plane khi chưa homography phù hợp.

Các trường debug đề xuất: `direction_vector`, `direction_state`, `direction_quality`, `direction_observed_span_s`. Không bắt buộc sửa public API nếu sidecar/debug đủ dùng; nếu có schema mới phải tương thích và có test.

### 6.3 Ranh giới tracklet trung thực

Giữ camera/epoch, temporary Track ID, segment ID, frame ID, timestamp, bottom-center, confirmed/observed đúng contract. Không nối qua epoch hoặc gap dài bằng điểm giả. Nếu continuity không đáng tin, dùng cơ chế segment hiện có để bắt đầu đoạn mới và ghi lý do; không tự tạo luật tách chỉ vì thiếu hướng hoặc người đổi hướng thật.

Đo chất lượng segment trước khi đề xuất thay đổi luật. Không kéo dài tracklet bằng prediction, không làm mất các đoạn rẽ để chỉ giữ đường đẹp. Không thay engine Common Path để che đầu vào tracking kém.

## 7. Benchmark có thể chứng minh cải thiện

### 7.1 Dữ liệu và cache

- Chọn tập tuning và holdout theo các đoạn thời gian tách biệt, tránh frame gần nhau rơi vào cả hai. Bao phủ nhỏ/xa, gần, che khuất, tile seam, giao cắt, đổi hướng và đứng yên.
- Mốc khởi đầu đề xuất: ít nhất 3 đoạn liên tục 10–20 giây cho MOT, có annotation đầy đủ bbox/ID và quy tắc visibility/ignore; bổ sung frame detection nếu cần. Quy mô này chỉ đủ kiểm tra ban đầu, không đại diện mọi camera.
- Người/đoạn không thể nhận diện đáng tin được đánh dấu theo quy tắc annotation công khai, không tùy ý xóa các case khó. Nếu chỉ gán nhãn ROI, xử lý vùng ngoài/biên bằng evaluator có ignore policy đúng, không tính detection ngoài vùng chưa gán nhãn thành false positive.
- Nhóm “small/far” xác định từ GT/ROI hoặc calibration trước benchmark, không dựa vào kích thước box dự đoán của từng candidate. Báo phân bố bbox, số GT và visibility. Không cắt track tùy tiện khi đổi nhóm kích thước để làm ID metrics sai.
- `tracking_cache.jsonl` là đầu ra tracker, không đủ để thử lại association. Muốn tune tracker trên cùng detection cần detection cache trước tracker có bbox/confidence/source/frame/coverage; nếu chưa có thì bổ sung tối thiểu.
- Detection cache phải lưu đến confidence floor thấp nhất cần thử. Muốn hạ dưới floor đã lưu hoặc đổi model/tile/preprocess/merge thì chạy lại detector. Tracking cache dùng để kiểm tra downstream/replay, không thay thế detector cache.
- Giữ fingerprint source/model/config/schema và không ghi đè artifact baseline.

### 7.2 Metric và cách tránh đánh giá sai

Dùng evaluator chuẩn như TrackEval cho IDF1, HOTA/AssA và CLEAR metrics khi có nhãn phù hợp [S5]. Các metric hướng/tracklet dưới đây là metric dự án, phải ghi định nghĩa rõ.

| Nhóm | Metric cần báo cáo |
|---|---|
| Detection | Precision/recall tại IoU 0,5; riêng small/far/occlusion; FP/frame; duplicate box; bbox localization |
| Identity | IDF1, HOTA/AssA, ID switches, fragmentation; báo raw count và denominator chuẩn hóa cố định |
| Continuity | Tỷ lệ thời gian GT visible có observed match đúng ID; đoạn đúng ID liên tục dài nhất; recovery sau occlusion; ghost duration |
| Tracklet | Purity theo GT identity, teleport/false join, độ dài observed thực; kiểm tra không tăng purity bằng cách chia mọi track thành đoạn cực ngắn |
| Hướng | Angular error so với hướng GT trong cửa sổ moving; tỷ lệ có hướng hợp lệ; false movement khi GT stationary; delay khi bắt đầu đi/rẽ |
| Vận hành | Detector/tracker FPS và p95 latency, dropped frames, peak memory; cùng phần cứng/config output |

Không suy chất lượng từ track dài hơn, ít `unique_track_ids` hơn, nhiều bbox hơn hoặc video mượt hơn. Ít switch do tracker không theo dõi người khó là thất bại; luôn báo recall/coverage cùng identity metrics. Hướng chỉ chính xác trên một vài track dễ nhưng thường xuyên unknown cũng phải hiện rõ qua coverage.

Angular error chỉ tính khi GT displacement đủ vượt noise; cửa sổ và tiêu chí chọn mẫu cố định giữa các variant. Không đo hướng chỉ so với track do chính thuật toán sinh ra rồi gọi đó là accuracy. Jitter có thể đánh giá residual so với annotation trên đoạn phù hợp, không phạt chuyển động thật.

### 7.3 Ablation tối thiểu

| Variant | Detector | Tracker | Mục đích |
|---|---|---|---|
| B0 | Baseline | Baseline | Mốc đối chiếu |
| T1 | Cùng detection cache | Confidence/lifecycle đã tune | Cô lập cấu hình tracker |
| T2 | Cùng detection cache | Scale/uncertainty/motion fix có chọn lọc | Cô lập association |
| D1 | Điều chỉnh detector nhỏ nhất nếu cần | Tracker baseline | Chứng minh lỗi từ detector |
| C1 | Detector được chọn | Tracker được chọn | Hiệu quả kết hợp |
| H1 | Cùng C1 | Thêm direction/filter output | Đo hướng và filter lag riêng |

Chỉ chạy nhánh có cơ sở, không grid search toàn bộ tổ hợp. Giữ precision, model và cadence cố định trừ variant có chủ đích thay chúng. Sau tune, dùng holdout chưa dùng để chọn tham số; ghi mọi thay đổi config thực tế.

### 7.4 Quyết định nghiệm thu

Trước khi chạy candidate, chốt metric chính theo lỗi baseline: ưu tiên small/far IDF1/association và ID switch, với detection precision/recall/coverage làm điều kiện bảo vệ. Ngưỡng dung sai phải dựa vào quy mô nhãn và nhiễu annotation, ghi trước kết quả; không tự đặt “ổn định 99%” khi chưa có dữ liệu.

Candidate chỉ được khuyến nghị khi có cải thiện identity/continuity trên small/far, không đánh đổi bằng ghép sai người, mất coverage hoặc hồi quy rõ trên người gần. Hướng phải cải thiện cùng coverage hoặc delay chấp nhận được. Báo kết quả theo từng clip và số mẫu; vài case đẹp không đủ kết luận.

Nếu thiếu nhãn, hoàn tất implementation, unit/regression tests, diagnostic video và báo `QUALITY_REVIEW_PENDING`; không công bố IDF1/HOTA hoặc “tracking ổn định” bằng cảm tính. Không dừng toàn bộ công việc chỉ vì chưa có ground truth, nhưng không promote âm thầm.

## 8. Kiểm thử và deliverable

Bổ sung test có ý nghĩa cho: box nhỏ lệch vài pixel; hai người gần nhau; confidence dao động high/low; khởi tạo/tentative; mất 1–vài frame; che khuất và recovery; dt/skip; đứng yên rồi đi; rẽ/quay đầu; tile seam/duplicate; observed vs prediction; direction unknown; epoch/reset và segment boundary. Có cả case synthetic với truth đã biết và case video thật.

Chạy suite liên quan, `compileall`, `git diff --check`; build frontend nếu public contract/UI bị đổi. Mốc 147 test pass trong tài liệu chỉ là lịch sử. Có smoke tích hợp Common Path nhưng giữ nguyên code/config thuật toán Common Path để đối chiếu input mới.

Bàn giao:

1. Code/patch và config candidate riêng, tên đề xuất `configs/shibuya-tracking-stable.yaml`; mọi key mới có schema và code tiêu thụ, rollback rõ.
2. Báo cáo `docs/small_person_tracking_report.md`: lỗi gốc, lựa chọn tham số, ablation, kết quả holdout và giới hạn.
3. Video trước/sau cùng thời gian, crop vùng xa, bbox/ID, observed/predicted, trail ngắn và hướng khi đủ bằng chứng; bảng lỗi có timestamp.
4. Metrics JSON/CSV; detector/tracking cache cần thiết; config resolved, source/model hash, commit và dependency versions.
5. Lệnh benchmark/replay thực sự được kiểm tra với CLI hiện có, không bịa tên cờ.
6. Kết luận `IMPROVED_AND_VERIFIED`, `QUALITY_REVIEW_PENDING` hoặc `NOT_IMPROVED`, kèm bằng chứng; xác nhận Common Path engine/config không bị thay để che lỗi.

Hãy bắt đầu bằng đọc tracker adapter và audit detector confidence floor, kiểm tra người nhỏ bị mất ở bước nào, rồi triển khai thay đổi có tác động rõ nhất. Tự giải quyết lựa chọn kỹ thuật trong phạm vi này; báo blocker cụ thể nếu thiếu dữ liệu/quyền chạy. Không mặc định mở rộng sang tối ưu tốc độ, huấn luyện lớn hoặc thay toàn kiến trúc.

## Tài liệu đối chiếu

Các nguồn sau hỗ trợ nguyên lý, không phải bằng chứng rằng một thay đổi chắc chắn cải thiện Shibuya. Luôn kiểm tra API và code của phiên bản đang cài. Thiết kế gate/diagnostic/đánh giá đặc thù ở trên là đề xuất cho dự án.

- [S1 — ByteTrack, bài báo gốc](https://arxiv.org/abs/2110.06864): association sử dụng cả detection score thấp.
- [S2 — Ultralytics tracking](https://docs.ultralytics.com/modes/track): tham số tracker và cơ chế high/low confidence.
- [S3 — Slicing Aided Hyper Inference](https://arxiv.org/abs/2202.06934): sliced inference cho đối tượng nhỏ.
- [S4 — OC-SORT, mã nguồn tác giả](https://github.com/noahcao/OC_SORT): hướng observation-centric để đối chứng khi cần.
- [S5 — TrackEval](https://github.com/JonathonLuiten/TrackEval): implementation tham chiếu cho các metric MOT.

Nguồn dự án: `tong_quan_hien_trang_du_an.md`, cập nhật 28/09/2026. Khi có khác biệt, dùng code/config/provenance đúng run làm căn cứ và ghi rõ thay đổi so với mốc tài liệu.
