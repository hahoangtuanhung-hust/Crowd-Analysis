# Prompt triển khai: detection theo chu kỳ, tracking đúng thời gian và tracklet trung thực

Ngày: 02/10/2026 — Dự án Crowd Analysis / Common Path.

## 1. Nhiệm vụ

Hãy triển khai một profile lấy mẫu detection để giảm chi phí YOLO tiled: thử lần lượt **mỗi N=1, 3, 5 frame nguồn chạy detector một lần**. Chỉ thử N=10 khi N=5 đã có bằng chứng chất lượng chấp nhận được. Mục tiêu là giảm số ảnh inference nhưng vẫn giữ Track ID, hướng và tracklet đủ tin cậy cho Common Path.

Người dùng chấp nhận video nguồn 30 FPS nhưng detector chạy thưa hơn, nếu chất lượng phân tích luồng di chuyển đạt yêu cầu. Vì vậy đợt này đánh giá **sampled detection cho live/analytics**, không bắt buộc detector xử lý đủ 30 frame/giây. Báo đúng source FPS, detection rate, tracker prediction/update rate và preview FPS.

Thực hiện code, config, test, benchmark và báo cáo; không chỉ đề xuất. Tự giải quyết lựa chọn kỹ thuật thông thường. Nếu thiếu GPU/video/annotation, hoàn tất phần code/test có thể làm và báo phần chưa xác minh; không bịa số liệu hoặc tự đánh dấu PASS.

## 2. Phạm vi và baseline

- Giảm tần suất **detector**, không chỉ giảm tracker update trong khi YOLO vẫn chạy mọi frame.
- Frame được chọn: detection thực → association/tracker update → điểm observed khi match có thật.
- Frame ở giữa: có thể prediction/coast để giữ state/preview, không chạy YOLO và không tạo evidence giả.
- Giữ nguyên model, backend, precision, imgsz, số tile/overlap, confidence threshold và merge ở phép so sánh đầu tiên. Với Shibuya hiện trạng, một lần detection gồm full frame + bốn tile.
- Không đồng thời triển khai TensorRT, batch mới, model mới, ReID, Motion ROI, dynamic tile hoặc refactor Common Path. Nếu backend đã đổi trước đó, cố định backend hiện tại cho mọi N và ghi rõ.
- Giữ thuật toán/renderer Common Path, support theo temporary Track ID, Path ID/màu/Top K, lifecycle và route memory. Không bù thiếu evidence bằng đường nội suy.
- Tạo nhánh/profile riêng từ baseline hiện tại, tên gợi ý `perf/sampled-detection`. Giữ N=1 làm mặc định và rollback; không tự merge/deploy profile chưa được kiểm chứng.

Đọc `AGENTS.md`, kiểm tra HEAD/working tree và giữ nguyên thay đổi của người dùng. Theo tài liệu 28/09/2026 tại `main@f99ec01`, dự án đã có `video.inference_interval`, capture/analytics bounded queues, ByteTrack adapter có `coast()` và advance theo frame gap. Hãy xác minh và tái sử dụng, không tạo scheduler/tracker song song trùng chức năng.

Đọc đường chạy local/web và `gpu_clip`/Modal batch thực tế; không giả định chúng đều dùng chung loop. Thống nhất semantics trong abstraction phù hợp và kiểm chứng từng entrypoint cần hỗ trợ.

## 3. Quy ước chu kỳ phải rõ ràng

Trong nhiệm vụ này **N là khoảng cách giữa hai lần detection**, không phải số frame bỏ qua. Ví dụ một epoch đánh số từ 0, N=5: detect ở frame 0, 5, 10, 15…; các frame giữa chỉ coast/predict nếu được xử lý.

Không sao chép trực tiếp semantics của thư viện khác: ví dụ `interval` của DeepStream là số batch bị bỏ qua [S1]. Kiểm tra `video.inference_interval` hiện tại có nghĩa gì, rồi ghi tài liệu/validation tương ứng; tránh N=5 vô tình thành chu kỳ 6 hoặc bị áp hai lần.

Với nguồn CFR 30 FPS, không có drop ngoài ý muốn và mỗi scan là 5 ảnh:

| N | Detection scans/giây nguồn | Khoảng quan sát | Ảnh detector/giây nguồn |
|---:|---:|---:|---:|
| 1 | 30 | 33,3 ms | 150 |
| 3 | 10 | 100 ms | 50 |
| 5 | 6 | 166,7 ms | 30 |
| 10 | 3 | 333,3 ms | 15 |

Đây là workload lý thuyết, không phải throughput đã đo. Không suy toàn pipeline tăng tốc đúng N lần.

## 4. Scheduler dựa trên nguồn, không dựa trên tốc độ máy

### 4.1 Lập lịch

1. Detect frame đầu tiên của epoch/session. Lưu source frame ID/PTS của lần detector thực sự được gọi, kể cả lần gọi trả về không có người.
2. Chọn frame tiếp khi đã đến hạn N source frames từ lần scan trước. Dùng điều kiện đến hạn, không chỉ `frame_id % N == 0`, vì live queue có thể đã bỏ đúng frame đó.
3. Nếu frame đến hạn bị rơi khỏi queue, scan frame khả dụng tiếp theo; ghi số frame trễ/lỡ hạn và cập nhật lịch từ scan thực tế. Không chạy bù nhiều scan trên cùng một ảnh hoặc tái tạo frame đã mất.
4. Không dùng số vòng loop hoặc wall time xử lý làm source cadence. Offline nhanh/chậm vẫn chọn cùng lịch nếu không drop. PTS phục vụ dt/aging; N được định nghĩa theo frame nguồn.
5. Với VFR hoặc RTSP không có PTS tin cậy, ghi chất lượng/fallback timestamp và actual inter-scan time; không coi N=5 luôn tương đương 167 ms. Nếu cần time-based scheduler, coi đó là mode riêng, không trộn ngầm trong thử nghiệm này.
6. Reset scheduler và các timestamp khi epoch đổi/restart/seek theo lifecycle hiện có. Chỉ cho chọn N lúc start session trong phiên bản đầu nếu chưa có contract hot-update; không thêm UI phức tạp không cần thiết.
7. Motion ROI vẫn tắt ở A/B này; không kết hợp hai bộ skip không có accounting. Không bật đồng thời `vid_stride` ở loader và interval ở pipeline khiến thực tế thành N².

### 4.2 Queue và frame bị bỏ

Giữ bounded queue. Offline benchmark dùng backpressure, không drop ngoài policy. Live giữ `drop_oldest` để kiểm soát latency, nhưng đếm riêng drop do quá tải và frame chủ động không detect.

Frame capture bị drop không phải frame đã chạy `coast()`. Khi frame kế tiếp tới, tiến state theo gap thật. Không tăng queue vô hạn để “giữ đủ frame”, không replay frame cũ làm detector rate trông đạt yêu cầu.

## 5. Tracker: prediction khác measurement update

### 5.1 Trên frame detection

- Chỉ gọi YOLO trên frame đã được chọn; giữ toàn bộ profile tiled baseline.
- Merge detection rồi association đúng một lần, theo thứ tự source time.
- Chỉ track match detection thật mới `observed=true`; confirmation là trạng thái riêng, không tự confirmed chỉ vì đã predict đủ lâu.
- Track không match trong vùng detector đã tìm nhận miss thật theo contract. Detector lỗi không được ngụy trang thành kết quả “không thấy người”.

### 5.2 Trên frame không detection

- Dùng prediction/coast phù hợp adapter hiện có, với `observed=false`, coverage `NOT_SEARCHED_BY_POLICY` và reason như `cadence_skip` trong metric/metadata nếu schema hỗ trợ.
- Không gọi `update([])` để mô phỏng skip. Detection đã chạy nhưng không có người và detector không chạy là hai sự kiện khác nhau.
- Không sao chép bbox/detection score cũ rồi đánh dấu như quan sát mới; nếu hiển thị score cũ phải có observation age.
- Không cập nhật direction evidence, hit streak, confirmation hoặc support bằng điểm prediction. Track mới không thể được phát hiện ở frame đã bỏ; báo discovery delay là một tradeoff của sampling.

### 5.3 Không advance Kalman hai lần

Audit adapter và tracker bên dưới có tự predict trong `update()` hay không. Chọn **một owner duy nhất** cho tiến thời gian. Ví dụ đã coast qua frame 1–4 thì frame 5 chỉ tiến phần thời gian chưa được tính; không lại advance toàn bộ gap 0→5 rồi predict thêm trong update.

Tách ít nhất ba khái niệm trong state hoặc metadata tương đương:

- `last_state_time`: state đã predict đến đâu;
- `last_observed_time`: lần cuối track match detection thật;
- `last_searched_time`: lần cuối vùng track thực sự được detector quét.

Tên field là đề xuất; không tạo trùng nếu đã có. Quy đổi dt nhất quán với đơn vị velocity/covariance của tracker. Không chuyển frame count sang giây ở một nơi mà quên motion model/gating còn lại.

Có thể tiến state chỉ khi cần frame output hoặc measurement nếu tối ưu hơn; kết quả vẫn phải đúng source time và được kiểm thử. Không bắt buộc chạy association 30 lần/giây khi chỉ có 6 lần measurement.

### 5.4 Confirmation, lost buffer và giới hạn dự đoán

- Frame không tìm không tăng detector-miss count giả, nhưng tuổi từ quan sát cuối vẫn tăng. Track không được sống vô hạn vì luôn được coast.
- Giữ giới hạn prediction-only age/TTL thực sự theo source time. Nếu gap vượt khả năng tin cậy, không tiếp tục hiển thị bbox như vị trí chắc chắn hoặc nối tracklet bằng prediction.
- Audit threshold đang tính theo source frames, processed updates hay số detection hits. Với N=5, “30 update” khác 30 frame nguồn; báo rõ conversion và test để không vô tình tăng timeout 5 lần.
- Confirmation vẫn cần số quan sát thực có bằng chứng. Sampling có thể làm confirmation chậm hơn; đo delay, không tự giảm xuống một box yếu để bù.
- Không tăng grace/gate vô hạn để giữ ID; kiểm tra false recovery khi hai người đi gần/giao cắt.

## 6. Tracklet, hướng và Common Path

Common Path hiện tại chỉ nhận `confirmed=true`, `observed=true`; duy trì quy tắc này.

1. Frame skip có thể cung cấp snapshot prediction cho preview, nhưng không thêm điểm vào evidence tracklet. Giữ raw observation và prediction phân biệt trong cache/schema.
2. Logic source-time, lifecycle, cooldown/TTL vẫn phải tiến khi không có observation. Không freeze Path ID/state forever vì bỏ qua analytics trên frame skip; dùng clock/snapshot update hiện có với chi phí phù hợp, không buộc recompute Common Path nặng mỗi frame.
3. Hướng tính từ các observed points cùng segment, có đủ source-time span/displacement vượt noise. Không tạo hướng bằng prediction rồi dùng chính hướng đó chứng minh tracking đúng. Đứng yên/thiếu bằng chứng phải có unknown hoặc stationary theo contract.
4. Audit `max_observation_gap`, giới hạn bước nhảy, min points/min duration, confirmation và các threshold tương đương thật sự tồn tại. N=5/N=10 có thể làm tracklet bị tách/loại ngay cả khi association đúng.
5. Phép A/B đầu tiên giữ tracker/Common Path thresholds như baseline và ghi rejection reason. Nếu cần điều chỉnh đơn vị thời gian hoặc một config liên quan sampling, tạo variant phụ riêng, giải thích lý do và đo hồi quy; không âm thầm nới rule hoặc thay algorithm gom/vẽ tuyến.
6. Khoảng quan sát 167–333 ms là ít mẫu hơn; không resample/nội suy thành “30 observed points/giây”. Nếu phát hiện hướng chậm hoặc bỏ lỡ đoạn rẽ, đó là kết quả cần báo cáo, không che bằng smoothing renderer.

## 7. Cấu hình và khả năng rollback

Ưu tiên dùng `video.inference_interval` nếu phù hợp. Bổ sung schema validation: số nguyên N≥1, giá trị không hợp lệ fail sớm. Không cần thêm tham số trùng nghĩa.

Tạo profile kế thừa baseline, tên gợi ý:

- `shibuya-sampled-n3.yaml`;
- `shibuya-sampled-n5.yaml`;
- `shibuya-sampled-n10.yaml` là experimental, chỉ benchmark sau gate N=5;
- baseline N=1 giữ nguyên làm reference và default.

Ghi N, scheduler version/phase rule, source timing, lifecycle conversions và skip reason trong config resolved/provenance. Thay N phải làm tracking cache fingerprint thay đổi; không reuse output N=1 như thể đã chạy tracker N=5.

Không thêm threshold/flag “để dành” mà code không sử dụng. Nếu UI có FPS metric, chỉ sửa tối thiểu để phân biệt tần suất; không cần thiết kế lại dashboard.

## 8. Metrics phải có

Phân biệt hai mẫu số: **giây video nguồn** và **giây wall time xử lý**. Offline chạy chậm vẫn có lịch 6 scan/giây nguồn nhưng chỉ thực thi được ít scan/giây thực.

| Metric | Ý nghĩa |
|---|---|
| `input_fps` | FPS nguồn hoặc tốc độ nhận nguồn, phải nêu loại |
| `source_processing_fps` | Số source frames pipeline thực sự đi qua / wall time xử lý |
| `detection_scans_per_source_second` | Số frame được detector quét / thời lượng nguồn |
| `detection_scans_per_wall_second` | Số scan thực thi / wall time |
| `detector_images` | Tổng full-frame/tile images, không nhầm với số model calls/batches |
| `association_updates`, `prediction_steps` | Đếm riêng measurement update và prediction |
| `policy_skipped_frames` | Frame chủ động không detection |
| `capture_drops`, `analytics_drops` | Frame mất do queue/quá tải, không gộp với policy skip |
| `inter_scan_gap`, `observation_age` | Khoảng thời gian scan thực và tuổi observation per-track |
| `preview_fps`, `frame_age_p95`, `e2e_p95` | Trải nghiệm live, định nghĩa mốc đo rõ |

Kèm stage p50/p95, queue age/size, CPU/RAM/GPU/VRAM, scan deadline misses, tracklet accepted/rejected reason. Cache recording không được lặp lại cùng observed point trên nhiều frame prediction.

Trong offline không drop, detect frame đầu và mỗi N frame, đối chiếu scan count với số frame/range/phase đã định nghĩa. Đối chiếu số ảnh detector với scan count và workload tiled thực tế. Nếu scheduler có ngoại lệ, liệt kê và đếm rõ.

## 9. Benchmark và quality gate

### 9.1 Trình tự

1. Audit và kiểm tra N=1 sau refactor: không thay behavior baseline ngoài bug fix đã được chứng minh và báo riêng.
2. Smoke 5–10 giây cho scheduling/time/coverage; N=1,3,5.
3. A/B cùng clip khoảng 65 giây và các đoạn khó: nhỏ/xa, crossing, occlusion, rẽ, đứng yên→đi và người xuất hiện ngắn giữa hai scan.
4. Chọn N=3 hoặc N=5 dựa trên chất lượng và throughput. Chỉ thử N=10 nếu N=5 đạt quality gate; nếu N=5 fail thì giữ N thấp hơn, không mặc định chạy N lớn hơn để có FPS đẹp.
5. Candidate tốt nhất chạy full clip gần 10 phút và live paced 30 FPS; báo cả kết quả offline và live, không coi chúng tương đương. Lặp baseline/candidate đủ để phân biệt hiệu quả với nhiễu, ưu tiên ít nhất hai lần trên clip A/B.

Giữ GPU, CPU allocation, backend, model, precision, imgsz, tile, output mode và dependency versions giống nhau. Các benchmark runtime dùng inference thật và output I/O thực tế; không dùng tracking/detection cache để công bố FPS mới.

### 9.2 Dữ liệu và đánh giá đúng sampling

Nếu đã có detection cache ở N=1, có thể lấy đúng các frame được chọn **trước tracker** để thử association nhanh, không dùng detection frame bị skip để cập nhật state. Đây chỉ là quality replay và phải mang nhãn đó; benchmark tốc độ vẫn chạy detector thật.

Đánh giá trên cùng các đoạn GT/holdout và cùng quy tắc visibility/ignore. Không chỉ chấm trên những frame candidate có detect rồi kết luận toàn bộ tracking tương đương:

- **Scan-time quality:** precision/recall người nhỏ và association ở các mốc scan. Dùng baseline tại cùng mốc để đối chiếu; báo đây là metric có điều kiện.
- **Full-timeline tracking:** nếu sản phẩm xuất prediction giữa scan thì chấm IDF1/HOTA trên cùng timeline, với prediction thực sự được xuất trong TTL hợp lệ. Nếu không xuất track ở frame skip, không nội suy offline để bù khi chấm.
- **Evidence quality:** scan density, observation gap/age, observed recall tại scan, thời gian đoạn GT có liên kết đúng ID, tracklet purity/fragmentation và missed short passages. Không đòi số observed samples bằng N=1 vì mục tiêu sampling là giảm số đó.
- **Downstream:** hướng đúng/sai/unknown, delay khi bắt đầu đi/rẽ, tracklet rejection và Common Path direction/continuity/stability với engine giữ nguyên. Không coi prediction đầy đủ là evidence đầy đủ.

Chưa có GT thì hoàn tất code/test, video review timestamp và report `QUALITY_PENDING`. Không suy IDF1/HOTA từ số ID hoặc vẻ mượt. GT không được tạo bằng chính baseline rồi gọi là annotation thật.

### 9.3 Tiêu chí chọn N

Chốt ngưỡng trước A/B trong report; các giá trị sau chỉ là đề xuất khởi đầu, không phải bảo đảm chất lượng:

- Full-timeline IDF1/HOTA giảm không quá 2 điểm trên thang 0–100; small/far scan-time recall giảm không quá 2 điểm so với reference tại cùng mốc, đồng thời kiểm tra full-timeline missed passages.
- Không có hồi quy false join nghiêm trọng ở các case crossing/occlusion đã gán nhãn; báo raw ID switch/fragmentation và denominator, không chỉ tỷ lệ phần trăm khi số case rất ít.
- Hướng và tracklet giữ đủ chất lượng cho các tuyến được review; báo delay/unknown coverage và các đoạn người xuất hiện ngắn bị bỏ lỡ. Gate không được chỉ dựa vào detector recall tại scan.
- Live đạt cadence thực tế gần mục tiêu theo nguồn, E2E/observation age không tăng vô hạn, queue bounded và không có drop quá tải ngoài ngưỡng đã công bố. Mục tiêu thử ban đầu có thể dùng output E2E p95≤250 ms; observation age và discovery delay phải báo riêng vì sampling tự tạo khoảng chờ.
- Workload/tổng thời gian xử lý giảm thực đo, không OOM/leak/deadlock. GPU utilization cao hoặc preview 30 FPS không đủ để PASS.

Nếu quality chưa đủ, không promote N=5/N=10. Nếu chỉ đạt sampled live, ghi chính xác **“nguồn 30 FPS, detector X scan/giây nguồn, preview Y FPS”**, không công bố detector full-frame 30 FPS.

## 10. Test trọng tâm

1. N=1,3,5,10 chọn đúng frame, first frame/EOF/short clip; config invalid bị từ chối.
2. Frame đến hạn bị drop thì scan frame khả dụng tiếp, không starvation do modulo hoặc chạy bù nhiều lần cùng ảnh.
3. Không nhân đôi sampling ở loader/pipeline; source ID/PTS không bị đánh lại thành chỉ số processed frame làm sai dt.
4. Coast rồi update không double-predict; kiểm thử motion/covariance phù hợp dt và gap thực, gồm dropped frames.
5. Skip không phải miss; detector gọi nhưng không có detection là miss thật. Prediction không tăng hit/observed/support hoặc gia hạn `last_observed_time`.
6. Ghost track vẫn hết TTL theo source time, confirmation dùng quan sát thật, state/route lifecycle không bị freeze khi không có evidence.
7. Epoch/restart/seek, batch/frame order nếu code đã batch; không trộn ID qua epoch hoặc tạo điểm từ tương lai.
8. Cache/replay giữ đúng N, coverage và timestamp; không duplicate observed point; rollback N=1 tương thích.
9. Video thực có tiny person, crossing, rẽ và dừng: metric/overlay phân biệt measured/predicted, arrow unknown khi thiếu evidence.

Chạy test hiện có liên quan, `compileall` cho module đã sửa và `git diff --check`; frontend build nếu API/UI thay đổi. Mốc 147 test pass là lịch sử, không coi đó là kết quả hiện tại. Không xóa/đổi test để che regression.

## 11. Deliverable

- Code tích hợp scheduler/coast/time semantics trong pipeline/adapter hiện có; config N=3/5 và N=10 experimental, baseline N=1 không bị thay mặc định.
- Report `docs/sampled_detection_tracking_report.md` hoặc mở rộng báo cáo phù hợp: baseline commit/model/config, thay đổi, thời gian thực đo, quality gate, các hạn chế và N khuyến nghị.
- Video trước/sau cùng clip, crop vùng xa, bbox/ID, observed/predicted, trail/hướng; timestamp các case fail.
- Metrics JSON/CSV, config resolved, source/model hash, run ID, version và cache provenance; không commit artifact lớn/secret hoặc ghi đè run cũ.
- Lệnh chạy benchmark/live/replay thật sự đã kiểm tra theo CLI hiện tại và cách rollback N=1. Giữ workflow Modal GPU commit Volume rồi kết thúc, download riêng.
- Kết luận một trong: `SAMPLED_PROFILE_VERIFIED`, `QUALITY_PENDING`, `QUALITY_REGRESSION`, `PERFORMANCE_NOT_MET`; nói rõ N nào đã thử, N=10 có được thử hay bị giữ lại bởi gate.

Mẫu bảng kết quả:

| N | Scan/s nguồn / wall | Ảnh detector | Source processing FPS | E2E p95 / observation age p95 | IDF1/HOTA | Small recall tại scan | ID switch/Frag | Missed passages | Tracklet/hướng | Quyết định |
|---:|---|---:|---:|---|---|---|---|---|---|---|
| 1 | Đo | Đo | Đo | Đo | Đo hoặc pending | Đo | Đo | Đo | Review | Reference |
| 3 | Đo | Đo | Đo | Đo | Đo hoặc pending | Đo | Đo | Đo | Review | Điền |
| 5 | Đo | Đo | Đo | Đo | Đo hoặc pending | Đo | Đo | Đo | Review | Điền |
| 10 | Chỉ thử khi gate cho phép | — | — | — | — | — | — | — | — | Experimental |

Không tự mở rộng sang job/GPU ngoài quyền tài nguyên đã có. Nếu bị chặn, hoàn tất patch/test/config/lệnh cụ thể trước khi báo lại. Không cần xin xác nhận lại việc thử N=3 và N=5 trong phạm vi đã được yêu cầu.

Hãy bắt đầu bằng audit `video.inference_interval`, `coast()` và phần `update()` tự predict của ByteTrack; triển khai đúng clock/evidence trước, rồi mới đo tốc độ. Giữ thử nghiệm sampling độc lập với nhánh TensorRT; sau khi chọn N mới có thể so backend với cùng N trong nhiệm vụ tiếp theo.

## Nguồn và giới hạn

- Hiện trạng: `tong_quan_hien_trang_du_an.md`, ngày 28/09/2026; xác minh code/config đúng commit hiện tại.
- [S1 — NVIDIA DeepStream reference application](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_ref_app_deepstream.html): tham khảo mô hình detector chạy thưa và tracker giữa các scan; không yêu cầu chuyển dự án sang DeepStream.
- [Ultralytics ByteTrack API](https://docs.ultralytics.com/reference/trackers/byte_tracker/): đối chiếu đúng phiên bản và adapter, không giả định upstream xử lý cadence như dự án.
- [APPTracker — nghiên cứu MOT ở frame rate thấp](https://infzhou.github.io/folder/Zhou_APPTracker_Improving_Tracking_Multiple_Objects_in_Low-Frame-Rate_Videos_MM_2022.pdf): cơ sở để kiểm tra association khi quan sát thưa; không chứng minh N=5/N=10 tốt cho Shibuya.

Các lựa chọn scheduler, field metrics và ngưỡng nghiệm thu ở trên là thiết kế đề xuất cho dự án, không phải kết quả đã đo.
