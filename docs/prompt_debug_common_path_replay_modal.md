# Prompt tiếp tục kiểm tra Common Path: replay cache, candidate diagnostics, tracking và UI

## 1. Nhiệm vụ

Bạn là Senior Computer Vision Engineer + Backend Engineer. Hãy tiếp tục kiểm tra và sửa đúng nguyên nhân trong project Crowd Analysis đã triển khai Directional Grid:

**Point Tracklets → Directional Histogram theo Grid → Directed Edge Flow → Candidate Path → Tracklet Validation → Hysteresis → Active Common Path.**

Không triển khai lại engine từ đầu. Mục tiêu của lượt này:

1. Biết chính xác candidate mất ở bước nào và vì sao.
2. Đánh giá chất lượng tracking bằng dữ liệu, không kết luận từ tổng Track IDs.
3. Kiểm tra một tuyến cục bộ có thật bên trong ảnh.
4. Render lại debug/candidate/Active Path từ cache, không chạy lại YOLO.
5. Kiểm chứng UI đang phát đúng bằng trình duyệt.
6. Chạy một job kiểm chứng bằng môi trường GPU Modal, sử dụng cache sẵn có.

Đây là yêu cầu triển khai và kiểm thử thực tế. Sau khảo sát ngắn, hãy sửa code, chạy test và xem output; không dừng ở việc đưa kế hoạch.

## 2. Bối cảnh từ báo cáo trước — phải xác minh lại

Đường dẫn repo được báo cáo: `D:/AITHUCCHIEN/VinFast/project-crownd-analysis`. Dùng workspace thực tế đang mở, không hardcode đường dẫn Windows nếu môi trường khác.

Các thành phần được báo cáo:

- `backend/app/analytics/directional_grid.py`
- `backend/app/analytics/engine.py`
- `configs/default.yaml`
- `modal_common_path.py`
- `scripts/common_path_clip.py`
- `docs/directional_grid_common_path_report.md`
- Run `outputs/common_path/dg-integration-20260919-final/`

Kết quả đã báo cáo, chưa coi là đã được xác minh ở lượt này:

- Clip 60 giây, 1.800 frame; GPU Tesla T4.
- 2.954 Track IDs; processing FPS khoảng 11.614.
- Common Path trả `insufficient_data`, thiếu complete tracklets/support.
- Cache v2 có thể dựng lại ByteTrack từ detections.
- Backend MJPEG có frame thay đổi, nhưng UI là `UI_NOT_VERIFIED`.
- Vòng trước đã dùng 3 GPU jobs.

Đọc README, AGENTS.md nếu có, report, manifest, resolved config và code liên quan. Giữ thay đổi trong worktree, chế độ legacy và khả năng rollback. Không đổi model hoặc kiến trúc streaming khi chưa xác định cần thiết.

## 3. Quy tắc cache và Modal trong lượt này

### 3.1. Ưu tiên cache, cấm YOLO ngầm

- Replay tracklets nếu chỉ sửa analytics.
- Replay detections rồi dựng lại tracker nếu sửa association/lifecycle.
- Render bằng video nguồn + kết quả replay đúng frame/timestamp. Không dùng video đã overlay làm nền rồi vẽ chồng.
- Không khởi tạo/load/warm-up/call YOLO trong runner replay, renderer hoặc UI replay.
- Thêm guard fail-fast và counter `detector_calls`; acceptance là `detector_calls == 0`.
- Ghi `inference_executed: false`, inference timing là N/A trong báo cáo. Không dùng “0 ms” để suy ra inference cực nhanh.
- Giữ cache nguồn bất biến; cache tracker/config mới có key và thư mục riêng.
- Không bổ sung detections giả, nối ID tùy tiện hoặc lặp cùng clip thành nhiều phút để tăng unique support.

### 3.2. Dùng GPU Modal mà không chạy lại YOLO

Yêu cầu “dùng GPU Modal” trong lượt này được thực hiện bằng **một remote replay/verification job trên container có GPU T4**, giữ loại GPU của baseline nếu khả dụng.

Trong job đó: kiểm tra device → đọc detections/tracklets cache → chạy phần cần kiểm chứng → analytics → render/encode → lưu output. Không gọi detector.

- Chẩn đoán/unit tests/replay thử ban đầu chạy CPU local để giảm chi phí; chỉ gửi config đã sửa và kiểm tra lên Modal.
- Một container, một job mới, timeout tối đa 600 giây; clip có cache tối đa 60 giây, chạy offline nhanh không sleep theo FPS.
- Đây là ngân sách riêng cho lượt tiếp theo, không yêu cầu chạy lại 3 job cũ. Không tự retry nhiều lần hoặc mở thêm job.
- Nếu cần job bổ sung/inference mới vì cache thiếu hoặc không tương thích, báo bằng chứng và đề nghị người dùng quyết định; không tự chạy YOLO.
- Kiểm tra CUDA/GPU thực tế, tên device và loại encoder. Không giả định container có NVENC hoặc mọi bước đều chạy GPU.
- Nếu có GPU nhưng analytics/render chạy CPU, ghi đúng `execution_host: modal_gpu_container` và device từng stage; không tuyên bố GPU đã tăng tốc analytics.
- Nếu GPU/credentials/quota không có, báo `MODAL_BLOCKED`; vẫn hoàn thành phần chẩn đoán cache local có thể làm, nhưng không giả nhận đã test Modal.
- Tái sử dụng runner/Volume hiện có; không public deployment mới, không ghi đè production và không upload token/`.env`.
- Đóng writer trước khi commit output Volume; tải đúng thư mục run về local, không tải toàn bộ Volume.

Đối chiếu API theo SDK của project với [Modal GPU](https://modal.com/docs/guide/gpu) và [Modal Volumes](https://modal.com/docs/guide/volumes). Chỉ việc cấp GPU không bảo đảm code CPU tự được tăng tốc.

## 4. Giai đoạn A — Xác minh input và tái hiện baseline

Trước khi thay logic:

1. Đọc manifest, xác định video nguồn, clip start/end, FPS/PTS, resize/letterbox, coordinate space, ROI và model/tracker config tạo cache.
2. Kiểm tra schema/version/hash của cache và sự liên tục frame/timestamp.
3. Xác minh 2.954 Track IDs là distinct track keys trong một camera/epoch, không phải tổng active tracks mỗi frame hoặc cộng lại nhiều lần replay.
4. Replay cấu hình hiện tại từ cache, ghi output baseline riêng.
5. So khớp các chỉ số có thể tái lập với report cũ. Nếu lệch, giải thích trước khi dùng làm baseline.

Tất cả window/confirmation/cooling dùng media event time. Wall clock chỉ dùng đo runtime. Với clip bắt đầu ở thời điểm khác 0, thời gian quan sát là từ đầu đoạn thực sự đã replay, không phải PTS tuyệt đối.

**Không mặc định kết luận “60 giây < 180 giây nên chưa có path”.** Kiểm tra short window 30 giây và warm-up có được hoạt động trên dữ liệu đã quan sát hay đang bị khóa tới 180 giây.

## 5. Giai đoạn B — Candidate diagnostics có thể đối soát

Thêm logging có cấu trúc tại từng bước. Chẩn đoán phải phân biệt:

- Không có edge đủ điều kiện nên chưa sinh candidate.
- Có candidate nhưng bị validation loại.
- Candidate hợp lệ đang chờ hysteresis.
- Candidate đạt nhưng không được chọn vì thua route khác.

### 5.1. Thống kê trước khi sinh candidate

Mỗi lần extract ghi:

- Số cell có evidence, số directed edges trước/sau filter.
- Histogram edge support và ngưỡng áp dụng.
- Số start/end seeds; số cặp có thể kết nối.
- Số nhánh bị prune vì edge support, ROI, direction, loop, max length.
- Search có hết beam/search budget hay không.
- Window đã quan sát bao lâu, warm-up state, earliest/latest evidence.

Nếu `candidate_generated == 0`, phải ghi lý do từ graph/search. Không chỉ in “insufficient_data”. Search bị cắt vì budget không chứng minh không tồn tại route.

### 5.2. Lý do và trạng thái candidate

Ít nhất hỗ trợ các reason code sau; map đúng với code thực tế:

| Reason | Ý nghĩa |
| --- | --- |
| `NO_SEEDS` | Chưa có đầu/cuối hợp lệ để tìm tuyến |
| `NO_CONNECTED_PATH` | Không tìm thấy kết nối với graph/điều kiện hiện tại |
| `INSUFFICIENT_EDGE_SUPPORT` | Một hoặc nhiều edge thiếu unique support |
| `DIRECTION_MISMATCH` | Tracklet đi ngược hướng candidate |
| `ORDER_MISMATCH` | Có cell chung nhưng sai thứ tự di chuyển |
| `INSUFFICIENT_COVERAGE` | Tracklet không bao phủ đủ độ dài tuyến |
| `INSUFFICIENT_UNIQUE_SUPPORT` | Thiếu số tracklet khác nhau hỗ trợ |
| `INSUFFICIENT_COMPLETE_TRACKS` | Thiếu track liên tục đi từ đầu đến cuối tuyến đang xét |
| `STALE_EVIDENCE` | Evidence đã quá hạn |
| `HYSTERESIS_CONFIRMING` | Đang chờ đủ thời gian xác nhận |
| `HYSTERESIS_MARGIN_NOT_MET` | Chưa vượt score Active đủ mức |
| `NOT_TOP_RANKED` | Hợp lệ nhưng có route khác được chọn |
| `SEARCH_BUDGET_REACHED` | Tìm kiếm bị giới hạn, chưa kết luận đầy đủ |
| `ACTIVATED` / `KEPT_ACTIVE` | Được chọn hoặc tiếp tục duy trì |

`HYSTERESIS_CONFIRMING` là pending, không phải rejected. Tracklet-level rejection khác candidate-level rejection; xuất hai bảng riêng.

### 5.3. Event schema và phép đếm

Mỗi evaluation có:

```text
run_id, variant, evaluation_id, event_time_s
candidate_id, route_scope, source_gate, target_gate
cell_sequence, direction, path_length
min_edge_support, support_tracks, complete_tracks
coverage_distribution, order_pass_count
score, active_score, confirmation_elapsed_s
decision: rejected | pending | eligible_not_selected | selected
primary_reason, evaluated_secondary_reasons
checks_not_evaluated, thresholds, evidence_track_keys
```

Yêu cầu:

- `candidate_id` ổn định giữa các lần đánh giá cùng route; `evaluation_id` phân biệt từng lần.
- Nếu short-circuit tại lỗi đầu tiên, ghi các bước sau là not_evaluated, không gán lỗi cho chúng.
- Primary reason là một lý do quyết định, không đếm một evaluation hai lần trong bảng tổng.
- Secondary reasons có thể trùng; ghi rõ tổng secondary counts không nhất thiết bằng số candidate.
- Báo riêng số route unique và số evaluation; cùng route xuất hiện 20 lần không phải 20 tuyến.
- Đối soát mỗi evaluation batch:

```text
generated evaluations
= rejected + pending + eligible_not_selected + selected
```

- Giới hạn evidence samples và kích thước log; counters vẫn chính xác. Diagnostics chạy theo chu kỳ extract, không dump toàn graph trên từng frame.
- Candidate bị filter trước scoring vẫn phải có dấu vết; không chỉ log những candidate còn sống sau tất cả bộ lọc.

Xuất `candidate_evaluations.jsonl`, `candidate_funnel.csv`, `tracklet_rejection_counts.csv`, `graph_diagnostics.jsonl` và `rejection_summary.json`.

## 6. Giai đoạn C — Chất lượng tracking

Từ baseline cache, xuất một hàng mỗi track key vào `tracking_quality.csv`.

Các chỉ số cần có:

- First/last observed time, observed span_seconds, số observation thật.
- Số frame prediction-only, tỷ lệ có observation và khoảng mất observation dài nhất.
- Quãng đường quan sát hợp lệ; không cộng đoạn teleport/gap bị cắt.
- Entry/exit cell, nguyên nhân kết thúc nếu có.
- Track bắt đầu trước clip/kết thúc sau clip hoặc trạng thái không biết do clip bị cắt.

Báo cáo aggregate:

- Số distinct track keys; median/p95 observed span.
- Tỷ lệ track ngắn hơn 1 giây và 2 giây, ghi mẫu số và quy tắc lọc.
- Phân bố active tracks theo frame, số track mới theo khoảng thời gian.
- Median/p95 khoảng mất observation, tỷ lệ track đủ dài để qua corridor đã chọn.
- Thống kê riêng completed tracks và tracks bị cắt bởi clip/ROI; không gọi duration quan sát được là toàn bộ tuổi thật của một người.

Để kiểm tra ID switch:

1. Tìm tối đa 10–20 đoạn nghi ngờ: track kết thúc rồi track khác xuất hiện gần vị trí/hướng dự kiến; giao cắt/occlusion; jump bất thường.
2. Xuất timestamp, track keys, ảnh trước/sau và lý do nghi ngờ.
3. Kiểm tra trực quan mẫu này trên video nguồn và debug overlay.
4. Nếu không có ground truth association, gọi là `suspected_id_switch`; không báo IDF1/HOTA/MOTA hay số ID switch chính xác.

Không tự ghép IDs chỉ vì gần nhau. Một người đi ra và người khác đi vào có thể tạo mẫu tương tự.

Nếu cần sửa ByteTrack, chọn tối đa 2 cấu hình có lý do rồi replay cùng detections cache. So sánh track fragmentation, merge nhầm và route support; không chọn chỉ vì số IDs giảm.

Không hạ ngưỡng detector dựa trên cache đã bị lọc ở ngưỡng cao hơn: detections bị loại trước khi cache không thể khôi phục bằng replay. Ghi hạn chế này nếu ảnh hưởng kết quả.

## 7. Giai đoạn D — Kiểm tra Common Path cục bộ

Chọn một hành lang chuyển động nhìn rõ trong clip hiện có:

1. Xem contact sheet/video nguồn để chọn ROI và hai gate nội bộ S/T theo hình học cảnh, không chọn theo kết quả thuật toán có sẵn.
2. Lưu `corridor_roi.json`, tọa độ, coordinate space, ảnh minh họa và lý do chọn.
3. Cho algorithm tìm tuyến trong corridor; không dùng đường vẽ tay làm polyline Active.
4. Đếm riêng tracklet đi S→T, T→S, partial và bị mất dấu.
5. Kiểm tra có nhiều tracklet thực sự đi liên tục cùng đoạn, đúng thứ tự và chiều hay không.

Phân biệt rõ:

| Scope | Complete nghĩa là |
| --- | --- |
| `global_od` | Track có evidence liên tục từ entrance tới exit của tuyến toàn cảnh |
| `local_corridor` | Track có evidence liên tục từ gate S tới gate T của corridor nội bộ |
| `partial_evidence` | Chỉ quan sát một phần; chưa xác nhận toàn tuyến đang xét |

Một track không cần đi từ biên màn hình tới biên màn hình để ủng hộ tuyến cục bộ. Nhưng complete trong local scope vẫn cần cùng track đi qua cả S và T.

Nếu code đang ép mọi candidate phải có track biên-to-biên toàn ảnh, tách validator theo scope, có feature flag/config và test. Giữ nguyên tiêu chuẩn global_od; đây là thay đổi phạm vi route, không vô hiệu hóa validation.

- Duy trì min unique support, ordered coverage, chống hướng ngược và hysteresis.
- Không cộng các track rời rạc thành complete track.
- Không tự giảm threshold để đảm bảo Active xuất hiện.
- Nếu không có corridor đủ evidence, xuất kết luận đó với số liệu; không dựng path minh họa rồi coi là kết quả.
- Thử cùng config trên một đoạn thời gian khác có trong cache nếu đủ mẫu; nếu chỉ đánh giá trên corridor/clip đã chọn, nêu rõ chưa chứng minh tổng quát.

## 8. Giai đoạn E — Sửa đúng nguyên nhân và render lại từ cache

Ưu tiên sửa theo dữ liệu diagnostics, một nhóm thay đổi mỗi lần:

- Clock/warm-up/window normalization.
- Dedupe unique support, histogram hướng, chuyển cell.
- Search/seed/filter làm mất candidate hợp lệ.
- Order/coverage hoặc complete definition sai phạm vi.
- Candidate identity/confirmation reset liên tục.
- Tracker association/lifecycle nếu evidence xác nhận.

Giữ baseline để so sánh cùng clip, same input hash và cùng event timeline. Không so config mới có context dài hơn với config cũ thiếu context mà không ghi khác biệt.

Xuất hai video:

1. `debug_candidates.mp4`: grid, hướng, candidate ID, scope, support, primary reason/pending time; Track ID chỉ bật ở debug. Giới hạn số candidate hiển thị để đọc được.
2. `active_common_path.mp4`: chỉ point tracking + Active/Cooling Path theo state; không bbox, individual trajectory hoặc candidate.

Render từ video nguồn và snapshot đúng timestamp. Common Path chỉ dùng evidence tới thời điểm frame đó; không dùng path tính từ cuối clip để vẽ ngược lên đầu clip.

Nếu chỉ render đoạn ngắn sau mốc 30 giây, replay phần lịch sử trước đó trước khi xuất frame, không reset engine ngay đầu đoạn preview.

Mở video/contact sheet/debug ảnh, kiểm tra thực tế. Nếu không có Active, debug vẫn phải cho thấy lý do. Ghi quan sát có timestamp vào `inspection.md`.

## 9. Giai đoạn F — Kiểm chứng UI bằng trình duyệt

Dùng browser tool có sẵn hoặc Playwright/browser automation đã cấu hình trong project. Kiểm tra khả năng mở trang thật trước, không coi frontend build thành công là browser test.

Các URL cũ `http://localhost:5173` và `http://127.0.0.1:8000` chỉ là gợi ý; xác minh process/port và endpoint trong môi trường hiện tại. Browser từ xa có localhost riêng, không tự giả định truy cập được backend trên máy người dùng.

Phục vụ kết quả từ cached replay qua đúng endpoint/session/player đang dùng. Label rõ `replay`, không gọi đây là live camera inference. Có thể kiểm tra UI local từ output Modal; điều đó chưa xác minh đường truyền trực tiếp Modal→browser, phải ghi đúng topology.

Kịch bản kiểm tra 10–15 giây, khởi tạo từ đủ lịch sử nếu muốn xem Active:

1. Mở UI, start đúng session/input, quan sát vùng video thực sự đang hiển thị.
2. Lưu ít nhất 3 screenshot vùng video ở các thời điểm khác nhau, có frame ID/media timestamp trong chế độ test.
3. Kiểm tra ảnh người/point trong vùng video thay đổi; không chỉ counter hoặc spinner thay đổi.
4. Kiểm tra frame/point cùng frame packet; Active Path đúng state/revision, video vẫn chạy khi không có Active.
5. Kiểm tra console error, failed requests, stream disconnect và dữ liệu có tiếp tục tới không.
6. Stop/start một lần nếu chức năng có sẵn; xác minh không tạo thêm detector worker và epoch/state reset đúng.
7. Không thêm vòng lặp frame cuối âm thầm; khi EOF báo ended hoặc replay loop có reset rõ ràng.

Theo player thực tế:

- Với video element: kiểm tra playback progress/decoded frames và screenshot.
- Với canvas: kiểm tra draw/version và screenshot vùng canvas.
- Với MJPEG img: dùng nhiều screenshot/capture vùng ảnh và metadata phía stream; `img.complete`, HTTP 200 hoặc request pending không đủ chứng minh đang phát.

Nếu không mở được browser, thử cách hỗ trợ sẵn phù hợp, ví dụ Playwright local. Không giả screenshot từ MP4 là screenshot UI. Nếu vẫn bị chặn, báo `UI_NOT_VERIFIED`, ghi lỗi cụ thể; hoàn thành các phần khác thay vì báo toàn bộ task pass.

Xuất `ui_verification.json` gồm URL thực tế, source mode, topology, browser/version, thời gian xem, evidence, console/network errors, kết luận và giới hạn.

## 10. Output và thứ tự chạy

Dùng thư mục mới `outputs/common_path/diagnostics-<run_id>/` hoặc output root hiện có. Không ghi đè `dg-integration-20260919-final`.

| Artifact | Mục đích |
| --- | --- |
| `manifest.json`, `config_resolved.yaml` | Cache/input hash, versions, variant, scope, clock, budgets |
| `baseline/`, `after/` | Kết quả trước/sau riêng biệt |
| `candidate_evaluations.jsonl` | Quyết định từng evaluation |
| `candidate_funnel.csv`, `rejection_summary.json` | Số lượng và nguyên nhân, phân biệt route/evaluation |
| `graph_diagnostics.jsonl`, `tracklet_rejection_counts.csv` | Pruning và lý do loại tracklet |
| `tracking_quality.csv`, `tracking_summary.json` | Tuổi track, track ngắn, gaps, censored tracks |
| `suspected_id_switches.json` | Đoạn nghi ngờ và evidence trực quan |
| `corridor_roi.json`, `corridor_validation.json` | Local scope và support thật |
| `debug_candidates.mp4`, `active_common_path.mp4` | Video render từ cache |
| `preview_contact_sheet.jpg`, `inspection.md` | Kết quả agent đã xem |
| `ui_screenshots/`, `ui_verification.json` | Bằng chứng browser thực tế |
| `modal_verification.json` | Job ID, GPU, stage devices, detector_calls, runtime, artifacts |
| `diagnostic_report.md` | Root cause, sửa đổi, kết quả và phần chưa kiểm chứng |

File có thể đổi tên cho phù hợp runner hiện tại nhưng manifest phải chỉ rõ đường dẫn. Nếu một bước bị chặn, ghi status/reason; không tạo output placeholder rồi coi là pass.

Thứ tự:

1. Preflight/cache audit + baseline replay.
2. Diagnostics funnel + tracking report + xem mẫu output.
3. Kiểm chứng corridor + sửa lỗi đã có bằng chứng.
4. Replay sau sửa + regression tests liên quan + render preview.
5. Một job Modal GPU kiểm chứng cùng cache/config đã chốt.
6. Tải/mở output, kiểm chứng UI bằng trình duyệt.
7. Báo cáo và lệnh tái hiện.

Không chạy full-video dài, training, sweep mọi config hay soak test trong lượt này. Dừng vòng thử thêm khi đã giải quyết lỗi cụ thể và đủ evidence.

## 11. Acceptance và báo cáo cuối

Acceptance:

- Biết số route/candidate evaluations, stage bị mất và lý do; phép đếm funnel đối soát được.
- Pending hysteresis không bị tính thành reject; không đếm lại route qua các chu kỳ thành tuyến mới.
- Có median/p95 tuổi track quan sát, short-track ratio, clip truncation và các mẫu nghi ngờ ID switch; không ngụy tạo ground truth.
- Local corridor có định nghĩa scope rõ; chỉ xác nhận path khi thật sự có support.
- Sửa lỗi có regression test tương ứng: counting, warm-up, order/coverage, local/global complete, hysteresis identity hoặc cache no-inference.
- Render đúng causal timeline; video được decode và đã xem bằng công cụ hỗ trợ.
- Detector bị khóa trong replay, `detector_calls == 0` cả local, Modal và UI replay.
- Có Modal evidence và browser evidence riêng; nếu thiếu phải đánh dấu riêng chưa verified.
- Không dùng FPS replay làm FPS YOLO/end-to-end live; không suy luận độ ổn định 180 giây/nhiều giờ từ clip 60 giây.

Báo cáo ngắn theo mẫu:

```text
ROOT CAUSE
- Nguyên nhân chính, file/function và evidence

CANDIDATE FUNNEL
- Unique routes / evaluation count
- Edge pruning / order / coverage / complete / pending hysteresis
- Not evaluated / search budget / selected

TRACKING
- Median/p95 observed span, short-track ratios, censored tracks
- Suspected ID switches đã xem và giới hạn kết luận

LOCAL CORRIDOR
- ROI, S/T, scope, support, complete tracks, state

CHANGED
- File/config và lý do sửa; rollback

MODAL + UI
- Job/GPU thực tế; detector_calls; stage devices
- Browser evidence hoặc blocker; replay/live topology

OUTPUT REVIEW
- Đường dẫn video/debug/report và nhận xét có timestamp

LIMITATIONS
- Những điều chưa chứng minh

REPRODUCE
- Lệnh thực tế cho replay, render, Modal và UI test
```

**Bắt đầu bằng đọc cache/manifest và tái hiện baseline. Đừng chạy YOLO để giải quyết một vấn đề chưa được truy vết trong analytics. Trước mỗi lần chỉnh tham số, chỉ ra candidate bị mất ở bước nào bằng số liệu và artifact.**
