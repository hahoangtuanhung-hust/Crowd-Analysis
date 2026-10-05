# Prompt: rà soát và xóa file không còn cần trong Crowd Analysis

Ngày: 02/10/2026.

## Nhiệm vụ

Bạn đang làm việc trực tiếp trong repository Crowd Analysis. Hãy kiểm tra các file, xác định file thực sự dư thừa và xóa những file đã có đủ bằng chứng không còn cần. Hoàn tất việc dọn dẹp và kiểm chứng, không chỉ đưa ra danh sách đề xuất.

**Nguyên tắc quyết định: chưa chắc không còn dùng thì giữ lại.** Không đặt chỉ tiêu phải xóa bao nhiêu file hoặc tiết kiệm bao nhiêu dung lượng. Một đợt audit kết luận không có file nào đủ an toàn để xóa vẫn là kết quả hợp lệ.

Người dùng đã cho phép xóa file được xác minh là không cần. Không hỏi lại từng file đủ điều kiện; tiếp tục xử lý nhóm chắc chắn, còn nhóm chưa rõ phải giữ nguyên và báo lý do. Không coi việc có bản sao hoặc Git history là bằng chứng file không còn dùng: đó chỉ là khả năng khôi phục.

## 1. Phạm vi và bối cảnh

- Phạm vi là working tree của repository hiện tại, không bao gồm dữ liệu trên Modal Volume, cloud storage, repository khác hoặc thư mục ngoài dự án.
- Mục tiêu là dọn file thừa, không thay đổi tính năng, thuật toán detect/tracking/Common Path, public API, schema cache hoặc hành vi runtime.
- Đọc `AGENTS.md` và tài liệu hiện trạng nếu có. Tài liệu `tong_quan_hien_trang_du_an.md` đối chiếu `main@f99ec01` chỉ là mốc cũ; phải kiểm tra code/config hiện tại.
- Dự án có backend Python/FastAPI, frontend React/TypeScript, config YAML có `extends`, script local/replay/benchmark, Modal batch/live, Docker và adapter Grand Central. Phải xét toàn bộ các đường chạy được hỗ trợ, không chỉ cấu hình production mặc định.
- Các engine `legacy`, `directional_grid`, `shadow`, `DominantLiveFlowEngine`, DD-CRP và Motion ROI có thể phục vụ rollback, compatibility hoặc nghiên cứu. “Không chạy mặc định” không đồng nghĩa “có thể xóa”.

## 2. Bảo vệ trạng thái trước khi dọn

1. Xác định repository root, branch, HEAD và trạng thái Git gồm staged, unstaged, untracked, ignored. Dùng danh sách file từ Git kết hợp `rg --files`; không coi kết quả `rg` mặc định là toàn bộ dự án vì có thể bỏ qua file ignored/hidden.
2. Ghi nhận thay đổi có sẵn của người dùng. Không reset, stash, checkout đè hoặc stage/commit toàn bộ working tree. File đang được sửa hoặc mới tạo mà chưa rõ mục đích phải giữ lại.
3. Không xóa file đang được job/worker tạo hoặc sử dụng. Nếu có agent/job khác đang sửa cùng đường dẫn, loại file đó khỏi đợt xóa; không tự dừng job để dọn.
4. Với file tracked định xóa, xác minh nội dung hiện tại trùng phiên bản Git có thể khôi phục và ghi commit/blob nguồn. Nếu có thay đổi chưa lưu thuộc người dùng, không xóa.
5. Với file untracked/ignored, mặc định giữ nếu không chứng minh được nguồn gốc và mục đích. Chỉ xóa file sinh tự động có thể tái tạo rẻ, cách tái tạo đã xác minh, không chứa dữ liệu duy nhất và không còn được dùng.
6. Nếu cần bản sao cho file không có trong Git, tạo bản sao nguyên vẹn ở vị trí riêng ngoài vùng sẽ xóa, giữ đường dẫn tương đối và checksum, rồi xác minh khôi phục được. Không sao chép secret ra nơi chia sẻ hoặc đưa nội dung secret vào report.

Chỉ backup những thứ thật sự cần; không nhân bản toàn bộ video/model/dataset để dọn vài file. Không xóa nếu chưa có phương án khôi phục phù hợp. Không làm việc này bằng lệnh xóa diện rộng hoặc lọc theo đuôi tên.

## 3. Lập danh mục và tìm nơi sử dụng

Đầu tiên phân loại: source, entrypoint, config, test/fixture, script vận hành, documentation, frontend asset, model/data, benchmark artifact, file sinh tự động và file chưa rõ.

Với mỗi ứng viên xóa, đọc nội dung/metadata phù hợp và kiểm tra:

- Import trực tiếp, re-export, package discovery, `__init__`, string module name, `importlib`, registry, decorator/plugin và import động.
- FastAPI router/lifespan, frontend route/lazy import, TypeScript path alias, Vite/public asset, CSS `url()`, template và đường dẫn được ghép lúc chạy.
- YAML `extends`, engine/backend selector, environment override, default/fallback path, config được chọn từ CLI/API hoặc tài liệu.
- Script chạy tay, `__main__`, console entrypoint, `subprocess`, glob, tên file ghép từ prefix/suffix và process khởi động độc lập.
- Dockerfile `COPY`, compose volume/command, Nginx, CI/CD, package/build metadata, dependency/lockfile, resource package, migration và lệnh trong README.
- Test, fixture, sample, notebook, báo cáo benchmark, quy trình rollback, script tải artifact và workflow Modal local/batch/live/replay.
- Lịch sử commit liên quan: file được thay thế khi nào, bởi file nào, còn lý do compatibility hay rollback không. Tuổi file chỉ là thông tin phụ.

Tìm cả full path, basename, module/symbol, key cấu hình và những biến thể đường dẫn có ý nghĩa; đọc ngữ cảnh thay vì chỉ đếm match. Nếu dùng công cụ phát hiện unused/dead code, kết quả chỉ tạo danh sách nghi vấn để kiểm tra tiếp.

**Không tìm thấy import hoặc không chạy trong một lần test không đủ kết luận file thừa.** Các script được gọi ngoài repository, entrypoint công khai hoặc asset được truy cập trực tiếp có thể không có reference nội bộ. Khi không xác định được caller ngoài dự án, giữ lại.

## 4. Điều kiện để được xóa

Gán từng ứng viên vào đúng một trạng thái:

| Trạng thái | Điều kiện | Hành động |
|---|---|---|
| `SAFE_TO_DELETE` | Có bằng chứng cụ thể đã hết vai trò, đã rà dependency/entrypoint liên quan, không thuộc nhóm cần bảo vệ, có cách khôi phục và kiểm chứng | Xóa theo danh sách đường dẫn cụ thể |
| `KEEP` | Còn được dùng hoặc có giá trị runtime, kiểm thử, vận hành, dữ liệu, audit, rollback | Giữ nguyên |
| `NEEDS_REVIEW` | Chưa đủ bằng chứng về công dụng, reference động/ngoài repo hoặc không thể kiểm chứng rủi ro liên quan | Giữ nguyên và ghi phần chưa rõ |

Ví dụ có thể đủ điều kiện sau khi kiểm tra: file tạm do một script audit đã kết thúc tạo ra; bản sao trung gian đã xác minh không còn caller và không chứa nội dung riêng; generated output nhẹ không được dùng làm input/fixture/deployment artifact; module nội bộ đã thay thế hoàn toàn, không còn contract hay workflow nào phụ thuộc.

Không tự động xóa file mang tên `old`, `backup`, `temp`, `test`, `draft`, `legacy`, file lâu không sửa hoặc file có hash giống file khác. File rỗng cũng có thể là marker/package file; hai file giống nội dung vẫn có thể phục vụ hai đường dẫn khác nhau.

## 5. Những nhóm cần giữ trong dự án này nếu chưa có bằng chứng riêng

- `tracking_cache.jsonl`, detection cache và metadata: có thể cần cho replay, kiểm chứng tracker/Common Path và tránh chạy GPU lại. “Tạo lại được” không có nghĩa là dư thừa hoặc rẻ để tạo lại.
- `outputs/`, MP4, contact sheet, ground truth, dataset, model `.pt/.onnx/.engine`, annotation và tài liệu review chưa hoàn tất. Không quét rồi xóa hàng loạt vì chúng bị `.gitignore`.
- `summary.json`, `manifest.json`, `provenance.json`, `config_resolved.yaml`, metrics, inspection và artifact benchmark: có thể là bằng chứng duy nhất của một run; xem chúng như một bộ dữ liệu có liên kết.
- Profile A/B, single-pass/FP16, Motion ROI/shadow, script benchmark/Grand Central/replay, engine rollback/compatibility: chỉ tắt hoặc không default là chưa đủ để xóa.
- Test/fixture, tài liệu kiến trúc, báo cáo cũ, prompt triển khai, script dùng thủ công: không có import code không đồng nghĩa không cần.
- File cấu hình môi trường/credential, config cá nhân, dependency lockfile, giấy phép, asset bản quyền và metadata triển khai.
- Thư mục dependency/môi trường như virtualenv hoặc `node_modules`: không dọn chỉ vì có thể cài lại; đợt này không phải reset môi trường.

Cache thuần sinh tự động chỉ được xóa khi không phục vụ job đang chạy và đã xác minh tái tạo được. Không đổi `.gitignore` để che file thừa hoặc coi file bị ignore là đã được xóa.

## 6. Thực hiện theo lô nhỏ và có manifest

Trước khi xóa, ghi bảng quyết định, ưu tiên dùng báo cáo hiện có; nếu chưa có, tạo `docs/repository_cleanup_report.md`. Với mỗi đường dẫn `SAFE_TO_DELETE`, ghi:

- Loại file, kích thước và trạng thái Git.
- Vai trò cũ, lý do hết vai trò, file thay thế nếu có.
- Nơi đã kiểm tra sử dụng, command/search hoặc call graph liên quan và kết quả.
- Commit/blob hoặc backup/checksum để khôi phục; với generated cache nhẹ thì ghi cách tái tạo.
- Kiểm thử/smoke sẽ kiểm chứng việc xóa và kết quả sau khi chạy.

Không ghi chung chung “unused” hay “không ai dùng” mà không có bằng chứng. Không đưa nội dung credential vào manifest.

Sau đó:

1. Chụp lại/hash ứng viên và kiểm tra trạng thái ngay trước khi xóa; nếu file đã thay đổi từ lúc audit thì giữ lại để tránh xóa thay đổi mới.
2. Xóa đúng từng path trong allowlist đã xác minh; không dùng `git clean -fdx`, `rm -rf` theo wildcard, recursive delete một thư mục chưa kiểm tra hết, hoặc `git reset --hard`.
3. Kiểm tra resolved path không vượt repository root; không đi theo symlink/junction để xóa target ngoài phạm vi. Không đụng `.git`, submodule hoặc worktree metadata.
4. Dọn theo nhóm chức năng nhỏ, kiểm chứng sau mỗi nhóm. Với file đang được tham chiếu bởi runtime/feature còn hỗ trợ, chuyển sang `KEEP`; không xóa caller/feature chỉ để biến file thành unused.
5. Nếu cần cập nhật link lịch sử tới một bản thay thế tương đương, sửa tối thiểu, giữ ngữ nghĩa/tài liệu nguồn. Nếu cần refactor lớn để xóa một file, giữ file và báo công việc riêng.
6. Không tự push, force-push, deploy hoặc xóa cloud artifact. Bàn giao diff sạch theo phạm vi, giữ nguyên staging và thay đổi của người dùng.

## 7. Kiểm chứng trước và sau

Đo baseline phù hợp trước khi xóa source/config/assets. Không dùng build cũ/cache cũ để kết luận source đã xóa không ảnh hưởng; dùng build sạch trong vị trí tạm khi cần, không xóa tùy tiện dữ liệu dự án.

Các kiểm tra nền tảng của dự án, xác minh lệnh theo repository thực tế:

```powershell
python -m pytest -q
python -m compileall -q backend scripts modal_common_path.py
git diff --check
```

Nếu phần frontend/build assets bị ảnh hưởng, chạy production build trong `frontend`:

```powershell
npm run build
```

Thêm smoke đúng đường bị ảnh hưởng: backend import/start và health, config `extends`/engine resolution, CLI help hoặc dry-run không phát sinh job, frontend asset loading, replay từ cache sẵn có nếu phù hợp. Mốc 147 test pass là lịch sử, không giả định số test hiện tại giống vậy.

Không chạy deploy hay job GPU tốn phí chỉ để chứng minh cleanup. Nếu không kiểm tra được đường GPU/RTSP do thiếu môi trường mà việc xóa có thể ảnh hưởng đường đó, giữ lại ứng viên đó; vẫn tiếp tục dọn những file khác có bằng chứng đầy đủ.

Nếu test/smoke fail do lô xóa, khôi phục đúng file/lô vừa xóa, kiểm chứng hồi phục rồi ghi nhận. Không sửa test, bỏ test, đổi config hoặc tắt tính năng để che lỗi cleanup. Phân biệt lỗi đã có ở baseline với lỗi mới và báo chính xác; baseline fail không tự động hợp thức hóa việc xóa.

## 8. Bàn giao cuối cùng

Trả lời bằng tiếng Việt, gồm:

1. **Đã xóa:** từng file hoặc nhóm tương đương có danh sách path đầy đủ, lý do/bằng chứng và dung lượng thực tế.
2. **Giữ lại:** các file từng nghi thừa nhưng còn phục vụ runtime, benchmark, dữ liệu hoặc rollback.
3. **Chưa rõ:** đường dẫn và bằng chứng còn thiếu; xác nhận chưa xóa.
4. **Kiểm chứng:** lệnh đã chạy, kết quả, phần chưa chạy và lý do.
5. **Khôi phục:** commit/blob hoặc backup/cách tái tạo cho các file đã xóa, áp dụng theo từng đường dẫn và không đè thay đổi mới của người dùng.
6. **Phạm vi thay đổi:** xác nhận không sửa thuật toán/tính năng, không xóa artifact cloud, không ảnh hưởng thay đổi ngoài nhiệm vụ; nếu có ngoại lệ phải nêu rõ.

Phân biệt dung lượng file đã bỏ khỏi working tree với dung lượng thực giải phóng. File tracked vẫn có thể nằm trong Git history, backup cũng chiếm dung lượng; không tuyên bố đã giảm tương ứng kích thước repository/ổ đĩa nếu chưa đo. Không rewrite Git history để thu nhỏ repo trong nhiệm vụ này.

Hãy bắt đầu audit và thực hiện ngay nhóm `SAFE_TO_DELETE`. Hoàn tất phần chắc chắn trước khi báo lại; với nhóm chưa rõ, giữ nguyên. Ưu tiên không mất chức năng và dữ liệu hơn việc làm cây thư mục trông gọn.
