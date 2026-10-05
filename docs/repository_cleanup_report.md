# Báo cáo rà soát và dọn dẹp repository

Thời điểm rà soát: 2026-10-02  
Repository: `D:\AITHUCCHIEN\VinFast\project-crownd-analysis`  
Nhánh/HEAD: `main` / `4269fc8f469653e8a9b740245fe33be80b677467`

## Phạm vi và nguyên tắc

- Đã quét tệp tracked, untracked và ignored; không có `AGENTS.md` trong cây làm việc.
- Mọi thay đổi đang có trước khi dọn dẹp, bao gồm các cấu hình tracking, mã nguồn, kiểm thử, tài liệu prompt/báo cáo và script replay chưa track, đều được **giữ nguyên**.
- Không có tệp tracked nào được chọn để xóa (`git ls-files --error-unmatch` không trả về mục tiêu nào); vì vậy không có Git blob nào cần làm bằng chứng khôi phục.
- Chỉ xóa cache sinh tự động sau khi xác minh đường dẫn tuyệt đối nằm trong repository, không phải symlink/junction, và không có dữ liệu nguồn/độc nhất.

## Quyết định trước khi xóa

Mọi đường dẫn dưới đây đều bị Git ignore. Mười bốn thư mục `__pycache__` đã được kiểm tra toàn bộ: chỉ chứa tệp `.pyc`. `.pytest_cache` chỉ chứa metadata/cache của pytest; `.ruff_cache` chỉ chứa metadata/cache của Ruff (payload được ignore bởi file `.gitignore` do cache tạo ra). Không có tiến trình dự án Python/Vite đang chạy tại thời điểm kiểm tra.

| Đường dẫn | Phân loại và bằng chứng | Dung lượng | Quyết định | Tạo lại/khôi phục |
|---|---|---:|---|---|
| `__pycache__` | 6 `.pyc`; ignore theo `.gitignore:1` | 73,812 B | SAFE_TO_DELETE | import Python hoặc `python -m compileall -q ...` |
| `backend/__pycache__` | 2 `.pyc`; ignore theo `.gitignore:1` | 1,004 B | SAFE_TO_DELETE | import/compile Python |
| `backend/app/__pycache__` | 3 `.pyc`; ignore theo `.gitignore:1` | 6,700 B | SAFE_TO_DELETE | import/compile Python |
| `backend/app/analytics/__pycache__` | 13 `.pyc`; ignore theo `.gitignore:1` | 421,955 B | SAFE_TO_DELETE | import/compile Python |
| `backend/app/api/__pycache__` | 2 `.pyc`; ignore theo `.gitignore:1` | 38,173 B | SAFE_TO_DELETE | import/compile Python |
| `backend/app/core/__pycache__` | 4 `.pyc`; ignore theo `.gitignore:1` | 88,828 B | SAFE_TO_DELETE | import/compile Python |
| `backend/app/datasets/__pycache__` | 6 `.pyc`; ignore theo `.gitignore:1` | 77,342 B | SAFE_TO_DELETE | import/compile Python |
| `backend/app/inference/__pycache__` | 3 `.pyc`; ignore theo `.gitignore:1` | 36,303 B | SAFE_TO_DELETE | import/compile Python |
| `backend/app/metrics/__pycache__` | 2 `.pyc`; ignore theo `.gitignore:1` | 9,986 B | SAFE_TO_DELETE | import/compile Python |
| `backend/app/schemas/__pycache__` | 3 `.pyc`; ignore theo `.gitignore:1` | 13,161 B | SAFE_TO_DELETE | import/compile Python |
| `backend/app/tracking/__pycache__` | 3 `.pyc`; ignore theo `.gitignore:1` | 47,711 B | SAFE_TO_DELETE | import/compile Python |
| `backend/app/video/__pycache__` | 6 `.pyc`; ignore theo `.gitignore:1` | 94,270 B | SAFE_TO_DELETE | import/compile Python |
| `scripts/__pycache__` | 23 `.pyc`; ignore theo `.gitignore:1` | 439,773 B | SAFE_TO_DELETE | import/compile Python |
| `tests/__pycache__` | 39 `.pyc`; ignore theo `.gitignore:1` | 725,496 B | SAFE_TO_DELETE | import/compile Python |
| `.pytest_cache` | cache kết quả pytest; ignore theo `.gitignore:3` | 3,617,502 B | SAFE_TO_DELETE | `python -m pytest -q` |
| `.ruff_cache` | cache phân tích Ruff, không chứa mã nguồn | 6,734 B | SAFE_TO_DELETE | `ruff check .` |
| `frontend/tsconfig.app.tsbuildinfo` | incremental TypeScript info; ignore theo `.gitignore:11`; SHA-256 `3410249E5B963FC8DD7C6C9C6E95DA9BD88E786B39E5680E11183F5F739BC0B2` | 593 B | SAFE_TO_DELETE | `cd frontend; npm run build` |
| `frontend/tsconfig.node.tsbuildinfo` | incremental TypeScript info; ignore theo `.gitignore:11`; SHA-256 `9541E424A62AC62357880AD5D2EB511CABC03714A9CA9B58C5620CF7B6319FF5` | 12,479 B | SAFE_TO_DELETE | `cd frontend; npm run build` |

Tổng danh sách trên: **5,711,822 B** (khoảng **5.45 MiB**), gồm 16 thư mục cache và 2 tệp build-info.

## Các mục được giữ lại

- Toàn bộ thay đổi Git hiện hữu và tệp untracked của người dùng: đặc biệt các cấu hình `shibuya-tracking-*`, mã tracking/pipeline, test, `scripts/replay_detection_cache.py`, cùng các prompt và báo cáo trong `docs/`.
- Dữ liệu, mô hình, video, artefact và kết quả: `data/`, `models/`, `outputs/`, `tmp/`, cache/dataset có khả năng chứa dữ liệu độc nhất; cũng như `.venv/`, `frontend/node_modules/`, `frontend/dist/`.
- Cấu hình frontend ignored `frontend/vite.config.js` và `frontend/vite.config.d.ts`: Vite cục bộ ưu tiên `vite.config.js` trước `vite.config.ts`, và hai cấu hình có nội dung khác nhau. Xóa có thể đổi hành vi dev/build, nên giữ lại và cần chủ sở hữu quyết định nếu muốn hợp nhất.
- Lockfile, tài liệu, fixtures, profiles/engines và script vận hành: không có bằng chứng là sinh tự động hoặc có thể thay thế rẻ.

## Kiểm chứng và khôi phục

- Trước khi xóa, `npm run build` trong `frontend/` đã thành công (`tsc -b && vite build`, 2,452 modules transformed). Đây cũng chứng minh hai tệp `.tsbuildinfo` có thể được tái tạo.
- Sau khi xóa, chạy `PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider` để kiểm tra mà không làm phát sinh lại Python/pytest cache; chạy thêm `git diff --check` và xác nhận 18 đường dẫn không còn tồn tại.
- Không cần backup vì tất cả mục xóa là cache/build metadata không track; các lệnh ở cột cuối tạo lại chúng. Không có dữ liệu hay mã nguồn bị xóa.

## Thực thi

Danh sách SAFE_TO_DELETE ở trên là allowlist duy nhất đã được dùng cho thao tác xóa.

- Đã xóa thành công **18/18** mục, đúng tổng **5,711,822 B**; không có tệp Git-tracked nào bị xóa.
- Hậu kiểm đã xác nhận cả 18 đường dẫn đều không còn tồn tại.
- `PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider` đã thành công sau dọn dẹp (chạy không ghi bytecode hay pytest cache). Có các cảnh báo deprecation sẵn có của `modal`, `starlette` và `fastapi`, nhưng không có lỗi kiểm thử.
- `git diff --check` đã thành công. `git status --short` chỉ còn các thay đổi/tệp untracked đã tồn tại trước đó và báo cáo này; không có cache nào trong danh sách xóa xuất hiện lại.
