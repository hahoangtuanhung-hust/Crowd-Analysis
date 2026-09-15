from pathlib import Path
import modal

# 1. Khởi tạo Modal App
app = modal.App("crowd-analysis")

# 2. Định nghĩa Persistent Volume để lưu trữ video và kết quả phân tích
volume = modal.Volume.from_name("crowd-analysis-data", create_if_missing=True)

# 3. Định nghĩa Docker Image trên Cloud
# Cài đặt các thư viện hệ thống cần cho OpenCV, Video I/O và dependencies của dự án
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1-mesa-glx", "libglib2.0-0", "ffmpeg")
    .pip_install(
        "fastapi>=0.115,<1",
        "numpy>=2.0,<3",
        "onnxruntime>=1.20,<2",
        "opencv-python-headless>=4.10,<6",
        "pyarrow>=18,<24",
        "psutil>=6,<8",
        "pydantic>=2.10,<3",
        "python-multipart>=0.0.18,<1",
        "pyyaml>=6,<7",
        "ultralytics>=8.4.116,<9",
        "uvicorn[standard]>=0.34,<1",
    )
    # Tải trước trọng số model YOLO vào image cache để tăng tốc khởi động
    .run_commands(
        "python -c 'from ultralytics import YOLO; YOLO(\"yolo26n.pt\")'"
    )
    # Mount file trọng số model yolo26n.pt nếu có sẵn
    .add_local_file("yolo26n.pt", remote_path="/root/yolo26n.pt")
    # Mount toàn bộ mã nguồn backend, scripts, configs vào container
    .add_local_dir("backend", remote_path="/root/backend")
    .add_local_dir("configs", remote_path="/root/configs")
    .add_local_dir("scripts", remote_path="/root/scripts")
)

# ---------------------------------------------------------------------------
# KỊCH BẢN 1: Chạy Batch Tracking trên GPU (ví dụ T4, A10G, hoặc L4)
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    gpu="T4",               # Có thể chọn: "T4", "A10G", "L4", hoặc bỏ qua nếu chạy CPU
    volumes={"/root/data": volume},
    timeout=1800,           # 30 phút timeout cho video dài
)
def run_tracking_remote(
    video_bytes: bytes,
    filename: str = "input.mp4",
    pipeline_type: str = "tracking",  # "tracking" hoặc "points"
) -> dict:
    import shutil
    import subprocess
    from pathlib import Path

    input_path = Path("/tmp") / filename
    input_path.write_bytes(video_bytes)

    # Thư mục tạm thời chứa riêng output của lần chạy này
    job_out_dir = Path("/tmp/job_outputs")
    if job_out_dir.exists():
        shutil.rmtree(job_out_dir)
    job_out_dir.mkdir(parents=True, exist_ok=True)

    persistent_out_dir = Path("/root/data/outputs")
    persistent_out_dir.mkdir(parents=True, exist_ok=True)

    stem = input_path.stem
    print(f"[Modal] Bắt đầu xử lý video ({pipeline_type}): {input_path}")

    if pipeline_type == "points":
        # Dùng cho phân tích điểm, luồng di chuyển zone (vd: Grand Central data.mp4)
        cmd = [
            "python", "-m", "scripts.process_point_tracks",
            str(input_path),
            "--output-dir", str(job_out_dir),
            "--zones", "/root/configs/zones.json",
        ]
    else:
        # Mặc định: Full tracking + Heatmap + Flow field + Overlay Video
        cmd = [
            "python", "-m", "scripts.run_tracking",
            str(input_path),
            "--output", str(job_out_dir / f"{stem}_overlay.mp4"),
            "--jsonl", str(job_out_dir / f"{stem}_tracks.jsonl"),
            "--occupancy-heatmap", str(job_out_dir / f"{stem}_occupancy.png"),
            "--movement-heatmap", str(job_out_dir / f"{stem}_movement.png"),
            "--flow-field", str(job_out_dir / f"{stem}_flow.png"),
            "--analytics-json", str(job_out_dir / f"{stem}_analytics.json"),
        ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print("Lỗi từ script:", result.stderr)
        raise RuntimeError(f"Tracking failed: {result.stderr}")

    # Thu thập tất cả các file đã được tạo ra
    output_files = {}
    for file_path in job_out_dir.iterdir():
        if file_path.is_file():
            data = file_path.read_bytes()
            output_files[file_path.name] = data
            # Đồng thời sao lưu vào Modal Persistent Volume
            (persistent_out_dir / file_path.name).write_bytes(data)

    volume.commit()
    print(f"[Modal] Đã thu thập {len(output_files)} files kết quả và lưu Persistent Volume.")

    return {
        "status": "success",
        "message": f"Processed {filename} ({pipeline_type}) successfully.",
        "files": output_files,
    }

# ---------------------------------------------------------------------------
# KỊCH BẢN 2: Host FastAPI Backend dưới dạng Serverless ASGI Web Endpoint
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    gpu="T4",               # Gắn GPU cho API nếu cần chạy inference realtime
    volumes={"/root/data": volume},
    timeout=600,
    scaledown_window=300,   # Giữ ấm container 5 phút sau request cuối
)
@modal.asgi_app()
def fastapi_app():
    import sys
    sys.path.insert(0, "/root")
    from backend.app.api import create_app
    from backend.app.core.config import load_config

    config = load_config(Path("/root/configs/default.yaml"))
    # Cấu hình đường dẫn lưu upload và kết quả vào Persistent Volume
    config.server.upload_directory = Path("/root/data/uploads")
    return create_app(config=config)


# CLI Local Entrypoint để test chạy script từ máy tính của bạn
@app.local_entrypoint()
def main(
    video_path: str = "data/videos/data.mp4",
    pipeline_type: str = "tracking",  # "tracking" (mặc định) hoặc "points" (Grand Central)
    output_dir: str = "output_modal", # Thư mục xuất kết quả tự động trên máy local
):
    video_file = Path(video_path)
    if not video_file.exists():
        print(f"File {video_path} không tồn tại trên máy local!")
        return

    # Tự động chọn 'points' nếu video là data.mp4 (Grand Central) trừ khi người dùng chỉ định rõ
    selected_pipeline = pipeline_type
    if pipeline_type == "tracking" and "data.mp4" in video_file.name.lower():
        print("[Gợi ý] Phát hiện video Grand Central (data.mp4), tự động dùng pipeline='points' (zone flows).")
        selected_pipeline = "points"

    print(f"Đang gửi {video_file} lên Modal GPU (pipeline: {selected_pipeline})...")
    content = video_file.read_bytes()
    result = run_tracking_remote.remote(
        content,
        filename=video_file.name,
        pipeline_type=selected_pipeline,
    )

    print("Thông báo từ Modal:", result.get("message"))

    # Tự động ghi tất cả file kết quả vào thư mục output_modal trên máy local
    local_out_dir = Path(output_dir)
    local_out_dir.mkdir(parents=True, exist_ok=True)

    files = result.get("files", {})
    if files:
        print(f"\n[Tự động tải về] Đang xuất {len(files)} file vào thư mục: {local_out_dir.resolve()}")
        for filename, data in files.items():
            dest = local_out_dir / filename
            dest.write_bytes(data)
            size_kb = len(data) / 1024
            if size_kb >= 1024:
                print(f"  ✓ {filename:<25} ({size_kb / 1024:.2f} MB)")
            else:
                print(f"  ✓ {filename:<25} ({size_kb:.1f} KB)")
        print(f"\n[Hoàn tất] Toàn bộ kết quả đã được lưu tại: {local_out_dir.resolve()}\n")
    else:
        print("[Cảnh báo] Không có file kết quả nào được trả về từ container.")
