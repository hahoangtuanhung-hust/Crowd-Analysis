from pathlib import Path

import modal

from backend.app.artifacts import POINT_TRACKING_ARTIFACTS

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
        "lap>=0.5.12",
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
    config_bytes: bytes = b"",
    zones_bytes: bytes = b"",
    max_frames: int | None = None,
) -> dict:
    import shutil
    import subprocess
    from pathlib import Path

    from backend.app.artifacts import validate_point_tracking_artifacts

    if pipeline_type not in ("tracking", "points"):
        raise ValueError("pipeline_type must be either 'tracking' or 'points'")

    input_path = Path("/tmp") / filename
    input_path.write_bytes(video_bytes)
    config_path = Path("/tmp/job_config.yaml")
    if config_bytes:
        config_path.write_bytes(config_bytes)
    else:
        shutil.copy2("/root/configs/default.yaml", config_path)
    zones_path = Path("/tmp/job_zones.json")
    if zones_bytes:
        zones_path.write_bytes(zones_bytes)
    else:
        shutil.copy2("/root/configs/zones.json", zones_path)

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
        cmd = [
            "python", "-m", "scripts.process_point_tracks",
            str(input_path),
            "--config", str(config_path),
            "--output-dir", str(job_out_dir),
            "--zones", str(zones_path),
        ]
        if max_frames is not None:
            cmd.extend(["--max-frames", str(max_frames)])
    else:
        if max_frames is not None:
            raise ValueError("max_frames is currently supported only by the points pipeline")
        cmd = [
            "python", "-m", "scripts.run_tracking",
            str(input_path),
            "--config", str(config_path),
            "--output", str(job_out_dir / f"{stem}_overlay.mp4"),
            "--jsonl", str(job_out_dir / f"{stem}_tracks.jsonl"),
            "--occupancy-heatmap", str(job_out_dir / f"{stem}_occupancy.png"),
            "--movement-heatmap", str(job_out_dir / f"{stem}_movement.png"),
            "--flow-field", str(job_out_dir / f"{stem}_flow.png"),
            "--analytics-json", str(job_out_dir / f"{stem}_analytics.json"),
        ]

    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    print(result.stdout)
    if result.returncode != 0:
        print("Lỗi từ script:", result.stderr)
        raise RuntimeError(f"Tracking failed: {result.stderr}")

    output_files = {}
    if pipeline_type == "points":
        artifacts = validate_point_tracking_artifacts(job_out_dir, reject_unexpected=True)
        for artifact_name, file_path in artifacts.items():
            data = file_path.read_bytes()
            output_files[artifact_name] = data
            (persistent_out_dir / artifact_name).write_bytes(data)
    else:
        for file_path in job_out_dir.iterdir():
            if file_path.is_file():
                data = file_path.read_bytes()
                output_files[file_path.name] = data
                (persistent_out_dir / file_path.name).write_bytes(data)

    volume.commit()
    print(f"[Modal] Đã thu thập {len(output_files)} files kết quả và lưu Persistent Volume.")

    return {
        "status": "success",
        "message": f"Processed {filename} ({pipeline_type}) successfully.",
        "files": output_files,
    }


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
    config.server.cors_origins = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]
    if Path("/root/yolo26n.pt").exists():
        config.detector.model = "/root/yolo26n.pt"
    return create_app(config=config)


# CLI Local Entrypoint để test chạy script từ máy tính của bạn
@app.local_entrypoint()
def main(
    video_path: str = "data/videos/data-test.mp4",
    pipeline_type: str = "tracking",  # "tracking" (mặc định) hoặc "points"
    output_dir: str = "output_modal", # Thư mục xuất kết quả tự động trên máy local
    config_path: str = "configs/default.yaml",
    zones_path: str = "configs/zones.json",
    max_frames: int | None = None,
):
    video_file = Path(video_path)
    if not video_file.exists():
        print(f"File {video_path} không tồn tại trên máy local!")
        return

    if pipeline_type not in ("tracking", "points"):
        raise ValueError("pipeline_type must be either 'tracking' or 'points'")
    config_file = Path(config_path)
    if not config_file.is_file():
        raise FileNotFoundError(f"Config {config_path} không tồn tại trên máy local")
    zones_file = Path(zones_path)
    if pipeline_type == "points" and not zones_file.is_file():
        raise FileNotFoundError(f"Zones {zones_path} không tồn tại trên máy local")
    if max_frames is not None and max_frames < 1:
        raise ValueError("max_frames must be positive")

    print(
        f"Đang gửi {video_file} lên Modal GPU "
        f"(pipeline: {pipeline_type}, config: {config_file})..."
    )
    content = video_file.read_bytes()
    result = run_tracking_remote.remote(
        content,
        filename=video_file.name,
        pipeline_type=pipeline_type,
        config_bytes=config_file.read_bytes(),
        zones_bytes=zones_file.read_bytes() if pipeline_type == "points" else b"",
        max_frames=max_frames,
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
