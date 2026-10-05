"""Ephemeral, one-job Modal GPU clip runner; artifacts move through a persistent Volume."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

import modal

VOLUME_NAME = "crowd-analysis-data"
CPU_REQUEST_DEFAULT = 2.0
CPU_REQUEST_CHOICES = (1.0, 2.0, 4.0, 8.0)
volume = modal.Volume.from_name(VOLUME_NAME)
app = modal.App("crowd-common-path-clip")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libglib2.0-0", "ffmpeg")
    .pip_install("lap>=0.5.12", "numpy>=2,<3", "opencv-python-headless>=4.10,<6",
                 "psutil>=6,<8", "pydantic>=2.10,<3", "pyyaml>=6,<7",
                 "ultralytics==8.4.155", "onnx>=1.17,<2",
                 "onnxslim>=0.1.71,<1", "tensorrt-cu12>=10.8,<11",
                 "torch==2.14.0", "torchvision==0.29.0")
    .add_local_dir("backend", remote_path="/root/backend")
    .add_local_file("scripts/common_path_clip.py", remote_path="/root/scripts/common_path_clip.py")
    .add_local_file("scripts/__init__.py", remote_path="/root/scripts/__init__.py")
    .add_local_file("yolo26n.pt", remote_path="/root/model/yolo26n.pt")
)


@app.function(image=image, gpu="T4", volumes={"/root/data": volume},
              timeout=3600, retries=0, cpu=CPU_REQUEST_DEFAULT)
def build_tensorrt_engine(model_hash: str, batch: int = 10,
                          imgsz: int = 1280) -> dict:
    """Build/cache one static TensorRT FP16 engine on the target T4."""
    import platform
    import torch
    import ultralytics
    import tensorrt
    from ultralytics import YOLO

    model = Path("/root/model/yolo26n.pt")
    if not model.is_file() or _sha256(model) != model_hash:
        raise ValueError("Source model missing/mismatched on TensorRT builder")
    key = (
        f"{model_hash[:16]}-u{ultralytics.__version__}-trt{tensorrt.__version__}"
        f"-t4-b{batch}-i{imgsz}-fp16"
    )
    engine_dir = Path("/root/data/common_path/engines")
    engine_dir.mkdir(parents=True, exist_ok=True)
    engine_path = engine_dir / f"{key}.engine"
    metadata_path = engine_dir / f"{key}.json"
    if engine_path.is_file() and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("engine_sha256") == _sha256(engine_path):
            metadata["cache_hit"] = True
            return metadata

    started = time.perf_counter()
    exported = YOLO(str(model)).export(
        format="engine",
        imgsz=imgsz,
        batch=batch,
        half=True,
        dynamic=False,
        device=0,
        workspace=4,
        simplify=True,
        verbose=False,
    )
    exported_path = Path(str(exported))
    if not exported_path.is_file():
        raise RuntimeError(f"TensorRT export did not create an engine: {exported}")
    shutil.copy2(exported_path, engine_path)
    metadata = {
        "backend": "tensorrt_fp16",
        "cache_hit": False,
        "engine_relative_path": str(engine_path.relative_to("/root/data")).replace("\\", "/"),
        "engine_sha256": _sha256(engine_path),
        "engine_size_bytes": engine_path.stat().st_size,
        "engine_build_seconds": round(time.perf_counter() - started, 3),
        "static_batch": batch,
        "input_shape": [batch, 3, imgsz, imgsz],
        "precision": "fp16",
        "gpu": torch.cuda.get_device_name(0),
        "ultralytics_version": ultralytics.__version__,
        "tensorrt_version": tensorrt.__version__,
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "source_model_sha256": model_hash,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    volume.commit()
    return metadata


@app.function(image=image, gpu="T4", volumes={"/root/data": volume,},
# A 10-minute Shibuya video is substantially slower than wall-clock in the
# current detector/render pipeline. Keep one invocation alive long enough for
# the batch to finish; the function still writes its manifest to the Volume.
              max_containers=1, timeout=14400, retries=0, cpu=CPU_REQUEST_DEFAULT)
def gpu_clip(source_hash: str, model_hash: str, config_text: str, start_seconds: float,
            duration_seconds: float | None, engine: str, run_id: str,
            profile_gpu: bool = False,
            profile_cpu: bool = False,
            detector_worker_mode: str = "",
            cpu_request_physical_cores: float = CPU_REQUEST_DEFAULT,
            inference_backend: str = "pytorch_fp32",
            tensorrt_engine_info: dict | None = None) -> dict:
    import datetime as dt
    import sys
    import threading
    import subprocess
    import psutil
    worker_threads = max(1, int(cpu_request_physical_cores))
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[variable] = str(worker_threads)
    sys.path.insert(0, "/root")
    import torch
    torch.set_num_threads(worker_threads)
    try:
        torch.set_num_interop_threads(min(2, worker_threads))
    except RuntimeError:
        # PyTorch permits configuring inter-op threads only before parallel
        # work starts. A pre-initialized runtime is still reported below.
        pass
    if not torch.cuda.is_available():
        raise RuntimeError("Modal GPU unavailable; inference refused")
    from scripts.common_path_clip import digest, run

    source = Path(f"/root/data/common_path/inputs/{source_hash}.mp4")
    source_model = Path("/root/model/yolo26n.pt")
    if not source.is_file() or digest(source) != source_hash or digest(source_model) != model_hash:
        raise ValueError("Input or model missing/mismatched on Modal worker")
    if inference_backend == "pytorch_fp32":
        model = source_model
        backend_model_hash = model_hash
    elif inference_backend == "tensorrt_fp16":
        if not tensorrt_engine_info:
            raise ValueError("TensorRT engine metadata is required")
        relative = Path(str(tensorrt_engine_info["engine_relative_path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Invalid TensorRT engine path")
        model = Path("/root/data") / relative
        backend_model_hash = str(tensorrt_engine_info["engine_sha256"])
        if not model.is_file() or digest(model) != backend_model_hash:
            raise ValueError("TensorRT engine missing/mismatched on Modal worker")
    else:
        raise ValueError(f"Unsupported inference backend: {inference_backend}")
    output = Path(f"/root/data/common_path/runs/{run_id}")
    config = Path(f"/tmp/{run_id}.yaml")
    config.write_text(config_text, encoding="utf-8")
    started = time.monotonic()
    wall_started = dt.datetime.now(dt.timezone.utc)
    process = psutil.Process()
    samples: list[dict[str, object]] = []
    stop_sampling = threading.Event()

    def sample_resources() -> None:
        process.cpu_percent(None)
        while not stop_sampling.wait(1.0):
            gpu_util = gpu_memory_used = gpu_memory_total = None
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, encoding="utf-8", errors="replace",
                    timeout=2, check=False,
                )
                fields = [item.strip() for item in result.stdout.splitlines()[0].split(",")]
                if len(fields) >= 3:
                    gpu_util, gpu_memory_used, gpu_memory_total = (float(fields[0]), float(fields[1]), float(fields[2]))
            except (OSError, IndexError, ValueError, subprocess.TimeoutExpired):
                pass
            samples.append({
                "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "cpu_percent": round(process.cpu_percent(None), 3),
                "ram_mb": round(process.memory_info().rss / 1024**2, 3),
                "gpu_utilization_percent": gpu_util,
                "gpu_memory_used_mb": gpu_memory_used,
                "gpu_memory_total_mb": gpu_memory_total,
            })

    sampler = threading.Thread(target=sample_resources, name="resource-sampler", daemon=True)
    sampler.start()
    try:
        manifest = run(source, output, config_path=config, model_path=model,
                       start_seconds=start_seconds, duration_seconds=duration_seconds,
                       engine=engine, run_id=run_id, input_hash=source_hash,
                       model_hash=backend_model_hash, profile_gpu=profile_gpu,
                       profile_cpu=profile_cpu,
                       detector_worker_mode=detector_worker_mode or None,
                       inference_backend=inference_backend)
        stop_sampling.set()
        sampler.join(timeout=3)
        elapsed = time.monotonic() - started
        completed_at = dt.datetime.now(dt.timezone.utc)
        (output / "resource_metrics.jsonl").write_text(
            "".join(json.dumps(item, separators=(",", ":")) + "\n" for item in samples),
            encoding="utf-8",
        )
        # `samples` also contains lifecycle events such as `warmup_completed`.
        # Only resource samples carry utilization fields, so aggregate optional
        # values instead of treating event rows as resource measurements.
        cpu_values = [float(value) for item in samples
                      if (value := item.get("cpu_percent")) is not None]
        ram_values = [float(value) for item in samples
                      if (value := item.get("ram_mb")) is not None]
        gpu_values = [float(value) for item in samples
                      if (value := item.get("gpu_utilization_percent")) is not None]
        gpu_memory_values = [float(value) for item in samples
                             if (value := item.get("gpu_memory_used_mb")) is not None]
        resources = {
            "started_at_utc": wall_started.isoformat(),
            "completed_at_utc": completed_at.isoformat(),
            "remote_wall_seconds": round(elapsed, 3),
            "sample_interval_seconds": 1,
            "sample_count": len(cpu_values),
            "cpu_percent_avg": round(sum(cpu_values) / len(cpu_values), 3) if cpu_values else None,
            "cpu_percent_peak": round(max(cpu_values), 3) if cpu_values else None,
            "ram_peak_mb": round(max(ram_values), 3) if ram_values else None,
            "gpu_utilization_avg_percent": round(sum(gpu_values) / len(gpu_values), 3) if gpu_values else None,
            "gpu_utilization_peak_percent": round(max(gpu_values), 3) if gpu_values else None,
            "gpu_memory_used_peak_mb": round(max(gpu_memory_values), 3) if gpu_memory_values else None,
            "gpu": torch.cuda.get_device_name(0),
            "cpu_request_physical_cores": cpu_request_physical_cores,
            "cpu_affinity_logical_cpus": (
                len(process.cpu_affinity()) if hasattr(process, "cpu_affinity") else None
            ),
            "threading": {
                "torch_intraop_threads": torch.get_num_threads(),
                "torch_interop_threads": torch.get_num_interop_threads(),
                "opencv_threads": __import__("cv2").getNumThreads(),
                "OMP_NUM_THREADS": os.getenv("OMP_NUM_THREADS"),
                "MKL_NUM_THREADS": os.getenv("MKL_NUM_THREADS"),
                "OPENBLAS_NUM_THREADS": os.getenv("OPENBLAS_NUM_THREADS"),
            },
        }
        (output / "resource_summary.json").write_text(json.dumps(resources, indent=2), encoding="utf-8")
        summary_path = output / "summary.json"
        if summary_path.is_file():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["timing"] = {
                "started_at_utc": resources["started_at_utc"],
                "completed_at_utc": resources["completed_at_utc"],
                "remote_wall_seconds": resources["remote_wall_seconds"],
            }
            summary["resources"] = resources
            summary["backend_artifact"] = (
                tensorrt_engine_info
                if inference_backend == "tensorrt_fp16"
                else {
                    "backend": "pytorch_fp32",
                    "source_model_sha256": model_hash,
                    "model_size_bytes": source_model.stat().st_size,
                }
            )
            summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        manifest["remote_wall_seconds"] = round(elapsed, 2)
        manifest["completed_at_utc"] = resources["completed_at_utc"]
        manifest["resource_summary"] = resources
        manifest["cpu_request_physical_cores"] = cpu_request_physical_cores
        manifest["gpu"] = torch.cuda.get_device_name(0)
        manifest["inference_backend"] = inference_backend
        manifest["backend_artifact"] = (
            tensorrt_engine_info
            if inference_backend == "tensorrt_fp16"
            else {"backend": "pytorch_fp32", "source_model_sha256": model_hash}
        )
        manifest["artifacts"] = sorted(path.name for path in output.iterdir() if path.is_file())
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        volume.commit()
        return manifest
    except Exception as exc:
        stop_sampling.set()
        sampler.join(timeout=3)
        output.mkdir(parents=True, exist_ok=True)
        (output / "manifest.json").write_text(json.dumps(dict(
            run_id=run_id, status="failed", error=str(exc),
            cpu_request_physical_cores=cpu_request_physical_cores,
            remote_wall_seconds=round(time.monotonic() - started, 2))), encoding="utf-8")
        volume.commit()
        raise


def _cli(*args: str) -> str:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    result = subprocess.run(["modal", "volume", *args], capture_output=True,
                            text=True, encoding="utf-8", errors="replace",
                            env=env, check=True)
    return result.stdout


def _sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _validate_cpu_request(cpu: float) -> float:
    value = float(cpu)
    if value not in CPU_REQUEST_CHOICES:
        choices = ", ".join(str(int(item)) for item in CPU_REQUEST_CHOICES)
        raise ValueError(f"cpu must be one of: {choices}")
    return value


@app.local_entrypoint()
def main(input: str = "data/videos/data-test.mp4", start_seconds: float = 0.,
         duration_seconds: float | None = None, engine: str = "tracklet_aggregation",
         mode: str = "offline_fast", config: str = "configs/default.yaml",
         run_id: str = "", cache_policy: str = "reuse",
         download_artifacts: bool = False,
         profile_gpu: bool = False,
         profile_cpu: bool = False,
         detector_worker_mode: str = "",
         inference_backend: str = "pytorch_fp32",
         cpu: float = CPU_REQUEST_DEFAULT) -> None:
    if mode != "offline_fast":
        raise ValueError("The batch runner only supports offline_fast; UI realtime is separate")
    if cache_policy not in ("reuse", "refresh") or engine not in ("legacy", "directional_grid", "tracklet_aggregation", "shadow"):
        raise ValueError("Invalid cache policy or engine")
    if start_seconds < 0:
        raise ValueError("Start must be >=0 seconds")
    if duration_seconds is not None and duration_seconds <= 0:
        raise ValueError("Duration must be >0 seconds when provided")
    if inference_backend not in ("pytorch_fp32", "tensorrt_fp16", "ab"):
        raise ValueError("inference_backend must be pytorch_fp32, tensorrt_fp16, or ab")
    cpu = _validate_cpu_request(cpu)
    source, settings, model = Path(input), Path(config), Path("yolo26n.pt")
    for path in (source, settings, model):
        if not path.is_file():
            raise FileNotFoundError(path)
    run_id = run_id or time.strftime("%Y%m%d-%H%M%S")
    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", run_id):
        raise ValueError("run_id must contain only letters, numbers, underscores, hyphens")
    backend_runs = (
        (("pytorch_fp32", f"{run_id}-torch"),
         ("tensorrt_fp16", f"{run_id}-trt"))
        if inference_backend == "ab"
        else ((inference_backend, run_id),)
    )
    for _, backend_run_id in backend_runs:
        if len(backend_run_id) > 64:
            raise ValueError("backend run_id exceeds 64 characters")
        local_output = Path("outputs/common_path") / backend_run_id
        if download_artifacts and local_output.exists():
            raise FileExistsError(local_output)
    source_hash, model_hash = _sha256(source), _sha256(model)
    # Resolve local `extends` profiles before sending the single immutable
    # config payload to the ephemeral worker.
    import yaml
    from backend.app.core.config import load_config
    resolved_config = load_config(settings)
    # Modal batch artifacts are review outputs: include the tracker boxes in
    # every rendered video while keeping the source profile unchanged on disk.
    resolved_config.visualization.show_bounding_boxes = True
    resolved_config_text = yaml.safe_dump(
        resolved_config.model_dump(mode="json"),
        sort_keys=False,
    )
    remote_input = f"common_path/inputs/{source_hash}.mp4"
    try:
        listing = _cli("ls", VOLUME_NAME, "common_path/inputs")
    except subprocess.CalledProcessError:
        listing = ""
    if cache_policy == "refresh" or f"{source_hash}.mp4" not in listing:
        # Modal Volume refuses overwriting an existing path unless --force is
        # explicit. Refresh is intentionally an overwrite; reuse remains
        # immutable and skips the upload when the hashed object exists.
        upload_args = ["put", VOLUME_NAME, str(source), remote_input]
        if cache_policy == "refresh":
            upload_args.insert(1, "--force")
        _cli(*upload_args)
    configured_gpu_clip = gpu_clip.with_options(cpu=cpu)
    engine_info = (
        build_tensorrt_engine.remote(model_hash, batch=10, imgsz=resolved_config.detector.imgsz)
        if inference_backend in ("tensorrt_fp16", "ab") else None
    )
    completed: list[dict] = []
    for backend_name, backend_run_id in backend_runs:
        manifest = configured_gpu_clip.remote(
            source_hash,
            model_hash,
            resolved_config_text,
            start_seconds,
            duration_seconds,
            engine,
            backend_run_id,
            profile_gpu=profile_gpu,
            profile_cpu=profile_cpu,
            detector_worker_mode=detector_worker_mode,
            cpu_request_physical_cores=cpu,
            inference_backend=backend_name,
            tensorrt_engine_info=(
                engine_info if backend_name == "tensorrt_fp16" else None
            ),
        )
        if manifest.get("status") != "success":
            raise RuntimeError(f"Remote run failed: {manifest}")
        remote_path = f"{VOLUME_NAME}:common_path/runs/{backend_run_id}"
        if not download_artifacts:
            completed.append({
                "run_id": backend_run_id,
                "remote_path": remote_path,
                "manifest": manifest,
                "next_command": (
                    "python scripts/download_modal_artifacts.py "
                    f"--run-id {backend_run_id}"
                ),
            })
            continue
        local_output = Path("outputs/common_path") / backend_run_id
        local_output.mkdir(parents=True, exist_ok=False)
        # The remote manifest may already list itself after finalization.
        for filename in dict.fromkeys([*manifest["artifacts"], "manifest.json"]):
            if Path(filename).name != filename:
                raise ValueError("Unexpected artifact name from worker")
            _cli("get", VOLUME_NAME, f"common_path/runs/{backend_run_id}/{filename}",
                 str(local_output / filename))
        local_manifest = json.loads(
            (local_output / "manifest.json").read_text(encoding="utf-8")
        )
        if not (local_output / "tracked_points_common_path.mp4").is_file():
            raise RuntimeError("MP4 was not downloaded; run is not inspected")
        completed.append({
            "local_output": str(local_output.resolve()),
            "manifest": local_manifest,
        })
    print(json.dumps({"status": "gpu_completed", "runs": completed}, indent=2))
    print("Review preview/contact sheet/debug images and write inspection.md before claiming success.")
