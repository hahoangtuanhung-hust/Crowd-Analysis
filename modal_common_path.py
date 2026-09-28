"""Ephemeral, one-job Modal GPU clip runner; artifacts move through a persistent Volume."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import time
from pathlib import Path

import modal

VOLUME_NAME = "crowd-analysis-data"
volume = modal.Volume.from_name(VOLUME_NAME)
app = modal.App("crowd-common-path-clip")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libglib2.0-0", "ffmpeg")
    .pip_install("lap>=0.5.12", "numpy>=2,<3", "opencv-python-headless>=4.10,<6",
                 "psutil>=6,<8", "pydantic>=2.10,<3", "pyyaml>=6,<7",
                 "ultralytics>=8.4.116,<9")
    .add_local_dir("backend", remote_path="/root/backend")
    .add_local_file("scripts/common_path_clip.py", remote_path="/root/scripts/common_path_clip.py")
    .add_local_file("scripts/__init__.py", remote_path="/root/scripts/__init__.py")
    .add_local_file("yolo26n.pt", remote_path="/root/model/yolo26n.pt")
)


@app.function(image=image, gpu="T4", volumes={"/root/data": volume},
# A 10-minute Shibuya video is substantially slower than wall-clock in the
# current detector/render pipeline. Keep one invocation alive long enough for
# the batch to finish; the function still writes its manifest to the Volume.
              max_containers=1, timeout=14400, retries=0)
def gpu_clip(source_hash: str, model_hash: str, config_text: str, start_seconds: float,
             duration_seconds: float | None, engine: str, run_id: str) -> dict:
    import datetime as dt
    import sys
    import threading
    import subprocess
    import psutil
    sys.path.insert(0, "/root")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Modal GPU unavailable; inference refused")
    from scripts.common_path_clip import digest, run

    source = Path(f"/root/data/common_path/inputs/{source_hash}.mp4")
    model = Path("/root/model/yolo26n.pt")
    if not source.is_file() or digest(source) != source_hash or digest(model) != model_hash:
        raise ValueError("Input or model missing/mismatched on Modal worker")
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
                       model_hash=model_hash)
        stop_sampling.set()
        sampler.join(timeout=3)
        elapsed = time.monotonic() - started
        completed_at = dt.datetime.now(dt.timezone.utc)
        (output / "resource_metrics.jsonl").write_text(
            "".join(json.dumps(item, separators=(",", ":")) + "\n" for item in samples),
            encoding="utf-8",
        )
        cpu_values = [float(item["cpu_percent"]) for item in samples]
        ram_values = [float(item["ram_mb"]) for item in samples]
        gpu_values = [float(item["gpu_utilization_percent"]) for item in samples
                      if item["gpu_utilization_percent"] is not None]
        gpu_memory_values = [float(item["gpu_memory_used_mb"]) for item in samples
                             if item["gpu_memory_used_mb"] is not None]
        resources = {
            "started_at_utc": wall_started.isoformat(),
            "completed_at_utc": completed_at.isoformat(),
            "remote_wall_seconds": round(elapsed, 3),
            "sample_interval_seconds": 1,
            "sample_count": len(samples),
            "cpu_percent_avg": round(sum(cpu_values) / len(cpu_values), 3) if cpu_values else None,
            "cpu_percent_peak": round(max(cpu_values), 3) if cpu_values else None,
            "ram_peak_mb": round(max(ram_values), 3) if ram_values else None,
            "gpu_utilization_avg_percent": round(sum(gpu_values) / len(gpu_values), 3) if gpu_values else None,
            "gpu_utilization_peak_percent": round(max(gpu_values), 3) if gpu_values else None,
            "gpu_memory_used_peak_mb": round(max(gpu_memory_values), 3) if gpu_memory_values else None,
            "gpu": torch.cuda.get_device_name(0),
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
            summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        manifest["remote_wall_seconds"] = round(elapsed, 2)
        manifest["completed_at_utc"] = resources["completed_at_utc"]
        manifest["resource_summary"] = resources
        manifest["gpu"] = torch.cuda.get_device_name(0)
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


@app.local_entrypoint()
def main(input: str = "data/videos/data-test.mp4", start_seconds: float = 0.,
         duration_seconds: float | None = None, engine: str = "tracklet_aggregation",
         mode: str = "offline_fast", config: str = "configs/default.yaml",
         run_id: str = "", cache_policy: str = "reuse",
         download_artifacts: bool = False) -> None:
    if mode != "offline_fast":
        raise ValueError("The batch runner only supports offline_fast; UI realtime is separate")
    if cache_policy not in ("reuse", "refresh") or engine not in ("legacy", "directional_grid", "tracklet_aggregation", "shadow"):
        raise ValueError("Invalid cache policy or engine")
    if start_seconds < 0:
        raise ValueError("Start must be >=0 seconds")
    if duration_seconds is not None and duration_seconds <= 0:
        raise ValueError("Duration must be >0 seconds when provided")
    source, settings, model = Path(input), Path(config), Path("yolo26n.pt")
    for path in (source, settings, model):
        if not path.is_file():
            raise FileNotFoundError(path)
    run_id = run_id or time.strftime("%Y%m%d-%H%M%S")
    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", run_id):
        raise ValueError("run_id must contain only letters, numbers, underscores, hyphens")
    local_output = Path("outputs/common_path") / run_id
    if download_artifacts and local_output.exists():
        raise FileExistsError(local_output)
    source_hash, model_hash = _sha256(source), _sha256(model)
    # Resolve local `extends` profiles before sending the single immutable
    # config payload to the ephemeral worker.
    import yaml
    from backend.app.core.config import load_config
    resolved_config_text = yaml.safe_dump(
        load_config(settings).model_dump(mode="json"),
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
    manifest = gpu_clip.remote(source_hash, model_hash, resolved_config_text,
                               start_seconds, duration_seconds, engine, run_id)
    if manifest.get("status") != "success":
        raise RuntimeError(f"Remote run failed: {manifest}")
    remote_path = f"{VOLUME_NAME}:common_path/runs/{run_id}"
    if not download_artifacts:
        print(json.dumps({
            "status": "gpu_completed",
            "run_id": run_id,
            "remote_path": remote_path,
            "manifest": manifest,
            "next_command": (
                f"python scripts/download_modal_artifacts.py --run-id {run_id}"
            ),
        }, indent=2))
        return
    local_output.mkdir(parents=True, exist_ok=False)
    for filename in [*manifest["artifacts"], "manifest.json"]:
        if Path(filename).name != filename:
            raise ValueError("Unexpected artifact name from worker")
        _cli("get", VOLUME_NAME, f"common_path/runs/{run_id}/{filename}",
             str(local_output / filename))
    local_manifest = json.loads((local_output / "manifest.json").read_text(encoding="utf-8"))
    if not (local_output / "tracked_points_common_path.mp4").is_file():
        raise RuntimeError("MP4 was not downloaded; run is not inspected")
    print(json.dumps(dict(local_output=str(local_output.resolve()), manifest=local_manifest),
                     indent=2))
    print("Review preview/contact sheet/debug images and write inspection.md before claiming success.")
