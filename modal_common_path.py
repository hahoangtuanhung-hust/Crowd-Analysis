"""Ephemeral, one-job Modal GPU clip runner; artifacts move through a persistent Volume."""

from __future__ import annotations

import hashlib
import json
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
              max_containers=1, timeout=3600, retries=0)
def gpu_clip(source_hash: str, model_hash: str, config_text: str, start_seconds: float,
             duration_seconds: float | None, engine: str, run_id: str) -> dict:
    import sys
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
    try:
        manifest = run(source, output, config_path=config, model_path=model,
                       start_seconds=start_seconds, duration_seconds=duration_seconds,
                       engine=engine, run_id=run_id, input_hash=source_hash,
                       model_hash=model_hash)
        manifest["remote_wall_seconds"] = round(time.monotonic() - started, 2)
        manifest["gpu"] = torch.cuda.get_device_name(0)
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        volume.commit()
        return manifest
    except Exception as exc:
        output.mkdir(parents=True, exist_ok=True)
        (output / "manifest.json").write_text(json.dumps(dict(
            run_id=run_id, status="failed", error=str(exc),
            remote_wall_seconds=round(time.monotonic() - started, 2))), encoding="utf-8")
        volume.commit()
        raise


def _cli(*args: str) -> str:
    result = subprocess.run(["modal", "volume", *args], capture_output=True,
                            text=True, check=True)
    return result.stdout


def _sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


@app.local_entrypoint()
def main(input: str = "data/videos/data-test.mp4", start_seconds: float = 0.,
         duration_seconds: float | None = None, engine: str = "directional_grid",
         mode: str = "offline_fast", config: str = "configs/default.yaml",
         run_id: str = "", cache_policy: str = "reuse") -> None:
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
    if local_output.exists():
        raise FileExistsError(local_output)
    source_hash, model_hash = _sha256(source), _sha256(model)
    remote_input = f"common_path/inputs/{source_hash}.mp4"
    try:
        listing = _cli("ls", VOLUME_NAME, "common_path/inputs")
    except subprocess.CalledProcessError:
        listing = ""
    if cache_policy == "refresh" or f"{source_hash}.mp4" not in listing:
        _cli("put", VOLUME_NAME, str(source), remote_input)
    manifest = gpu_clip.remote(source_hash, model_hash, settings.read_text(encoding="utf-8"),
                               start_seconds, duration_seconds, engine, run_id)
    if manifest.get("status") != "success":
        raise RuntimeError(f"Remote run failed: {manifest}")
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
