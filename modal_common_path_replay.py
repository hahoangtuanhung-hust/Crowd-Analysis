"""One cache-only Modal GPU-container verification job; never imports a detector."""

from __future__ import annotations

import json
import re
import subprocess
import time
from pathlib import Path

import modal

VOLUME_NAME = "crowd-analysis-data"
volume = modal.Volume.from_name(VOLUME_NAME)
app = modal.App("crowd-common-path-cache-verification")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libglib2.0-0", "ffmpeg", "pciutils")
    .pip_install("numpy>=2,<3", "opencv-python-headless>=4.10,<6",
                 "psutil>=6,<8", "pydantic>=2.10,<3", "pyyaml>=6,<7")
    .add_local_dir("backend", remote_path="/root/backend")
    .add_local_file("scripts/diagnose_common_path.py",
                    remote_path="/root/scripts/diagnose_common_path.py")
    .add_local_file("scripts/__init__.py", remote_path="/root/scripts/__init__.py")
)


@app.function(image=image, gpu="T4", volumes={"/root/data": volume},
              max_containers=1, timeout=600, retries=0)
def verify_cache(source_hash: str, cache_run_id: str, config_text: str,
                 run_id: str) -> dict:
    import sys
    sys.path.insert(0, "/root")
    from scripts.diagnose_common_path import _digest, run_diagnostics

    gpu_name = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True
    ).strip().splitlines()[0]
    if not gpu_name:
        raise RuntimeError("Modal GPU device check returned no GPU")
    source = Path(f"/root/data/common_path/inputs/{source_hash}.mp4")
    cache = Path(f"/root/data/common_path/runs/{cache_run_id}/tracking_cache.jsonl")
    cache_meta = cache.with_suffix(".meta.json")
    if not source.is_file() or _digest(source) != source_hash:
        raise ValueError("Source video is missing or hash-mismatched on the Modal Volume")
    if not cache.is_file() or not cache_meta.is_file():
        raise FileNotFoundError("Existing tracking cache is unavailable on the Modal Volume")
    output = Path(f"/root/data/common_path/diagnostics/{run_id}")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite Modal diagnostic run: {output}")
    config = Path(f"/tmp/{run_id}.yaml")
    config.write_text(config_text, encoding="utf-8")
    started = time.monotonic()
    manifest = run_diagnostics(
        source, cache, config, output, run_id,
        execution_host="modal_gpu_container", gpu_device=gpu_name,
    )
    encoders = subprocess.check_output(["ffmpeg", "-hide_banner", "-encoders"],
                                       text=True, stderr=subprocess.STDOUT)
    verification = dict(
        status="success", run_id=run_id,
        modal_function_call_id=modal.current_function_call_id(),
        modal_input_id=modal.current_input_id(),
        gpu_requested="T4", gpu_device=gpu_name,
        execution_host="modal_gpu_container",
        stage_devices=manifest["stage_devices"],
        detector_calls=manifest["detector_calls"],
        detector_module_loaded=manifest["detector_module_loaded"],
        inference_executed=manifest["inference_executed"],
        inference_timing_ms=None,
        nvenc_available="h264_nvenc" in encoders,
        encoder_used="cpu_opencv_mp4v",
        cache_key=manifest["cache_key"], source_hash=manifest["source_hash"],
        frame_count=manifest["frame_count"],
        runtime_seconds=round(time.monotonic() - started, 3),
    )
    (output / "modal_verification.json").write_text(
        json.dumps(verification, indent=2), encoding="utf-8"
    )
    manifest["modal_verification"] = verification
    manifest["artifacts"] = sorted(
        str(path.relative_to(output)).replace("\\", "/")
        for path in output.rglob("*") if path.is_file()
    )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    volume.commit()
    return manifest


def _cli(*args: str) -> str:
    result = subprocess.run(["modal", "volume", *args], capture_output=True,
                            text=True, check=True)
    return result.stdout


@app.local_entrypoint()
def main(source_hash: str, cache_run_id: str = "dg-integration-20260919-final",
         config: str = "configs/default.yaml", run_id: str = "") -> None:
    if not re.fullmatch(r"[0-9a-f]{64}", source_hash):
        raise ValueError("source_hash must be a SHA-256 hex digest")
    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", cache_run_id):
        raise ValueError("Invalid cache_run_id")
    run_id = run_id or time.strftime("diagnostics-modal-%Y%m%d-%H%M%S")
    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", run_id):
        raise ValueError("Invalid run_id")
    config_path = Path(config)
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    local_output = Path("outputs/common_path") / run_id
    if local_output.exists():
        raise FileExistsError(local_output)
    manifest = verify_cache.remote(
        source_hash, cache_run_id, config_path.read_text(encoding="utf-8"), run_id
    )
    if manifest.get("status") != "success":
        raise RuntimeError(f"Remote cache verification failed: {manifest}")
    local_output.mkdir(parents=True, exist_ok=False)
    for relative_name in manifest["artifacts"]:
        relative = Path(relative_name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unexpected artifact path from Modal worker")
        destination = local_output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        _cli("get", VOLUME_NAME,
             f"common_path/diagnostics/{run_id}/{relative_name}", str(destination))
    print(json.dumps(dict(local_output=str(local_output.resolve()), manifest=manifest), indent=2))


if __name__ == "__main__":
    raise SystemExit("Use `modal run modal_common_path_replay.py ...`")
