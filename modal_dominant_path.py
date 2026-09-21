"""Render dominant-direction Common Path from the Shibuya cache on Modal T4."""

from __future__ import annotations

import json
import re
import subprocess
import time
from pathlib import Path

import modal


VOLUME_NAME = "crowd-analysis-data"
SOURCE_HASH = "ae75cf06369007e9745c0f951e06e07826a2abc5dcf36276bc1ef06050aca04c"
volume = modal.Volume.from_name(VOLUME_NAME)
app = modal.App("crowd-dominant-common-path")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libglib2.0-0", "ffmpeg")
    .pip_install(
        "numpy>=2,<3", "opencv-python-headless>=4.10,<6",
        "pydantic>=2.10,<3", "pyyaml>=6,<7",
    )
    .add_local_dir("backend", remote_path="/root/backend")
    .add_local_file(
        "scripts/render_dominant_path.py",
        remote_path="/root/scripts/render_dominant_path.py",
    )
    .add_local_file("scripts/__init__.py", remote_path="/root/scripts/__init__.py")
)


@app.function(
    image=image,
    gpu="T4",
    volumes={"/root/data": volume},
    max_containers=1,
    timeout=600,
    retries=0,
)
def render_remote(cache_run_id: str, config_text: str, run_id: str) -> dict[str, object]:
    import sys

    sys.path.insert(0, "/root")
    from scripts.render_dominant_path import digest, run

    gpu_name = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True
    ).strip().splitlines()[0]
    if not gpu_name:
        raise RuntimeError("Modal GPU device check returned no GPU")
    source = Path(f"/root/data/common_path/inputs/{SOURCE_HASH}.mp4")
    cache = Path(f"/root/data/common_path/runs/{cache_run_id}/tracking_cache.jsonl")
    if not source.is_file() or digest(source) != SOURCE_HASH:
        raise ValueError("Shibuya source is missing or hash-mismatched")
    if not cache.is_file() or not cache.with_suffix(".meta.json").is_file():
        raise FileNotFoundError("Shibuya tracking cache is missing")
    output = Path(f"/root/data/common_path/dominant_runs/{run_id}")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite Modal run: {output}")
    config = Path(f"/tmp/{run_id}.yaml")
    config.write_text(config_text, encoding="utf-8")
    manifest = run(
        source,
        cache,
        config,
        output,
        run_id,
        execution_host="modal_gpu_container",
        gpu_device=gpu_name,
    )
    manifest["modal_function_call_id"] = modal.current_function_call_id()
    manifest["modal_input_id"] = modal.current_input_id()
    manifest["gpu_requested"] = "T4"
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    volume.commit()
    return manifest


def _cli(*args: str) -> None:
    subprocess.run(["modal", "volume", *args], check=True)


@app.local_entrypoint()
def main(
    cache_run_id: str = "shibuya-live-integration-20260920",
    config: str = "configs/shibuya.yaml",
    run_id: str = "",
) -> None:
    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", cache_run_id):
        raise ValueError("Invalid cache_run_id")
    run_id = run_id or time.strftime("shibuya-dominant-%Y%m%d-%H%M%S")
    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", run_id):
        raise ValueError("Invalid run_id")
    config_path = Path(config)
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    local_output = Path("outputs/common_path") / run_id
    if local_output.exists():
        raise FileExistsError(local_output)
    manifest = render_remote.remote(
        cache_run_id, config_path.read_text(encoding="utf-8"), run_id
    )
    if manifest.get("status") != "success":
        raise RuntimeError(f"Remote render failed: {manifest}")
    local_output.mkdir(parents=True)
    for filename in manifest["artifacts"]:
        if Path(str(filename)).name != filename:
            raise ValueError("Unexpected artifact name from Modal worker")
        _cli(
            "get", VOLUME_NAME,
            f"common_path/dominant_runs/{run_id}/{filename}",
            str(local_output / filename),
        )
    print(json.dumps({"local_output": str(local_output.resolve()), "manifest": manifest}, indent=2))


if __name__ == "__main__":
    raise SystemExit("Use `modal run modal_dominant_path.py ...`")
