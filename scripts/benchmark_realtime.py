#!/usr/bin/env python3
"""Benchmark runner: launches controlled Modal variants and collects processing_fps.

Usage (from project root):
    python scripts/benchmark_realtime.py --input data/videos/shibuya.mp4
    python scripts/benchmark_realtime.py --input data/videos/shibuya.mp4 --variant fp16-batch2

Variants:
  baseline    FP32 tiled (shibuya.yaml) — production throughput path
  fp32-batch2 FP32 tiled, two source frames / ten detector images
  fp16        FP16 tiled (shibuya-fp16.yaml) — production throughput path
  fp16-batch2 FP16 tiled, two source frames / ten detector images
  fp16-batch4 FP16 tiled, four source frames / twenty detector images
  candidate   Selected FP16 tiled source-batch-2 profile

Each variant writes artifacts to outputs/benchmark/<run_id>/.
A summary CSV is appended to outputs/benchmark/benchmark_summary.csv.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

CONFIGS = {
    "baseline": "configs/shibuya.yaml",
    "fp32-batch2": "configs/shibuya-fp32-batch2.yaml",
    "fp16": "configs/shibuya-fp16.yaml",
    "fp16-batch2": "configs/shibuya-fp16-batch2.yaml",
    "fp16-batch4": "configs/shibuya-fp16-batch4.yaml",
    "candidate": "configs/shibuya-realtime-candidate.yaml",
}

# Keep the default experiment matrix small and orthogonal.  Batch 4 was slower
# in the 25-second smoke, while candidate currently duplicates fp16-batch2.
DEFAULT_VARIANTS = ("baseline", "fp32-batch2", "fp16", "fp16-batch2")

PROFILE_GPU = {
    # Throughput variants never synchronize every frame. Use the underlying
    # Modal --profile-gpu flag only for a separate short diagnostic trace.
    "baseline": False,
    "fp32-batch2": False,
    "fp16": False,
    "fp16-batch2": False,
    "fp16-batch4": False,
    "candidate": False,  # Production path — no sync
}

SUMMARY_FIELDS = [
    "cpu_request_physical_cores",
    "source_batch_size_requested", "source_batch_size_effective",
    "max_detector_images_per_batch", "source_batch_fallback_reason",
    "inference_batches_total", "detector_invocations_total",
    "detector_images_total", "detector_images_per_source_frame",
    "processing_seconds", "artifact_finalize_seconds", "detector_warmup_ms",
    "processing_fps", "inference_p50_ms", "inference_p95_ms",
    "tracking_p50_ms", "tracking_p95_ms", "analytics_p50_ms",
    "analytics_p95_ms", "render_p50_ms", "render_p95_ms",
    "encode_p50_ms", "encode_p95_ms",
]


def _extract_manifest(output: str) -> dict:
    """Extract the final local-entrypoint JSON from mixed Modal log output."""
    decoder = json.JSONDecoder()
    matches: list[dict] = []
    for offset, character in enumerate(output):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(output[offset:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and value.get("status") == "gpu_completed":
            matches.append(value)
    if not matches:
        return {}
    manifest = matches[-1].get("manifest")
    return manifest if isinstance(manifest, dict) else {}


def _compatible_summary_path(base: Path, fieldnames: list[str]) -> Path:
    """Return an empty or schema-compatible CSV path without corrupting older runs."""
    candidate = base
    version = 1
    while candidate.exists():
        with candidate.open("r", newline="", encoding="utf-8") as source:
            if next(csv.reader(source), []) == fieldnames:
                return candidate
        version += 1
        candidate = base.with_name(f"{base.stem}_v{version}{base.suffix}")
    return candidate


def run_variant(variant: str, input_path: str, duration: float | None,
                run_suffix: str, cpu: float = 2.0) -> dict:
    config = CONFIGS[variant]
    profile_gpu = PROFILE_GPU[variant]
    ts = time.strftime("%Y%m%d-%H%M%S")
    run_id = f"bench-{variant}-{run_suffix}-{ts}"
    cmd = [
        sys.executable, "-m", "modal", "run", "modal_common_path.py",
        f"--input={input_path}",
        f"--config={config}",
        f"--run-id={run_id}",
        f"--cpu={cpu}",
    ]
    if duration is not None:
        cmd.append(f"--duration-seconds={duration}")
    if profile_gpu:
        cmd.append("--profile-gpu")
    print(f"\n[benchmark] Starting variant={variant!r} run_id={run_id!r}")
    print(f"  config={config}  profile_gpu={profile_gpu}  cpu={cpu}")
    started = time.monotonic()
    child_env = os.environ.copy()
    # Modal/Rich emits Unicode status glyphs. Windows' legacy cp1252 console
    # codec cannot encode them when the CLI is launched behind a captured pipe.
    child_env["PYTHONIOENCODING"] = "utf-8"
    child_env["PYTHONUTF8"] = "1"
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=child_env,
    )
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    elapsed = time.monotonic() - started
    manifest = _extract_manifest(result.stdout)
    row = {"variant": variant, "run_id": run_id, "wall_s": round(elapsed, 1),
           "exit_code": result.returncode, "status": manifest.get("status", "failed")}
    row.update({field: manifest.get(field) for field in SUMMARY_FIELDS})
    return row


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Source video path on local disk")
    parser.add_argument("--duration-seconds", type=float, default=None)
    parser.add_argument("--variant", choices=list(CONFIGS), default=None,
                        help=("Run only this variant (default: baseline, fp32-batch2, "
                              "fp16 and fp16-batch2)"))
    parser.add_argument("--run-suffix", default=time.strftime("%H%M"),
                        help="Short suffix for run IDs (default: HHMM timestamp)")
    parser.add_argument("--cpu", type=float, choices=(1.0, 2.0, 4.0, 8.0), default=2.0,
                        help="Modal physical CPU request (default: 2; use 8 for rollback)")
    args = parser.parse_args()

    variants = [args.variant] if args.variant else list(DEFAULT_VARIANTS)
    results = []
    for variant in variants:
        row = run_variant(
            variant,
            args.input,
            args.duration_seconds,
            args.run_suffix,
            cpu=args.cpu,
        )
        results.append(row)
        print(f"[benchmark] Finished: {row}")

    summary_path = Path("outputs/benchmark/benchmark_summary.csv")
    fieldnames = ["variant", "run_id", "wall_s", "exit_code", "status", *SUMMARY_FIELDS]
    summary_path = _compatible_summary_path(summary_path, fieldnames)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not summary_path.exists()
    with summary_path.open("a", newline="", encoding="utf-8") as sink:
        writer = csv.DictWriter(sink, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(results)

    print(f"\n[benchmark] Summary appended to {summary_path}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
