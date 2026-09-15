from __future__ import annotations

import argparse
import json
import platform
from datetime import UTC, datetime
from pathlib import Path

import psutil

from backend.app.core.config import load_config
from backend.app.core.session import ProcessingSession
from backend.app.inference import UltralyticsPersonDetector
from backend.app.video import OverlayOptions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark the complete crowd-analysis pipeline")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--video", default="data/videos/example-people.mp4")
    parser.add_argument("--output", default="benchmarks/pipeline_profile.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_config = load_config(args.config)
    detector = UltralyticsPersonDetector(base_config.detector)
    runs: list[dict] = []

    profiles = [("default", interval, OverlayOptions()) for interval in (1, 2, 3)] + [
        (
            "all_overlays",
            1,
            OverlayOptions(
                detection=True, tracking=True, trajectory=True, heatmap=True, zones=True
            ),
        )
    ]
    for profile_name, interval, overlay in profiles:
        video_config = base_config.video.model_copy(update={"inference_interval": interval})
        config = base_config.model_copy(update={"video": video_config})
        session = ProcessingSession(
            camera_id="benchmark-camera",
            source_uri=args.video,
            source_kind="video",
            realtime=False,
            detector=detector,
            config=config,
        )
        session.set_overlay(overlay)
        session.start()
        if not session.wait(timeout=180.0):
            session.stop()
            raise TimeoutError(f"Pipeline benchmark timed out for inference interval {interval}")
        runs.append(
            {
                "profile": profile_name,
                "inference_interval": interval,
                "session": session.snapshot().__dict__
                if hasattr(session.snapshot(), "__dict__")
                else {
                    "status": session.snapshot().status,
                    "frame_version": session.snapshot().frame_version,
                },
                "summary": {
                    "processed_frames": session.analytics.summary().processed_frames,
                    "unique_tracks": session.analytics.summary().unique_track_count,
                    "peak_count": session.analytics.summary().peak_crowd_count,
                },
                "metrics": session.metrics(),
            }
        )
        print(f"Completed full-pipeline profile={profile_name}, interval={interval}", flush=True)

    output = {
        "measured_at_utc": datetime.now(UTC).isoformat(),
        "hardware": {
            "processor": platform.processor() or platform.machine(),
            "physical_cores": psutil.cpu_count(logical=False),
            "logical_cores": psutil.cpu_count(),
            "ram_gb": round(psutil.virtual_memory().total / 1024**3, 1),
            "gpu": None,
        },
        "video": args.video,
        "runs": runs,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Wrote pipeline profile to {output_path}", flush=True)


if __name__ == "__main__":
    main()
