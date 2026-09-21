"""Serve the real UI pipeline from a Modal detection cache; performs no model inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import threading
from dataclasses import dataclass, field
from pathlib import Path

import uvicorn

from backend.app.api import create_app
from backend.app.core.config import load_config
from backend.app.core.session import SessionManager
from backend.app.schemas import Detection, FramePacket


@dataclass
class ReplayCounters:
    detector_calls: int = 0
    cache_reads: int = 0
    cache_resets: int = 0
    cache_reader_instances: int = 0
    cache_skipped_rows: int = 0
    cache_mismatch_count: int = 0
    last_cache_frame_id: int = -1
    last_cache_timestamp_s: float = 0.0
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)


class CachedDetector:
    def __init__(self, cache: Path, counters: ReplayCounters | None = None) -> None:
        self._source = cache.open("r", encoding="utf-8")
        self._lock = threading.Lock()
        self._counters = counters or ReplayCounters()
        with self._counters.lock:
            self._counters.cache_reader_instances += 1

    def reset(self) -> None:
        with self._lock:
            self._source.seek(0)
            with self._counters.lock:
                self._counters.cache_resets += 1
                self._counters.last_cache_frame_id = -1
                self._counters.last_cache_timestamp_s = 0.0

    def detect(self, _frame) -> list[Detection]:
        with self._lock:
            row = self._read_row()
            return [Detection(**item) for item in row[2]["detections"]] if row else []

    def _read_row(self) -> tuple[int, float, dict] | None:
        line = self._source.readline()
        if not line:
            return None
        payload = json.loads(line)
        event_time_s = payload.get("event_time_s", payload.get("source_timestamp"))
        if event_time_s is None:
            raise ValueError("Cache row requires event_time_s or source_timestamp")
        frame_id = int(payload["frame_id"])
        timestamp = float(event_time_s)
        with self._counters.lock:
            self._counters.cache_reads += 1
            self._counters.last_cache_frame_id = frame_id
            self._counters.last_cache_timestamp_s = timestamp
        return frame_id, timestamp, payload

    def detect_packet(self, packet: FramePacket) -> list[Detection]:
        with self._lock:
            while (row := self._read_row()) is not None:
                frame_id, timestamp, payload = row
                if frame_id < packet.frame_id:
                    with self._counters.lock:
                        self._counters.cache_skipped_rows += 1
                    continue
                if frame_id != packet.frame_id or abs(timestamp - packet.source_timestamp) > 0.05:
                    with self._counters.lock:
                        self._counters.cache_mismatch_count += 1
                    raise ValueError(
                        f"Cache/source mismatch at frame {packet.frame_id}: "
                        f"cached frame {frame_id}, timestamp {timestamp:.3f}s vs "
                        f"source {packet.source_timestamp:.3f}s"
                    )
                return [Detection(**item) for item in payload["detections"]]
            with self._counters.lock:
                self._counters.cache_mismatch_count += 1
            raise ValueError(f"Cache ended before source frame {packet.frame_id}")


class ReplaySessionManager(SessionManager):
    def __init__(self, config, cache: Path, counters: ReplayCounters) -> None:
        self._replay_counters = counters
        meta = json.loads(cache.with_suffix(".meta.json").read_text(encoding="utf-8"))
        self._expected_source_hash = str(meta["source_hash"])
        self._cache_frame_count = int(meta["frame_count"]) if "frame_count" in meta else None
        super().__init__(
            config,
            detector_factory=lambda _config: CachedDetector(cache, counters),
        )

    def start(self, *, source_uri: str, source_kind: str, realtime: bool,
              camera_id: str = "camera-01"):
        if source_kind != "upload":
            raise ValueError("Replay accepts only the source video matching the cache")
        digest = hashlib.sha256()
        with Path(source_uri).open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != self._expected_source_hash:
            raise ValueError("Uploaded source video SHA-256 does not match replay cache")
        return super().start(
            source_uri=source_uri, source_kind=source_kind,
            realtime=realtime, camera_id=camera_id,
            max_frames=self._cache_frame_count,
        )

    def runtime_info(self) -> dict:
        with self._replay_counters.lock:
            counters = {
                "detector_calls": self._replay_counters.detector_calls,
                "cache_reads": self._replay_counters.cache_reads,
                "cache_resets": self._replay_counters.cache_resets,
                "cache_reader_instances": self._replay_counters.cache_reader_instances,
                "cache_skipped_rows": self._replay_counters.cache_skipped_rows,
                "cache_mismatch_count": self._replay_counters.cache_mismatch_count,
                "last_cache_frame_id": self._replay_counters.last_cache_frame_id,
                "last_cache_timestamp_s": self._replay_counters.last_cache_timestamp_s,
            }
        return {
            "mode": "replay",
            "source_mode": "modal_detection_cache",
            "inference_executed": False,
            "detector_workers": 0,
            **counters,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if not args.cache.is_file() or not args.cache.with_suffix(".meta.json").is_file():
        raise FileNotFoundError("Detection cache and matching .meta.json are required")
    config = load_config(args.config)
    counters = ReplayCounters()
    manager = ReplaySessionManager(config, args.cache, counters)
    uvicorn.run(create_app(config, manager), host="127.0.0.1", port=args.port,
                log_level="info")


if __name__ == "__main__":
    main()
