from pathlib import Path

import pytest
from pydantic import ValidationError

from backend.app.core.config import AppConfig, DetectorConfig, TrackerConfig, load_config


def test_default_config_loads() -> None:
    config = load_config(Path("configs/default.yaml"))
    assert config.detector.model == "yolo26n.pt"
    assert config.detector.classes == [0]
    assert config.detector.max_det == 1000
    assert config.detector.imgsz == 1280
    assert config.detector.confidence == config.tracker.track_low_thresh == 0.05
    assert config.tracker.association_mode == "hybrid"
    assert config.video.queue_size == 4


def test_tracker_rejects_inverted_thresholds() -> None:
    with pytest.raises(ValidationError):
        TrackerConfig(track_low_thresh=0.5, track_high_thresh=0.2)


def test_detector_threshold_must_feed_bytetrack_low_confidence_stage() -> None:
    with pytest.raises(ValidationError, match="detector.confidence"):
        AppConfig(
            detector=DetectorConfig(confidence=0.2),
            tracker=TrackerConfig(track_low_thresh=0.1),
        )


def test_config_rejects_unknown_keys() -> None:
    with pytest.raises(ValidationError):
        AppConfig.model_validate({"unknown": True})
