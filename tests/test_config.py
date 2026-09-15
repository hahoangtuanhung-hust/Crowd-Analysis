from pathlib import Path

import pytest
from pydantic import ValidationError

from backend.app.core.config import AppConfig, TrackerConfig, load_config


def test_default_config_loads() -> None:
    config = load_config(Path("configs/default.yaml"))
    assert config.detector.model == "yolo26n.pt"
    assert config.detector.classes == [0]
    assert config.detector.max_det == 1000
    assert config.detector.imgsz == 960
    assert config.video.queue_size == 4


def test_tracker_rejects_inverted_thresholds() -> None:
    with pytest.raises(ValidationError):
        TrackerConfig(track_low_thresh=0.5, track_high_thresh=0.2)


def test_config_rejects_unknown_keys() -> None:
    with pytest.raises(ValidationError):
        AppConfig.model_validate({"unknown": True})
