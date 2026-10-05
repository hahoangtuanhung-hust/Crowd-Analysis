from pathlib import Path

import pytest
from pydantic import ValidationError

from backend.app.core.config import (
    AppConfig,
    CommonPathStyleConfig,
    DetectorConfig,
    TrackerConfig,
    load_config,
)


def test_default_config_loads() -> None:
    config = load_config(Path("configs/default.yaml"))
    assert config.detector.model == "yolo26n.pt"
    assert config.detector.classes == [0]
    assert config.detector.max_det == 1000
    assert config.detector.imgsz == 1280
    assert config.detector.confidence == config.tracker.track_low_thresh == 0.05
    assert config.detector.tiled_inference is False
    assert config.tracker.association_mode == "hybrid"
    assert config.video.queue_size == 4


def test_shibuya_config_enables_small_person_detection() -> None:
    config = load_config(Path("configs/shibuya.yaml"))

    assert config.detector.tiled_inference is True
    assert (config.detector.tile_rows, config.detector.tile_columns) == (2, 2)
    assert config.detector.tile_include_full_frame is True
    assert len(config.detector.ignore_regions) == 3
    assert config.tracker.new_track_thresh == 0.08
    assert config.tracker.stationary_lost_track_grace_frames == 30


def test_shibuya_far2_candidate_changes_only_normalized_tile_geometry() -> None:
    baseline = load_config(Path("configs/shibuya-overlap-batch2.yaml"))
    candidate = load_config(Path("configs/shibuya-overlap-batch2-far2.yaml"))

    baseline_detector = baseline.detector.model_dump()
    candidate_detector = candidate.detector.model_dump()
    assert baseline_detector.pop("tile_regions_normalized") == []
    assert candidate_detector.pop("tile_regions_normalized") == [
        (0.0, 0.0, 0.55625, 0.555556),
        (0.44375, 0.0, 1.0, 0.555556),
    ]
    assert candidate_detector == baseline_detector
    assert candidate.video == baseline.video
    assert candidate.tracker == baseline.tracker
    assert candidate.analytics == baseline.analytics


def test_shibuya_realtime_candidate_resolves_relative_parent() -> None:
    config = load_config(Path("configs/shibuya-realtime-candidate.yaml"))

    assert config.detector.precision == "fp16"
    assert config.detector.tiled_inference is True
    assert config.detector.tile_include_full_frame is True
    assert config.video.source_batch_size == 2
    assert config.video.max_detector_images_per_batch == 10
    assert config.analytics.common_path.max_paths == 3


def test_shibuya_tracking_stable_only_lowers_recovery_pool_and_enables_candidate_gates() -> None:
    baseline = load_config(Path("configs/shibuya.yaml"))
    candidate = load_config(Path("configs/shibuya-tracking-stable.yaml"))

    assert candidate.detector.confidence == candidate.tracker.track_low_thresh == 0.02
    assert candidate.detector.tile_edge_min_confidence == 0.04
    assert candidate.tracker.track_high_thresh == baseline.tracker.track_high_thresh
    assert candidate.tracker.new_track_thresh == baseline.tracker.new_track_thresh
    assert candidate.tracker.association_size_adaptive is True
    assert candidate.tracker.association_hard_gate is True
    assert candidate.tracker.direction_diagnostics_enabled is True


def test_shibuya_fp16_variants_keep_three_common_paths() -> None:
    baseline = load_config(Path("configs/shibuya.yaml"))
    for path in (
        "configs/shibuya-fp16.yaml",
        "configs/shibuya-fp16-batch2.yaml",
        "configs/shibuya-fp16-batch4.yaml",
        "configs/shibuya-realtime-candidate.yaml",
    ):
        config = load_config(Path(path))
        assert config.analytics.common_path.max_paths == 3
        assert config.tracker == baseline.tracker


def test_shibuya_fp32_batch2_changes_only_offline_batch_limits() -> None:
    baseline = load_config(Path("configs/shibuya.yaml"))
    batch2 = load_config(Path("configs/shibuya-fp32-batch2.yaml"))

    assert batch2.detector == baseline.detector
    assert batch2.tracker == baseline.tracker
    assert batch2.analytics == baseline.analytics
    assert batch2.video.source_batch_size == 2
    assert batch2.video.max_detector_images_per_batch == 10


def test_tracker_rejects_inverted_thresholds() -> None:
    with pytest.raises(ValidationError):
        TrackerConfig(track_low_thresh=0.5, track_high_thresh=0.2)


def test_tracker_rejects_shorter_stationary_grace_period() -> None:
    with pytest.raises(ValidationError, match="stationary_lost_track_grace_frames"):
        TrackerConfig(lost_track_grace_frames=4, stationary_lost_track_grace_frames=3)


def test_tracker_rejects_invalid_adaptive_gate_and_direction_hysteresis() -> None:
    with pytest.raises(ValidationError, match="association_max_distance_pixels"):
        TrackerConfig(
            association_distance_floor_pixels=20,
            association_max_distance_pixels=10,
        )
    with pytest.raises(ValidationError, match="direction_stationary_exit"):
        TrackerConfig(
            direction_stationary_enter_speed_pixels_s=5,
            direction_stationary_exit_speed_pixels_s=4,
        )
    with pytest.raises(ValidationError, match="direction_min_observed_span"):
        TrackerConfig(
            direction_history_seconds=0.1,
            direction_min_observed_span_seconds=0.2,
        )


def test_detector_threshold_must_feed_bytetrack_low_confidence_stage() -> None:
    with pytest.raises(ValidationError, match="detector.confidence"):
        AppConfig(
            detector=DetectorConfig(confidence=0.2),
            tracker=TrackerConfig(track_low_thresh=0.1),
        )


def test_config_rejects_unknown_keys() -> None:
    with pytest.raises(ValidationError):
        AppConfig.model_validate({"unknown": True})


def test_common_path_style_rejects_inverted_widths() -> None:
    with pytest.raises(ValidationError, match="min_width_pixels"):
        CommonPathStyleConfig(min_width_pixels=20, max_width_pixels=10)
