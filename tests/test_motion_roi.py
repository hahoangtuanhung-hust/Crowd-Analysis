from __future__ import annotations

import numpy as np

from backend.app.core.config import (
    MotionBackgroundConfig,
    MotionCoverageConfig,
    MotionFallbackConfig,
    MotionROIConfig,
    MotionTrackingConfig,
)
from backend.app.video.motion_roi import MotionROIPlanner


class _MaskSequence:
    def __init__(self, foreground: list[bool]) -> None:
        self._foreground = iter(foreground)

    def apply(self, frame: np.ndarray, *, learningRate: float) -> np.ndarray:
        del learningRate
        value = 255 if next(self._foreground) else 0
        return np.full(frame.shape[:2], value, dtype=np.uint8)


def _config(*, shadow_mode: bool = False) -> MotionROIConfig:
    return MotionROIConfig(
        shadow_mode=shadow_mode,
        background=MotionBackgroundConfig(
            analysis_width=160,
            warmup_min_s=0.0,
            warmup_max_s=0.5,
            static_confirm_s=0.1,
            max_mask_age_s=1.0,
        ),
        coverage=MotionCoverageConfig(periodic_scan_interval_s=0.5),
        tracking=MotionTrackingConfig(
            max_observation_gap_s=1.0,
            prediction_grace_s=1.5,
        ),
        fallback=MotionFallbackConfig(soft_recovery_confirm_s=0.2),
    )


def _plan(planner: MotionROIPlanner, timestamp_s: float):
    frame = np.zeros((90, 160, 3), dtype=np.uint8)
    return planner.plan(
        frame,
        timestamp_s,
        tile_regions=((0, 0, 80, 90), (80, 0, 160, 90)),
        reference_image_count=3,
    )


def test_zero_pts_timeline_leaves_warmup() -> None:
    planner = MotionROIPlanner(_config())
    planner._subtractor = _MaskSequence([False, False])  # type: ignore[assignment]

    first = _plan(planner, 0.0)
    later = _plan(planner, 0.2)

    assert first.state == "WARMUP"
    assert later.state == "HYBRID"
    assert "BACKGROUND_WARMUP" not in later.reasons


def test_shadow_execution_does_not_postpone_virtual_periodic_scan() -> None:
    config = _config(shadow_mode=True)
    # Keep the test focused on the virtual reference clock rather than the
    # independent static-mask confirmation gate.
    config.background.static_confirm_s = 0.0
    planner = MotionROIPlanner(config)
    planner._subtractor = _MaskSequence([False] * 6)  # type: ignore[assignment]

    plans = [_plan(planner, timestamp) for timestamp in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)]

    assert all(plan.scan_type == "reference" for plan in plans)
    assert plans[1].proposed_scan_type == "skip"
    assert plans[-1].proposed_scan_type == "reference"
    assert "PERIODIC_COVERAGE_DEADLINE" in plans[-1].reasons


def test_recovery_requires_continuously_healthy_mask() -> None:
    config = _config()
    config.coverage.periodic_scan_interval_s = 1.0
    planner = MotionROIPlanner(config)
    planner._subtractor = _MaskSequence(  # type: ignore[assignment]
        [False, False, True, False, False, False]
    )

    assert _plan(planner, 0.0).state == "WARMUP"
    assert _plan(planner, 0.2).state == "HYBRID"
    unhealthy = _plan(planner, 0.6)
    first_healthy = _plan(planner, 0.7)
    still_recovering = _plan(planner, 0.8)
    recovered = _plan(planner, 0.91)

    assert unhealthy.state == "RECOVER"
    assert "MASK_UNHEALTHY" in unhealthy.reasons
    assert first_healthy.state == "RECOVER"
    assert still_recovering.state == "RECOVER"
    assert recovered.state == "HYBRID"


def test_stale_motion_sample_fails_safe_to_reference() -> None:
    config = _config()
    config.background.max_mask_age_s = 0.15
    planner = MotionROIPlanner(config)
    planner._subtractor = _MaskSequence([False, False])  # type: ignore[assignment]

    _plan(planner, 0.0)
    stale = _plan(planner, 0.2)

    assert stale.scan_type == "reference"
    assert stale.state == "RECOVER"
    assert "MASK_STALE" in stale.reasons
