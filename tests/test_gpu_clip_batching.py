import cv2
import numpy as np

from scripts.common_path_clip import read_source_batch, resolve_source_batch_size


def test_source_batch_is_capped_by_detector_image_limit() -> None:
    assert resolve_source_batch_size(
        requested=8,
        max_detector_images=20,
        detector_images_per_frame=5,
        motion_enabled=False,
        batch_supported=True,
        replay_cache=False,
    ) == (4, "MAX_DETECTOR_IMAGES_PER_BATCH")


def test_motion_feedback_and_replay_force_sequential_source_frames() -> None:
    assert resolve_source_batch_size(
        requested=4,
        max_detector_images=20,
        detector_images_per_frame=5,
        motion_enabled=True,
        batch_supported=True,
        replay_cache=False,
    ) == (1, "MOTION_ROI_TRACK_FEEDBACK")
    assert resolve_source_batch_size(
        requested=4,
        max_detector_images=20,
        detector_images_per_frame=5,
        motion_enabled=False,
        batch_supported=True,
        replay_cache=True,
    ) == (1, "CACHE_REPLAY")


def test_read_source_batch_flushes_partial_eof_in_source_order() -> None:
    class FakeCapture:
        def __init__(self) -> None:
            self.frames = [
                np.full((2, 3, 3), value, dtype=np.uint8)
                for value in (10, 20, 30)
            ]
            self.index = 0

        def read(self):
            if self.index >= len(self.frames):
                return False, None
            frame = self.frames[self.index]
            self.index += 1
            return True, frame

        def get(self, prop: int) -> float:
            assert prop == cv2.CAP_PROP_POS_MSEC
            return self.index * 100.0

    capture = FakeCapture()
    frames, metadata, last_timestamp = read_source_batch(
        capture,
        start_frame_id=7,
        end_frame_exclusive=11,
        source_batch_size=4,
        fps=10.0,
        last_timestamp=0.6,
    )

    assert [int(frame[0, 0, 0]) for frame in frames] == [10, 20, 30]
    assert [item["frame_id"] for item in metadata] == [7, 8, 9]
    assert [item["timestamp"] for item in metadata] == [0.7, 0.8, 0.9]
    assert last_timestamp == 0.9
