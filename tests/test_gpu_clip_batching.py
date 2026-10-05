import cv2
import numpy as np

from scripts.common_path_clip import (
    OfflineOverlapPipeline,
    materialize_overlap_batch,
    read_source_batch,
    resolve_source_batch_size,
)


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


def test_overlap_pipeline_is_bounded_and_preserves_source_order() -> None:
    class FakeCapture:
        def __init__(self) -> None:
            self.frames = [
                np.full((2, 3, 3), value, dtype=np.uint8)
                for value in (10, 20, 30, 40, 50)
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

    class FakeDetector:
        def prepare_source_batch(self, frames):
            return list(frames), [[None] for _ in frames], 0.2

        def infer_prepared_batch(self, inputs):
            predictions = [[int(frame[0, 0, 0])] for frame in inputs]
            return predictions, {
                "batch_size": len(inputs),
                "model_predict_ms": 4.0,
                "ultralytics_predict_ms": 4.0,
                "result_transfer_ms": 0.4,
                "yolo_preprocess_total_ms": 1.0,
                "yolo_inference_total_ms": 2.5,
                "yolo_postprocess_total_ms": 0.5,
                "actual_tensor_shapes": [list(frame.shape) for frame in inputs],
            }

        def merge_prepared_frame(self, frame, predictions, metadata):
            assert metadata == [None]
            return predictions[0], 0.3

    pipeline = OfflineOverlapPipeline(
        capture=FakeCapture(),
        detector=FakeDetector(),
        start_frame=0,
        end_frame_exclusive=5,
        source_batch_size=2,
        fps=10.0,
        last_timestamp=0.0,
        queue_batches=1,
    )
    observed: list[tuple[int, int]] = []
    batch_sizes: list[int] = []
    pipeline.start()
    try:
        sequence = 0
        while (batch := pipeline.next_batch()) is not None:
            sequence += 1
            materialized = materialize_overlap_batch(
                pipeline.detector,
                batch,
                detector_images_per_frame=1,
                inference_batch_id=sequence,
            )
            batch_sizes.append(len(materialized))
            observed.extend(
                (int(item["frame_id"]), int(item["detections"][0]))
                for item in materialized
            )
    finally:
        pipeline.close()

    assert batch_sizes == [2, 2, 1]
    assert observed == [(0, 10), (1, 20), (2, 30), (3, 40), (4, 50)]
    assert pipeline.input_queue_peak <= 1
    assert pipeline.result_queue_peak <= 1


def test_overlap_pipeline_batches_only_deadline_detection_frames() -> None:
    class FakeCapture:
        def __init__(self) -> None:
            self.frames = [
                np.full((2, 3, 3), value, dtype=np.uint8)
                for value in range(10)
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

    class FakeDetector:
        observed_batches: list[list[int]] = []

        def prepare_source_batch(self, frames):
            self.observed_batches.append([int(frame[0, 0, 0]) for frame in frames])
            return list(frames), [[None] for _ in frames], 0.2

        def infer_prepared_batch(self, inputs):
            return [[int(frame[0, 0, 0])] for frame in inputs], {
                "batch_size": len(inputs),
                "model_predict_ms": 4.0,
                "actual_tensor_shapes": [list(frame.shape) for frame in inputs],
            }

        def merge_prepared_frame(self, frame, predictions, metadata):
            return predictions[0], 0.3

    detector = FakeDetector()
    pipeline = OfflineOverlapPipeline(
        capture=FakeCapture(),
        detector=detector,
        start_frame=0,
        end_frame_exclusive=10,
        source_batch_size=2,
        inference_interval=3,
        fps=10.0,
        last_timestamp=0.0,
        queue_batches=1,
    )
    rows: list[dict] = []
    pipeline.start()
    try:
        sequence = 0
        while (batch := pipeline.next_batch()) is not None:
            sequence += 1
            rows.extend(materialize_overlap_batch(
                detector,
                batch,
                detector_images_per_frame=1,
                inference_batch_id=sequence,
            ))
    finally:
        pipeline.close()

    assert detector.observed_batches == [[0, 3], [6, 9]]
    assert [row["frame_id"] for row in rows] == list(range(10))
    assert [row["intentional_skip"] for row in rows] == [
        False, True, True, False, True, True, False, True, True, False
    ]
    assert [row["detections"] for row in rows if not row["intentional_skip"]] == [
        [0], [3], [6], [9]
    ]
    assert pipeline.detection_scan_deadline_misses == 0


def test_overlap_pipeline_flushes_trailing_coast_only_frames() -> None:
    class FakeCapture:
        def __init__(self) -> None:
            self.index = 0

        def read(self):
            if self.index >= 11:
                return False, None
            frame = np.full((2, 3, 3), self.index, dtype=np.uint8)
            self.index += 1
            return True, frame

        def get(self, prop: int) -> float:
            return self.index * 100.0

    class FakeDetector:
        def prepare_source_batch(self, frames):
            return list(frames), [[None] for _ in frames], 0.0

        def infer_prepared_batch(self, inputs):
            return [[int(frame[0, 0, 0])] for frame in inputs], {
                "batch_size": len(inputs)
            }

        def merge_prepared_frame(self, frame, predictions, metadata):
            return predictions[0], 0.0

    detector = FakeDetector()
    pipeline = OfflineOverlapPipeline(
        capture=FakeCapture(), detector=detector, start_frame=0,
        end_frame_exclusive=11, source_batch_size=2, inference_interval=3,
        fps=10.0, last_timestamp=0.0, queue_batches=1,
    )
    rows = []
    pipeline.start()
    try:
        sequence = 0
        while (batch := pipeline.next_batch()) is not None:
            sequence += 1
            rows.extend(materialize_overlap_batch(
                detector, batch, detector_images_per_frame=1,
                inference_batch_id=sequence,
            ))
    finally:
        pipeline.close()
    assert [row["frame_id"] for row in rows] == list(range(11))
    assert rows[-1]["intentional_skip"] is True
    assert rows[-1]["detections"] == []
