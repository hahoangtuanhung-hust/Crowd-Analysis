import numpy as np

from backend.app.core.config import DetectorConfig
from backend.app.inference.ultralytics_detector import (
    UltralyticsPersonDetector,
    _in_ignore_region,
    _merge_detections,
    _same_detection,
    _tile_ranges,
)
from backend.app.schemas import Detection


def test_tile_ranges_cover_axis_with_overlap() -> None:
    ranges = _tile_ranges(1280, 2, 0.2)

    assert ranges[0][0] == 0
    assert ranges[-1][1] == 1280
    assert ranges[0][1] > ranges[1][0]


def test_merge_detections_suppresses_tile_duplicates() -> None:
    detections = [
        Detection(0, 0, 20, 40, 0.9),
        Detection(4, 5, 16, 35, 0.8),
        Detection(50, 10, 70, 45, 0.7),
    ]

    merged = _merge_detections(detections, iou_threshold=0.45, max_detections=100)

    assert merged == [detections[0], detections[2]]


def test_merge_detections_keeps_overlapping_people_from_same_tile() -> None:
    detections = [
        Detection(100, 40, 140, 150, 0.92),
        Detection(106, 45, 135, 145, 0.88),
    ]

    merged = _merge_detections(
        detections,
        iou_threshold=0.45,
        max_detections=100,
        source_ids=[2, 2],
    )

    assert merged == detections


def test_merge_detections_still_suppresses_aligned_cross_tile_duplicate() -> None:
    detections = [
        Detection(100, 40, 140, 150, 0.92),
        Detection(102, 42, 138, 148, 0.88),
    ]

    merged = _merge_detections(
        detections,
        iou_threshold=0.45,
        max_detections=100,
        source_ids=[1, 2],
    )

    assert merged == [detections[0]]


def test_vectorized_merge_matches_exhaustive_reference() -> None:
    rng = np.random.default_rng(20260930)
    detections: list[Detection] = []
    source_ids: list[int] = []
    for index in range(160):
        x1 = float(rng.uniform(-20.0, 1260.0))
        y1 = float(rng.uniform(-10.0, 700.0))
        width = float(rng.uniform(6.0, 90.0))
        height = float(rng.uniform(12.0, 150.0))
        confidence = float(rng.uniform(0.04, 0.99))
        class_id = int(index % 2)
        detections.append(Detection(
            x1, y1, x1 + width, y1 + height, confidence, class_id
        ))
        source_ids.append(index % 5)
        if index % 3 == 0:
            jitter = rng.uniform(-1.5, 1.5, size=4)
            detections.append(Detection(
                x1 + float(jitter[0]),
                y1 + float(jitter[1]),
                x1 + width + float(jitter[2]),
                y1 + height + float(jitter[3]),
                confidence - 0.001,
                class_id,
            ))
            source_ids.append((index + 1) % 5)

    def exhaustive(ids: list[int] | None) -> list[Detection]:
        selected: list[Detection] = []
        selected_sources: list[int | None] = []
        for original_index, candidate in sorted(
            enumerate(detections),
            key=lambda item: item[1].confidence,
            reverse=True,
        ):
            candidate_source = ids[original_index] if ids is not None else None
            duplicate = any(
                previous.class_id == candidate.class_id
                and (ids is None or previous_source != candidate_source)
                and _same_detection(candidate, previous, 0.45)
                for previous, previous_source in zip(
                    selected, selected_sources, strict=True
                )
            )
            if duplicate:
                continue
            selected.append(candidate)
            selected_sources.append(candidate_source)
            if len(selected) >= 150:
                break
        return selected

    assert _merge_detections(
        detections,
        iou_threshold=0.45,
        max_detections=150,
        source_ids=source_ids,
    ) == exhaustive(source_ids)
    assert _merge_detections(
        detections,
        iou_threshold=0.45,
        max_detections=150,
    ) == exhaustive(None)


def test_normalized_ignore_region_filters_screen_detection() -> None:
    detection = Detection(550, 50, 650, 150, 0.8)

    assert _in_ignore_region(
        detection,
        width=1280,
        height=720,
        regions=[(0.4, 0.05, 0.6, 0.25)],
    )


def test_fp32_prediction_does_not_pass_deprecated_half_flag() -> None:
    class FakeModel:
        def __init__(self) -> None:
            self.options: dict = {}

        def predict(self, **options):
            self.options = options
            return []

    detector = object.__new__(UltralyticsPersonDetector)
    detector._config = DetectorConfig(precision="fp32")
    detector._device = None
    detector._model = FakeModel()

    assert detector._predict([np.zeros((8, 8, 3), dtype=np.uint8)]) == []
    assert "half" not in detector._model.options


def test_fp16_prediction_uses_current_quantize_option() -> None:
    class FakeModel:
        def __init__(self) -> None:
            self.options: dict = {}

        def predict(self, **options):
            self.options = options
            return []

    detector = object.__new__(UltralyticsPersonDetector)
    detector._config = DetectorConfig(precision="fp16")
    detector._device = "cuda"
    detector._model = FakeModel()

    assert detector._predict([np.zeros((8, 8, 3), dtype=np.uint8)]) == []
    assert detector._model.options["quantize"] == 16
    assert "half" not in detector._model.options


def test_tiled_detector_batches_full_frame_and_tiles_in_one_model_call() -> None:
    class FakeModel:
        def __init__(self) -> None:
            self.calls = 0
            self.batch_size = 0

        def predict(self, **options):
            self.calls += 1
            self.batch_size = len(options["source"])
            return [type("Result", (), {"boxes": None})() for _ in options["source"]]

    detector = object.__new__(UltralyticsPersonDetector)
    detector._config = DetectorConfig(
        tiled_inference=True,
        tile_rows=2,
        tile_columns=2,
        tile_include_full_frame=True,
    )
    detector._device = None
    detector._model = FakeModel()
    detector._last_inference_stats = {}

    assert detector.detect(np.zeros((720, 1280, 3), dtype=np.uint8)) == []
    assert detector._model.calls == 1
    assert detector._model.batch_size == 5
    assert detector.last_inference_stats["inference_images"] == 5
    assert detector.last_inference_stats["shape_source"] == (
        "source_images_before_ultralytics_letterbox"
    )


def test_warmup_uses_configured_source_batch_shape() -> None:
    detector = object.__new__(UltralyticsPersonDetector)
    calls: list[int] = []
    detector.detect_batch = lambda frames: calls.append(len(frames)) or []  # type: ignore[method-assign]
    detector.detect = lambda frame: (_ for _ in ()).throw(AssertionError(frame))  # type: ignore[method-assign]

    detector.warmup(
        frame_width=16,
        frame_height=8,
        warmup_passes=2,
        source_batch_size=4,
    )

    assert calls == [4, 4]


def test_multi_frame_batch_preserves_per_frame_tiled_results_and_flushes_partial() -> None:
    class FakeTensor:
        def __init__(self, value):
            self.value = np.asarray(value)

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.value

    class FakeBoxes:
        def __init__(self, confidence: float) -> None:
            self.xyxy = FakeTensor([[5.0, 5.0, 25.0, 45.0]])
            self.conf = FakeTensor([confidence])
            self.cls = FakeTensor([0])

        def __len__(self) -> int:
            return 1

    class FakeModel:
        def __init__(self) -> None:
            self.calls = 0

        def predict(self, **options):
            self.calls += 1
            return [
                type("Result", (), {"boxes": FakeBoxes(0.5 + float(frame.mean()) / 510.0)})()
                for frame in options["source"]
            ]

    def make_detector() -> UltralyticsPersonDetector:
        detector = object.__new__(UltralyticsPersonDetector)
        detector._config = DetectorConfig(
            tiled_inference=True,
            tile_rows=2,
            tile_columns=2,
            tile_include_full_frame=True,
        )
        detector._device = None
        detector._model = FakeModel()
        detector._last_inference_stats = {}
        return detector

    frames = [
        np.full((72, 128, 3), value, dtype=np.uint8)
        for value in (0, 40, 80)
    ]
    individual_detector = make_detector()
    expected = [individual_detector.detect(frame) for frame in frames]
    batch_detector = make_detector()

    actual = batch_detector.detect_batch(frames)

    assert actual == expected
    assert batch_detector._model.calls == 1
    assert batch_detector.last_inference_stats["source_batch_size"] == 3
    assert batch_detector.last_inference_stats["inference_images"] == 15
