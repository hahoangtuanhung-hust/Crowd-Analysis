from dataclasses import replace

import numpy as np
import pytest

from backend.app.analytics.tracklet_aggregation import TrackletAggregationEngine, _Tracklet
from backend.app.analytics.spatial import SpatialTransformer
from backend.app.core.config import AppConfig, DetectorConfig, TrackletAggregationConfig
from backend.app.inference.ultralytics_detector import (
    UltralyticsPersonDetector, _merge_detections, _merge_detections_reference, _pack_rgb_chw,
)
from backend.app.schemas import Detection
from scripts.common_path_clip import OfflineOverlapPipeline, materialize_overlap_batch, count_inference_batches


def test_direct_rgb_pack_is_byte_identical_for_strided_crops():
    rng = np.random.default_rng(48)
    frame = rng.integers(0, 256, (39, 77, 3), dtype=np.uint8)
    crops = [frame[2:34, 3:67], frame[3:35, 4:68]]
    reference = np.ascontiguousarray(np.stack(crops)[..., ::-1].transpose(0, 3, 1, 2))
    packed = _pack_rgb_chw(crops)
    assert packed.flags.c_contiguous
    assert packed.dtype == np.uint8
    assert np.array_equal(packed, reference)


def test_producer_preprocess_keeps_letterbox_pixels_and_batch_ownership():
    from ultralytics.data.augment import LetterBox

    detector = object.__new__(UltralyticsPersonDetector)
    detector._config = DetectorConfig(imgsz=320, tiled_inference=True, offline_cpu_preprocess=True)
    detector._worker_mode = "no_sync_profile"
    rng = np.random.default_rng(91)
    frames = [rng.integers(0, 256, (180, 320, 3), dtype=np.uint8) for _ in range(2)]
    first, metadata, _ = detector.prepare_source_batch(frames)
    reference_images = [LetterBox((320, 320), auto=False)(image=image) for image in first]
    reference = np.ascontiguousarray(np.stack(reference_images)[..., ::-1].transpose(0, 3, 1, 2))
    assert np.array_equal(first.preprocessed, reference)
    assert len(metadata) == 2 and all(len(crops) == 5 for crops in metadata)
    saved = first.preprocessed.copy()
    second, _, _ = detector.prepare_source_batch([np.zeros_like(frames[0])])
    assert not np.shares_memory(first.preprocessed, second.preprocessed)
    assert np.array_equal(first.preprocessed, saved)


def test_rectangular_letterbox_preserves_content_scale_and_stride_alignment():
    from ultralytics.data.augment import LetterBox

    rng = np.random.default_rng(29)
    frame = rng.integers(0, 256, (720, 1280, 3), dtype=np.uint8)
    detector = object.__new__(UltralyticsPersonDetector)
    detector._config = DetectorConfig(tiled_inference=True)
    inputs, _, _ = detector.prepare_source_batch([frame])
    for image in inputs:
        square = LetterBox((1280, 1280), auto=False)(image=image)
        rectangular = LetterBox((768, 1280), auto=False)(image=image)
        assert np.array_equal(square[256:1024], rectangular)
    with pytest.raises(ValueError, match="multiples of 32"):
        DetectorConfig(input_shape=(750, 1280))


def perspective_detector():
    detector = object.__new__(UltralyticsPersonDetector)
    detector._config = DetectorConfig(imgsz=640)
    detector._config.perspective_regions.enabled = True
    far = detector._config.perspective_regions.far
    far.y_max, far.imgsz, far.tiles.cols = 0.38, 960, 2
    detector._last_inference_stats = {}
    detector.calls = []

    def predict(inputs, imgsz=None):
        detector.calls.append((len(inputs), imgsz))
        detector._last_inference_stats = {"inference_images": len(inputs), "model_invocations": 1}
        return [[Detection(image.shape[1] / 2 - 5, image.shape[0] / 2 - 5,
                           image.shape[1] / 2 + 5, image.shape[0] / 2 + 5, 0.9)]
                for image in inputs]

    detector._predict = predict
    return detector


def test_perspective_batch_and_overlap_keep_all_far_crops():
    frames = [np.full((100, 200, 3), value, np.uint8) for value in (0, 40)]
    single = perspective_detector()
    expected = [single.detect(frame) for frame in frames]
    batch = perspective_detector()
    assert batch.detect_batch(frames) == expected
    assert batch.calls == [(2, 640), (4, 960)]
    assert batch.reference_image_count(200, 100) == 3
    assert batch.last_inference_stats["detector_images_far"] == 4

    class Capture:
        index = 0

        def read(self):
            if self.index == len(frames):
                return False, None
            frame = frames[self.index]
            self.index += 1
            return True, frame

        def get(self, prop):
            return self.index * 100

    overlap_detector = perspective_detector()
    pipeline = OfflineOverlapPipeline(capture=Capture(), detector=overlap_detector,
                                      start_frame=0, end_frame_exclusive=2, source_batch_size=2,
                                      fps=10, last_timestamp=0, queue_batches=1)
    pipeline.start()
    try:
        rows = materialize_overlap_batch(overlap_detector, pipeline.next_batch(),
                                         detector_images_per_frame=3, inference_batch_id=1)
        assert [row["detections"] for row in rows] == expected
        assert [row["detector_stats"]["detector_images_far"] for row in rows] == [2, 2]
        assert sum(row["detector_stats"]["model_invocations"] for row in rows) == 2
        assert pipeline.next_batch() is None
    finally:
        pipeline.close()


def test_perspective_far_rows_and_fractional_resize_mapping():
    detector = perspective_detector()
    detector._config.perspective_regions.far.tiles.rows = 2
    assert detector.reference_image_count(200, 100) == 5
    detector._config.perspective_regions.strategy = "region_only"
    policy = detector._config.perspective_regions
    policy.far.tiles.rows, policy.far.tiles.cols = 1, 1
    policy.far.y_min, policy.far.y_max = 0.2, 0.7
    policy.far.upscale.enabled, policy.far.upscale.scale = True, 1.3
    policy.middle.y_max = policy.near.y_max = 0

    def predict(inputs, imgsz=None):
        detector._last_inference_stats = {"inference_images": len(inputs), "model_invocations": 1}
        h, w = inputs[0].shape[:2]
        return [[Detection(0, 0, w, h, 0.9)]]

    detector._predict = predict
    result = detector.detect(np.zeros((103, 101, 3), np.uint8))[0]
    assert result.xyxy == pytest.approx((0, 20, 101, 72))


def test_perspective_boundary_duplicates_are_merged_in_source_coordinates():
    detector = perspective_detector()
    policy = detector._config.perspective_regions
    policy.strategy = "region_only"
    policy.far.y_max, policy.far.tiles.cols, policy.far.imgsz = 0.6, 1, 640
    policy.middle.y_min, policy.middle.y_max = 0.4, 1.0
    policy.near.y_max = 0

    def predict(inputs, imgsz=None):
        detector._last_inference_stats = {"inference_images": len(inputs)}
        return [[Detection(100, 480, 120, 520, 0.9)],
                [Detection(100, 80, 120, 120, 0.8)]]

    detector._predict = predict
    assert detector.detect(np.zeros((1000, 1000, 3), np.uint8)) == [
        Detection(100, 480, 120, 520, 0.9),
    ]


@pytest.mark.parametrize("with_sources", [False, True])
def test_block_merge_matches_reference_across_blocks_and_confidence_ties(with_sources):
    rng = np.random.default_rng(42)
    boxes = rng.uniform(0, 300, (700, 4))
    boxes[:, 2:] = boxes[:, :2] + rng.uniform(1, 100, (700, 2))
    detections = [Detection(*map(float, box), float(rng.choice([0.1, 0.5, 0.9])), i % 2)
                  for i, box in enumerate(boxes)]
    sources = (np.arange(700) % 5).tolist() if with_sources else None
    assert _merge_detections(detections, iou_threshold=0.45, max_detections=170,
                             source_ids=sources) == _merge_detections_reference(
        detections, iou_threshold=0.45, max_detections=170, source_ids=sources,
    )


def test_link_cache_distinguishes_interior_geometry_with_equal_endpoints():
    analyzer = TrackletAggregationEngine(TrackletAggregationConfig(), SpatialTransformer.pixel(1000, 1000))
    points = np.asarray([[0.1, 0.2], [0.2, 0.2], [0.3, 0.2]])
    first = _Tracklet(("cam", "epoch", 1), 1, 0, 1, points, np.asarray([1., 0.]), .2, .2, 1, 1)
    second = replace(first, key=("cam", "epoch", 2), points=points.copy())
    bent = replace(second, points=np.asarray([[0.1, 0.2], [0.2, 0.8], [0.3, 0.2]]))
    analyzer._link_score(first, second)
    analyzer._link_score(first, bent)
    assert len(analyzer._link_score_cache) == 2
    assert analyzer.link_score_cache_hits == 0


def test_zero_score_link_cache_is_bounded():
    analyzer = TrackletAggregationEngine(TrackletAggregationConfig(max_matching_tracklets=32),
                                         SpatialTransformer.pixel(1000, 1000))
    points = np.asarray([[0.1, 0.2], [0.3, 0.2]])
    first = _Tracklet(("cam", "epoch", 1), 1, 0, 1, points, np.asarray([1., 0.]), .2, .2, 1, 1)
    for track_id in range(8300):
        second = replace(first, key=("cam", "epoch", track_id + 2), direction=np.asarray([-1., 0.]))
        assert analyzer._link_score(first, second) == 0
    assert len(analyzer._link_score_cache) <= 8192


def test_perspective_and_uniform_motion_coverage_cannot_be_combined():
    with pytest.raises(ValueError, match="cannot combine"):
        AppConfig.model_validate({"detector": {"tiled_inference": True,
                                               "perspective_regions": {"enabled": True}},
                                  "motion_roi": {"enabled": True}})


def test_multi_resolution_invocations_do_not_inflate_source_batch_count():
    rows = [{"inference_batch_id": 1, "model_invocations": 2},
            {"inference_batch_id": 1, "model_invocations": 0},
            {"inference_batch_id": 2, "model_invocations": 2},
            {"inference_batch_id": None, "model_invocations": 0}]
    assert count_inference_batches(rows) == 2
    assert count_inference_batches([{"model_invocations": 2}]) == 1
def test_window_fps_uses_wall_time_instead_of_video_timestamps(tmp_path):
    import csv
    from scripts.common_path_clip import _write_stage_reports

    rows = [{"event_time_s": 0.0, "pipeline_elapsed_seconds": 0.1},
            {"event_time_s": 0.03333, "pipeline_elapsed_seconds": 0.2},
            {"event_time_s": 60.0, "pipeline_elapsed_seconds": 0.3}]
    _write_stage_reports(tmp_path, rows)
    with (tmp_path / "long_run_metrics.csv").open() as source:
        windows = list(csv.DictReader(source))
    assert [float(window["processing_fps"]) for window in windows] == [10.0, 10.0]


def test_performance_flags_reuse_cache_but_input_canvas_changes_key():
    from scripts.common_path_clip import cache_key

    reference = AppConfig()
    candidate = reference.model_copy(deep=True)
    candidate.detector.offline_cpu_preprocess = True
    candidate.detector.cuda_graph_inference = True
    expected = cache_key("source", "model", reference, 0, 25)
    assert cache_key("source", "model", candidate, 0, 25) == expected
    candidate.detector.input_shape = (768, 1280)
    assert cache_key("source", "model", candidate, 0, 25) != expected
