from __future__ import annotations

import json
import math
import os
from collections import Counter
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
from numpy.typing import NDArray

CoordinateMode = Literal["pixel_space", "ground_plane"]
TrajectorySource = Literal["ground_truth", "prediction"]

SOURCE_VIDEO_FPS = 25.0
ANNOTATION_STEP_FRAMES = 20
ANNOTATION_FPS = SOURCE_VIDEO_FPS / ANNOTATION_STEP_FRAMES


def _configured_path(name: str, fallback: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser() if value else fallback


@dataclass(frozen=True, slots=True)
class GrandCentralPaths:
    root: Path
    video: Path
    annotations: Path
    homography: Path
    processed: Path
    output: Path

    @classmethod
    def from_env(cls, root: str | Path | None = None) -> GrandCentralPaths:
        dataset_root = Path(
            root or os.environ.get("GC_DATASET_ROOT", "data/grand-central")
        ).expanduser()
        raw = dataset_root / "raw"
        return cls(
            root=dataset_root,
            video=_configured_path(
                "GC_VIDEO_PATH", raw / "video" / "grand_central.mp4"
            ),
            annotations=_configured_path(
                "GC_ANNOTATION_DIR", raw / "annotations"
            ),
            homography=raw / "homography",
            processed=_configured_path(
                "GC_PROCESSED_DIR", dataset_root / "processed"
            ),
            output=_configured_path("GC_OUTPUT_DIR", dataset_root / "outputs"),
        )

    def create(self) -> None:
        for path in (
            self.video.parent,
            self.annotations,
            self.homography,
            self.processed,
            self.output / "tracking",
            self.output / "heatmaps",
            self.output / "popular_paths",
            self.output / "zone_flows",
            self.output / "benchmarks",
        ):
            path.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True, slots=True)
class RawAnnotationPoint:
    pedestrian_id: int
    frame_id: int
    x: float
    y: float
    annotation_file: str


@dataclass(frozen=True, slots=True)
class NormalizedTrajectoryPoint:
    camera_id: str
    source: TrajectorySource
    frame_id: int
    timestamp: float
    track_id: int
    x: float | None
    y: float | None
    x1: float | None
    y1: float | None
    x2: float | None
    y2: float | None
    foot_x: float
    foot_y: float
    world_x: float | None
    world_y: float | None
    confidence: float | None
    zone_id: str | None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class GrandCentralDataset:
    """Adapter for the CVPR 2015 Grand Central walking-path annotations.

    Every annotation file is one pedestrian path encoded as repeated
    ``x y source_frame_id`` triples. The first token is x and the second is y;
    this was verified against actual 1920x1080 frames because the historical
    OpenTraj loader assigns the two variables in the opposite order.
    """

    camera_id = "grand-central-01"
    source_fps = SOURCE_VIDEO_FPS
    annotation_fps = ANNOTATION_FPS

    def __init__(self, paths: GrandCentralPaths | None = None) -> None:
        self.paths = paths or GrandCentralPaths.from_env()

    def load_metadata(self) -> dict[str, Any]:
        metadata_path = self.paths.processed / "metadata.json"
        if metadata_path.exists():
            with metadata_path.open("r", encoding="utf-8") as stream:
                return json.load(stream)
        return {
            "dataset": "Grand Central Station walking paths",
            "camera_id": self.camera_id,
            "source_video_fps": self.source_fps,
            "annotation_fps": self.annotation_fps,
            "paths": self.path_metadata(),
        }

    def path_metadata(self) -> dict[str, str]:
        return {
            "root": str(self.paths.root.resolve()),
            "video": str(self.paths.video.resolve()),
            "annotations": str(self.paths.annotations.resolve()),
            "homography": str(self.paths.homography.resolve()),
            "processed": str(self.paths.processed.resolve()),
            "output": str(self.paths.output.resolve()),
        }

    def annotation_files(self) -> list[Path]:
        if not self.paths.annotations.is_dir():
            raise FileNotFoundError(
                f"Grand Central annotation directory does not exist: "
                f"{self.paths.annotations}"
            )
        files = [path for path in self.paths.annotations.glob("*.txt") if path.is_file()]
        return sorted(files, key=self._annotation_sort_key)

    def iter_annotations(
        self, *, duration_seconds: float | None = None
    ) -> Iterator[RawAnnotationPoint]:
        max_frame = (
            math.floor(duration_seconds * self.source_fps)
            if duration_seconds is not None
            else None
        )
        for path in self.annotation_files():
            try:
                pedestrian_id = int(path.stem)
            except ValueError as exc:
                raise ValueError(f"Annotation filename must be numeric: {path.name}") from exc
            tokens = path.read_text(encoding="utf-8").split()
            if not tokens:
                continue
            if len(tokens) % 3:
                raise ValueError(
                    f"Corrupted annotation {path}: expected x/y/frame triples, "
                    f"found {len(tokens)} tokens"
                )
            previous_frame = -1
            for offset in range(0, len(tokens), 3):
                try:
                    x = float(tokens[offset])
                    y = float(tokens[offset + 1])
                    frame_value = float(tokens[offset + 2])
                except ValueError as exc:
                    raise ValueError(
                        f"Corrupted annotation {path} at token {offset + 1}"
                    ) from exc
                if not all(math.isfinite(value) for value in (x, y, frame_value)):
                    raise ValueError(f"Non-finite annotation value in {path}")
                frame_id = int(frame_value)
                if not math.isclose(frame_value, frame_id, abs_tol=1e-6):
                    raise ValueError(f"Frame ID is not an integer in {path}: {frame_value}")
                if frame_id < previous_frame:
                    raise ValueError(f"Frame IDs are not sorted in {path}")
                previous_frame = frame_id
                if max_frame is not None and frame_id > max_frame:
                    continue
                yield RawAnnotationPoint(
                    pedestrian_id=pedestrian_id,
                    frame_id=frame_id,
                    x=x,
                    y=y,
                    annotation_file=path.name,
                )

    def load_trajectories(
        self,
        *,
        duration_seconds: float | None = None,
        coordinate_mode: CoordinateMode = "pixel_space",
        zones: list[dict[str, Any]] | None = None,
    ) -> Iterator[NormalizedTrajectoryPoint]:
        homography = None
        if coordinate_mode == "ground_plane":
            homography = self.load_homography(required=True)

        previous_by_pedestrian: dict[int, int] = {}
        segment_by_pedestrian: Counter[int] = Counter()
        for raw in self.iter_annotations(duration_seconds=duration_seconds):
            previous_frame = previous_by_pedestrian.get(raw.pedestrian_id)
            if previous_frame is not None and raw.frame_id - previous_frame > ANNOTATION_STEP_FRAMES:
                segment_by_pedestrian[raw.pedestrian_id] += 1
            previous_by_pedestrian[raw.pedestrian_id] = raw.frame_id
            segment = segment_by_pedestrian[raw.pedestrian_id]
            track_id = raw.pedestrian_id * 100 + segment

            world_x: float | None = None
            world_y: float | None = None
            if homography is not None:
                world_x, world_y = self.transform_point(raw.x, raw.y, homography)
            zone_id = self._zone_at(raw.x, raw.y, zones or [])
            yield NormalizedTrajectoryPoint(
                camera_id=self.camera_id,
                source="ground_truth",
                frame_id=raw.frame_id,
                timestamp=raw.frame_id / self.source_fps,
                track_id=track_id,
                x=raw.x,
                y=raw.y,
                x1=None,
                y1=None,
                x2=None,
                y2=None,
                foot_x=raw.x,
                foot_y=raw.y,
                world_x=world_x,
                world_y=world_y,
                confidence=None,
                zone_id=zone_id,
            )

    def load_homography(self, *, required: bool = False) -> NDArray[np.float64] | None:
        candidates = (
            self.paths.processed / "homography.json",
            self.paths.homography / "H.json",
            self.paths.homography / "homography.json",
        )
        path = next((candidate for candidate in candidates if candidate.is_file()), None)
        if path is None:
            if required:
                raise FileNotFoundError(
                    f"No homography JSON found under {self.paths.homography}"
                )
            return None
        with path.open("r", encoding="utf-8") as stream:
            raw = json.load(stream)
        if isinstance(raw, dict):
            matrix_value = raw.get("matrix", raw.get("homog"))
            if matrix_value is None:
                raise ValueError(f"Homography JSON has no 'matrix' or 'homog' key: {path}")
        else:
            matrix_value = raw
        matrix = np.asarray(matrix_value, dtype=np.float64)
        self.validate_homography(matrix)
        return matrix

    @staticmethod
    def validate_homography(matrix: NDArray[np.float64]) -> None:
        if matrix.shape != (3, 3):
            raise ValueError(f"Homography must have shape (3, 3), got {matrix.shape}")
        if not np.isfinite(matrix).all():
            raise ValueError("Homography contains non-finite values")
        determinant = float(np.linalg.det(matrix))
        if math.isclose(determinant, 0.0, abs_tol=1e-12):
            raise ValueError("Homography matrix is singular")

    @staticmethod
    def transform_point(
        x: float, y: float, matrix: NDArray[np.float64]
    ) -> tuple[float, float]:
        transformed = matrix @ np.asarray([x, y, 1.0], dtype=np.float64)
        if math.isclose(float(transformed[2]), 0.0, abs_tol=1e-12):
            raise ValueError("Homography maps point to infinity")
        return float(transformed[0] / transformed[2]), float(transformed[1] / transformed[2])

    def video_metadata(self) -> dict[str, Any]:
        if not self.paths.video.is_file():
            raise FileNotFoundError(f"Grand Central video does not exist: {self.paths.video}")
        capture = cv2.VideoCapture(str(self.paths.video))
        try:
            if not capture.isOpened():
                raise ValueError(f"Unable to open Grand Central video: {self.paths.video}")
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            return {
                "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "fps": fps,
                "frame_count": frame_count,
                "duration_seconds": frame_count / fps if fps > 0 else None,
                "size_bytes": self.paths.video.stat().st_size,
            }
        finally:
            capture.release()

    def validate(self) -> dict[str, Any]:
        files = self.annotation_files()
        point_count = 0
        frame_counts: Counter[int] = Counter()
        pedestrian_ids: set[int] = set()
        x_min = math.inf
        x_max = -math.inf
        y_min = math.inf
        y_max = -math.inf
        for point in self.iter_annotations():
            point_count += 1
            frame_counts[point.frame_id] += 1
            pedestrian_ids.add(point.pedestrian_id)
            x_min = min(x_min, point.x)
            x_max = max(x_max, point.x)
            y_min = min(y_min, point.y)
            y_max = max(y_max, point.y)
        if not point_count:
            raise ValueError(f"No annotation points found in {self.paths.annotations}")

        video = self.video_metadata()
        video_duration = float(video["duration_seconds"] or 0.0)
        overlapping_frames = [
            frame for frame in frame_counts if frame / self.source_fps <= video_duration
        ]
        homography = self.load_homography(required=False)
        return {
            "valid": True,
            "camera_id": self.camera_id,
            "raw_format": "one pedestrian per file; repeated x y source_frame_id triples",
            "coordinate_order": "x_then_y",
            "source_video_fps": self.source_fps,
            "annotation_fps": self.annotation_fps,
            "annotation_step_frames": ANNOTATION_STEP_FRAMES,
            "annotation_files": len(files),
            "pedestrians": len(pedestrian_ids),
            "annotation_points": point_count,
            "annotated_frames": len(frame_counts),
            "frame_id_min": min(frame_counts),
            "frame_id_max": max(frame_counts),
            "people_per_frame_average": round(point_count / len(frame_counts), 3),
            "people_per_frame_max": max(frame_counts.values()),
            "coordinate_bounds": {
                "x_min": x_min,
                "x_max": x_max,
                "y_min": y_min,
                "y_max": y_max,
            },
            "video": video,
            "video_timebase_mapping": {
                "rule": "timestamp = annotation_frame_id / 25; video_frame = round(timestamp * video_fps)",
                "overlapping_annotated_frames": len(overlapping_frames),
                "last_overlapping_annotation_frame": max(overlapping_frames)
                if overlapping_frames
                else None,
            },
            "homography": {
                "available": homography is not None,
                "valid": homography is not None,
                "world_unit": "unverified",
            },
            "paths": self.path_metadata(),
        }

    @staticmethod
    def _annotation_sort_key(path: Path) -> tuple[int, str]:
        try:
            return int(path.stem), path.name
        except ValueError:
            return 2**31 - 1, path.name

    @staticmethod
    def _zone_at(x: float, y: float, zones: list[dict[str, Any]]) -> str | None:
        for zone in zones:
            points = np.asarray(zone.get("points", []), dtype=np.float32)
            if points.shape[0] >= 3 and cv2.pointPolygonTest(points, (x, y), False) >= 0:
                return str(zone["zone_id"])
        return None

    def zone_at(self, x: float, y: float, zones: list[dict[str, Any]]) -> str | None:
        return self._zone_at(x, y, zones)
