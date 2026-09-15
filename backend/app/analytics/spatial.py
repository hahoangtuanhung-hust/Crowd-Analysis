from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class SpatialTransformer:
    mode: Literal["pixel", "ground"]
    width: float
    height: float
    matrix: NDArray[np.float64] | None = None

    @classmethod
    def pixel(cls, width: int, height: int) -> SpatialTransformer:
        if width <= 0 or height <= 0:
            raise ValueError("Pixel plane dimensions must be positive")
        return cls(mode="pixel", width=float(width), height=float(height))

    @classmethod
    def ground_plane(
        cls,
        source_points: Sequence[Sequence[float]],
        *,
        width: float,
        height: float,
    ) -> SpatialTransformer:
        points = np.asarray(source_points, dtype=np.float32)
        if points.shape != (4, 2) or not np.isfinite(points).all():
            raise ValueError("source_points must contain four finite x/y pairs")
        if width <= 0 or height <= 0:
            raise ValueError("Ground plane dimensions must be positive")
        if abs(cv2.contourArea(points)) < 1.0:
            raise ValueError("Calibration points must form a non-degenerate quadrilateral")

        destination = np.asarray(
            [[0.0, 0.0], [width, 0.0], [width, height], [0.0, height]],
            dtype=np.float32,
        )
        matrix = cv2.getPerspectiveTransform(points, destination).astype(np.float64)
        return cls(mode="ground", width=float(width), height=float(height), matrix=matrix)

    def transform(self, x: float, y: float) -> tuple[float, float]:
        if self.matrix is None:
            return float(x), float(y)
        source = np.asarray([[[x, y]]], dtype=np.float64)
        transformed = cv2.perspectiveTransform(source, self.matrix)[0, 0]
        return float(transformed[0]), float(transformed[1])

    def inverse_transform(self, x: float, y: float) -> tuple[float, float]:
        if self.matrix is None:
            return float(x), float(y)
        inverse = np.linalg.inv(self.matrix)
        source = np.asarray([[[x, y]]], dtype=np.float64)
        transformed = cv2.perspectiveTransform(source, inverse)[0, 0]
        return float(transformed[0]), float(transformed[1])

    def grid_cell(
        self,
        x: float,
        y: float,
        *,
        grid_width: int,
        grid_height: int,
    ) -> tuple[int, int] | None:
        epsilon = max(self.width, self.height, 1.0) * 1e-9
        if not (-epsilon <= x <= self.width + epsilon and -epsilon <= y <= self.height + epsilon):
            return None
        x = min(self.width, max(0.0, x))
        y = min(self.height, max(0.0, y))
        column = min(grid_width - 1, int(x / self.width * grid_width))
        row = min(grid_height - 1, int(y / self.height * grid_height))
        return row, column
